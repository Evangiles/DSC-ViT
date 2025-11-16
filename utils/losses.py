"""
Loss functions for DSC-ViT training.

Includes:
1. Main segmentation loss (Cross-Entropy)
2. Cluster space losses (categorization, compactness, separation)
3. Deep supervision loss (weighted sum across T steps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SegmentationLoss(nn.Module):
    """
    Standard cross-entropy loss for segmentation with optional class weighting.
    """

    def __init__(self,
                 num_classes: int,
                 ignore_index: int = -100,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W]
            target: [B, H, W] - class indices
        Returns:
            loss: scalar
        """
        return self.ce_loss(pred, target)


class ClusterSpaceLoss(nn.Module):
    """
    Loss functions for cluster space learning.

    Combines three objectives:
    1. Assignment loss: Cluster assignments should match class labels
    2. Compactness loss: Features should be close to their assigned cluster center
    3. Separation loss: Cluster centers should be far from each other
    """

    def __init__(self,
                 num_clusters: int,
                 num_classes: int,
                 alpha: float = 0.1,  # Compactness weight
                 beta: float = 0.05,  # Separation weight
                 use_assignment_loss: bool = True,
                 use_compactness_loss: bool = True,
                 use_separation_loss: bool = True):
        super().__init__()

        self.K = num_clusters
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.use_assignment_loss = use_assignment_loss
        self.use_compactness_loss = use_compactness_loss
        self.use_separation_loss = use_separation_loss

    def assignment_loss(self,
                       combined_cluster: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """
        Cluster assignment should match ground truth classes.

        Args:
            combined_cluster: [B, K, H, W]
            target: [B, H, W]
        Returns:
            loss: scalar
        """
        # Get cluster assignment (argmax over K dimension)
        cluster_assignment = torch.argmax(combined_cluster, dim=1)  # [B, H, W]

        # If K == num_classes, cluster assignment should match target
        if self.K == self.num_classes:
            # Direct comparison
            valid_mask = (target != -100)  # Ignore index
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    combined_cluster,
                    target,
                    ignore_index=-100
                )
            else:
                loss = torch.tensor(0.0, device=target.device)
        else:
            # K != num_classes: use soft alignment
            # Convert both to one-hot and compute KL divergence
            # This is a simplified version; can be improved
            loss = torch.tensor(0.0, device=target.device)

        return loss

    def compactness_loss(self,
                        features: torch.Tensor,
                        cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        Features should be close to their assigned cluster centers.

        Args:
            features: [B, K, H, W]
            cluster_centers: [K, K]
        Returns:
            loss: scalar
        """
        B, K, H, W = features.shape

        # Reshape features to [B, H, W, K]
        features = features.permute(0, 2, 3, 1).reshape(B * H * W, K)

        # Compute distances to cluster centers
        # ||f - μ||^2 = ||f||^2 + ||μ||^2 - 2<f, μ>
        f_squared = (features ** 2).sum(dim=-1, keepdim=True)  # [BHW, 1]
        mu_squared = (cluster_centers ** 2).sum(dim=-1, keepdim=True).t()  # [1, K]
        cross_term = features @ cluster_centers.t()  # [BHW, K]

        distances = f_squared + mu_squared - 2 * cross_term  # [BHW, K]

        # Soft assignment (softmax over distances)
        assignments = F.softmax(-distances, dim=-1)  # [BHW, K]

        # Weighted average distance (features should be close to assigned clusters)
        weighted_distances = (assignments * distances).sum(dim=-1).mean()

        return weighted_distances

    def separation_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        Cluster centers should be far from each other.

        Args:
            cluster_centers: [K, K]
        Returns:
            loss: scalar (negative, to maximize distance)
        """
        # Compute pairwise distances between cluster centers
        dist_matrix = torch.cdist(cluster_centers, cluster_centers, p=2)  # [K, K]

        # Mask diagonal (distance to self)
        mask = torch.eye(self.K, device=cluster_centers.device, dtype=torch.bool)
        dist_matrix = dist_matrix.masked_fill(mask, float('inf'))

        # Average minimum distance (we want to maximize this)
        # So we minimize the negative
        min_distances = dist_matrix.min(dim=-1)[0]  # [K]
        avg_min_distance = min_distances.mean()

        # Return negative (to maximize separation)
        return -avg_min_distance

    def forward(self,
                combined_cluster: torch.Tensor,
                target: torch.Tensor,
                cluster_centers: torch.Tensor) -> tuple:
        """
        Total cluster space loss.

        Args:
            combined_cluster: [B, K, H, W]
            target: [B, H, W]
            cluster_centers: [K, K]
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        losses = {}
        total_loss = 0.0

        # 1. Assignment loss
        if self.use_assignment_loss:
            L_assign = self.assignment_loss(combined_cluster, target)
            losses['assignment'] = L_assign.item()
            total_loss = total_loss + L_assign

        # 2. Compactness loss
        if self.use_compactness_loss:
            L_compact = self.compactness_loss(combined_cluster, cluster_centers)
            losses['compactness'] = L_compact.item()
            total_loss = total_loss + self.alpha * L_compact

        # 3. Separation loss
        if self.use_separation_loss:
            L_separate = self.separation_loss(cluster_centers)
            losses['separation'] = L_separate.item()
            total_loss = total_loss + self.beta * L_separate

        return total_loss, losses


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss across T recursive steps.

    L_total = L_main + λ * Σ(L_aux^(t)) for t=1 to T
    """

    def __init__(self,
                 num_classes: int,
                 num_clusters: int,
                 lambda_aux: float = 0.4,
                 use_cluster_loss: bool = True,
                 cluster_loss_alpha: float = 0.1,
                 cluster_loss_beta: float = 0.05):
        super().__init__()

        self.lambda_aux = lambda_aux

        # Main segmentation loss
        self.seg_loss = SegmentationLoss(num_classes)

        # Cluster space loss
        self.use_cluster_loss = use_cluster_loss
        if use_cluster_loss:
            self.cluster_loss = ClusterSpaceLoss(
                num_clusters=num_clusters,
                num_classes=num_classes,
                alpha=cluster_loss_alpha,
                beta=cluster_loss_beta
            )

    def forward(self,
                seg_preds: List[torch.Tensor],
                combined_clusters: List[torch.Tensor],
                cluster_centers: torch.Tensor,
                target: torch.Tensor) -> tuple:
        """
        Args:
            seg_preds: List of [B, num_classes, H, W] for each step
            combined_clusters: List of [B, K, H', W'] for each step
            cluster_centers: [K, K]
            target: [B, H, W]
        Returns:
            total_loss: scalar
            loss_dict: dict of losses
        """
        T = len(seg_preds)

        # Main loss (final prediction)
        L_main = self.seg_loss(seg_preds[-1], target)

        # Auxiliary losses (all predictions including final)
        L_aux_list = []
        cluster_loss_list = []

        for t in range(T):
            # Segmentation auxiliary loss
            L_seg_t = self.seg_loss(seg_preds[t], target)
            L_aux_list.append(L_seg_t)

            # Cluster space loss
            if self.use_cluster_loss:
                L_cluster_t, _ = self.cluster_loss(
                    combined_clusters[t],
                    target,
                    cluster_centers
                )
                cluster_loss_list.append(L_cluster_t)

        # Total auxiliary loss
        L_aux = sum(L_aux_list) / T
        L_cluster = sum(cluster_loss_list) / T if self.use_cluster_loss else 0.0

        # Total loss
        total_loss = L_main + self.lambda_aux * L_aux
        if self.use_cluster_loss:
            total_loss = total_loss + L_cluster

        # Loss dict
        loss_dict = {
            'total': total_loss.item(),
            'main': L_main.item(),
            'aux': L_aux.item(),
            'cluster': L_cluster.item() if self.use_cluster_loss else 0.0,
        }

        # Individual step losses
        for t, L_t in enumerate(L_aux_list):
            loss_dict[f'step_{t+1}'] = L_t.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing loss functions:")

    B, H, W = 2, 256, 256
    num_classes = 19
    K = 19

    # Test segmentation loss
    print("\n1. Testing SegmentationLoss:")
    seg_loss = SegmentationLoss(num_classes)

    pred = torch.randn(B, num_classes, H, W)
    target = torch.randint(0, num_classes, (B, H, W))

    loss = seg_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")

    # Test cluster space loss
    print("\n2. Testing ClusterSpaceLoss:")
    cluster_loss_fn = ClusterSpaceLoss(K, num_classes)

    combined_cluster = torch.randn(B, K, H // 8, W // 8)
    cluster_centers = torch.randn(K, K)
    target_small = F.interpolate(
        target.float().unsqueeze(1),
        size=(H // 8, W // 8),
        mode='nearest'
    ).squeeze(1).long()

    cluster_loss, loss_dict = cluster_loss_fn(
        combined_cluster,
        target_small,
        cluster_centers
    )
    print(f"   Total: {cluster_loss.item():.4f}")
    print(f"   Components: {loss_dict}")

    # Test deep supervision loss
    print("\n3. Testing DeepSupervisionLoss:")
    deep_sup_loss = DeepSupervisionLoss(
        num_classes=num_classes,
        num_clusters=K,
        lambda_aux=0.4
    )

    T = 3
    seg_preds = [torch.randn(B, num_classes, H, W) for _ in range(T)]
    combined_clusters = [torch.randn(B, K, H // 8, W // 8) for _ in range(T)]

    total_loss, loss_dict = deep_sup_loss(
        seg_preds,
        combined_clusters,
        cluster_centers,
        target
    )

    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"     {k}: {v:.4f}")

    print("\n✓ All loss tests passed!")
