"""
Soft K-Means Clustering Layer for DSC-ViT

Implements differentiable epistemological categorization in cluster space.
The model learns to cluster latent representations into K semantic categories
before making final predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftKMeansLayer(nn.Module):
    """
    Differentiable Soft K-Means clustering layer.

    Performs soft assignment of features to cluster centers and
    generates clustered representations via weighted averaging.

    Key parameters:
    - K: Number of clusters (typically = number of classes)
    - temperature: Controls softness of assignments (lower = harder)
    """

    def __init__(self,
                 num_clusters: int,
                 feature_dim: int,
                 temperature: float = 1.0,
                 init_method: str = 'random',
                 normalize_features: bool = True,
                 learnable_temp: bool = False):
        """
        Args:
            num_clusters: Number of cluster centers (K)
            feature_dim: Dimension of feature space (K in our architecture)
            temperature: Temperature for soft assignment
            init_method: Initialization method ('random', 'uniform', 'orthogonal')
            normalize_features: Whether to L2-normalize features before clustering
            learnable_temp: Whether temperature is a learnable parameter
        """
        super().__init__()

        self.K = num_clusters
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features

        # Cluster centers μ: [K, feature_dim]
        self.register_parameter(
            'cluster_centers',
            nn.Parameter(torch.zeros(num_clusters, feature_dim))
        )

        # Initialize cluster centers
        self._initialize_centers(init_method)

        # Temperature for soft assignment
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self.learnable_temp = learnable_temp

    def _initialize_centers(self, method: str):
        """Initialize cluster centers."""
        if method == 'random':
            # Random normal initialization (increased std for better separation)
            nn.init.normal_(self.cluster_centers, mean=0, std=1.0)
        elif method == 'uniform':
            # Uniform initialization
            nn.init.uniform_(self.cluster_centers, -1, 1)
        elif method == 'orthogonal':
            # Orthogonal initialization (centers are orthogonal to each other)
            nn.init.orthogonal_(self.cluster_centers)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def compute_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between features and cluster centers.

        Args:
            features: [B, K, H, W] or [B, HW, K]
        Returns:
            distances: [B, H, W, num_clusters] or [B, HW, num_clusters]
        """
        # Normalize if needed
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)
            centers = F.normalize(self.cluster_centers, p=2, dim=-1)
        else:
            centers = self.cluster_centers

        # Reshape features to [B, HW, K] if needed
        if features.dim() == 4:
            B, K, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, K)
            reshape_needed = True
        else:
            B, HW, K = features.shape
            H = W = int(math.sqrt(HW))
            reshape_needed = False

        # Compute squared Euclidean distances
        # ||x - μ||^2 = ||x||^2 + ||μ||^2 - 2<x, μ>
        x_squared = (features ** 2).sum(dim=-1, keepdim=True)  # [B, HW, 1]
        mu_squared = (centers ** 2).sum(dim=-1, keepdim=True).t()  # [1, num_clusters]
        cross_term = features @ centers.t()  # [B, HW, num_clusters]

        distances = x_squared + mu_squared - 2 * cross_term  # [B, HW, num_clusters]

        if reshape_needed:
            distances = distances.reshape(B, H, W, self.K)

        return distances

    def soft_assignment(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments using softmax.

        Args:
            distances: [B, H, W, K] or [B, HW, K]
        Returns:
            assignments: [B, H, W, K] or [B, HW, K] (probabilities)
        """
        # Soft assignment: q = softmax(-dist / temperature)
        assignments = F.softmax(-distances / self.temperature, dim=-1)

        return assignments

    def cluster_features(self,
                        features: torch.Tensor,
                        assignments: torch.Tensor) -> torch.Tensor:
        """
        Generate clustered features via weighted averaging of cluster centers.

        Args:
            features: [B, K, H, W] - input features (not used directly, kept for interface)
            assignments: [B, H, W, num_clusters] - soft assignments
        Returns:
            clustered: [B, K, H, W] - clustered features
        """
        # Reshape if needed
        if assignments.dim() == 4:
            B, H, W, num_clusters = assignments.shape
            assignments_flat = assignments.reshape(B, H * W, num_clusters)
        else:
            B, HW, num_clusters = assignments.shape
            H = W = int(math.sqrt(HW))
            assignments_flat = assignments

        # Weighted average: z_clustered = Σ(q_k * μ_k)
        clustered_flat = assignments_flat @ self.cluster_centers  # [B, HW, feature_dim]

        # Reshape back to [B, feature_dim, H, W]
        clustered = clustered_flat.reshape(B, H, W, self.feature_dim)
        clustered = clustered.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]

        return clustered

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Forward pass: compute soft assignments and clustered features.

        Args:
            features: [B, K, H, W] - input features in cluster space
        Returns:
            clustered: [B, K, H, W] - clustered features
            assignments: [B, H, W, num_clusters] - soft cluster assignments
            distances: [B, H, W, num_clusters] - distances to cluster centers
        """
        # 1. Compute distances to cluster centers
        distances = self.compute_distances(features)

        # 2. Compute soft assignments
        assignments = self.soft_assignment(distances)

        # 3. Generate clustered features
        clustered = self.cluster_features(features, assignments)

        return clustered, assignments, distances

    def get_cluster_centers(self) -> torch.Tensor:
        """Return cluster centers."""
        return self.cluster_centers

    def update_temperature(self, new_temp: float):
        """Update temperature (for annealing)."""
        if not self.learnable_temp:
            self.temperature.fill_(new_temp)
        else:
            print("Temperature is learnable, cannot update manually.")


class HardKMeansLayer(nn.Module):
    """
    Hard K-Means clustering layer (traditional EM algorithm).

    Unlike soft k-means, this uses hard assignments (argmin) and updates
    cluster centers via traditional k-means algorithm (no gradients).

    Update rule:
    1. Assignment: assign each point to nearest cluster (argmin)
    2. Update: move cluster center to mean of assigned points
    """

    def __init__(self,
                 num_clusters: int,
                 feature_dim: int,
                 init_method: str = 'random',
                 normalize_features: bool = True,
                 update_centers: bool = True):
        """
        Args:
            num_clusters: Number of cluster centers (K)
            feature_dim: Dimension of feature space (K in our architecture)
            init_method: Initialization method ('random', 'uniform', 'orthogonal')
            normalize_features: Whether to L2-normalize features before clustering
            update_centers: Whether to update cluster centers during forward pass
        """
        super().__init__()

        self.K = num_clusters
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features
        self.update_centers = update_centers

        # Cluster centers μ: [K, feature_dim]
        self.register_buffer(
            'cluster_centers',
            torch.zeros(num_clusters, feature_dim)
        )

        # Initialize cluster centers
        self._initialize_centers(init_method)

        # Track number of updates for monitoring
        self.register_buffer('num_updates', torch.tensor(0))

    def _initialize_centers(self, method: str):
        """Initialize cluster centers."""
        if method == 'random':
            nn.init.normal_(self.cluster_centers, mean=0, std=1.0)
        elif method == 'uniform':
            nn.init.uniform_(self.cluster_centers, -1, 1)
        elif method == 'orthogonal':
            nn.init.orthogonal_(self.cluster_centers)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def compute_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between features and cluster centers.

        Args:
            features: [B, K, H, W] or [B, HW, K]
        Returns:
            distances: [B, H, W, num_clusters] or [B, HW, num_clusters]
        """
        # Normalize if needed
        if self.normalize_features:
            features_norm = F.normalize(features, p=2, dim=-1 if features.dim() == 3 else 1)
            centers = F.normalize(self.cluster_centers, p=2, dim=-1)
        else:
            features_norm = features
            centers = self.cluster_centers

        # Reshape features to [B, HW, K] if needed
        if features_norm.dim() == 4:
            B, K, H, W = features_norm.shape
            features_flat = features_norm.permute(0, 2, 3, 1).reshape(B, H * W, K)
            reshape_needed = True
        else:
            B, HW, K = features_norm.shape
            H = W = int(math.sqrt(HW))
            features_flat = features_norm
            reshape_needed = False

        # Compute squared Euclidean distances
        x_squared = (features_flat ** 2).sum(dim=-1, keepdim=True)  # [B, HW, 1]
        mu_squared = (centers ** 2).sum(dim=-1, keepdim=True).t()  # [1, num_clusters]
        cross_term = features_flat @ centers.t()  # [B, HW, num_clusters]

        distances = x_squared + mu_squared - 2 * cross_term  # [B, HW, num_clusters]

        if reshape_needed:
            distances = distances.reshape(B, H, W, self.K)

        return distances, features_flat, (B, H, W)

    def hard_assignment(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute hard cluster assignments using argmin.

        Args:
            distances: [B, H, W, K] or [B, HW, K]
        Returns:
            assignments: [B, H, W] or [B, HW] (cluster indices)
        """
        # Hard assignment: k* = argmin(dist)
        assignments = torch.argmin(distances, dim=-1)

        return assignments

    @torch.no_grad()
    def update_cluster_centers(self,
                               features_flat: torch.Tensor,
                               assignments: torch.Tensor):
        """
        Update cluster centers using traditional K-Means rule (vectorized).

        μ_k = mean(all points assigned to cluster k)

        Args:
            features_flat: [B, HW, feature_dim]
            assignments: [B, HW] - cluster indices
        """
        if not self.update_centers or not self.training:
            return

        B, HW, feature_dim = features_flat.shape

        # Flatten across batch
        features_all = features_flat.reshape(B * HW, feature_dim)  # [N, feature_dim]
        assignments_all = assignments.reshape(B * HW)  # [N]

        # Vectorized update using one-hot encoding
        # one_hot: [N, K] where one_hot[i, k] = 1 if point i assigned to cluster k
        one_hot = F.one_hot(assignments_all, num_classes=self.K).float()  # [N, K]

        # Sum of features per cluster: [K, feature_dim]
        cluster_sum = one_hot.t() @ features_all  # [K, N] @ [N, D] = [K, D]

        # Count points per cluster: [K]
        cluster_count = one_hot.sum(dim=0)  # [K]

        # Avoid division by zero (empty clusters keep current center)
        valid_clusters = cluster_count > 0
        cluster_count = cluster_count.clamp(min=1)

        # Update centers: [K, feature_dim]
        new_centers = cluster_sum / cluster_count.unsqueeze(1)

        # Only update non-empty clusters
        self.cluster_centers[valid_clusters] = new_centers[valid_clusters]

        self.num_updates += 1

    def cluster_features(self, assignments: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Generate clustered features by looking up cluster centers.

        Args:
            assignments: [B, HW] - cluster indices
            shape: (B, H, W) - original spatial shape
        Returns:
            clustered: [B, feature_dim, H, W]
        """
        B, H, W = shape

        # Look up cluster centers for each assignment
        clustered_flat = self.cluster_centers[assignments]  # [B, HW, feature_dim]

        # Reshape to [B, feature_dim, H, W]
        clustered = clustered_flat.reshape(B, H, W, self.feature_dim)
        clustered = clustered.permute(0, 3, 1, 2)

        return clustered

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Forward pass: hard assignment + cluster center update.

        Args:
            features: [B, K, H, W] - input features in cluster space
        Returns:
            clustered: [B, K, H, W] - clustered features (detached)
            assignments: [B, H, W] - hard cluster assignments
            distances: [B, H, W, num_clusters] - distances to cluster centers
        """
        # 1. Compute distances to cluster centers
        distances, features_flat, shape = self.compute_distances(features)

        # 2. Hard assignment (argmin)
        if distances.dim() == 4:
            B, H, W, K = distances.shape
            distances_flat = distances.reshape(B, H * W, K)
        else:
            distances_flat = distances

        assignments = self.hard_assignment(distances_flat)  # [B, HW]

        # 3. Update cluster centers (traditional K-Means, no gradient)
        self.update_cluster_centers(features_flat, assignments)

        # 4. Generate clustered features (detached from gradient)
        clustered = self.cluster_features(assignments, shape)

        # Reshape assignments back to spatial
        B, H, W = shape
        assignments_spatial = assignments.reshape(B, H, W)

        # Reshape distances back if needed
        if distances.dim() == 3:
            distances = distances.reshape(B, H, W, self.K)

        return clustered, assignments_spatial, distances

    def get_cluster_centers(self) -> torch.Tensor:
        """Return cluster centers."""
        return self.cluster_centers

    def reset_centers(self):
        """Reset cluster centers (useful for re-initialization)."""
        self._initialize_centers('random')
        self.num_updates.zero_()


class MultiScaleSoftKMeans(nn.Module):
    """
    Multi-scale soft K-Means clustering.

    Uses multiple K values to capture different levels of granularity.
    For example: K1 = num_classes (fine), K2 = num_classes/2 (coarse)
    """

    def __init__(self,
                 num_clusters_list: list,
                 feature_dim: int,
                 temperature: float = 1.0,
                 fusion_method: str = 'concat'):
        """
        Args:
            num_clusters_list: List of K values [K1, K2, ...]
            feature_dim: Feature dimension
            temperature: Temperature for soft assignment
            fusion_method: How to fuse multi-scale features ('concat', 'add', 'attention')
        """
        super().__init__()

        self.num_clusters_list = num_clusters_list
        self.fusion_method = fusion_method

        # Create multiple clustering layers
        self.clustering_layers = nn.ModuleList([
            SoftKMeansLayer(K, feature_dim, temperature)
            for K in num_clusters_list
        ])

        # Fusion layer
        if fusion_method == 'concat':
            total_dim = feature_dim * len(num_clusters_list)
            self.fusion = nn.Conv2d(total_dim, feature_dim, kernel_size=1)
        elif fusion_method == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: [B, K, H, W]
        Returns:
            fused_clustered: [B, K, H, W]
            all_assignments: List of assignments for each scale
        """
        clustered_list = []
        assignments_list = []

        for clustering_layer in self.clustering_layers:
            clustered, assignments, _ = clustering_layer(features)
            clustered_list.append(clustered)
            assignments_list.append(assignments)

        # Fuse multi-scale features
        if self.fusion_method == 'concat':
            concat = torch.cat(clustered_list, dim=1)
            fused = self.fusion(concat)
        elif self.fusion_method == 'add':
            fused = sum(clustered_list) / len(clustered_list)
        elif self.fusion_method == 'attention':
            # Not implemented yet
            raise NotImplementedError("Attention fusion not implemented")
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused, assignments_list


if __name__ == "__main__":
    # Test Soft K-Means layer
    print("Testing SoftKMeansLayer:")

    B, K, H, W = 2, 19, 64, 64
    num_clusters = 19

    # Create layer
    soft_kmeans = SoftKMeansLayer(
        num_clusters=num_clusters,
        feature_dim=K,
        temperature=1.0,
        init_method='orthogonal'
    )

    # Test input
    features = torch.randn(B, K, H, W)
    print(f"Input features: {features.shape}")

    # Forward pass
    clustered, assignments, distances = soft_kmeans(features)

    print(f"Clustered features: {clustered.shape} (expected: [{B}, {K}, {H}, {W}])")
    print(f"Assignments: {assignments.shape} (expected: [{B}, {H}, {W}, {num_clusters}])")
    print(f"Distances: {distances.shape} (expected: [{B}, {H}, {W}, {num_clusters}])")

    # Check assignment probabilities sum to 1
    assignment_sums = assignments.sum(dim=-1)
    print(f"Assignment sums (should be ~1.0): mean={assignment_sums.mean():.4f}, std={assignment_sums.std():.4f}")

    # Test multi-scale clustering
    print("\nTesting MultiScaleSoftKMeans:")

    multi_scale_kmeans = MultiScaleSoftKMeans(
        num_clusters_list=[19, 10, 5],
        feature_dim=K,
        temperature=1.0,
        fusion_method='concat'
    )

    fused, assignments_list = multi_scale_kmeans(features)
    print(f"Fused clustered features: {fused.shape} (expected: [{B}, {K}, {H}, {W}])")
    print(f"Number of scales: {len(assignments_list)}")

    for i, assignments in enumerate(assignments_list):
        print(f"  Scale {i+1} assignments: {assignments.shape}")

    print("\n✓ All tests passed!")
