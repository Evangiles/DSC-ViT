"""
Swin-DSC-ViT: Swin Transformer + Deep Supervised Clustering

Combines:
1. Swin Transformer: TRUE multi-scale feature extraction (64×64, 32×32, 16×16, 8×8)
2. FPN Fusion: Top-down multi-scale fusion
3. DSC-ViT Recursive Clustering: Iterative refinement with soft k-means
4. Deep Supervision: Multi-step gradient updates

Architecture:
    Swin Encoder (multi-scale) → FPN Fusion → Recursive Clustering → Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .swin_encoder import SwinEncoder
from .soft_kmeans import SoftKMeansLayer
from .projections import ProjectionLayers, GatedFusion, AttentionFusion


class SwinDSCViT(nn.Module):
    """
    Swin Transformer based DSC-ViT with TRUE multi-scale features.

    Key improvements over ViT-based DSC-ViT:
    - Multi-scale features: 64×64, 32×32, 16×16, 8×8 (not just 16×16)
    - Better boundary precision from high-res features
    - Better semantics from low-res features
    - FPN-style fusion for optimal feature utilization
    """

    def __init__(self,
                 # Image and output specs
                 image_channels: int = 3,
                 num_classes: int = 21,
                 img_size: int = 256,

                 # Clustering specs
                 num_clusters: Optional[int] = None,
                 cluster_temperature: float = 1.0,

                 # Architecture specs
                 latent_dim: int = 768,
                 swin_model_name: str = 'swin_base_patch4_window7_224',
                 use_pretrained_swin: bool = True,

                 # TRM-style recursive refinement
                 num_latent_updates: int = 4,
                 num_recursive_steps: int = 2,
                 fusion_method: str = 'residual',

                 # Training settings
                 freeze_swin: bool = False,
                 use_deep_supervision: bool = True,

                 # Multi-scale settings
                 extract_stages: List[int] = [0, 1, 2, 3],
                 fusion_resolution: int = 16):  # Resolution for clustering (16×16)
        """
        Args:
            image_channels: Input channels (C)
            num_classes: Number of segmentation classes
            img_size: Input image size
            num_clusters: Number of clusters (K). If None, K = num_classes
            cluster_temperature: Temperature for soft K-Means
            latent_dim: Feature dimension (D)
            swin_model_name: Swin model name from timm
            use_pretrained_swin: Use ImageNet pretrained weights
            num_latent_updates: Number of z updates per recursive step (n)
            num_recursive_steps: Number of recursive cycles (T)
            fusion_method: Feature fusion method ('residual', 'gated', 'attention')
            freeze_swin: Freeze Swin encoder
            use_deep_supervision: Enable deep supervision
            extract_stages: Which Swin stages to extract (0-3)
            fusion_resolution: Target resolution after FPN fusion (default: 16)
        """
        super().__init__()

        self.C = image_channels
        self.num_classes = num_classes
        self.K = num_clusters if num_clusters is not None else num_classes
        self.D = latent_dim
        self.n = num_latent_updates
        self.T = num_recursive_steps
        self.img_size = img_size
        self.fusion_method = fusion_method
        self.use_deep_supervision = use_deep_supervision
        self.fusion_resolution = fusion_resolution

        print(f"\n{'='*70}")
        print(f"Initializing Swin-DSC-ViT:")
        print(f"{'='*70}")
        print(f"  Image size: {img_size}×{img_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Num clusters (K): {self.K}")
        print(f"  Latent dim (D): {latent_dim}")
        print(f"  Latent updates (n): {self.n}")
        print(f"  Recursive steps (T): {self.T}")
        print(f"  Fusion resolution: {fusion_resolution}×{fusion_resolution}")
        print(f"  Swin model: {swin_model_name}")
        print(f"  Fusion method: {fusion_method}")

        # 1. Swin Encoder (multi-scale)
        self.encoder = SwinEncoder(
            model_name=swin_model_name,
            pretrained=use_pretrained_swin,
            img_size=img_size,
            output_dim=latent_dim,
            freeze_encoder=freeze_swin,
            extract_stages=extract_stages
        )

        # 2. FPN-style fusion layers
        # Fuse multi-scale features to target resolution
        num_skip = len(extract_stages) - 1  # Number of skip connections
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
            for _ in range(num_skip)
        ])

        self.fpn_output = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_skip)
        ])

        # 3. Projection layers (D ↔ K)
        self.projections = ProjectionLayers(
            image_channels=image_channels,
            num_clusters=self.K,
            latent_dim=latent_dim
        )

        # 4. Soft K-Means clustering
        self.clustering_layer = SoftKMeansLayer(
            num_clusters=self.K,
            feature_dim=self.K,
            temperature=cluster_temperature,
            init_method='orthogonal'
        )

        # 5. Fusion mechanism
        if fusion_method == 'gated':
            self.fusion = GatedFusion(num_clusters=self.K)
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(num_clusters=self.K)
        else:
            self.fusion = None

        # 6. Answer network (TRM-style y update)
        self.answer_network = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.GroupNorm(32, latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.GroupNorm(32, latent_dim)
        )

        # 7. Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=1),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, num_classes, kernel_size=1)
        )

        # 8. Q-head for ACT
        self.q_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=1),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, 1, kernel_size=1)
        )

        # 9. Final upsampling to original resolution
        self.upsample = nn.Upsample(
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        )

        print(f"{'='*70}\n")

    def fuse_multiscale_features(self, main_features: torch.Tensor,
                                skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        FPN-style top-down fusion to target resolution.

        Args:
            main_features: [B, D, H_deep, W_deep] - deepest stage (e.g., 8×8)
            skip_features: List of [B, D, H_i, W_i] - intermediate stages
                          Ordered shallow to deep (e.g., [64×64, 32×32, 16×16])
        Returns:
            fused: [B, D, fusion_resolution, fusion_resolution]
        """
        x = main_features  # Start from deepest

        # Top-down pathway
        for i in range(len(skip_features) - 1, -1, -1):
            # Upsample to match skip feature size
            target_size = skip_features[i].shape[2:]
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

            # Lateral connection + fusion
            lateral = self.fpn_lateral[i](skip_features[i])
            x = x + lateral
            x = self.fpn_output[i](x)

        # Resize to target fusion resolution if needed
        if x.shape[2] != self.fusion_resolution:
            x = F.interpolate(x, size=(self.fusion_resolution, self.fusion_resolution),
                            mode='bilinear', align_corners=False)

        return x

    def update_reasoning_latent(self, x_fused: torch.Tensor,
                               image_cluster: torch.Tensor,
                               y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update reasoning latent z via clustering.

        Args:
            x_fused: [B, D, H', W'] - fused multi-scale features (cached)
            image_cluster: [B, K, H', W'] - image in cluster space (cached)
            y: [B, D, H', W'] - answer latent
            z: [B, D, H', W'] - reasoning latent
        Returns:
            z_new: [B, D, H', W']
            z_clustered: [B, K, H', W']
        """
        # Project to cluster space
        z_cluster = self.projections.latent_to_cluster(z)  # D → K

        # Soft K-Means clustering
        z_clustered, _, _ = self.clustering_layer(z_cluster)

        # Project back to latent space
        z_latent = self.projections.cluster_to_latent(z_clustered)  # K → D

        # Combine with multi-scale features and answer
        z_new = z_latent + x_fused + y

        return z_new, z_clustered

    def update_answer_latent(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Update answer latent y."""
        combined = y + z
        y_new = self.answer_network(combined)
        return y_new

    def latent_recursion(self, x_fused: torch.Tensor, image_cluster: torch.Tensor,
                        y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One cycle of latent recursion (n z-updates + 1 y-update)."""
        combined_cluster = None
        for i in range(self.n):
            z, combined_cluster = self.update_reasoning_latent(x_fused, image_cluster, y, z)

        y = self.update_answer_latent(y, z)
        return y, z, combined_cluster

    def deep_recursion(self, x_fused: torch.Tensor, image_cluster: torch.Tensor,
                      y: torch.Tensor, z: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                 torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deep recursion with gradient approximation."""
        # T-1 iterations without gradient
        with torch.no_grad():
            for t in range(self.T - 1):
                y, z, _ = self.latent_recursion(x_fused, image_cluster, y, z)

        # Last iteration with gradient
        y, z, combined_cluster = self.latent_recursion(x_fused, image_cluster, y, z)

        # Generate predictions
        seg_pred = self.seg_head(y)
        q_logit = self.q_head(y)

        return (y.detach(), z.detach()), seg_pred, q_logit, combined_cluster

    def forward(self, image: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                z: Optional[torch.Tensor] = None) -> Tuple:
        """
        Forward pass with Swin encoder + recursive clustering.

        Args:
            image: [B, C, H, W]
            y: [B, D, H', W'] - answer latent (optional)
            z: [B, D, H', W'] - reasoning latent (optional)
        Returns:
            (y_detached, z_detached): For next supervision step
            seg_pred_full: [B, num_classes, H, W]
            q_logit_full: [B, 1, H, W]
            combined_cluster: [B, K, H', W']
            image_cluster: [B, K, H', W']
        """
        B, C, H, W = image.shape

        # Resize if needed
        if H != self.img_size or W != self.img_size:
            image_resized = F.interpolate(image, size=(self.img_size, self.img_size),
                                         mode='bilinear', align_corners=False)
        else:
            image_resized = image

        # 1. Multi-scale feature extraction with Swin
        main_feat, skip_feats = self.encoder(image_resized)

        # 2. FPN fusion to target resolution
        x_fused = self.fuse_multiscale_features(main_feat, skip_feats)  # [B, D, fusion_res, fusion_res]

        # 3. Project image to cluster space
        image_cluster = self.projections.image_to_cluster(image_resized)  # [B, K, H_enc, W_enc]
        if image_cluster.shape[2] != self.fusion_resolution:
            image_cluster = F.interpolate(image_cluster,
                                         size=(self.fusion_resolution, self.fusion_resolution),
                                         mode='bilinear', align_corners=False)

        # 4. Initialize y and z
        if y is None:
            y = x_fused.clone()
        if z is None:
            z = torch.zeros_like(y)

        # 5. Deep recursion with clustering
        (y_detached, z_detached), seg_pred, q_logit, combined_cluster = self.deep_recursion(
            x_fused, image_cluster, y, z
        )

        # 6. Upsample predictions to full resolution
        seg_pred_full = self.upsample(seg_pred)
        q_logit_full = self.upsample(q_logit)

        # Resize to original input size if needed
        if seg_pred_full.shape[2] != H or seg_pred_full.shape[3] != W:
            seg_pred_full = F.interpolate(seg_pred_full, size=(H, W),
                                         mode='bilinear', align_corners=False)
            q_logit_full = F.interpolate(q_logit_full, size=(H, W),
                                        mode='bilinear', align_corners=False)

        return (y_detached, z_detached), seg_pred_full, q_logit_full, combined_cluster, image_cluster

    def get_cluster_centers(self) -> torch.Tensor:
        """Return learned cluster centers."""
        return self.clustering_layer.get_cluster_centers()

    def update_temperature(self, new_temp: float):
        """Update clustering temperature."""
        if hasattr(self.clustering_layer, 'update_temperature'):
            self.clustering_layer.update_temperature(new_temp)


if __name__ == "__main__":
    print("Testing Swin-DSC-ViT:")

    # Create model
    model = SwinDSCViT(
        image_channels=3,
        num_classes=21,
        img_size=256,
        num_clusters=21,
        latent_dim=768,
        swin_model_name='swin_tiny_patch4_window7_224',  # Use tiny for testing
        use_pretrained_swin=False,
        num_latent_updates=4,
        num_recursive_steps=2,
        fusion_method='residual',
        fusion_resolution=16
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Test input
    image = torch.randn(2, 3, 256, 256)
    print(f"\nInput: {image.shape}")

    # Test forward (single supervision step)
    print("\nTesting single supervision step...")
    (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(image)

    print(f"\nOutputs:")
    print(f"  y: {y.shape}")
    print(f"  z: {z.shape}")
    print(f"  seg_pred: {seg_pred.shape}")
    print(f"  q_logit: {q_logit.shape}")
    print(f"  combined_cluster: {combined_cluster.shape}")

    # Test deep supervision loop
    print(f"\nTesting Deep Supervision (N_sup=3)...")
    y, z = None, None
    for step in range(3):
        (y, z), seg_pred, q_logit, _, _ = model(image, y, z)
        print(f"  Step {step+1}: seg_pred {seg_pred.shape}")

    print("\n✓ Swin-DSC-ViT integration complete!")
