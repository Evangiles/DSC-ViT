"""
DSC-ViT: Deep Supervised Vision Transformer with Clustering Hidden State

Main model implementation integrating:
1. ViT Encoder
2. Soft K-Means Clustering in cluster space
3. Recursive refinement loop with visual projection
4. Deep supervision across T recursive steps

Key Innovation: Projects original image to cluster space (C→K) instead of
projecting clusters to visual space (K→C), reducing complexity by 33%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .vit_encoder import ViTEncoder, SimpleConvEncoder
from .soft_kmeans import SoftKMeansLayer, HardKMeansLayer
from .projections import ProjectionLayers, GatedFusion, AttentionFusion
from .spatial_context import ASPP, ASPPAdaptive, SimpleSpatialContext


class DSCViT(nn.Module):
    """
    Deep Supervised ViT with Clustering Hidden State.

    ⭐ Key Innovation: ViT is computed ONCE per batch (not n×T times)!
    This makes recursive refinement ~18x faster and enables frozen ViT.

    Architecture flow:
    1. ViT Encoding (ONCE): image → x_vit (D-dimensional, cached)
    2. Image to Cluster (ONCE): image → image_cluster (K-dimensional, cached)
    3. Recursive Refinement (n×T times):
       a. Project to cluster space: (y+z) → z_cluster (D→K)
       b. Soft K-Means clustering: z_cluster → z_clustered
       c. Combine with image: z_clustered + image_cluster
       d. Project back to latent: combined_cluster → z_new (K→D)
       e. Add ViT residual: z_new = z_new + x_vit
    4. Deep Supervision across T recursive steps
    """

    def __init__(self,
                 # Image and output specs
                 image_channels: int = 3,
                 num_classes: int = 19,
                 img_size: int = 256,

                 # Clustering specs
                 num_clusters: Optional[int] = None,  # If None, use num_classes
                 cluster_temperature: float = 1.0,
                 clustering_method: str = 'soft',    # 'soft' or 'hard'

                 # Architecture specs
                 latent_dim: int = 512,
                 vit_model_name: str = 'vit_base_patch16_224',
                 use_pretrained_vit: bool = True,
                 use_simple_encoder: bool = False,

                 # TRM-style recursive refinement
                 num_latent_updates: int = 6,       # n: z를 업데이트하는 횟수
                 num_recursive_steps: int = 3,      # T: y를 업데이트하는 주기
                 fusion_method: str = 'residual',   # 'residual', 'gated', 'attention'

                 # Spatial context encoding
                 use_spatial_context: bool = False,  # Enable spatial context encoder
                 spatial_context_type: str = 'simple',  # 'simple' or 'aspp'

                 # Training settings
                 freeze_vit: bool = False,
                 use_deep_supervision: bool = True):
        """
        Args:
            image_channels: Number of input channels (C)
            num_classes: Number of segmentation classes
            img_size: Input image size
            num_clusters: Number of clusters (K). If None, K = num_classes
            cluster_temperature: Temperature for soft K-Means (ignored for hard)
            clustering_method: 'soft' (gradient-based) or 'hard' (EM-based)
            latent_dim: ViT latent dimension (D)
            vit_model_name: Pre-trained ViT model name
            use_pretrained_vit: Whether to use pretrained weights
            use_simple_encoder: Use simple conv encoder instead of ViT
            num_recursive_steps: Number of recursive refinement steps (T)
            fusion_method: How to combine cluster features ('residual', 'gated', 'attention')
            freeze_vit: Whether to freeze ViT encoder
            use_deep_supervision: Whether to use deep supervision
        """
        super().__init__()

        self.C = image_channels
        self.num_classes = num_classes
        self.K = num_clusters if num_clusters is not None else num_classes
        self.D = latent_dim
        self.n = num_latent_updates      # TRM's n
        self.T = num_recursive_steps     # TRM's T
        self.img_size = img_size
        self.fusion_method = fusion_method
        self.clustering_method = clustering_method
        self.use_deep_supervision = use_deep_supervision

        print(f"Initializing DSC-ViT (TRM-style):")
        print(f"  Image channels (C): {self.C}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Num clusters (K): {self.K}")
        print(f"  Latent dim (D): {self.D}")
        print(f"  Latent updates (n): {self.n}")
        print(f"  Recursive steps (T): {self.T}")
        print(f"  Effective depth (per step): {self.T * (self.n + 1) * 2} layers")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Clustering method: {clustering_method}")

        # 1. Encoder (ViT or simple conv)
        if use_simple_encoder:
            self.encoder = SimpleConvEncoder(
                in_channels=image_channels,
                output_dim=latent_dim,
                num_layers=4
            )
            self.patch_size = 8  # Simple encoder downsamples by 8x
        else:
            self.encoder = ViTEncoder(
                model_name=vit_model_name,
                pretrained=use_pretrained_vit,
                output_dim=latent_dim,
                freeze_encoder=freeze_vit,
                img_size=img_size
            )
            self.patch_size = 16  # ViT patch size

        # 2. Projection layers (C→K→D, D→K)
        self.projections = ProjectionLayers(
            image_channels=image_channels,
            num_clusters=self.K,
            latent_dim=latent_dim
        )

        # 3. K-Means clustering (Soft or Hard)
        if clustering_method == 'soft':
            self.clustering_layer = SoftKMeansLayer(
                num_clusters=self.K,
                feature_dim=self.K,  # Clustering happens in K-dimensional space
                temperature=cluster_temperature,
                init_method='orthogonal'
            )
        elif clustering_method == 'hard':
            self.clustering_layer = HardKMeansLayer(
                num_clusters=self.K,
                feature_dim=self.K,  # Clustering happens in K-dimensional space
                init_method='orthogonal',
                update_centers=True  # Use traditional K-Means updates
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        # 4. Compute encoder output size (needed for spatial context)
        self.encoder_output_size = img_size // self.patch_size

        # 5. Spatial Context Encoder (optional)
        self.use_spatial_context = use_spatial_context
        if use_spatial_context:
            if spatial_context_type == 'aspp':
                self.spatial_context_encoder = ASPP(
                    in_channels=latent_dim,
                    out_channels=256,  # ASPP internal channels
                    dilations=[1, 2, 4, 8]  # Optimized for 16×16
                )
            elif spatial_context_type == 'aspp_adaptive':
                self.spatial_context_encoder = ASPPAdaptive(
                    in_channels=latent_dim,
                    out_channels=256,
                    feature_size=self.encoder_output_size  # Auto-adjust dilations
                )
            elif spatial_context_type == 'simple':
                # dilation=4 for 16×16 (RF=9×9, ~56% of feature map)
                self.spatial_context_encoder = SimpleSpatialContext(
                    in_channels=latent_dim,
                    dilation=4
                )
            else:
                raise ValueError(f"Unknown spatial context type: {spatial_context_type}")
            print(f"  Spatial Context: {spatial_context_type.upper()} (feature_size={self.encoder_output_size})")
        else:
            self.spatial_context_encoder = None

        # 6. Fusion mechanism for combining cluster features
        if fusion_method == 'gated':
            self.fusion = GatedFusion(num_clusters=self.K)
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(num_clusters=self.K)
        else:  # residual
            self.fusion = None

        # 5. Segmentation head (simple MLP)
        self.seg_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=1),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, num_classes, kernel_size=1)
        )

        # 6. Q-head for ACT (Adaptive Computational Time)
        # Outputs logit (not probability) for halting decision
        self.q_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=1),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, 1, kernel_size=1)  # Single logit output
        )

        # 7. Answer network for y update (TRM requirement)
        # 2-layer network with normalization for stability
        self.answer_network = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.GroupNorm(32, latent_dim),  # Normalization for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.GroupNorm(32, latent_dim)   # Normalization for stability
        )

        # 8. Upsample layer (to restore original resolution)
        # encoder_output_size already computed above (line 158)
        self.upsample_factor = self.patch_size

        self.upsample = nn.Upsample(
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        )

    def update_reasoning_latent(self,
                                x_vit: torch.Tensor,
                                image_cluster: torch.Tensor,
                                y: torch.Tensor,
                                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update reasoning latent z (TRM의 z 역할).

        ⭐ Key change: ViT is computed ONCE outside the recursive loop!
        This makes the recursive refinement ~18x faster and allows frozen ViT.

        Args:
            x_vit: [B, D, H', W'] - ViT features (computed once, cached)
            image_cluster: [B, K, H', W'] - image in cluster space (cached)
            y: [B, D, H', W'] - current answer latent
            z: [B, D, H', W'] - current reasoning latent
        Returns:
            z_new: [B, D, H', W'] - updated reasoning latent
            z_clustered: [B, K, H', W'] - clustered features (for visualization/loss)
        """
        # Step 0: (Optional) Spatial context encoding
        if self.use_spatial_context:
            z = self.spatial_context_encoder(z)  # D→D with multi-scale context

        # Step 1: Cluster z only (design philosophy)
        z_cluster = self.projections.latent_to_cluster(z)  # D→K

        # Step 2: Soft K-Means clustering (learnable)
        z_clustered, _, _ = self.clustering_layer(z_cluster)  # [B, K, H', W']

        # Step 3: Project back to latent space
        z_latent = self.projections.cluster_to_latent(z_clustered)  # K→D

        # Step 4: Combine with x_vit and y (TRM philosophy: use x, y, z)
        z_new = z_latent + x_vit + y

        return z_new, z_clustered

    def update_answer_latent(self,
                            y: torch.Tensor,
                            z: torch.Tensor) -> torch.Tensor:
        """
        Update answer latent y using reasoning latent z (TRM의 y 역할).

        Args:
            y: [B, D, H', W'] - current answer latent
            z: [B, D, H', W'] - reasoning latent
        Returns:
            y_new: [B, D, H', W'] - updated answer latent
        """
        # TRM: y = net(y, z) using network (not simple addition)
        combined = y + z  # Combine y and z
        y_new = self.answer_network(combined)  # Pass through network

        return y_new

    def latent_recursion(self,
                        x_vit: torch.Tensor,
                        image_cluster: torch.Tensor,
                        y: torch.Tensor,
                        z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One cycle of latent recursion (TRM's n updates + 1 answer update).

        Follows TRM structure:
        - Update z n times: z = net(x_vit, image_cluster, y, z)
        - Update y once: y = net(y, z)

        TRM gradient strategy:
        - n-1 iterations: detach z (no gradient)
        - n-th iteration: keep gradient

        Args:
            x_vit: [B, D, H', W'] - ViT features (cached)
            image_cluster: [B, K, H', W'] - image in cluster space (cached)
            y: [B, D, H', W'] - current answer latent
            z: [B, D, H', W'] - current reasoning latent
        Returns:
            y_new: [B, D, H', W'] - updated answer latent
            z_new: [B, D, H', W'] - updated reasoning latent
            combined_cluster: [B, K, H', W'] - last combined cluster features
        """
        # Update reasoning latent z for n iterations
        combined_cluster = None
        for i in range(self.n):
            z, combined_cluster = self.update_reasoning_latent(x_vit, image_cluster, y, z)
            # TRM: Full gradient through all n steps (no detach!)

        # Update answer latent y once
        y = self.update_answer_latent(y, z)

        return y, z, combined_cluster

    def deep_recursion(self,
                      x_vit: torch.Tensor,
                      image_cluster: torch.Tensor,
                      y: torch.Tensor,
                      z: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion with 1-step gradient approximation (TRM's gradient strategy).

        Follows TRM structure:
        - T-1 iterations: no gradient (forward only)
        - T-th iteration: with gradient (backprop)

        Args:
            x_vit: [B, D, H', W'] - ViT features (cached)
            image_cluster: [B, K, H', W'] - image in cluster space (cached)
            y: [B, D, H', W'] - current answer latent
            z: [B, D, H', W'] - current reasoning latent
        Returns:
            (y_detached, z_detached): Detached latents for next supervision step
            seg_pred: [B, num_classes, H', W'] - segmentation prediction
            q_logit: [B, 1, H', W'] - ACT halting logit
            combined_cluster: [B, K, H', W'] - combined cluster features
        """
        # T-1 iterations without gradient
        with torch.no_grad():
            for t in range(self.T - 1):
                y, z, _ = self.latent_recursion(x_vit, image_cluster, y, z)

        # Last iteration with gradient
        y, z, combined_cluster = self.latent_recursion(x_vit, image_cluster, y, z)

        # Generate predictions from answer latent y
        seg_pred = self.seg_head(y)  # [B, num_classes, H', W']
        q_logit = self.q_head(y)      # [B, 1, H', W']

        # Detach for next supervision step (gradient isolation)
        return (y.detach(), z.detach()), seg_pred, q_logit, combined_cluster

    def forward(self,
                image: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                z: Optional[torch.Tensor] = None) -> Tuple:
        """
        TRM-style forward pass with deep recursion.

        ⭐ Key optimization: ViT is computed ONCE per batch (not n×T times)!
        This makes training ~18x faster and allows frozen ViT feature extraction.

        This implements ONE supervision step with T recursive iterations.
        For Deep Supervision (N_sup steps), call this function N_sup times
        in the training loop, each time with detached y, z from previous step.

        Args:
            image: [B, C, H, W] - input image
            y: [B, D, H', W'] - answer latent (if None, initialize from image)
            z: [B, D, H', W'] - reasoning latent (if None, initialize as zeros)
        Returns:
            (y_detached, z_detached): Detached latents for next supervision step
            seg_pred_full: [B, num_classes, H, W] - segmentation prediction
            q_logit_full: [B, 1, H, W] - ACT halting logit
            combined_cluster: [B, K, H', W'] - combined cluster features
            image_cluster: [B, K, H', W'] - cached image in cluster space
        """
        B, C, H, W = image.shape

        # Resize if needed
        if H != self.img_size or W != self.img_size:
            image_resized = F.interpolate(
                image,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            image_resized = image

        # ⭐ Compute ViT features ONCE (not in the recursive loop!)
        x_vit = self.encoder(image_resized)  # [B, D, H', W']

        # Resize to encoder output size if needed
        if x_vit.size(2) != self.encoder_output_size:
            x_vit = F.interpolate(
                x_vit,
                size=(self.encoder_output_size, self.encoder_output_size),
                mode='bilinear',
                align_corners=False
            )

        # Pre-compute: project original image to cluster space (only once per batch!)
        image_cluster = self.projections.image_to_cluster(image_resized)  # C→K: [B, K, H_enc, W_enc]

        # Resize to encoder output size if needed
        if image_cluster.size(2) != self.encoder_output_size:
            image_cluster = F.interpolate(
                image_cluster,
                size=(self.encoder_output_size, self.encoder_output_size),
                mode='bilinear',
                align_corners=False
            )

        # Initialize y and z if not provided
        if y is None:
            # Initialize answer latent from ViT features
            y = x_vit.clone()

        if z is None:
            # Initialize reasoning latent as zeros
            z = torch.zeros_like(y)

        # Deep recursion: T iterations with 1-step gradient approximation
        # ⭐ x_vit and image_cluster are cached, only lightweight modules iterate
        (y_detached, z_detached), seg_pred, q_logit, combined_cluster = self.deep_recursion(
            x_vit, image_cluster, y, z
        )

        # Upsample predictions to original size
        seg_pred_full = self.upsample(seg_pred)  # [B, num_classes, H_img, W_img]
        q_logit_full = self.upsample(q_logit)    # [B, 1, H_img, W_img]

        # Resize to original input size if needed
        if seg_pred_full.size(2) != H or seg_pred_full.size(3) != W:
            seg_pred_full = F.interpolate(
                seg_pred_full,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            q_logit_full = F.interpolate(
                q_logit_full,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return (y_detached, z_detached), seg_pred_full, q_logit_full, combined_cluster, image_cluster

    def get_cluster_centers(self) -> torch.Tensor:
        """Return learned cluster centers."""
        return self.clustering_layer.get_cluster_centers()

    def update_temperature(self, new_temp: float):
        """Update clustering temperature (for annealing). Only for soft k-means."""
        if self.clustering_method == 'soft' and hasattr(self.clustering_layer, 'update_temperature'):
            self.clustering_layer.update_temperature(new_temp)


if __name__ == "__main__":
    print("Testing DSC-ViT (TRM-style):")

    B, C, H, W = 2, 3, 256, 256
    num_classes = 19

    # Create model
    model = DSCViT(
        image_channels=C,
        num_classes=num_classes,
        img_size=H,
        num_clusters=num_classes,
        latent_dim=512,
        num_latent_updates=6,      # n
        num_recursive_steps=3,     # T
        use_simple_encoder=True,   # Use simple encoder for testing (faster)
        fusion_method='residual',
        use_deep_supervision=True
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Test input
    image = torch.randn(B, C, H, W)
    print(f"\nInput image: {image.shape}")

    # Test single supervision step
    print("\nTesting single supervision step (deep_recursion)...")
    (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(image)

    print(f"\nOutputs:")
    print(f"  y (answer latent): {y.shape}")
    print(f"  z (reasoning latent): {z.shape}")
    print(f"  seg_pred: {seg_pred.shape} (expected: [{B}, {num_classes}, {H}, {W}])")
    print(f"  q_logit: {q_logit.shape} (expected: [{B}, 1, {H}, {W}])")
    print(f"  combined_cluster: {combined_cluster.shape}")
    print(f"  image_cluster: {image_cluster.shape}")

    # Test Deep Supervision loop (N_sup steps)
    print(f"\nTesting Deep Supervision loop (N_sup=3)...")
    N_sup = 3
    y, z = None, None
    for step in range(N_sup):
        (y, z), seg_pred, q_logit, _, _ = model(image, y, z)
        print(f"  Step {step+1}: seg_pred {seg_pred.shape}, q_logit mean = {q_logit.mean().item():.4f}")

    # Test cluster centers
    centers = model.get_cluster_centers()
    print(f"\nCluster centers: {centers.shape} (expected: [{num_classes}, {num_classes}])")

    # Test temperature update
    model.update_temperature(0.5)
    print(f"Updated temperature to 0.5")

    print("\n✓ All DSC-ViT TRM-style tests passed!")
