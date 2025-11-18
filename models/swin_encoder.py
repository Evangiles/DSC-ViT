"""
Swin Transformer Encoder for DSC-ViT.

Provides TRUE multi-scale features with hierarchical architecture:
- Stage 1: [B, 96,  64×64]   (High resolution, low semantics)
- Stage 2: [B, 192, 32×32]   (Medium resolution)
- Stage 3: [B, 384, 16×16]   (Low resolution, high semantics)
- Stage 4: [B, 768, 8×8]     (Lowest resolution, highest semantics)

Unlike standard ViT which maintains 16×16 throughout, Swin provides
hierarchical features at multiple resolutions for better segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import timm


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder wrapper for multi-scale feature extraction.

    Supports:
    - Swin-T (Tiny): 28M params
    - Swin-S (Small): 50M params
    - Swin-B (Base): 88M params
    - Swin-L (Large): 197M params
    """

    def __init__(self,
                 model_name: str = 'swin_base_patch4_window7_224',
                 pretrained: bool = True,
                 img_size: int = 256,
                 output_dim: int = 768,
                 freeze_encoder: bool = False,
                 extract_stages: List[int] = [0, 1, 2, 3]):
        """
        Args:
            model_name: Swin model name from timm
                       Options: swin_tiny_patch4_window7_224
                               swin_small_patch4_window7_224
                               swin_base_patch4_window7_224
                               swin_large_patch4_window7_224
            pretrained: Use ImageNet pretrained weights
            img_size: Input image size
            output_dim: Output dimension for main features
            freeze_encoder: Freeze all Swin parameters
            extract_stages: Which stages to extract (0-3)
                           Stage 0: 64×64 (or H/4×W/4)
                           Stage 1: 32×32 (or H/8×W/8)
                           Stage 2: 16×16 (or H/16×W/16)
                           Stage 3: 8×8   (or H/32×W/32)
        """
        super().__init__()

        self.model_name = model_name
        self.img_size = img_size
        self.output_dim = output_dim
        self.extract_stages = sorted(extract_stages)

        # Load Swin Transformer with feature extraction
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,  # Enable multi-scale feature extraction
            out_indices=extract_stages,  # Which stages to output
        )

        # Get feature dimensions at each stage
        self.feature_info = self.swin.feature_info
        self.stage_dims = [info['num_chs'] for info in self.feature_info]
        self.stage_reductions = [info['reduction'] for info in self.feature_info]

        print(f"\nSwin Transformer Encoder: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Image size: {img_size}")
        print(f"  Extracting from stages: {extract_stages}")
        print(f"\nStage Information:")
        for i, (dim, reduction) in enumerate(zip(self.stage_dims, self.stage_reductions)):
            spatial = img_size // reduction
            print(f"  Stage {extract_stages[i]}: {dim} channels, {spatial}×{spatial} spatial")

        # Projection heads to unify dimensions
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.GELU()
            ) for dim in self.stage_dims
        ])

        # Freeze if requested
        if freeze_encoder:
            for param in self.swin.parameters():
                param.requires_grad = False
            print("  Swin encoder frozen")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract multi-scale features from Swin Transformer.

        Args:
            x: [B, C, H, W] input image
        Returns:
            main_features: [B, D, H', W'] - deepest stage features
            skip_features: List of [B, D, H_i, W_i] - intermediate stage features
                          Ordered from shallow to deep (excluding deepest)
        """
        # Extract multi-scale features from Swin
        # Returns list of features at different scales
        stage_features = self.swin(x)  # List of [B, H_i, W_i, C_i] (NHWC format)

        # Project all features to output_dim
        projected_features = []
        for i, feat in enumerate(stage_features):
            # Convert from NHWC to NCHW format
            if feat.dim() == 4 and feat.shape[-1] == self.stage_dims[i]:
                feat = feat.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            proj_feat = self.projections[i](feat)  # [B, D, H_i, W_i]
            projected_features.append(proj_feat)

        # Return deepest as main, others as skip connections
        main_features = projected_features[-1]
        skip_features = projected_features[:-1]

        return main_features, skip_features

    def forward_all_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get features from all stages (useful for debugging/visualization).

        Args:
            x: [B, C, H, W]
        Returns:
            features: List of [B, D, H_i, W_i] for each stage
        """
        stage_features = self.swin(x)
        projected_features = []
        for i, feat in enumerate(stage_features):
            # Convert from NHWC to NCHW format
            if feat.dim() == 4 and feat.shape[-1] == self.stage_dims[i]:
                feat = feat.permute(0, 3, 1, 2)
            projected_features.append(self.projections[i](feat))
        return projected_features


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) style decoder for multi-scale fusion.

    Fuses features from multiple scales using top-down pathway with
    lateral connections (skip connections from encoder).
    """

    def __init__(self,
                 feature_dim: int = 768,
                 num_classes: int = 21,
                 num_stages: int = 4,
                 img_size: int = 256):
        """
        Args:
            feature_dim: Unified feature dimension (D)
            num_classes: Number of output classes
            num_stages: Number of encoder stages (default: 4 for Swin)
            img_size: Original image size for final upsampling
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size

        # Top-down pathway: upsample and fuse
        # Stage 3 (8×8) -> Stage 2 (16×16) -> Stage 1 (32×32) -> Stage 0 (64×64)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
            for _ in range(num_stages - 1)  # Lateral connections
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages - 1)  # FPN fusion layers
        ])

        # Final segmentation head
        # Upsample from highest resolution (64×64 for 256×256 input)
        # to full resolution (256×256)
        self.seg_head = nn.Sequential(
            # 64×64 -> 128×128
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),

            # 128×128 -> 256×256
            nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True),

            # Final classification
            nn.Conv2d(feature_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, main_features: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        FPN-style top-down fusion.

        Args:
            main_features: [B, D, H_deep, W_deep] - deepest stage (e.g., 8×8)
            skip_features: List of [B, D, H_i, W_i] - [Stage0, Stage1, Stage2]
                          Ordered from shallow to deep (e.g., [64×64, 32×32, 16×16])
        Returns:
            output: [B, num_classes, img_size, img_size]
        """
        # Start from deepest features
        x = main_features  # [B, D, 8, 8]

        # Top-down pathway: fuse with skip connections
        # Process in reverse order: Stage 2 -> Stage 1 -> Stage 0
        for i in range(len(skip_features) - 1, -1, -1):
            # Upsample x to match skip feature size
            target_size = skip_features[i].shape[2:]
            x_up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

            # Lateral connection
            lateral = self.lateral_convs[i](skip_features[i])

            # Fuse
            x = x_up + lateral

            # Apply FPN conv
            x = self.fpn_convs[i](x)

        # Now x is at the shallowest resolution (e.g., 64×64)
        # Upsample to full resolution and classify
        output = self.seg_head(x)  # [B, num_classes, img_size, img_size]

        return output


if __name__ == "__main__":
    print("="*70)
    print("Testing Swin Transformer Encoder")
    print("="*70)

    # Test encoder
    encoder = SwinEncoder(
        model_name='swin_tiny_patch4_window7_224',  # Use tiny for faster testing
        pretrained=False,
        img_size=256,
        output_dim=768,
        extract_stages=[0, 1, 2, 3]
    )

    # Test input
    x = torch.randn(2, 3, 256, 256)
    print(f"\nInput: {x.shape}")

    # Forward
    main_feat, skip_feats = encoder(x)

    print(f"\n{'='*70}")
    print("Multi-Scale Features (TRUE multi-scale!):")
    print(f"{'='*70}")
    for i, feat in enumerate(skip_feats):
        print(f"Stage {i}: {feat.shape} - {feat.shape[2]}×{feat.shape[3]} resolution")
    print(f"Stage {len(skip_feats)} (main): {main_feat.shape} - {main_feat.shape[2]}×{main_feat.shape[3]} resolution")

    # Test decoder
    print(f"\n{'='*70}")
    print("Testing FPN Decoder:")
    print(f"{'='*70}")
    decoder = FPNDecoder(
        feature_dim=768,
        num_classes=21,
        num_stages=4,
        img_size=256
    )

    output = decoder(main_feat, skip_feats)
    print(f"Decoder output: {output.shape}")

    print(f"\n{'='*70}")
    print("✓ Swin Transformer encoder ready!")
    print(f"{'='*70}")
