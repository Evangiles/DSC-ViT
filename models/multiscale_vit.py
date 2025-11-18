"""
Multi-Scale Vision Transformer Encoder for DSC-ViT.

Extracts features from multiple depths of ViT for U-Net/FPN-style skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import timm


class MultiScaleViTEncoder(nn.Module):
    """
    Multi-scale ViT encoder that extracts features from multiple depths.

    Architecture:
    - Shallow layers (blocks 3-4): High resolution, low semantics (edges, textures)
    - Middle layers (blocks 7-8): Medium resolution, medium semantics
    - Deep layers (blocks 11-12): Low resolution, high semantics (object categories)

    For ViT-B/16:
    - 12 transformer blocks total
    - We extract from blocks: [3, 7, 11] (shallow, mid, deep)
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 output_dim: int = 768,
                 freeze_encoder: bool = False,
                 img_size: int = 256,
                 extract_layers: List[int] = [3, 7, 11]):
        """
        Args:
            model_name: ViT model name
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            freeze_encoder: Freeze ViT weights
            img_size: Input image size
            extract_layers: Which transformer blocks to extract features from
                           (0-indexed, for ViT-B/16 with 12 blocks: [3, 7, 11])
        """
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.img_size = img_size
        self.extract_layers = sorted(extract_layers)

        # Load ViT with feature extraction capability
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            features_only=False,  # We'll manually extract
        )

        # Get ViT embedding dimension
        if hasattr(self.vit, 'embed_dim'):
            self.vit_dim = self.vit.embed_dim
        elif hasattr(self.vit, 'num_features'):
            self.vit_dim = self.vit.num_features
        else:
            raise AttributeError("Cannot determine ViT embedding dimension")

        print(f"Multi-Scale ViT: {model_name}, embedding dim: {self.vit_dim}")
        print(f"  Extracting from layers: {extract_layers}")

        # Projection heads for each scale
        self.projections = nn.ModuleDict({
            f'layer_{layer}': nn.Sequential(
                nn.Linear(self.vit_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for layer in extract_layers
        })

        # Freeze if requested
        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT encoder frozen")

        # Patch size (usually 16 for ViT-B/16)
        self.patch_size = self.vit.patch_embed.patch_size[0] if hasattr(self.vit, 'patch_embed') else 16

    def forward_features_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from ViT.

        Args:
            x: [B, C, H, W] input image
        Returns:
            features: List of [B, D, H', W'] features at different depths
                     where H' = H // patch_size, W' = W // patch_size
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.vit.patch_embed(x)  # [B, num_patches, vit_dim]

        # Add class token (ViT includes [CLS] token)
        if hasattr(self.vit, 'cls_token'):
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, vit_dim]

        # Add positional embedding (after adding CLS token)
        if hasattr(self.vit, 'pos_embed'):
            x = x + self.vit.pos_embed

        # Apply position dropout
        if hasattr(self.vit, 'pos_drop'):
            x = self.vit.pos_drop(x)

        # Extract features from specified layers
        features = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)

            if i in self.extract_layers:
                # Remove [CLS] token (first token)
                feat = x[:, 1:, :] if hasattr(self.vit, 'cls_token') else x

                # Reshape to spatial: [B, num_patches, D] -> [B, D, H', W']
                num_patches = feat.shape[1]
                H_feat = W_feat = int(num_patches ** 0.5)
                feat = feat.transpose(1, 2).reshape(B, self.vit_dim, H_feat, W_feat)

                # Project to output_dim
                # [B, vit_dim, H', W'] -> [B, D, H', W']
                feat = feat.permute(0, 2, 3, 1)  # [B, H', W', vit_dim]
                feat = self.projections[f'layer_{i}'](feat)  # [B, H', W', D]
                feat = feat.permute(0, 3, 1, 2)  # [B, D, H', W']

                features.append(feat)

        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multi-scale features.

        Args:
            x: [B, C, H, W]
        Returns:
            main_features: [B, D, H', W'] - features from deepest layer
            skip_features: List of [B, D, H', W'] - features from intermediate layers
        """
        all_features = self.forward_features_multiscale(x)

        # Return deepest (most semantic) as main, others as skip connections
        main_features = all_features[-1]
        skip_features = all_features[:-1]

        return main_features, skip_features


class UNetDecoder(nn.Module):
    """
    Simple U-Net style decoder with skip connections.

    Upsamples from low resolution (16×16) to full resolution (256×256)
    while incorporating skip connections from encoder.
    """

    def __init__(self,
                 feature_dim: int = 768,
                 num_classes: int = 21,
                 skip_channels: List[int] = None):
        """
        Args:
            feature_dim: Feature dimension (D)
            num_classes: Number of output classes
            skip_channels: Channels for skip connections (if different per scale)
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Upsampling blocks with skip connections
        # Assume 3 scales: 16×16 (main), 16×16 (mid), 16×16 (shallow)
        # Note: ViT outputs same spatial size for all layers (16×16 for patch16)
        # So we only need to fuse features, not upsample between them

        # Fusion layers for skip connections
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(2)  # 2 skip connections (mid, shallow)
        ])

        # Final upsampling to original resolution
        # 16×16 -> 256×256 (16x upsampling)
        self.final_upsample = nn.Sequential(
            # 16×16 -> 32×32
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),

            # 32×32 -> 64×64
            nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True),

            # 64×64 -> 128×128
            nn.ConvTranspose2d(feature_dim // 4, feature_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 8),
            nn.ReLU(inplace=True),

            # 128×128 -> 256×256
            nn.ConvTranspose2d(feature_dim // 8, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, main_features: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            main_features: [B, D, 16, 16] - deepest features
            skip_features: List of [B, D, 16, 16] - [shallow, mid] features
        Returns:
            output: [B, num_classes, 256, 256]
        """
        x = main_features

        # Fuse skip connections (reverse order: mid -> shallow)
        for i, skip in enumerate(reversed(skip_features)):
            # Concatenate and fuse
            x = torch.cat([x, skip], dim=1)  # [B, 2D, 16, 16]
            x = self.fusion_blocks[i](x)     # [B, D, 16, 16]

        # Upsample to full resolution
        output = self.final_upsample(x)  # [B, num_classes, 256, 256]

        return output


if __name__ == "__main__":
    print("Testing Multi-Scale ViT Encoder:")

    # Create encoder
    encoder = MultiScaleViTEncoder(
        model_name='vit_base_patch16_224',
        pretrained=False,  # Fast testing
        output_dim=768,
        img_size=256,
        extract_layers=[3, 7, 11]
    )

    # Test input
    x = torch.randn(2, 3, 256, 256)

    # Forward
    main_feat, skip_feats = encoder(x)

    print(f"\nInput: {x.shape}")
    print(f"Main features (deep): {main_feat.shape}")
    print(f"Skip features:")
    for i, feat in enumerate(skip_feats):
        print(f"  Layer {encoder.extract_layers[i]}: {feat.shape}")

    # Test decoder
    decoder = UNetDecoder(feature_dim=768, num_classes=21)
    output = decoder(main_feat, skip_feats)
    print(f"\nDecoder output: {output.shape}")

    print("\n✓ Multi-scale ViT encoder ready!")
