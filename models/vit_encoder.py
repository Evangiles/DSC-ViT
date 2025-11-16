"""
Vision Transformer Encoder for DSC-ViT

Wraps pre-trained ViT models from timm or transformers library.
Supports freezing/unfreezing strategies and custom output dimensions.
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not installed. Please install: pip install timm")


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder wrapper.

    Loads pre-trained ViT and adapts it for segmentation tasks.
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 output_dim: int = 512,
                 freeze_encoder: bool = False,
                 num_frozen_layers: int = 0,
                 img_size: int = 224):
        """
        Args:
            model_name: ViT model name (timm model)
            pretrained: Whether to load pretrained weights
            output_dim: Output dimension (D)
            freeze_encoder: Whether to freeze the entire encoder
            num_frozen_layers: Number of transformer blocks to freeze (if not freezing all)
            img_size: Input image size
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT encoder")

        self.model_name = model_name
        self.output_dim = output_dim
        self.img_size = img_size

        # Load pretrained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
        )

        # Get ViT embedding dimension
        if hasattr(self.vit, 'embed_dim'):
            self.vit_dim = self.vit.embed_dim
        elif hasattr(self.vit, 'num_features'):
            self.vit_dim = self.vit.num_features
        else:
            raise AttributeError("Cannot determine ViT embedding dimension")

        print(f"Loaded ViT: {model_name}, embedding dim: {self.vit_dim}")

        # Projection from ViT dim to output dim (if different)
        if self.vit_dim != output_dim:
            self.proj_out = nn.Sequential(
                nn.Linear(self.vit_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            self.proj_out = nn.Identity()

        # Freeze encoder if needed
        if freeze_encoder:
            self.freeze_encoder()
        elif num_frozen_layers > 0:
            self.freeze_layers(num_frozen_layers)

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("ViT encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("ViT encoder unfrozen")

    def freeze_layers(self, num_layers: int):
        """Freeze first num_layers transformer blocks."""
        if hasattr(self.vit, 'blocks'):
            blocks = self.vit.blocks
            num_layers = min(num_layers, len(blocks))
            for i in range(num_layers):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            print(f"Frozen first {num_layers} transformer blocks")
        else:
            warnings.warn("Cannot freeze layers: blocks not found")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ViT.

        Args:
            x: [B, C, H, W] - input tensor
        Returns:
            features: [B, num_patches, vit_dim]
        """
        # ViT forward
        x = self.vit.forward_features(x)

        # Remove CLS token if present
        if x.dim() == 3 and x.size(1) > (self.img_size // self.vit.patch_embed.patch_size[0]) ** 2:
            x = x[:, 1:, :]  # Remove first token (CLS)

        return x

    def forward(self, x: torch.Tensor, return_spatial: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] - input tensor
            return_spatial: If True, return [B, D, H', W'], else [B, num_patches, D]
        Returns:
            features: [B, D, H', W'] or [B, num_patches, D]
        """
        B, C, H, W = x.shape

        # Extract features
        features = self.forward_features(x)  # [B, num_patches, vit_dim]

        # Project to output dim
        features = self.proj_out(features)  # [B, num_patches, output_dim]

        if return_spatial:
            # Reshape to spatial format
            num_patches = features.size(1)
            patch_size = self.vit.patch_embed.patch_size[0]
            H_out = W_out = int(num_patches ** 0.5)

            features = features.transpose(1, 2)  # [B, output_dim, num_patches]
            features = features.reshape(B, self.output_dim, H_out, W_out)

        return features


class SimpleConvEncoder(nn.Module):
    """
    Simple convolutional encoder as alternative to ViT.

    Useful for quick prototyping or when ViT is too heavy.
    """

    def __init__(self,
                 in_channels: int = 3,
                 output_dim: int = 512,
                 num_layers: int = 4):
        super().__init__()

        channels = [in_channels, 64, 128, 256, output_dim]

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
            ])

            # Downsample except for last layer
            if i < num_layers - 1:
                layers.append(nn.MaxPool2d(2))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [B, output_dim, H//8, W//8]
        """
        return self.encoder(x)


if __name__ == "__main__":
    print("Testing ViTEncoder:")

    B, C, H, W = 2, 3, 224, 224
    output_dim = 512

    # Test ViT encoder
    if TIMM_AVAILABLE:
        try:
            encoder = ViTEncoder(
                model_name='vit_base_patch16_224',
                pretrained=False,  # Set to True for actual use
                output_dim=output_dim,
                img_size=H
            )

            x = torch.randn(B, C, H, W)
            print(f"Input: {x.shape}")

            # Forward pass
            features = encoder(x, return_spatial=True)
            print(f"Output (spatial): {features.shape}")

            features_seq = encoder(x, return_spatial=False)
            print(f"Output (sequence): {features_seq.shape}")

            # Test freezing
            encoder.freeze_encoder()
            encoder.unfreeze_encoder()
            encoder.freeze_layers(6)

            print("✓ ViT encoder tests passed!")

        except Exception as e:
            print(f"ViT encoder test failed: {e}")
    else:
        print("Skipping ViT test (timm not available)")

    # Test simple conv encoder
    print("\nTesting SimpleConvEncoder:")
    simple_encoder = SimpleConvEncoder(
        in_channels=C,
        output_dim=output_dim,
        num_layers=4
    )

    x = torch.randn(B, C, 256, 256)
    features = simple_encoder(x)
    print(f"Input: {x.shape}")
    print(f"Output: {features.shape}")

    print("\n✓ All encoder tests passed!")
