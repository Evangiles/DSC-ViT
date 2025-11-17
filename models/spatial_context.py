"""
Spatial Context Encoding modules for DSC-ViT.

Enhances feature representations with multi-scale spatial context
before clustering to improve boundary precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """
    Atrous (Dilated) Convolution block.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        dilation: Dilation rate
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Module):
    """
    Global Average Pooling + 1x1 Conv (captures image-level context).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.gap(x)  # [B, C, 1, 1]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # Upsample to original spatial size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) adapted for low-resolution features.

    Captures multi-scale context using parallel atrous convolutions
    with different dilation rates.

    ⚠️ NOTE: Dilation rates are adjusted for 16×16 feature maps.
    Original DeepLab uses [1, 6, 12, 18] for 32×32+ features.
    For 16×16, we use smaller rates to avoid gridding artifacts.

    Args:
        in_channels: Input feature channels (D)
        out_channels: Output feature channels (typically D)
        dilations: List of dilation rates (default: [1, 2, 4, 8] for 16×16)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 256,
                 dilations: list = [1, 2, 4, 8]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 conv (captures local features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different dilation rates
        self.aspp_convs = nn.ModuleList([
            ASPPConv(in_channels, out_channels, dilation)
            for dilation in dilations[1:]  # Skip dilation=1 (already covered by conv1)
        ])

        # Global average pooling
        self.global_pool = ASPPPooling(in_channels, out_channels)

        # Fusion: concatenate all branches
        num_branches = 1 + len(dilations[1:]) + 1  # conv1 + aspp_convs + global_pool
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W] - latent features
        Returns:
            out: [B, D, H, W] - context-enhanced features
        """
        # Multiple parallel branches
        res = []

        # 1x1 conv
        res.append(self.conv1(x))

        # Atrous convolutions
        for aspp_conv in self.aspp_convs:
            res.append(aspp_conv(x))

        # Global pooling
        res.append(self.global_pool(x))

        # Concatenate all branches
        res = torch.cat(res, dim=1)  # [B, out_channels * num_branches, H, W]

        # Project back to original channel dimension
        out = self.project(res)  # [B, D, H, W]

        return out


class ASPPAdaptive(nn.Module):
    """
    Adaptive ASPP that automatically adjusts dilation rates based on feature map size.

    Args:
        in_channels: Input feature channels (D)
        out_channels: Output feature channels (typically 256)
        feature_size: Expected feature map spatial size (H=W, e.g., 16, 32, 64)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 256,
                 feature_size: int = 16):
        super().__init__()

        # Auto-select dilation rates based on feature size
        if feature_size <= 16:
            # Conservative for small features
            dilations = [1, 2, 4, 8]
        elif feature_size <= 32:
            # Moderate for medium features
            dilations = [1, 3, 6, 12]
        else:
            # Aggressive for large features (DeepLab original)
            dilations = [1, 6, 12, 18]

        # Use base ASPP with auto-selected dilations
        self.aspp = ASPP(in_channels, out_channels, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aspp(x)


class SimpleSpatialContext(nn.Module):
    """
    Simplified spatial context encoder (lighter than ASPP).

    Uses two dilated convolutions to capture local and broader context.

    ⚠️ NOTE: Dilation rate adjusted for 16×16 feature maps.
    Uses dilation=4 instead of 6 to avoid gridding artifacts.

    Args:
        in_channels: Input channels (D)
        dilation: Dilation rate for broader context (default: 4 for 16×16)
    """
    def __init__(self, in_channels: int, dilation: int = 4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W]
        Returns:
            out: [B, D, H, W]
        """
        local = self.conv1(x)
        context = self.conv2(x)
        fused = torch.cat([local, context], dim=1)
        out = self.fusion(fused)
        return out
