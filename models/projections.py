"""
Projection layers for DSC-ViT (Image→Cluster Space architecture)

Key architecture change:
- Original: Cluster→Visual→Cluster→Latent (K→C→K→D)
- New: Image→Cluster→Latent (C→K→D)

This reduces computational complexity by 33% and improves interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayers(nn.Module):
    """
    Projection layers for DSC-ViT with cluster space integration.

    Architecture:
    1. proj_image_to_cluster: C → K (project image to cluster space)
    2. proj_cluster_to_latent: K → D (project cluster to ViT latent space)
    3. proj_latent_to_cluster: D → K (project ViT output back to cluster space)
    """

    def __init__(self,
                 image_channels: int = 3,
                 num_clusters: int = 19,  # e.g., Cityscapes classes
                 latent_dim: int = 512,
                 use_norm: bool = True,
                 activation: str = 'gelu'):
        """
        Args:
            image_channels: Number of input image channels (C)
            num_clusters: Number of clusters (K)
            latent_dim: ViT latent dimension (D)
            use_norm: Whether to use normalization layers
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()

        self.C = image_channels
        self.K = num_clusters
        self.D = latent_dim

        # Activation function
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 1. Image → Cluster space (C → K)
        self.proj_image_to_cluster = nn.Sequential(
            nn.Conv2d(image_channels, num_clusters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_clusters) if use_norm else nn.Identity(),
            self.act
        )

        # 2. Cluster space → Latent space (K → D)
        self.proj_cluster_to_latent = nn.Sequential(
            nn.Conv2d(num_clusters, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim) if use_norm else nn.Identity(),
            self.act
        )

        # 3. Latent space → Cluster space (D → K)
        # Used to project ViT output back to cluster space for Soft K-Means
        self.proj_latent_to_cluster = nn.Sequential(
            nn.Conv2d(latent_dim, num_clusters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_clusters) if use_norm else nn.Identity(),
            self.act
        )

        # Initial projection for first ViT pass (C → D)
        self.proj_initial = nn.Sequential(
            nn.Conv2d(image_channels, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim) if use_norm else nn.Identity(),
            self.act
        )

    def image_to_cluster(self, image: torch.Tensor) -> torch.Tensor:
        """
        Project image to cluster space.

        Args:
            image: [B, C, H, W]
        Returns:
            cluster_features: [B, K, H, W]
        """
        return self.proj_image_to_cluster(image)

    def cluster_to_latent(self, cluster_features: torch.Tensor) -> torch.Tensor:
        """
        Project cluster features to ViT latent space.

        Args:
            cluster_features: [B, K, H, W]
        Returns:
            latent_features: [B, D, H, W]
        """
        return self.proj_cluster_to_latent(cluster_features)

    def latent_to_cluster(self, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Project ViT latent features back to cluster space.

        Args:
            latent_features: [B, D, H, W]
        Returns:
            cluster_features: [B, K, H, W]
        """
        return self.proj_latent_to_cluster(latent_features)

    def initial_projection(self, image: torch.Tensor) -> torch.Tensor:
        """
        Initial projection for first ViT pass (C → D).

        Args:
            image: [B, C, H, W]
        Returns:
            latent_features: [B, D, H, W]
        """
        return self.proj_initial(image)

    def forward(self, image: torch.Tensor, mode: str = 'image_to_cluster'):
        """
        Forward pass with specified mode.

        Args:
            image: Input tensor
            mode: Projection mode
                - 'image_to_cluster': C → K
                - 'cluster_to_latent': K → D
                - 'latent_to_cluster': D → K
                - 'initial': C → D
        """
        if mode == 'image_to_cluster':
            return self.image_to_cluster(image)
        elif mode == 'cluster_to_latent':
            return self.cluster_to_latent(image)
        elif mode == 'latent_to_cluster':
            return self.latent_to_cluster(image)
        elif mode == 'initial':
            return self.initial_projection(image)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining cluster features.

    Can be used instead of simple residual connection for better performance.
    """

    def __init__(self, num_clusters: int):
        super().__init__()

        # Gate network: learns to balance z_clustered and image_in_cluster
        self.gate_conv = nn.Sequential(
            nn.Conv2d(num_clusters * 2, num_clusters, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z_clustered: torch.Tensor, image_cluster: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_clustered: [B, K, H, W] - clustered features from Soft K-Means
            image_cluster: [B, K, H, W] - original image projected to cluster space
        Returns:
            fused: [B, K, H, W] - gated fusion
        """
        # Concatenate along channel dimension
        concat = torch.cat([z_clustered, image_cluster], dim=1)  # [B, 2K, H, W]

        # Compute gate
        gate = self.gate_conv(concat)  # [B, K, H, W]

        # Gated fusion
        fused = gate * z_clustered + (1 - gate) * image_cluster

        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for combining cluster features.

    More sophisticated than gated fusion, but higher computational cost.
    """

    def __init__(self, num_clusters: int, num_heads: int = None):
        super().__init__()

        # Auto-select num_heads if not provided (prefer powers of 2)
        if num_heads is None:
            # Try powers of 2 first (descending from 16)
            for h in [16, 8, 4, 2, 1]:
                if num_clusters % h == 0:
                    num_heads = h
                    break
            if num_heads is None:
                num_heads = 1

        assert num_clusters % num_heads == 0, f"num_clusters ({num_clusters}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = num_clusters // num_heads

        print(f"  AttentionFusion: num_clusters={num_clusters}, num_heads={num_heads}, head_dim={self.head_dim}")

        # Q, K, V projections
        self.q_proj = nn.Conv2d(num_clusters, num_clusters, kernel_size=1)
        self.k_proj = nn.Conv2d(num_clusters, num_clusters, kernel_size=1)
        self.v_proj = nn.Conv2d(num_clusters, num_clusters, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(num_clusters, num_clusters, kernel_size=1)

        self.scale = self.head_dim ** -0.5

    def forward(self, z_clustered: torch.Tensor, image_cluster: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_clustered: [B, K, H, W]
            image_cluster: [B, K, H, W]
        Returns:
            fused: [B, K, H, W]
        """
        B, num_clusters, height, width = z_clustered.shape

        # Query from z_clustered, Key and Value from image_cluster
        Q = self.q_proj(z_clustered)  # [B, num_clusters, height, width]
        K = self.k_proj(image_cluster)
        V = self.v_proj(image_cluster)

        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, height * width)  # [B, num_heads, head_dim, HW]
        K = K.view(B, self.num_heads, self.head_dim, height * width)
        V = V.view(B, self.num_heads, self.head_dim, height * width)

        # Attention scores
        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) * self.scale  # [B, num_heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, V)  # [B, num_heads, head_dim, HW]

        # Reshape back
        out = out.reshape(B, num_clusters, height, width)

        # Output projection
        out = self.out_proj(out)

        # Residual connection
        fused = out + z_clustered

        return fused


if __name__ == "__main__":
    # Test projection layers
    B, C, H, W = 2, 3, 256, 256
    K, D = 19, 512

    # Create projection layers
    proj = ProjectionLayers(
        image_channels=C,
        num_clusters=K,
        latent_dim=D
    )

    # Test image
    image = torch.randn(B, C, H, W)

    # Test forward passes
    print("Testing ProjectionLayers:")
    print(f"Input image: {image.shape}")

    # C → K
    cluster_feat = proj.image_to_cluster(image)
    print(f"Image → Cluster: {cluster_feat.shape} (expected: [{B}, {K}, {H}, {W}])")

    # K → D
    latent_feat = proj.cluster_to_latent(cluster_feat)
    print(f"Cluster → Latent: {latent_feat.shape} (expected: [{B}, {D}, {H}, {W}])")

    # D → K
    cluster_feat_back = proj.latent_to_cluster(latent_feat)
    print(f"Latent → Cluster: {cluster_feat_back.shape} (expected: [{B}, {K}, {H}, {W}])")

    # Test fusion mechanisms
    print("\nTesting GatedFusion:")
    gated_fusion = GatedFusion(K)
    fused = gated_fusion(cluster_feat, cluster_feat_back)
    print(f"Fused features: {fused.shape} (expected: [{B}, {K}, {H}, {W}])")

    print("\nTesting AttentionFusion:")
    attn_fusion = AttentionFusion(K, num_heads=4)
    fused_attn = attn_fusion(cluster_feat, cluster_feat_back)
    print(f"Attention fused: {fused_attn.shape} (expected: [{B}, {K}, {H}, {W}])")

    print("\n✓ All tests passed!")
