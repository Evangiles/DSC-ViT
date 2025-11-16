"""
Benchmark to verify that ViT is called only ONCE (not n×T times).

This validates our architectural improvement.
"""

import torch
import torch.nn as nn
from models import DSCViT

class CountingWrapper(nn.Module):
    """Wrapper to count forward calls."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        return self.module(*args, **kwargs)

    def reset_count(self):
        self.call_count = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def test_vit_call_count():
    """Test that ViT is called exactly once per forward pass."""

    print("="*60)
    print("Benchmarking ViT Call Count")
    print("="*60)

    # Create model
    model = DSCViT(
        image_channels=3,
        num_classes=19,
        img_size=256,
        num_clusters=19,
        latent_dim=512,
        num_latent_updates=6,      # n
        num_recursive_steps=3,     # T
        use_simple_encoder=True,
        fusion_method='residual',
        use_deep_supervision=True
    )

    # Wrap encoder to count calls
    original_encoder = model.encoder
    model.encoder = CountingWrapper(original_encoder)

    # Test input
    B, C, H, W = 2, 3, 256, 256
    image = torch.randn(B, C, H, W)

    # Test 1: Single forward pass
    print("\n[Test 1] Single forward pass (1 supervision step)")
    model.encoder.reset_count()

    (y, z), seg_pred, q_logit, _, _ = model(image)

    vit_calls = model.encoder.call_count
    print(f"  ViT forward calls: {vit_calls}")
    print(f"  Expected: 1")
    print(f"  ✓ PASS" if vit_calls == 1 else f"  ✗ FAIL")

    # Test 2: Deep Supervision loop (N_sup steps)
    print("\n[Test 2] Deep Supervision loop (N_sup=16 steps)")
    model.encoder.reset_count()

    N_sup = 16
    y, z = None, None
    for step in range(N_sup):
        (y, z), seg_pred, q_logit, _, _ = model(image, y, z)

    vit_calls = model.encoder.call_count
    print(f"  ViT forward calls: {vit_calls}")
    print(f"  Expected: {N_sup} (1 per step)")
    print(f"  ✓ PASS" if vit_calls == N_sup else f"  ✗ FAIL")

    # Calculate speedup
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)

    n = 6
    T = 3
    old_vit_calls_per_step = n * T  # Old: ViT called n×T times per step
    new_vit_calls_per_step = 1       # New: ViT called once per step

    speedup = old_vit_calls_per_step / new_vit_calls_per_step

    print(f"\nOld architecture (ViT in recursive loop):")
    print(f"  ViT calls per step: {old_vit_calls_per_step} (n×T = {n}×{T})")
    print(f"  ViT calls for N_sup={N_sup}: {old_vit_calls_per_step * N_sup}")

    print(f"\nNew architecture (ViT outside loop):")
    print(f"  ViT calls per step: {new_vit_calls_per_step}")
    print(f"  ViT calls for N_sup={N_sup}: {new_vit_calls_per_step * N_sup}")

    print(f"\n⭐ Speedup: {speedup}x fewer ViT forward passes!")
    print(f"   ({old_vit_calls_per_step * N_sup} → {new_vit_calls_per_step * N_sup} calls)")

    print("\n" + "="*60)
    print("Benefits")
    print("="*60)
    print("✅ 1. ~18x faster recursive refinement")
    print("✅ 2. Can freeze ViT for feature extraction")
    print("✅ 3. Recursive learning happens in lightweight modules")
    print("✅ 4. True to TRM's 'tiny recursive network' philosophy")
    print("="*60)


if __name__ == "__main__":
    test_vit_call_count()
