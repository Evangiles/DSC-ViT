"""
Test that num_clusters can be different from num_classes.

Tests:
1. num_clusters = num_classes (21 = 21) - should work
2. num_clusters > num_classes (32 > 21) - should work
3. num_clusters < num_classes (16 < 21) - should work
"""

import torch
import sys

print("="*60)
print("Testing Cluster-Class Mismatch Compatibility")
print("="*60)

# Add project root to path
sys.path.insert(0, '/workspace/DSC-ViT')

from models import DSCViT
from utils import DeepSupervisionLoss

def test_config(num_classes, num_clusters, test_name):
    """Test a specific configuration."""
    print(f"\n[{test_name}]")
    print(f"  num_classes:  {num_classes}")
    print(f"  num_clusters: {num_clusters}")

    try:
        # Create model
        model = DSCViT(
            num_classes=num_classes,
            num_clusters=num_clusters,
            latent_dim=512,
            use_simple_encoder=True,  # Fast test
            num_latent_updates=2,
            num_recursive_steps=2
        )

        # Create loss
        criterion = DeepSupervisionLoss(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_assignment_loss=False,  # ⭐ Unsupervised
            use_cluster_loss=True
        )

        # Test forward pass
        B, C, H, W = 2, 3, 256, 256
        image = torch.randn(B, C, H, W)
        target = torch.randint(0, num_classes, (B, H, W))

        (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(image)

        print(f"  ✓ Model forward: seg_pred={seg_pred.shape}")
        print(f"  ✓ Cluster output: combined_cluster={combined_cluster.shape}")

        # Test loss
        cluster_centers = model.get_cluster_centers()
        loss, loss_dict = criterion.cluster_loss(
            combined_cluster,
            torch.nn.functional.interpolate(
                target.float().unsqueeze(1),
                size=(model.encoder_output_size, model.encoder_output_size),
                mode='nearest'
            ).squeeze(1).long(),
            cluster_centers
        )

        print(f"  ✓ Loss computed: {loss.item():.4f}")
        print(f"  ✓ Loss components: {loss_dict}")

        # Check assignment_loss
        if 'assignment' in loss_dict:
            print(f"  ⚠ Assignment loss present (should be 0 if use_assignment_loss=False)")
        else:
            print(f"  ✓ Assignment loss skipped (correct for unsupervised)")

        print(f"  ✅ Test PASSED")
        return True

    except Exception as e:
        print(f"  ❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run tests
print("\n" + "="*60)
print("Test Cases")
print("="*60)

results = []

# Test 1: Same (21 = 21)
results.append(test_config(21, 21, "Test 1: num_clusters = num_classes"))

# Test 2: More clusters (32 > 21)
results.append(test_config(21, 32, "Test 2: num_clusters > num_classes"))

# Test 3: Fewer clusters (16 < 21)
results.append(test_config(21, 16, "Test 3: num_clusters < num_classes"))

# Test 4: Many more clusters (64 > 21)
results.append(test_config(21, 64, "Test 4: num_clusters >> num_classes"))

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
passed = sum(results)
total = len(results)
print(f"Passed: {passed}/{total}")

if passed == total:
    print("\n✅ All tests passed!")
    print("\n💡 Key insight:")
    print("  - num_clusters can be DIFFERENT from num_classes")
    print("  - With use_assignment_loss=False (unsupervised):")
    print("    • Clusters discover patterns freely")
    print("    • Seg_head learns to map patterns → classes")
    print("  - More clusters (32, 64) = more fine-grained patterns")
    print("  - Fewer clusters (16) = coarser patterns")
else:
    print(f"\n❌ {total - passed} test(s) failed")
    sys.exit(1)
