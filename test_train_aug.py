"""
Test if torchvision VOCSegmentation supports 'train_aug' option.

train_aug = VOC 2012 train + SBD = 10,582 images (SOTA standard)
"""

from torchvision.datasets import VOCSegmentation

print("="*60)
print("Testing VOC 'train_aug' (SOTA augmented training set)")
print("="*60)

try:
    # SOTA 표준 "증강 세트" (10,582장)
    dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train_aug',  # 'train'이 아닌 'train_aug'
        download=True
    )

    print(f"\n✓ train_aug dataset loaded successfully!")
    print(f"  Total images: {len(dataset):,}")
    print(f"  Expected: 10,582 images")

    # Test loading a sample
    img, mask = dataset[0]
    print(f"\n  Sample test:")
    print(f"    Image type: {type(img)}")
    print(f"    Mask type: {type(mask)}")

    print("\n" + "="*60)
    if len(dataset) == 10582:
        print("✓ SUCCESS: train_aug works perfectly!")
    else:
        print(f"⚠ WARNING: Expected 10,582 but got {len(dataset)}")
    print("="*60)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\ntrain_aug may not be supported in this torchvision version")
    print("Alternative: Manually combine VOC train + SBD")
