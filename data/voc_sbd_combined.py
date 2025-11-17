"""
Combined VOC + SBD dataset for PASCAL VOC segmentation.

Combines:
- PASCAL VOC 2012 (Kaggle): 1,464 train images
- SBD train_noval: 9,118 images (excludes VOC 2012 val)
Total: 10,582 training images (SOTA standard)

Usage:
    from data.voc_sbd_combined import get_combined_dataset

    train_dataset, val_dataset = get_combined_dataset(
        transform_train=train_transform,
        transform_val=val_transform
    )
"""

import os
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets import SBDataset
from .voc_kaggle import VOCSegmentationKaggle


def get_combined_dataset(transform_train=None, transform_val=None, voc_root=None):
    """
    Create combined VOC + SBD training dataset.

    Args:
        transform_train: Training transforms (Albumentations or torchvision)
        transform_val: Validation transforms
        voc_root: Root directory for VOC dataset (auto-detected if None)

    Returns:
        train_dataset: Combined VOC train (1,464) + SBD train_noval (9,118) = 10,582 images
        val_dataset: VOC validation (1,449 images)
    """

    # 1. VOC Kaggle dataset
    if voc_root is None:
        # Auto-detect Kaggle cache location
        kaggle_cache = Path.home() / '.cache' / 'kagglehub' / 'datasets'
        voc_root = kaggle_cache / 'sovitrath' / 'voc-2012-segmentation-data' / 'versions' / '1' / 'voc_2012_segmentation_data'

        if not voc_root.exists():
            raise FileNotFoundError(
                f"VOC dataset not found at {voc_root}\n"
                "Please download first:\n"
                "  python -c \"import kagglehub; kagglehub.dataset_download('sovitrath/voc-2012-segmentation-data')\""
            )

    print(f"Loading VOC from: {voc_root}")

    voc_train = VOCSegmentationKaggle(
        root=str(voc_root),
        split='train',
        transform=transform_train
    )

    voc_val = VOCSegmentationKaggle(
        root=str(voc_root),
        split='valid',
        transform=transform_val
    )

    # 2. SBD dataset (train_noval split - excludes VOC 2012 val)
    print(f"Loading SBD from: ./data")

    # Check if SBD is downloaded
    sbd_path = Path('./data/benchmark_RELEASE/dataset')
    if not sbd_path.exists():
        raise FileNotFoundError(
            f"SBD dataset not found at {sbd_path}\n"
            "Please download first:\n"
            "  python download_sbd.py"
        )

    # Create wrapper for SBD to match VOCSegmentationKaggle interface
    class SBDWrapper(torch.utils.data.Dataset):
        """Wrapper for SBD to handle transform compatibility."""

        def __init__(self, sbd_dataset, transform):
            self.sbd_dataset = sbd_dataset
            self.transform = transform

        def __len__(self):
            return len(self.sbd_dataset)

        def __getitem__(self, idx):
            image, target = self.sbd_dataset[idx]

            # SBD returns PIL images, convert to numpy for Albumentations
            if self.transform is not None:
                import numpy as np
                image_np = np.array(image)
                target_np = np.array(target)

                # Apply transform
                transformed = self.transform(image=image_np, mask=target_np)
                image_t = transformed['image']
                target_t = transformed['mask']

                # Convert to tensors if not already
                if isinstance(image_t, np.ndarray):
                    image_t = torch.from_numpy(image_t).permute(2, 0, 1).float() / 255.0
                    # ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_t = (image_t - mean) / std

                if isinstance(target_t, np.ndarray):
                    target_t = torch.from_numpy(target_t).long()

                return image_t, target_t
            else:
                # No transform, return as-is
                return image, target

    sbd_dataset_raw = SBDataset(
        root='./data',
        image_set='train_noval',  # IMPORTANT: excludes VOC 2012 val
        mode='segmentation',
        download=False  # Should be already downloaded
    )

    # Wrap SBD with transform compatibility
    sbd_train = SBDWrapper(sbd_dataset_raw, transform_train)

    # 3. Combine VOC train + SBD train_noval
    train_combined = ConcatDataset([voc_train, sbd_train])

    # Print statistics
    print(f"\n{'='*60}")
    print("VOC + SBD Dataset Statistics")
    print(f"{'='*60}")
    print(f"VOC train:       {len(voc_train):,} images")
    print(f"SBD train_noval: {len(sbd_train):,} images (excludes VOC val)")
    print(f"{'─'*60}")
    print(f"Combined train:  {len(train_combined):,} images ⭐")
    print(f"VOC validation:  {len(voc_val):,} images")
    print(f"{'='*60}")
    print(f"Expected: 10,582 train images (SOTA standard)")
    print(f"Actual:   {len(train_combined):,} train images")
    if len(train_combined) == 10582:
        print(f"✓ Dataset size matches SOTA standard!")
    else:
        print(f"⚠ Dataset size differs from expected")
    print(f"{'='*60}\n")

    return train_combined, voc_val


if __name__ == "__main__":
    # Test combined dataset
    print("="*60)
    print("Testing VOC + SBD Combined Dataset")
    print("="*60)

    try:
        train, val = get_combined_dataset()

        print(f"\n✓ Train dataset: {len(train):,} images")
        print(f"✓ Val dataset:   {len(val):,} images")

        # Test loading samples from each source
        print("\n[1] Testing VOC sample (index 0, from VOC)...")
        img, mask = train[0]
        print(f"  Image: {type(img)}")
        print(f"  Mask:  {type(mask)}")
        if hasattr(img, 'shape'):
            print(f"  Image shape: {img.shape}")
        if hasattr(mask, 'shape'):
            print(f"  Mask shape:  {mask.shape}")

        print("\n[2] Testing SBD sample (index 2000, from SBD)...")
        img, mask = train[2000]
        print(f"  Image: {type(img)}")
        print(f"  Mask:  {type(mask)}")
        if hasattr(img, 'shape'):
            print(f"  Image shape: {img.shape}")
        if hasattr(mask, 'shape'):
            print(f"  Mask shape:  {mask.shape}")

        print("\n" + "="*60)
        print("✓ Combined dataset ready for training!")
        print("="*60)
        print(f"\nTo use in training:")
        print(f"  from data.voc_sbd_combined import get_combined_dataset")
        print(f"  train_dataset, val_dataset = get_combined_dataset()")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease download datasets first:")
        print(f"  1. VOC: python -c \"import kagglehub; kagglehub.dataset_download('sovitrath/voc-2012-segmentation-data')\"")
        print(f"  2. SBD: python download_sbd.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
