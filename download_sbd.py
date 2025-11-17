"""
Download and test Semantic Boundaries Dataset (SBD).

SBD contains annotations from 11,355 images taken from PASCAL VOC 2011.

IMPORTANT: For VOC+SBD training, use train_noval split to avoid overlap with VOC 2012 val!
Expected: train_noval = 9,118 images (for combining with VOC train 1,464 = 10,582 total)
"""

import os
import torch
from pathlib import Path
from torchvision.datasets import SBDataset
from torch.utils.data import DataLoader


def download_sbd():
    """Download and test SBD dataset."""

    print("="*60)
    print("Downloading Semantic Boundaries Dataset (SBD)")
    print("="*60)
    print("IMPORTANT: We will download train_noval for VOC+SBD training")
    print("="*60)

    # Download train_noval split FIRST (most important for training)
    print("\n[1] Downloading train_noval split (recommended for VOC+SBD)...")
    train_noval_dataset = SBDataset(
        root='./data',
        image_set='train_noval',  # Excludes VOC 2012 val images
        mode='segmentation',
        download=True
    )
    print(f"✓ Train_noval split: {len(train_noval_dataset):,} images (expected: 9,118)")

    # Download train split (for reference)
    print("\n[2] Downloading train split (includes VOC val, not recommended)...")
    train_dataset = SBDataset(
        root='./data',
        image_set='train',
        mode='segmentation',
        download=True  # Should reuse downloaded files
    )
    print(f"✓ Train split: {len(train_dataset):,} images (expected: ~8,498)")

    # Download val split (for reference)
    print("\n[3] Downloading val split...")
    val_dataset = SBDataset(
        root='./data',
        image_set='val',
        mode='segmentation',
        download=True  # Should reuse downloaded files
    )
    print(f"✓ Val split: {len(val_dataset):,} images (expected: ~2,857)")

    # Verify dataset structure
    print("\n[4] Verifying dataset structure...")
    sbd_root = Path('./data/benchmark_RELEASE/dataset')
    if sbd_root.exists():
        print(f"✓ SBD root found: {sbd_root}")

        # Check subdirectories
        img_dir = sbd_root / 'img'
        cls_dir = sbd_root / 'cls'

        if img_dir.exists():
            num_images = len(list(img_dir.glob('*.jpg')))
            print(f"  Images: {num_images:,} (.jpg files in img/)")

        if cls_dir.exists():
            num_labels = len(list(cls_dir.glob('*.mat')))
            print(f"  Labels: {num_labels:,} (.mat files in cls/)")

        # Check split files
        train_txt = sbd_root / 'train.txt'
        val_txt = sbd_root / 'val.txt'
        train_noval_txt = sbd_root / 'train_noval.txt'

        if train_noval_txt.exists():
            with open(train_noval_txt) as f:
                train_noval_ids = [line.strip() for line in f]
            print(f"  train_noval.txt: {len(train_noval_ids):,} IDs")

        if train_txt.exists():
            with open(train_txt) as f:
                train_ids = [line.strip() for line in f]
            print(f"  train.txt: {len(train_ids):,} IDs")
    else:
        print(f"✗ SBD root not found: {sbd_root}")

    # Test loading a sample from train_noval
    print("\n[5] Testing sample loading from train_noval...")
    try:
        image, target = train_noval_dataset[0]
        print(f"  Image type: {type(image)}")
        print(f"  Target type: {type(target)}")
        if hasattr(image, 'shape'):
            print(f"  Image shape: {image.shape}")
        if hasattr(target, 'shape'):
            print(f"  Target shape: {target.shape}")
            if hasattr(target, 'unique'):
                unique_classes = target.unique()
                print(f"  Unique classes in mask: {unique_classes.tolist()[:10]}...")  # First 10
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")

    # Summary
    print("\n" + "="*60)
    print("SBD Dataset Summary")
    print("="*60)
    print(f"Total images: ~11,355")
    print(f"\nSplits:")
    print(f"  - train_noval:  {len(train_noval_dataset):,} images ⭐ USE THIS for VOC+SBD")
    print(f"  - train:        {len(train_dataset):,} images")
    print(f"  - val:          {len(val_dataset):,} images")
    print(f"\n⭐ IMPORTANT:")
    print(f"  For VOC+SBD training, use train_noval (excludes VOC 2012 val)")
    print(f"  Expected total: VOC train (1,464) + SBD train_noval (9,118) = 10,582")
    print(f"\nDataset stored in: ./data/benchmark_RELEASE/dataset/")
    print("="*60)
    print("\n✓ SBD dataset ready!")

    return train_noval_dataset


if __name__ == "__main__":
    download_sbd()
