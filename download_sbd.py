"""
Download and test Semantic Boundaries Dataset (SBD).

SBD contains annotations from 11,355 images taken from PASCAL VOC 2011.
"""

import torch
from torchvision.datasets import SBDataset
from torch.utils.data import DataLoader

def download_sbd():
    """Download and test SBD dataset."""

    print("="*60)
    print("Downloading Semantic Boundaries Dataset (SBD)")
    print("="*60)

    # Download train split
    print("\n[1] Downloading train split...")
    train_dataset = SBDataset(
        root='./data',
        image_set='train',
        mode='segmentation',  # Use segmentation mode
        download=True
    )
    print(f"✓ Train split: {len(train_dataset)} images")

    # Download val split
    print("\n[2] Downloading val split...")
    val_dataset = SBDataset(
        root='./data',
        image_set='val',
        mode='segmentation',
        download=True
    )
    print(f"✓ Val split: {len(val_dataset)} images")

    # Download train_noval split (excludes VOC 2012 val images)
    print("\n[3] Downloading train_noval split...")
    train_noval_dataset = SBDataset(
        root='./data',
        image_set='train_noval',
        mode='segmentation',
        download=True
    )
    print(f"✓ Train_noval split: {len(train_noval_dataset)} images")

    # Test loading a sample
    print("\n[4] Testing sample loading...")
    image, target = train_dataset[0]
    print(f"    Image type: {type(image)}")
    print(f"    Target type: {type(target)}")

    print("\n" + "="*60)
    print("SBD Dataset Summary")
    print("="*60)
    print(f"Total images: ~11,355")
    print(f"Splits:")
    print(f"  - train:        {len(train_dataset):,} images")
    print(f"  - val:          {len(val_dataset):,} images")
    print(f"  - train_noval:  {len(train_noval_dataset):,} images")
    print(f"\nNote: train_noval excludes VOC 2012 val images")
    print(f"Dataset stored in: ./data/")
    print("="*60)
    print("\n✓ SBD dataset ready!")


if __name__ == "__main__":
    download_sbd()
