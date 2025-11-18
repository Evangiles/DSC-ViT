"""
Download ADE20K dataset for semantic segmentation.

ADE20K Dataset:
- 150 semantic categories
- 20,210 training images
- 2,000 validation images
- More challenging than PASCAL VOC (21 classes)
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"Downloading from {url}")
    print(f"Destination: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}% [{count * block_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB]", end='')

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print("\n✓ Download complete!")


def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete!")


def download_ade20k(data_root='./data'):
    """
    Download ADE20K dataset.

    Downloads from MIT CSAIL official mirror.
    """

    print("=" * 60)
    print("Downloading ADE20K Dataset")
    print("=" * 60)

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # ADE20K official download URLs
    # Note: These are the official MIT CSAIL links
    base_url = "http://data.csail.mit.edu/places/ADEchallenge"

    # ADEChallengeData2016.zip contains images and annotations
    dataset_url = f"{base_url}/ADEChallengeData2016.zip"
    zip_path = data_root / "ADEChallengeData2016.zip"
    extract_path = data_root

    # Check if already downloaded
    ade20k_path = data_root / "ADEChallengeData2016"
    if ade20k_path.exists():
        print(f"\n✓ ADE20K already exists at {ade20k_path}")
        print("Skipping download.")
    else:
        # Download dataset
        print(f"\n[1] Downloading ADE20K dataset...")
        print(f"    Size: ~923 MB (compressed)")
        download_file(dataset_url, zip_path)

        # Extract dataset
        print(f"\n[2] Extracting dataset...")
        extract_zip(zip_path, extract_path)

        # Clean up zip file
        print(f"\n[3] Cleaning up...")
        zip_path.unlink()
        print("✓ Removed zip file")

    # Verify structure
    print("\n" + "=" * 60)
    print("Verifying Dataset Structure")
    print("=" * 60)

    images_train = ade20k_path / "images" / "training"
    images_val = ade20k_path / "images" / "validation"
    annot_train = ade20k_path / "annotations" / "training"
    annot_val = ade20k_path / "annotations" / "validation"

    if images_train.exists():
        num_train_images = len(list(images_train.glob("*.jpg")))
        print(f"✓ Training images:   {num_train_images:,} (expected: ~20,210)")
    else:
        print("✗ Training images not found")

    if images_val.exists():
        num_val_images = len(list(images_val.glob("*.jpg")))
        print(f"✓ Validation images: {num_val_images:,} (expected: ~2,000)")
    else:
        print("✗ Validation images not found")

    if annot_train.exists():
        num_train_annot = len(list(annot_train.glob("*.png")))
        print(f"✓ Training annotations:   {num_train_annot:,}")
    else:
        print("✗ Training annotations not found")

    if annot_val.exists():
        num_val_annot = len(list(annot_val.glob("*.png")))
        print(f"✓ Validation annotations: {num_val_annot:,}")
    else:
        print("✗ Validation annotations not found")

    # Dataset info
    print("\n" + "=" * 60)
    print("ADE20K Dataset Summary")
    print("=" * 60)
    print(f"Classes:     150 semantic categories")
    print(f"Train:       ~20,210 images")
    print(f"Validation:  ~2,000 images")
    print(f"Location:    {ade20k_path}")
    print(f"\nDataset structure:")
    print(f"  {ade20k_path}/")
    print(f"    ├── images/")
    print(f"    │   ├── training/     (*.jpg)")
    print(f"    │   └── validation/   (*.jpg)")
    print(f"    └── annotations/")
    print(f"        ├── training/     (*.png)")
    print(f"        └── validation/   (*.png)")
    print("=" * 60)
    print("\n✓ ADE20K dataset ready!")

    return ade20k_path


if __name__ == "__main__":
    download_ade20k()
