"""
ADE20K dataset loader for semantic segmentation.

ADE20K is a large-scale scene parsing benchmark with:
- 150 semantic categories
- 20,210 training images
- 2,000 validation images
- Diverse indoor/outdoor scenes
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class ADE20KSegmentation(Dataset):
    """
    ADE20K Semantic Segmentation Dataset.

    Dataset structure:
        ADEChallengeData2016/
        ├── images/
        │   ├── training/     (20,210 .jpg files)
        │   └── validation/   (2,000 .jpg files)
        └── annotations/
            ├── training/     (20,210 .png files)
            └── validation/   (2,000 .png files)
    """

    # ADE20K has 150 classes (+ 1 for background/unlabeled)
    NUM_CLASSES = 150

    # Class names (first 20 for reference)
    CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
        # ... (130 more classes)
    ]

    def __init__(self,
                 root: str = './data/ADEChallengeData2016',
                 split: str = 'training',
                 transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of ADE20K dataset
            split: 'training' or 'validation'
            transform: Albumentations transform (applied to both image and mask)
        """
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Paths
        self.images_dir = self.root / 'images' / split
        self.annotations_dir = self.root / 'annotations' / split

        # Verify paths exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")

        # Get image list
        self.images = sorted(list(self.images_dir.glob('*.jpg')))
        self.masks = sorted(list(self.annotations_dir.glob('*.png')))

        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})"

        print(f"ADE20K {split}: {len(self.images):,} images loaded")
        print(f"  Images: {self.images_dir}")
        print(f"  Masks:  {self.annotations_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [C, H, W] tensor (float32, normalized)
            mask: [H, W] tensor (int64, class indices 0-149)
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path)

        # Convert to numpy for albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # ADE20K masks are stored as RGB images where R channel contains class index
        # Extract R channel (class indices are 0-150, where 0 is background)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]  # R channel

        # Apply transforms
        if self.transform is not None:
            # SegmentationTransform returns (image, mask) tuple, not dict
            transformed = self.transform(image=image_np, mask=mask_np)

            # Check if it's a tuple or dict
            if isinstance(transformed, tuple):
                image_tensor, mask_tensor = transformed
            else:
                image_np = transformed['image']
                mask_np = transformed['mask']

                # Convert to tensors
                if isinstance(image_np, np.ndarray):
                    # Normalize and convert to tensor
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    # Apply ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_tensor = (image_tensor - mean) / std
                else:
                    image_tensor = image_np

                if isinstance(mask_np, np.ndarray):
                    mask_tensor = torch.from_numpy(mask_np).long()
                else:
                    mask_tensor = mask_np.long()
        else:
            # No transform
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            mask_tensor = torch.from_numpy(mask_np).long()

        # Convert class indices: ADE20K uses 0=unlabeled, 1-150=classes
        # We'll use 0-149 as classes and -100 as ignore index (PyTorch default)
        # Clip values first to prevent out-of-range
        mask_tensor = torch.clamp(mask_tensor, 0, 150)

        # Shift: 0 (unlabeled) → -100 (ignore), 1-150 → 0-149 (classes)
        mask_tensor = mask_tensor - 1  # Now: -1 (was 0), 0-149 (was 1-150)
        mask_tensor[mask_tensor == -1] = -100  # -1 (unlabeled) → -100 (ignore)

        return image_tensor, mask_tensor

    @staticmethod
    def get_num_classes():
        """Return number of classes (excluding background/ignore)."""
        return ADE20KSegmentation.NUM_CLASSES


def get_ade20k_dataset(root='./data/ADEChallengeData2016',
                       transform_train=None,
                       transform_val=None):
    """
    Create ADE20K train and validation datasets.

    Args:
        root: Root directory of ADE20K dataset
        transform_train: Training transforms
        transform_val: Validation transforms

    Returns:
        train_dataset: Training dataset (~20,210 images)
        val_dataset: Validation dataset (~2,000 images)
    """
    train_dataset = ADE20KSegmentation(
        root=root,
        split='training',
        transform=transform_train
    )

    val_dataset = ADE20KSegmentation(
        root=root,
        split='validation',
        transform=transform_val
    )

    print(f"\n{'='*60}")
    print("ADE20K Dataset Statistics")
    print(f"{'='*60}")
    print(f"Train:      {len(train_dataset):,} images")
    print(f"Validation: {len(val_dataset):,} images")
    print(f"Classes:    {ADE20KSegmentation.NUM_CLASSES}")
    print(f"{'='*60}\n")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test ADE20K dataset loader
    print("=" * 60)
    print("Testing ADE20K Dataset Loader")
    print("=" * 60)

    # Test without transforms
    print("\n[1] Testing training split...")
    try:
        train_dataset = ADE20KSegmentation(
            root='./data/ADEChallengeData2016',
            split='training'
        )

        # Load first sample
        image, mask = train_dataset[0]
        print(f"  Image: {image.shape}, dtype={image.dtype}")
        print(f"  Mask: {mask.shape}, dtype={mask.dtype}")
        print(f"  Mask unique values: {torch.unique(mask).tolist()[:10]}...")  # First 10
        print(f"  Mask min/max: [{mask.min()}, {mask.max()}]")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n[2] Testing validation split...")
    try:
        val_dataset = ADE20KSegmentation(
            root='./data/ADEChallengeData2016',
            split='validation'
        )

        # Load first sample
        image, mask = val_dataset[0]
        print(f"  Image: {image.shape}, dtype={image.dtype}")
        print(f"  Mask: {mask.shape}, dtype={mask.dtype}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Classes: {ADE20KSegmentation.NUM_CLASSES}")
    print(f"Train: {len(train_dataset):,} images")
    print(f"Valid: {len(val_dataset):,} images")
    print("=" * 60)
    print("\n✓ ADE20K dataset loader ready!")
