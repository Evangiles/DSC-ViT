"""
Data augmentation transforms for segmentation.

Uses albumentations for efficient augmentation.
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Get training augmentation pipeline.

    Includes TRM paper's dihedral transformations and additional augmentations.

    Args:
        img_size: Target image size
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),

        # Dihedral transformations (8 possible: 4 rotations × 2 flips)
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Transpose(p=1.0),
            A.Rotate(limit=[90, 90], p=1.0),
            A.Rotate(limit=[180, 180], p=1.0),
            A.Rotate(limit=[270, 270], p=1.0),
            A.NoOp(),  # No transformation
        ], p=0.8),

        # Random crop (if enabled)
        A.RandomCrop(img_size, img_size, p=0.5) if img_size < 512 else A.NoOp(),

        # Color augmentations (TRM paper mentions color jitter)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),

        # Additional augmentations
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.NoOp(),
        ], p=0.3),

        # Normalize (ImageNet stats)
        A.Normalize(mean=mean, std=std),

        # To tensor
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Get validation transforms (no augmentation).

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class SegmentationTransform:
    """
    Wrapper for albumentations transforms for segmentation.

    Handles image and mask together.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        """
        Apply transform to image and mask.

        Args:
            image: PIL Image or numpy array [H, W, C]
            mask: PIL Image or numpy array [H, W]
        Returns:
            image_tensor: [C, H, W]
            mask_tensor: [H, W]
        """
        # Convert PIL to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)

        image_tensor = transformed['image']  # Already tensor from ToTensorV2

        # Check if mask is already a tensor (from ToTensorV2)
        mask_data = transformed['mask']
        if isinstance(mask_data, torch.Tensor):
            mask_tensor = mask_data.long()
        else:
            mask_tensor = torch.from_numpy(mask_data).long()

        return image_tensor, mask_tensor


if __name__ == "__main__":
    print("Testing transforms:")

    # Create dummy image and mask
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 21, (512, 512), dtype=np.uint8)

    # Test training transforms
    print("\n1. Training transforms:")
    train_transform = SegmentationTransform(get_train_transforms(256))
    img_t, mask_t = train_transform(img, mask)
    print(f"   Image: {img_t.shape}, dtype={img_t.dtype}, range=[{img_t.min():.2f}, {img_t.max():.2f}]")
    print(f"   Mask: {mask_t.shape}, dtype={mask_t.dtype}, unique={mask_t.unique().tolist()[:5]}...")

    # Test validation transforms
    print("\n2. Validation transforms:")
    val_transform = SegmentationTransform(get_val_transforms(256))
    img_v, mask_v = val_transform(img, mask)
    print(f"   Image: {img_v.shape}, dtype={img_v.dtype}")
    print(f"   Mask: {mask_v.shape}, dtype={mask_v.dtype}")

    print("\n✓ Transforms test passed!")
