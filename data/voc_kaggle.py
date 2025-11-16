"""
PASCAL VOC 2012 Segmentation from Kaggle.

Downloaded via kagglehub from: sovitrath/voc-2012-segmentation-data
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class VOCSegmentationKaggle(Dataset):
    """
    PASCAL VOC 2012 Segmentation from Kaggle.

    Structure:
    - train_images/ : 1,464 .jpg files
    - train_labels/ : 1,464 .png files (segmentation masks)
    - valid_images/ : 1,449 .jpg files
    - valid_labels/ : 1,449 .png files
    """

    # 20 object classes + 1 background = 21 classes
    VOC_CLASSES = [
        'background',
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    NUM_CLASSES = 21

    def __init__(self,
                 root='/root/.cache/kagglehub/datasets/sovitrath/voc-2012-segmentation-data/versions/1/voc_2012_segmentation_data',
                 split='train',  # 'train' or 'valid'
                 transform=None):
        """
        Args:
            root: Root directory of kaggle dataset
            split: 'train' or 'valid'
            transform: SegmentationTransform object
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Set paths
        self.image_dir = self.root / f'{split}_images'
        self.label_dir = self.root / f'{split}_labels'

        # Get file list
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

        print(f"VOC Kaggle {split}: {len(self.images)} images loaded")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Label dir: {self.label_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: [C, H, W] tensor
            mask: [H, W] tensor with class indices (0-20, -100 for ignore)
        """
        # Load image
        img_name = self.images[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = self.label_dir / mask_name
        mask = Image.open(mask_path)

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).long()

        # VOC uses 255 as ignore index (boundary pixels)
        mask[mask == 255] = -100  # PyTorch ignore index

        return image, mask


if __name__ == "__main__":
    from transforms import SegmentationTransform, get_train_transforms, get_val_transforms

    print("="*60)
    print("Testing VOC Kaggle Dataset")
    print("="*60)

    # Test train split
    print("\n[1] Testing train split...")
    train_transform = SegmentationTransform(get_train_transforms(256))
    train_dataset = VOCSegmentationKaggle(
        split='train',
        transform=train_transform
    )

    # Load sample
    img, mask = train_dataset[0]
    print(f"    Image: {img.shape}, dtype={img.dtype}")
    print(f"    Mask: {mask.shape}, dtype={mask.dtype}")
    print(f"    Mask unique: {mask.unique().tolist()}")

    # Test valid split
    print("\n[2] Testing valid split...")
    val_transform = SegmentationTransform(get_val_transforms(256))
    val_dataset = VOCSegmentationKaggle(
        split='valid',
        transform=val_transform
    )

    img, mask = val_dataset[0]
    print(f"    Image: {img.shape}, dtype={img.dtype}")
    print(f"    Mask: {mask.shape}, dtype={mask.dtype}")

    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Classes: {VOCSegmentationKaggle.NUM_CLASSES}")
    print(f"Train: {len(train_dataset)} images")
    print(f"Valid: {len(val_dataset)} images")
    print("="*60)
    print("\n✓ VOC Kaggle dataset ready!")
