"""
Dataset loaders for semantic segmentation.

PASCAL VOC 2012 using torchvision's built-in dataset.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation as TorchVisionVOC

from .transforms import SegmentationTransform


class VOCSegmentation(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset.

    Uses torchvision's built-in VOCSegmentation for automatic download.

    Splits:
    - train: 1,464 images (official VOC 2012 train)
    - val: 1,449 images (official VOC 2012 val)

    Reference:
    - VOC: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
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
                 root='./data',
                 split='train',  # 'train' or 'val'
                 transform=None,
                 download=True):
        """
        Args:
            root: Root directory for dataset
            split: Dataset split ('train' or 'val')
            transform: SegmentationTransform object
            download: Whether to download if not present
        """
        self.root = root
        self.split = split
        self.transform = transform

        # Use torchvision's VOCSegmentation (handles download automatically)
        self.voc_dataset = TorchVisionVOC(
            root=root,
            year='2012',
            image_set=split,
            download=download
        )

        print(f"VOC {split}: {len(self.voc_dataset)} images loaded")

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            image: [C, H, W] tensor
            mask: [H, W] tensor with class indices (0-20)
        """
        # Load image and mask from torchvision
        image, mask = self.voc_dataset[idx]

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).long()

        # VOC uses 255 as ignore index (boundary pixels)
        mask[mask == 255] = -100  # PyTorch ignore index

        return image, mask


def get_dataset(dataset_name='voc', root='./data', split='train', transform=None):
    """
    Factory function to get dataset.

    Args:
        dataset_name: 'voc'
        root: Data root directory
        split: Dataset split
        transform: Transform to apply
    Returns:
        Dataset object
    """
    if dataset_name.lower() in ['voc', 'pascal', 'pascalvoc']:
        return VOCSegmentation(root=root, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    from .transforms import get_train_transforms, get_val_transforms

    print("="*60)
    print("Testing PASCAL VOC 2012 Dataset (torchvision)")
    print("="*60)

    # Test train split
    print("\n[1] Testing train split...")
    train_transform = SegmentationTransform(get_train_transforms(256))
    train_dataset = VOCSegmentation(
        root='./data',
        split='train',
        transform=train_transform,
        download=True
    )
    print(f"    Total train images: {len(train_dataset)}")

    # Test loading
    img, mask = train_dataset[0]
    print(f"    Sample - Image: {img.shape}, Mask: {mask.shape}")
    print(f"    Mask unique values: {mask.unique().tolist()[:10]}...")

    # Test val split
    print("\n[2] Testing val split...")
    val_transform = SegmentationTransform(get_val_transforms(256))
    val_dataset = VOCSegmentation(
        root='./data',
        split='val',
        transform=val_transform,
        download=True
    )
    print(f"    Total val images: {len(val_dataset)}")

    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Classes: {VOCSegmentation.NUM_CLASSES} (20 objects + 1 background)")
    print(f"Class names: {', '.join(VOCSegmentation.VOC_CLASSES[:5])}...")
    print(f"\nDataset splits:")
    print(f"  - Train: {len(train_dataset)} images")
    print(f"  - Val:   {len(val_dataset)} images")
    print("="*60)
