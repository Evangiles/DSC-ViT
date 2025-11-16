"""
Analyze PASCAL VOC dataset to see how many classes per image.
"""
import torch
from data.voc_kaggle import VOCSegmentationKaggle
from data.transforms import SegmentationTransform, get_val_transforms
from collections import Counter
import numpy as np


def analyze_dataset(split='train'):
    """Analyze how many classes are in each image."""
    print(f"\n{'='*60}")
    print(f"Analyzing {split.upper()} dataset")
    print(f"{'='*60}")

    # Load dataset
    transform = SegmentationTransform(get_val_transforms(256))
    dataset = VOCSegmentationKaggle(split=split, transform=transform)

    # Analyze each image
    num_classes_per_image = []
    images_with_multiple_classes = 0

    for idx in range(len(dataset)):
        image, mask = dataset[idx]

        # Get unique classes (excluding -100 ignore index)
        unique_classes = mask[mask != -100].unique()
        num_classes = len(unique_classes)
        num_classes_per_image.append(num_classes)

        if num_classes > 2:  # More than background + 1 object
            images_with_multiple_classes += 1

        # Print first 5 examples
        if idx < 5:
            print(f"\nImage {idx}:")
            print(f"  Unique classes: {unique_classes.tolist()}")
            print(f"  Num classes: {num_classes}")
            # Map to class names
            class_names = [VOCSegmentationKaggle.VOC_CLASSES[c] for c in unique_classes.tolist()]
            print(f"  Classes: {class_names}")

    # Statistics
    num_classes_per_image = np.array(num_classes_per_image)

    print(f"\n{'-'*60}")
    print(f"Statistics:")
    print(f"{'-'*60}")
    print(f"Total images: {len(dataset)}")
    print(f"Images with 1 class (background only): {(num_classes_per_image == 1).sum()}")
    print(f"Images with 2 classes (background + 1 object): {(num_classes_per_image == 2).sum()}")
    print(f"Images with 3+ classes (multiple objects): {(num_classes_per_image >= 3).sum()}")
    print(f"")
    print(f"Images with multiple object classes (>2 total): {images_with_multiple_classes}")
    print(f"Percentage: {100 * images_with_multiple_classes / len(dataset):.1f}%")
    print(f"")
    print(f"Min classes per image: {num_classes_per_image.min()}")
    print(f"Max classes per image: {num_classes_per_image.max()}")
    print(f"Mean classes per image: {num_classes_per_image.mean():.2f}")
    print(f"Median classes per image: {np.median(num_classes_per_image):.0f}")

    # Distribution
    print(f"\nDistribution of number of classes:")
    counter = Counter(num_classes_per_image)
    for num in sorted(counter.keys()):
        print(f"  {num} classes: {counter[num]:4d} images ({100*counter[num]/len(dataset):5.1f}%)")

    return num_classes_per_image


if __name__ == "__main__":
    # Analyze train split
    train_stats = analyze_dataset('train')

    # Analyze valid split
    val_stats = analyze_dataset('valid')

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_stats)} images, mean {train_stats.mean():.2f} classes/image")
    print(f"Valid: {len(val_stats)} images, mean {val_stats.mean():.2f} classes/image")
    print(f"{'='*60}\n")
