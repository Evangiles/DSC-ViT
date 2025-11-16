"""
Evaluation metrics for segmentation.

Includes:
- IoU (Intersection over Union)
- mIoU (mean IoU)
- Pixel Accuracy
- Class-wise metrics
"""

import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """
    Compute segmentation metrics.
    """

    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions.

        Args:
            pred: [B, num_classes, H, W] or [B, H, W]
            target: [B, H, W]
        """
        # Convert predictions to class indices if needed
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)  # [B, H, W]

        # Move to CPU and convert to numpy
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # Filter out ignore index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def get_iou(self) -> np.ndarray:
        """
        Compute IoU for each class.

        Returns:
            iou: [num_classes] array
        """
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection

        # Avoid division by zero
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if union[i] > 0:
                iou[i] = intersection[i] / union[i]

        return iou

    def get_miou(self) -> float:
        """
        Compute mean IoU.

        Returns:
            miou: scalar
        """
        iou = self.get_iou()
        # Only average over classes that appear in the dataset
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        if valid_classes.sum() > 0:
            miou = iou[valid_classes].mean()
        else:
            miou = 0.0
        return miou

    def get_pixel_accuracy(self) -> float:
        """
        Compute pixel accuracy.

        Returns:
            accuracy: scalar
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0
        return accuracy

    def get_class_accuracy(self) -> np.ndarray:
        """
        Compute accuracy for each class.

        Returns:
            class_acc: [num_classes] array
        """
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)

        class_acc = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if total[i] > 0:
                class_acc[i] = correct[i] / total[i]

        return class_acc

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Returns:
            metrics_dict: dict with all metrics
        """
        iou = self.get_iou()
        miou = self.get_miou()
        pixel_acc = self.get_pixel_accuracy()
        class_acc = self.get_class_accuracy()

        metrics = {
            'mIoU': miou,
            'pixel_acc': pixel_acc,
        }

        # Add per-class IoU
        for i in range(self.num_classes):
            metrics[f'IoU_class_{i}'] = iou[i]
            metrics[f'Acc_class_{i}'] = class_acc[i]

        return metrics

    def print_summary(self, class_names: List[str] = None):
        """
        Print metric summary.

        Args:
            class_names: Optional list of class names
        """
        metrics = self.get_summary()

        print(f"\n{'='*60}")
        print(f"Segmentation Metrics Summary")
        print(f"{'='*60}")
        print(f"mIoU: {metrics['mIoU']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")

        print(f"\nPer-class IoU:")
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            iou = metrics[f'IoU_class_{i}']
            acc = metrics[f'Acc_class_{i}']
            print(f"  {class_name:20s}: IoU={iou:.4f}, Acc={acc:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Testing SegmentationMetrics:")

    num_classes = 5
    B, H, W = 2, 64, 64

    metrics = SegmentationMetrics(num_classes)

    # Generate random predictions and targets
    pred = torch.randn(B, num_classes, H, W)
    target = torch.randint(0, num_classes, (B, H, W))

    # Update metrics
    metrics.update(pred, target)

    # Get metrics
    print(f"mIoU: {metrics.get_miou():.4f}")
    print(f"Pixel Accuracy: {metrics.get_pixel_accuracy():.4f}")

    # Print summary
    class_names = [f"Class_{i}" for i in range(num_classes)]
    metrics.print_summary(class_names)

    print("✓ Metrics test passed!")
