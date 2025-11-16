"""
Test training with synthetic data to verify the pipeline works.

This allows us to test the model without downloading datasets.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import yaml

from models import DSCViT
from utils import DeepSupervisionLoss, SegmentationMetrics


def create_synthetic_dataset(num_samples=100, img_size=256, num_classes=21):
    """Create synthetic dataset for testing."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    # Create realistic segmentation masks (not just random noise)
    masks = torch.zeros(num_samples, img_size, img_size, dtype=torch.long)

    # Add some structured patterns to masks
    for i in range(num_samples):
        # Background
        masks[i] = 0

        # Add some random objects
        num_objects = torch.randint(1, 5, (1,)).item()
        for obj in range(num_objects):
            class_id = torch.randint(1, num_classes, (1,)).item()
            x = torch.randint(0, img_size - 50, (1,)).item()
            y = torch.randint(0, img_size - 50, (1,)).item()
            w = torch.randint(20, 80, (1,)).item()
            h = torch.randint(20, 80, (1,)).item()
            masks[i, y:y+h, x:x+w] = class_id

    return TensorDataset(images, masks)


def test_training():
    """Test training pipeline with synthetic data."""

    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Override settings for quick test
    config['training']['epochs'] = 3
    config['training']['num_supervision_steps'] = 4  # Reduce from 16
    config['training']['batch_size'] = 4
    config['model']['use_simple_encoder'] = True  # Use simple encoder (no timm needed)

    print("="*60)
    print("Testing DSC-ViT Training Pipeline")
    print("="*60)
    print(f"Config:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  N_sup: {config['training']['num_supervision_steps']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Simple encoder: {config['model']['use_simple_encoder']}")

    # Create synthetic datasets
    print("\nCreating synthetic datasets...")
    train_dataset = create_synthetic_dataset(num_samples=100, img_size=256, num_classes=21)
    val_dataset = create_synthetic_dataset(num_samples=20, img_size=256, num_classes=21)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle num_clusters (if None, use num_classes)
    num_clusters = config['model'].get('num_clusters') or config['model']['num_classes']

    model = DSCViT(
        image_channels=config['model']['image_channels'],
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        num_clusters=num_clusters,
        latent_dim=config['model']['latent_dim'],
        vit_model_name=config['model']['vit_model_name'],
        use_pretrained_vit=False,  # Don't use pretrained (no download)
        use_simple_encoder=config['model']['use_simple_encoder'],
        num_latent_updates=config['model']['num_latent_updates'],
        num_recursive_steps=config['model']['num_recursive_steps'],
        fusion_method=config['model']['fusion_method'],
        freeze_vit=False,
        use_deep_supervision=True
    ).to(device)

    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss and optimizer
    criterion = DeepSupervisionLoss(
        num_classes=config['model']['num_classes'],
        num_clusters=num_clusters,
        lambda_aux=config['loss']['lambda_aux'],
        use_cluster_loss=config['loss']['use_cluster_loss'],
        cluster_loss_alpha=config['loss']['cluster_loss_alpha'],
        cluster_loss_beta=config['loss']['cluster_loss_beta']
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        weight_decay=config['optimizer']['weight_decay']
    )

    metrics = SegmentationMetrics(config['model']['num_classes'])

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    N_sup = config['training']['num_supervision_steps']

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Deep Supervision Loop
            y, z = None, None
            batch_loss = 0.0

            for step in range(N_sup):
                # Forward
                (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(images, y, z)

                # Loss
                loss = criterion.seg_loss(seg_pred, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

                # ACT early stopping
                if config['training'].get('use_act', False):
                    if q_logit.mean() > 0:
                        break

            epoch_loss += batch_loss / N_sup
            num_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={batch_loss/N_sup:.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % 1 == 0:
            model.eval()
            metrics.reset()
            val_loss = 0.0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    # Run all N_sup steps
                    y, z = None, None
                    for step in range(N_sup):
                        (y, z), seg_pred, q_logit, _, _ = model(images, y, z)

                    loss = criterion.seg_loss(seg_pred, targets)
                    val_loss += loss.item()
                    metrics.update(seg_pred, targets)

            val_metrics = metrics.get_summary()
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
            print(f"  Val mIoU: {val_metrics['mIoU']:.4f}")
            print(f"  Val Pixel Acc: {val_metrics['pixel_acc']:.4f}")

    print("\n" + "="*60)
    print("✓ Training test completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download PASCAL VOC manually if needed")
    print("  2. Run full training: python train.py --config configs/default.yaml")
    print("="*60)


if __name__ == "__main__":
    test_training()
