"""
Training script for DSC-ViT.

Usage:
    python train.py --config configs/default.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
from pathlib import Path

from models import DSCViT
from utils import DeepSupervisionLoss, SegmentationMetrics
from data import get_dataset, VOCSegmentationKaggle
from data.transforms import SegmentationTransform, get_train_transforms, get_val_transforms


class Trainer:
    """
    Trainer for DSC-ViT.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        # Handle num_clusters (if None, use num_classes)
        num_clusters = config['model'].get('num_clusters') or config['model']['num_classes']

        self.model = DSCViT(
            image_channels=config['model']['image_channels'],
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size'],
            num_clusters=num_clusters,
            latent_dim=config['model']['latent_dim'],
            vit_model_name=config['model']['vit_model_name'],
            use_pretrained_vit=config['model']['use_pretrained_vit'],
            use_simple_encoder=config['model']['use_simple_encoder'],
            num_latent_updates=config['model']['num_latent_updates'],
            num_recursive_steps=config['model']['num_recursive_steps'],
            fusion_method=config['model']['fusion_method'],
            freeze_vit=config['model']['freeze_vit'],
            use_deep_supervision=True
        ).to(self.device)

        # Loss function
        self.criterion = DeepSupervisionLoss(
            num_classes=config['model']['num_classes'],
            num_clusters=num_clusters,
            lambda_aux=config['loss']['lambda_aux'],
            use_cluster_loss=config['loss']['use_cluster_loss'],
            cluster_loss_alpha=config['loss']['cluster_loss_alpha'],
            cluster_loss_beta=config['loss']['cluster_loss_beta']
        )

        # Optimizer
        if config['optimizer']['name'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['optimizer']['lr'],
                betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                weight_decay=config['optimizer']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']['name']}")

        # Scheduler
        if config['scheduler']['name'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['scheduler']['min_lr']
            )
        elif config['scheduler']['name'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['scheduler']['step_size'],
                gamma=config['scheduler']['gamma']
            )
        else:
            self.scheduler = None

        # Metrics
        self.metrics = SegmentationMetrics(config['model']['num_classes'])

        # EMA (Exponential Moving Average)
        self.use_ema = config['training'].get('use_ema', False)
        if self.use_ema:
            self.ema_decay = config['training']['ema_decay']
            self.ema_model = self._create_ema_model()

        # Experiment directory
        self.exp_dir = Path(config['training']['exp_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def _create_ema_model(self):
        """Create EMA model."""
        ema_model = DSCViT(
            image_channels=self.config['model']['image_channels'],
            num_classes=self.config['model']['num_classes'],
            img_size=self.config['model']['img_size'],
            num_clusters=self.config['model'].get('num_clusters', None),
            latent_dim=self.config['model']['latent_dim'],
            vit_model_name=self.config['model']['vit_model_name'],
            use_pretrained_vit=False,  # Don't load pretrained for EMA
            use_simple_encoder=self.config['model']['use_simple_encoder'],
            num_latent_updates=self.config['model']['num_latent_updates'],
            num_recursive_steps=self.config['model']['num_recursive_steps'],
            fusion_method=self.config['model']['fusion_method'],
            freeze_vit=False,
            use_deep_supervision=True
        ).to(self.device)

        # Initialize with current model weights
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()

        # Freeze EMA model
        for param in ema_model.parameters():
            param.requires_grad = False

        return ema_model

    def _update_ema(self):
        """Update EMA model weights."""
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch with TRM-style Deep Supervision.

        Deep Supervision (TRM): Each batch is processed N_sup times.
        Each supervision step:
        1. Forward with deep_recursion (T iterations, gradient on last)
        2. Compute loss and backward
        3. Optimizer step
        4. Detach y, z for next step (gradient isolation)
        5. ACT early stopping if q_logit > 0
        """
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        loss_dict_total = {'main': 0.0, 'cluster': 0.0}

        N_sup = self.config['training']['num_supervision_steps']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Initialize y and z as None (model will initialize them)
            y, z = None, None

            # Deep Supervision Loop: process same batch N_sup times
            for step in range(N_sup):

                # TRM-style forward: deep_recursion with gradient strategy
                (y, z), seg_pred, q_logit, combined_cluster, image_cluster = self.model(
                    images, y, z
                )

                # Compute loss for this supervision step
                cluster_centers = self.model.get_cluster_centers()

                # Main segmentation loss
                loss = self.criterion.seg_loss(seg_pred, targets)
                loss_dict_total['main'] += loss.item()

                # Cluster space loss (if enabled)
                if self.criterion.use_cluster_loss:
                    # Downsample target to match cluster size
                    target_small = torch.nn.functional.interpolate(
                        targets.float().unsqueeze(1),
                        size=combined_cluster.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()

                    cluster_loss, _ = self.criterion.cluster_loss(
                        combined_cluster,
                        target_small,
                        cluster_centers
                    )
                    loss = loss + cluster_loss
                    loss_dict_total['cluster'] += cluster_loss.item()

                # ACT training loss (TRM paper: train q_head to predict correctness)
                if self.config['training'].get('use_act', False):
                    # Compute target: 1 if prediction matches ground truth, 0 otherwise
                    with torch.no_grad():
                        y_pred_classes = torch.argmax(seg_pred, dim=1)  # [B, H, W]
                        # Per-pixel correctness, then average to get confidence score
                        target_halt = (y_pred_classes == targets).float().mean(dim=[1, 2], keepdim=True)  # [B, 1, 1]

                    # q_logit: [B, 1, H, W] - spatial halt prediction
                    # Expand target to match q_logit shape
                    target_halt_expanded = target_halt.unsqueeze(-1).expand_as(q_logit)  # [B, 1, H, W]

                    # Binary cross-entropy: train q_head to predict correctness
                    act_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        q_logit, target_halt_expanded
                    )
                    loss = loss + act_loss
                    loss_dict_total['main'] += act_loss.item()

                # Backward pass (for this supervision step only)
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config['training'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                # Optimizer step (for this supervision step only)
                self.optimizer.step()

                # Update EMA
                self._update_ema()

                # Accumulate losses
                total_loss += loss.item()
                total_steps += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'step': f'{step+1}/{N_sup}'
                })

                # ACT early stopping (TRM paper: halt if q_logit > 0)
                if self.config['training'].get('use_act', False):
                    # q_logit is [B, 1, H, W], take mean across batch and spatial dims
                    q_mean = q_logit.mean().item()

                    # TRM paper uses threshold=0: stop if model is confident answer is correct
                    act_threshold = self.config['training'].get('act_threshold', 0.0)
                    if q_mean > act_threshold:
                        break

        # Average losses
        total_loss /= total_steps
        for key in loss_dict_total:
            loss_dict_total[key] /= total_steps

        return total_loss, loss_dict_total

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on validation set with TRM-style Deep Supervision.

        During validation, we run all N_sup steps to get the best prediction.
        """
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        self.metrics.reset()
        total_loss = 0.0

        N_sup = self.config['training']['num_supervision_steps']

        pbar = tqdm(val_loader, desc="Validation")

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Initialize y and z as None
            y, z = None, None

            # Run all N_sup steps (no early stopping in validation)
            for step in range(N_sup):
                (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(
                    images, y, z
                )

            # Use final prediction
            # Compute loss
            loss = self.criterion.seg_loss(seg_pred, targets)
            total_loss += loss.item()

            # Update metrics
            self.metrics.update(seg_pred, targets)

        # Average loss
        total_loss /= len(val_loader)

        # Get metrics
        metrics = self.metrics.get_summary()

        return total_loss, metrics

    def train(self, train_loader, val_loader):
        """Full training loop."""
        best_miou = 0.0

        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

            # Train
            train_loss, train_loss_dict = self.train_epoch(train_loader, epoch)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"  Main: {train_loss_dict['main']:.4f}, Cluster: {train_loss_dict['cluster']:.4f}")

            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss, val_metrics = self.validate(val_loader)

                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val mIoU: {val_metrics['mIoU']:.4f}")
                print(f"Val Pixel Acc: {val_metrics['pixel_acc']:.4f}")

                # Save best model
                if val_metrics['mIoU'] > best_miou:
                    best_miou = val_metrics['mIoU']
                    self.save_checkpoint(epoch, val_metrics['mIoU'], is_best=True)
                    print(f"New best mIoU: {best_miou:.4f}")

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, 0.0, is_best=False)

        print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")

    def save_checkpoint(self, epoch, miou, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'miou': miou,
            'config': self.config
        }

        if self.use_ema:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        # Save regular checkpoint
        save_path = self.exp_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, save_path)

        # Save best checkpoint
        if is_best:
            best_path = self.exp_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--debug', action='store_true',
                       help='Use small subset for debugging')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create datasets
    print("\n" + "="*60)
    print("Loading PASCAL VOC 2012 Dataset")
    print("="*60)

    # Training dataset
    train_transform = SegmentationTransform(
        get_train_transforms(
            img_size=config['model']['img_size']
        )
    )
    # Use Kaggle VOC dataset
    train_dataset = VOCSegmentationKaggle(
        split='train',
        transform=train_transform
    )

    # Validation dataset
    val_transform = SegmentationTransform(
        get_val_transforms(
            img_size=config['model']['img_size']
        )
    )
    val_dataset = VOCSegmentationKaggle(
        split='valid',
        transform=val_transform
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} images ({config['data']['train_split']})")
    print(f"  Val:   {len(val_dataset)} images ({config['data']['val_split']})")

    # Debug mode: use small subset
    if args.debug:
        print("\n⚠️  DEBUG MODE: Using small subset")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(20, len(val_dataset))))
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val:   {len(val_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    print("="*60)

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
