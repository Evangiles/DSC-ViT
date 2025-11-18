"""
Training script for DSC-ViT.

Usage:
    python train.py --config configs/default.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
from pathlib import Path
import numpy as np

# Use non-interactive backend for matplotlib (no GUI windows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import DSCViT, SwinDSCViT
from utils import DeepSupervisionLoss, SegmentationMetrics
from data import get_dataset, VOCSegmentationKaggle
from data.voc_sbd_combined import get_combined_dataset
from data.transforms import SegmentationTransform, get_train_transforms, get_val_transforms


class Trainer:
    """
    Trainer for DSC-ViT and Swin-DSC-ViT.
    """

    @staticmethod
    def _create_model(config, for_ema=False):
        """
        Factory method to create the appropriate model (DSCViT or SwinDSCViT).

        Args:
            config: Configuration dictionary
            for_ema: If True, don't load pretrained weights

        Returns:
            Model instance
        """
        num_clusters = config['model'].get('num_clusters') or config['model']['num_classes']

        # Check if we should use Swin-DSC-ViT
        use_swin = 'swin_model_name' in config['model']

        if use_swin:
            # Create Swin-DSC-ViT
            model = SwinDSCViT(
                image_channels=config['model']['image_channels'],
                num_classes=config['model']['num_classes'],
                img_size=config['model']['img_size'],
                num_clusters=num_clusters,
                cluster_temperature=config.get('cluster_temperature', {}).get('initial', 1.0),
                latent_dim=config['model']['latent_dim'],
                swin_model_name=config['model']['swin_model_name'],
                use_pretrained_swin=config['model']['use_pretrained_swin'] and not for_ema,
                num_latent_updates=config['model']['num_latent_updates'],
                num_recursive_steps=config['model']['num_recursive_steps'],
                fusion_method=config['model']['fusion_method'],
                freeze_swin=config['model'].get('freeze_swin', False) if not for_ema else False,
                use_deep_supervision=True,
                extract_stages=config['model'].get('extract_stages', [0, 1, 2, 3]),
                fusion_resolution=config['model'].get('fusion_resolution', 16)
            )
        else:
            # Create standard DSC-ViT
            model = DSCViT(
                image_channels=config['model']['image_channels'],
                num_classes=config['model']['num_classes'],
                img_size=config['model']['img_size'],
                num_clusters=num_clusters,
                clustering_method=config['model'].get('clustering_method', 'soft'),
                latent_dim=config['model']['latent_dim'],
                vit_model_name=config['model']['vit_model_name'],
                use_pretrained_vit=config['model']['use_pretrained_vit'] and not for_ema,
                use_simple_encoder=config['model']['use_simple_encoder'],
                num_latent_updates=config['model']['num_latent_updates'],
                num_recursive_steps=config['model']['num_recursive_steps'],
                fusion_method=config['model']['fusion_method'],
                use_spatial_context=config['model'].get('use_spatial_context', False),
                spatial_context_type=config['model'].get('spatial_context_type', 'simple'),
                spatial_context_dropout=config['model'].get('spatial_context_dropout', 0.1),
                freeze_vit=config['model']['freeze_vit'] if not for_ema else False,
                use_deep_supervision=True
            )

        return model

    def __init__(self, config, enable_visualization=False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_visualization = enable_visualization

        # Create model
        self.model = self._create_model(config, for_ema=False).to(self.device)

        # Get num_clusters from config
        num_clusters = config['model'].get('num_clusters') or config['model']['num_classes']

        # Loss function
        self.criterion = DeepSupervisionLoss(
            num_classes=config['model']['num_classes'],
            num_clusters=num_clusters,
            lambda_aux=config['loss']['lambda_aux'],
            use_cluster_loss=config['loss']['use_cluster_loss'],
            cluster_loss_alpha=config['loss']['cluster_loss_alpha'],
            cluster_loss_beta=config['loss']['cluster_loss_beta'],
            use_assignment_loss=config['loss'].get('use_assignment_loss', True),
            use_compactness_loss=config['loss'].get('use_compactness_loss', True),
            use_separation_loss=config['loss'].get('use_separation_loss', True),
            use_focal_loss=config['loss'].get('use_focal_loss', False),
            focal_alpha=config['loss'].get('focal_alpha', 0.25),
            focal_gamma=config['loss'].get('focal_gamma', 2.0),
            use_bce=config['loss'].get('use_bce', False),
            bce_weight=config['loss'].get('bce_weight', 0.5)
        )

        # Optimizer with parameter groups (different LR for spatial context)
        if config['optimizer']['name'] == 'adamw':
            # Separate parameters for spatial context (if it exists)
            spatial_context_lr_scale = config['optimizer'].get('spatial_context_lr_scale', 0.1)

            if hasattr(self.model, 'spatial_context_encoder') and self.model.spatial_context_encoder is not None:
                # Spatial context parameters
                spatial_params = list(self.model.spatial_context_encoder.parameters())
                spatial_param_ids = set(id(p) for p in spatial_params)

                # Other parameters (excluding spatial context)
                other_params = [p for p in self.model.parameters() if id(p) not in spatial_param_ids]

                # Create parameter groups with different learning rates
                param_groups = [
                    {
                        'params': other_params,
                        'lr': config['optimizer']['lr'],
                        'weight_decay': config['optimizer']['weight_decay']
                    },
                    {
                        'params': spatial_params,
                        'lr': config['optimizer']['lr'] * spatial_context_lr_scale,
                        'weight_decay': config['optimizer']['weight_decay']
                    }
                ]

                print(f"\nOptimizer parameter groups:")
                print(f"  Main model:     lr={config['optimizer']['lr']:.2e}, params={len(other_params)}")
                print(f"  Spatial context: lr={config['optimizer']['lr'] * spatial_context_lr_scale:.2e}, params={len(spatial_params)}")
            else:
                # No spatial context, use all parameters with same LR
                param_groups = self.model.parameters()

            self.optimizer = optim.AdamW(
                param_groups,
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
        ema_model = self._create_model(self.config, for_ema=True
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
        loss_dict_total = {
            'main': 0.0,
            'main_primary': 0.0,
            'main_bce': 0.0,
            'aux': 0.0,
            'cluster': 0.0
        }

        N_sup = self.config['training']['num_supervision_steps']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Pre-compute downsampled target (once per batch, not N_sup times)
            # This optimization avoids redundant interpolation operations
            target_small = None
            if self.criterion.use_cluster_loss:
                encoder_size = self.model.encoder_output_size
                target_small = torch.nn.functional.interpolate(
                    targets.float().unsqueeze(1),
                    size=(encoder_size, encoder_size),
                    mode='nearest'
                ).squeeze(1).long()

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

                # Main segmentation loss (with detailed breakdown)
                loss, loss_detail = self.criterion.seg_loss(seg_pred, targets, return_dict=True)
                loss_dict_total['main'] += loss.item()
                loss_dict_total['main_primary'] += loss_detail['primary']
                loss_dict_total['main_bce'] += loss_detail['bce']
                # Note: aux is not computed in this training loop structure (using N_sup instead of T-step collection)

                # Cluster space loss (if enabled)
                if self.criterion.use_cluster_loss:
                    # Use pre-computed downsampled target (cached)
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
            print(f"  Main: {train_loss_dict['main']:.4f} (Focal/CE: {train_loss_dict['main_primary']:.4f}, BCE: {train_loss_dict['main_bce']:.4f})")
            print(f"  Cluster: {train_loss_dict['cluster']:.4f}")

            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss, val_metrics = self.validate(val_loader)

                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val mIoU: {val_metrics['mIoU']:.4f}")
                print(f"Val Pixel Acc: {val_metrics['pixel_acc']:.4f}")

                # Visualize clustering results (if enabled)
                if self.enable_visualization:
                    self.visualize_clustering(val_loader, epoch, num_samples=3)

                # Save best model
                if val_metrics['mIoU'] > best_miou:
                    best_miou = val_metrics['mIoU']
                    self.save_checkpoint(epoch, val_metrics['mIoU'], is_best=True)
                    print(f"New best mIoU: {best_miou:.4f}")

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Temperature annealing for cluster assignments
            if 'cluster_temperature' in self.config:
                temp_config = self.config['cluster_temperature']
                progress = (epoch + 1) / self.config['training']['epochs']

                # Compute annealed temperature
                if temp_config['schedule'] == 'linear':
                    new_temp = temp_config['initial'] + progress * (temp_config['final'] - temp_config['initial'])
                elif temp_config['schedule'] == 'cosine':
                    import math
                    new_temp = temp_config['final'] + 0.5 * (temp_config['initial'] - temp_config['final']) * \
                              (1 + math.cos(math.pi * progress))
                elif temp_config['schedule'] == 'exponential':
                    new_temp = temp_config['initial'] * (temp_config['final'] / temp_config['initial']) ** progress
                else:
                    new_temp = temp_config['initial']

                # Update temperature in all clustering layers
                for module in self.model.modules():
                    if hasattr(module, 'update_temperature'):
                        module.update_temperature(new_temp)

                if epoch % 10 == 0:  # Print every 10 epochs
                    print(f"Cluster temperature: {new_temp:.4f}")

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

    def visualize_clustering(self, val_loader, epoch, num_samples=3):
        """
        Visualize clustering results on validation samples.

        Saves images showing:
        - Original image
        - Ground truth segmentation
        - Predicted segmentation
        - Cluster assignments
        """
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        # Create visualization directory
        vis_dir = self.exp_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        # Get a batch from validation loader
        images, targets = next(iter(val_loader))
        images = images.to(self.device)
        targets = targets.to(self.device)

        # Limit to num_samples
        images = images[:num_samples]
        targets = targets[:num_samples]

        N_sup = self.config['training']['num_supervision_steps']

        with torch.no_grad():
            # Track clustering evolution across recursive steps
            T = model.T  # Number of recursive steps
            cluster_evolution = []

            # Run model and capture clusters at each recursive step
            y, z = None, None

            # Run N_sup-1 steps normally
            for step in range(N_sup - 1):
                (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(
                    images, y, z
                )

            # On last step, capture intermediate clusters
            for t in range(T):
                # One recursive step
                (y, z), seg_pred, q_logit, combined_cluster, image_cluster = model(
                    images, y, z
                )

                # Get cluster assignments (cluster z only, not y+z)
                z_cluster = model.projections.latent_to_cluster(z)
                _, assignments_t, _ = model.clustering_layer(z_cluster)

                # assignments_t shape: [B, H', W', K] where H'=W'=16
                # Convert to hard assignments [B, H', W']
                if assignments_t.dim() == 4:
                    # Get hard cluster assignment (argmax over K dimension)
                    assignments_t = assignments_t.argmax(dim=-1)  # [B, H', W']
                elif assignments_t.dim() == 3:
                    # If [B, HW, K]
                    assignments_t = assignments_t.argmax(dim=-1)  # [B, HW]
                    B_vis = images.shape[0]
                    H_vis = W_vis = int(assignments_t.shape[1] ** 0.5)
                    assignments_t = assignments_t.reshape(B_vis, H_vis, W_vis)

                # Now assignments_t is [B, 16, 16] - upsample to [B, 256, 256]
                H_target, W_target = images.shape[2], images.shape[3]
                print(f"[DEBUG] Before upsample: assignments_t.shape = {assignments_t.shape}")
                print(f"[DEBUG] Target size: ({H_target}, {W_target})")

                assignments_t = F.interpolate(
                    assignments_t.unsqueeze(1).float(),  # [B, 1, 16, 16]
                    size=(H_target, W_target),           # (256, 256)
                    mode='nearest'
                ).squeeze(1).long()  # [B, 256, 256]

                print(f"[DEBUG] After upsample: assignments_t.shape = {assignments_t.shape}")

                cluster_evolution.append(assignments_t.clone())

            # Use final assignments
            assignments = cluster_evolution[-1]

        # Move to CPU for visualization
        images_cpu = images.cpu()
        targets_cpu = targets.cpu()
        seg_pred_cpu = seg_pred.argmax(dim=1).cpu()
        assignments_cpu = assignments.cpu()
        cluster_evolution_cpu = [c.cpu() for c in cluster_evolution]

        # Create figure with evolution
        num_rows = num_samples
        num_cols = 3 + len(cluster_evolution)  # Original, GT, Prediction, + T cluster steps
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Original image
            img = images_cpu[i].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Sample {i+1}: Original')
            axes[i, 0].axis('off')

            # Ground truth
            gt = targets_cpu[i].numpy()
            gt_vis = np.ma.masked_where(gt == -100, gt)  # Mask ignore index
            axes[i, 1].imshow(gt_vis, cmap='tab20', vmin=0, vmax=20)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Prediction
            pred = seg_pred_cpu[i].numpy()
            axes[i, 2].imshow(pred, cmap='tab20', vmin=0, vmax=20)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

            # Cluster evolution across T steps
            for t_idx, cluster_t in enumerate(cluster_evolution_cpu):
                cluster_map = cluster_t[i].numpy()
                axes[i, 3 + t_idx].imshow(cluster_map, cmap='nipy_spectral',
                                         vmin=0, vmax=model.K-1)
                axes[i, 3 + t_idx].set_title(f'Cluster T={t_idx+1}')
                axes[i, 3 + t_idx].axis('off')

        plt.tight_layout()
        save_path = vis_dir / f'clustering_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--debug', action='store_true',
                       help='Use small subset for debugging')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable clustering visualization (saves PNG files)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create datasets
    print("\n" + "="*60)
    print("Loading Combined VOC + SBD Dataset")
    print("="*60)

    # Create transforms
    train_transform = SegmentationTransform(
        get_train_transforms(
            img_size=config['model']['img_size']
        )
    )
    val_transform = SegmentationTransform(
        get_val_transforms(
            img_size=config['model']['img_size']
        )
    )

    # Get combined VOC + SBD dataset
    train_dataset, val_dataset = get_combined_dataset(
        transform_train=train_transform,
        transform_val=val_transform
    )

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
    trainer = Trainer(config, enable_visualization=args.visualize)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
