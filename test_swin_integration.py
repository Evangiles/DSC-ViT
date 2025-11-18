"""
Quick test to verify Swin-DSC-ViT integration with trainer.
"""

import torch
import yaml
from train import Trainer

# Load Swin config
with open('configs/swin.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("Testing Swin-DSC-ViT Integration")
print("=" * 70)

# Override some settings for quick test
config['training']['batch_size'] = 2
config['training']['num_workers'] = 0
config['training']['num_supervision_steps'] = 2
config['model']['use_pretrained_swin'] = False  # Faster for testing

# Create trainer
print("\nCreating trainer...")
trainer = Trainer(config, enable_visualization=False)

print("\nModel type:", type(trainer.model).__name__)
print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters())/1e6:.2f}M")

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(2, 3, 256, 256).to(trainer.device)

with torch.no_grad():
    (y, z), seg_pred, q_logit, combined_cluster, image_cluster = trainer.model(dummy_input)

print(f"\nOutputs:")
print(f"  y: {y.shape}")
print(f"  z: {z.shape}")
print(f"  seg_pred: {seg_pred.shape}")
print(f"  q_logit: {q_logit.shape}")
print(f"  combined_cluster: {combined_cluster.shape}")
print(f"  image_cluster: {image_cluster.shape}")

# Test EMA model creation
if config['training'].get('use_ema', False):
    print("\nTesting EMA model creation...")
    ema_model = trainer._create_ema_model()
    print(f"EMA model type: {type(ema_model).__name__}")
    print(f"EMA model parameters: {sum(p.numel() for p in ema_model.parameters())/1e6:.2f}M")

print("\n" + "=" * 70)
print("✓ Swin-DSC-ViT integration test passed!")
print("=" * 70)
