"""
Check prediction distribution to diagnose low mIoU.
"""
import torch
import yaml
import numpy as np
from models import DSCViT
from data import get_ade20k_dataset, get_val_transforms
from data.transforms import SegmentationTransform
from torch.utils.data import DataLoader

# Load config
with open('configs/ade20k.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model
print("Loading model...")
model = DSCViT(
    image_channels=3,
    num_classes=150,
    img_size=256,
    num_clusters=8,
    clustering_method='soft',
    latent_dim=512,
    vit_model_name='vit_base_patch16_224',
    use_pretrained_vit=True,
    use_simple_encoder=False,
    num_latent_updates=6,
    num_recursive_steps=3,
    fusion_method='attention',
    use_spatial_context=False,
    freeze_vit=True,
    use_deep_supervision=True
).cuda()

# Load checkpoint
ckpt = torch.load('experiments/ade20k/best_model.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load validation data
print("Loading dataset...")
val_transform = SegmentationTransform(
    get_val_transforms(img_size=256)
)
_, val_dataset = get_ade20k_dataset(
    root='./data/ADEChallengeData2016',
    transform_train=None,
    transform_val=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Collect predictions and ground truth
print("Analyzing predictions...")
pred_counts = np.zeros(150)
gt_counts = np.zeros(150)
total_pixels = 0

with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        if i >= 20:  # Check first 20 batches
            break

        images = images.cuda()

        # Forward
        y = torch.zeros(images.size(0), 512, 16, 16).cuda()
        z = torch.zeros_like(y)

        outputs = model(images, y, z)
        seg_pred = outputs[1]  # [B, 150, H, W]

        # Get predictions
        preds = seg_pred.argmax(dim=1).cpu().numpy()
        gt = masks.cpu().numpy()

        # Count predictions
        for c in range(150):
            pred_counts[c] += (preds == c).sum()
            gt_counts[c] += (gt == c).sum()
        total_pixels += preds.size

print("\n" + "="*80)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*80)

print('\nTop 15 PREDICTED classes:')
top_pred = np.argsort(pred_counts)[::-1][:15]
for c in top_pred:
    pct = pred_counts[c] / total_pixels * 100
    gt_pct = gt_counts[c] / total_pixels * 100
    print(f'  Class {c:3d}: Pred={pct:>5.2f}%, GT={gt_pct:>5.2f}%')

print('\nTop 15 GROUND TRUTH classes:')
top_gt = np.argsort(gt_counts)[::-1][:15]
for c in top_gt:
    pct = pred_counts[c] / total_pixels * 100
    gt_pct = gt_counts[c] / total_pixels * 100
    print(f'  Class {c:3d}: Pred={pct:>5.2f}%, GT={gt_pct:>5.2f}%')

zero_pred = (pred_counts == 0).sum()
zero_gt = (gt_counts == 0).sum()

print(f'\nClasses NEVER predicted: {zero_pred}/150')
print(f'Classes NOT in ground truth (these 20 batches): {zero_gt}/150')
print(f'\nEntropy of predictions: {-(pred_counts/total_pixels * np.log(pred_counts/total_pixels + 1e-10)).sum():.4f}')
print(f'Max entropy (uniform): {np.log(150):.4f}')
print("="*80)
