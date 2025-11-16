# DSC-ViT: Deep Supervised Vision Transformer with Clustering Hidden State

PyTorch implementation of DSC-ViT for image segmentation, inspired by the Tiny Recursive Model (TRM) paper.

## Key Innovations

### 1. **Cluster Space Integration**
Instead of projecting cluster features to visual space (K→C→K→D), we project the original image directly to cluster space (C→K→D). This:
- Reduces computational complexity by 33%
- Improves interpretability (cluster activations directly visible)
- Maintains better information flow throughout recursion

### 2. **⭐ ViT Outside Recursive Loop (NEW!)**
Unlike naive implementations that call ViT n×T times per step, we compute ViT features **once** and cache them:
- **18x speedup** in recursive refinement (288 → 16 ViT calls for N_sup=16)
- Enables **frozen ViT** as feature extractor (pre-trained knowledge preserved)
- **Lightweight modules** (projections, clustering) handle recursive learning
- True to TRM's philosophy: "tiny recursive network" with ~5M trainable parameters

## Architecture Overview

```
Input Image (H×W×C)
  ↓
  ⭐ ViT Encoding (ONCE): image → x_vit [B, D, H', W'] [cached]
  ↓
  ⭐ Project to Cluster (ONCE): image → image_cluster [B, K, H', W'] [cached]
  ↓
  ┌─────────── Deep Supervision Loop (N_sup=16 times) ───────────┐
  │                                                               │
  │  ┌─────────── Recursive Loop (T=3 times) ───────────┐       │
  │  │  ┌── Latent Update Loop (n=6 times) ──────┐     │       │
  │  │  │                                         │     │       │
  │  │  │  1. Project to Cluster: (y+z) → D→K    │     │       │
  │  │  │  2. Soft K-Means Clustering in K-space │     │       │
  │  │  │  3. Combine: z_clustered + image_cluster│    │       │
  │  │  │  4. Project back: K→D                   │     │       │
  │  │  │  5. Add ViT residual: z = z + x_vit   │     │       │
  │  │  │                                         │     │       │
  │  │  └─────────────────────────────────────────┘     │       │
  │  │  ↓                                                │       │
  │  │  Answer Update: y = y + z                        │       │
  │  └───────────────────────────────────────────────────┘       │
  │  ↓                                                            │
  │  Segmentation Prediction from y                              │
  │  ↓                                                            │
  │  Loss Computation → Backward → Optimizer Step                │
  │  ↓                                                            │
  │  y, z = y.detach(), z.detach() [gradient isolation]         │
  │  ↓                                                            │
  │  Early Stop? (ACT) → Yes: break | No: continue               │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
  ↓
Final Prediction (after N_sup refinements)
```

### Deep Supervision Mechanism (Key Innovation from TRM)

**Core Concept**: Each batch is refined **N_sup times** (up to 16), with independent backward passes per step.

```python
for each batch in dataloader:
    z_latent = initial_latent

    for step in range(N_sup):  # Deep Supervision Loop
        # T recursive refinements
        for t in range(T):
            z_latent = recursive_step(z_latent, image_cluster)

        # Compute loss and update weights
        loss = compute_loss(z_latent, target)
        loss.backward()  # ⭐ Gradient for this step only
        optimizer.step()

        # ⭐ Detach: gradient isolation
        z_latent = z_latent.detach()

        # Early stopping (ACT)
        if accuracy > threshold:
            break  # Move to next batch
```

**Why This Works**:
1. **Progressive Refinement**: Each step improves the latent representation
2. **Gradient Isolation**: Detaching prevents gradient flow across steps
3. **Effective Depth**: Emulates 16×3×layers = 48× effective network depth
4. **Memory Efficient**: Only backprop through one step at a time

## Installation

```bash
# Clone repository
cd Segmentation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Test the Model

Run individual module tests:

```bash
# Test projection layers
python -m models.projections

# Test Soft K-Means
python -m models.soft_kmeans

# Test ViT encoder
python -m models.vit_encoder

# Test full DSC-ViT model
python -m models.dsc_vit

# Test loss functions
python -m utils.losses

# Test metrics
python -m utils.metrics

# ⭐ Benchmark ViT optimization (verify 18x speedup)
python benchmark_vit_calls.py
```

### 2. Training

```bash
# Train with default config
python train.py --config configs/default.yaml
```

### 3. Custom Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (K, D, T, fusion method)
- Loss weights (λ_aux, α, β)
- Optimizer settings (lr, weight decay)
- Training settings (batch size, epochs)

## Model Configuration

### Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `num_clusters` | K | num_classes | Number of cluster centers |
| `latent_dim` | D | 512 | ViT latent dimension |
| `num_recursive_steps` | T | 3 | Recursive refinement steps |
| `fusion_method` | - | residual | Cluster fusion: residual/gated/attention |
| `lambda_aux` | λ | 0.4 | Deep supervision weight |
| `cluster_loss_alpha` | α | 0.1 | Compactness loss weight |
| `cluster_loss_beta` | β | 0.05 | Separation loss weight |

### Ablation Study Configurations

Create configs for each experiment:

**Baseline (T=1, no clustering)**:
```yaml
num_recursive_steps: 1
use_cluster_loss: false
```

**Full DSC-ViT**:
```yaml
num_recursive_steps: 3
use_cluster_loss: true
fusion_method: 'residual'
```

**Over-clustering (K=2×classes)**:
```yaml
num_clusters: 38  # 2 × 19 for Cityscapes
```

**Gated Fusion**:
```yaml
fusion_method: 'gated'
```

## Project Structure

```
segmentation/
├── models/
│   ├── __init__.py
│   ├── projections.py      # C→K→D, D→K projections + fusion
│   ├── soft_kmeans.py      # Differentiable K-Means clustering
│   ├── vit_encoder.py      # ViT encoder wrapper
│   └── dsc_vit.py          # Main DSC-ViT model
├── utils/
│   ├── __init__.py
│   ├── losses.py           # Deep supervision + cluster losses
│   └── metrics.py          # mIoU, pixel accuracy
├── configs/
│   └── default.yaml        # Default configuration
├── experiments/            # Training logs and checkpoints
├── data/                   # Dataset directory
├── train.py               # Training script
├── requirements.txt
├── CLAUDE.md             # Claude Code guidance
└── README.md
```

## Expected Results

### Performance Targets

| Metric | Baseline ViT (T=1) | DSC-ViT (T=3) | Target Improvement |
|--------|-------------------|---------------|-------------------|
| mIoU | ~60% | ~65-68% | +5-8% |
| Parameters | ~27M | ~7M | -74% |
| Inference Time | 1× | ~3× | T times slower |

### Ablation Studies

1. **Cluster Space vs Visual Space**
   - Hypothesis: Cluster space (C→K) outperforms visual space (K→C)
   - Metric: mIoU, training speed, memory usage

2. **K Optimization**
   - Test K = 0.5×, 1.0×, 2.0×, 4.0× num_classes
   - Metric: mIoU, cluster purity, interpretability

3. **Fusion Methods**
   - Residual vs Gated vs Attention
   - Metric: mIoU, parameter count, training time

4. **Recursive Steps**
   - T = 1, 2, 3, 4, 5
   - Metric: mIoU vs computational cost

## Visualization

To visualize cluster activations:

```python
from models import DSCViT
import torch
import matplotlib.pyplot as plt

# Load model
model = DSCViT(...)
model.eval()

# Forward pass
image = torch.randn(1, 3, 256, 256)
seg_preds, combined_clusters = model(image, return_intermediate=True)

# Visualize cluster activations for each step
for t in range(len(combined_clusters)):
    cluster_feat = combined_clusters[t][0]  # [K, H, W]

    # Plot each cluster channel
    fig, axes = plt.subplots(1, K, figsize=(20, 4))
    for k in range(K):
        axes[k].imshow(cluster_feat[k].cpu().detach(), cmap='hot')
        axes[k].set_title(f'Cluster {k}')
    plt.savefig(f'cluster_step_{t+1}.png')
```

## Citation

This work is inspired by:

```bibtex
@article{trm2025,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

## TODOs

- [ ] Implement dataset loaders (Cityscapes, PASCAL VOC, ADE20K)
- [ ] Add data augmentation pipeline (dihedral, color jitter)
- [ ] Implement temperature annealing for Soft K-Means
- [ ] Add visualization utilities (cluster maps, attention maps)
- [ ] Implement evaluation script
- [ ] Add TensorBoard/WandB logging
- [ ] Multi-GPU training support
- [ ] Mixed precision training (FP16)
- [ ] Cluster-class correspondence matrix visualization

## License

MIT License
