# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **DSC-ViT (Deep Supervised ViT with Clustering Hidden State)**, an image segmentation model that combines Vision Transformers with recursive reasoning and soft clustering mechanisms. The model is inspired by the Tiny Recursive Model (TRM) paper (arXiv:2510.04871v1) but adapted for computer vision tasks.

## Core Architecture Concepts

### Key Components

1. **Vision Transformer Encoder**: Pre-trained ViT (e.g., ViT-B/16) serves as the backbone
2. **Soft K-Means Clustering Layer**: Performs differentiable epistemological categorization in latent space
3. **Recursive Refinement Loop**: T-step iterative improvement similar to TRM's deep supervision
4. **Visual Projection Loop**: Projects clustered latent states back to visual space and combines with original image

### Model Flow (DSC-ViT) - OPTIMIZED ARCHITECTURE

**Key Innovations**:
1. Project original image to cluster space (C→K) instead of clusters to visual space (K→C)
2. **⭐ ViT computed ONCE** (not n×T times) - 18x speedup!

```
Input Image (H×W×C)
  ↓
  ⭐ ViT Encoding (ONCE): image → x_vit [B, D, H', W'] [cached]
  ⭐ Project to Cluster (ONCE): image → image_cluster [B, K, H', W'] [cached]
  ↓
  ┌─────────── Recursive Loop (T times) ───────────┐
  │  ┌── Latent Update (n times) ─────────────┐   │
  │  │                                         │   │
  │  │  Project to Cluster: (y+z) → D→K       │   │
  │  │  ↓                                      │   │
  │  │  Soft K-Means Clustering in K-space    │   │
  │  │  ↓                                      │   │
  │  │  Combine: z_clustered + image_cluster  │   │
  │  │  ↓                                      │   │
  │  │  Project back: K→D                      │   │
  │  │  ↓                                      │   │
  │  │  Add ViT residual: z = z + x_vit      │   │
  │  │                                         │   │
  │  └─────────────────────────────────────────┘   │
  │  ↓                                              │
  │  Answer Update: y = y + z                      │
  │  ↓                                              │
  │  Segmentation Prediction                       │
  │                                                 │
  └─────────────────────────────────────────────────┘
  ↓
Final Output (H×W×num_classes)
```

**Advantages**:
- **18x speedup**: ViT called once (not n×T=18 times) per step
- **33% fewer projections**: C→K→D vs K→C→K→D
- **Frozen ViT support**: Pre-trained features preserved, recursive learning in lightweight modules
- **Better information preservation**: Task-specific cluster space
- **Improved interpretability**: Cluster activations directly meaningful
- **True "tiny" recursion**: ~5M trainable parameters (projections + clustering)

### Key Architectural Decisions

**ViT Placement (critical optimization)**:
- **Naive approach**: Call ViT inside recursive loop → 18x slower, can't freeze
- **⭐ DSC-ViT**: ViT computed once, cached → lightweight modules iterate
- **Benefit**: Frozen ViT as feature extractor + fast recursive refinement

**Recursive Input Construction**:
- **TRM**: `Input = Feature + Latent` (direct combination)
- **DSC-ViT**:
  - `z_cluster = Proj_D→K(y + z)` (project to cluster space)
  - `z_new = Proj_K→D(clustering(z_cluster) + image_cluster)` (back to latent)
  - `z = z_new + x_vit` (add ViT residual)

**Latent State Refinement**:
- Uses differentiable Soft K-Means clustering instead of standard transformer operations
- **K clusters can differ from num_classes** (unsupervised pattern discovery)
- With `use_assignment_loss=False`: clusters discover patterns freely, segmentation head maps K→C
- Cluster centers μ^(t) are learned per recursive step

**Deep Supervision (CRITICAL - TRM's Core Mechanism)**:
- **NOT** multi-task learning across T steps!
- **IS** progressive refinement of same batch across N_sup steps
- Each batch processed up to N_sup=16 times with independent backward passes
- Structure: `for batch → for step in N_sup → [T recursive steps] → loss.backward() → optimizer.step() → z.detach()`
- Gradient isolation via detach() between supervision steps
- Early stopping via ACT (Adaptive Computational Time)

## Mathematical Formulation

### Soft K-Means Clustering
```
z_clustered^(t) = Σ(k=1 to K) q_{i,j,k}^(t) · μ_k^(t)
```
where q^(t) represents soft assignment probabilities

### Visual Projection Loop
```
1. z_visual^(t) = Projection_Visual(z_clustered^(t))  # D→C channels
2. x_combined = Original_Image + z_visual^(t)          # H×W×C
3. x'^(t+1) = Projection_D(x_combined)                # C→D channels
```

## Theoretical Motivation

The model integrates two cognitive capabilities:

1. **Epistemological Categorization**: Learning "what to differentiate from what" through soft clustering (edges, color groups, object boundaries)
2. **Recursive Refinement**: Iteratively improving reasoning by feeding refined representations back through the visual space

This differs from TRM's approach to puzzle-solving by focusing on visual understanding and segmentation-specific inductive biases.

## Comparison with TRM (Base Paper)

### TRM Key Insights
- Uses tiny 2-layer networks with only 7M parameters
- Achieves 45% on ARC-AGI-1, 8% on ARC-AGI-2 (beating most LLMs with <0.01% parameters)
- Core mechanism: recursive latent reasoning (z) + answer refinement (y)
- Deep supervision with N_sup=16 steps, T=3 recursion cycles, n=6 latent updates
- No fixed-point theorem needed (simpler than HRM)

### DSC-ViT Adaptations for Segmentation
- Replaces TRM's latent recursion with **Soft K-Means clustering**
- Adds **visual space projection loop** (latent→visual→combine→latent)
- Uses **pre-trained ViT** instead of training from scratch
- Applies recursive reasoning to **dense prediction** (per-pixel classification) instead of single-answer problems

## Implementation Guidelines

### Model Components to Implement

1. **Projection Layers**:
   - `Projection_D`: Maps visual features (C channels) to ViT latent space (D dimensions)
   - `Projection_Visual`: Maps clustered latent (D) back to visual space (C channels)
   - Typically use 1×1 convolutions for channel adjustment

2. **Soft K-Means Layer**:
   - Learnable cluster centers: `μ^(t)` shape `[K, D]`
   - Compute soft assignments via distance/similarity metric
   - Generate weighted average: `z_clustered = Σ q * μ`

3. **Recursive Loop**:
   - Run T iterations (e.g., T=3)
   - Each iteration: encode → cluster → project to visual → combine → project to latent
   - Maintain gradient flow through all T steps (no detaching except for input preparation)

4. **Segmentation Head**:
   - Simple MLP or 1×1 conv from final z^(T) to per-pixel class probabilities

### Training Considerations

- **Batch Size**: 32-64 (ADE20K), 768 from TRM paper (too large for most GPUs)
- **Hidden Size**: 512 (D dimension)
- **Learning Rate**: 5e-5 (ADE20K), 1e-4 (VOC) with warmup
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **Deep Supervision Steps**: N_sup = 8-16
- **Recursive Steps**: T = 3
- **Clustering**: K can differ from num_classes
  - **Fewer clusters (K < C)**: Coarser patterns, faster training
  - **Equal (K = C)**: Direct class-cluster correspondence
  - **More clusters (K > C)**: Fine-grained patterns, better boundaries
  - Recommended: K = C for initial baseline, then experiment
- **Loss Weighting**: Balance between segmentation loss and cluster loss
- **AMP**: Enable for 2-3x speedup (`use_amp: true`)

### Key Differences from Standard ViT Training

- Requires **multiple forward passes** (T times) per training step
- **Deep supervision** at each recursive step (not just final output)
- **Visual-latent projection loop** creates residual connections through image space
- Cluster centers evolve during training (learnable parameters updated per step)

## Expected Behavior

The model should learn to:
1. Categorize image regions into meaningful clusters (object boundaries, textures, colors)
2. Progressively refine segmentation through recursive iterations
3. Leverage original image information at each step (via visual projection loop)
4. Achieve better segmentation than single-pass ViT through iterative reasoning

## Research Context

This project explores whether TRM's recursive reasoning approach (successful on puzzle tasks like Sudoku, Maze, ARC-AGI) can transfer to dense computer vision tasks when combined with:
- Explicit clustering mechanisms (Soft K-Means)
- Visual-space feedback loops
- Pre-trained vision encoders

The hypothesis is that epistemological categorization + recursive visual refinement will improve segmentation quality, especially on complex scenes requiring hierarchical reasoning.

---

## Implementation Status

### ✅ Completed (2024-11-17)

#### 1. **Core Architecture**
- ✅ Vision Transformer encoder with timm integration
- ✅ Soft K-Means clustering layer (differentiable)
- ✅ Projection layers (C↔D↔K transformations)
- ✅ Recursive refinement loop (n=6 latent updates, T=3 recursive steps)
- ✅ Deep supervision training loop (N_sup=16 steps per batch)
- ✅ Segmentation head with upsampling
- ✅ **Spatial Context Encoding Module** (Optional)
  - `SimpleSpatialContext`: Lightweight dilated convolutions (dilation=4)
  - `ASPP`: DeepLab V3 style with optimized dilations [1,2,4,8] for 16×16
  - `ASPPAdaptive`: Auto-adjusts dilation rates based on feature map size
  - Prevents gridding artifacts at low resolutions
  - See `SPATIAL_CONTEXT.md` for detailed analysis

#### 2. **Critical Optimizations**
- ✅ **ViT Placement Optimization (18x speedup)**
  - Moved ViT encoding outside recursive loop
  - Reduced ViT calls from 288 to 16 per N_sup=16 training step
  - Enables frozen ViT as feature extractor
  - Verified with `benchmark_vit_calls.py`

- ✅ **ACT (Adaptive Computational Time) - TRM Paper Implementation**
  - Added BCE loss to train q_head for correctness prediction
  - Changed threshold from arbitrary value to 0 (TRM specification)
  - Enables adaptive early stopping per batch
  - Target: `(y_pred == y_true)` per-pixel accuracy

#### 3. **Dataset Integration**
- ✅ PASCAL VOC 2012 Segmentation (via Kaggle)
  - 21 classes (20 objects + background)
  - 1,464 train images, 1,449 validation images
  - Custom `VOCSegmentationKaggle` loader in `data/voc_kaggle.py`
  - Augmentation pipeline (flip, crop, color jitter, dihedral)

- ✅ **ADE20K Scene Parsing Dataset**
  - 150 classes (complex scene parsing benchmark)
  - 20,210 train images, 2,000 validation images
  - Custom `ADE20KSegmentation` loader in `data/ade20k.py`
  - Automated download script: `download_ade20k.py`
  - Config: `configs/ade20k.yaml` with optimized hyperparameters

#### 4. **Training Infrastructure**
- ✅ Deep Supervision training loop (TRM-style)
  - N_sup=16 supervision steps per batch
  - Gradient isolation via detach() between steps
  - ACT early stopping with learned q_head
- ✅ **AMP (Automatic Mixed Precision) Training**
  - torch.cuda.amp integration for 2-3x speedup
  - ~40% memory reduction
  - Configurable via `use_amp` flag
- ✅ EMA (Exponential Moving Average) model
- ✅ Cluster loss (compactness + separation)
- ✅ Segmentation metrics (mIoU, pixel accuracy, per-class IoU)
- ✅ Checkpoint saving (best model + periodic)
- ✅ **PCA-based Cluster Visualization**
  - For K>50: PCA reduces K-dimensional cluster centers to RGB
  - Preserves semantic similarity in color space
  - Prevents rainbow gradient artifacts

#### 5. **Testing & Validation**
- ✅ `test_training.py` - Synthetic data testing (no dataset download needed)
- ✅ `benchmark_vit_calls.py` - Verify ViT optimization (18x reduction confirmed)
- ✅ `test_cluster_mismatch.py` - Verify num_clusters ≠ num_classes compatibility
- ✅ `check_predictions.py` - Analyze prediction distribution to diagnose issues
- ✅ `analyze_model_params.py` - Detailed parameter breakdown by module
- ✅ Module tests for all components (projections, soft k-means, encoder, losses)

#### 6. **Documentation**
- ✅ CLAUDE.md - Project overview and architecture details
- ✅ README.md - Setup instructions and usage examples
- ✅ TRM_ARCHITECTURE.md - TRM paper analysis
- ✅ **SPATIAL_CONTEXT.md** - Spatial context encoding guide
  - Dilation rate analysis for 16×16 resolution
  - SimpleSpatialContext vs ASPP vs ASPPAdaptive comparison
  - Why original DeepLab dilations don't work at low resolutions
- ✅ .gitignore - Proper data/cache exclusions

### 📊 Performance Metrics

**ViT Optimization:**
- Old: 288 ViT calls (n×T×N_sup = 6×3×16)
- New: 16 ViT calls (1×N_sup = 1×16)
- **Speedup: 18.0x fewer ViT forward passes**

**Model Size (ADE20K, K=8):**
- Total parameters: ~87.08M (86.24M ViT + 0.84M task-specific)
- Trainable (frozen ViT): ~1.24M (1.42% of total)
  - Answer Network: 0.53M (42.6% of trainable) - recursive reasoning
  - Segmentation Head: 0.17M (13.8% of trainable)
  - Q-head (ACT): 0.13M (10.7% of trainable)
  - Projections: 0.01M (1.0% of trainable)
  - Clustering: 64 params
  - Fusion: 288 params
- With Spatial Context: +1.5M (SimpleSpatialContext) or +3-4M (ASPP)
- Memory efficient: ~0.17 GB training (AMP, excluding activations/gradients)

### 🚀 Ready to Train

The implementation is complete and ready for full-scale training:

```bash
# Quick test with synthetic data (no download)
python test_training.py

# Full training on PASCAL VOC 2012 (21 classes)
python train.py --config configs/default.yaml

# Full training on ADE20K (150 classes)
python train.py --config configs/ade20k.yaml

# With spatial context encoding
# Edit configs/ade20k.yaml:
#   use_spatial_context: true
#   spatial_context_type: 'simple'  # or 'aspp'

# Debug mode (small subset)
python train.py --config configs/ade20k.yaml --debug

# Analyze model parameters
python analyze_model_params.py

# Check prediction distribution (requires trained model)
python check_predictions.py
```

### 🔬 Next Steps (Research)

1. **Critical Issues to Address**
   - **Low validation mIoU (~8%)** on ADE20K with K=8
     - Hypothesis: Information bottleneck (150 classes → 8 clusters)
     - Solution: Increase num_clusters to 150+ or use K=num_classes
   - **EMA validation lag**: Disabled for now, may re-enable with lower decay
   - **uv CUDA compatibility**: Fixed by reinstalling torch with cu128 flag

2. **Hyperparameter Tuning**
   - **num_clusters sensitivity**: Test K ∈ {8, 16, 32, 64, 150, 300}
   - ACT threshold (currently 0.0 per TRM paper)
   - Optimal N_sup (deep supervision steps)
   - Cluster loss weighting (α=1.0, β=0.5)
   - Learning rate for 150-class dataset

3. **Ablation Studies**
   - ViT frozen vs fine-tuned
   - With/without spatial context (SimpleSpatialContext vs ASPP)
   - With/without cluster loss
   - Different fusion methods (residual vs gated vs attention)
   - num_clusters vs num_classes impact on mIoU

4. **Scaling Experiments**
   - Different ViT backbones (ViT-Tiny, ViT-Small for speed)
   - Multi-scale inference
   - Test on Cityscapes (19 classes)

5. **Analysis & Visualization**
   - Per-class IoU distribution (identify which classes fail)
   - Cluster center evolution during training
   - ACT stopping distribution across batches
   - Per-step segmentation quality improvement
   - Prediction entropy analysis
