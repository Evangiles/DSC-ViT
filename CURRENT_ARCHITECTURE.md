# DSC-ViT: Current Architecture & Data Flow

**Last Updated:** 2024-11-16
**Model Version:** TRM-Aligned (with clustering)

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Model Components](#model-components)
3. [Data Flow](#data-flow)
4. [Forward Pass Details](#forward-pass-details)
5. [Training Loop](#training-loop)
6. [Loss Computation](#loss-computation)
7. [Key Differences from TRM](#key-differences-from-trm)

---

## Architecture Overview

### High-Level Structure

```
Input Image [B, 3, 256, 256]
    ↓
    ⭐ ViT Encoding (ONCE): → x_vit [B, D, 16, 16] (cached)
    ⭐ Image to Cluster (ONCE): → image_cluster [B, K, 16, 16] (cached)
    ↓
┌─────────── Deep Supervision Loop (N_sup=16 times) ───────────┐
│   ┌── Deep Recursion (T-1=2 times, no grad) ──────────┐     │
│   │   ┌── Latent Recursion (n=6, T=3) ────────────┐   │     │
│   │   │   ┌── z update (n=6 times) ────────┐       │   │     │
│   │   │   │  1. Project z to cluster: D→K  │       │   │     │
│   │   │   │  2. Soft K-Means clustering    │       │   │     │
│   │   │   │  3. Project back: K→D          │       │   │     │
│   │   │   │  4. Combine: z+x_vit+y         │       │   │     │
│   │   │   │  (Full gradient through all 6) │       │   │     │
│   │   │   └────────────────────────────────┘       │   │     │
│   │   │   ┌── y update (once) ─────────────┐       │   │     │
│   │   │   │  y = answer_network(y + z)     │       │   │     │
│   │   │   └────────────────────────────────┘       │   │     │
│   │   └─────────────────────────────────────────────┘   │     │
│   └─────────────────────────────────────────────────────┘     │
│   ┌── Deep Recursion (1 time, WITH grad) ──────────────┐     │
│   │   (Same as above, but gradients flow)              │     │
│   └─────────────────────────────────────────────────────┘     │
│   ↓                                                            │
│   Segmentation Head: y → seg_pred [B, 21, 256, 256]          │
│   Q-Head: y → q_logit [B, 1, 256, 256]                       │
│   ↓                                                            │
│   Loss.backward() → optimizer.step()                          │
│   ↓                                                            │
│   y, z = y.detach(), z.detach() (gradient isolation)         │
└────────────────────────────────────────────────────────────────┘
    ↓
Final Output [B, 21, 256, 256]
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D` | 512 | Latent dimension (ViT feature dim) |
| `K` | 21 | Number of clusters (= num_classes) |
| `C` | 3 | Image channels (RGB) |
| `H, W` | 256, 256 | Image resolution |
| `H', W'` | 16, 16 | ViT output resolution (patch_size=16) |
| `n` | 6 | Number of z updates per recursion |
| `T` | 3 | Number of recursive steps |
| `N_sup` | 16 | Deep supervision steps |

---

## Model Components

### 1. **Vision Transformer Encoder** (`self.encoder`)

**Input:** `[B, 3, 256, 256]` (RGB image)
**Output:** `[B, 512, 16, 16]` (ViT features)

```python
self.encoder = ViTEncoder(
    model_name='vit_base_patch16_224',
    pretrained=True,
    output_dim=512,
    freeze_encoder=True,  # ⭐ Frozen (not trainable)
    img_size=256
)
```

**Key Properties:**
- Pre-trained on ImageNet
- Frozen during training (feature extractor only)
- Called **ONCE** per batch (18x speedup)
- Patch size: 16 → output resolution: 256/16 = 16

---

### 2. **Projection Layers** (`self.projections`)

Three projection modules for space transformations:

```python
self.projections = ProjectionLayers(
    image_channels=3,
    num_clusters=21,
    latent_dim=512
)
```

**2.1. Image → Cluster** (`image_to_cluster`)
- Input: `[B, 3, H, W]`
- Output: `[B, K, H', W']`
- Used: Computed ONCE and cached

**2.2. Latent → Cluster** (`latent_to_cluster`)
- Input: `[B, D, H', W']`
- Output: `[B, K, H', W']`
- Used: In z update loop (n times per recursion)

**2.3. Cluster → Latent** (`cluster_to_latent`)
- Input: `[B, K, H', W']`
- Output: `[B, D, H', W']`
- Used: After clustering in z update

---

### 3. **Soft K-Means Clustering** (`self.clustering_layer`)

**Input:** `[B, K, H', W']` (features in cluster space)
**Output:** `[B, K, H', W']` (clustered features)

```python
self.clustering_layer = SoftKMeansLayer(
    num_clusters=21,
    feature_dim=21,
    temperature=1.0,
    init_method='orthogonal'
)
```

**Algorithm:**
```python
# 1. Compute distances to cluster centers
distances = ||x - μ_k||²  # [B, H'W', K]

# 2. Soft assignment (temperature-scaled softmax)
q = softmax(-distances / temperature)  # [B, H'W', K]

# 3. Weighted sum
z_clustered = Σ(q_k * μ_k)  # [B, H'W', K]
```

**Learnable Parameters:**
- Cluster centers: `μ` shape `[K, K]`

---

### 4. **Answer Network** (`self.answer_network`)

**Input:** `[B, D, H', W']` (y + z combined)
**Output:** `[B, D, H', W']` (updated y)

```python
self.answer_network = nn.Sequential(
    nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
    nn.GroupNorm(32, latent_dim),  # ✅ Normalization added
    nn.ReLU(inplace=True),
    nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
    nn.GroupNorm(32, latent_dim)   # ✅ Normalization added
)
```

**Purpose:** Implements TRM's `y = net(y, z)` update

**Features:**
- ✅ GroupNorm for stability (32 groups)
- ⚠️ No residual connection (intentional for TRM alignment)
- ✅ Proper normalization between layers

---

### 5. **Segmentation Head** (`self.seg_head`)

**Input:** `[B, D, H', W']` (answer latent y)
**Output:** `[B, 21, H', W']` → upsampled to `[B, 21, 256, 256]`

```python
self.seg_head = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 21, kernel_size=1)
)
```

---

### 6. **Q-Head (ACT)** (`self.q_head`)

**Input:** `[B, D, H', W']` (answer latent y)
**Output:** `[B, 1, H', W']` (halt logits)

```python
self.q_head = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 1, kernel_size=1)
)
```

**Purpose:** Predict when to stop deep supervision
**Threshold:** 0.0 (TRM paper)

---

## Data Flow

### Initialization (Once per batch)

```python
# Step 1: Encode image with ViT (ONCE!)
x_vit = encoder(images)  # [B, 3, 256, 256] → [B, 512, 16, 16]

# Step 2: Project image to cluster space (ONCE!)
image_cluster = projections.image_to_cluster(images)  # [B, 3, 256, 256] → [B, 21, 16, 16]

# Step 3: Initialize latent states
y = torch.zeros(B, D, H', W')  # Answer latent
z = torch.zeros(B, D, H', W')  # Reasoning latent
```

---

### Deep Supervision Loop (N_sup = 16 iterations)

```python
for step in range(N_sup):
    # Deep recursion with gradient control
    (y, z), seg_pred, q_logit, _, _ = model(images, y, z)

    # Compute loss
    loss = criterion(seg_pred, targets, q_logit, ...)

    # Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Gradient isolation
    y = y.detach()
    z = z.detach()

    # ACT: Early stopping
    if q_logit.mean() > 0.0:
        break
```

---

### Deep Recursion (T = 3 steps)

```python
def deep_recursion(x_vit, image_cluster, y, z):
    # T-1 iterations without gradient
    with torch.no_grad():
        for t in range(T - 1):  # 2 times
            y, z, _ = latent_recursion(x_vit, image_cluster, y, z)

    # 1 iteration WITH gradient
    y, z, combined_cluster = latent_recursion(x_vit, image_cluster, y, z)

    # Predictions
    seg_pred = upsample(seg_head(y))  # [B, 21, 256, 256]
    q_logit = upsample(q_head(y))     # [B, 1, 256, 256]

    return (y.detach(), z.detach()), seg_pred, q_logit, ...
```

**Gradient Strategy:**
- T-1 iterations: `no_grad()` (fast forward)
- Last iteration: Full gradient through n=6 steps

---

### Latent Recursion (n = 6 updates)

```python
def latent_recursion(x_vit, image_cluster, y, z):
    # Update z for n iterations
    for i in range(n):  # 6 times
        z, _ = update_reasoning_latent(x_vit, image_cluster, y, z)
        # ⭐ NO DETACH! Full gradient through all 6 steps

    # Update y once
    y = update_answer_latent(y, z)

    return y, z, combined_cluster
```

---

### Z Update (Reasoning Latent)

```python
def update_reasoning_latent(x_vit, image_cluster, y, z):
    # Step 1: Cluster z only (design philosophy)
    z_cluster = projections.latent_to_cluster(z)  # [B, D, 16, 16] → [B, K, 16, 16]

    # Step 2: Soft K-Means clustering
    z_clustered, _, _ = clustering_layer(z_cluster)  # [B, K, 16, 16]

    # Step 3: Project back to latent space
    z_latent = projections.cluster_to_latent(z_clustered)  # [B, K, 16, 16] → [B, D, 16, 16]

    # Step 4: Combine with x_vit and y (TRM philosophy)
    z_new = z_latent + x_vit + y  # Element-wise addition

    return z_new, z_clustered
```

**TRM Alignment:**
- ✅ Uses x, y, z (all three inputs)
- ⚠️ Simple addition (not network)
- ⚠️ Clustering on z only (intentional difference)

---

### Y Update (Answer Latent)

```python
def update_answer_latent(y, z):
    # Combine y and z
    combined = y + z  # [B, D, 16, 16]

    # Pass through network (TRM requirement)
    y_new = answer_network(combined)  # [B, D, 16, 16]

    return y_new
```

**TRM Alignment:**
- ✅ Uses network (not simple addition)
- ⚠️ No residual connection
- ⚠️ Separate network (TRM uses same network as z update)

---

## Forward Pass Details

### Complete Forward Pass

**Input:**
- `images`: `[B, 3, 256, 256]`
- `y_prev`: `[B, D, 16, 16]` or `None`
- `z_prev`: `[B, D, 16, 16]` or `None`

**Output:**
- `(y, z)`: Updated latent states `[B, D, 16, 16]`
- `seg_pred`: Segmentation prediction `[B, 21, 256, 256]`
- `q_logit`: Halt logit `[B, 1, 256, 256]`
- `combined_cluster`: Clustered features `[B, K, 16, 16]`
- `image_cluster`: Image in cluster space `[B, K, 16, 16]`

### Shape Transformations

```
images           [B, 3, 256, 256]
   ↓ ViT
x_vit            [B, 512, 16, 16]
   ↓ image_to_cluster
image_cluster    [B, 21, 16, 16]
   ↓ initialize
y, z             [B, 512, 16, 16]

# Z Update Loop (n=6)
z                [B, 512, 16, 16]
   ↓ latent_to_cluster
z_cluster        [B, 21, 16, 16]
   ↓ clustering
z_clustered      [B, 21, 16, 16]
   ↓ cluster_to_latent
z_latent         [B, 512, 16, 16]
   ↓ + x_vit + y
z_new            [B, 512, 16, 16]

# Y Update
y + z            [B, 512, 16, 16]
   ↓ answer_network
y_new            [B, 512, 16, 16]

# Segmentation
y                [B, 512, 16, 16]
   ↓ seg_head
seg_pred_small   [B, 21, 16, 16]
   ↓ upsample
seg_pred         [B, 21, 256, 256]
```

---

## Training Loop

### Single Epoch

```python
for batch_idx, (images, targets) in enumerate(train_loader):
    # 1. Initialize latent states
    y, z = None, None

    # 2. Deep supervision loop
    for step in range(N_sup):  # 16 iterations
        # 2.1. Forward pass
        (y, z), seg_pred, q_logit, _, _ = model(images, y, z)

        # 2.2. Compute loss
        loss, loss_dict = criterion(
            seg_pred, targets,
            q_logit=q_logit,
            combined_cluster=...,
            z_clustered=...
        )

        # 2.3. Backward
        loss.backward()

        # 2.4. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 2.5. Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # 2.6. Detach for next step (gradient isolation)
        y = y.detach()
        z = z.detach()

        # 2.7. ACT early stopping
        q_mean = torch.sigmoid(q_logit).mean()
        if q_mean > 0.0:
            break

    # 3. Update EMA model
    if use_ema:
        ema_model.update(model)
```

### Key Training Details

| Setting | Value | Description |
|---------|-------|-------------|
| Optimizer | AdamW | β1=0.9, β2=0.95 |
| Learning Rate | 1e-4 | ⚠️ May be too high (causing divergence) |
| Weight Decay | 0.1 | |
| Gradient Clip | 1.0 | ⚠️ May need 0.5 |
| Warmup Epochs | 5 | Linear warmup |
| Scheduler | Cosine | min_lr=1e-6 |
| Batch Size | 32 | (default config) |
| EMA Decay | 0.999 | |

---

## Loss Computation

### Total Loss Components

```python
total_loss = main_loss + λ_aux * aux_loss + cluster_loss
```

### 1. **Main Loss** (Segmentation)

```python
# Focal Loss (if enabled)
focal_loss = -α * (1 - p_t)^γ * log(p_t)

# BCE Loss (if enabled)
bce_loss = BCE(pred, target_one_hot)

# Combined
main_loss = focal_loss + β * bce_loss
```

**Current Config:**
- `use_focal_loss: true`
- `focal_alpha: 0.25`
- `focal_gamma: 2.0`
- `use_bce: true`
- `bce_weight: 0.5`

### 2. **Auxiliary Loss** (Deep Supervision)

```python
# Weighted sum across N_sup steps
aux_loss = Σ(weight_i * loss_i)
```

**Weight:** `λ_aux = 0.4`

### 3. **Cluster Loss** (Compactness + Separation)

```python
# Compactness: pixels close to cluster centers
compactness = Σ||x - μ_assigned||²

# Separation: cluster centers far apart
separation = -Σ(i≠j) ||μ_i - μ_j||²

# Combined
cluster_loss = α * compactness + β * separation
```

**Current Config:**
- `use_cluster_loss: true`
- `cluster_loss_alpha: 1.0`
- `cluster_loss_beta: 0.5`

### 4. **ACT Loss** (Halt Prediction)

```python
# Target: 1 if prediction correct, 0 otherwise
target_halt = (seg_pred.argmax(1) == targets).float()

# BCE with logits
act_loss = BCE_with_logits(q_logit, target_halt)
```

---

## Key Differences from TRM

### ✅ Aligned with TRM

| Feature | TRM | DSC-ViT | Status |
|---------|-----|---------|--------|
| z uses x, y, z | ✅ | ✅ | Aligned |
| y uses network | ✅ | ✅ | Aligned |
| Full n-step gradient | ✅ | ✅ | Aligned |
| ACT threshold = 0 | ✅ | ✅ | Aligned |
| Deep supervision | ✅ | ✅ | Aligned |
| Gradient isolation | ✅ | ✅ | Aligned |

### ⚠️ Intentional Differences

| Feature | TRM | DSC-ViT | Reason |
|---------|-----|---------|--------|
| Clustering | ❌ | ✅ | Core DSC-ViT innovation |
| Single network | ✅ | ❌ | Clustering requires separate functions |
| x as input | ✅ | Residual | ViT optimization (18x speedup) |

### ❌ Potential Issues

| Issue | Impact | Priority |
|-------|--------|----------|
| ~~No normalization in answer_network~~ | ~~Instability~~ | ✅ Fixed |
| Simple addition for x+y+z | Limited expressiveness | ⚠️ Medium |
| No residual in y update | Intentional (TRM alignment) | ℹ️ Design choice |
| High learning rate | Gradient explosion | 🔥 High |

---

## Current Issues & Observations

### ✅ Resolved Issues

1. **~~No Normalization in Answer Network~~** (Fixed)
   - Solution: Added GroupNorm(32) between Conv2d layers
   - Impact: Improved training stability

### 🔥 Remaining Critical Issues

1. **Gradient Explosion** (If occurs during training)
   - Cause: Full gradient through n=6 steps
   - Solution: Lower LR or stronger grad clip

2. **Model Divergence** (If occurs during training)
   - Cause: Potentially unstable y ↔ z feedback loop
   - Solution: Monitor training, adjust hyperparameters

### ⚠️ Potential Improvements

1. **Better Fusion for x+y+z**
   - Current: Element-wise addition
   - Proposed: Learnable MLP or gated fusion
   - Trade-off: Simplicity vs expressiveness

2. **Multi-scale Features**
   - Current: Only 16×16
   - Proposed: 8×8, 16×16, 32×32
   - Trade-off: Memory vs performance

3. **Residual Connections in Y Update**
   - Current: `y = net(y+z)` (TRM-aligned)
   - Alternative: `y = y + net(y+z)`
   - Note: Current design follows TRM paper

---

## Model Size

### Parameter Count

```
Total:        ~86M parameters
  - ViT:      ~85M (frozen, not trainable)
  - Trainable: ~5M

Trainable breakdown:
  - Projections:        ~2M
  - Clustering:         <1M
  - Answer Network:     ~500K
  - Seg Head:           ~1M
  - Q Head:             ~500K
```

### Memory Usage (batch_size=32)

```
Forward pass:  ~8 GB
Backward pass: ~12 GB
Total:         ~20 GB (single GPU)
```

---

## Conclusion

**Current Status:** Model architecture is TRM-aligned with stability improvements (GroupNorm added).

**Next Steps:**
1. 🚀 **Immediate:** Run full-scale training on PASCAL VOC 2012
2. 📊 **Short-term:** Monitor training stability and adjust hyperparameters
3. ⚠️ **Medium-term:** Ablation studies (fusion methods, cluster loss weights)
4. 🔬 **Long-term:** Scale up (larger datasets, multi-scale features)

**Key Insight:** The clustering-based approach is conceptually sound. With GroupNorm added to the answer network, the model should have improved stability while maintaining TRM's recursive reasoning and the clustering innovation.

**Architecture Status:**
- ✅ ViT optimization (18x speedup)
- ✅ TRM-aligned recursion (n=6, T=3)
- ✅ Answer network normalization
- ✅ Deep supervision + ACT
- ⚠️ Awaiting full training results
