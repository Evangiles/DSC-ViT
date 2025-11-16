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

### Model Flow (DSC-ViT) - REVISED ARCHITECTURE

**Key Innovation**: Project original image to cluster space (C→K) instead of clusters to visual space (K→C).

```
Input Image (H×W×C)
  → Project to Cluster Space (C→K) [computed once, cached]
  → Initialize ViT Input (C→D)
  ↓
  ┌─────────── Recursive Loop (T times) ───────────┐
  │                                                 │
  │  ViT Encoding → Latent State (D)               │
  │  ↓                                              │
  │  Project to Cluster Space (D→K)                │
  │  ↓                                              │
  │  Soft K-Means Clustering in K-space            │
  │  ↓                                              │
  │  Combine with Image Cluster Features (K+K→K)   │
  │  ↓                                              │
  │  Project back to Latent Space (K→D)            │
  │  ↓                                              │
  │  Segmentation Prediction                       │
  │                                                 │
  └─────────────────────────────────────────────────┘
  ↓
Final Output (H×W×num_classes)
```

**Advantages**:
- 33% fewer projection operations (C→K→D vs K→C→K→D)
- Better information preservation (task-specific cluster space)
- Improved interpretability (cluster activations directly meaningful)
- Unified computation in cluster space

### Key Architectural Decisions

**Recursive Input Construction (differs from TRM)**:
- **TRM**: `Input_D = Feature_D + Latent_D` (direct latent space combination)
- **DSC-ViT**: `Input_D = Projection_D(Feature_Visual + Projection_Visual(Latent_D))` (visual space loop)

**Latent State Refinement**:
- Uses differentiable Soft K-Means clustering instead of standard transformer operations
- K clusters = number of segmentation classes
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

- **Batch Size**: 768 (from TRM paper)
- **Hidden Size**: 512 (D dimension)
- **Learning Rate**: 1e-4 with 2K warmup iterations
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **Deep Supervision Steps**: N_sup = 16
- **Recursive Steps**: T = 3
- **Clustering**: K = number of segmentation classes
- **Loss Weighting**: Balance λ between main CE loss and auxiliary categorization losses

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
