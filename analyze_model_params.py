"""
Analyze DSC-ViT model parameters by module.

Usage:
    python analyze_model_params.py
"""

import torch
import yaml
from models import DSCViT


def count_parameters(model, module_name=""):
    """Count parameters in a module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_model_structure(config_path='configs/ade20k.yaml'):
    """Analyze model structure and parameter counts."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle num_clusters
    num_clusters = config['model'].get('num_clusters') or config['model']['num_classes']

    # Create model
    model = DSCViT(
        image_channels=config['model']['image_channels'],
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        num_clusters=num_clusters,
        clustering_method=config['model'].get('clustering_method', 'soft'),
        latent_dim=config['model']['latent_dim'],
        vit_model_name=config['model']['vit_model_name'],
        use_pretrained_vit=config['model']['use_pretrained_vit'],
        use_simple_encoder=config['model']['use_simple_encoder'],
        num_latent_updates=config['model']['num_latent_updates'],
        num_recursive_steps=config['model']['num_recursive_steps'],
        fusion_method=config['model']['fusion_method'],
        use_spatial_context=config['model'].get('use_spatial_context', False),
        spatial_context_type=config['model'].get('spatial_context_type', 'simple'),
        freeze_vit=config['model']['freeze_vit'],
        use_deep_supervision=True
    )

    print("="*80)
    print("DSC-ViT Model Parameter Analysis")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Image size: {config['model']['img_size']}")
    print(f"  - Num classes: {config['model']['num_classes']}")
    print(f"  - Num clusters: {num_clusters}")
    print(f"  - Latent dim: {config['model']['latent_dim']}")
    print(f"  - ViT model: {config['model']['vit_model_name']}")
    print(f"  - Freeze ViT: {config['model']['freeze_vit']}")
    print(f"  - Fusion method: {config['model']['fusion_method']}")
    print(f"  - Spatial context: {config['model'].get('use_spatial_context', False)}")

    print("\n" + "="*80)
    print("Parameter Breakdown by Module")
    print("="*80)

    modules = []

    # 1. ViT Encoder
    total, trainable = count_parameters(model.encoder)
    modules.append(("1. ViT Encoder (vit_base_patch16_224)", total, trainable))
    print(f"\n{'1. ViT Encoder (vit_base_patch16_224)':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")
    print(f"   Trainable:  {trainable:>12,} ({trainable/1e6:>6.2f}M)")
    print(f"   Frozen:     {total-trainable:>12,} ({(total-trainable)/1e6:>6.2f}M)")

    # 2. Projection Layers
    total, trainable = count_parameters(model.projections)
    modules.append(("2. Projection Layers (C→K, K→D, D→K)", total, trainable))
    print(f"\n{'2. Projection Layers':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # Detailed projection breakdown
    if hasattr(model.projections, 'proj_image_to_cluster'):
        proj_params = sum(p.numel() for p in model.projections.proj_image_to_cluster.parameters())
        print(f"     - image_to_cluster (C→K):  {proj_params:>10,} ({proj_params/1e6:.2f}M)")

    if hasattr(model.projections, 'proj_cluster_to_latent'):
        proj_params = sum(p.numel() for p in model.projections.proj_cluster_to_latent.parameters())
        print(f"     - cluster_to_latent (K→D): {proj_params:>10,} ({proj_params/1e6:.2f}M)")

    if hasattr(model.projections, 'proj_latent_to_cluster'):
        proj_params = sum(p.numel() for p in model.projections.proj_latent_to_cluster.parameters())
        print(f"     - latent_to_cluster (D→K): {proj_params:>10,} ({proj_params/1e6:.2f}M)")

    if hasattr(model.projections, 'proj_initial'):
        proj_params = sum(p.numel() for p in model.projections.proj_initial.parameters())
        print(f"     - initial (C→D):           {proj_params:>10,} ({proj_params/1e6:.2f}M)")

    # 3. Clustering Layer
    total, trainable = count_parameters(model.clustering_layer)
    modules.append(("3. Clustering Layer (Soft K-Means)", total, trainable))
    print(f"\n{'3. Clustering Layer (Soft K-Means)':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")
    print(f"   Cluster centers: [{model.K}, {model.K}]")

    # 4. Spatial Context (if enabled)
    if model.use_spatial_context:
        total, trainable = count_parameters(model.spatial_context_encoder)
        modules.append(("4. Spatial Context Encoder", total, trainable))
        print(f"\n{'4. Spatial Context Encoder':<50}")
        print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # 5. Fusion
    if model.fusion is not None:
        total, trainable = count_parameters(model.fusion)
        modules.append(("5. Fusion Module (Attention)", total, trainable))
        print(f"\n{'5. Fusion Module (Attention)':<50}")
        print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # 6. Segmentation Head
    total, trainable = count_parameters(model.seg_head)
    modules.append(("6. Segmentation Head", total, trainable))
    print(f"\n{'6. Segmentation Head':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # 7. Q-head (ACT)
    total, trainable = count_parameters(model.q_head)
    modules.append(("7. Q-head (ACT)", total, trainable))
    print(f"\n{'7. Q-head (Adaptive Computational Time)':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # 8. Answer Network
    total, trainable = count_parameters(model.answer_network)
    modules.append(("8. Answer Network (y update)", total, trainable))
    print(f"\n{'8. Answer Network (y update, TRM-style)':<50}")
    print(f"   Total:      {total:>12,} ({total/1e6:>6.2f}M)")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_all = total_all - trainable_all

    print(f"\nTotal Parameters:      {total_all:>12,} ({total_all/1e6:>6.2f}M)")
    print(f"Trainable Parameters:  {trainable_all:>12,} ({trainable_all/1e6:>6.2f}M)")
    print(f"Frozen Parameters:     {frozen_all:>12,} ({frozen_all/1e6:>6.2f}M)")
    print(f"\nTrainable Ratio:       {trainable_all/total_all*100:>6.2f}%")

    # Breakdown table
    print("\n" + "="*80)
    print("Module-wise Breakdown")
    print("="*80)
    print(f"{'Module':<50} {'Total':>12} {'Trainable':>12} {'%':>6}")
    print("-"*80)

    for name, total, trainable in modules:
        pct = (trainable / trainable_all * 100) if trainable_all > 0 else 0
        print(f"{name:<50} {total:>12,} {trainable:>12,} {pct:>5.1f}%")

    print("="*80)

    # Memory estimation
    print("\n" + "="*80)
    print("Memory Estimation")
    print("="*80)

    # FP32: 4 bytes per parameter
    # FP16 (AMP): 2 bytes per parameter + 4 bytes for optimizer states
    mem_fp32 = total_all * 4 / 1024**3  # GB
    mem_fp16_model = total_all * 2 / 1024**3  # GB
    mem_fp16_opt = trainable_all * 8 / 1024**3  # GB (Adam: 2 states * 4 bytes)
    mem_fp16_total = mem_fp16_model + mem_fp16_opt

    print(f"\nModel Weights (FP32):        {mem_fp32:>6.2f} GB")
    print(f"Model Weights (FP16/AMP):    {mem_fp16_model:>6.2f} GB")
    print(f"Optimizer States (FP32):     {mem_fp16_opt:>6.2f} GB")
    print(f"Total Training Memory (AMP): {mem_fp16_total:>6.2f} GB")
    print(f"\nNote: Does not include activations, gradients, or batch data")

    return model


if __name__ == '__main__':
    model = analyze_model_structure()
