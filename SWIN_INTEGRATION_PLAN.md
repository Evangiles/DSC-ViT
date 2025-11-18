# Swin-DSC-ViT Integration Plan

## Architecture Overview

```
Input Image (256×256)
    ↓
Swin Encoder (Multi-Scale)
    ├─ Stage 0: 64×64, 96 channels  → Project → 768 channels
    ├─ Stage 1: 32×32, 192 channels → Project → 768 channels  
    ├─ Stage 2: 16×16, 384 channels → Project → 768 channels
    └─ Stage 3: 8×8,   768 channels → Project → 768 channels
    ↓
FPN Fusion (Top-Down)
    ↓
Fused Features (64×64, 768 channels)
    ↓
**DSC-ViT Recursive Clustering** (Same as before)
    ├─ Latent Update (n times)
    │   ├─ Project to Cluster Space (D → K)
    │   ├─ Soft K-Means Clustering
    │   └─ Project back (K → D)
    └─ Answer Update (y = y + z)
    ↓
Segmentation Head
    ↓
Output (256×256, num_classes)
```

## Key Design Decisions

### 1. **Where to Apply Multi-Scale Fusion?**
**Decision**: Use FPN to fuse multi-scale features BEFORE recursive clustering

**Rationale**:
- Recursive clustering operates on fixed-dimension latent space
- Multi-scale fusion provides rich initial features
- Keeps recursive logic simple and focused

### 2. **What Resolution for Clustering?**
**Options**:
- A) 64×64 (highest resolution from Swin Stage 0)
- B) 16×16 (original DSC-ViT resolution)  
- C) 8×8 (lowest resolution from Swin Stage 3)

**Decision**: 16×16 (Option B)

**Rationale**:
- Balance between detail and computational cost
- Compatible with existing DSC-ViT architecture
- 16×16 has good semantic information from Swin Stage 2

### 3. **Spatial Context + Swin?**
**Decision**: Start WITHOUT spatial context, add later if needed

**Rationale**:
- Swin already provides multi-scale receptive fields
- Avoid over-complication
- Test Swin benefit first, then add spatial context

## Implementation Steps

1. ✓ Create SwinEncoder (multi-scale extraction)
2. ✓ Create FPNDecoder (multi-scale fusion)
3. ⏳ Create SwinDSCViT (Swin + Recursive Clustering)
   - Use Swin encoder instead of ViT
   - FPN fusion to 16×16 resolution
   - Apply existing recursive clustering logic
   - Use FPN final upsampling to 256×256
4. ⏳ Create config file (configs/swin.yaml)
5. ⏳ Test end-to-end

## Expected Benefits

- ✅ **Better boundaries**: High-res features (64×64) capture fine details
- ✅ **Better semantics**: Deep features (8×8) provide global context
- ✅ **Better small objects**: Multi-scale helps detect objects at all sizes
- ✅ **Better mIoU**: Expected +3~7% improvement over single-scale ViT

## Next Steps

Continue with step 3: Implement SwinDSCViT model
