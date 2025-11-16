# DSC-ViT 역전파 분석: z vs (y+z) 클러스터링

## 핵심 변경사항

| | **잘못된 구현 (Old)** | **올바른 구현 (New)** |
|---|---|---|
| **클러스터링 대상** | `y + z` | `z` only |
| **설계 철학** | Answer + Reasoning 혼합 | Reasoning만 정제 |
| **y의 역할** | 클러스터링에 직접 관여 | 클러스터링과 분리 |

---

## 1. Forward Pass 비교

### 1.1 Old (WRONG): y+z 클러스터링

```python
def update_reasoning_latent(x_vit, image_cluster, y, z):
    # Step 1: y와 z를 합침
    combined_input = y + z  # [B, 512, 16, 16]

    # Step 2: 합친 것을 클러스터 공간으로 투영
    z_cluster = proj_latent_to_cluster(combined_input)  # [B, 21, 16, 16]

    # Step 3: Soft K-Means 클러스터링
    z_clustered, _, _ = clustering_layer(z_cluster)  # [B, 21, 16, 16]

    # Step 4: 이미지와 결합
    combined_cluster = z_clustered + image_cluster  # [B, 21, 16, 16]

    # Step 5: 다시 latent 공간으로
    z_new = proj_cluster_to_latent(combined_cluster)  # [B, 512, 16, 16]

    # Step 6: ViT residual
    z = z_new + x_vit

    return z, combined_cluster

def update_answer_latent(y, z):
    y = y + z
    return y
```

### 1.2 New (CORRECT): z만 클러스터링

```python
def update_reasoning_latent(x_vit, image_cluster, y, z):
    # Step 1: z만 클러스터 공간으로 투영
    z_cluster = proj_latent_to_cluster(z)  # [B, 21, 16, 16]

    # Step 2: Soft K-Means 클러스터링
    z_clustered, _, _ = clustering_layer(z_cluster)  # [B, 21, 16, 16]

    # Step 3: 이미지와 결합
    combined_cluster = z_clustered + image_cluster  # [B, 21, 16, 16]

    # Step 4: 다시 latent 공간으로
    z_new = proj_cluster_to_latent(combined_cluster)  # [B, 512, 16, 16]

    # Step 5: ViT residual
    z = z_new + x_vit

    return z, combined_cluster

def update_answer_latent(y, z):
    y = y + z  # y는 z로부터 업데이트받음
    return y
```

---

## 2. Computational Graph 비교

### 2.1 Old (WRONG): y+z 클러스터링

```
Input: image, y, z

Forward Flow:
image → x_vit (ViT encoder)
image → image_cluster (projection)

y ─┐
   ├→ (y + z) → z_cluster → z_clustered → combined_cluster → z_new → z
z ─┘                                                                    ↓
                                                                   y = y + z
                                                                        ↓
                                                                    seg_pred
                                                                        ↓
                                                                      Loss
```

**문제점:**
- `y`가 클러스터링 과정에 **직접 개입**
- `y`의 변화가 클러스터링 결과에 즉시 영향
- Reasoning latent와 Answer latent의 역할 구분 모호

### 2.2 New (CORRECT): z만 클러스터링

```
Input: image, y, z

Forward Flow:
image → x_vit (ViT encoder)
image → image_cluster (projection)

z → z_cluster → z_clustered → combined_cluster → z_new → z
                                                           ↓
y ────────────────────────────────────────────────→ y = y + z
                                                           ↓
                                                       seg_pred
                                                           ↓
                                                         Loss
```

**설계 철학:**
- **z (Reasoning latent)**: 클러스터링을 통해 정제됨
- **y (Answer latent)**: 정제된 z를 누적하여 답안 구축
- 명확한 역할 분리: z는 사고 과정, y는 답안 누적

---

## 3. Backward Pass (역전파) 비교

### 3.1 Old (WRONG): y+z 클러스터링

```
Loss (scalar)
  ↓ ∂L/∂seg_pred
seg_pred [B, 21, 256, 256]
  ↓ ∂L/∂y
y [B, 512, 16, 16]
  ├→ Path 1: Direct gradient
  │   ∂L/∂y (from seg_pred)
  │
  └→ Path 2: Through clustering
      ∂L/∂y (from y+z clustering)
        ↓ ∂L/∂(y+z)
      (y + z) [B, 512, 16, 16]
        ↓ ∂L/∂z_cluster
      z_cluster [B, 21, 16, 16]
        ↓ ∂L/∂z_clustered (through Soft K-Means)
      z_clustered [B, 21, 16, 16]
        ↓ ∂L/∂combined_cluster
      combined_cluster [B, 21, 16, 16]
        ↓ ∂L/∂z_new
      z_new [B, 512, 16, 16]
        ↓ ∂L/∂z
      z [B, 512, 16, 16]

Total gradient for y:
∂L/∂y = ∂L/∂y_direct + ∂L/∂y_clustering

Total gradient for z:
∂L/∂z = ∂L/∂z_from_y+z + ∂L/∂z_from_update
```

**Gradient Flow 특성:**
- ✅ `y`가 **2개 경로**로 gradient 받음
  1. Segmentation loss에서 직접
  2. Clustering loss를 통해
- ❌ `y`가 클러스터링에 영향 → **설계 의도와 불일치**
- ❌ `z`도 `y`의 영향을 받음 → Reasoning latent가 Answer에 종속

### 3.2 New (CORRECT): z만 클러스터링

```
Loss (scalar)
  ↓ ∂L/∂seg_pred
seg_pred [B, 21, 256, 256]
  ↓ ∂L/∂y
y [B, 512, 16, 16]
  ↓ ∂L/∂z (through y = y + z)
z [B, 512, 16, 16]
  ├→ Path 1: From y update
  │   ∂L/∂z (from ∂L/∂y)
  │
  └→ Path 2: From z update
      ∂L/∂z_new (through z = z_new + x_vit)
        ↓ ∂L/∂combined_cluster
      combined_cluster [B, 21, 16, 16]
        ↓ ∂L/∂z_clustered
      z_clustered [B, 21, 16, 16]
        ↓ ∂L/∂z_cluster (through Soft K-Means)
      z_cluster [B, 21, 16, 16]
        ↓ ∂L/∂z (through projection)
      z [B, 512, 16, 16]

Total gradient for y:
∂L/∂y = ∂L/∂y_segmentation  (단일 경로)

Total gradient for z:
∂L/∂z = ∂L/∂z_from_y + ∂L/∂z_clustering  (2개 경로)
```

**Gradient Flow 특성:**
- ✅ `y`는 **segmentation loss에서만** gradient
- ✅ `z`는 **2개 경로**로 gradient:
  1. `y` 업데이트를 통해 (간접)
  2. Clustering loss를 통해 (직접)
- ✅ 명확한 역할: `z`는 클러스터링으로 정제, `y`는 누적

---

## 4. Gradient 크기 비교

### 4.1 y의 Gradient

#### Old (y+z):
```
∂L/∂y = ∂L_seg/∂y + ∂L_cluster/∂y

여기서:
- ∂L_seg/∂y: Segmentation loss 기여
- ∂L_cluster/∂y: Clustering loss 기여 (y+z를 통해)

→ y가 clustering에 직접 관여하므로 gradient 복잡
```

#### New (z only):
```
∂L/∂y = ∂L_seg/∂y

여기서:
- ∂L_seg/∂y: Segmentation loss만 기여
- Clustering은 y에 직접 영향 없음

→ y의 gradient가 단순하고 명확
```

### 4.2 z의 Gradient

#### Old (y+z):
```
∂L/∂z = ∂L/∂(y+z) + ∂L/∂z_update

문제:
- ∂L/∂(y+z)는 y의 영향을 포함
- z가 독립적으로 최적화되지 않음
```

#### New (z only):
```
∂L/∂z = ∂L/∂y × ∂y/∂z + ∂L/∂z_cluster

여기서:
- ∂L/∂y × ∂y/∂z: y = y + z를 통한 간접 gradient
- ∂L/∂z_cluster: Clustering을 통한 직접 gradient

→ z가 클러스터링에서 독립적으로 최적화
```

---

## 5. 학습 동역학 변화

### 5.1 Old (y+z): 결합된 학습

| Iteration | y 업데이트 | z 업데이트 | 문제점 |
|---|---|---|---|
| 1 | y₁ = y₀ + z₀<br>Clustering(y₁ + z₀) | z₁ from Clustering(y₁ + z₀) | y와 z가 상호 의존 |
| 2 | y₂ = y₁ + z₁<br>Clustering(y₂ + z₁) | z₂ from Clustering(y₂ + z₁) | y 변화가 z에 즉시 영향 |
| ... | ... | ... | 누적 효과가 혼재 |

**특징:**
- ❌ y의 변화가 다음 z 업데이트에 즉시 반영
- ❌ z가 "이전 답안 + 현재 추론"의 혼합물 클러스터링
- ❌ 역할 구분 불명확

### 5.2 New (z only): 분리된 학습

| Iteration | y 업데이트 | z 업데이트 | 설계 의도 |
|---|---|---|---|
| 1 | y₁ = y₀ + z₀ | z₁ from Clustering(z₀) | z는 순수 reasoning |
| 2 | y₂ = y₁ + z₁ | z₂ from Clustering(z₁) | y는 z를 누적만 |
| 3 | y₃ = y₂ + z₂ | z₃ from Clustering(z₂) | 명확한 역할 분리 |

**특징:**
- ✅ z는 **이전 reasoning 결과**만 클러스터링
- ✅ y는 **정제된 reasoning을 누적**하는 역할만
- ✅ TRM 논문 철학과 일치: z는 사고, y는 답안

---

## 6. TRM 논문과의 비교

### TRM (Tiny Recursive Model) 설계:

```python
# TRM의 구조 (conceptual)
def recursive_step(input_features, y, z):
    # z: Reasoning latent (사고 과정)
    z_new = reasoning_network(input_features, z)  # z만 정제

    # y: Answer latent (답안 누적)
    y_new = y + z_new  # 정제된 z를 누적

    return y_new, z_new
```

**TRM의 핵심:**
- `z`: Scratch pad for reasoning (매 step 초기화 가능)
- `y`: Accumulated answer (계속 누적)

### DSC-ViT 설계 의도:

```python
# New (CORRECT) - TRM 철학 준수
z_cluster = proj_latent_to_cluster(z)  # z만 클러스터링 (reasoning 정제)
z_clustered = clustering_layer(z_cluster)
z_new = ... (z_clustered와 image 결합)

y = y + z  # 정제된 z를 y에 누적

# Old (WRONG) - TRM 철학 위배
combined = y + z  # y와 z를 혼합 → 역할 구분 파괴
z_cluster = proj_latent_to_cluster(combined)  # 혼합물 클러스터링
```

---

## 7. 손실 함수 Gradient 전파

### 7.1 Segmentation Loss

```python
L_seg = CrossEntropy(seg_pred, target)
seg_pred = seg_head(y)
```

#### Old (y+z):
```
∂L_seg/∂y → 직접 전파
∂L_seg/∂z → ∂y/∂z = 1 (from y = y + z)
           → ∂(y+z)/∂z = 1 (from clustering input)

→ z가 2개 경로로 gradient 받음 (혼재)
```

#### New (z only):
```
∂L_seg/∂y → 직접 전파
∂L_seg/∂z → ∂y/∂z = 1 (from y = y + z)

→ z는 y를 통해서만 seg loss gradient
→ 명확한 단일 경로
```

### 7.2 Cluster Loss

```python
L_cluster = ClusterLoss(combined_cluster, cluster_centers)
```

#### Old (y+z):
```
∂L_cluster/∂combined_cluster
  ↓
∂L_cluster/∂z_clustered
  ↓
∂L_cluster/∂z_cluster
  ↓
∂L_cluster/∂(y+z)
  ├→ ∂L_cluster/∂y  (y에 직접 영향!)
  └→ ∂L_cluster/∂z

→ y와 z 모두 cluster loss 영향받음
```

#### New (z only):
```
∂L_cluster/∂combined_cluster
  ↓
∂L_cluster/∂z_clustered
  ↓
∂L_cluster/∂z_cluster
  ↓
∂L_cluster/∂z  (z만 영향받음)

→ y는 cluster loss와 독립적
→ z만 clustering으로 정제
```

---

## 8. 학습 효과 차이

### 8.1 Old (y+z): 혼재된 최적화

**효과:**
- `y`가 clustering loss에 직접 노출 → 중간 추론 결과도 clustering에 영향
- `z`가 `y`의 영향을 받음 → reasoning이 이전 답안에 종속
- **문제:** 역할 구분 없이 모든 것이 뒤섞임

**수렴 특성:**
- Gradient가 복잡하게 얽혀있음
- y와 z가 동시에 변하면서 불안정할 수 있음
- 역할 분리 없이 jointly optimize

### 8.2 New (z only): 분리된 최적화

**효과:**
- `z`만 clustering loss에 노출 → reasoning만 정제
- `y`는 segmentation loss만 영향 → 답안 구축에 집중
- **장점:** 명확한 역할 분리

**수렴 특성:**
- `z`: Clustering으로 의미 있는 feature 학습
- `y`: `z`를 누적하여 최종 답안 구축
- TRM 논문의 recursive reasoning 철학 구현

---

## 9. 파라미터 업데이트 비교

### Projection Layers

| 파라미터 | Old (y+z) | New (z only) |
|---|---|---|
| `proj_latent_to_cluster` | ∂L/∂W includes ∂(y+z) | ∂L/∂W from z only |
| `proj_cluster_to_latent` | 동일 | 동일 |
| `clustering_layer.centers` | Gradient from (y+z) | Gradient from z |

**차이:**
- Old: Cluster centers가 y+z의 혼합 표현 학습
- New: Cluster centers가 순수 reasoning latent 표현 학습

---

## 10. 실험적 예측

### 10.1 Old (y+z) 예상 결과:

- ❌ Cluster visualization이 혼재된 표현
- ❌ y의 역할 불명확
- ❌ TRM 철학 위배
- ⚠️ 성능은 괜찮을 수 있으나 해석 어려움

### 10.2 New (z only) 예상 결과:

- ✅ Cluster visualization이 reasoning 과정 명확히 표현
- ✅ y는 답안, z는 추론으로 역할 분리
- ✅ TRM 철학 준수
- ✅ 해석 가능성 증가
- ⚠️ 초기 학습은 더 느릴 수 있음 (z가 독립적으로 학습)

---

## 11. 결론

### 왜 z만 클러스터링해야 하는가?

1. **TRM 설계 철학 준수**
   - z: Reasoning latent (매 step 정제)
   - y: Answer latent (누적)

2. **역할 명확화**
   - Clustering은 reasoning 정제 도구
   - y는 정제된 결과만 축적

3. **Gradient 단순화**
   - y는 segmentation에만 집중
   - z는 clustering에 집중

4. **해석 가능성**
   - z의 클러스터링 evolution을 보면 reasoning 과정 추적 가능
   - y는 최종 답안 표현

### 변경사항 요약

```diff
# models/dsc_vit.py
- combined_input = y + z
- z_cluster = self.projections.latent_to_cluster(combined_input)
+ z_cluster = self.projections.latent_to_cluster(z)
```

**이 한 줄 차이가 전체 학습 동역학을 바꿉니다.**

---

## 부록: Gradient 계산 예시

### A.1 Old (y+z) 상세 계산

```python
# Forward
y_t, z_t = current_latents
combined = y_t + z_t  # [B, 512, 16, 16]
z_cluster = W_ltc @ combined  # W_ltc: latent_to_cluster weights
z_clustered = SoftKMeans(z_cluster, μ)  # μ: cluster centers
combined_cluster = z_clustered + image_cluster
z_new = W_ctl @ combined_cluster  # W_ctl: cluster_to_latent weights
z_t+1 = z_new + x_vit
y_t+1 = y_t + z_t+1

# Backward (simplified)
∂L/∂y_t+1 = ∂L_seg/∂y_t+1

∂L/∂z_t+1 = ∂L/∂y_t+1 × ∂y_t+1/∂z_t+1 = ∂L/∂y_t+1 × 1

∂L/∂combined = ∂L/∂z_clustered × ∂SoftKMeans/∂z_cluster × W_ltc^T
∂L/∂y_t = ∂L/∂combined × 1
∂L/∂z_t = ∂L/∂combined × 1 + ∂L/∂z_t+1 (through residual)
```

### A.2 New (z only) 상세 계산

```python
# Forward
y_t, z_t = current_latents
z_cluster = W_ltc @ z_t  # Only z
z_clustered = SoftKMeans(z_cluster, μ)
combined_cluster = z_clustered + image_cluster
z_new = W_ctl @ combined_cluster
z_t+1 = z_new + x_vit
y_t+1 = y_t + z_t+1

# Backward (simplified)
∂L/∂y_t+1 = ∂L_seg/∂y_t+1

∂L/∂z_t+1 = ∂L/∂y_t+1 × 1

∂L/∂z_t = ∂L/∂z_clustered × ∂SoftKMeans/∂z_cluster × W_ltc^T
        + ∂L/∂z_t+1 (through residual)

∂L/∂y_t = ∂L/∂y_t+1  (no clustering path!)
```

**핵심 차이:**
- Old: `∂L/∂y_t`가 clustering path 포함
- New: `∂L/∂y_t`는 segmentation path만
