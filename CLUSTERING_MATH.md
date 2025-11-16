# DSC-ViT 클러스터링 과정: 수학적 설명

## 개요

DSC-ViT는 **Soft K-Means Clustering**을 사용하여 잠재 공간에서 미분 가능한 인식론적 범주화를 수행합니다. 이 문서는 클러스터링 메커니즘에 대한 엄밀한 수학적 설명을 제공합니다.

---

## 1. 기호 정의

| 기호 | 차원 | 설명 |
|------|------|------|
| `x` | `[B, C, H, W]` | 입력 이미지 |
| `z` | `[B, D, H', W']` | 추론 잠재 변수 (ViT 특징) |
| `y` | `[B, D, H', W']` | 답변 잠재 변수 (누적) |
| `f` | `[B, K, H', W']` | 클러스터 공간의 특징 (K차원) |
| `μ_k` | `[K]` | k번째 클러스터 중심 |
| `μ` | `[K_clusters, K]` | 모든 클러스터 중심 |
| `K` | 스칼라 | 클러스터 공간의 특징 차원 |
| `K_clusters` | 스칼라 | 클러스터 개수 (= num_classes = 21) |
| `τ` | 스칼라 | 온도 매개변수 (Temperature) |

---

## 2. Soft K-Means 클러스터링

### 2.1 문제 정의

특징 `f ∈ ℝ^(B×K×H'×W')`가 주어졌을 때, 다음을 수행:
1. 각 공간 위치를 K_clusters개의 클러스터 중심에 할당
2. 가중 평균을 통해 클러스터링된 특징 생성
3. End-to-end 학습을 위한 미분 가능성 유지

**목표**: 클러스터 내 분산을 최소화하면서 클러스터 간 분리를 최대화

---

### 2.2 단계 1: 거리 계산

각 공간 위치 `(h, w)`에 대해 모든 클러스터 중심까지의 **유클리드 거리 제곱** 계산:

```
d_{i,k} = ||f_i - μ_k||²
```

여기서:
- `f_i ∈ ℝ^K`: 위치 i의 특징 벡터
- `μ_k ∈ ℝ^K`: k번째 클러스터 중심
- `i`: 공간 인덱스 (i ∈ {1, ..., H'×W'})
- `k`: 클러스터 인덱스 (k ∈ {1, ..., K_clusters})

**효율적 계산** (명시적 루프 회피):

```
d_{i,k} = ||f_i||² + ||μ_k||² - 2⟨f_i, μ_k⟩
```

행렬 형태:
```python
# f: [B, H'W', K] (reshape 후)
# μ: [K_clusters, K]

f_squared = (f ** 2).sum(dim=-1, keepdim=True)      # [B, H'W', 1]
μ_squared = (μ ** 2).sum(dim=-1, keepdim=True).t()  # [1, K_clusters]
cross_term = f @ μ.t()                               # [B, H'W', K_clusters]

distances = f_squared + μ_squared - 2 * cross_term   # [B, H'W', K_clusters]
```

**계산 복잡도**: `O(B × H'W' × K × K_clusters)`

---

### 2.3 단계 2: Soft Assignment (소프트 할당)

온도 스케일링된 softmax를 사용하여 **소프트 할당 확률** 계산:

```
q_{i,k} = exp(-d_{i,k} / τ) / Σ_j exp(-d_{i,j} / τ)
```

여기서:
- `q_{i,k}`: 위치 `i`가 클러스터 `k`에 속할 확률
- `τ > 0`: 온도 매개변수 ("부드러움" 제어)
- `Σ_k q_{i,k} = 1`: 확률 합은 1

**온도 효과**:
- `τ → 0`: **Hard assignment** (one-hot, argmax)
  ```
  q_{i,k} ≈ 𝟙[k = argmin_j d_{i,j}]
  ```
- `τ → ∞`: **균등 할당** (모든 클러스터 동일 확률)
  ```
  q_{i,k} ≈ 1/K_clusters
  ```
- `τ = 1`: **표준 소프트 할당** (균형)

**구현**:
```python
# distances: [B, H'W', K_clusters]
# τ: 스칼라

q = softmax(-distances / τ, dim=-1)  # [B, H'W', K_clusters]
```

---

### 2.4 단계 3: 가중 클러스터 집계

클러스터 중심의 가중 평균으로 **클러스터링된 특징** 생성:

```
f̃_i = Σ_{k=1}^{K_clusters} q_{i,k} · μ_k
```

여기서:
- `f̃_i ∈ ℝ^K`: 위치 i의 클러스터링된 특징
- `q_{i,k}`: 단계 2의 소프트 할당 가중치
- `μ_k ∈ ℝ^K`: 클러스터 중심 (학습 가능한 파라미터)

**행렬 형태**:
```python
# q: [B, H'W', K_clusters]
# μ: [K_clusters, K]

f_clustered = q @ μ  # [B, H'W', K]
```

**직관**: 각 위치의 특징은 할당 확률로 가중된 클러스터 중심들의 **혼합**이 됨

---

## 3. 특징 정규화 (선택적)

안정성과 수렴을 개선하기 위해 특징과 클러스터 중심을 **L2 정규화** 가능:

```
f̂_i = f_i / ||f_i||₂
μ̂_k = μ_k / ||μ_k||₂
```

**정규화된 거리**는 **코사인 유사도**가 됨:
```
d_{i,k} = ||f̂_i - μ̂_k||² = 2 - 2⟨f̂_i, μ̂_k⟩ = 2(1 - cos(f̂_i, μ̂_k))
```

**이점**:
- 스케일 불변 클러스터링
- 더 나은 gradient flow
- 빠른 수렴

**구현** (soft_kmeans.py:90-94):
```python
if self.normalize_features:
    f = F.normalize(f, p=2, dim=-1)       # 특징 L2 정규화
    μ = F.normalize(self.cluster_centers, p=2, dim=-1)  # 중심 L2 정규화
```

---

## 4. 손실 함수

### 4.1 Assignment Loss (할당 손실, 지도 학습)

**목표**: 클러스터 할당이 실제 클래스 레이블과 일치해야 함

```
ℒ_assign = CrossEntropy(combined_cluster, target)
```

여기서:
- `combined_cluster ∈ ℝ^(B×K_clusters×H'×W')`: 클러스터 공간의 로짓
- `target ∈ ℤ^(B×H'×W')`: 실제 세그먼테이션 레이블

**수학적 형태**:
```
ℒ_assign = -1/(B·H'·W') Σ_{b,h,w} log( exp(c_{b,y_{b,h,w},h,w}) / Σ_k exp(c_{b,k,h,w}) )
```

여기서:
- `c`: combined_cluster 로짓
- `y_{b,h,w}`: 위치 (b,h,w)의 실제 레이블

**베이스라인 (랜덤 추측)**:
```
ℒ_assign^random = -log(1/K_clusters) = log(21) ≈ 3.04  (VOC 21 클래스)
```

---

### 4.2 Compactness Loss (밀집도 손실, 비지도 학습)

**목표**: 특징들이 할당된 클러스터 중심에 **가까워야** 함

```
ℒ_compact = 1/(B·H'·W') Σ_{b,h,w} Σ_k q_{b,h,w,k} · d_{b,h,w,k}
```

여기서:
- `q_{b,h,w,k}`: 소프트 할당 확률
- `d_{b,h,w,k} = ||f_{b,h,w} - μ_k||²`: 거리 제곱

**직관**: 클러스터 중심까지의 **가중 평균 거리** 최소화

**구현** (utils/losses.py:109-140):
```python
def compactness_loss(self, features, cluster_centers):
    # features: [B, K, H', W']
    # cluster_centers: [K_clusters, K]

    # 1. 거리 계산
    distances = compute_distances(features, cluster_centers)  # [B, H'W', K_clusters]

    # 2. 소프트 할당
    q = softmax(-distances, dim=-1)  # [B, H'W', K_clusters]

    # 3. 가중 거리
    loss = (q * distances).sum(dim=-1).mean()  # 스칼라

    return loss
```

**범위**: `[0, ∞)` (낮을수록 좋음)

---

### 4.3 Separation Loss (분리 손실, 비지도 학습)

**목표**: 클러스터 중심들이 서로 **멀리** 떨어져 있어야 함

```
ℒ_separate = -1/K_clusters Σ_{k=1}^{K_clusters} min_{j≠k} ||μ_k - μ_j||₂
```

**직관**: 클러스터 중심 간 **최소 쌍별 거리** 최대화

**왜 최소값?** **모든 쌍**이 분리되도록 보장하기 위해 (평균이 아님)

**구현** (utils/losses.py:142-164):
```python
def separation_loss(self, cluster_centers):
    # cluster_centers: [K_clusters, K]

    # 1. 쌍별 거리
    dist_matrix = torch.cdist(cluster_centers, cluster_centers, p=2)  # [K_clusters, K_clusters]

    # 2. 대각선 마스킹
    mask = torch.eye(K_clusters, dtype=torch.bool)
    dist_matrix = dist_matrix.masked_fill(mask, float('inf'))

    # 3. 클러스터당 최소 거리
    min_distances = dist_matrix.min(dim=-1)[0]  # [K_clusters]

    # 4. 평균 및 부호 반전 (최대화를 위해)
    return -min_distances.mean()
```

**범위**: `(-∞, 0]` (0에 가까울수록 좋음, 즉 큰 분리)

---

### 4.4 총 클러스터 손실

```
ℒ_cluster = ℒ_assign + α · ℒ_compact + β · ℒ_separate
```

여기서:
- `α`: 밀집도 가중치 (기본값: 1.0)
- `β`: 분리 가중치 (기본값: 0.5)

**하이퍼파라미터 튜닝**:
- **α 너무 작음**: 클러스터가 밀집된 그룹 형성 안 함 → 일반화 성능 저하
- **α 너무 큼**: 훈련 데이터에 과적합, 경직된 클러스터
- **β 너무 작음**: 클러스터 중심이 함께 붕괴 → mode collapse
- **β 너무 큼**: 클러스터 중심이 극단으로 밀림 → 할당 성능 저하

**권장 설정** (configs/default.yaml):
```yaml
cluster_loss_alpha: 1.0   # Assignment loss(~3.0)와 균형
cluster_loss_beta: 0.5    # Compactness 가중치의 절반
```

---

## 5. DSC-ViT 클러스터링 파이프라인

### 5.1 전체 아키텍처

```
입력 x [B, C, H, W]
  ↓
ViT 인코더 (배치당 1회)
  ↓
x_vit [B, D, H', W']  (캐시됨)
  ↓
┌─────── 재귀 루프 (T번) ───────┐
│                               │
│  y, z: 답변 & 추론 잠재변수    │
│  ↓                            │
│  Proj_{D→K}: (y+z) → f        │  ← 클러스터 공간으로 투영
│       [B, K, H', W']          │
│  ↓                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  │ Soft K-Means 클러스터링 │ │
│  │                          │ │
│  │ 1. 거리: d = ||f - μ||²  │ │
│  │ 2. 할당: q = softmax(-d/τ)│ │
│  │ 3. 집계: f̃ = Σ q_k·μ_k   │ │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  ↓                            │
│  f̃ [B, K, H', W']  (클러스터링)│
│  ↓                            │
│  결합: f̃ + image_cluster       │
│  ↓                            │
│  Proj_{K→D}: combined → z_new │  ← 잠재 공간으로 역투영
│  ↓                            │
│  z = z_new + x_vit  (잔차)    │
│  ↓                            │
│  y = y + z  (답변 업데이트)    │
│                               │
└───────────────────────────────┘
  ↓
세그먼테이션 헤드: y → output [B, num_classes, H, W]
```

### 5.2 단계별 순전파

**초기화** (t=0):
```
y^(0) = 0
z^(0) = 0
x_vit = ViT(x)  (1회 계산, 캐시)
image_cluster = Proj_{C→K}(x)  (1회 계산, 캐시)
```

**재귀 반복** (t = 1, ..., T):
```
1. f^(t) = Proj_{D→K}(y^(t-1) + z^(t-1))         # K-공간으로 투영

2. 클러스터링:
   d^(t) = ||f^(t) - μ||²                        # 거리
   q^(t) = softmax(-d^(t) / τ^(t))               # 소프트 할당
   f̃^(t) = Σ_k q^(t)_k · μ_k                     # 클러스터링된 특징

3. combined^(t) = f̃^(t) + image_cluster         # 이미지 정보 추가

4. z_new^(t) = Proj_{K→D}(combined^(t))          # D-공간으로 복귀

5. z^(t) = z_new^(t) + x_vit                     # 잔차 연결

6. y^(t) = y^(t-1) + z^(t)                       # 답변 누적
```

**최종 출력**:
```
seg_pred = SegHead(y^(T))  # [B, num_classes, H, W]
```

---

## 6. Temperature Annealing (온도 어닐링)

### 6.1 스케줄

훈련 중 온도 `τ`를 점진적으로 감소:

```
τ(epoch) = τ_final + 0.5·(τ_init - τ_final)·(1 + cos(π·progress))
```

여기서 `progress = epoch / total_epochs`

**예시** (100 에포크):
```
Epoch   0: τ = 2.00  (매우 부드러움)
Epoch  25: τ = 1.55
Epoch  50: τ = 1.05  (중간)
Epoch  75: τ = 0.55
Epoch 100: τ = 0.10  (매우 날카로움, 거의 hard)
```

### 6.2 할당에 미치는 영향

**높은 온도 (τ = 2.0)** - 학습 초기:
```
q = [0.23, 0.19, 0.21, 0.18, 0.19]  # 부드럽고 탐색적
```

**낮은 온도 (τ = 0.1)** - 학습 후기:
```
q = [0.92, 0.02, 0.03, 0.01, 0.02]  # 날카롭고 결정적
```

**이점**:
1. **탐색 → 활용**: 탐색을 위해 부드럽게 시작, 확정을 위해 날카롭게 종료
2. **그래디언트 흐름**: 소프트 할당이 초기에 더 나은 그래디언트 제공
3. **수렴**: 날카로운 할당이 테스트 시 불확실성 감소

---

## 7. 그래디언트 흐름

### 7.1 클러스터링을 통한 역전파

**순전파**:
```
f → d → q → f̃
```

**역전파** (연쇄 법칙):
```
∂ℒ/∂f = (∂ℒ/∂f̃) · (∂f̃/∂q) · (∂q/∂d) · (∂d/∂f)
```

**주요 그래디언트**:

1. **할당에 대한 클러스터링된 특징의 미분**:
   ```
   ∂f̃_i/∂q_{i,k} = μ_k
   ```

2. **거리에 대한 소프트 할당의 미분**:
   ```
   ∂q_{i,k}/∂d_{i,j} = (1/τ) · q_{i,k} · (𝟙[j=k] - q_{i,j})
   ```

3. **특징에 대한 거리의 미분**:
   ```
   ∂d_{i,k}/∂f_i = 2(f_i - μ_k)
   ```

**클러스터 중심 그래디언트**:
```
∂ℒ/∂μ_k = Σ_i (∂ℒ/∂f̃_i) · q_{i,k} + ∂ℒ_compact/∂μ_k + ∂ℒ_separate/∂μ_k
```

### 7.2 수치 안정성

**문제**: `d`가 클 경우 `exp(-d/τ)` 오버플로우 가능

**해결책**: softmax 전에 최댓값 빼기 (표준 트릭):
```python
def stable_softmax(logits):
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - logits_max)
    return exp_logits / exp_logits.sum(dim=-1, keepdim=True)
```

PyTorch의 `F.softmax`는 이미 구현되어 있음.

---

## 8. 학습 역학

### 8.1 예상 손실 진화

| Epoch | ℒ_assign | ℒ_compact | ℒ_separate | ℒ_cluster | 해석 |
|-------|----------|-----------|------------|-----------|------|
| 0 | 3.04 | 2.5 | -1.0 | 3.04 + 2.5 - 0.5 = 5.04 | 랜덤 초기화 |
| 10 | 2.0 | 1.5 | -2.0 | 2.0 + 1.5 - 1.0 = 2.5 | 클러스터 학습 중 |
| 50 | 0.8 | 0.5 | -5.0 | 0.8 + 0.5 - 2.5 = -1.2 | 좋은 클러스터 |
| 100 | 0.3 | 0.1 | -8.0 | 0.3 + 0.1 - 4.0 = -3.6 | 잘 분리됨 |

**참고**: 분리 항 때문에 총 손실이 **음수**가 될 수 있음!

### 8.2 진단 메트릭

**클러스터링 불량** (ℒ_assign ≈ 3.0):
- 할당이 랜덤
- 클러스터 중심이 클래스와 정렬되지 않음
- **해결**: α, β 증가 또는 초기화 개선

**Mode collapse** (ℒ_separate ≈ 0):
- 모든 클러스터 중심이 멀리 떨어짐 → 좋음!
- `ℒ_separate > -1.0`이면 → 중심들이 너무 가까움

**과적합** (Train ℒ_compact → 0, Val ℒ_compact > 1):
- 훈련 데이터에서 클러스터가 너무 밀집
- **해결**: α 감소 또는 정규화 추가

---

## 9. 구현 참조

### 9.1 핵심 파일

- **`models/soft_kmeans.py`**: Soft K-Means 레이어 구현
  - 80-117줄: 거리 계산
  - 119-131줄: 소프트 할당
  - 133-161줄: 클러스터링된 특징 생성

- **`utils/losses.py`**: 클러스터링 손실 함수
  - 74-107줄: 할당 손실
  - 109-140줄: 밀집도 손실
  - 142-164줄: 분리 손실

- **`train.py`**: 온도 어닐링
  - 353-376줄: 코사인 어닐링 스케줄

### 9.2 주요 하이퍼파라미터

```yaml
# configs/default.yaml

model:
  num_clusters: 21  # K_clusters (num_classes와 동일)
  latent_dim: 512   # D

loss:
  cluster_loss_alpha: 1.0   # α (밀집도 가중치)
  cluster_loss_beta: 0.5    # β (분리 가중치)

cluster_temperature:
  initial: 2.0      # τ_init (부드럽게 시작)
  final: 0.1        # τ_final (날카롭게 종료)
  schedule: 'cosine'
```

---

## 10. 수학적 성질

### 10.1 볼록성

**할당 손실**: 로짓에 대해 볼록 (cross-entropy)

**밀집도 손실**: 비볼록 (소프트 할당에 의존)

**분리 손실**: 비볼록 (min 연산)

**총 손실**: 비볼록 → 신중한 초기화와 어닐링 필요

### 10.2 수렴 보장

**Soft K-Means**는 전역 최적해로의 수렴을 보장하지 않지만:
- 온도 어닐링이 불량한 국소 최솟값 회피에 도움
- 지도 할당 손실이 강력한 가이드 제공
- 잔차 연결이 그래디언트 소실 방지

### 10.3 표준 K-Means와의 관계

**표준 K-Means** (hard assignment):
```
1. 할당: c_i = argmin_k ||f_i - μ_k||²
2. 업데이트: μ_k = mean({f_i : c_i = k})
```

**Soft K-Means** (미분 가능):
```
1. 할당: q_{i,k} = softmax(-||f_i - μ_k||² / τ)
2. 업데이트: μ_k ← μ_k - η·∂ℒ/∂μ_k  (경사 하강법)
```

**핵심 차이**: 소프트 할당이 역전파를 통한 **end-to-end 학습** 가능!

---

## 11. Ablation Studies (권장 실험)

클러스터링 메커니즘 검증을 위한 테스트:

1. **클러스터링 없음** (베이스라인): D→num_classes 직접 투영
2. **Hard K-Means** (τ → 0): 미분 불가능, argmax 할당
3. **온도 어닐링 없음**: 고정 τ = 1.0
4. **분리 손실 없음**: 할당 + 밀집도만
5. **다른 K_clusters**: K < num_classes 또는 K > num_classes

**예상 순위** (mIoU):
```
전체 모델 > 어닐링 없음 > 분리 없음 > Hard K-Means > 클러스터링 없음
```

---

## 참고문헌

1. **Tiny Recursive Model (TRM)**: arXiv:2510.04871v1
2. **Soft K-Means**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
3. **Temperature scaling**: Hinton et al. (2015). Distilling the Knowledge in a Neural Network.
4. **Supervised Contrastive Learning**: Khosla et al. (2020). Supervised Contrastive Learning.

---

**마지막 업데이트**: 2024-11-16
