# TRM (Tiny Recursive Model) 구조 분석

## 논문 정보
- **제목**: Less is More: Recursive Reasoning with Tiny Networks
- **arXiv**: 2510.04871v1
- **저자**: Alexia Jolicoeur-Martineau (Samsung SAIL Montreal)

---

## 1. 핵심 아이디어

### 문제점
- LLM은 CoT + TTC를 써도 어려운 문제(Sudoku, Maze, ARC-AGI)에서 한계
- HRM은 복잡하고 biological argument에 의존

### 해결책
TRM은 **단일 tiny network (2-layer)로 recursive reasoning**을 수행:
- 7M parameters로 45% ARC-AGI-1, 8% ARC-AGI-2 달성
- LLM 대비 <0.01% parameter로 더 높은 성능

---

## 2. TRM vs HRM 비교

### HRM (Hierarchical Reasoning Model)
```python
# 두 개의 네트워크 사용
f_L(z_L, z_H, x)  # Low-level network (고빈도)
f_H(z_L, z_H)     # High-level network (저빈도)

# 두 개의 latent feature
z_L  # Low-level latent
z_H  # High-level latent (output과 대응)
```

**복잡한 점**:
- 왜 두 개 네트워크인가? → Biological argument
- 왜 두 개 latent인가? → Hierarchical processing
- Fixed-point theorem 가정 (문제 있음)

### TRM (Tiny Recursive Model)
```python
# 단일 네트워크 사용
net(x, y, z)  # x=input, y=answer, z=reasoning latent

# 두 개의 feature (하지만 의미 명확)
y  # Current answer (이전: z_H)
z  # Reasoning latent (이전: z_L)
```

**명확한 해석**:
- `y`: 현재 예측 답안 (embedded form)
- `z`: 추론 과정의 latent (chain-of-thought 같은 역할)
- 입력 `x`를 주면 → `z`를 업데이트 → `z`로 `y`를 개선

---

## 3. TRM 구조 상세

### 3.1 주요 파라미터

| 파라미터 | 기호 | 값 | 의미 |
|---------|------|-----|------|
| Latent recursion | n | 6 | z를 업데이트하는 횟수 |
| Answer update frequency | T | 3 | y를 업데이트하는 주기 |
| Deep supervision steps | N_sup | 16 | 같은 배치를 반복 학습하는 횟수 |
| Network layers | - | 2 | Transformer 레이어 수 |

### 3.2 Latent Recursion (n번)

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):  # z만 n번 업데이트
        z = net(x, y, z)

    y = net(y, z)  # z로 y 개선

    return y, z
```

**역할**:
- `z` (reasoning latent)를 n=6번 반복적으로 정제
- 마지막에 정제된 `z`로 answer `y` 업데이트
- 이것이 한 번의 "reasoning cycle"

### 3.3 Deep Recursion (T번 반복 + gradient 전략)

```python
def deep_recursion(x, y, z, n=6, T=3):
    # ⭐ T-1번은 gradient 없이
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)

    # ⭐ 마지막 T번째만 gradient
    y, z = latent_recursion(x, y, z, n)

    return (y.detach(), z.detach()), output_head(y), Q_head(y)
```

**1-step Gradient Approximation**:
- 총 `n*T = 6*3 = 18`번의 recursion
- 하지만 **마지막 n+1 = 7번만 backprop**
- 처음 11번은 no_grad (메모리 절약)
- HRM의 fixed-point theorem 없이도 작동

**왜 이렇게?**
- Deep equilibrium model처럼 fixed-point까지 iterate하지 않음
- 그냥 T-1번은 forward만 → 마지막만 backprop
- 논문: "removes entirely the need to assume that a fixed-point is reached"

### 3.4 Deep Supervision (N_sup번 반복)

```python
# Outer loop: 각 배치마다
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init

    # Inner loop: 같은 배치를 최대 N_sup=16번 학습
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)

        loss = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))

        loss.backward()  # ⭐ 각 step마다
        opt.step()
        opt.zero_grad()

        if q_hat > 0:  # early-stopping (ACT)
            break
```

**핵심 포인트**:
1. **각 배치를 여러 번 학습**: 같은 (x_input, y_true)를 최대 16번
2. **독립적 backward**: 각 step마다 loss.backward() + opt.step()
3. **Gradient 격리**: detach()로 이전 step과 차단
4. **Progressive refinement**: (y, z)가 점점 정답에 가까워짐

### 3.5 ACT (Adaptive Computational Time)

```python
# Q-head output (logit, not probability)
q_hat = Q_head(y)  # Logit 값

# Early stopping (논문 Figure 3)
if q_hat > 0:  # Logit > 0 means P(halt) > 0.5
    break  # 다음 배치로
```

**Loss 계산**:
```python
# Binary cross-entropy (q_hat은 logit)
loss += binary_cross_entropy(q_hat, (y_hat == y_true))
```

**효과**:
- 쉬운 문제: 1-2 step만 사용
- 어려운 문제: 16 step 모두 사용
- Sudoku-Extreme 평균: <2 steps (논문 보고)

**주의**: `q_hat`은 sigmoid 이전의 **logit 값**이므로, 0을 기준으로 판단

---

## 4. TRM의 Effective Depth

### 정의 (논문 Table 1)
**Effective depth per supervision step** = T × (n + 1) × n_layers

### 계산 (TRM with T=3, n=6, 2-layer)
```
Effective depth (per step) = T × (n + 1) × n_layers
                           = 3 × (6 + 1) × 2
                           = 3 × 7 × 2
                           = 42 layers
```

### Backprop Depth (per step)
- 전체 42 layers 중, T-1번은 no_grad
- 마지막 1번의 latent_recursion만 backprop
```
Backprop depth (per step) = (n + 1) × n_layers
                          = (6 + 1) × 2
                          = 14 layers
```

### Deep Supervision 전체 계산
- **하나의 supervision step**: 42 layers (effective depth)
- **N_sup = 16번 반복**: 42 × 16 = 672 layers의 **총 계산량**
- **메모리**: 2-layer network만 저장 (detach 덕분)
- **매우 깊은 네트워크를 메모리 효율적으로 모방**

### 요약
| 측정 항목 | 값 | 의미 |
|----------|-----|------|
| Network depth | 2 | 실제 네트워크 레이어 수 |
| Effective depth (per step) | 42 | 하나의 supervision step의 계산 깊이 |
| Backprop depth (per step) | 14 | gradient가 흐르는 깊이 |
| Total computation | 672 | N_sup=16 × 42 layers |

---

## 5. TRM vs HRM 주요 차이

| 구성 요소 | HRM | TRM |
|----------|-----|-----|
| 네트워크 수 | 2개 (f_L, f_H) | 1개 (net) |
| 파라미터 | 27M | 7M |
| 레이어 수 | 4 | 2 |
| n (L steps) | 2 | 6 |
| T (H steps) | 2 | 3 |
| Gradient 전략 | 1-step approx (복잡) | Full recursion (단순) |
| 이론적 근거 | Fixed-point + biology | 불필요 |
| Sudoku-Extreme | 55% | 87.4% |

---

## 6. 핵심 혁신 요약

### 6.1 "Less is More"
- **더 작은 네트워크** (4-layer → 2-layer): 과적합 감소
- **단일 네트워크**: f_L과 f_H 통합 (입력으로 구분)
- **명확한 해석**: y=answer, z=reasoning (biological argument 불필요)

### 6.2 No Fixed-Point Theorem
- HRM: fixed-point 가정 → 문제 있음
- TRM: 그냥 T번 recursion → 마지막만 backprop
- 더 단순하고 효과적

### 6.3 Deep Supervision의 진정한 의미
```
Step 1: (y_0, z_0) → recursion → (y_1, z_1) [학습] → detach
Step 2: (y_1, z_1) → recursion → (y_2, z_2) [학습] → detach
...
Step N: (y_N-1, z_N-1) → recursion → (y_N, z_N) [학습]
```

각 step이 이전 결과를 **개선**하는 것이지, 처음부터 다시 학습하는 것이 아님!

---

## 7. TRM Training Flow 요약

```python
for epoch in epochs:
    for batch (x, y_true) in dataloader:  # 배치 루프
        y, z = initialize()

        for step in range(N_sup):  # Deep supervision (최대 16번)

            # T-1번: no gradient
            with torch.no_grad():
                for t in range(T-1):  # T-1 = 2번
                    for i in range(n):  # n = 6번
                        z = net(x, y, z)
                    y = net(y, z)

            # T번째: with gradient
            for i in range(n):  # n = 6번
                z = net(x, y, z)
            y = net(y, z)

            # Prediction
            y_pred = output_head(y)

            # Loss & Update
            loss = cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Detach for next step
            y = y.detach()
            z = z.detach()

            # Early stopping (ACT)
            if should_stop(y_pred, y_true):
                break
```

---

## 8. DSC-ViT 적용 시 고려사항

### 8.1 필요한 수정
1. **n 파라미터 추가**: latent recursion 횟수
2. **Gradient 전략**: T-1번 no_grad, T번째만 grad
3. **Single network**: 하나의 네트워크로 z 업데이트 + y 업데이트

### 8.2 구조 매핑

| TRM | DSC-ViT |
|-----|---------|
| x (input) | image_cluster (C→K projected) |
| y (answer) | z_latent (D-dim latent) |
| z (reasoning) | ? (새로 추가 필요) |
| net(x, y, z) | recursive_step(x, y, z) |
| output_head(y) | seg_head(z_latent) |

### 8.3 현재 구현의 문제
- **n 파라미터 없음**: z를 몇 번 업데이트할지 명시 안됨
- **Gradient 전략 없음**: 모든 T step에 gradient
- **y/z 구분 불명확**: TRM처럼 명확히 분리 안됨

---

## 9. 검증 완료 사항

1. ✅ Deep supervision = 같은 배치를 N_sup번 반복
2. ✅ 각 step마다 독립적 backward + optimizer.step()
3. ✅ detach()로 gradient 격리
4. ✅ n번 z 업데이트, T번 주기로 y 업데이트
5. ✅ T-1번은 no_grad, T번째만 gradient
6. ✅ y와 z를 명확히 분리
7. ✅ Effective depth = T × (n+1) × n_layers = 42 (per step)
8. ✅ ACT 조건 = q_hat > 0 (logit 기준)

---

## 10. 다음 단계

1. **이 MD 파일 검토**: 구조 이해가 맞는지 확인
2. **DSC-ViT 재설계**: TRM 구조에 맞게 수정
3. **구현**: n, T, gradient 전략 반영
4. **테스트**: 작은 데이터셋으로 학습 확인
