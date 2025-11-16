# PASCAL VOC 2012 데이터셋 설정 가이드

이 문서는 DSC-ViT 프로젝트에서 PASCAL VOC 2012 Segmentation 데이터셋을 설정하는 전체 워크플로우를 설명합니다.

## 📋 목차

1. [프로젝트 구조 파악](#1-프로젝트-구조-파악)
2. [데이터셋 다운로드 방법](#2-데이터셋-다운로드-방법)
3. [환경 설정 (uv 사용)](#3-환경-설정-uv-사용)
4. [데이터셋 다운로드](#4-데이터셋-다운로드)
5. [데이터셋 검증](#5-데이터셋-검증)
6. [다음 단계](#6-다음-단계)

---

## 1. 프로젝트 구조 파악

### 핵심 디렉토리 구조

```
DSC-ViT/
├── models/              # 모델 아키텍처
│   ├── dsc_vit.py      # DSC-ViT 메인 모델
│   ├── vit_encoder.py  # ViT 인코더
│   ├── soft_kmeans.py  # 소프트 K-평균 클러스터링
│   └── projections.py  # 투영 레이어
│
├── data/               # 데이터셋 로더
│   ├── datasets.py     # torchvision VOC 로더 (방법 1)
│   ├── voc_kaggle.py   # Kaggle VOC 로더 (방법 2) ✅
│   └── transforms.py   # 데이터 증강
│
├── utils/              # 손실 함수 & 메트릭
├── configs/            # 설정 파일
└── train.py            # 학습 스크립트
```

### 주요 특징

- **모델**: DSC-ViT (Deep Supervised ViT with Clustering)
- **태스크**: Semantic Segmentation
- **데이터셋**: PASCAL VOC 2012 (21 classes)
- **학습 방식**: TRM-style Deep Supervision

---

## 2. 데이터셋 다운로드 방법

프로젝트에서 지원하는 두 가지 방법:

### 방법 1: torchvision 자동 다운로드

```python
# data/datasets.py 사용
# torchvision이 자동으로 다운로드
from data import VOCSegmentation

dataset = VOCSegmentation(
    root='./data',
    split='train',
    download=True  # 자동 다운로드
)
```

**장점:**
- 가장 간단한 방법
- 공식 torchvision API 사용
- 자동 다운로드 및 압축 해제

**단점:**
- 일부 환경에서 다운로드 실패 가능
- 네트워크 제한 시 문제 발생 가능

### 방법 2: Kaggle 데이터셋 (✅ 선택한 방법)

```python
# data/voc_kaggle.py 사용
# Kaggle에서 직접 다운로드
from data import VOCSegmentationKaggle

dataset = VOCSegmentationKaggle(
    root='/root/.cache/kagglehub/datasets/.../voc_2012_segmentation_data',
    split='train'
)
```

**장점:**
- 안정적인 다운로드
- 명확한 데이터 구조
- Kaggle API를 통한 빠른 다운로드

**단점:**
- kagglehub 패키지 필요
- 수동 다운로드 단계 필요

---

## 3. 환경 설정 (uv 사용)

이 프로젝트는 **uv** (Python 패키지 관리자)를 사용합니다.

### 3.1 의존성 설치

```bash
# uv로 모든 의존성 설치
uv add -r requirements.txt
```

### 3.2 필수 패키지

`requirements.txt`에 포함된 주요 패키지:

```txt
# 핵심 의존성
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0

# 데이터 처리
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
albumentations>=1.3.0

# Kaggle 데이터셋용
kagglehub>=0.2.0

# 메트릭 & 시각화
scikit-learn>=1.3.0
matplotlib>=3.7.0
tensorboard>=2.13.0
wandb>=0.15.0

# 유틸리티
tqdm>=4.65.0
pyyaml>=6.0
einops>=0.6.0
```

### 3.3 실행 방법

uv를 사용할 때는 항상 `uv run` 접두사를 붙여야 합니다:

```bash
# ❌ 일반 실행 (작동 안 함)
python train.py

# ✅ uv 실행 (올바른 방법)
uv run python train.py
```

---

## 4. 데이터셋 다운로드

### 4.1 Kaggle 데이터셋 다운로드

```bash
uv run python -c "
import kagglehub
path = kagglehub.dataset_download('sovitrath/voc-2012-segmentation-data')
print(f'✓ Downloaded to: {path}')
"
```

### 4.2 다운로드 과정

```
Downloading from https://www.kaggle.com/api/v1/datasets/download/...
Extracting files...
✓ Downloaded to: /root/.cache/kagglehub/datasets/sovitrath/voc-2012-segmentation-data/versions/1
```

**다운로드 크기**: 316MB (압축)

**소요 시간**: 약 8-10초 (네트워크 속도에 따라 다름)

### 4.3 데이터셋 구조 확인

```bash
# 다운로드된 데이터셋 구조 확인
ls -la /root/.cache/kagglehub/datasets/sovitrath/voc-2012-segmentation-data/versions/1/voc_2012_segmentation_data/
```

**결과:**

```
voc_2012_segmentation_data/
├── train_images/    # 1,464개의 .jpg 파일
├── train_labels/    # 1,464개의 .png 파일 (세그멘테이션 마스크)
├── valid_images/    # 1,449개의 .jpg 파일
└── valid_labels/    # 1,449개의 .png 파일
```

---

## 5. 데이터셋 검증

### 5.1 데이터 로더 테스트

```bash
uv run python data/voc_kaggle.py
```

### 5.2 테스트 결과

```
============================================================
Testing VOC Kaggle Dataset
============================================================

[1] Testing train split...
VOC Kaggle train: 1464 images loaded
  Image dir: /root/.cache/kagglehub/datasets/.../train_images
  Label dir: /root/.cache/kagglehub/datasets/.../train_labels
    Image: torch.Size([3, 256, 256]), dtype=torch.float32
    Mask: torch.Size([256, 256]), dtype=torch.int64
    Mask unique: [-100, 0, 1, 15]

[2] Testing valid split...
VOC Kaggle valid: 1449 images loaded
  Image dir: /root/.cache/kagglehub/datasets/.../valid_images
  Label dir: /root/.cache/kagglehub/datasets/.../valid_labels
    Image: torch.Size([3, 256, 256]), dtype=torch.float32
    Mask: torch.Size([256, 256]), dtype=torch.int64

============================================================
Dataset Summary
============================================================
Classes: 21
Train: 1464 images
Valid: 1449 images
============================================================

✓ VOC Kaggle dataset ready!
```

### 5.3 검증 항목

✅ **이미지 Shape**: `[3, 256, 256]` (C, H, W)
- 3 채널 (RGB)
- 256x256 해상도 (증강 후)

✅ **마스크 Shape**: `[256, 256]` (H, W)
- 픽셀별 클래스 인덱스 (0-20)
- -100: ignore index (경계 픽셀)

✅ **클래스 개수**: 21
- 20개 객체 클래스
- 1개 배경 클래스

✅ **데이터 개수**:
- Train: 1,464장
- Valid: 1,449장

### 5.4 VOC 클래스 목록

```python
VOC_CLASSES = [
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
```

---

## 6. 다음 단계

데이터셋이 준비되었으니 이제 학습을 시작할 수 있습니다!

### 6.1 빠른 테스트 (합성 데이터)

데이터셋 다운로드 없이 모델 파이프라인을 테스트:

```bash
uv run python test_training.py
```

### 6.2 전체 학습 시작

PASCAL VOC 2012 데이터셋으로 전체 학습:

```bash
uv run python train.py --config configs/default.yaml
```

### 6.3 디버그 모드

작은 서브셋으로 빠른 디버깅:

```bash
uv run python train.py --config configs/default.yaml --debug
```

### 6.4 학습 설정 (configs/default.yaml)

```yaml
model:
  num_classes: 21
  latent_dim: 512
  vit_model_name: 'vit_base_patch16_224'
  num_latent_updates: 6      # n
  num_recursive_steps: 3     # T
  num_supervision_steps: 16  # N_sup

training:
  epochs: 100
  batch_size: 8
  lr: 1.0e-4
  use_act: true  # Adaptive Computation Time
  use_ema: true  # Exponential Moving Average
```

---

## 📊 요약

### 완료된 단계

1. ✅ 프로젝트 구조 파악
2. ✅ 데이터셋 다운로드 방법 선택 (Kaggle)
3. ✅ uv로 의존성 설치
4. ✅ Kaggle에서 PASCAL VOC 2012 다운로드
5. ✅ 데이터셋 구조 확인
6. ✅ 데이터 로더 테스트 및 검증

### 주요 경로

- **데이터셋 루트**: `/root/.cache/kagglehub/datasets/sovitrath/voc-2012-segmentation-data/versions/1/voc_2012_segmentation_data`
- **프로젝트 루트**: `/workspace/DSC-ViT`
- **설정 파일**: `configs/default.yaml`

### 핵심 명령어

```bash
# 데이터셋 다운로드
uv run python -c "import kagglehub; kagglehub.dataset_download('sovitrath/voc-2012-segmentation-data')"

# 데이터셋 검증
uv run python data/voc_kaggle.py

# 학습 시작
uv run python train.py --config configs/default.yaml
```

---

## 🔧 문제 해결

### 문제 1: "No module named 'kagglehub'"

**해결책:**
```bash
uv add kagglehub
# 또는
uv pip install kagglehub
```

### 문제 2: torchvision 자동 다운로드 실패

**해결책:**
- Kaggle 방법(방법 2) 사용
- 이 문서의 4단계 참고

### 문제 3: uv run 실행 오류

**해결책:**
```bash
# pyproject.toml이 없는 경우
# 먼저 의존성 설치
uv add -r requirements.txt

# 그 다음 실행
uv run python <script.py>
```

---

## 📚 참고 자료

- **프로젝트 문서**: `README.md`, `CLAUDE.md`, `TRM_ARCHITECTURE.md`
- **PASCAL VOC 공식**: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- **Kaggle 데이터셋**: https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
- **TRM 논문**: arXiv:2510.04871v1

---

**작성일**: 2024-11-16
**상태**: ✅ 데이터셋 준비 완료, 학습 준비됨
