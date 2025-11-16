# VOC + SBD 데이터셋 결합 계획

## 목표

PASCAL VOC 2012 (Kaggle)와 SBD를 결합하여 SOTA 표준 학습 세트 (10,582장) 생성

## 데이터셋 구성

### 현재 다운로드된 데이터

1. **PASCAL VOC 2012 (Kaggle)**
   - 경로: `/root/.cache/kagglehub/datasets/sovitrath/voc-2012-segmentation-data/versions/1/voc_2012_segmentation_data/`
   - Train: 1,464장 (`train_images/`, `train_labels/`)
   - Valid: 1,449장 (`valid_images/`, `valid_labels/`)

2. **SBD (Semantic Boundaries Dataset)**
   - 경로: `./data/benchmark_RELEASE/dataset/`
   - Images: 8,818장 (`img/`)
   - Labels: 11,355장 (`cls/*.mat`)
   - Train split: 8,498장 (`train.txt`)

## 결합 방법

### Option 1: ConcatDataset (추천)

```python
from torch.utils.data import ConcatDataset
from data.voc_kaggle import VOCSegmentationKaggle
from torchvision.datasets import SBDataset

# VOC train
voc_train = VOCSegmentationKaggle(
    root='/root/.cache/kagglehub/...',
    split='train',
    transform=transform
)

# SBD train_noval (VOC 2012 val 제외)
sbd_train = SBDataset(
    root='./data',
    image_set='train_noval',
    mode='segmentation',
    transform=transform
)

# 결합
combined = ConcatDataset([voc_train, sbd_train])
# Total: 1,464 + 9,118 = 10,582
```

### Option 2: 커스텀 Dataset 클래스

`data/voc_sbd_combined.py` 구현 필요

## 예상 결과

- **Train**: 10,582장 (VOC 1,464 + SBD 9,118)
- **Val**: 1,449장 (VOC valid)
- **학습 시간**: 기존 대비 7.2배 증가

## 주의사항

1. **Split 겹침 방지**: SBD는 반드시 `train_noval` 사용
2. **Transform 일관성**: 두 데이터셋에 동일한 transform 적용
3. **Label 형식**: VOC와 SBD 모두 동일한 21 classes 사용
