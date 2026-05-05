"""MedMNIST 데이터로더 + Albumentations augmentation

설계 원칙:
    1. MedMNIST 라이브러리의 기본 Dataset 대신 직접 wrapping
       → torchvision transform 의존성을 끊고 Albumentations 사용
    2. Train/Val transform 명확히 분리
       → val에는 augmentation 적용 안 함 (당연한 것 같지만 자주 실수함)
    3. RTAI 본 프로젝트와 동일한 스택
       → Albumentations + 그레이스케일 + ImageNet 평균/표준편차 X
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import medmnist
from medmnist import INFO


# ============================================================
# MedMNIST 메타정보 조회
# ============================================================
def get_medmnist_info(dataset_name: str) -> dict:
    """MedMNIST 데이터셋의 task 종류, 채널 수, 클래스 수 조회.
    
    INFO는 medmnist 라이브러리에 내장된 dict로, 각 데이터셋의
    'task'(binary/multi-class/multi-label), 'n_channels', 'label' 등을 제공.
    """
    info = INFO[dataset_name]
    return {
        "task": info["task"],            # "binary-class" / "multi-label, binary-class" / etc.
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "labels": info["label"],
    }


# ============================================================
# 커스텀 Dataset
# ============================================================
class MedMNISTDataset(Dataset):
    """MedMNIST를 Albumentations와 호환되게 wrapping
    
    왜 직접 만드는가:
        - medmnist 라이브러리의 기본 Dataset은 torchvision transform 전제
        - 우리는 Albumentations를 쓰고 싶음 (RTAI 본 프로젝트와 동일 스택)
        - 직접 구현으로 데이터 흐름을 명확히 이해
    """
    def __init__(self, dataset_name: str, split: str, size: int, transform=None):
        # medmnist의 동적 클래스 로딩 (예: PneumoniaMNIST 클래스 가져오기)
        DataClass = getattr(medmnist, INFO[dataset_name]["python_class"])
        self.ds = DataClass(split=split, download=True, size=size)
        self.transform = transform
        self.task = INFO[dataset_name]["task"]
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        # MedMNIST는 (PIL Image, label) 반환
        img, label = self.ds[idx]
        
        # PIL → numpy (Albumentations는 numpy 입력)
        img = np.array(img)
        
        # 그레이스케일이면 (H, W) → (H, W, 1) 명시
        # Albumentations는 (H, W, C) 형태를 기대함
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        
        # Augmentation 적용
        if self.transform:
            img = self.transform(image=img)["image"]
        
        # Label 처리: task별로 다름
        if self.task == "multi-label, binary-class":
            # ChestMNIST 등: 라벨이 [0, 1, 0, 1, ...] 같은 multi-hot
            label = torch.tensor(label, dtype=torch.float32)
        else:
            # PneumoniaMNIST 등: 라벨이 [0] 또는 [1] 같은 단일 클래스
            # squeeze로 (1,) → 스칼라 변환
            label = torch.tensor(label, dtype=torch.long).squeeze()
        
        return img, label


# ============================================================
# Augmentation 파이프라인
# ============================================================
def build_transforms(size: int, train: bool, use_aug: bool = True):
    """Train/Val transform 분리
    
    1주차 핵심 원칙:
        - Train과 Val의 augmentation 명확히 분리 (val에는 적용 X)
        - 그레이스케일 정규화: ImageNet mean/std 대신 0.5/0.5 사용
          (1채널이라 ImageNet 통계가 의미 없음)
    
    RTAI 본 프로젝트 적용 시 주의:
        - HorizontalFlip은 RT 필름에 적용 가능한지 도메인 검토 필요
          (용접부 비대칭이면 부적절)
        - 명암 조정(BrightnessContrast)은 필름 노출 변동을 모사하므로 유용
    """
    if train and use_aug:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.Normalize(mean=[0.5], std=[0.5]),  # 1채널용
            ToTensorV2(),                          # numpy → torch tensor
        ])
    else:
        # Val/Test: augmentation 없이 정규화만
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])


# ============================================================
# DataLoader 생성
# ============================================================
def build_dataloaders(cfg: dict):
    """Config 기반 train/val/test DataLoader 생성"""
    name = cfg["data"]["dataset"]
    size = cfg["data"]["size"]
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    use_aug = cfg["data"]["augmentation"]
    
    # Transform 생성 (train/val 분리)
    train_tf = build_transforms(size, train=True, use_aug=use_aug)
    val_tf = build_transforms(size, train=False)
    
    # Dataset 생성
    train_ds = MedMNISTDataset(name, "train", size, train_tf)
    val_ds = MedMNISTDataset(name, "val", size, val_tf)
    test_ds = MedMNISTDataset(name, "test", size, val_tf)
    
    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # DataLoader: train만 shuffle=True
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,           # GPU 전송 속도 ↑
        drop_last=True,            # 마지막 불완전 batch 버림 (BN 안정성)
        persistent_workers=(nw > 0),  # epoch 사이에 worker 재사용 (속도 ↑)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# 디버깅용 — 단독 실행 시 한 batch 시각화
# ============================================================
if __name__ == "__main__":
    """사용법: python -m src.data
    
    한 배치를 뽑아서 shape, 값 범위, 라벨 분포를 출력.
    학습 시작 전 데이터 흐름을 검증하는 핵심 단계.
    """
    from src.utils import load_config
    
    cfg = load_config("configs/default.yaml")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    
    # 한 batch 추출
    imgs, labels = next(iter(train_loader))
    
    print(f"\n[Sanity Check]")
    print(f"  imgs.shape   = {imgs.shape}")     # (B, C, H, W) 기대
    print(f"  imgs.dtype   = {imgs.dtype}")
    print(f"  imgs range   = [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  imgs mean    = {imgs.mean():.3f}")
    print(f"  imgs std     = {imgs.std():.3f}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  labels.dtype = {labels.dtype}")
    print(f"  labels[:10]  = {labels[:10]}")
    
    # 메타정보
    info = get_medmnist_info(cfg["data"]["dataset"])
    print(f"\n[Dataset Info]")
    print(f"  task    = {info['task']}")
    print(f"  classes = {info['n_classes']} ({list(info['labels'].values())})")