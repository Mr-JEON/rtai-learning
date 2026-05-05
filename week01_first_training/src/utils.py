"""공통 유틸리티: seed 고정, config 로드, device 선택, 출력 디렉토리 관리"""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 seed 고정.
    
    주의: deterministic=True는 속도 저하를 유발하므로 1주차에선 미사용.
    본 프로젝트에서 정밀 재현이 필요할 때만 활성화.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # 필요 시 활성화

    # cudnn.benchmark 주의:
    # PyTorch nightly 2.12 + 5090 (sm_120) + 가변 batch size 조합에서
    # cudnn이 잘못된 커널을 선택하여 segfault 발생 (Week 01 디버깅).
    # benchmark=False가 안전. stable PyTorch 출시 시 재검토.
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """YAML config 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """GPU 가용성 확인 후 device 반환"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"[GPU] {gpu_name} (Compute Capability {cc[0]}.{cc[1]})")
        return device
    print("[GPU] CUDA 사용 불가, CPU로 진행")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """학습 가능 파라미터 수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_output_dir(base: str, exp_name: str) -> Path:
    """outputs/{exp_name}/ 디렉토리 생성.
    
    구조:
        outputs/{exp_name}/
        ├── checkpoints/  ← 모델 가중치
        └── tb/           ← TensorBoard 로그
    """
    out = Path(base) / exp_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "tb").mkdir(exist_ok=True)
    return out


# ============================================================
# 디버깅용 — 단독 실행 시 동작 확인
# ============================================================
if __name__ == "__main__":
    """사용법: python -m src.utils"""
    set_seed(42)
    print("Seed set to 42")
    
    device = get_device()
    print(f"Device: {device}")
    
    cfg = load_config("configs/default.yaml")
    print(f"Loaded config: experiment.name = {cfg['experiment']['name']}")
    
    out = make_output_dir("outputs", "test_dir")
    print(f"Output dir: {out}")
    print(f"  checkpoints: {(out / 'checkpoints').exists()}")
    print(f"  tb: {(out / 'tb').exists()}")