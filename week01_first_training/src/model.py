"""timm 기반 모델 생성

설계 원칙:
    - timm을 통해 1000+ pretrained 모델을 한 줄로 사용 가능
    - in_chans=1 옵션으로 RGB pretrained → 그레이스케일 자동 변환
    - 백본 교체가 config 한 줄 수정으로 가능 (실험 비교 용이)

본 RTAI 프로젝트와의 관계:
    - 본 프로젝트도 timm을 backbone source로 사용
    - 모듈 4 (결함 분류)는 ViT/Swin/EfficientNet 비교 예정
    - in_chans=1 패턴은 RT 필름에 그대로 적용
"""
import timm
import torch.nn as nn

from src.data import get_medmnist_info


def build_model(cfg: dict) -> tuple[nn.Module, dict]:
    """timm으로 모델 생성
    
    Args:
        cfg: config dict (load_config로 로드한 결과)
    
    Returns:
        (model, info): 모델과 데이터셋 메타정보
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    
    # 데이터셋 메타정보로부터 클래스 수 자동 결정
    info = get_medmnist_info(data_cfg["dataset"])
    num_classes = info["n_classes"]
    
    # timm으로 모델 생성
    # 핵심 인자:
    #   - in_chans=1: 첫 conv layer를 1채널 입력으로 자동 변환
    #   - num_classes: 마지막 classifier head를 자동 교체
    #   - pretrained: ImageNet 사전학습 가중치 사용
    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        in_chans=model_cfg["in_chans"],
        num_classes=num_classes,
        drop_rate=model_cfg["drop_rate"],
    )
    
    return model, info


# ============================================================
# 디버깅용 — 모델 구조 + forward 동작 확인
# ============================================================
if __name__ == "__main__":
    """사용법: python -m src.model
    
    모델 생성 + 더미 입력으로 forward 동작 확인.
    출력 shape이 (B, num_classes)인지 검증.
    """
    import torch
    from src.utils import load_config, get_device, count_parameters
    
    cfg = load_config("configs/default.yaml")
    device = get_device()
    
    # 모델 생성
    model, info = build_model(cfg)
    model = model.to(device)
    
    print(f"\n[Model Info]")
    print(f"  name        = {cfg['model']['name']}")
    print(f"  pretrained  = {cfg['model']['pretrained']}")
    print(f"  in_chans    = {cfg['model']['in_chans']}")
    print(f"  num_classes = {info['n_classes']}")
    print(f"  parameters  = {count_parameters(model):,}")
    
    # 더미 입력으로 forward 테스트
    # 실제 데이터와 동일한 shape: (B, 1, H, W)
    B, C, H, W = 4, cfg['model']['in_chans'], cfg['data']['size'], cfg['data']['size']
    dummy = torch.randn(B, C, H, W, device=device)
    
    print(f"\n[Forward Test]")
    print(f"  Input shape  = {tuple(dummy.shape)}")
    
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    
    print(f"  Output shape = {tuple(out.shape)}")
    print(f"  Output range = [{out.min():.3f}, {out.max():.3f}]")
    print(f"  Expected     = ({B}, {info['n_classes']})")
    
    # 검증
    assert out.shape == (B, info["n_classes"]), \
        f"❌ Output shape mismatch: got {tuple(out.shape)}"
    print(f"\n✅ Forward pass OK")