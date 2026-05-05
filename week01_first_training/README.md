# Week 01: First Training Pipeline

산업 검사 분야 적용을 목표로 한 첫 PyTorch 학습 파이프라인 구축.
MedMNIST의 PneumoniaMNIST를 사용해 그레이스케일 의료영상에 ResNet50 fine-tuning.

## 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | PneumoniaMNIST (4,708 train / 524 val / 624 test) |
| 모델 | ResNet50 (timm, ImageNet pretrained, in_chans=1) |
| 학습 시간 | 6초 (10 epochs, RTX 5090, AMP on) |
| Best Val AUC | 0.9988 (epoch 8) |
| Best Val Acc | 0.9847 |
| Test AUC | 0.9833 |
| Test Acc | 0.9167 |

## 학습 내용

### 코드 구조
- `configs/default.yaml` — 모든 하이퍼파라미터 분리
- `src/utils.py` — 재사용 헬퍼 (seed, device, output dir)
- `src/data.py` — Albumentations 기반 DataLoader
- `src/model.py` — timm wrapper (in_chans=1로 그레이스케일 자동 변환)
- `src/train.py` — 학습 루프 (AMP + cosine schedule + sanity check)

### 주요 결정
- timm을 backbone source로 (Apache 2.0)
- Albumentations로 augmentation (RT 필름과 동일 스택)
- 1채널 정규화: mean=0.5, std=0.5 (ImageNet 통계 대신)
- AdamW + cosine schedule + warmup_epochs=1
- AMP 사용 (FP16 텐서 코어 활용)

### 핵심 패턴
- **`in_chans=1` trick**: timm이 첫 conv layer 가중치를 채널 평균으로 자동 변환.
  ImageNet pretrained 효과를 그레이스케일에 그대로 가져옴.
- **Sanity check**: 본 학습 전 1-batch overfit 테스트로 데이터/모델/loss 검증.
- **Best checkpoint 자동 저장**: val AUC 기준으로 best.pt 갱신.

## TensorBoard 분석에서 얻은 인사이트

1. **Pretrained의 위력**: epoch 0부터 AUC 0.98+. ImageNet 특징이 의료영상에도
   거의 그대로 transfer됨.
2. **AMP의 trade-off**: 학습 중 약간의 노이즈 (val loss peak at epoch 4) 발생하나,
   최종 성능은 동등 또는 약간 우수.
3. **Cosine scheduler의 자동 안정화**: epoch 4의 overfit 신호를 lr decay가
   자동으로 보정. 사람 개입 불필요.
4. **Val vs Test gap**: PneumoniaMNIST는 test가 val보다 어려운 분포.
   본 프로젝트에서는 split 설계 신중히 해야 함.

## 환경 디버깅 경험 (값진 학습)

오늘 6번의 segfault를 만나며 ML 환경 셋업의 어려움을 직접 경험.

### 발견한 ABI 충돌
| 라이브러리 | 문제 버전 | 해결 |
|---|---|---|
| pandas | 3.0.2 | 2.3.x로 다운그레이드 |
| scipy | 1.17.1 | 1.15.x로 다운그레이드 (sklearn import 시 segfault) |
| numpy | 2.4.4 | 1.26.x로 다운그레이드 (PyTorch nightly 호환성) |
| albumentations | 2.0.8 | 1.4.x (OpenCV 4.13과 stack smashing) |
| opencv | 4.13.0.92 | 4.10.x로 |
| cudnn.benchmark | True | False (5090 + 가변 batch에서 segfault) |

### 핵심 교훈
- `pip check`는 ABI 충돌을 못 잡음. import/실행 시점에만 드러남
- "최신이 최선이 아니다" — 검증된 보수적 버전 명시가 안전
- Sanity check + 디버그 print + `-u` 플래그가 segfault 진단의 핵심
- PyTorch nightly + 5090 + 매우 최신 라이브러리는 검증 부족 영역

## 본 RTAI 프로젝트로의 적용

이 1주차 코드는 **본 프로젝트에 거의 그대로 가져갈 수 있는 골격**.

| 패턴 | 본 프로젝트 적용 |
|---|---|
| `in_chans=1` 그레이스케일 | RT 필름 16bit grayscale에 그대로 |
| Albumentations 기반 augmentation | 동일 (HorizontalFlip은 도메인 검토 필요) |
| timm + cosine schedule + AMP | 동일 |
| Sanity check 패턴 | 모든 모듈(ROI, IQI, defect seg)에 적용 |
| Config-driven 학습 | 모듈별 config 분리 |
| 검증된 라이브러리 버전 | requirements.txt 그대로 재사용 |

## 다음 주 (Week 02)

- 백본 비교: ResNet50 vs EfficientNet vs Swin
- 더 큰 데이터셋 (size: 64 → 224)
- Multi-label classification (ChestMNIST)

## 명령어 메모

```bash
# 환경 활성화
conda activate rtai-learning

# 학습 실행
python -u -m src.train --config configs/default.yaml

# TensorBoard
tensorboard --logdir outputs --port 6006
# 브라우저: http://localhost:6006

# Sanity check (데이터로더만)
python -m src.data
```