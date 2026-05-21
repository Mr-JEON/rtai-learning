# `train.py` 심층 분석 — Week 01 학습용

> 이 문서는 `week01_first_training/src/train.py`를 **줄단위로 훑는 게 아니라**,
> "왜 이렇게 짰는가", "다른 선택지는 무엇인가", "직접 망가뜨려 보면 무엇이 보이는가"를
> 학습하기 위한 워크스루다. 원본 파일을 옆에 띄워놓고 줄번호를 따라가며 읽으면 좋다.

---

## 0. 학습 목표

이 파일을 읽고 나면 다음을 **남에게 설명**할 수 있어야 한다.

1. AMP(Automatic Mixed Precision)의 forward/backward 순서와 GradScaler가 하는 일
2. Gradient clipping을 AMP와 결합할 때 `scaler.unscale_()`이 왜 먼저 와야 하는가
3. Warmup + cosine annealing scheduler의 수식과 모양
4. Binary vs multi-class vs multi-label task에서 AUC를 계산하는 세 가지 분기
5. `model.train()` / `model.eval()` / `@torch.no_grad()` / `set_to_none=True`의 의미
6. Sanity overfit 테스트가 무엇을 잡아내고 무엇을 못 잡아내는가
7. Best checkpoint 저장 패턴과 재현성을 위해 dict에 넣는 항목들

---

## 1. 파일 전체 골격

`train.py`의 함수 6개 + 진입점:

| 함수 | 줄번호 | 역할 |
|---|---|---|
| `sanity_overfit` | `34-62` | 1배치 N step 학습으로 데이터/모델/loss 무결성 검증 |
| `train_one_epoch` | `68-108` | AMP forward + backward + clip + step + 로깅 |
| `evaluate` | `114-152` | val/test 평가, task별 AUC/ACC 계산 |
| `make_scheduler` | `158-170` | LambdaLR로 warmup + cosine annealing 구현 |
| `main` | `176-303` | 전체 오케스트레이션 |
| `__main__` | `306-310` | CLI argparse |

**의존성 그래프**:
```
main
 ├── load_config / set_seed / get_device / make_output_dir  (utils)
 ├── build_dataloaders  (data)
 ├── build_model  (model)
 ├── sanity_overfit
 │    └── train_one_epoch loop 안에서도 동일 패턴 사용
 ├── train_one_epoch
 ├── evaluate
 └── make_scheduler
```

읽는 순서 추천: `make_scheduler` (가장 쉬움) → `sanity_overfit` → `evaluate` → `train_one_epoch` → `main`.

---

## 2. `make_scheduler` — Warmup + Cosine (line 158-170)

가장 짧지만 학습 곡선의 모양을 결정하는 함수.

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(progress * math.pi))
```

### 동작 분석

`LambdaLR`은 매 `scheduler.step()`마다 `lr = base_lr * lr_lambda(epoch)`로 lr을 재설정한다.

- **Warmup 구간** (`epoch < warmup_epochs`):
  - `epoch=0` → `lr_lambda = 1/W`
  - `epoch=W-1` → `lr_lambda = 1.0` (target lr 도달)
  - 선형 증가. 큰 batch 학습 초기의 gradient noise를 부드럽게 한다.
- **Cosine 구간** (`epoch >= warmup_epochs`):
  - `progress=0` → `cos(0) = 1` → `lr_lambda = 1.0`
  - `progress=1` → `cos(π) = -1` → `lr_lambda = 0.0`
  - 0.5 * (1 + cos) 는 1에서 시작해서 0까지 부드럽게 감쇠.

### 왜 timm.scheduler를 안 썼는가?

`timm`에는 `CosineLRScheduler`가 있어서 한 줄이면 끝나지만, **1주차 목표가 "내부 동작 이해"**이기 때문에 LambdaLR로 수식을 직접 작성. 본 RTAI 프로젝트로 넘어갈 때는 `timm.scheduler.CosineLRScheduler`로 교체해도 무방.

### 함정

- `scheduler.step()`은 **epoch 단위로** 호출되고 있다 (`train.py:259`). step 단위로 부르려면 `lr_lambda`의 인자 의미가 바뀐다.
- `max(1, ...)`로 0 나눗셈을 막아둔 게 보임. `epochs == warmup_epochs`인 극단 케이스 방어.

### 실험 거리
- `epochs=10, warmup=2`일 때 매 epoch의 lr 값을 손으로 계산해보고 `train/lr` 텐서보드 곡선과 일치하는지 확인.
- warmup 없이(0) 학습 vs warmup=3으로 학습 — train_loss 초반 곡선이 어떻게 달라지나?

---

## 3. `sanity_overfit` — 1배치 overfit 테스트 (line 34-62)

```python
imgs, labels = next(iter(loader))   # 1 배치만
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(steps):
    opt.zero_grad()
    loss = criterion(model(imgs), labels)
    loss.backward(); opt.step()
```

### 이 테스트가 잡아내는 것

- **데이터 → 모델 → loss 경로의 shape/dtype 오류** (forward가 터지면 즉시 stop)
- **loss 함수가 task에 맞지 않음** (예: BCEWithLogitsLoss에 long 라벨 넣으면 에러)
- **lr이 너무 작아 학습 자체가 안 일어남** (loss가 안 떨어짐)
- **모델이 입력에 의존하지 않음** (예: in_chans 잘못 → 어떤 입력에도 같은 출력)
- **라벨이 무작위로 섞임** (1배치도 overfit 안 됨)

### 이 테스트가 못 잡는 것

- **train/val split이 섞여 있음** (data leak)
- **augmentation이 너무 강해서 train만 망가짐** (1배치는 augmentation이 매번 다르게 적용되지만 본 학습보다는 표본이 적음)
- **lr이 본 학습에 비해 너무 큼** (1배치는 워낙 쉬워서 통과)
- **scheduler 버그** (이 함수 안에선 scheduler 안 씀)

### 함정

- 이 함수는 **모델 가중치를 변경한다**. 그래서 `main`에서 sanity 후에 모델/optimizer/scheduler/scaler를 전부 재생성한다 (`train.py:228-241`).
- 더 깔끔한 방법: sanity 직전에 `state_dict = copy.deepcopy(model.state_dict())` 떠놓고 끝나면 `load_state_dict`. 1주차 회고에 적어둘 만한 개선 포인트.
- `next(iter(loader))`는 매번 같은 배치를 보장하지 않는다 (shuffle 때문). 같은 배치를 고정하려면 `torch.manual_seed`로 통제하거나 인덱스 고정.

### 임계값 `losses[-1] >= losses[0] * 0.5` 의미

50 step 만에 loss가 절반 미만으로 떨어져야 통과. 너무 후한 기준이라 통과해도 안심하면 안 됨 — 1배치는 보통 1/10 미만까지 떨어져야 정상.

### 실험 거리
- lr을 `1e-7`로 바꾸고 돌려보기 → 의도적으로 실패시켜서 메시지 형태 확인.
- model에 `nn.init.zeros_`로 weight 초기화 후 돌려보기 → 학습이 일어나는지?
- 라벨을 `torch.randint(0, 2, ...)`로 무작위로 바꾸고 돌려보기 → overfit이 가능해야 정상(파라미터가 충분히 많다면).

---

## 4. `train_one_epoch` — AMP + GradScaler + Clip (line 68-108)

이 함수가 PyTorch 학습 루프의 **표준 패턴 6단계**를 다 담고 있다. 한 step의 골격:

```python
optimizer.zero_grad(set_to_none=True)                # (1) grad 초기화
with autocast(device_type="cuda", enabled=...):      # (2) forward in fp16
    logits = model(imgs)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()                        # (3) backward (scaled loss)
if grad_clip > 0:
    scaler.unscale_(optimizer)                       # (4) grad unscale 먼저
    torch.nn.utils.clip_grad_norm_(...)              #     그 다음 clip
scaler.step(optimizer)                               # (5) optimizer step
scaler.update()                                      # (6) scale factor 갱신
```

### 각 단계의 의미

**(1) `set_to_none=True`**
- 디폴트(`zero_grad()`)는 grad 텐서를 0으로 덮어쓴다.
- `set_to_none=True`는 grad를 `None`으로 만든다 — 메모리 할당 자체를 줄임.
- PyTorch 2.0+에서는 디폴트가 `True`로 바뀌었지만, 명시하는 편이 의도가 분명.

**(2) `autocast`**
- 컨텍스트 안의 연산을 fp16(또는 bf16)로 자동 변환.
- conv/matmul은 fp16, loss 계산은 fp32로 자동 분리 (정밀도 손실 방어).
- `enabled=scaler.is_enabled()`로 묶어서 AMP off일 때 no-op.

**(3) `scaler.scale(loss).backward()`**
- fp16 gradient는 underflow(너무 작아서 0이 됨) 위험이 있음.
- loss에 큰 상수(예: 65536)를 곱해서 gradient를 정상 범위로 끌어올림.
- backward 결과 gradient는 그 상수배만큼 부풀어 있는 상태.

**(4) `scaler.unscale_(optimizer)` → `clip_grad_norm_`**
- 핵심 포인트. **gradient가 부풀어 있는 채로 clip하면 의미 없는 값**으로 자름.
- `unscale_`로 원래 크기로 되돌린 후 clip → 의도한 임계값으로 자름.
- `unscale_`는 같은 optimizer에 대해 step 전에 한 번만 호출 가능 (안전장치 있음).

**(5) `scaler.step(optimizer)`**
- 내부에서 gradient에 inf/NaN 있는지 검사 → 있으면 이 step skip, scale factor 절반으로 감소.
- 없으면 정상 optimizer.step 호출 (이미 unscale 된 상태).

**(6) `scaler.update()`**
- 일정 step 동안 NaN 없었으면 scale factor를 2배로 증가 → 다음 step에서 더 적극적으로 사용.

### `non_blocking=True` (line 76-77)

`pin_memory=True`인 DataLoader에서 `.to(device, non_blocking=True)`로 보내면 CPU→GPU 전송이 **현재 stream과 비동기로** 진행된다. 다음 연산이 이 텐서를 사용할 때 자동 동기화. 학습 throughput에 의외로 큰 영향.

### running_loss 누적 패턴 (line 98)

```python
running_loss += loss.item() * imgs.size(0)
n_samples += imgs.size(0)
# epoch 끝나면: avg = running_loss / n_samples
```

왜 `imgs.size(0)`을 곱하나? — 마지막 배치가 batch_size보다 작을 수 있기 때문에 (drop_last=True여도 일반화된 패턴). 단순히 `loss.item()`만 누적하면 작은 배치가 동일 가중치로 들어가 평균이 왜곡됨.

### 실험 거리
- `use_amp=False`로 돌리고 throughput / 메모리 비교.
- `grad_clip=0`으로 끄고, 일부러 lr을 키워서 loss spike 발생시켜 보기. clip이 있을 때와 비교.
- `pin_memory=False`로 바꾸고 학습 시간 측정.

---

## 5. `evaluate` — Task별 메트릭 분기 (line 114-152)

세 가지 task 분기가 있다.

| Task | 출력 활성함수 | 메트릭 |
|---|---|---|
| `multi-label, binary-class` | sigmoid | macro AUC, threshold=0.5 accuracy |
| binary (n_classes=2) | softmax | AUC(positive class만), argmax accuracy |
| multi-class (n_classes>2) | softmax | AUC(ovr=one-vs-rest), argmax accuracy |

### `@torch.no_grad()` 데코레이터 (line 114)

함수 전체를 grad 비활성화 컨텍스트로 감싼다. inference 속도/메모리 모두 이득.
대안: 함수 안에서 `with torch.no_grad():` 블록. 데코레이터가 더 명확.

### `model.eval()` (line 117)

이게 빠지면 **버그**. BatchNorm과 Dropout이 train 모드 그대로 작동.
- BN: running mean/var 대신 mini-batch 통계 사용 → 평가 불안정
- Dropout: 평가 중에도 뉴런 무작위 제거 → 결과 비결정적

### CPU로 가져오는 이유 (line 131-132)

```python
all_logits.append(logits.cpu())
all_labels.append(labels.cpu())
```

`sklearn.metrics`는 numpy 입력만 받음. epoch 끝에 한 번 `.cat()` 후 `.numpy()`로 변환.
GPU에 그대로 누적하면 메모리 부족 + sklearn 호출 직전에 옮겨야 함.

### Binary AUC의 `probs[:, 1]` (line 146-147)

이진 분류에서 `softmax` 출력은 `[P(0), P(1)]` 형태.
`roc_auc_score`는 positive class의 확률만 받아야 정확. 그래서 `probs[:, 1]`.

### Multi-class AUC `multi_class="ovr"` (line 149)

- `ovr` (one-vs-rest): 각 클래스를 positive vs 나머지로 보고 AUC 계산 후 평균.
- `ovo` (one-vs-one): 모든 클래스 쌍에 대해 AUC 계산 — 시간 더 걸림.
- 클래스 불균형이 심하면 결과 차이가 크다. RTAI 본 프로젝트에서 결함 클래스가 unbalanced하면 재검토.

### 실험 거리
- `model.eval()` 빼고 evaluate 두 번 호출 → val_loss/auc가 매번 다르게 나오는지 확인.
- `@torch.no_grad()` 빼고 메모리 사용량 비교.
- ChestMNIST(multi-label)로 데이터셋 바꾸고 분기 자동 전환 확인.

---

## 6. `main` — 오케스트레이션 (line 176-303)

전체 흐름을 9단계로 나눠 볼 수 있다.

1. **Config / seed / device / out_dir / writer** (177-188): 환경 셋업
2. **DataLoader 빌드** (191): `build_dataloaders` 호출
3. **모델 빌드** (194-197): `build_model`로 timm 모델 + meta info
4. **Loss 선택** (200-203): task에 따라 BCE / CE 분기
5. **Optimizer / Scheduler / Scaler** (206-219): AdamW + warmup-cosine + GradScaler
6. **Sanity check** (222-241): overfit 테스트 → **모델 재생성**
7. **학습 루프** (251-289): epoch 반복 → train → eval → log → checkpoint
8. **Best checkpoint으로 test** (293-300): 최종 일반화 성능
9. **TensorBoard close** (302)

### Sanity 후 모델 재생성 (line 227-241)

```python
# 모델을 다시 초기 상태로 — sanity가 모델 가중치를 변경했으므로
model, _ = build_model(cfg)
model = model.to(device)
optimizer = torch.optim.AdamW(...)
scheduler = make_scheduler(...)
scaler = GradScaler(enabled=...)
```

**문제점**: optimizer/scheduler/scaler까지 재생성할 필요 없다 (모델 파라미터에 묶여있긴 하지만, 새 model의 파라미터로 다시 묶으면 됨). 더 큰 문제는 timm pretrained weight를 다시 다운로드하지는 않더라도 **다시 to(device)하고 GPU 메모리 한 번 더 잡는** 비용.

**개선안 A**: sanity 전 state_dict 백업.
```python
import copy
backup = copy.deepcopy(model.state_dict())
sanity_overfit(...)
model.load_state_dict(backup)
```

**개선안 B**: sanity가 모델을 안 건드리게 — 모델을 클론해서 사용.
```python
import copy
sanity_model = copy.deepcopy(model)
sanity_overfit(sanity_model, ...)
del sanity_model
```

둘 다 1주차 회고에 "개선 포인트"로 적어둘 만함.

### Checkpoint dict 구조 (line 282-288)

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "metrics": val_metrics,
    "config": cfg,
}, ckpt_path)
```

- `model_state`: weight만. `model` 객체 자체를 저장하면 클래스 정의가 바뀌면 로드 깨짐.
- `optimizer_state`: **학습 재개**할 때 필요 (momentum 등). 추론만 할 거면 불필요.
- `config`: 어떤 하이퍼파라미터로 학습했는지 동봉 → 모델 파일만 받아도 재현 가능.
- 빠진 것: `scheduler_state`, `scaler_state`, `rng_state`. 학습 재개를 진지하게 하려면 추가 필요.

### `torch.load(..., weights_only=False)` (line 293)

PyTorch 2.4+에서 디폴트가 `weights_only=True`로 바뀌었음. 우리는 dict에 cfg, metrics 등 텐서 외 객체도 들었기 때문에 `False`로 명시 (보안 trade-off: untrusted checkpoint은 절대 `False`로 로드하지 말 것).

### Best metric 선택 (line 246, 279)

`metric_name = cfg["logging"]["metric"]` — config에서 "auc" / "acc" / "loss" 등 선택 가능하게 했음.
주의: loss는 작을수록 좋으므로 `>` 비교가 잘못된다. 현재 코드는 auc/acc 가정. config에서 loss를 선택하면 best가 안 잡힘.

### 실험 거리
- `metric: loss`로 바꿔서 돌려보기 → best가 첫 epoch만 잡히는 버그 재현.
- scheduler를 빼고 fixed lr로 돌려보고 best AUC 비교.
- checkpoint에 `scheduler.state_dict()`, `scaler.state_dict()` 추가하고, 학습 중단 후 재개 기능 구현해보기 (1주차 이후 도전 과제).

---

## 7. 디자인 결정 요약 (cheat sheet)

| # | 결정 | 이유 |
|---|---|---|
| 1 | Lightning 안 씀 | 학습 사이클 손으로 짜는 게 1주차 목표 |
| 2 | AMP 사용 | 5090에서 fp16/bf16 native, 속도/메모리 모두 이득 |
| 3 | `set_to_none=True` | 메모리 절약 + 명시적 의도 표현 |
| 4 | grad clip + AMP 순서 (unscale → clip) | 부풀린 grad로 clip하면 무의미 |
| 5 | LambdaLR로 warmup+cosine 수기 작성 | timm 의존 줄이고 수식 이해 |
| 6 | Sanity overfit을 train 전에 강제 | 데이터/모델/loss 무결성 빠르게 검증 |
| 7 | Best checkpoint를 metric 기준으로 저장 | 마지막 epoch가 best 아닐 수 있음 |
| 8 | Test는 best checkpoint으로 평가 | val로 model selection, test로 일반화 측정 |
| 9 | `cudnn.benchmark=False` (utils) | 5090 + nightly에서 segfault 회피 |

---

## 8. 의도적으로 망가뜨려 보기 (가장 학습 효과 큰 부분)

각 실험에 대해 "어떤 신호가 보이면 정상/이상"을 미리 예측하고 돌릴 것. 예측이 맞으면 이해한 것이고, 틀리면 이해의 구멍이 있다는 신호.

- [ ] **AMP off** (`use_amp: false`) — throughput, 메모리, val_auc 비교
- [ ] **grad_clip=0** + `lr=1e-1` — loss spike / NaN 재현
- [ ] **augmentation off** (`augmentation: false`) — train_loss vs val_loss gap 관찰
- [ ] **`model.eval()` 제거** — val_loss/auc가 매번 달라지는지 (BN/Dropout 영향)
- [ ] **scheduler 제거** (고정 lr) — best epoch 위치 / 곡선 모양 변화
- [ ] **`warmup_epochs=0`** — 초반 train_loss 진동 확인
- [ ] **백본 교체**: `resnet18` → `efficientnet_b0` → `vit_tiny_patch16_224`
  - vit는 28x28 PneumoniaMNIST에 패치 사이즈 안 맞음 → `size: 224`로 키워야 함
- [ ] **sanity_overfit에서 lr=1e-7** — 의도적 실패 메시지 확인
- [ ] **데이터셋 교체**: PneumoniaMNIST → ChestMNIST (multi-label로 task 분기 자동 전환되는지)

---

## 9. 자주 빠지는 함정

- `model.eval()` 안 부르고 evaluate
- `optimizer.zero_grad()` 안 부르고 학습 → gradient 누적되어 폭주
- AMP grad clip 순서 (`unscale_` 후 `clip`) — 위에 자세히
- DataLoader `num_workers > 0` + Albumentations 조합에서 fork 충돌 (특히 macOS) — Linux는 보통 괜찮음
- `roc_auc_score`에 single-class 라벨만 들어가면 에러 (val set이 너무 작거나 1클래스만 들어간 경우)
- `tqdm`의 `pbar.set_postfix` 안의 평균 loss는 **누적 평균**, iter loss와 다름
- checkpoint 저장 시 `model.state_dict()`만 저장하면 학습 재개 불가
- `torch.load`의 `weights_only` 디폴트 변경 (2.4+)

---

## 10. 다음 단계

이 워크스루를 다 소화했다면:

1. 위 "망가뜨려 보기" 체크리스트 절반 이상 실행 + TensorBoard 곡선 비교 캡처
2. Sanity overfit 개선안(개선안 A/B) 둘 중 하나로 리팩토링
3. 학습 재개 기능 (`scheduler_state`, `scaler_state` 저장 + `--resume` 옵션) 추가
4. Week 02로 넘어가기 (segmentation으로 task 전환)

---

> **참고**: 이 문서는 `train.py`의 학습 흐름을 빠르게 기억해내는 용도. 실제 코드 수정 시에는
> 원본을 다시 읽고 줄번호를 재확인할 것 (코드가 바뀌면 줄번호는 어긋난다).
