"""학습 루프 — Raw PyTorch (Lightning 등 wrapper 사용 안 함)

설계 원칙:
    - 모든 단계를 명시적으로: forward → loss → backward → step → eval
    - AMP, Scheduler, Checkpoint 모두 직접 처리
    - RTAI 본 프로젝트에서 그대로 확장 가능한 구조
    
1주차 학습 목적:
    - 학습 사이클을 손으로 짜보면서 PyTorch 내부 동작 이해
    - sanity overfit 패턴 익히기
    - TensorBoard 곡선 보고 학습 상태 판단하는 감각 키우기
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data import build_dataloaders
from src.model import build_model
from src.utils import (
    set_seed, load_config, get_device, count_parameters, make_output_dir
)


# ============================================================
# Sanity check: 1-batch overfit
# ============================================================
def sanity_overfit(model, loader, criterion, device, task, steps=50):
    """단일 batch로 빠르게 학습시켜 loss가 떨어지는지 확인.
    
    이 테스트가 실패하면 본 학습은 절대 성공하지 않음.
    데이터/모델/loss 중 하나가 잘못된 신호.
    
    1주차의 가장 중요한 디버깅 패턴.
    """
    model.train()  # 모델을 학습 모드로 전환 (dropout/batchnorm 등 학습 동작 활성화)
    
    # 전체 loader에서 첫 batch만 추출 — 1-batch overfit에 사용
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)      # 이미지 텐서를 GPU로 이동
    labels = labels.to(device)  # 라벨 텐서도 GPU로 이동
    
    # 간단한 AdamW 옵티마이저 생성 (sanity 체크용, learning rate 고정)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []  # epoch마다 loss 기록 리스트

    for i in range(steps):
        opt.zero_grad()  # 이전 gradient를 0으로 초기화 (필수)
        
        logits = model(imgs)  # 모델 forward pass (output shape: [batch, ...])
        loss = criterion(logits, labels)  # 손실 함수 계산 (예: CrossEntropy 등)
        
        loss.backward()  # gradient 계산 (역전파)
        opt.step()       # 가중치 업데이트
        
        losses.append(loss.item())  # 현재 step의 손실 값을 리스트에 저장

    # 초기 loss와 마지막 loss 비교 출력
    print(f"  loss[0]={losses[0]:.4f} → loss[-1]={losses[-1]:.4f}")
    
    # loss가 충분히 감소(최소 절반 이하)하지 않으면 데이터/모델/loss에 문제 있음
    if losses[-1] >= losses[0] * 0.5:
        print("  ⚠️  loss가 충분히 떨어지지 않음 — 데이터/모델/loss 점검 필요")
        return False

    print("  ✅ Sanity OK (1-batch에 정상 overfit)")
    return True


# ============================================================
# 한 epoch 학습
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, writer, log_every, grad_clip, task):
    model.train()
    running_loss = 0.0
    n_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for it, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision으로 forward + loss
        with autocast(device_type="cuda", enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        # Backward (AMP 호환)
        scaler.scale(loss).backward()
        
        # Gradient clipping (학습 안정화)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 통계 누적
        running_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
        
        # 로깅
        if (it + 1) % log_every == 0:
            avg = running_loss / n_samples
            pbar.set_postfix(loss=f"{avg:.4f}")
            global_step = epoch * len(loader) + it
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
    
    return running_loss / n_samples


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, task):
    """전체 validation/test set에 대해 평가."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_logits, all_labels = [], []
    
    for imgs, labels in tqdm(loader, desc="[Val]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = running_loss / n_samples
    
    if task == "multi-label, binary-class":
        probs = torch.sigmoid(all_logits).numpy()
        auc = roc_auc_score(all_labels.numpy(), probs, average="macro")
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(all_labels.numpy().flatten(), preds.flatten())
    else:
        probs = torch.softmax(all_logits, dim=1).numpy()
        preds = all_logits.argmax(dim=1).numpy()
        if probs.shape[1] == 2:
            auc = roc_auc_score(all_labels.numpy(), probs[:, 1])
        else:
            auc = roc_auc_score(all_labels.numpy(), probs, multi_class="ovr")
        acc = accuracy_score(all_labels.numpy(), preds)
    
    return {"loss": avg_loss, "auc": auc, "acc": acc}


# ============================================================
# LR scheduler with warmup
# ============================================================
def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """Warmup + Cosine annealing scheduler.
    
    Warmup: 첫 N epoch에서 lr을 0 → target으로 점진적 증가
    Cosine: 이후 lr을 cosine 곡선으로 0까지 감쇠
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(progress * math.pi))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 메인
# ============================================================
def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    
    # 출력 디렉토리
    out_dir = make_output_dir(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"]
    )
    print(f"[Output] {out_dir}")
    
    writer = SummaryWriter(out_dir / "tb")
    
    # ----- 데이터 -----
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    
    # ----- 모델 -----
    model, info = build_model(cfg)
    model = model.to(device)
    print(f"[Model] {cfg['model']['name']}, params={count_parameters(model):,}")
    print(f"[Task]  {info['task']}, classes={info['n_classes']}")
    
    # ----- Loss -----
    if info["task"] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ----- Optimizer & Scheduler -----
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=train_cfg["epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
    )
    
    # AMP scaler (use_amp=False면 no-op처럼 동작)
    scaler = GradScaler(enabled=train_cfg["use_amp"])
    
    # ----- Sanity Check (1주차 핵심 디버깅 패턴) -----
    print("\n[Sanity] 1-batch overfit test...")
    if not sanity_overfit(model, train_loader, criterion, device, info["task"]):
        print("⚠️  Sanity check 실패. 본 학습 진행 전 점검 필요.")
        # 일단 진행하되 경고만 띄움
    
    # 모델을 다시 초기 상태로 — sanity가 모델 가중치를 변경했으므로
    print("\n[Reset] Sanity check 후 모델 재생성...")
    model, _ = build_model(cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=train_cfg["epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
    )
    scaler = GradScaler(enabled=train_cfg["use_amp"])
    
    # ----- 학습 루프 -----
    best_metric = -float("inf")
    log_every = cfg["logging"]["log_every"]
    metric_name = cfg["logging"]["metric"]
    epochs = train_cfg["epochs"]
    
    print(f"\n[Training] {epochs} epochs 시작...")
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, writer, log_every, train_cfg["grad_clip"],
            info["task"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device, info["task"])
        
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"lr={cur_lr:.2e}"
        )
        
        # TensorBoard
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("train/lr", cur_lr, epoch)
        
        # Best checkpoint 저장
        if val_metrics[metric_name] > best_metric:
            best_metric = val_metrics[metric_name]
            ckpt_path = out_dir / "checkpoints" / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": val_metrics,
                "config": cfg,
            }, ckpt_path)
            print(f"  → best {metric_name}={best_metric:.4f} saved")
    
    # ----- Test (best checkpoint으로) -----
    print("\n[Test] best checkpoint으로 평가...")
    ckpt = torch.load(out_dir / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, info["task"])
    print(
        f"[Test] loss={test_metrics['loss']:.4f}, "
        f"auc={test_metrics['auc']:.4f}, "
        f"acc={test_metrics['acc']:.4f}"
    )
    
    writer.close()
    print(f"\n✅ 학습 완료. 결과: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)