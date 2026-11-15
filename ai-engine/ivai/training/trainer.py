"""Training / evaluation loops with AMP, LR scheduling, early stopping and checkpointing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ivai.training.callbacks import EarlyStopping, ModelCheckpoint


@dataclass
class TrainConfig:
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-6
    early_stopping_patience: int = 8
    min_delta: float = 0.0
    amp: bool = True
    # Phase 3: learning rate of the classification head when it differs from ``lr``
    # (fine-tuning: backbone at ``lr``, head at ``head_lr``). The groups themselves are
    # built by ``ivai.models.registry.parameter_groups`` and passed to ``fit``.
    head_lr: float | None = None


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    probs: np.ndarray = field(repr=False)


def _amp_enabled(device: torch.device, amp: bool) -> bool:
    return amp and device.type == "cuda"


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp_enabled(device, amp)):
            logits = model(x)
            loss = criterion(logits, y)
        # GradScaler multiplies the loss to avoid fp16 gradient underflow, then unscales.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> EpochResult:
    model.eval()
    total_loss, n = 0.0, 0
    ys, preds, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp_enabled(device, amp)):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        p = torch.softmax(logits.float(), dim=1)
        ys.append(y.cpu())
        preds.append(p.argmax(1).cpu())
        probs.append(p.cpu())
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return EpochResult(total_loss / n, float((y_true == y_pred).mean()), y_true, y_pred, torch.cat(probs).numpy())


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: Path,
    log=print,
    param_groups: list[dict] | None = None,
) -> dict:
    """Train with validation-loss model selection. Returns the history and timings.

    ``param_groups=None`` is the historical Phase 2 path: ``Adam(model.parameters())``
    with a single learning rate. Otherwise the given groups (each with its own ``lr``)
    are used; ReduceLROnPlateau then scales every group by the same factor.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if param_groups is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=_amp_enabled(device, cfg.amp))
    stopper = EarlyStopping(cfg.early_stopping_patience, cfg.min_delta)
    ckpt = ModelCheckpoint(checkpoint_path)

    history, stopped_early = [], False
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        group_lrs = {f"lr_{g.get('name', i)}": g["lr"] for i, g in enumerate(optimizer.param_groups)}             if len(optimizer.param_groups) > 1 else {}
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg.amp)
        val = evaluate(model, val_loader, criterion, device, cfg.amp)
        scheduler.step(val.loss)
        improved = ckpt.step(val.loss, epoch, model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "lr": lr, **group_lrs, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val.loss, "val_acc": val.accuracy, "epoch_time_s": dt, "best": improved})
        log(f"epoch {epoch:02d} | lr {lr:.1e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val.loss:.4f} acc {val.accuracy:.4f} | {dt:.1f}s{'  *best*' if improved else ''}")
        if stopper.step(val.loss):
            stopped_early = True
            log(f"early stopping: no val-loss improvement for {cfg.early_stopping_patience} epochs")
            break
    total = time.perf_counter() - t_start
    return {
        "history": history,
        "best_epoch": ckpt.best_epoch,
        "best_val_loss": ckpt.best,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "train_time_s": total,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
    }
