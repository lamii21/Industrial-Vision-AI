"""Early stopping and best-model checkpointing, both driven by the validation loss."""

from __future__ import annotations

import math
from pathlib import Path

import torch


class EarlyStopping:
    """Stop when the monitored value has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if value < self.best - self.min_delta:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ModelCheckpoint:
    """Save the model weights each time the validation loss reaches a new minimum."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best = math.inf
        self.best_epoch: int | None = None

    def step(self, value: float, epoch: int, model: torch.nn.Module, extra: dict | None = None) -> bool:
        if value < self.best:
            self.best, self.best_epoch = value, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": value, **(extra or {})}, self.path)
            return True
        return False


def load_checkpoint(model: torch.nn.Module, path: Path, map_location: str | torch.device = "cpu") -> dict:
    """Load weights saved by ``ModelCheckpoint`` (``weights_only=True``: no arbitrary code execution)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
