"""PyTorch dataset built on the Phase 1 split CSVs (read-only).

* The split CSVs are the single source of truth: file list and ``label_idx``.
* Images are loaded once in memory as uint8 tensors (1 x 200 x 200): 1 800 images
  are ~70 MB, so disk I/O disappears from the training loop and ``num_workers=0``
  is enough (no worker start-up cost on Windows, no extra source of randomness).
* **Test guard**: requesting the ``test`` split raises unless ``allow_test=True``.
  No Phase 2 script sets this flag.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ivai.config import NEU_PROCESSED_DIR, SPLITS_DIR
from ivai.data.index import load_gray

VALID_SPLITS = ("train", "val", "test")


class TestSetAccessError(RuntimeError):
    """Raised when code tries to read the test split without explicit permission."""

    __test__ = False  # not a pytest test class despite its name


def read_split(split: str, splits_dir: Path = SPLITS_DIR, allow_test: bool = False) -> pd.DataFrame:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    if split == "test" and not allow_test:
        raise TestSetAccessError("The test split is locked during model development (pass allow_test=True only for the final evaluation).")
    return pd.read_csv(splits_dir / f"neu_cls_{split}.csv")


def load_train_stats(splits_dir: Path = SPLITS_DIR) -> tuple[float, float]:
    """Normalisation statistics computed on the train split during Phase 1."""
    stats = json.loads((splits_dir / "train_pixel_stats.json").read_text(encoding="utf-8"))
    return float(stats["mean"]), float(stats["std"])


def build_transforms(train: bool, mean: float, std: float, aug: dict | None = None) -> v2.Compose:
    """uint8 tensor (1,H,W) -> normalised float tensor. Augmentations only if ``train``."""
    ops: list = [v2.ToDtype(torch.float32, scale=True)]  # [0,255] -> [0,1]
    if train and aug:
        if aug.get("hflip", 0) > 0:
            ops.append(v2.RandomHorizontalFlip(p=aug["hflip"]))
        if aug.get("vflip", 0) > 0:
            ops.append(v2.RandomVerticalFlip(p=aug["vflip"]))
        b, c = aug.get("brightness", 0), aug.get("contrast", 0)
        if b or c:
            ops.append(v2.ColorJitter(brightness=b, contrast=c))
            ops.append(v2.Lambda(lambda x: x.clamp(0.0, 1.0)))
    ops.append(v2.Normalize(mean=[mean], std=[std]))
    return v2.Compose(ops)


class NEUDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform: Callable | None = None,
        allow_test: bool = False,
        splits_dir: Path = SPLITS_DIR,
        images_root: Path = NEU_PROCESSED_DIR,
    ):
        self.split = split
        self.df = read_split(split, splits_dir, allow_test).reset_index(drop=True)
        self.transform = transform
        self.filenames: list[str] = self.df["filename"].tolist()
        self.targets = torch.tensor(self.df["label_idx"].to_numpy(), dtype=torch.long)
        paths = [images_root / split / lbl / f for lbl, f in zip(self.df["label"], self.df["filename"])]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}. Run scripts/prepare_splits.py.")
        self.images = torch.stack([torch.tensor(load_gray(p)).unsqueeze(0) for p in paths])  # N,1,H,W uint8 (copy)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[i]
