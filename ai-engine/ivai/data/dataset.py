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


# ImageNet statistics expected by the torchvision pretrained weights (Phase 3).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_normalization(normalization: str, splits_dir: Path = SPLITS_DIR) -> tuple:
    """``"dataset"`` -> train statistics of Phase 1 (floats, 1 channel);
    ``"imagenet"`` -> ImageNet statistics (3-tuples, 3 channels)."""
    if normalization == "dataset":
        return load_train_stats(splits_dir)
    if normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(f"Unknown normalization {normalization!r}")


def _gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(1,H,W) -> (3,H,W): the grey channel is replicated (pretrained models expect RGB)."""
    return x.repeat(3, 1, 1)


def build_transforms(train: bool, mean, std, aug: dict | None = None, channels: int = 1) -> v2.Compose:
    """uint8 tensor (1,H,W) -> normalised float tensor. Augmentations only if ``train``.

    ``channels=1`` with float ``mean``/``std`` is the Phase 2 pipeline (unchanged).
    ``channels=3``: the augmented grey image is replicated to 3 channels just before the
    normalisation (``mean``/``std`` are then 3-tuples, e.g. ImageNet statistics).
    """
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
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
    if channels == 3:
        ops.append(v2.Lambda(_gray_to_rgb))
    mean_l = list(mean) if isinstance(mean, (list, tuple)) else [mean]
    std_l = list(std) if isinstance(std, (list, tuple)) else [std]
    if len(mean_l) != channels or len(std_l) != channels:
        raise ValueError(f"mean/std must have {channels} value(s)")
    ops.append(v2.Normalize(mean=mean_l, std=std_l))
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
