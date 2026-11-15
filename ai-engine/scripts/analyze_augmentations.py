"""Empirical check of the geometric augmentations, on the TRAIN split only.

Questions answered
1. Orientation: is there a dominant texture orientation per class? Flips keep the
   orientation of a structure (a vertical streak stays vertical) whereas a 90°
   rotation turns vertical into horizontal. Measure: log ratio of the vertical-
   to horizontal-gradient energy, log(E|d/dx| / E|d/dy|) > 0 means vertical
   structures dominate (intensity varies mostly along x).
2. Asymmetry: does the class mean image differ from its mirrored version (left/right,
   top/bottom) by more than the sampling noise? If a class had a real top/bottom
   asymmetry (e.g. defects always "pointing" in the rolling direction), a vertical
   flip would create unrealistic samples.

Usage:  python scripts/analyze_augmentations.py
Outputs: reports/figures/custom_cnn/aug_orientation.png and
         reports/experiments/custom_cnn/augmentation_check.json
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivai.config import FIGURES_DIR, NEU_PROCESSED_DIR, REPORTS_DIR, SPLITS_DIR, load_data_config
from ivai.data.index import load_gray

OUT_FIG = FIGURES_DIR / "custom_cnn"
OUT_EXP = REPORTS_DIR / "experiments" / "custom_cnn"


def orientation_index(img: np.ndarray) -> float:
    g = img.astype(np.float64)
    gx = np.abs(np.diff(g, axis=1)).mean()  # variation along x -> vertical structures
    gy = np.abs(np.diff(g, axis=0)).mean()  # variation along y -> horizontal structures
    return float(np.log(gx / gy))


def asymmetry(imgs: np.ndarray, axis: int, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """RMS difference between the mean image and its mirror, and the same statistic
    under random flips (null hypothesis: the class is symmetric)."""
    mean = imgs.mean(0)
    observed = float(np.sqrt(((mean - np.flip(mean, axis=axis)) ** 2).mean()))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_boot):
        mask = rng.random(len(imgs)) < 0.5
        mixed = np.where(mask[:, None, None], np.flip(imgs, axis=axis + 1), imgs).mean(0)
        null.append(np.sqrt(((mixed - np.flip(mixed, axis=axis)) ** 2).mean()))
    return observed, float(np.percentile(null, 95))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_data_config()
    train = pd.read_csv(SPLITS_DIR / "neu_cls_train.csv")  # train only: test never read
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_EXP.mkdir(parents=True, exist_ok=True)

    rows, orient = [], {}
    for cls in cfg.classes:
        files = train.loc[train.label == cls, "filename"]
        imgs = np.stack([load_gray(NEU_PROCESSED_DIR / "train" / cls / f) for f in files]).astype(np.float64)
        oi = np.array([orientation_index(i) for i in imgs])
        oi_rot = np.array([orientation_index(np.rot90(i)) for i in imgs])
        oi_hf = np.array([orientation_index(np.fliplr(i)) for i in imgs])
        oi_vf = np.array([orientation_index(np.flipud(i)) for i in imgs])
        orient[cls] = oi
        lr_obs, lr_null = asymmetry(imgs, axis=1)
        tb_obs, tb_null = asymmetry(imgs, axis=0)
        rows.append({
            "class": cls,
            "orientation_mean": round(float(oi.mean()), 3),
            "share_vertical_dominant": round(float((oi > 0).mean()), 3),
            "orientation_after_hflip": round(float(oi_hf.mean()), 3),
            "orientation_after_vflip": round(float(oi_vf.mean()), 3),
            "orientation_after_rot90": round(float(oi_rot.mean()), 3),
            "lr_asymmetry": round(lr_obs, 2), "lr_null95": round(lr_null, 2),
            "tb_asymmetry": round(tb_obs, 2), "tb_null95": round(tb_null, 2),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.boxplot([orient[c] for c in cfg.classes], tick_labels=list(cfg.classes), widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": "#2a78d6"}, medianprops={"color": "#0b0b0b"},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#52514e", "markeredgecolor": "none"})
    ax.axhline(0, color="#52514e", linewidth=1, linestyle="--")
    ax.set_ylabel("log(E|∂x| / E|∂y|)")
    ax.set_title("Orientation des textures (train) : > 0 = structures verticales dominantes", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "aug_orientation.png", dpi=150, facecolor="white")

    (OUT_EXP / "augmentation_check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT_FIG / 'aug_orientation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
