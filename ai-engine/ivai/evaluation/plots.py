"""Figures for training curves and confusion matrices (matplotlib, saved as PNG)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (colour-blind safe), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_training_curves(history: list[dict], best_epoch: int, title: str, path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
        for k, (split, label) in enumerate([("train", "train (augmenté, dropout actif)"), ("val", "validation")]):
            ax.plot(ep, [h[f"{split}_{key}"] for h in history], color=SERIES[k], linewidth=2,
                    marker="o", markersize=3.5, label=label)
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"meilleure epoch ({best_epoch})", xy=(best_epoch, 0.5), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED, rotation=90, va="center")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        _style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str, path: Path) -> None:
    """Counts in each cell; colour = row-normalised rate (recall of the true class)."""
    cm = np.asarray(cm)
    rates = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]:g}", ha="center", va="center", fontsize=10,
                        color="white" if rates[i, j] > 0.6 else TEXT)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Taux par ligne (rappel)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
