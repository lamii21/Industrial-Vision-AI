"""List and display the validation errors of every run of an experiment.

Usage:  python scripts/plot_val_errors.py --experiment custom_cnn [--experiment custom_cnn_30ep ...]

Outputs: reports/experiments/<first exp>/val_errors.csv
         reports/figures/<first exp>/val_errors.png  (YOLO boxes in yellow, for context only)
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from ivai.config import FIGURES_DIR, NEU_PROCESSED_DIR, REPORTS_DIR, SPLITS_DIR  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="append", required=True)
    args = ap.parse_args()

    rows = []
    for exp in args.experiment:
        for run in sorted((REPORTS_DIR / "experiments" / exp).glob("seed_*")):
            if run.name.count("_") != 1:
                continue  # skip tagged extra runs
            p = pd.read_csv(run / "val_predictions.csv")
            for r in p[p.true != p.pred].itertuples():
                rows.append({"experiment": exp, "run": run.name, "filename": r.filename, "true": r.true,
                             "pred": r.pred, "confidence": round(r.confidence, 3),
                             "multi_defect": bool(r.has_secondary_defects)})
    errors = pd.DataFrame(rows)
    n_runs = sum(1 for exp in args.experiment for r in (REPORTS_DIR / "experiments" / exp).glob("seed_*") if r.name.count("_") == 1)
    out_exp = REPORTS_DIR / "experiments" / args.experiment[0]
    errors.to_csv(out_exp / "val_errors.csv", index=False)
    print(errors.to_string(index=False))
    if errors.empty:
        return 0

    per_file = (errors.groupby(["filename", "true"])
                .agg(n=("run", "size"), pred=("pred", lambda s: s.mode()[0]),
                     cmin=("confidence", "min"), cmax=("confidence", "max"))
                .reset_index().sort_values("n", ascending=False))
    boxes = pd.read_csv(SPLITS_DIR / "neu_cls_boxes.csv")
    fig, axes = plt.subplots(1, len(per_file), figsize=(3.2 * len(per_file), 3.9), squeeze=False)
    for ax, r in zip(axes[0], per_file.itertuples()):
        img = Image.open(NEU_PROCESSED_DIR / "val" / r.true / r.filename).convert("L")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        for b in boxes[boxes.filename == r.filename].itertuples():
            ax.add_patch(mpatches.Rectangle(((b.cx - b.w / 2) * 200, (b.cy - b.h / 2) * 200), b.w * 200, b.h * 200,
                                            fill=False, edgecolor="#eda100", linewidth=1.5))
        ax.set_title(f"{r.filename}\n{r.true} → {r.pred}\n{r.n}/{n_runs} runs, conf. {r.cmin:.2f}–{r.cmax:.2f}", fontsize=8.5)
        ax.axis("off")
    fig.suptitle("Images de validation mal classées (toutes runs confondues)", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / args.experiment[0] / "val_errors.png"
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
