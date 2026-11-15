"""Aggregate the runs of one experiment over its seeds (validation metrics only).

Usage:  python scripts/summarize_runs.py --experiment custom_cnn

Standard deviation = sample std (ddof=1) over the n seeds. With n = 3 it is a rough
estimate of the run-to-run variability, not a confidence interval.

Outputs: reports/experiments/<exp>/summary.json, summary.md
         reports/figures/<exp>/summary_val_curves.png, summary_f1_per_class.png,
         summary_confusion_val_sum.png
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ivai.config import FIGURES_DIR, REPORTS_DIR, load_data_config  # noqa: E402
from ivai.evaluation.plots import GRID, MUTED, SERIES, plot_confusion_matrix  # noqa: E402

GLOBAL = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "loss"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="custom_cnn")
    args = ap.parse_args()
    classes = load_data_config().classes
    exp_dir = REPORTS_DIR / "experiments" / args.experiment
    fig_dir = FIGURES_DIR / args.experiment

    # Main runs only: folders named exactly seed_<N> (tagged extra runs are excluded)
    runs = sorted((p for p in exp_dir.glob("seed_*") if p.name.split("_")[1].isdigit() and p.name.count("_") == 1),
                  key=lambda p: int(p.name.split("_")[1]))
    data = [json.loads((r / "run.json").read_text(encoding="utf-8")) for r in runs]
    if not data:
        raise SystemExit(f"No runs in {exp_dir}")
    seeds = [d["seed"] for d in data]

    rows = []
    for d in data:
        m = d["best_validation_metrics"]
        rows.append({"seed": d["seed"], **{k: m[k] for k in GLOBAL},
                     "n_errors": m["n_errors"], "best_epoch": d["training"]["best_epoch"],
                     "epochs_run": d["training"]["epochs_run"], "stopped_early": d["training"]["stopped_early"],
                     "train_time_s": d["training"]["train_time_s"],
                     "acc_multi_defect": m["accuracy_multi_defect_images"]})
    per_seed = pd.DataFrame(rows).set_index("seed")
    num = per_seed.drop(columns=["stopped_early"]).astype(float)
    agg = pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1)})

    f1 = pd.DataFrame({d["seed"]: {c: d["best_validation_metrics"]["per_class"][c]["f1"] for c in classes} for d in data})
    prec = pd.DataFrame({d["seed"]: {c: d["best_validation_metrics"]["per_class"][c]["precision"] for c in classes} for d in data})
    rec = pd.DataFrame({d["seed"]: {c: d["best_validation_metrics"]["per_class"][c]["recall"] for c in classes} for d in data})
    per_class = pd.DataFrame({"precision_mean": prec.mean(axis=1), "recall_mean": rec.mean(axis=1),
                              "f1_mean": f1.mean(axis=1), "f1_std": f1.std(axis=1, ddof=1)})
    cm_sum = np.sum([np.array(d["best_validation_metrics"]["confusion_matrix"]) for d in data], axis=0)

    summary = {
        "experiment": args.experiment, "seeds": seeds, "n_runs": len(data), "split": "validation",
        "std": "sample standard deviation (ddof=1)",
        "per_seed": per_seed.reset_index().to_dict(orient="records"),
        "mean_std": agg.to_dict(orient="index"),
        "per_class": per_class.to_dict(orient="index"),
        "confusion_matrix_sum": cm_sum.tolist(),
        "parameters": data[0]["architecture"]["parameters"]["total"],
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fmt = lambda k, p=4: f"{agg.loc[k, 'mean']:.{p}f} ± {agg.loc[k, 'std']:.{p}f}"  # noqa: E731
    md = [f"# {args.experiment}: validation results over seeds {seeds}", "",
          "| Seed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Val loss | Errors /270 | Best epoch | Epochs run | Early stop | Time (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in per_seed.iterrows():
        md.append(f"| {s} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} | {r.f1_macro:.4f} | "
                  f"{r.loss:.4f} | {int(r.n_errors)} | {int(r.best_epoch)} | {int(r.epochs_run)} | {r.stopped_early} | {r.train_time_s:.1f} |")
    md.append(f"| **mean ± std** | {fmt('accuracy')} | {fmt('precision_macro')} | {fmt('recall_macro')} | {fmt('f1_macro')} | "
              f"{fmt('loss')} | {fmt('n_errors', 1)} | {fmt('best_epoch', 1)} | {fmt('epochs_run', 1)} | | {fmt('train_time_s', 1)} |")
    md += ["", "| Class | Precision (mean) | Recall (mean) | F1 mean ± std |", "|---|---|---|---|"]
    for c, r in per_class.iterrows():
        md.append(f"| {c} | {r.precision_mean:.4f} | {r.recall_mean:.4f} | {r.f1_mean:.4f} ± {r.f1_std:.4f} |")
    (exp_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- figures ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, (d, r) in enumerate(zip(data, runs)):
        h = pd.read_csv(r / "history.csv")
        axes[0].plot(h.epoch, h.val_loss, color=SERIES[k], linewidth=2, label=f"seed {d['seed']}")
        axes[0].plot(h.epoch, h.train_loss, color=SERIES[k], linewidth=1, linestyle=":", alpha=0.9)
        axes[1].plot(h.epoch, h.val_acc, color=SERIES[k], linewidth=2, label=f"seed {d['seed']}")
        be = d["training"]["best_epoch"]
        axes[0].plot(be, h.loc[h.epoch == be, "val_loss"].item(), "o", color=SERIES[k], markersize=8,
                     markeredgecolor="white", markeredgewidth=2)
    axes[0].set_ylabel("Loss (trait plein : validation, pointillé : train)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Accuracy de validation")
    axes[1].set_ylim(0.7, 1.0)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.legend(frameon=False)
    fig.suptitle(f"{args.experiment} : convergence des {len(data)} seeds (point = epoch retenue)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "summary_val_curves.png", dpi=150, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(classes))
    bars = ax.bar(x, per_class.f1_mean, yerr=per_class.f1_std, color=SERIES[0], width=0.55, capsize=4,
                  error_kw={"ecolor": MUTED, "linewidth": 1})
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in per_class.f1_mean], padding=6, fontsize=8)
    ax.set_xticks(x, classes, rotation=20)
    lo = max(0.0, float((per_class.f1_mean - per_class.f1_std).min()) - 0.05)
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel("F1 (validation)")
    ax.set_title(f"F1 par classe : moyenne ± écart-type sur {len(data)} seeds", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "summary_f1_per_class.png", dpi=150, facecolor="white")
    plt.close(fig)

    plot_confusion_matrix(cm_sum, classes, f"Validation : somme des {len(data)} seeds ({cm_sum.sum()} prédictions)",
                          fig_dir / "summary_confusion_val_sum.png")
    print(f"\n-> {exp_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
