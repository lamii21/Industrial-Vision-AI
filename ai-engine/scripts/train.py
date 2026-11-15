"""Train one model with one seed. Validation is used for model selection; the
test split is never loaded.

Usage:
    python scripts/train.py --config configs/custom_cnn.yaml --seed 42
    python scripts/train.py --config configs/custom_cnn.yaml --seed 42 --tag repro   # extra run, separate folder

Outputs (per run):
    artifacts/models/<experiment>/seed_<N>[_<tag>]/best.pt          weights (not versioned)
    reports/experiments/<experiment>/seed_<N>[_<tag>]/run.json      full traceability
                                                     history.csv    per-epoch metrics
                                                     val_predictions.csv
    reports/figures/<experiment>/seed_<N>[_<tag>]_curves.png, _confusion_val.png
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import ivai
from ivai.config import ENGINE_DIR, FIGURES_DIR, REPORTS_DIR, SPLITS_DIR, load_data_config
from ivai.data.dataset import NEUDataset, build_transforms, load_train_stats
from ivai.evaluation.metrics import classification_metrics
from ivai.evaluation.plots import plot_confusion_matrix, plot_training_curves
from ivai.models.analysis import count_parameters, empirical_receptive_field, theoretical_receptive_field
from ivai.models.custom_cnn import CustomCNN
from ivai.training.callbacks import load_checkpoint
from ivai.training.trainer import TrainConfig, evaluate, fit
from ivai.utils.seed import seed_worker, set_seed
from ivai.utils.system import prevent_sleep

ARTIFACTS_DIR = ENGINE_DIR.parent / "artifacts" / "models"
MODELS = {"custom_cnn": CustomCNN}


def build_model(mcfg: dict) -> torch.nn.Module:
    kwargs = {k: v for k, v in mcfg.items() if k != "name"}
    return MODELS[mcfg["name"]](**kwargs)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="", help="suffix for an extra run (e.g. reproducibility check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = load_data_config()
    exp = cfg["experiment"]
    run_name = f"seed_{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = REPORTS_DIR / "experiments" / exp / run_name
    fig_dir = FIGURES_DIR / exp
    ckpt_path = ARTIFACTS_DIR / exp / run_name / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
    generator = set_seed(args.seed, deterministic=tcfg.get("deterministic", True))
    device = torch.device(args.device)

    # ---- data (train + val only) -------------------------------------------
    mean, std = load_train_stats()
    train_ds = NEUDataset("train", build_transforms(True, mean, std, cfg["data"].get("augment")))
    val_ds = NEUDataset("val", build_transforms(False, mean, std))
    overlap = set(train_ds.filenames) & set(val_ds.filenames)
    if overlap:
        raise RuntimeError(f"train/val overlap: {sorted(overlap)[:5]}")
    bs = cfg["data"]["batch_size"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=generator,
                              worker_init_fn=seed_worker, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=0, pin_memory=pin)

    # ---- model --------------------------------------------------------------
    model = build_model(cfg["model"])
    params = count_parameters(model)
    rf = theoretical_receptive_field(model)
    rf["measured_geometric"] = empirical_receptive_field(model)
    print(f"[{exp} | {run_name}] device={device} params={params['total']:,} receptive field={rf['receptive_field']}px")

    # ---- training -----------------------------------------------------------
    train_cfg = TrainConfig(**{k: v for k, v in tcfg.items() if k in TrainConfig.__dataclass_fields__})
    with prevent_sleep():
        result = fit(model, train_loader, val_loader, train_cfg, device, ckpt_path)

    # ---- validation metrics of the selected (best val-loss) checkpoint -------
    load_checkpoint(model, ckpt_path, map_location=device)
    val = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device, train_cfg.amp)
    metrics = classification_metrics(val.y_true, val.y_pred, data_cfg.classes)
    metrics["loss"] = val.loss

    # Predictions: val rows only (metadata of other splits is never joined).
    index = pd.read_csv(SPLITS_DIR / "neu_cls_index.csv", usecols=["filename", "split", "has_secondary_defects"])
    index = index[index["split"] == "val"].drop(columns="split")
    pred_df = pd.DataFrame({
        "filename": val_ds.filenames,
        "true": [data_cfg.classes[i] for i in val.y_true],
        "pred": [data_cfg.classes[i] for i in val.y_pred],
        "confidence": val.probs.max(1),
        **{f"p_{c}": val.probs[:, k] for k, c in enumerate(data_cfg.classes)},
    }).merge(index, on="filename", how="left")
    pred_df.to_csv(run_dir / "val_predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False, float_format="%.6f")

    multi = pred_df["has_secondary_defects"].astype(bool)
    metrics["accuracy_mono_defect_images"] = float((pred_df.true == pred_df.pred)[~multi].mean())
    metrics["accuracy_multi_defect_images"] = float((pred_df.true == pred_df.pred)[multi].mean()) if multi.any() else None
    metrics["n_multi_defect_images"] = int(multi.sum())

    best_row = next(h for h in result["history"] if h["epoch"] == result["best_epoch"])
    run = {
        "experiment": exp,
        "run": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": {"name": cfg["model"]["name"], "config": cfg["model"], "repr": str(model),
                         "parameters": params, "receptive_field": rf},
        "hyperparameters": {"batch_size": bs, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                            "scheduler": "ReduceLROnPlateau(val_loss)", "selection": "min val_loss",
                            **tcfg, "augment": cfg["data"].get("augment")},
        "seed": args.seed,
        "data": {"train_size": len(train_ds), "val_size": len(val_ds), "test_used": False,
                 "normalization": {"mean": mean, "std": std, "source": "train_pixel_stats.json"},
                 "split_csv_sha256": json.loads((SPLITS_DIR / "split_meta.json").read_text())["csv_sha256"]},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "device": str(device),
                        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                        "python": platform.python_version(), "ivai": ivai.__version__,
                        "deterministic": tcfg.get("deterministic", True)},
        "training": {"best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                     "stopped_early": result["stopped_early"], "train_time_s": round(result["train_time_s"], 2),
                     "mean_epoch_time_s": round(result["train_time_s"] / result["epochs_run"], 3),
                     "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                     "train_loss_at_best": best_row["train_loss"], "train_acc_at_best": best_row["train_acc"]},
        "best_validation_metrics": metrics,
        "checkpoint": str(ckpt_path.relative_to(ENGINE_DIR.parent)).replace("\\", "/"),
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_training_curves(result["history"], result["best_epoch"],
                         f"CNN personnalisé, seed {args.seed}{' (' + args.tag + ')' if args.tag else ''} : convergence",
                         fig_dir / f"{run_name}_curves.png")
    plot_confusion_matrix(metrics["confusion_matrix"], data_cfg.classes,
                          f"Validation, seed {args.seed} (epoch {result['best_epoch']}) : acc {metrics['accuracy']:.3f}",
                          fig_dir / f"{run_name}_confusion_val.png")

    print(f"best epoch {result['best_epoch']} | val loss {val.loss:.4f} | acc {metrics['accuracy']:.4f} | "
          f"F1 macro {metrics['f1_macro']:.4f} | {result['train_time_s']:.1f}s")
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
