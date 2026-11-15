"""Train one model with one seed. Validation is used for model selection; the
test split is never loaded.

Usage:
    python scripts/train.py --config configs/custom_cnn.yaml --seed 42
    python scripts/train.py --config configs/custom_cnn.yaml --seed 42 --tag nonreg     # extra run, separate folder
    python scripts/train.py --config configs/resnet50_ft.yaml --seed 42                 # Phase 3 (E1)
    python scripts/train.py --config configs/resnet50_ft.yaml --seed 42 --train-fraction 0.1   # E3 -> resnet50_ft_f010

Outputs (per run):
    artifacts/models/<experiment>/seed_<N>[_<tag>]/best.pt          weights (not versioned)
    reports/experiments/<experiment>/seed_<N>[_<tag>]/run.json      full traceability
                                                     history.csv    per-epoch metrics
                                                     val_predictions.csv
                                                     train_subset.csv (E3 only)
    reports/figures/<experiment>/seed_<N>[_<tag>]_curves.png, _confusion_val.png

Phase 3 rules (docs/phase3_protocol.md):
* never overwrites: the script stops if the run folder or the checkpoint already exists;
* never downloads weights: pretrained weights must already be in the local cache
  (scripts/download_weights.py);
* for the custom CNN at fraction 1.0 the code path (data, loader, optimiser, order of
  random-number consumption) is the Phase 2 one, so a re-run is bit-identical.
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
from torch.utils.data import DataLoader, Subset

import ivai
from ivai.config import ENGINE_DIR, FIGURES_DIR, REPORTS_DIR, SPLITS_DIR, load_data_config
from ivai.data.dataset import NEUDataset, build_transforms, resolve_normalization
from ivai.data.subsets import VIRTUAL_EPOCH_SIZE, fraction_tag, make_train_loader, stratified_nested_subset
from ivai.evaluation.metrics import classification_metrics
from ivai.evaluation.plots import plot_confusion_matrix, plot_training_curves
from ivai.models.analysis import count_parameters, empirical_receptive_field, theoretical_receptive_field
from ivai.models.pretrained import WEIGHTS, weights_cache_path
from ivai.models.registry import build_model, default_input, parameter_groups
from ivai.training.callbacks import load_checkpoint
from ivai.training.trainer import TrainConfig, evaluate, fit
from ivai.utils.seed import set_seed
from ivai.utils.system import prevent_sleep

ARTIFACTS_DIR = ENGINE_DIR.parent / "artifacts" / "models"
DISPLAY = {"custom_cnn": "CNN personnalisé", "resnet50": "ResNet50", "mobilenetv2": "MobileNetV2"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="", help="suffix for an extra run (e.g. reproducibility check)")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="E3: stratified nested train subset (0.1, 0.25, 0.5)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = load_data_config()
    mcfg = cfg["model"]
    fraction = args.train_fraction
    if not 0 < fraction <= 1:
        raise SystemExit("--train-fraction must be in (0, 1]")
    exp = cfg["experiment"] + ("" if fraction == 1.0 else f"_{fraction_tag(fraction)}")
    run_name = f"seed_{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = REPORTS_DIR / "experiments" / exp / run_name
    fig_dir = FIGURES_DIR / exp
    ckpt_path = ARTIFACTS_DIR / exp / run_name / "best.pt"
    # ---- never overwrite a previous run (Phase 2 results included) ----------
    for p in (run_dir, ckpt_path):
        if p.exists():
            raise SystemExit(f"Refusing to overwrite existing {p}. Use a new --tag or a new experiment name.")
    if mcfg.get("pretrained") and not weights_cache_path(WEIGHTS[mcfg["name"]]).exists():
        raise SystemExit(f"Pretrained weights for {mcfg['name']} are not in the local cache: run scripts/download_weights.py first.")
    run_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
    generator = set_seed(args.seed, deterministic=tcfg.get("deterministic", True))
    device = torch.device(args.device)

    # ---- data (train + val only) -------------------------------------------
    inp = cfg["data"].get("input") or default_input(mcfg)
    mean, std = resolve_normalization(inp["normalization"])
    train_ds = NEUDataset("train", build_transforms(True, mean, std, cfg["data"].get("augment"), channels=inp["channels"]))
    val_ds = NEUDataset("val", build_transforms(False, mean, std, channels=inp["channels"]))
    overlap = set(train_ds.filenames) & set(val_ds.filenames)
    if overlap:
        raise RuntimeError(f"train/val overlap: {sorted(overlap)[:5]}")
    bs = cfg["data"]["batch_size"]
    pin = device.type == "cuda"
    subset_info = None
    if fraction == 1.0:
        train_used, virtual = train_ds, None   # historical loader (shuffle=True)
    else:
        idx = stratified_nested_subset(train_ds.df, fraction, seed=args.seed)
        train_used, virtual = Subset(train_ds, idx.tolist()), VIRTUAL_EPOCH_SIZE
        sub_df = train_ds.df.iloc[idx][["filename", "label", "label_idx"]]
        sub_df.to_csv(run_dir / "train_subset.csv", index=False)
        subset_info = {"fraction_nominal": fraction, "fraction_real": round(len(idx) / len(train_ds), 4),
                       "n_images": int(len(idx)), "per_class": sub_df["label"].value_counts().sort_index().to_dict(),
                       "subset_seed": args.seed, "virtual_epoch_size": VIRTUAL_EPOCH_SIZE,
                       "rule": "per class: permutation (numpy default_rng(seed)) of train files sorted by name, first floor(n*f)"}
    train_loader = make_train_loader(train_used, bs, generator, pin, virtual_epoch_size=virtual)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=0, pin_memory=pin)

    # ---- model --------------------------------------------------------------
    model = build_model(mcfg)
    params = count_parameters(model)
    rf = None
    if mcfg["name"] == "custom_cnn":  # Phase 2 analysis, same calls in the same order
        rf = theoretical_receptive_field(model)
        rf["measured_geometric"] = empirical_receptive_field(model)
    print(f"[{exp} | {run_name}] device={device} params={params['total']:,} trainable={params['trainable']:,}"
          + (f" receptive field={rf['receptive_field']}px" if rf else ""))

    # ---- training -----------------------------------------------------------
    train_cfg = TrainConfig(**{k: v for k, v in tcfg.items() if k in TrainConfig.__dataclass_fields__})
    groups = parameter_groups(model, mcfg, train_cfg.lr, train_cfg.head_lr)
    with prevent_sleep():
        result = fit(model, train_loader, val_loader, train_cfg, device, ckpt_path, param_groups=groups)

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
    pretrained = None
    if mcfg.get("pretrained"):
        w = WEIGHTS[mcfg["name"]]
        pretrained = {"weights": f"{type(w).__name__}.{w.name}", "url": w.url, "file": Path(w.url).name}
    run = {
        "experiment": exp,
        "run": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": {"name": mcfg["name"], "config": mcfg, "repr": str(model),
                         "parameters": params, "receptive_field": rf, "pretrained": pretrained},
        "hyperparameters": {"batch_size": bs, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                            "scheduler": "ReduceLROnPlateau(val_loss)", "selection": "min val_loss",
                            **tcfg, "augment": cfg["data"].get("augment"),
                            "optimizer_groups": None if groups is None else
                            [{"name": g["name"], "lr": g["lr"], "n_params": sum(p.numel() for p in g["params"])} for g in groups]},
        "seed": args.seed,
        "data": {"train_size": len(train_used), "val_size": len(val_ds), "test_used": False,
                 "input": inp,
                 "normalization": {"mean": mean, "std": std,
                                   "source": "train_pixel_stats.json" if inp["normalization"] == "dataset" else "ImageNet"},
                 "train_subset": subset_info,
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
                     "steps_per_epoch": len(train_loader),
                     "train_loss_at_best": best_row["train_loss"], "train_acc_at_best": best_row["train_acc"]},
        "best_validation_metrics": metrics,
        "checkpoint": str(ckpt_path.relative_to(ENGINE_DIR.parent)).replace("\\", "/"),
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    label = DISPLAY.get(mcfg["name"], mcfg["name"]) + ("" if fraction == 1.0 else f" ({fraction:.0%} du train)")
    plot_training_curves(result["history"], result["best_epoch"],
                         f"{label}, seed {args.seed}{' (' + args.tag + ')' if args.tag else ''} : convergence",
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
