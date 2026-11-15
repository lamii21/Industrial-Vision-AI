"""Loading and aggregating experiment runs for the Phase 3 comparison module.

Reads only the per-run files written by ``scripts/train.py`` (``run.json``,
``val_predictions.csv``): validation data only, never the test split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ivai.config import ENGINE_DIR, REPORTS_DIR
from ivai.evaluation.calibration import brier, calibration_report, per_sample_nll
from ivai.models.registry import build_model, default_input
from ivai.training.callbacks import load_checkpoint

SEEDS = (42, 123, 2026)


def main_seed_runs(exp_dir: Path) -> list[Path]:
    """Folders named exactly ``seed_<N>`` (tagged extra runs such as ``seed_42_repro`` are excluded)."""
    runs = [p for p in exp_dir.glob("seed_*") if p.is_dir() and p.name.count("_") == 1 and p.name.split("_")[1].isdigit()]
    return sorted(runs, key=lambda p: int(p.name.split("_")[1]))


@dataclass
class Experiment:
    name: str
    runs: dict[int, dict] = field(repr=False)            # seed -> run.json
    preds: dict[int, pd.DataFrame] = field(repr=False)   # seed -> val predictions sorted by filename
    classes: tuple[str, ...] = ()

    @property
    def seeds(self) -> list[int]:
        return sorted(self.runs)

    def probs_and_labels(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        df = self.preds[seed]
        probs = df[[f"p_{c}" for c in self.classes]].to_numpy(float)
        y = df["true"].map({c: i for i, c in enumerate(self.classes)}).to_numpy(int)
        return probs, y

    def per_image_nll(self, seeds=None) -> np.ndarray:
        """(n_seeds, n_images), images sorted by file name."""
        return np.stack([per_sample_nll(*self.probs_and_labels(s)) for s in (seeds or self.seeds)])

    def filenames(self, seed: int) -> list[str]:
        return self.preds[seed]["filename"].tolist()


def load_experiment(name: str, classes, reports_dir: Path = REPORTS_DIR) -> Experiment | None:
    exp_dir = reports_dir / "experiments" / name
    runs, preds = {}, {}
    for r in main_seed_runs(exp_dir):
        run = json.loads((r / "run.json").read_text(encoding="utf-8"))
        if run.get("data", {}).get("test_used", False):
            raise RuntimeError(f"{r}: run.json says the test split was used")
        runs[run["seed"]] = run
        preds[run["seed"]] = pd.read_csv(r / "val_predictions.csv").sort_values("filename").reset_index(drop=True)
    if not runs:
        return None
    files = {tuple(df["filename"]) for df in preds.values()}
    if len(files) != 1:
        raise RuntimeError(f"{name}: seeds do not share the same validation images")
    return Experiment(name, runs, preds, tuple(classes))


def check_aligned(*exps: Experiment) -> None:
    """Paired comparisons need the same seeds and the same validation images in the same order."""
    ref = exps[0]
    for e in exps[1:]:
        if e.seeds != ref.seeds:
            raise ValueError(f"{e.name} and {ref.name} do not have the same seeds")
        for s in ref.seeds:
            if e.filenames(s) != ref.filenames(s):
                raise ValueError(f"{e.name} and {ref.name}: different validation images")


def _mean_std(values) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "per_seed": v.tolist()}


def summarize(exp: Experiment, with_calibration: bool = True) -> dict:
    """Mean ± std (ddof=1) over seeds of every validation metric of protocol §9.1-9.3."""
    rows = {k: [] for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "n_errors", "nll", "brier",
                            "ece", "best_epoch", "epochs_run", "train_time_s")}
    recall = {c: [] for c in exp.classes}
    calib = {}
    for s in exp.seeds:
        m = exp.runs[s]["best_validation_metrics"]
        probs, y = exp.probs_and_labels(s)
        for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "n_errors"):
            rows[k].append(m[k])
        rows["nll"].append(float(per_sample_nll(probs, y).mean()))
        rows["brier"].append(brier(probs, y))
        for k in ("best_epoch", "epochs_run", "train_time_s"):
            rows[k].append(exp.runs[s]["training"][k])
        for c in exp.classes:
            recall[c].append(m["per_class"][c]["recall"])
        if with_calibration:
            calib[s] = calibration_report(probs, y)
            rows["ece"].append(calib[s]["ece"]["value"])
    if not with_calibration:
        rows.pop("ece")
    run0 = exp.runs[exp.seeds[0]]
    return {
        "experiment": exp.name, "seeds": exp.seeds,
        "metrics": {k: _mean_std(v) for k, v in rows.items()},
        "mean_recall_per_class": {c: float(np.mean(v)) for c, v in recall.items()},
        "parameters": run0["architecture"]["parameters"],
        "train_subset": run0.get("data", {}).get("train_subset"),
        "calibration_per_seed": calib,
    }


def learning_curve(points: dict[float, dict]) -> dict:
    """``points``: real fraction -> summary. F1/accuracy/NLL vs real fraction and the area
    under the F1 curve (trapezoidal rule, x = real fraction, protocol §9.7)."""
    xs = sorted(points)
    curve = {k: [points[x]["metrics"][k]["mean"] for x in xs] for k in ("f1_macro", "accuracy", "nll")}
    std = {k: [points[x]["metrics"][k]["std"] for x in xs] for k in ("f1_macro", "accuracy", "nll")}
    auc = float(np.trapezoid(curve["f1_macro"], xs)) if len(xs) > 1 else None
    return {"fractions": xs, "mean": curve, "std": std, "auc_f1": auc}


def model_from_run(run: dict, device: str | torch.device = "cpu", load_weights: bool = True) -> tuple[torch.nn.Module, dict]:
    """Rebuild a trained model from its ``run.json`` (weights from its checkpoint; ImageNet
    weights are never needed because the checkpoint contains every parameter)."""
    mcfg = dict(run["architecture"]["config"])
    if mcfg["name"] != "custom_cnn":
        mcfg["pretrained"] = False
        mcfg["freeze_backbone"] = False
    model = build_model(mcfg)
    if load_weights:
        load_checkpoint(model, ENGINE_DIR.parent / run["checkpoint"], map_location=device)
    inp = run.get("data", {}).get("input") or default_input(run["architecture"]["config"])
    return model.to(device).eval(), inp
