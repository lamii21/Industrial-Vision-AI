"""E5 (complementary) - robustness of the E1 models to acquisition perturbations of the
VALIDATION images (docs/phase3_protocol.md §9.6). Descriptive only: never used to select
a model. The test split is never loaded.

Usage:
    python scripts/evaluate_robustness.py
    python scripts/evaluate_robustness.py --experiments custom_cnn --out <dir>
Outputs: reports/robustness/phase3/robustness.json, robustness.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ivai.config import REPORTS_DIR, load_data_config
from ivai.data.dataset import NEUDataset, build_transforms, resolve_normalization
from ivai.evaluation.calibration import nll
from ivai.evaluation.comparison import load_experiment, model_from_run
from ivai.evaluation.metrics import classification_metrics
from ivai.evaluation.robustness import PERTURBATIONS, PerturbedDataset
from ivai.training.trainer import evaluate

E1 = ["custom_cnn", "resnet50_ft", "mobilenetv2_ft"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=E1)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "robustness" / "phase3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if (args.out / "robustness.json").exists() and not args.force:
        raise SystemExit(f"{args.out / 'robustness.json'} exists; refusing to overwrite.")
    args.out.mkdir(parents=True, exist_ok=True)
    classes = load_data_config().classes
    device = torch.device(args.device)
    raw_val = NEUDataset("val", transform=None)          # uint8 images, validation only
    conditions = [("clean", None), *PERTURBATIONS]

    results = {}
    for name in args.experiments:
        exp = load_experiment(name, classes)
        if exp is None:
            print(f"skip {name}: no runs")
            continue
        results[name] = {}
        for seed in exp.seeds:
            model, inp = model_from_run(exp.runs[seed], device)
            mean, std = resolve_normalization(inp["normalization"])
            tf = build_transforms(False, mean, std, channels=inp["channels"])
            amp = bool(exp.runs[seed]["hyperparameters"].get("amp", True))
            per_cond = {}
            for kind, level in conditions:
                ds = PerturbedDataset(raw_val, kind, level, tf)
                res = evaluate(model, DataLoader(ds, batch_size=64), torch.nn.CrossEntropyLoss(), device, amp)
                m = classification_metrics(res.y_true, res.y_pred, classes)
                per_cond[f"{kind}" + ("" if level is None else f"@{level}")] = {
                    "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "n_errors": m["n_errors"],
                    "nll": nll(res.probs, res.y_true)}
            results[name][seed] = per_cond
            print(f"{name} seed {seed}: " + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in per_cond.items()))

    summary = {}
    for name, by_seed in results.items():
        conds = list(next(iter(by_seed.values())))
        summary[name] = {}
        for c in conds:
            vals = {k: [by_seed[s][c][k] for s in by_seed] for k in ("accuracy", "f1_macro", "nll")}
            clean = {k: [by_seed[s]["clean"][k] for s in by_seed] for k in ("accuracy", "nll")}
            summary[name][c] = {**{k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                                   for k, v in vals.items()},
                                "accuracy_drop_vs_clean": float(np.mean(clean["accuracy"]) - np.mean(vals["accuracy"]))}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "validation",
           "protocol": "docs/phase3_protocol.md §9.6 (descriptive, not a selection criterion)",
           "perturbations": [list(p) for p in PERTURBATIONS], "per_seed": results, "summary": summary}
    (args.out / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# E5 - robustness on perturbed validation images (descriptive)", "",
          "| Condition | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for c in (next(iter(summary.values())) if summary else {}):
        md.append(f"| {c} | " + " | ".join(f"{summary[n][c]['accuracy']['mean']:.3f}" for n in summary) + " |")
    (args.out / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
