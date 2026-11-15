"""Compare two training runs for bit-exact reproducibility.

Usage:
    python scripts/compare_runs.py <run_dir_A> <run_dir_B> <ckpt_A> <ckpt_B>

Compares every per-epoch value of history.csv (except wall-clock times), the
validation predictions/probabilities and every tensor of the saved weights.
Exit code 0 if identical, 1 otherwise (the differences are printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    a, b, ca, cb = map(Path, sys.argv[1:5])
    ok = True

    ha, hb = pd.read_csv(a / "history.csv"), pd.read_csv(b / "history.csv")
    cols = [c for c in ha.columns if c != "epoch_time_s"]
    if ha.shape != hb.shape:
        print(f"history: different number of epochs ({len(ha)} vs {len(hb)})")
        ok = False
    else:
        diff = (ha[cols] != hb[cols]).any(axis=1)
        print(f"history: {len(ha)} epochs, {int(diff.sum())} epoch(s) differ")
        if diff.any():
            ok = False
            d = (ha[cols].select_dtypes("number") - hb[cols].select_dtypes("number")).abs().max()
            print("  max abs difference per column:\n" + d.to_string())

    pa, pb = pd.read_csv(a / "val_predictions.csv"), pd.read_csv(b / "val_predictions.csv")
    same_pred = (pa["pred"] == pb["pred"]).all()
    prob_cols = [c for c in pa.columns if c.startswith("p_")]
    max_prob_diff = float((pa[prob_cols] - pb[prob_cols]).abs().max().max())
    print(f"val predictions identical: {same_pred}; max |prob diff| = {max_prob_diff:.2e}")
    ok &= bool(same_pred) and max_prob_diff == 0.0

    sa = torch.load(ca, map_location="cpu", weights_only=True)
    sb = torch.load(cb, map_location="cpu", weights_only=True)
    diffs = [k for k in sa["state_dict"] if not torch.equal(sa["state_dict"][k], sb["state_dict"][k])]
    print(f"weights: {len(sa['state_dict'])} tensors, {len(diffs)} differ; best epoch {sa['epoch']} vs {sb['epoch']}")
    ok &= not diffs and sa["epoch"] == sb["epoch"]

    print("BIT-EXACT REPRODUCTION" if ok else "RUNS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
