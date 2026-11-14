"""Verify the materialised split: no leakage, stratification, ratios, readable files.

Usage:  python scripts/verify_splits.py
Exit code 0 if every check passes.
"""

from __future__ import annotations

import sys

import pandas as pd

from ivai.config import NEU_PROCESSED_DIR, SPLITS_DIR, load_data_config
from ivai.data import verify
from ivai.data.split import SPLITS


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    cfg = load_data_config()
    index = pd.read_csv(SPLITS_DIR / "neu_cls_index.csv")
    splits = {s: pd.read_csv(SPLITS_DIR / f"neu_cls_{s}.csv") for s in SPLITS}

    # The materialised folders must contain exactly the files listed in the CSVs.
    on_disk = {s: sorted(p.name for p in (NEU_PROCESSED_DIR / s).rglob("*.jpg")) for s in SPLITS}
    folders_match = all(on_disk[s] == sorted(splits[s]["filename"]) for s in SPLITS)

    results = [
        verify.CheckResult("Dossiers = CSV de split", folders_match, ", ".join(f"{s}={len(on_disk[s])}" for s in SPLITS)),
        verify.check_disjoint_files(splits),
        verify.check_groups_not_split(index),
        verify.check_ratios(index, cfg),
        verify.check_stratification(index, cfg),
        verify.check_files_readable(index, NEU_PROCESSED_DIR, cfg),
        verify.check_no_exact_duplicates_across(splits, NEU_PROCESSED_DIR),
        verify.check_no_near_duplicates_across(splits, NEU_PROCESSED_DIR, cfg),
    ]
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
