"""Post-split integrity checks (leakage, balance, file integrity).

Every check returns a ``CheckResult``; ``scripts/verify_splits.py`` prints them
and exits with a non-zero code if any check fails, so it can be used in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ivai.config import DataConfig
from ivai.data.duplicates import compute_hashes, hamming_matrix, pixel_correlation, TRANSFORMS
from ivai.data.index import load_gray, pixel_sha256
from ivai.data.split import SPLITS


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_disjoint_files(splits: dict[str, pd.DataFrame]) -> CheckResult:
    overlaps = []
    for a_i, a in enumerate(SPLITS):
        for b in SPLITS[a_i + 1 :]:
            common = set(splits[a]["filename"]) & set(splits[b]["filename"])
            if common:
                overlaps.append(f"{a}/{b}: {len(common)}")
    return CheckResult("Aucun fichier partagé entre subsets", not overlaps, "; ".join(overlaps) or "0 overlap")


def check_no_exact_duplicates_across(splits: dict[str, pd.DataFrame], processed_dir: Path) -> CheckResult:
    """Recompute pixel hashes from the materialised copies (not from the CSV)."""
    owner: dict[str, str] = {}
    leaks = []
    for split in SPLITS:
        for row in splits[split].itertuples(index=False):
            h = pixel_sha256(load_gray(processed_dir / split / row.label / row.filename))
            if h in owner and owner[h] != split:
                leaks.append(f"{row.filename} ({owner[h]}<->{split})")
            owner.setdefault(h, split)
    return CheckResult("Aucun doublon exact entre subsets", not leaks, f"{len(leaks)} leak(s)" + (": " + ", ".join(leaks[:5]) if leaks else ""))


def check_no_near_duplicates_across(
    splits: dict[str, pd.DataFrame], processed_dir: Path, cfg: DataConfig
) -> CheckResult:
    """pHash (8 transforms) + pixel correlation between every pair of subsets."""
    imgs, hashes = {}, {}
    for split in SPLITS:
        imgs[split] = [load_gray(processed_dir / split / r.label / r.filename) for r in splits[split].itertuples(index=False)]
        hashes[split] = compute_hashes(imgs[split])
    leaks, min_dist = [], 10**6
    for a_i, a in enumerate(SPLITS):
        for b in SPLITS[a_i + 1 :]:
            base = hashes[a][:, 0, :]
            for t in range(len(TRANSFORMS)):
                d = hamming_matrix(base, hashes[b][:, t, :])
                min_dist = min(min_dist, int(d.min()))
                for i, j in zip(*np.nonzero(d <= cfg.phash_max_distance)):
                    corr = pixel_correlation(imgs[a][i], np.ascontiguousarray(TRANSFORMS[t](imgs[b][j])))
                    if corr >= cfg.min_pixel_correlation:
                        leaks.append(f"{splits[a].iloc[i].filename}<->{splits[b].iloc[j].filename} (d={d[i, j]}, r={corr:.3f})")
    leaks = sorted(set(leaks))
    detail = f"{len(leaks)} leak(s), distance pHash minimale inter-subsets = {min_dist}"
    if leaks:
        detail += ": " + ", ".join(leaks[:5])
    return CheckResult("Aucun quasi-doublon entre subsets", not leaks, detail)


def check_groups_not_split(df: pd.DataFrame) -> CheckResult:
    n_split = df.groupby("group")["split"].nunique()
    bad = int((n_split > 1).sum())
    return CheckResult("Groupes de doublons non scindés", bad == 0, f"{bad} groupe(s) scindé(s)")


def check_stratification(df: pd.DataFrame, cfg: DataConfig, tol: float = 0.01) -> CheckResult:
    """Each class proportion in each subset must be within ``tol`` of the global proportion."""
    global_p = df["label"].value_counts(normalize=True)
    worst = 0.0
    for split in SPLITS:
        p = df.loc[df["split"] == split, "label"].value_counts(normalize=True).reindex(cfg.classes, fill_value=0)
        worst = max(worst, float((p - global_p.reindex(cfg.classes)).abs().max()))
    table = pd.crosstab(df["label"], df["split"]).reindex(columns=list(SPLITS))
    return CheckResult(
        "Stratification respectée", worst <= tol, f"écart max de proportion = {worst:.4f}\n{table.to_string()}"
    )


def check_ratios(df: pd.DataFrame, cfg: DataConfig, tol: float = 0.01) -> CheckResult:
    r = df["split"].value_counts(normalize=True)
    target = {"train": 1 - cfg.val_ratio - cfg.test_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio}
    err = max(abs(r.get(s, 0) - target[s]) for s in SPLITS)
    sizes = ", ".join(f"{s}={int((df['split'] == s).sum())}" for s in SPLITS)
    return CheckResult("Proportions 70/15/15", err <= tol, f"{sizes} (écart max {err:.4f})")


def check_files_readable(df: pd.DataFrame, processed_dir: Path, cfg: DataConfig) -> CheckResult:
    bad = []
    for row in df.itertuples(index=False):
        p = processed_dir / row.split / row.label / row.filename
        try:
            g = load_gray(p)
            if g.shape != tuple(cfg.expected_size)[::-1]:
                bad.append(f"{row.filename}: shape {g.shape}")
        except Exception as exc:  # noqa: BLE001 - report any decoding problem
            bad.append(f"{row.filename}: {exc}")
    return CheckResult("Fichiers lisibles et taille attendue", not bad, f"{len(df) - len(bad)}/{len(df)} OK" + (": " + "; ".join(bad[:5]) if bad else ""))
