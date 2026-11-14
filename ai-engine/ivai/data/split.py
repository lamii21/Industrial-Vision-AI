"""Stratified, group-aware and reproducible train / validation / test split.

* **Stratified**: each subset keeps the class proportions (here 6 x 1/6).
* **Group-aware**: images flagged as duplicates of each other (same ``group``)
  are always placed in the same subset, so no leakage through near-copies.
* **Reproducible**: the result depends only on the index content and the seed.
  The split is also saved as CSV files that are versioned with Git, so the exact
  same files are used by every experiment, even if the code changes later.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "val", "test")


def stratified_group_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_col: str = "label",
    group_col: str = "group",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a new ``split`` column.

    The split is computed on groups (one row per group, labelled by the group's
    majority class) and then expanded to images.
    """
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError(f"Invalid ratios: val={val_ratio}, test={test_ratio}")
    if group_col not in df:
        df = df.assign(**{group_col: range(len(df))})

    groups = (
        df.groupby(group_col)[label_col]
        .agg(lambda s: s.value_counts().index[0])
        .rename("group_label")
        .reset_index()
        .sort_values(group_col)  # deterministic order before shuffling
    )

    rest, test = train_test_split(
        groups, test_size=test_ratio, stratify=groups["group_label"], random_state=seed
    )
    # val_ratio is expressed relative to the full dataset
    rel_val = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        rest, test_size=rel_val, stratify=rest["group_label"], random_state=seed
    )

    assignment = {g: "train" for g in train[group_col]}
    assignment.update({g: "val" for g in val[group_col]})
    assignment.update({g: "test" for g in test[group_col]})

    out = df.copy()
    out["split"] = out[group_col].map(assignment)
    if out["split"].isna().any():
        raise RuntimeError("Some images were not assigned to a split")
    return out


def save_split_csvs(df: pd.DataFrame, splits_dir: Path, prefix: str = "neu_cls") -> dict[str, Path]:
    """Write one CSV per subset (sorted, minimal columns) for versioning."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    cols = ["filename", "label", "label_idx", "group", "pixel_sha256"]
    for split in SPLITS:
        p = splits_dir / f"{prefix}_{split}.csv"
        df.loc[df["split"] == split, cols].sort_values("filename").to_csv(p, index=False)
        paths[split] = p
    return paths


def materialize_split(df: pd.DataFrame, raw_dir: Path, out_dir: Path) -> None:
    """Copy images into ``out_dir/<split>/<class>/`` (torchvision ImageFolder layout)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for row in df.itertuples(index=False):
        dst = out_dir / row.split / row.label / row.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_dir / row.relpath, dst)
