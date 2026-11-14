import pandas as pd
import pytest

from ivai.data.split import stratified_group_split

CLASSES = ["a", "b", "c", "d", "e", "f"]


def make_df(per_class=300, groups=None):
    rows = [{"filename": f"{c}_{i}.jpg", "label": c} for c in CLASSES for i in range(per_class)]
    df = pd.DataFrame(rows)
    df["group"] = range(len(df)) if groups is None else groups(df)
    return df


def test_exact_counts_and_stratification():
    out = stratified_group_split(make_df(), 0.15, 0.15, seed=42)
    table = pd.crosstab(out["label"], out["split"])
    assert (table["train"] == 210).all()
    assert (table["val"] == 45).all()
    assert (table["test"] == 45).all()


def test_reproducible_with_same_seed():
    a = stratified_group_split(make_df(), 0.15, 0.15, seed=42)
    b = stratified_group_split(make_df(), 0.15, 0.15, seed=42)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_different_seed_changes_split():
    a = stratified_group_split(make_df(), 0.15, 0.15, seed=42)
    b = stratified_group_split(make_df(), 0.15, 0.15, seed=7)
    assert (a["split"] != b["split"]).any()


def test_subsets_are_disjoint_and_complete():
    out = stratified_group_split(make_df(), 0.15, 0.15, seed=0)
    assert out["split"].notna().all()
    assert out["filename"].is_unique
    assert set(out["split"]) == {"train", "val", "test"}


def test_duplicate_groups_stay_together():
    # pairs of consecutive images form a duplicate group
    df = make_df(groups=lambda d: d.index // 2)
    out = stratified_group_split(df, 0.15, 0.15, seed=1)
    assert (out.groupby("group")["split"].nunique() == 1).all()


def test_invalid_ratio_raises():
    with pytest.raises(ValueError):
        stratified_group_split(make_df(), 0.6, 0.6, seed=0)
