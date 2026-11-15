"""Phase 3 - E3 stratified nested subsets and virtual epochs (protocol §4, §8)."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from ivai.config import SPLITS_DIR
from ivai.data.dataset import read_split
from ivai.data.subsets import (VIRTUAL_EPOCH_SIZE, fraction_tag, make_train_loader, per_class_count,
                               stratified_nested_subset)

TRAIN = read_split("train")


def test_per_class_counts_match_protocol():
    assert [per_class_count(210, f) for f in (0.10, 0.25, 0.50, 1.0)] == [21, 52, 105, 210]
    with pytest.raises(ValueError):
        per_class_count(210, 0)


@pytest.mark.parametrize("fraction,total,real", [(0.10, 126, 0.1000), (0.25, 312, 0.2476), (0.50, 630, 0.5000)])
def test_subset_sizes_and_stratification(fraction, total, real):
    idx = stratified_nested_subset(TRAIN, fraction, seed=42)
    assert len(idx) == len(set(idx)) == total
    assert round(len(idx) / len(TRAIN), 4) == real
    counts = TRAIN.iloc[idx]["label"].value_counts()
    assert set(counts) == {total // 6} and len(counts) == 6


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_subsets_are_nested(seed):
    s10, s25, s50 = (set(stratified_nested_subset(TRAIN, f, seed)) for f in (0.10, 0.25, 0.50))
    assert s10 < s25 < s50


def test_subsets_reproducible_and_seed_dependent():
    a = stratified_nested_subset(TRAIN, 0.10, 42)
    assert np.array_equal(a, stratified_nested_subset(TRAIN, 0.10, 42))
    assert not np.array_equal(a, stratified_nested_subset(TRAIN, 0.10, 123))


def test_subset_does_not_depend_on_row_order():
    shuffled = TRAIN.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = set(TRAIN.iloc[stratified_nested_subset(TRAIN, 0.25, 42)]["filename"])
    b = set(shuffled.iloc[stratified_nested_subset(shuffled, 0.25, 42)]["filename"])
    assert a == b


def test_subsets_contain_train_files_only():
    index = pd.read_csv(SPLITS_DIR / "neu_cls_index.csv", usecols=["filename", "split"])
    not_train = set(index.loc[index.split != "train", "filename"])  # val + test file NAMES only
    for f in (0.10, 0.25, 0.50):
        assert not set(TRAIN.iloc[stratified_nested_subset(TRAIN, f, 42)]["filename"]) & not_train
    with pytest.raises(ValueError):
        stratified_nested_subset(index.assign(label_idx=0), 0.1, 42)


def test_subset_draw_does_not_touch_torch_rng():
    state = torch.get_rng_state()
    stratified_nested_subset(TRAIN, 0.10, 42)
    assert torch.equal(state, torch.get_rng_state())


def test_fraction_tags():
    assert [fraction_tag(f) for f in (0.10, 0.25, 0.50)] == ["f010", "f025", "f050"]


@pytest.mark.parametrize("n,expected", [(126, {10}), (312, {4, 5}), (630, {2})])
def test_virtual_epoch_reuse(n, expected):
    ds = TensorDataset(torch.arange(n))
    g = torch.Generator().manual_seed(42)
    loader = make_train_loader(ds, 32, g, pin_memory=False, virtual_epoch_size=VIRTUAL_EPOCH_SIZE)
    assert len(loader) == 40
    seen = Counter()
    for (b,) in loader:
        seen.update(b.tolist())
    assert sum(seen.values()) == VIRTUAL_EPOCH_SIZE
    assert set(seen.values()) == expected
    assert len(seen) == n  # every image is used in each virtual epoch


def test_full_fraction_virtual_sampler_equals_historical_order():
    ds = TensorDataset(torch.arange(1260))

    def orders(loader):
        return [torch.cat([b[0] for b in loader]).tolist() for _ in range(4)]

    hist = make_train_loader(ds, 32, torch.Generator().manual_seed(42), False, None)
    virt = make_train_loader(ds, 32, torch.Generator().manual_seed(42), False, VIRTUAL_EPOCH_SIZE)
    assert orders(hist) == orders(virt)
    assert hist.sampler.num_samples == 1260 and len(hist) == 40
