"""Dataset tests: split usage, test-set lock, leakage, shapes, normalisation, augmentation scope."""

import pandas as pd
import pytest
import torch

from ivai.config import NEU_PROCESSED_DIR, SPLITS_DIR
from ivai.data.dataset import NEUDataset, TestSetAccessError, build_transforms, load_train_stats, read_split

needs_data = pytest.mark.skipif(not (NEU_PROCESSED_DIR / "train").exists(), reason="run prepare_splits.py first")
MEAN, STD = load_train_stats()


def test_test_split_is_locked():
    with pytest.raises(TestSetAccessError):
        read_split("test")
    with pytest.raises(TestSetAccessError):
        NEUDataset("test")


def test_unknown_split():
    with pytest.raises(ValueError):
        read_split("validation")


def test_no_leakage_between_sets():
    """File names and pixel hashes: train, val and test are pairwise disjoint.
    (Only the test *file list* from the index is read, never its images.)"""
    index = pd.read_csv(SPLITS_DIR / "neu_cls_index.csv", usecols=["filename", "split", "pixel_sha256", "group"])
    train, val = read_split("train"), read_split("val")
    test = index[index.split == "test"]
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a.filename) & set(b.filename)
        assert not set(a.pixel_sha256) & set(b.pixel_sha256)
        assert not set(a.group) & set(b.group)  # duplicate groups never cross subsets


def test_split_csvs_unchanged():
    """The Phase 1 CSVs must still match the hashes recorded when they were created."""
    import hashlib
    import json

    meta = json.loads((SPLITS_DIR / "split_meta.json").read_text())
    for split, h in meta["csv_sha256"].items():
        assert hashlib.sha256((SPLITS_DIR / f"neu_cls_{split}.csv").read_bytes()).hexdigest() == h


@needs_data
def test_dataset_sizes_and_shapes():
    ds = NEUDataset("val", build_transforms(False, MEAN, STD))
    assert len(ds) == 270
    x, y = ds[0]
    assert x.shape == (1, 200, 200) and x.dtype == torch.float32
    assert y.dtype == torch.long and 0 <= y.item() < 6
    assert torch.bincount(ds.targets, minlength=6).tolist() == [45] * 6
    assert len(NEUDataset("train")) == 1260


@needs_data
def test_train_normalisation_is_standard():
    ds = NEUDataset("train", build_transforms(False, MEAN, STD))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    assert abs(x.mean().item()) < 1e-3
    assert abs(x.std().item() - 1) < 1e-3


@needs_data
def test_validation_transform_is_deterministic():
    """No augmentation outside the train split."""
    ds = NEUDataset("val", build_transforms(False, MEAN, STD))
    assert torch.equal(ds[5][0], ds[5][0])


def test_eval_transform_has_no_random_op():
    names = [t.__class__.__name__ for t in build_transforms(False, MEAN, STD, {"hflip": 0.5, "vflip": 0.5}).transforms]
    assert names == ["ToDtype", "Normalize"]


def test_train_augmentations_are_flips_and_jitter_only():
    aug = {"hflip": 0.5, "vflip": 0.5, "brightness": 0.2, "contrast": 0.2}
    names = [t.__class__.__name__ for t in build_transforms(True, MEAN, STD, aug).transforms]
    assert names == ["ToDtype", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Lambda", "Normalize"]
    assert not any("Rotation" in n for n in names)


def test_hflip_is_a_mirror():
    x = torch.randint(0, 256, (1, 8, 8), dtype=torch.uint8)
    t = build_transforms(True, 0.0, 1.0, {"hflip": 1.0})
    assert torch.allclose(t(x), torch.flip(x.float() / 255, dims=[2]))


def test_jitter_keeps_range_and_shape():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 200, 200), dtype=torch.uint8)
    t = build_transforms(True, 0.0, 1.0, {"brightness": 0.2, "contrast": 0.2})
    for _ in range(5):
        y = t(x)
        assert y.shape == (1, 200, 200) and y.min() >= 0 and y.max() <= 1
