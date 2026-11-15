"""Phase 3 - data pipeline, trainer extensions and NON-REGRESSION of the Phase 2 custom CNN."""

import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms import v2

from ivai.config import CONFIGS_DIR, NEU_PROCESSED_DIR, REPORTS_DIR
from ivai.data.dataset import (IMAGENET_MEAN, IMAGENET_STD, NEUDataset, build_transforms, load_train_stats,
                               resolve_normalization)
from ivai.data.subsets import make_train_loader
from ivai.models.analysis import count_parameters, empirical_receptive_field, theoretical_receptive_field
from ivai.models.registry import build_model, default_input, parameter_groups
from ivai.training.trainer import TrainConfig, fit
from ivai.utils.seed import set_seed

needs_data = pytest.mark.skipif(not (NEU_PROCESSED_DIR / "train").exists(), reason="run prepare_splits.py first")
MEAN, STD = load_train_stats()
AUG = {"hflip": 0.5, "vflip": 0.5, "brightness": 0.2, "contrast": 0.2}
HISTORY_KEYS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "best"]


# ---------------------------------------------------------------- transforms
def test_phase2_transform_pipeline_unchanged():
    """Same operations, same parameters as Phase 2 for the custom CNN (1 channel, train stats)."""
    for train in (True, False):
        tf = build_transforms(train, MEAN, STD, AUG if train else None)
        names = [type(t).__name__ for t in tf.transforms]
        assert names == (["ToDtype", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Lambda", "Normalize"]
                         if train else ["ToDtype", "Normalize"])
        norm = tf.transforms[-1]
        assert norm.mean == [MEAN] and norm.std == [STD]


def test_phase2_eval_transform_values():
    x = torch.randint(0, 256, (1, 200, 200), dtype=torch.uint8)
    out = build_transforms(False, MEAN, STD)(x)
    assert torch.equal(out, (x.float() / 255 - MEAN) / STD) or torch.allclose(out, (x.float() / 255 - MEAN) / STD, atol=1e-6)
    assert out.shape == (1, 200, 200)


def test_imagenet_transform_replicates_grey_channel():
    x = torch.randint(0, 256, (1, 200, 200), dtype=torch.uint8)
    mean, std = resolve_normalization("imagenet")
    assert (mean, std) == (IMAGENET_MEAN, IMAGENET_STD)
    out = build_transforms(False, mean, std, channels=3)(x)
    assert out.shape == (3, 200, 200)
    grey = [out[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c] for c in range(3)]
    assert torch.allclose(grey[0], grey[1], atol=1e-6) and torch.allclose(grey[0], grey[2], atol=1e-6)
    assert torch.allclose(grey[0], x[0].float() / 255, atol=1e-6)


def test_imagenet_train_transform_augments_before_replication():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 200, 200), dtype=torch.uint8)
    tf = build_transforms(True, IMAGENET_MEAN, IMAGENET_STD, AUG, channels=3)
    names = [type(t).__name__ for t in tf.transforms]
    assert names.index("ColorJitter") < names.index("Lambda", names.index("ColorJitter") + 2) < names.index("Normalize")
    out = tf(x)
    grey = [out[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c] for c in range(3)]
    assert torch.allclose(grey[0], grey[2], atol=1e-6)  # same augmentation on the 3 channels
    assert not any(isinstance(t, (v2.RandomRotation, v2.RandomAffine)) for t in tf.transforms)


def test_transform_argument_checks():
    with pytest.raises(ValueError):
        build_transforms(False, IMAGENET_MEAN, IMAGENET_STD, channels=1)
    with pytest.raises(ValueError):
        build_transforms(False, MEAN, STD, channels=3)
    with pytest.raises(ValueError):
        resolve_normalization("unknown")


# ---------------------------------------------------------------- trainer
def _toy(n=64):
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(n, 3, 16, 16, generator=g), torch.randint(0, 6, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0)), \
        DataLoader(TensorDataset(x, y), batch_size=32)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 14 * 14, 8))
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.fc(torch.relu(self.body(x)))


def test_single_group_history_keys_unchanged(tmp_path):
    tl, vl = _toy()
    res = fit(Tiny(), tl, vl, TrainConfig(max_epochs=2, amp=False), torch.device("cpu"), tmp_path / "a.pt", log=lambda *_: None)
    assert list(res["history"][0]) == HISTORY_KEYS


def test_param_groups_lr_ratio_kept_by_scheduler(tmp_path):
    torch.manual_seed(0)
    m = Tiny()
    groups = [{"params": list(m.body.parameters()), "lr": 1e-4, "name": "backbone"},
              {"params": list(m.fc.parameters()), "lr": 1e-3, "name": "head"}]
    tl, vl = _toy()
    cfg = TrainConfig(max_epochs=6, amp=False, scheduler_patience=0, early_stopping_patience=100, head_lr=1e-3, lr=1e-4)
    res = fit(m, tl, vl, cfg, torch.device("cpu"), tmp_path / "b.pt", log=lambda *_: None, param_groups=groups)
    h = res["history"]
    assert {"lr_backbone", "lr_head"} <= set(h[0])
    assert h[0]["lr_backbone"] == 1e-4 and h[0]["lr_head"] == 1e-3 and h[0]["lr"] == 1e-4
    for row in h:
        assert row["lr_head"] / row["lr_backbone"] == pytest.approx(10.0)


def test_frozen_parameters_not_updated_by_fit(tmp_path):
    torch.manual_seed(0)
    m = Tiny()
    for p in m.body.parameters():
        p.requires_grad = False
    before = {k: v.clone() for k, v in m.body.state_dict().items()}
    tl, vl = _toy()
    fit(m, tl, vl, TrainConfig(max_epochs=2, amp=False), torch.device("cpu"), tmp_path / "c.pt", log=lambda *_: None,
        param_groups=[{"params": list(m.fc.parameters()), "lr": 1e-3, "name": "head"}])
    assert all(torch.equal(before[k], v) for k, v in m.body.state_dict().items())


# ---------------------------------------------------------------- non-regression of Phase 2
@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase 2 history was produced on CUDA")
def test_custom_cnn_reproduces_phase2_history(tmp_path):
    """Rebuild the seed-42 run exactly as scripts/train.py does (same calls, same order of
    random-number consumption) and compare the first 2 epochs with the Phase 2 history.csv."""
    cfg = yaml.safe_load((CONFIGS_DIR / "custom_cnn.yaml").read_text(encoding="utf-8"))
    mcfg, tcfg = cfg["model"], cfg["training"]
    gen = set_seed(42, deterministic=tcfg["deterministic"])
    inp = cfg["data"].get("input") or default_input(mcfg)
    mean, std = resolve_normalization(inp["normalization"])
    train_ds = NEUDataset("train", build_transforms(True, mean, std, cfg["data"]["augment"], channels=inp["channels"]))
    val_ds = NEUDataset("val", build_transforms(False, mean, std, channels=inp["channels"]))
    bs = cfg["data"]["batch_size"]
    tl = make_train_loader(train_ds, bs, gen, True, virtual_epoch_size=None)
    vl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=0, pin_memory=True)
    model = build_model(mcfg)
    count_parameters(model)
    theoretical_receptive_field(model)
    empirical_receptive_field(model)
    train_cfg = TrainConfig(**{**{k: v for k, v in tcfg.items() if k in TrainConfig.__dataclass_fields__}, "max_epochs": 2})
    groups = parameter_groups(model, mcfg, train_cfg.lr, train_cfg.head_lr)
    assert groups is None
    res = fit(model, tl, vl, train_cfg, torch.device("cuda"), tmp_path / "nonreg.pt", log=lambda *_: None, param_groups=groups)
    ref = pd.read_csv(REPORTS_DIR / "experiments" / "custom_cnn" / "seed_42" / "history.csv").iloc[:2]
    for h, (_, r) in zip(res["history"], ref.iterrows()):
        for k in ("lr", "train_loss", "train_acc", "val_loss", "val_acc"):
            assert abs(h[k] - r[k]) <= 1e-6, (h["epoch"], k, h[k], r[k])  # CSV stores 6 decimals


@needs_data
def test_subset_dataset_feeds_virtual_epochs():
    ds = NEUDataset("train", build_transforms(True, *resolve_normalization("imagenet"), AUG, channels=3))
    from ivai.data.subsets import VIRTUAL_EPOCH_SIZE, stratified_nested_subset
    idx = stratified_nested_subset(ds.df, 0.10, 42)
    loader = make_train_loader(Subset(ds, idx.tolist()), 32, torch.Generator().manual_seed(42), False, VIRTUAL_EPOCH_SIZE)
    n = 0
    for x, y in loader:
        assert x.shape[1:] == (3, 200, 200)
        n += len(y)
    assert n == VIRTUAL_EPOCH_SIZE
