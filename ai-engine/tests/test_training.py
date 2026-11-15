"""Training-loop tests: reproducibility (same seed -> identical run), callbacks."""

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from ivai.config import NEU_PROCESSED_DIR
from ivai.data.dataset import NEUDataset, build_transforms, load_train_stats
from ivai.models.custom_cnn import CustomCNN
from ivai.training.callbacks import EarlyStopping
from ivai.training.trainer import TrainConfig, fit
from ivai.utils.seed import set_seed

needs_data = pytest.mark.skipif(not (NEU_PROCESSED_DIR / "train").exists(), reason="run prepare_splits.py first")
AUG = {"hflip": 0.5, "vflip": 0.5, "brightness": 0.2, "contrast": 0.2}


def test_early_stopping():
    es = EarlyStopping(patience=2)
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.95)
    assert es.step(0.92)  # 2 epochs without improvement


def short_run(seed: int, device: str, tmp_path, amp: bool = True):
    gen = set_seed(seed, deterministic=True)
    mean, std = load_train_stats()
    train = Subset(NEUDataset("train", build_transforms(True, mean, std, AUG)), range(0, 1260, 10))  # 126 images
    val = Subset(NEUDataset("val", build_transforms(False, mean, std)), range(0, 270, 5))  # 54 images
    tl = DataLoader(train, batch_size=16, shuffle=True, generator=gen)
    vl = DataLoader(val, batch_size=32)
    model = CustomCNN()
    res = fit(model, tl, vl, TrainConfig(max_epochs=2, amp=amp), torch.device(device),
              tmp_path / f"{seed}_{device}.pt", log=lambda *_: None)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return [(h["train_loss"], h["val_loss"], h["val_acc"]) for h in res["history"]], state


@needs_data
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_same_seed_gives_identical_run(device, tmp_path):
    h1, w1 = short_run(42, device, tmp_path)
    h2, w2 = short_run(42, device, tmp_path)
    assert h1 == h2  # exact float equality, not approximate
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


@needs_data
def test_different_seed_gives_different_run(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h1, _ = short_run(42, device, tmp_path)
    h2, _ = short_run(123, device, tmp_path)
    assert h1 != h2
