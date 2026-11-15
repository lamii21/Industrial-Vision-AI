"""Model-level tests: shapes, forward, loss, devices, save/load, size, receptive field."""

import pytest
import torch
from torch import nn

from ivai.models.analysis import count_parameters, empirical_receptive_field, theoretical_receptive_field
from ivai.models.custom_cnn import CustomCNN
from ivai.training.callbacks import ModelCheckpoint, load_checkpoint

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return CustomCNN()


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_output_shape(model, batch):
    out = model.eval()(torch.randn(batch, 1, 200, 200))
    assert out.shape == (batch, 6)


def test_feature_map_shape(model):
    assert model.forward_features(torch.randn(2, 1, 200, 200)).shape == (2, 256, 12, 12)


def test_rejects_wrong_channel_count(model):
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 3, 200, 200))


def test_forward_returns_finite_logits_not_probabilities(model):
    torch.manual_seed(1)
    out = model.eval()(torch.randn(8, 1, 200, 200) * 3)
    assert torch.isfinite(out).all()
    # logits are unconstrained: rows do not sum to 1 (no softmax inside forward)
    assert not torch.allclose(out.sum(1), torch.ones(8), atol=1e-3)


def test_loss_and_gradients(model):
    x, y = torch.randn(4, 1, 200, 200), torch.tensor([0, 1, 2, 5])
    loss = nn.CrossEntropyLoss()(model.train()(x), y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_can_overfit_tiny_batch(model):
    """Sanity check: the optimisation pipeline is able to decrease the loss."""
    torch.manual_seed(0)
    x, y = torch.randn(8, 1, 200, 200), torch.arange(8) % 6
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    first = crit(model(x), y).item()
    for _ in range(30):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_parameter_count(model):
    p = count_parameters(model)
    # 288 + 18432 + 73728 + 294912 (convs, no bias) + 2*(32+64+128+256) (BN) + 256*6 + 6 (linear)
    assert p["total"] == p["trainable"] == 389_862


def test_receptive_field(model):
    assert theoretical_receptive_field(model) == {"receptive_field": 46, "cumulative_stride": 16}
    assert empirical_receptive_field(model, geometric=True) == 46


@cuda
def test_cpu_gpu_consistency(model):
    model.eval()
    x = torch.randn(4, 1, 200, 200)
    out_cpu = model(x)
    out_gpu = model.cuda()(x.cuda())
    assert out_gpu.device.type == "cuda"
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-3, rtol=1e-3)
    back = model.cpu()(x)
    assert torch.allclose(out_cpu, back, atol=1e-6)


def test_save_and_load_weights(model, tmp_path):
    model.eval()
    x = torch.randn(2, 1, 200, 200)
    ref = model(x)
    ckpt = ModelCheckpoint(tmp_path / "best.pt")
    assert ckpt.step(0.5, epoch=3, model=model)
    assert not ckpt.step(0.7, epoch=4, model=model)  # worse -> not saved
    fresh = CustomCNN().eval()
    meta = load_checkpoint(fresh, tmp_path / "best.pt")
    assert meta["epoch"] == 3 and meta["val_loss"] == 0.5
    assert torch.equal(fresh(x), ref)
