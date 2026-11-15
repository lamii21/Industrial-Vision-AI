"""Phase 3 - latency/cost helpers (E4) and validation perturbations (E5)."""

import pytest
import torch
from torch import nn

from ivai.evaluation.benchmark import count_macs, measure_latency, measure_preprocessing, state_dict_size_mb
from ivai.evaluation.robustness import PERTURBATIONS, PerturbedDataset, perturb


class Counting(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.lin(x)


def test_latency_warmup_iterations_and_stats():
    m = Counting()
    r = measure_latency(m, (8, 4), "cpu", n_warmup=3, n_iter=7, num_threads=2)
    assert m.calls == 10
    assert r["n_iter"] == 7 and r["n_warmup"] == 3 and r["batch"] == 8
    assert 0 < r["q1_ms"] <= r["median_ms"] <= r["q3_ms"]
    assert r["throughput_img_s"] == pytest.approx(8 / r["median_ms"] * 1000)
    assert r["precision"] == "fp32" and r["num_threads"] == 2
    assert not m.training  # eval mode


def test_latency_restores_thread_count():
    before = torch.get_num_threads()
    measure_latency(Counting(), (1, 4), "cpu", 1, 2, num_threads=1)
    assert torch.get_num_threads() == before


def test_amp_latency_cpu_refused():
    with pytest.raises(ValueError):
        measure_latency(Counting(), (1, 4), "cpu", 1, 1, amp=True)


def test_macs_and_size_known_layers():
    conv = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1, bias=False))
    assert count_macs(conv, (1, 1, 10, 10)) == 900          # 10*10 outputs x 9 MACs
    lin = nn.Linear(10, 5)                                 # 55 fp32 parameters
    assert state_dict_size_mb(lin) == pytest.approx(55 * 4 / 2**20)


def test_preprocessing_timer():
    r = measure_preprocessing(lambda x: x.float() / 255, torch.zeros(1, 8, 8, dtype=torch.uint8), 2, 5)
    assert r["n_iter"] == 5 and r["median_ms"] >= 0


def test_perturbation_levels_match_protocol():
    assert PERTURBATIONS == (("brightness", 0.6), ("brightness", 1.4), ("contrast", 0.6), ("contrast", 1.4),
                             ("gaussian_noise", 0.02), ("gaussian_noise", 0.05), ("gaussian_blur", 1.0),
                             ("gaussian_blur", 2.0), ("jpeg", 50), ("jpeg", 20))


@pytest.mark.parametrize("kind,level", PERTURBATIONS)
def test_perturbations_deterministic_and_in_range(kind, level):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(30, 220, (1, 64, 64), dtype=torch.uint8, generator=g)
    a, b = perturb(x, kind, level, 3), perturb(x, kind, level, 3)
    assert torch.equal(a, b) and a.shape == x.shape and a.dtype == torch.float32
    assert a.min() >= 0 and a.max() <= 1
    assert not torch.equal(a, x.float() / 255)


def test_noise_depends_on_image_index_and_identity_levels():
    x = torch.full((1, 16, 16), 128, dtype=torch.uint8)
    assert not torch.equal(perturb(x, "gaussian_noise", 0.05, 0), perturb(x, "gaussian_noise", 0.05, 1))
    assert torch.allclose(perturb(x, "brightness", 1.0, 0), x.float() / 255)
    with pytest.raises(TypeError):
        perturb(x.float(), "brightness", 0.6, 0)


def test_perturbed_dataset_clean_is_standard_pipeline():
    raw = [(torch.randint(0, 256, (1, 8, 8), dtype=torch.uint8), 0) for _ in range(3)]
    tf = lambda x: x.float() / 255 if x.dtype == torch.uint8 else x  # noqa: E731
    clean = PerturbedDataset(raw, "clean", None, tf)
    assert all(torch.equal(clean[i][0], raw[i][0].float() / 255) for i in range(3))
    blurred = PerturbedDataset(raw, "gaussian_blur", 1.0, tf)
    assert len(blurred) == 3 and blurred[0][1] == 0
