"""Phase 3 - torchvision models, registry, linear probe, deterministic pooling.
No weights are downloaded: architectures are built with ``pretrained=False``; the test
that compares with the ImageNet weights is skipped while they are not in the local cache."""

import copy

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from ivai.models.analysis import count_parameters
from ivai.models.pretrained import (HEAD_MODULE, WEIGHTS, WeightsNotAvailableError, build_torchvision_model,
                                    head_parameter_names, load_imagenet_state_dict, weights_cache_path)
from ivai.models.registry import build_model, default_input, parameter_groups

# Measured in Phase 3 step 1 (protocol §5).
EXPECTED_PARAMS = {"resnet50": 23_520_326, "mobilenetv2": 2_231_558}
EXPECTED_HEAD = {"resnet50": 12_294, "mobilenetv2": 7_686}
ARCHS = ["resnet50", "mobilenetv2"]


@pytest.fixture(scope="module", params=ARCHS)
def arch(request):
    return request.param


def scratch(name, **kw):
    return build_torchvision_model(name, pretrained=False, **kw)


def test_parameter_counts_with_six_class_head(arch):
    m = scratch(arch)
    assert count_parameters(m)["total"] == EXPECTED_PARAMS[arch]
    assert sum(p.numel() for n, p in m.named_parameters() if n in head_parameter_names(m, arch)) == EXPECTED_HEAD[arch]


def test_output_is_six_logits(arch):
    m = scratch(arch).eval()
    out = m(torch.randn(2, 3, 200, 200))
    assert out.shape == (2, 6)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out.softmax(1).sum(1), out.sum(1))  # logits, not probabilities


def test_architecture_unchanged_except_last_layer(arch):
    from torchvision.models import mobilenet_v2, resnet50
    ref = resnet50(weights=None) if arch == "resnet50" else mobilenet_v2(weights=None)
    m = scratch(arch)
    ref_keys = {k: v.shape for k, v in ref.state_dict().items()}
    keys = {k: v.shape for k, v in m.state_dict().items()}
    assert ref_keys.keys() == keys.keys()
    changed = {k for k in keys if keys[k] != ref_keys[k]}
    head = "fc." if arch == "resnet50" else "classifier.1."
    assert changed == {head + "weight", head + "bias"}


def test_global_pooling_dispatches_to_mean(arch):
    """AdaptiveAvgPool2d with a 1x1 output must run as ``mean`` (deterministic on CUDA), not as
    the ``_adaptive_avg_pool2d`` kernel whose backward is non-deterministic (protocol §13.2)."""

    class OpLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.add(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    m = scratch(arch)
    x = torch.randn(2, 3, 64, 64)
    with OpLog() as log:
        m(x).sum().backward()
    assert "mean" in log.ops
    assert not {o for o in log.ops if "adaptive_avg_pool" in o}


def test_no_implicit_download(tmp_path, monkeypatch, arch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(WeightsNotAvailableError):
        build_torchvision_model(arch, pretrained=True)
    assert not any(tmp_path.rglob("*.pth"))


@pytest.mark.skipif(not all(weights_cache_path(w).exists() for w in WEIGHTS.values()),
                    reason="ImageNet weights not downloaded yet (scripts/download_weights.py)")
def test_imagenet_weights_are_loaded(arch):
    m = build_torchvision_model(arch, pretrained=True)
    ref = load_imagenet_state_dict(WEIGHTS[arch])
    head = "fc." if arch == "resnet50" else "classifier.1."
    # The 2017 ResNet50 V1 file predates the BatchNorm ``num_batches_tracked`` buffers
    # (PyTorch 0.4.1). BatchNorm fills them in on load; they are unused with momentum=0.1.
    missing = [k for k in m.state_dict() if k not in ref]
    assert all(k.endswith("num_batches_tracked") for k in missing), missing
    assert all(bn.momentum == 0.1 for bn in m.modules() if isinstance(bn, torch.nn.modules.batchnorm._BatchNorm))
    for k, v in m.state_dict().items():
        if not k.startswith(head) and k not in missing:
            assert torch.equal(v, ref[k]), k


def test_probe_requires_pretrained():
    with pytest.raises(ValueError):
        build_torchvision_model("resnet50", pretrained=False, freeze_backbone=True)


def _probe(arch):
    """Linear probe built on random weights (the freezing logic does not depend on the values)."""
    from ivai.models.pretrained import HEAD_MODULE, _freeze_backbone
    m = scratch(arch)
    _freeze_backbone(m, HEAD_MODULE[arch])
    return m


def test_probe_trains_head_only(arch):
    m = _probe(arch)
    head = head_parameter_names(m, arch)
    assert {n for n, p in m.named_parameters() if p.requires_grad} == head
    assert count_parameters(m)["trainable"] == EXPECTED_HEAD[arch]


def test_probe_keeps_backbone_in_eval_mode(arch):
    m = _probe(arch)
    m.train()
    assert m.training
    head_mod = m.fc if arch == "resnet50" else m.classifier
    for name, mod in m.named_modules():
        if mod is m:
            continue
        in_head = any(mod is h for h in head_mod.modules())
        assert mod.training == in_head, name
    # deepcopy keeps the behaviour (subclass, not a patched bound method)
    c = copy.deepcopy(m)
    c.train()
    assert all(not mod.training for mod in c.modules() if isinstance(mod, nn.BatchNorm2d))


def test_probe_step_leaves_backbone_and_bn_stats_unchanged(arch):
    torch.manual_seed(0)
    m = _probe(arch)
    groups = parameter_groups(m, {"name": arch, "freeze_backbone": True}, lr=1e-3, head_lr=None)
    opt = torch.optim.Adam(groups)
    prefix = HEAD_MODULE[arch] + "."
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.train()
    loss = nn.functional.cross_entropy(m(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    opt.step()
    after = m.state_dict()
    for k in before:
        if not k.startswith(prefix):  # weights, running_mean/var, num_batches_tracked
            assert torch.equal(before[k], after[k]), k
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))


def test_registry_builds_every_model():
    cnn = build_model({"name": "custom_cnn", "in_channels": 1, "num_classes": 6, "channels": [32, 64, 128, 256], "dropout": 0.5})
    assert count_parameters(cnn)["total"] == 389_862
    for arch in ARCHS:
        assert count_parameters(build_model({"name": arch, "num_classes": 6, "pretrained": False}))["total"] == EXPECTED_PARAMS[arch]
    with pytest.raises(ValueError):
        build_model({"name": "vit"})


def test_default_input_convention():
    assert default_input({"name": "custom_cnn"}) == {"channels": 1, "normalization": "dataset"}
    for arch in ARCHS:
        assert default_input({"name": arch}) == {"channels": 3, "normalization": "imagenet"}


def test_parameter_groups(arch):
    cnn_cfg = {"name": "custom_cnn"}
    assert parameter_groups(build_model({**cnn_cfg}), cnn_cfg, 1e-3, None) is None  # historical path
    m = scratch(arch)
    g = parameter_groups(m, {"name": arch}, lr=1e-4, head_lr=1e-3)
    assert [x["name"] for x in g] == ["backbone", "head"]
    assert [x["lr"] for x in g] == [1e-4, 1e-3]
    ids = [id(p) for x in g for p in x["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))  # a partition of all parameters
    head_ids = {id(p) for n, p in m.named_parameters() if n in head_parameter_names(m, arch)}
    assert {id(p) for p in g[1]["params"]} == head_ids
    with pytest.raises(ValueError):
        parameter_groups(_probe(arch), {"name": arch, "freeze_backbone": True}, 1e-3, 1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_training_is_bit_identical(arch):
    """Two short AMP training runs with deterministic algorithms give identical weights."""
    from ivai.utils.seed import set_seed

    def run():
        set_seed(0, deterministic=True)
        m = scratch(arch).cuda()
        groups = parameter_groups(m, {"name": arch}, 1e-4, 1e-3)
        opt = torch.optim.Adam(groups)
        scaler = torch.amp.GradScaler("cuda")
        g = torch.Generator().manual_seed(1)
        m.train()
        for _ in range(3):
            x, y = torch.randn(8, 3, 200, 200, generator=g).cuda(), torch.randint(0, 6, (8,), generator=g).cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    a, b = run(), run()
    assert all(torch.equal(a[k], b[k]) for k in a)
