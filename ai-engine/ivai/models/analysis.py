"""Model inspection: parameters, tensor shapes and receptive field."""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())  # BN running mean/var (not learned)
    return {"total": total, "trainable": trainable, "non_trainable_buffers": buffers}


def parameters_per_layer(model: nn.Module) -> list[tuple[str, str, int]]:
    rows = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own:
            rows.append((name, module.__class__.__name__, own))
    return rows


@torch.no_grad()
def shape_trace(model: nn.Module, input_shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Output shape of every leaf module for one forward pass (batch size 1)."""
    trace, hooks = [], []
    for name, m in model.named_modules():
        if not list(m.children()):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: trace.append((n, tuple(o.shape)))))
    was_training = model.training
    model.eval()
    try:
        model(torch.zeros(1, *input_shape))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return trace


def theoretical_receptive_field(model: nn.Module) -> dict[str, int]:
    """Receptive field of one unit of the last feature map, from layer arithmetic.

    r_out = r_in + (k - 1) * j_in ;  j_out = j_in * s   (j = jump / cumulative stride)
    """
    r, j = 1, 1
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            r = r + (k - 1) * j
            j = j * s
    return {"receptive_field": r, "cumulative_stride": j}


def empirical_receptive_field(model: nn.Module, input_size: int = 200, geometric: bool = True) -> int:
    """Measured receptive field: back-propagate from the central unit of the last
    feature map and measure the extent of the non-zero input gradient.

    * ``geometric=True``: MaxPool layers are replaced by AvgPool layers with the
      same kernel/stride, so the gradient flows through every position of each
      pooling window -> measures the geometric receptive field (must equal the
      theoretical value).
    * ``geometric=False``: keeps MaxPool; the gradient only flows through the
      selected maxima, so the measured extent is the (input-dependent) part of
      the receptive field actually used for this input.

    Model in eval mode (BN uses running stats), weights set to positive values so
    that no gradient path is cancelled by ReLU on a positive input.
    """
    import copy

    m = copy.deepcopy(model).eval().double()
    if geometric:
        for parent in m.modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.MaxPool2d):
                    setattr(parent, name, nn.AvgPool2d(child.kernel_size, child.stride))
    for p in m.parameters():
        nn.init.constant_(p, 0.05)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
    x = torch.rand(1, 1, input_size, input_size, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    x.requires_grad_(True)
    feats = m.forward_features(x)
    _, _, h, w = feats.shape
    feats[0, :, h // 2, w // 2].sum().backward()
    nz = (x.grad[0, 0].abs() > 0).nonzero()
    rows = nz[:, 0].max() - nz[:, 0].min() + 1
    cols = nz[:, 1].max() - nz[:, 1].min() + 1
    return int(max(rows, cols))
