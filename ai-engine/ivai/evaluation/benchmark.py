"""Deployment cost measurements (docs/phase3_protocol.md §9.5).

Latency protocol: eval mode + ``torch.inference_mode()``, fixed input, warm-up then timed
iterations, CUDA synchronised around each call, **median and IQR** reported (the laptop
timing varies up to 2x, Phase 2). Preprocessing is measured separately.
"""

from __future__ import annotations

import ctypes
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> float:
    """Multiply-accumulate operations of one forward pass (convolutions and matmuls only),
    measured on CPU in eval mode."""
    was_training = model.training
    device = next(model.parameters()).device
    model.eval().cpu()
    try:
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.zeros(*input_shape))
    finally:
        model.to(device).train(was_training)
    return fc.get_total_flops() / 2


def state_dict_size_mb(model: nn.Module) -> float:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 2**20


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(times_ms: list[float], batch: int, n_warmup: int) -> dict:
    t = np.asarray(times_ms)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "iqr_ms": float(q3 - q1),
            "mean_ms": float(t.mean()), "min_ms": float(t.min()), "throughput_img_s": float(batch / med * 1000),
            "n_iter": int(len(t)), "n_warmup": int(n_warmup), "batch": int(batch)}


def measure_latency(model: nn.Module, input_shape: tuple[int, ...], device: str | torch.device = "cpu",
                    n_warmup: int = 50, n_iter: int = 200, amp: bool = False, num_threads: int | None = None) -> dict:
    """Latency of ``model(x)`` for a fixed input of shape ``input_shape`` (batch first)."""
    device = torch.device(device)
    if amp and device.type != "cuda":
        raise ValueError("AMP latency is only measured on CUDA")
    old_threads = torch.get_num_threads()
    if num_threads is not None and device.type == "cpu":
        torch.set_num_threads(num_threads)
    model = model.eval().to(device)
    x = torch.randn(*input_shape, generator=torch.Generator().manual_seed(0)).to(device)
    times = []
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            for _ in range(n_warmup):
                model(x)
            _sync(device)
            for _ in range(n_iter):
                t0 = time.perf_counter()
                model(x)
                _sync(device)
                times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old_threads)
    out = _stats(times, input_shape[0], n_warmup)
    out.update({"device": device.type, "precision": "fp16-amp" if amp else "fp32",
                "num_threads": num_threads if device.type == "cpu" else None})
    return out


def measure_preprocessing(transform, image: torch.Tensor, n_warmup: int = 50, n_iter: int = 200) -> dict:
    """Latency of the evaluation transform on one uint8 image (CPU)."""
    for _ in range(n_warmup):
        transform(image)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        transform(image)
        times.append((time.perf_counter() - t0) * 1000)
    return _stats(times, 1, n_warmup)


def on_ac_power() -> bool | None:
    """True on mains power, False on battery, None if unknown (non-Windows)."""
    if platform.system() != "Windows":
        return None

    class _PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    s = _PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None
    return {0: False, 1: True}.get(s.ACLineStatus)
