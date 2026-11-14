"""Check the Python / CUDA / PyTorch environment and run a real computation on the GPU.

Usage:  python scripts/check_env.py
Exit code 0 if everything required is present, 1 otherwise.
"""

from __future__ import annotations

import importlib
import platform
import sys
import time

REQUIRED = ["numpy", "pandas", "matplotlib", "sklearn", "scipy", "PIL", "cv2", "yaml", "torch", "torchvision"]


def main() -> int:
    ok = True
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info < (3, 11):
        print("  !! Python >= 3.11 required")
        ok = False

    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<12}: {getattr(mod, '__version__', 'ok')}")
        except ImportError as exc:
            print(f"{name:<12}: MISSING ({exc})")
            ok = False
    if not ok:
        return 1

    import torch

    print(f"\nCUDA build  : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("GPU         : NOT AVAILABLE -> training would run on CPU (slow). See docs/phase1.md.")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU         : {props.name}, {props.total_memory / 1024**3:.1f} GiB, compute capability {props.major}.{props.minor}")
    print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    # Real workload: a matrix product on GPU compared with CPU to prove the GPU computes correctly.
    torch.manual_seed(0)
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    ref = a @ b
    a_gpu, b_gpu = a.cuda(), b.cuda()
    # Warm-up: the first CUDA call initialises the context and cuBLAS (hundreds of ms).
    # Timing it would be misleading -> same rule later for model latency benchmarks.
    for _ in range(3):
        _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    _ = a @ b
    cpu_ms = (time.perf_counter() - t0) * 1000
    # TF32 may be used on Ampere GPUs -> compare with a relative tolerance
    max_rel = ((out.cpu() - ref).abs().max() / ref.abs().max()).item()
    print(f"matmul 2048 : GPU {gpu_ms:.2f} ms vs CPU {cpu_ms:.2f} ms, max relative error {max_rel:.2e}")

    # Mixed precision (AMP) convolution, as used later in training
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y = conv(torch.randn(32, 3, 224, 224, device="cuda"))
    print(f"AMP conv    : output {tuple(y.shape)} dtype {y.dtype}")
    print(f"GPU memory  : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB peak in this test")

    if max_rel > 1e-2:
        print("  !! GPU result differs from CPU result")
        return 1
    print("\nEnvironment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
