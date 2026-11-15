"""Reproducibility helpers.

Bit-exact reproducibility on GPU needs, in addition to the seeds:
* cuDNN in deterministic mode with autotuning (benchmark) disabled;
* ``torch.use_deterministic_algorithms(True)``: PyTorch raises an error if an
  operation without a deterministic implementation is used (so nothing silently
  breaks reproducibility);
* ``CUBLAS_WORKSPACE_CONFIG`` set *before* the first cuBLAS call.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG and return a ``torch.Generator`` for the DataLoader shuffling."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/random seeds from the torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
