"""E3 train subsets and virtual epochs (docs/phase3_protocol.md §4 and §8).

* Subsets are drawn from the **train split only**, stratified by class and **nested**:
  for each class (fixed order of ``label_idx``), the train images sorted by file name are
  permuted with ``numpy.random.default_rng(seed)`` and the first ``k`` are kept, with
  ``k = floor(n_class * fraction)``. The permutation does not depend on ``fraction``, so
  10 % ⊂ 25 % ⊂ 50 % for a given seed, and it does not depend on the architecture, so all
  models with the same seed see exactly the same subset. The global torch RNG is not used.
* A **virtual epoch** always contains ``VIRTUAL_EPOCH_SIZE`` = 1 260 samples (40 steps of
  32), whatever the subset size: ``RandomSampler(num_samples=1260, replacement=False)``
  chains full permutations of the subset, so each image is seen floor(1260/n) or
  ceil(1260/n) times per epoch. Patience, scheduler and the 60-epoch cap therefore mean
  the same number of optimisation steps for every fraction and architecture.
* Fraction 1.0 keeps the historical Phase 2 loader (``shuffle=True``), which is
  verified to produce the same order as the virtual-epoch sampler.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from ivai.utils.seed import seed_worker

VIRTUAL_EPOCH_SIZE = 1260  # size of the full train split (Phase 1)
FRACTIONS = (0.10, 0.25, 0.50)


def per_class_count(n_class: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return math.floor(n_class * fraction + 1e-9)  # 210*0.1 = 21.000000000000004 -> 21; 210*0.25 -> 52


def stratified_nested_subset(train_df: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    """Sorted positional indices (into ``train_df``) of the subset."""
    if "split" in train_df.columns and set(train_df["split"]) != {"train"}:
        raise ValueError("subsets are drawn from the train split only")
    df = train_df.reset_index(drop=True)
    if fraction == 1.0:
        return np.arange(len(df))
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(df["label_idx"].unique()):
        pos = df.index[df["label_idx"] == label].to_numpy()
        pos = pos[np.argsort(df.loc[pos, "filename"].to_numpy(), kind="stable")]
        perm = rng.permutation(pos)
        chosen.append(perm[: per_class_count(len(pos), fraction)])
    return np.sort(np.concatenate(chosen))


def fraction_tag(fraction: float) -> str:
    """0.1 -> 'f010', 0.25 -> 'f025', 0.5 -> 'f050'."""
    return f"f{round(fraction * 100):03d}"


def make_train_loader(dataset: Dataset, batch_size: int, generator: torch.Generator, pin_memory: bool,
                      virtual_epoch_size: int | None = None) -> DataLoader:
    if virtual_epoch_size is None:  # historical Phase 2 construction, unchanged
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                          worker_init_fn=seed_worker, num_workers=0, pin_memory=pin_memory)
    sampler = RandomSampler(dataset, replacement=False, num_samples=virtual_epoch_size, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, generator=generator,
                      worker_init_fn=seed_worker, num_workers=0, pin_memory=pin_memory)
