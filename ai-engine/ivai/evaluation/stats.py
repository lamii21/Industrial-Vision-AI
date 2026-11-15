"""Statistical comparisons and the frozen selection rule (docs/phase3_protocol.md §10-11).

The selection functions implement the protocol literally; they must not be tuned after
the results are known.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import beta, binomtest

EQUIVALENCE_MARGIN_IMAGES = 2      # §10.1
MIN_CLASS_RECALL = 0.95            # §10.1
N_BOOTSTRAP = 10_000               # §10.2
BOOTSTRAP_SEED = 0                 # §10.2
LATENCY_TIE_RELATIVE = 0.05        # §10.4


# ---------------------------------------------------------------- descriptive tests
def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for an error rate k/n."""
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired per-image correctness. b = A right / B wrong,
    c = A wrong / B right; p-value = two-sided binomial test of b vs c (p = 1 if b + c = 0)."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b_.shape:
        raise ValueError("paired arrays must have the same shape")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "discordant": b + c, "p_value": p,
            "both_wrong": int((~a & ~b_).sum())}


# ---------------------------------------------------------------- §10.2 paired NLL rule
def paired_nll_comparison(nll_a: np.ndarray, nll_b: np.ndarray, n_boot: int = N_BOOTSTRAP,
                          seed: int = BOOTSTRAP_SEED) -> dict:
    """``nll_a``, ``nll_b``: per-image NLL, shape (n_seeds, n_images), same seeds and the same
    image order for both models.

    d_s = mean_i(nll_a[s] - nll_b[s]);  Delta_i = mean_s(nll_a[:, i] - nll_b[:, i]);
    bootstrap: resample the images (with replacement) -> 95 % percentile CI [L, U] of mean(Delta).
    A >> B  iff  all d_s < 0 and U < 0;   B >> A  iff  all d_s > 0 and L > 0;   otherwise "close".
    """
    a, b = np.asarray(nll_a, float), np.asarray(nll_b, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("expected two arrays of identical shape (n_seeds, n_images)")
    d_s = (a - b).mean(axis=1)
    delta = (a - b).mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), (n_boot, len(delta)))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    if np.all(d_s < 0) and hi < 0:
        verdict = "A_better"
    elif np.all(d_s > 0) and lo > 0:
        verdict = "B_better"
    else:
        verdict = "close"
    return {"d_s": d_s.tolist(), "d_mean": float(delta.mean()), "ci95": [lo, hi], "verdict": verdict,
            "n_boot": n_boot, "bootstrap_seed": seed}


# ---------------------------------------------------------------- §10 selection
def select_default_model(candidates: Mapping[str, Mapping], per_image_nll: Mapping[str, np.ndarray],
                         cpu_latency_ms: Mapping[str, float], file_size_mb: Mapping[str, float],
                         margin: float = EQUIVALENCE_MARGIN_IMAGES, min_recall: float = MIN_CLASS_RECALL) -> dict:
    """Frozen selection procedure.

    ``candidates[name]``: {"mean_errors": float, "mean_recall_per_class": {class: float}}
    ``per_image_nll[name]``: (n_seeds, n_images), identical seed and image order for all models.
    ``cpu_latency_ms[name]``: median CPU latency, batch 1, fp32, 4 threads.
    Returns the chosen model and every intermediate decision (for the report).
    """
    names = list(candidates)
    # Step 1 - eligibility
    best_err = min(candidates[n]["mean_errors"] for n in names)
    eligibility = {}
    for n in names:
        err_ok = candidates[n]["mean_errors"] <= best_err + margin + 1e-12
        rec_ok = all(r >= min_recall for r in candidates[n]["mean_recall_per_class"].values())
        eligibility[n] = {"errors_ok": bool(err_ok), "recall_ok": bool(rec_ok), "eligible": bool(err_ok and rec_ok)}
    eligible = [n for n in names if eligibility[n]["eligible"]]
    # Step 2 - pairwise paired NLL comparisons
    pairwise, dominated = {}, set()
    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            res = paired_nll_comparison(per_image_nll[a], per_image_nll[b])
            pairwise[f"{a} vs {b}"] = res
            if res["verdict"] == "A_better":
                dominated.add(b)
            elif res["verdict"] == "B_better":
                dominated.add(a)
    # Step 3 - non-dominated set (never empty: the relation is acyclic, see protocol §10.3)
    non_dominated = [n for n in eligible if n not in dominated]
    assert non_dominated, "non-dominated set cannot be empty"
    # Step 4 - CPU latency, then file size if the two best medians differ by < 5 %
    by_latency = sorted(non_dominated, key=lambda n: cpu_latency_ms[n])
    chosen, tie = by_latency[0], False
    if len(by_latency) > 1:
        l0, l1 = cpu_latency_ms[by_latency[0]], cpu_latency_ms[by_latency[1]]
        if (l1 - l0) / l0 < LATENCY_TIE_RELATIVE:
            tie = True
            chosen = min(by_latency[:2], key=lambda n: file_size_mb[n])
    return {"chosen": chosen, "best_mean_errors": best_err, "eligibility": eligibility, "eligible": eligible,
            "pairwise_nll": pairwise, "dominated": sorted(dominated), "non_dominated": non_dominated,
            "latency_order": by_latency, "latency_tie_broken_by_size": tie,
            "rule": "docs/phase3_protocol.md §10 (frozen)"}
