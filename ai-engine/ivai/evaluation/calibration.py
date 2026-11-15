"""Probabilistic metrics (docs/phase3_protocol.md §9.2-9.3).

All functions take the **raw** softmax probabilities written in ``val_predictions.csv``
(no temperature scaling, no recalibration on the validation set).

* NLL and Brier are the reference metrics (proper scoring rules).
* ECE is **descriptive only**: with 270 images almost all predictions fall in the last
  bin, so it is reported with a bootstrap interval, the noise floor of a perfectly
  calibrated model, the equal-mass variant and the bin occupancy (protocol §11.4).
"""

from __future__ import annotations

import numpy as np

N_BINS = 15


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def per_sample_nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """-log p(y_i | x_i) for each sample."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=int)
    return -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(per_sample_nll(probs, y).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - 1[y=k])^2 (range [0, 2])."""
    probs = np.asarray(probs, dtype=float)
    return float(((probs - _onehot(np.asarray(y, dtype=int), probs.shape[1])) ** 2).sum(1).mean())


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS, scheme: str = "width") -> dict:
    """Expected calibration error of the top-1 confidence.

    ``scheme="width"``: equal-width bins on [0, 1]; a value on a bin edge goes to the lower bin
    (bins are (lo, hi], the first one includes 0). ``scheme="mass"``: bins with equal counts
    (``numpy.array_split`` of the sorted confidences).
    Returns the ECE and, per bin, count / mean confidence / accuracy.
    """
    conf, corr = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    n = len(conf)
    if scheme == "width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        groups = [np.flatnonzero(idx == b) for b in range(n_bins)]
    elif scheme == "mass":
        groups = np.array_split(np.argsort(conf, kind="stable"), n_bins)
        edges = None
    else:
        raise ValueError("scheme must be 'width' or 'mass'")
    bins, total = [], 0.0
    for g in groups:
        if len(g) == 0:
            bins.append({"count": 0, "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[g].mean()), float(corr[g].mean())
        total += len(g) / n * abs(a - c)
        bins.append({"count": int(len(g)), "confidence": c, "accuracy": a})
    return {"ece": float(total), "scheme": scheme, "n_bins": n_bins,
            "edges": None if edges is None else edges.tolist(), "bins": bins,
            "non_empty_bins": int(sum(b["count"] > 0 for b in bins))}


def ece_bootstrap(confidence, correct, n_boot: int = 2000, seed: int = 0, n_bins: int = N_BINS) -> tuple[float, float]:
    """95 % percentile interval of the equal-width ECE (resampling the images)."""
    conf, corr = np.asarray(confidence, float), np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    vals = [ece(conf[i], corr[i], n_bins)["ece"] for i in rng.integers(0, len(conf), (n_boot, len(conf)))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece_noise_floor(confidence, n_draws: int = 2000, seed: int = 0, n_bins: int = N_BINS) -> dict:
    """ECE of a *perfectly calibrated* model with the same confidences: each prediction is
    correct with probability equal to its confidence. Observed ECE values inside this
    distribution cannot be told apart from perfect calibration."""
    conf = np.asarray(confidence, float)
    rng = np.random.default_rng(seed)
    vals = [ece(conf, (rng.random(len(conf)) < conf).astype(float), n_bins)["ece"] for _ in range(n_draws)]
    return {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)), "n_draws": n_draws}


def calibration_report(probs: np.ndarray, y: np.ndarray, high_conf: float = 0.9, seed: int = 0) -> dict:
    """Every metric of protocol §9.2-9.3 for one run."""
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    width = ece(conf, correct, scheme="width")
    lo, hi = ece_bootstrap(conf, correct, seed=seed)
    return {
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "error_confidences": sorted((float(c) for c in conf[correct == 0]), reverse=True),
        "n_errors_conf_above": {"threshold": high_conf, "count": int(((correct == 0) & (conf > high_conf)).sum())},
        "ece": {"value": width["ece"], "bootstrap95": [lo, hi], "noise_floor": ece_noise_floor(conf, seed=seed),
                "equal_mass": ece(conf, correct, scheme="mass")["ece"], "non_empty_bins": width["non_empty_bins"],
                "occupancy": [b["count"] for b in width["bins"]], "bins": width["bins"],
                "note": "descriptive only - not a selection criterion (protocol §9.3, §11.4)"},
    }
