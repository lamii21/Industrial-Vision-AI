"""Phase 3 - probabilistic metrics, statistical tests and the frozen selection rule,
checked on hand-computable examples."""

import math

import numpy as np
import pytest

from ivai.evaluation.calibration import (brier, calibration_report, ece, ece_bootstrap, ece_noise_floor, nll,
                                         per_sample_nll)
from ivai.evaluation.stats import (clopper_pearson, mcnemar_exact, paired_nll_comparison,
                                   select_default_model)

P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.4, 0.1]])
Y = np.array([0, 1, 2, 1])


def test_nll_by_hand():
    assert per_sample_nll(P, Y) == pytest.approx([-math.log(0.7), -math.log(0.8), -math.log(0.4), -math.log(0.4)])
    assert nll(P, Y) == pytest.approx(np.mean([-math.log(v) for v in (0.7, 0.8, 0.4, 0.4)]))


def test_brier_by_hand():
    # (0.09+0.04+0.01) + (0.01+0.04+0.01) + (0.09+0.09+0.36) + (0.25+0.36+0.01) = 0.14+0.06+0.54+0.62
    assert brier(P, Y) == pytest.approx((0.14 + 0.06 + 0.54 + 0.62) / 4)
    assert brier(np.eye(3), np.arange(3)) == 0.0


def test_ece_equal_width_by_hand():
    conf = np.array([0.95, 0.95, 0.65, 0.65])
    correct = np.array([1, 0, 1, 1])
    r = ece(conf, correct, n_bins=10)
    # bin (0.9,1]: conf .95 acc .5 -> |.45|*.5 ; bin (0.6,0.7]: conf .65 acc 1 -> |.35|*.5
    assert r["ece"] == pytest.approx(0.5 * 0.45 + 0.5 * 0.35)
    assert r["non_empty_bins"] == 2
    assert sum(b["count"] for b in r["bins"]) == 4


def test_ece_edge_goes_to_lower_bin_and_zero_when_calibrated():
    r = ece(np.array([0.5]), np.array([1.0]), n_bins=10)
    assert r["bins"][4]["count"] == 1  # 0.5 belongs to (0.4, 0.5]
    assert ece(np.ones(5), np.ones(5))["ece"] == 0.0


def test_ece_equal_mass():
    conf = np.array([0.6, 0.7, 0.8, 0.9])
    correct = np.array([0, 1, 1, 1])
    r = ece(conf, correct, n_bins=2, scheme="mass")
    assert [b["count"] for b in r["bins"]] == [2, 2]
    assert r["ece"] == pytest.approx(0.5 * abs(0.5 - 0.65) + 0.5 * abs(1.0 - 0.85))


def test_noise_floor_and_bootstrap_reproducible():
    conf = np.full(270, 0.99)
    a, b = ece_noise_floor(conf, n_draws=200), ece_noise_floor(conf, n_draws=200)
    assert a == b and a["median"] > 0
    correct = np.ones(270)
    lo, hi = ece_bootstrap(conf, correct, n_boot=200)
    assert lo == pytest.approx(0.01) and hi == pytest.approx(0.01)  # always |1 - 0.99|


def test_calibration_report_uses_raw_probabilities():
    rep = calibration_report(P, Y)
    assert rep["nll"] == pytest.approx(nll(P, Y))
    assert rep["error_confidences"] == [0.5]  # sample 4: predicted class 0 with p = 0.5
    assert rep["n_errors_conf_above"]["count"] == 0
    assert "descriptive" in rep["ece"]["note"]


def test_clopper_pearson_matches_protocol_table():
    lo, hi = clopper_pearson(1, 270)
    assert (round(lo * 270, 1), round(hi * 270, 1)) == (0.0, 5.5)
    assert clopper_pearson(0, 270)[0] == 0.0
    assert round(clopper_pearson(0, 270)[1] * 270, 1) == 3.7


def test_mcnemar_exact():
    a = np.array([True] * 6 + [True] * 10)
    b = np.array([False] * 6 + [True] * 10)
    r = mcnemar_exact(a, b)
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (6, 0)
    assert r["p_value"] == pytest.approx(0.03125)
    assert mcnemar_exact(a, a)["p_value"] == 1.0
    assert mcnemar_exact(a[:5], b[:5])["p_value"] == pytest.approx(0.0625)


def _nll_pair(rng, shift, n=270, seeds=3, noise=0.01):
    base = rng.exponential(0.02, size=n)
    a = np.stack([base + rng.normal(0, noise, n) for _ in range(seeds)])
    b = np.stack([base + shift + rng.normal(0, noise, n) for _ in range(seeds)])
    return a, b


def test_paired_nll_rule():
    rng = np.random.default_rng(1)
    a, b = _nll_pair(rng, shift=0.02)          # B clearly worse everywhere
    assert paired_nll_comparison(a, b)["verdict"] == "A_better"
    assert paired_nll_comparison(b, a)["verdict"] == "B_better"
    assert paired_nll_comparison(a, a.copy())["verdict"] == "close"
    # consistent image-level CI but one seed of the opposite sign -> close
    a2, b2 = a.copy(), b.copy()
    b2[1] = a2[1] - 0.001
    r = paired_nll_comparison(a2, b2)
    assert r["verdict"] == "close"
    assert r["d_mean"] == pytest.approx(np.mean(r["d_s"]))
    assert paired_nll_comparison(a, b) == paired_nll_comparison(a, b)  # fixed bootstrap seed


def test_paired_nll_requires_same_shape():
    with pytest.raises(ValueError):
        paired_nll_comparison(np.zeros((3, 10)), np.zeros((3, 9)))


def _cands(errors, recalls=None):
    return {n: {"mean_errors": e, "mean_recall_per_class": {"c": (recalls or {}).get(n, 1.0)}} for n, e in errors.items()}


def test_selection_eligibility_margin_and_recall():
    rng = np.random.default_rng(0)
    nlls = {n: np.abs(rng.normal(0.02, 0.001, (3, 270))) for n in "ABC"}
    sel = select_default_model(_cands({"A": 1.0, "B": 3.0, "C": 3.4}, {"B": 0.94}), nlls,
                               {"A": 10, "B": 1, "C": 1}, {"A": 1, "B": 1, "C": 1})
    assert sel["eligibility"]["A"]["eligible"]
    assert not sel["eligibility"]["B"]["eligible"]   # recall 0.94 < 0.95
    assert not sel["eligibility"]["C"]["eligible"]   # 3.4 > 1.0 + 2
    assert sel["chosen"] == "A"


def test_selection_nll_domination_then_latency_then_size():
    rng = np.random.default_rng(2)
    a, b = _nll_pair(rng, shift=0.02)
    c = a + rng.normal(0, 0.0005, a.shape)    # close to A
    sel = select_default_model(_cands({"A": 1.0, "B": 1.0, "C": 1.0}), {"A": a, "B": b, "C": c},
                               {"A": 10.0, "B": 1.0, "C": 5.0}, {"A": 1, "B": 1, "C": 1})
    assert "B" in sel["dominated"]             # B has a clearly worse NLL even though it is the fastest
    assert sel["chosen"] == "C"                # A and C close in NLL -> lower latency
    sel2 = select_default_model(_cands({"A": 1.0, "C": 1.0}), {"A": a, "C": c},
                                {"A": 5.1, "C": 5.0}, {"A": 2.0, "C": 90.0})
    assert sel2["latency_tie_broken_by_size"] and sel2["chosen"] == "A"   # < 5 % latency gap -> smaller file
