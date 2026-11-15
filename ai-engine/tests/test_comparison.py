"""Phase 3 - comparison module on synthetic experiments and (read-only) on the Phase 2 runs."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ivai.config import ENGINE_DIR, REPORTS_DIR, load_data_config
from ivai.evaluation.comparison import check_aligned, learning_curve, load_experiment, main_seed_runs, summarize

CLASSES = load_data_config().classes
FILES = [f"img_{i:03d}.jpg" for i in range(30)]


def make_exp(root, name, seeds=(42, 123, 2026), acc_noise=0.0, fraction=None, test_used=False):
    rng = np.random.default_rng(len(name))
    for s in seeds:
        d = root / "experiments" / name / f"seed_{s}"
        d.mkdir(parents=True)
        y = np.arange(len(FILES)) % 6
        logits = rng.normal(0, 1, (len(FILES), 6))
        logits[np.arange(len(FILES)), y] += 4 - acc_noise
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        pred = p.argmax(1)
        df = pd.DataFrame({"filename": FILES, "true": [CLASSES[i] for i in y], "pred": [CLASSES[i] for i in pred],
                           "confidence": p.max(1), **{f"p_{c}": p[:, k] for k, c in enumerate(CLASSES)}})
        df.sample(frac=1, random_state=s).to_csv(d / "val_predictions.csv", index=False)  # order must not matter
        per_class = {c: {"precision": 1.0, "recall": float((pred[y == k] == k).mean()), "f1": 1.0, "support": 5}
                     for k, c in enumerate(CLASSES)}
        run = {"seed": s, "experiment": name,
               "best_validation_metrics": {"accuracy": float((pred == y).mean()), "f1_macro": 0.9, "precision_macro": 0.9,
                                           "recall_macro": 0.9, "n_errors": int((pred != y).sum()), "per_class": per_class},
               "training": {"best_epoch": 10, "epochs_run": 18, "train_time_s": 1.0},
               "architecture": {"parameters": {"total": 100, "trainable": 100}},
               "data": {"test_used": test_used,
                        "train_subset": None if fraction is None else {"fraction_real": fraction}}}
        (d / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "experiments" / name / "seed_42_repro").mkdir()   # tagged run: must be ignored


def test_load_and_summarize(tmp_path):
    make_exp(tmp_path, "a")
    exp = load_experiment("a", CLASSES, tmp_path)
    assert exp.seeds == [42, 123, 2026]
    assert [p.name for p in main_seed_runs(tmp_path / "experiments" / "a")] == ["seed_42", "seed_123", "seed_2026"]
    assert exp.filenames(42) == sorted(FILES)
    s = summarize(exp)
    assert s["metrics"]["nll"]["mean"] > 0 and len(s["metrics"]["nll"]["per_seed"]) == 3
    assert exp.per_image_nll().shape == (3, len(FILES))
    assert set(s["calibration_per_seed"]) == {42, 123, 2026}
    assert load_experiment("missing", CLASSES, tmp_path) is None


def test_refuses_runs_that_used_the_test_split(tmp_path):
    make_exp(tmp_path, "bad", test_used=True)
    with pytest.raises(RuntimeError):
        load_experiment("bad", CLASSES, tmp_path)


def test_alignment_check(tmp_path):
    make_exp(tmp_path, "a")
    make_exp(tmp_path, "b", seeds=(42, 123))
    with pytest.raises(ValueError):
        check_aligned(load_experiment("a", CLASSES, tmp_path), load_experiment("b", CLASSES, tmp_path))


def test_learning_curve_auc():
    pts = {x: {"metrics": {k: {"mean": v, "std": 0.0} for k in ("f1_macro", "accuracy", "nll")}}
           for x, v in [(0.1, 0.8), (0.5, 0.9), (1.0, 1.0)]}
    c = learning_curve(pts)
    assert c["fractions"] == [0.1, 0.5, 1.0]
    assert c["auc_f1"] == pytest.approx(0.4 * 0.85 + 0.5 * 0.95)


def test_compare_script_end_to_end_on_synthetic_runs(tmp_path):
    rep = tmp_path / "reports"
    for n in ("custom_cnn", "resnet50_ft", "mobilenetv2_ft"):
        make_exp(rep, n)
    make_exp(rep, "custom_cnn_f010", fraction=0.1, acc_noise=2.0)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ENGINE_DIR / "scripts" / "compare_models.py"), "--reports-dir", str(rep),
                        "--out", str(out), "--benchmark", str(tmp_path / "none.json")],
                       capture_output=True, text=True, encoding="utf-8", cwd=ENGINE_DIR)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert res["selection"]["status"] == "pending"          # no benchmark yet -> no selection
    assert set(res["pairwise_e1"]) == {"custom_cnn vs resnet50_ft", "custom_cnn vs mobilenetv2_ft",
                                       "resnet50_ft vs mobilenetv2_ft"}
    assert res["e3_learning_curves"]["custom_cnn"]["fractions"] == [0.1, 1.0]
    assert (out / "reliability_diagram.png").exists() and (out / "e3_learning_curves.png").exists()
    r2 = subprocess.run([sys.executable, str(ENGINE_DIR / "scripts" / "compare_models.py"), "--reports-dir", str(rep),
                         "--out", str(out)], capture_output=True, text=True, encoding="utf-8", cwd=ENGINE_DIR)
    assert r2.returncode != 0  # refuses to overwrite without --force


@pytest.mark.skipif(not (REPORTS_DIR / "experiments" / "custom_cnn" / "seed_42").exists(), reason="Phase 2 runs absent")
def test_phase2_probabilities_are_raw():
    """NLL recomputed from val_predictions.csv equals the stored validation loss (read-only check)."""
    exp = load_experiment("custom_cnn", CLASSES)
    s = summarize(exp, with_calibration=False)
    for seed, v in zip(exp.seeds, s["metrics"]["nll"]["per_seed"]):
        assert abs(v - exp.runs[seed]["best_validation_metrics"]["loss"]) < 1e-4
