"""Pin every number quoted in README.md to the results files in results/.

If a results file is regenerated and the README is not updated, this fails.
Run:  python -m pytest tests/   (needs numpy, scikit-learn)

README convention: mean ± sample standard deviation (ddof=1) over the
15 runs = 5 folds x 3 seeds.
"""
import json
import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


# ---------------------------------------------------------------- stage 1

def test_stage1_has_15_runs_on_official_folds():
    d = load("cv_runs.json")
    assert d["classes"][d["cough_index"]] == "coughing"
    assert sorted((r["seed"], r["fold"]) for r in d["runs"]) == [
        (s, f) for s in (0, 1, 2) for f in (1, 2, 3, 4, 5)
    ]
    assert all(len(r["y"]) == 80 for r in d["runs"])  # 400 clips / 5 folds


def test_stage1_10way_accuracy_0838_pm_0041():
    d = load("cv_runs.json")
    m, s = mean_sd([r["acc"] for r in d["runs"]])
    assert m == pytest.approx(0.838, abs=0.0005)
    assert s == pytest.approx(0.041, abs=0.0005)
    # the stored accuracy must be the argmax of the stored logits
    for r in d["runs"]:
        acc = float((np.argmax(r["logits"], axis=1) == np.asarray(r["y"])).mean())
        assert acc == pytest.approx(r["acc"], abs=1e-9)


def test_stage1_cough_vs_rest_auc_0986_pm_0011():
    d = load("cv_runs.json")
    ci = d["cough_index"]
    aucs = []
    for r in d["runs"]:
        L = np.asarray(r["logits"])
        p = np.exp(L - L.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        aucs.append(roc_auc_score(np.asarray(r["y"]) == ci, p[:, ci]))
    m, s = mean_sd(aucs)
    assert m == pytest.approx(0.986, abs=0.0005)
    assert s == pytest.approx(0.011, abs=0.0005)


def test_stage1_linear_baseline_0428_pm_0037():
    d = load("cv_runs.json")
    m, s = mean_sd([r["baseline_acc"] for r in d["runs"]])
    assert m == pytest.approx(0.428, abs=0.0005)
    assert s == pytest.approx(0.037, abs=0.0005)


def test_stage1_cough_recall_and_laughing_confusion():
    """README: cough recalled 75 %, 15 % called laughing, 7 % called sneezing."""
    d = load("cv_runs.json")
    cls = d["classes"]
    y = np.concatenate([r["y"] for r in d["runs"]])
    pred = np.concatenate([np.argmax(r["logits"], axis=1) for r in d["runs"]])
    cough = y == cls.index("coughing")
    n = cough.sum()
    assert n == 120  # 40 clips x 3 seeds
    assert (pred[cough] == cls.index("coughing")).sum() / n == pytest.approx(0.75, abs=0.005)
    assert (pred[cough] == cls.index("laughing")).sum() / n == pytest.approx(0.15, abs=0.005)
    assert (pred[cough] == cls.index("sneezing")).sum() / n == pytest.approx(0.07, abs=0.005)


# ---------------------------------------------------------------- stage 2

def test_stage2_dataset_size_2063_split_1587_476():
    d = load("coughtype_runs.json")
    wet = d["types"].index("wet")
    for seed in (0, 1, 2):
        y = np.concatenate([r["y"] for r in d["runs"] if r["seed"] == seed])
        assert len(y) == 2063
        assert int((y == wet).sum()) == 476
        assert int((y != wet).sum()) == 1587


def test_stage2_headline_auc_and_balanced_accuracy():
    d = load("coughtype_runs.json")
    m, s = mean_sd([r["auc"] for r in d["runs"]])
    assert (m, s) == (pytest.approx(0.705, abs=0.0005), pytest.approx(0.027, abs=0.0005))
    m, s = mean_sd([r["bal_acc"] for r in d["runs"]])
    assert (m, s) == (pytest.approx(0.655, abs=0.0005), pytest.approx(0.023, abs=0.0005))


def test_stage2_linear_baseline_and_majority():
    d = load("coughtype_runs.json")
    m, s = mean_sd([r["base_auc"] for r in d["runs"]])
    assert (m, s) == (pytest.approx(0.559, abs=0.0005), pytest.approx(0.025, abs=0.0005))
    m, s = mean_sd([r["base_bal"] for r in d["runs"]])
    assert (m, s) == (pytest.approx(0.540, abs=0.0005), pytest.approx(0.023, abs=0.0005))
    assert np.mean([r["majority_acc"] for r in d["runs"]]) == pytest.approx(0.769, abs=0.0005)


def test_stage2_operating_points_match_runs():
    """results/stage2_operating_points.json must be derivable from coughtype_runs.json."""
    d = load("coughtype_runs.json")
    op = load("stage2_operating_points.json")
    diff = np.array([r["auc"] - r["base_auc"] for r in d["runs"]])
    p = op["paired_cnn_vs_baseline_auc"]
    assert p["n_runs"] == 15 and p["cnn_wins"] == 15
    assert p["mean_diff"] == pytest.approx(diff.mean(), abs=1e-9)
    assert p["min_diff"] == pytest.approx(diff.min(), abs=1e-9)
    wet = d["types"].index("wet")
    y = np.concatenate([r["y"] for r in d["runs"]])
    pred = np.concatenate([r["pred"] for r in d["runs"]])
    a = op["argmax_operating_point"]
    assert a["wet_recall"] == pytest.approx(((pred == wet) & (y == wet)).sum() / (y == wet).sum(), abs=1e-9)
    assert a["dry_recall"] == pytest.approx(((pred != wet) & (y != wet)).sum() / (y != wet).sum(), abs=1e-9)


def test_stage2_readme_operating_point_numbers():
    op = load("stage2_operating_points.json")
    p = op["paired_cnn_vs_baseline_auc"]
    assert p["mean_diff"] == pytest.approx(0.146, abs=0.0005)
    assert p["ci95_half"] == pytest.approx(0.018, abs=0.0005)
    assert p["min_diff"] == pytest.approx(0.077, abs=0.0005)
    a = op["argmax_operating_point"]
    assert a["wet_recall"] == pytest.approx(0.74, abs=0.005)
    assert a["dry_recall"] == pytest.approx(0.57, abs=0.005)
    assert a["wet_precision"] == pytest.approx(0.34, abs=0.005)
    assert a["flagged_wet_share"] == pytest.approx(0.50, abs=0.005)
    f = op["sensitivity_at_fixed_specificity"]
    assert f["spec_90"]["mean"] == pytest.approx(0.26, abs=0.005)
    assert f["spec_80"]["mean"] == pytest.approx(0.46, abs=0.005)
    assert f["spec_70"]["mean"] == pytest.approx(0.59, abs=0.005)
    assert op["pooled_oof_auc_per_seed"]["mean"] == pytest.approx(0.701, abs=0.0005)
    assert op["pooled_oof_auc_per_seed"]["sd"] == pytest.approx(0.004, abs=0.0005)
