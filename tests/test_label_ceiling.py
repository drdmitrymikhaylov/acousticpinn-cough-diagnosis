"""Pin the README section "What the 77 % allows" to results/label_ceiling_check.json.

The label-noise family and its ceilings are recomputed here from the stored
(a, b, w) triples; the model's kappa and the fold spread are recomputed from
coughtype_runs.json; the bootstrap intervals are read from the JSON (2 000
resamples, seeded, 12 s to regenerate).  Needs numpy and scikit-learn, like the
other tests.
"""
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
LC = json.load(open(os.path.join(RES, "label_ceiling_check.json")))
OP = json.load(open(os.path.join(RES, "stage2_operating_points.json")))
RUNS = json.load(open(os.path.join(RES, "coughtype_runs.json")))
README = " ".join(open(os.path.join(HERE, "README.md"), encoding="utf-8").read().split())

FAM = LC["label_noise"]["family"]
SEEDS = ("0", "1", "2")


def f2(x):
    return "%.2f" % x


def f3(x):
    return "%.3f" % x


def pct(x):
    return "%.0f %%" % (100 * x)


def ci(v, nd=3):
    return "[%s, %s]" % (("%%.%df" % nd) % v[0], ("%%.%df" % nd) % v[1])


def ceiling(a, b, w):
    lw = w * (1 - b) + (1 - w) * a
    q = w * (1 - b) / lw
    r = w * b / (1 - lw)
    auc = q * (1 - r) + 0.5 * (q * r + (1 - q) * (1 - r))
    po = lw * q + (1 - lw) * (1 - r)
    pe = lw * w + (1 - lw) * (1 - w)
    return dict(auc=auc, sensitivity=q, specificity=1 - r, balanced_accuracy=0.5 * (q + 1 - r),
                agreement=po, kappa=(po - pe) / (1 - pe))


def kappa_binary(y, pred):
    po = (y == pred).mean()
    pe = y.mean() * pred.mean() + (1 - y.mean()) * (1 - pred.mean())
    return (po - pe) / (1 - pe)


def pooled(seed):
    wet = RUNS["types"].index("wet")
    rs = [r for r in RUNS["runs"] if r["seed"] == int(seed)]
    y = np.concatenate([np.asarray(r["y"]) for r in rs]) == wet
    p = np.concatenate([np.asarray(r["prob"]) for r in rs])
    pred = np.concatenate([np.asarray(r["pred"]) for r in rs]) == wet
    return y, p, pred


# ------------------------------------------------------------- README rows
def family_rows():
    rows = []
    for c in FAM:
        rows.append("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            f2(c["a"]), " (symmetric)" if c["symmetric"] else "", f2(c["b"]), f2(c["w"]), f3(c["auc"]),
            f3(c["sensitivity"]), f3(c["specificity"]), f3(c["balanced_accuracy"]), f2(c["kappa"]),
            pct(c["cnn_share_of_ceiling_auc"])))
    return rows


def kappa_rows():
    rows = []
    for p, k in zip(LC["physician_pairs"], LC["label_noise"]["physician_kappa_per_pair_at_subset_marginals"]):
        rows.append("| physician %d vs physician %d | %d | %s | %s |" % (p["i"], p["j"], p["n"], f3(p["agreement"]), f2(k)))
    L = LC["label_noise"]
    rows.append("| all four, pooled | %d | %s | %s |" % (
        LC["physician_overlapping_judgements"], f3(LC["physician_agreement"]), f2(L["physician_kappa_at_subset_marginals"])))
    S = LC["model_kappa_summary"]
    rows.append("| CNN against the label, class-weighted argmax (flags %s) | 2 063 per seed | %s | %s |" % (
        pct(np.mean([LC["model_kappa"][s]["argmax_flagged"] for s in SEEDS])), f3(S["argmax_agreement_mean"]), f2(S["at_argmax_mean"])))
    rows.append("| CNN against the label, best threshold (flags %s) | 2 063 per seed | -- | %s |" % (
        pct(S["best_flagged_mean"]), f2(S["best_mean"])))
    rows.append("| a model reproducing the truth exactly, against one-rater labels | -- | %s to %s | %s to %s |" % (
        f3(min(c["agreement"] for c in FAM)), f3(max(c["agreement"] for c in FAM)),
        f2(min(c["kappa"] for c in FAM)), f2(max(c["kappa"] for c in FAM))))
    return rows


def interval_rows():
    B = LC["bootstrap"]
    sd = lambda vals: float(np.std(vals, ddof=1))
    rows = [
        "| pooled out-of-fold ROC-AUC | %s | %s | %s |" % (
            f3(B["0"]["auc"]), ci(B["0"]["auc_ci"]), f3(sd([B[s]["auc"] for s in SEEDS]))),
    ]
    for k, spec in enumerate((90, 80, 70)):
        rows.append("| wet coughs caught at %d %% specificity | %s | %s | %s |" % (
            spec, f2(B["0"]["sens"][k]), ci(B["0"]["sens_ci"][k], 2), f2(sd([B[s]["sens"][k] for s in SEEDS]))))
    rows.append("| balanced accuracy at the argmax threshold | %s | %s | %s |" % (
        f3(B["0"]["bal_acc_argmax"]), ci(B["0"]["bal_acc_ci"]), f3(sd([B[s]["bal_acc_argmax"] for s in SEEDS]))))
    rows.append("| kappa at the argmax threshold | %s | %s | %s |" % (
        f2(LC["model_kappa"]["0"]["at_argmax"]), ci(B["0"]["kappa_argmax_ci"], 2),
        f2(sd([LC["model_kappa"][s]["at_argmax"] for s in SEEDS]))))
    return rows


# ------------------------------------------------------------------ checks
def test_stored_numbers_are_the_ones_on_the_page():
    for s in SEEDS:
        y, p, pred = pooled(s)
        assert abs(roc_auc_score(y, p) - LC["bootstrap"][s]["auc"]) < 1e-12
        assert abs(LC["bootstrap"][s]["auc"] - OP["pooled_oof_auc_per_seed"]["values"][int(s)]) < 1e-12
        assert abs(kappa_binary(y, pred) - LC["model_kappa"][s]["at_argmax"]) < 1e-12
        assert abs(pred.mean() - LC["model_kappa"][s]["argmax_flagged"]) < 1e-12
        assert abs((y == pred).mean() - LC["model_kappa"][s]["argmax_agreement"]) < 1e-12
    assert LC["n_recordings"] == 2063 and LC["n_wet"] == 476
    assert LC["physician_overlapping_judgements"] == 565 and f3(LC["physician_agreement"]) == "0.770"
    assert sum(p["n"] for p in LC["physician_pairs"]) == 565


def test_label_noise_family_is_consistent_and_the_ceilings_follow():
    agree, lw = LC["physician_agreement"], LC["labelled_wet_share"]
    assert abs(lw - 476 / 2063) < 1e-12
    e = 0.5 * (1 - math.sqrt(2 * agree - 1))
    assert abs(LC["label_noise"]["symmetric_error"] - e) < 1e-12 and f3(e) == "0.133"
    assert sum(c["symmetric"] for c in FAM) == 1
    for c in FAM:
        a, b, w = c["a"], c["b"], c["w"]
        assert abs(w * ((1 - b) ** 2 + b ** 2) + (1 - w) * ((1 - a) ** 2 + a ** 2) - agree) < 1e-9
        assert abs(w * (1 - b) + (1 - w) * a - lw) < 1e-9
        for k, v in ceiling(a, b, w).items():
            assert abs(c[k] - v) < 1e-12, k
        assert abs((OP["pooled_oof_auc_per_seed"]["mean"] - 0.5) / (c["auc"] - 0.5) - c["cnn_share_of_ceiling_auc"]) < 1e-12
    aucs = [c["auc"] for c in FAM]
    assert f2(min(aucs)) == "0.71" and f2(max(aucs)) == "0.85"
    sym = [c for c in FAM if c["symmetric"]][0]
    assert f3(sym["auc"]) == "0.739" and f3(sym["sensitivity"]) == "0.502" and f2(sym["w"]) == "0.13"
    assert pct(sym["cnn_share_of_ceiling_auc"]) == "84 %"
    assert pct(FAM[0]["cnn_share_of_ceiling_auc"]) == "57 %" and pct(FAM[-1]["cnn_share_of_ceiling_auc"]) == "97 %"
    assert f2(FAM[0]["b"]) == "0.50" and f2(FAM[0]["w"]) == "0.46"
    assert f2(FAM[-1]["a"]) == "0.15" and f2(FAM[-1]["b"]) == "0.00" and f2(FAM[-1]["w"]) == "0.10"
    assert f2(min(c["kappa"] for c in FAM)) == "0.52" and f2(max(c["kappa"] for c in FAM)) == "0.59"
    bals = [c["balanced_accuracy"] for c in FAM]
    assert f2(min(bals)) == "0.71" and f2(max(bals)) == "0.85"


def test_kappa_scale():
    L = LC["label_noise"]
    lw = LC["labelled_wet_share"]
    pe = lw * lw + (1 - lw) * (1 - lw)
    assert abs((LC["physician_agreement"] - pe) / (1 - pe) - L["physician_kappa_at_subset_marginals"]) < 1e-12
    assert f2(L["physician_kappa_at_subset_marginals"]) == "0.35"
    ks = L["physician_kappa_per_pair_at_subset_marginals"]
    for p, k in zip(LC["physician_pairs"], ks):
        assert abs((p["agreement"] - pe) / (1 - pe) - k) < 1e-12
    with3 = [k for p, k in zip(LC["physician_pairs"], ks) if 3 in (p["i"], p["j"])]
    without3 = [k for p, k in zip(LC["physician_pairs"], ks) if 3 not in (p["i"], p["j"])]
    assert len(with3) == 3 and f2(min(with3)) == "0.07" and f2(max(with3)) == "0.11"
    assert f2(min(without3)) == "0.50" and f2(max(without3)) == "0.80"
    S = LC["model_kappa_summary"]
    assert f2(S["at_argmax_mean"]) == "0.22" and f2(S["best_mean"]) == "0.25"
    assert pct(S["best_flagged_mean"]) == "30 %" and f3(S["argmax_agreement_mean"]) == "0.609"
    for s in SEEDS:
        assert LC["model_kappa"][s]["best"]["kappa"] >= LC["model_kappa"][s]["at_argmax"]


def test_bootstrap_intervals():
    B = LC["bootstrap"]
    for s in SEEDS:
        assert B[s]["n_boot"] == 2000
        lo, hi = B[s]["auc_ci"]
        assert lo < B[s]["auc"] < hi and 0.02 < hi - lo < 0.06
    assert ci(B["0"]["auc_ci"]) == "[0.681, 0.730]"
    assert [ci(c, 2) for c in B["0"]["sens_ci"]] == ["[0.22, 0.32]", "[0.41, 0.50]", "[0.53, 0.64]"]
    assert ci(B["0"]["bal_acc_ci"]) == "[0.634, 0.679]"
    assert ci(B["0"]["kappa_argmax_ci"], 2) == "[0.19, 0.26]"
    # the seed spread the page quotes against the sampling spread
    assert f3(np.std([B[s]["auc"] for s in SEEDS], ddof=1)) == "0.004"
    assert f3(B["0"]["auc_sd"]) == "0.013"
    assert all(0.02 <= v <= 0.035 for s in SEEDS for v in B[s]["sens_sd"])
    # the page's "+/- 0.00" at 80 % and 70 % is a spread over three retrainings, not an uncertainty
    seed_sd = [float(np.std([B[s]["sens"][k] for s in SEEDS], ddof=1)) for k in range(3)]
    assert f2(seed_sd[1]) == "0.00" and f2(seed_sd[2]) == "0.00" and f2(seed_sd[0]) == "0.03"


def test_fold_spread_is_sampling():
    F = LC["fold_spread"]
    run_aucs = np.array([r["auc"] for r in RUNS["runs"]])
    assert abs(run_aucs.std(ddof=1) - F["sd_over_15_runs"]) < 1e-12
    means = {str(f): float(np.mean([r["auc"] for r in RUNS["runs"] if r["fold"] == f])) for f in range(1, 6)}
    for f, v in means.items():
        assert abs(v - F["fold_means"][f]) < 1e-12
    assert f3(F["sd_over_15_runs"]) == "0.027" and f3(F["sd_expected_from_413_recording_subsamples"]) == "0.027"
    assert f3(F["sd_of_fold_means"]) == "0.021"
    lo, hi = F["subsample_ci"]
    assert ci(F["subsample_ci"]) == "[0.652, 0.757]"
    assert lo < min(means.values()) and max(means.values()) < hi


def test_readme_tables_and_prose():
    for r in family_rows() + kappa_rows() + interval_rows():
        assert r in README, r
    for s in (
        "AUC 0.71 to 0.85",
        "0.739 under symmetric errors",
        "57 % to 97 % of the way from chance to the ceiling",
        "84 % under symmetric errors",
        "a wet cough half the time",
        "0.35 on the kappa scale",
        "0.07 to 0.11",
        "0.50 to 0.80",
        "0.22 at the argmax threshold and 0.25 at its best threshold",
        "0.52 to 0.59",
        "[0.681, 0.730]",
        "[0.41, 0.50]",
        "0.027 against 0.027",
        "[0.652, 0.757]",
    ):
        assert s in README, s


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
