"""Pin the clip-level reading of stage 1 (README: "Is that 120 predictions or
40 clips?") to results/cv_runs.json and results/stage1_clip_checks.json.

Everything that can be recomputed from the stored logits is recomputed here.
Only the source-recording counts come from the JSON alone: they need ESC-50's
metadata file, which is not redistributed in this repository.

Run:  python -m pytest tests/   (needs numpy, scikit-learn)
"""
import json
import math
import os
from collections import Counter

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def readme():
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        return " ".join(f.read().split())          # the page wraps lines; prose must not care


def pct(k, n):
    return int(100.0 * k / n + 0.5)                # half-up, not Python's half-to-even


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def two_sided_even_split(a, b):
    """Exact two-sided binomial p for a split a : b under p = 1/2."""
    n, k = a + b, max(a, b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stacked():
    d = load("cv_runs.json")
    C, ci = d["classes"], d["cough_index"]
    y, logits = None, {}
    for s in SEEDS:
        runs = sorted((r for r in d["runs"] if r["seed"] == s), key=lambda r: r["fold"])
        ys = np.concatenate([r["y"] for r in runs])
        assert y is None or (ys == y).all()        # same clips, same order, every seed
        y = ys
        logits[s] = np.concatenate([np.asarray(r["logits"], dtype=float) for r in runs])
    pred = {s: logits[s].argmax(1) for s in SEEDS}
    n_right = sum((pred[s] == y).astype(int) for s in SEEDS)
    return C, ci, y, logits, pred, n_right


def cough_scores(logits, ci):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, ci]


def class_row(c, k, y, pred, n_right, bold=False):
    m = y == k
    right = int(sum((pred[s][m] == k).sum() for s in SEEDS))
    maj, never = int((n_right[m] >= 2).sum()), int((n_right[m] == 0).sum())
    b = "**" if bold else ""
    return f"| {b}{c}{b} | {b}{right} ({pct(right, 120)} %){b} | {b}{maj}{b} | {b}{never}{b} |"


def operating_rows(y, logits, ci):
    rows = {"zero": [], "fa3": [], "fa18": [], "c36": [], "c38": [], "c40": []}
    for s in SEEDS:
        sc = cough_scores(logits[s], ci)
        pos = np.sort(sc[y == ci])[::-1]
        neg = np.sort(sc[y != ci])[::-1]
        rows["zero"].append(int((pos > neg[0]).sum()))
        rows["fa3"].append(int((pos > neg[3]).sum()))
        rows["fa18"].append(int((pos > neg[18]).sum()))
        for k in (36, 38, 40):
            rows[f"c{k}"].append(int((neg >= pos[k - 1]).sum()))
    label = {
        "zero": "Coughs caught (of 40) before the first false alarm",
        "fa3": "Coughs caught with 3 false alarms in the 360 other clips (0.8 %)",
        "fa18": "Coughs caught with 18 false alarms (5 %)",
        "c36": "False alarms needed to catch 36 of 40",
        "c38": "False alarms needed to catch 38 of 40",
        "c40": "False alarms needed to catch all 40",
    }
    return rows, [f"| {label[k]} | " + " | ".join(str(v) for v in rows[k]) + " |" for k in rows]


# --------------------------------------------------------------------- tests

def test_cough_errors_sit_on_clips_not_on_seeds():
    C, ci, y, _, pred, n_right = stacked()
    cough = y == ci
    assert [int((pred[s][cough] == ci).sum()) for s in SEEDS] == [30, 30, 30]
    by = Counter(n_right[cough].tolist())
    assert (by[3], by[2], by[1], by[0]) == (25, 7, 1, 7)
    assert 120 - 90 == 30 and 3 * by[0] == 21      # 21 of the 30 errors: never-right clips
    lo, hi = wilson(30, 40)
    assert (pct(lo, 1), pct(hi, 1)) == (60, 86)
    assert int(((pred[0] == pred[1]) & (pred[1] == pred[2])).sum()) == 344
    assert int((n_right == 0).sum()) == 43
    t = readme()
    assert "**7 clips are missed by all three seeds** -- 21 of the 30 errors" in t
    assert "95 % interval for 30 of 40: 60–86 %" in t
    assert "the same answer on 344 of the 400 clips, and 43 clips are never classified correctly" in t


def test_cough_is_not_the_second_hardest_class():
    C, ci, y, _, pred, n_right = stacked()
    right = {c: int(sum((pred[s][y == k] == k).sum() for s in SEEDS)) for k, c in enumerate(C)}
    hardest = sorted(C, key=right.get)
    assert hardest[:4] == ["drinking_sipping", "breathing", "coughing", "laughing"]
    maj = {c: int((n_right[y == k] >= 2).sum()) for k, c in enumerate(C)}
    assert sorted(C, key=maj.get)[:4] == ["drinking_sipping", "breathing", "laughing", "coughing"]
    t = readme()
    for c in hardest[:5]:
        assert class_row(c, C.index(c), y, pred, n_right, bold=(c == "coughing")) in t
    assert "Cough is the second-hardest class" not in t
    lo, hi = wilson(32, 40)
    assert f"the interval on 32 of 40 is {pct(lo, 1)}–{pct(hi, 1)} %" in t


def test_laughing_against_sneezing_counted_by_clip():
    C, ci, y, _, pred, n_right = stacked()
    never = [i for i in np.where(y == ci)[0] if n_right[i] == 0]
    always = Counter(C[pred[0][i]] for i in never
                     if pred[0][i] == pred[1][i] == pred[2][i])
    assert dict(always) == {"laughing": 5, "sneezing": 2}
    major = Counter()
    for i in np.where(y == ci)[0]:
        top, n = Counter(pred[s][i] for s in SEEDS).most_common(1)[0]
        if top != ci and n >= 2:
            major[C[top]] += 1
    assert dict(major) == {"laughing": 6, "sneezing": 2}
    assert pct(two_sided_even_split(5, 2), 1) == 45
    assert pct(two_sided_even_split(6, 2), 1) == 29
    lau, sne = C.index("laughing"), C.index("sneezing")
    back = {k: (sum(int((pred[s][y == k] == ci).sum()) for s in SEEDS),
                len({int(i) for s in SEEDS for i in np.where(y == k)[0] if pred[s][i] == ci}))
            for k in (lau, sne)}
    assert back[lau] == (8, 3) and back[sne] == (5, 3)
    t = readme()
    assert "45 % (five against two) or 29 % (six against two) of the time" in t
    assert "8 predictions on **3 clips**, and sneezes called cough are 5 predictions on 3 clips" in t


def test_operating_table_is_the_one_the_logits_give():
    C, ci, y, logits, _, _ = stacked()
    rows, lines = operating_rows(y, logits, ci)
    t = readme()
    for line in lines:
        assert line in t, line
    aucs = [roc_auc_score(y == ci, cough_scores(logits[s], ci)) for s in SEEDS]
    assert "AUC " + ", ".join(f"{a:.3f}" for a in aucs) in t
    share = sorted(38 / (38 + fa) for fa in rows["c38"])
    assert f"**{pct(share[0], 1)}–{pct(share[-1], 1)} % of the alarms are coughs**" in t
    assert (min(rows["fa3"]), max(rows["fa3"])) == (28, 30)
    assert (pct(min(rows["c38"]), 360), pct(max(rows["c38"]), 360)) == (9, 13)


def test_false_alarms_at_high_recall_come_from_sneezing_first():
    C, ci, y, logits, _, _ = stacked()
    tot = Counter()
    for s in SEEDS:
        sc = cough_scores(logits[s], ci)
        thr = np.sort(sc[y == ci])[::-1][37]
        tot.update(C[k] for k in y[(y != ci) & (sc >= thr)])
    assert sum(tot.values()) == 120
    assert tot.most_common(3) == [("sneezing", 35), ("laughing", 31), ("drinking_sipping", 22)]
    assert "35 of the 120 over the three seeds, laughing 31, drinking 22" in readme()


def test_seed_spread_and_three_seed_average():
    C, ci, y, logits, pred, _ = stacked()
    right = [int((pred[s] == y).sum()) for s in SEEDS]
    assert right == [343, 331, 331]
    acc = np.array(right) / 400
    assert f"{acc.mean():.3f} ± {acc.std(ddof=1):.3f}" == "0.838 ± 0.017"
    prob = 0
    for s in SEEDS:
        e = np.exp(logits[s] - logits[s].max(axis=1, keepdims=True))
        prob = prob + e / e.sum(axis=1, keepdims=True)
    assert int((prob.argmax(1) == y).sum()) == 345
    t = readme()
    assert "343, 331 and 331 right: 0.838 ± 0.017 across seeds" in t
    assert "gets 345 -- two clips more than the best single seed" in t


def test_results_file_agrees_with_the_logits_and_carries_the_sources():
    C, ci, y, logits, pred, n_right = stacked()
    j = load("stage1_clip_checks.json")
    for k, c in enumerate(C):
        m = y == k
        assert j["per_class"][c]["pooled_right_of_120"] == int(sum((pred[s][m] == k).sum() for s in SEEDS))
        assert j["per_class"][c]["majority_right_of_40"] == int((n_right[m] >= 2).sum())
        assert j["per_class"][c]["never_right"] == int((n_right[m] == 0).sum())
    rows, _ = operating_rows(y, logits, ci)
    for i, s in enumerate(SEEDS):
        o = j["cough_vs_rest_per_seed"][str(s)]
        assert [o["caught_at_false_alarms"][k] for k in ("0", "3", "18")] == \
            [rows["zero"][i], rows["fa3"][i], rows["fa18"][i]]
        assert [o["false_alarms_to_catch"][k]["false_alarms_of_360"] for k in ("36", "38", "40")] == \
            [rows["c36"][i], rows["c38"][i], rows["c40"][i]]
    c = j["cough"]
    assert c["distinct_clips_with_an_error"] == len(c["clips_with_an_error"]) == 15
    for e in c["clips_with_an_error"]:             # ESC-50 names: fold-id-take-target, cough = 24
        fold, fid, _, target = e["filename"][:-4].split("-")
        assert (int(fold), int(fid), target) == (e["fold"], e["freesound_id"], "24")
    assert c["always_laughing_freesound_ids"] == [63679, 87794, 87795, 87799, 157296]
    src = j["sources"]
    assert src["total"] == 326
    assert src["per_class"]["coughing"]["sources"] == 39
    assert src["per_class"]["sneezing"]["sources"] == 40
    assert (src["per_class"]["crying_baby"]["sources"],
            src["per_class"]["crying_baby"]["largest_source_clips"]) == (18, 7)
    t = readme()
    assert "The 400 clips come from 326 source recordings" in t
    assert "the 40 coughs from 39, the 40 sneezes from 40, but the 40 `crying_baby` clips from 18, one of which supplies seven" in t
    assert "(87794, 87795, 87799)" in t
