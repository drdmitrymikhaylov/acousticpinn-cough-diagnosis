# Changelog

## 2026-11-16

- **What the 77 % allows** (new README subsection,
  `results/label_ceiling_check.json`, `tests/test_label_ceiling.py`, 6 tests).
  The physicians' 77.0 % agreement and the 476/2 063 labelled-wet share pin a
  one-parameter family of label-error models; along it a model that
  reproduced the truth exactly would score AUC 0.71–0.85 against one-rater
  labels (0.739 under symmetric 13 % errors), so the CNN's 0.701 is 57–97 % of
  the way from chance to the ceiling (84 % symmetric). The perfect model's
  sensitivity against the labels is 0.50 under symmetric errors.
- **Corrected:** the 0.770 in the results table is raw agreement, not
  balanced accuracy (two always-mostly-dry raters agree 65 % by prevalence).
  On the kappa scale the physicians sit at 0.35 (pairs 0.07–0.80; the three
  pairs with physician 3 at 0.07–0.11), the CNN against the labels at 0.22
  (argmax) and 0.25 (best threshold), a perfect model at 0.52–0.59.
- **Qualified:** the stage-2 ± are retraining spreads. Recording-level
  bootstrap: AUC 0.706 [0.681, 0.730]; sensitivity at 90/80/70 %
  specificity [0.22, 0.32] / [0.41, 0.50] / [0.53, 0.64] where the page's
  seed spread reads ± 0.03 / 0.00 / 0.00. The fold spread (sd 0.027) equals
  the sd of random 413-recording subsamples (0.027): no fold is harder than
  another.
- Limitations bullet on single-rater labels now carries the kappa reading.

## 2026-10-19

- LICENSE now holds the standard CC BY 4.0 legal text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-10-05

- Stage 1 read clip by clip (new README section "Is that 120 predictions or
  40 clips?", `results/stage1_clip_checks.json`, `tests/test_stage1_clips.py`,
  7 tests). The 30 cough errors sit on 15 clips; 7 of the 40 cough clips are
  missed by all three seeds and account for 21 of the 30.
- **Correction:** cough is not "the second-hardest class". Breathing is
  below it (88 against 90 of 120 predictions); by clips cough is fourth from
  the bottom, and breathing, coughing and laughing cannot be ranked on 40
  clips each.
- **Qualified:** "laughing, not sneezing" rests on 6 clips against 2
  (p = 0.29; 5 against 2 among the never-recognised, p = 0.45). As a detector
  at 38 of 40 coughs caught, sneezing is the largest source of false alarms
  (35 of 120 over the seeds, laughing 31), and 44-55 % of the alarms are
  coughs. Under 1 % false alarms the detector finds 28-30 coughs of 40.
- Stated the unit: 400 clips from 326 source recordings (crying_baby: 40
  clips from 18).
- Housekeeping in the same commit: three macOS `._*` files removed and
  ignored; the run-time note no longer names the laptop platform; the
  ore-body link points at the repository's current name.

## 2026-09-17

- Stage 2 read at operating points: CNN beats the linear baseline on 15/15
  paired fold×seed runs (ΔAUC +0.146 ± 0.036, worst +0.077); at 90 %
  specificity on dry coughs only 26 % of wet coughs are caught; the argmax
  threshold flags 50 % of recordings as wet when 23 % are. Seed-pooled AUC
  0.701 ± 0.004 shows the ± 0.027 run spread is mostly fold composition.
  New `results/stage2_operating_points.json`; README section added.
- Run-level results (`results/cv_runs.json`, `results/coughtype_runs.json`)
  now published, with `tests/test_readme_numbers.py` pinning every number
  on the page to them (10 tests).
