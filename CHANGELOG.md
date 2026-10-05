# Changelog

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
