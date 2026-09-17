# Changelog

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
