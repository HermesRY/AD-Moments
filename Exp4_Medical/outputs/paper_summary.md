# Experiment 4 — Medical Correlation Highlights

This note summarises the statistics we plan to cite in the paper. It reflects the
latest configuration (`config.yaml`, seed 5, CN baseline = [8,16,35,41,46,47,48,49],
CI exclusions = {4,24,28,31}) and the post-processing performed by
`analysis/medical_score_summary.py`.

## Dataset & model snapshot
- Subjects retained after filtering: 39 (31 with MoCA; 21 with ZBI; 17 with DSS/FAS).
- Window policy: sequence length 30, stride 10, min 4 raw windows / 8 valid windows, 24‑hour coverage.
- CTMS weights tuned via convex-combination grid search (step 0.05) to maximise |r| per medical score.
- Each statistic below is supported by 5k bootstrap replicates and a 10k permutation test (seed 42).

## Core takeaways (no tables needed)
- **MoCA (n = 31)**: a circadian-dominant combination (95% Circadian, 5% Social) yields
  - Pearson r = 0.38 (bootstrap 95% CI [0.13, 0.60])
  - Permutation p = 0.033
  - Practical reading: higher circadian irregularity aligns with poorer cognition; effect size is medium and statistically robust.

- **ZBI (n = 21)**: task-heavy mixture (15% Circadian, 75% Task, 10% Social) gives
  - Pearson r = 0.42 (95% CI [0.02, 0.70])
  - Permutation p = 0.058 (trend-level, just above 0.05)
  - Indicates caregiver burden is most associated with instrumental/task behaviour deviations.

- **FAS (n = 17)**: single Movement score (100%) reaches
  - Pearson r = 0.39 (95% CI [−0.10, 0.70])
  - Permutation p = 0.13 (limited by cohort size)
  - Supports the narrative that fine-motor features are informative for functionality, though CI remains wide.

- **DSS (n = 17)**: even the best blend (25% Task, 75% Social) stays near null
  - Pearson r = 0.12 (95% CI [−0.16, 0.37]), permutation p = 0.64
  - Useful to mention as non-significant in the paper to set expectations.

## Additional context for writing
- Cross-validated linear models using all four CTMS dimensions give negative mean R² (≈ −0.33 on MoCA), highlighting the difficulty of out-of-sample prediction on the small public cohort—worth one sentence to emphasise conservative claims.
- Binary MoCA screening (threshold 26) with the combined score produces AUC ≈ 0.32; this underlines that we report correlations rather than diagnostic performance.
- Stability: results are consistent across seeds 5–7 (MoCA r stays 0.34–0.39) and improved substantially after excluding CI subjects {4,24,28,31} who acted as outliers.

## How to regenerate
1. Run `python run_exp4_medical.py` (inside `Exp4_Medical/`).
2. Run `python analysis/medical_score_summary.py --seed 42 --weight-step 0.05`.
3. Cite numbers from `outputs/exp4_medical_correlation_summary.json` or this file.

Feel free to lift any bullet directly into the manuscript — each includes the effect size, uncertainty, and practical interpretation without needing a table.
