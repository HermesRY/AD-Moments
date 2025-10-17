# Experiment 2 — Daily CTMS Pattern (Clean Release)

This directory contains the public-facing pipeline for Experiment 2, comparing
daily CTMS anomaly patterns between cognitively normal (CN) and cognitively
impaired (CI) cohorts. The workflow mirrors `Exp1_CTMS` and runs entirely on the
sample dataset bundled with this repository.

- **Data**: `../sample_data/dataset_one_month.jsonl`
- **Model**: `../Model/ctms_model.py`
- **Runner**: `run_exp2_pattern.py`
- **Config**: `config.yaml`
- **Outputs**: `outputs/`

## Workflow
1. Load the unified dataset and construct sliding windows over each subject’s
   action sequence.
2. Calibrate CTMS encoder norms on a fixed CN cohort (see `config.yaml`).
3. Score the remaining CN and CI windows using positive-only (directional) CTMS
   deviations.
4. Aggregate hourly mean scores (30-minute bins from 10:00–19:30) and render the
   CN vs CI temporal anomaly figure alongside reproducible CSV/JSON artifacts.

## Key configuration knobs
- **`split.cn_train_ids`**: CN subjects used to estimate healthy CTMS baselines.
- **`filter.drop_subjects`**: evaluation exclusions for the clean cohort
  (publication default removes CN anon_ids 18, 42, 43).
- **`ctms.alpha`**: weights applied to the four latent CTMS components when
  combining per-window z-scores.
- **`anomaly.threshold`**: weighted score threshold to mark a window as anomalous
  (set to `0.0` to keep every positive deviation).
- **`time_bins`**: start/end hour and bin size for the temporal aggregation.

## Usage
From this folder:

```bash
pip install -r ../requirements.txt
python run_exp2_pattern.py
```

Generated artifacts:
- `outputs/exp2_daily_pattern.(png|pdf)` — publication-style CN vs CI curve
- `outputs/exp2_daily_pattern_rates.csv` — per-bin exposures and mean scores
- `outputs/exp2_daily_pattern_summary.json` — headline metrics (peak hour,
  mean-score ratio, subject lists)
- `outputs/exp2_daily_pattern_subject_metrics.csv` — subject-level aggregates
- `outputs/exp2_config_used.json` — resolved configuration for provenance

For exploratory heuristics and alternative cohort selections, see
`../Exp2_Pattern_backup/`.
