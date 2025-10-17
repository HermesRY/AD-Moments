````markdown
# Experiment 3 — CTMS Classification (Clean Release)

This folder hosts the public-ready CI/CN classification pipeline that
demonstrates how CTMS anomaly scores separate cognitively impaired (CI) from
cognitively normal (CN) participants. It mirrors the style of `Exp1_CTMS` and
`Exp2_Pattern` and runs end-to-end on the included sample dataset.

- **Data**: `../sample_data/dataset_one_month.jsonl`
- **Model**: `../Model/ctms_model.py`
- **Runner**: `run_exp3_ctms.py`
- **Config**: `config.yaml`
- **Outputs**: `outputs/`

## Workflow overview
1. Load the unified dataset and create sliding windows of actions for every
   subject.
2. Fit global CTMS baseline statistics on a fixed CN cohort (see
   `split.cn_train_ids`).
3. Score the remaining CN/CI subjects with directional CTMS deviations,
   optionally building subject-specific ("personal") baselines from the early
   portion of each timeline.
4. Aggregate subject scores, sweep thresholds to maximise F1, and report the
   final sensitivity, specificity, and CI/CN score ratio alongside a plot.

## Key configuration knobs
- **`split.cn_train_ids`** — CN subjects used to estimate the reference baseline.
- **`filter.drop_subjects`** — subjects excluded from evaluation in the clean
  cohort (same defaults as Experiment 2).
- **`ctms.alpha`** — weights applied to the four CTMS latent components when
  combining z-scores.
- **`scoring.personal_fraction` / `personal_min_windows`** — controls the
  personal baseline window before evaluation begins.
- **`classification.threshold_percentiles`** — candidate percentiles used in the
  threshold sweep (F1 maximisation).

## Usage
From this folder:

```bash
pip install -r ../requirements.txt
python run_exp3_ctms.py
```

Generated artifacts:
- `outputs/exp3_classification_summary.json` — headline metrics and subject IDs
  used for training/evaluation (F1, sensitivity, specificity, CI/CN score ratio).
- `outputs/exp3_classification_subject_metrics.csv` — subject-level score
  aggregates and predictions.
- `outputs/exp3_score_distribution.png` — publication-style box/strip plot with
  the optimal threshold overlayed.
- `outputs/exp3_threshold_metrics.csv` — every threshold candidate evaluated
  during the sweep (helpful when manually selecting an operating point).

## Reference results

The table below summarises two operating points that we plan to reference in the
public report. Both runs were executed with the clean runner in this folder —
the only difference is whether the personal baseline is enabled and the fully
public hyper-parameters used for the sweep.

| Method | Personalisation | Accuracy | Precision | Recall / Sensitivity | F1 | CI/CN rate ratio |
| ------ | --------------- | -------- | --------- | -------------------- | --- | ---------------- |
| AD‑Moments (No Personal.) | ✗ (set `personal_fraction` 0.0, `window_threshold` 0.9) | 0.634 | 0.788 | 0.765 | 0.776 | 0.77× |
| **AD‑Moments (Personal.)** | ✓ (`personal_fraction` 0.2, default config) | 0.829 | 0.829 | 1.000 | 0.907 | 1.67× |

Reproducing the no-personal baseline only requires editing `config.yaml` before
running the script:

```yaml
scoring:
  use_abs_z: false
  personal_fraction: 0.0
  personal_min_windows: 0
  window_threshold: 0.9
```

Remember to restore the published defaults (`personal_fraction: 0.2`,
`personal_min_windows: 15`, `window_threshold: 1.4`) when switching back to the
personalised setting.

For legacy exploratory scripts (logistic regression baselines, detailed
personalisation sweeps, etc.) refer to `../../docs/Exp3_CTMS_backup/`.
````