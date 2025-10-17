````markdown
# Experiment 4 — CTMS × Medical Correlation (Clean Release)

This folder packages a publication-ready workflow that connects CTMS behavioural
signatures with standardised clinical assessments (MoCA, ZBI, DSS, FAS). The
script mirrors the structure of Experiments 1–3 and runs end-to-end on the
public month-long dataset bundled with AD-Moments.

- **Data**: `../sample_data/dataset_one_month.jsonl`
- **Metadata**: `../sample_data/subjects_public.json`
- **Model**: `../Model/ctms_model.py`
- **Runner**: `run_exp4_medical.py`
- **Config**: `config.yaml`
- **Outputs**: `outputs/`

## Workflow overview
1. Load the month-long action streams and enrich them with subject metadata.
2. Generate CTMS embeddings on sliding windows and keep only daytime windows
   with adequate coverage.
3. Build reference statistics from a fixed CN cohort (see `split.cn_baseline_ids`).
4. Aggregate subject-level deviation scores for each CTMS dimension plus their
   weighted combination.
5. Join with medical assessments and report Pearson correlations, saving the
   results and a reproducible visualisation.

## Usage
From this folder:

```bash
pip install -r ../../requirements.txt
python run_exp4_medical.py
```

Key artefacts:
- `outputs/exp4_medical_subject_metrics.csv` — per-subject CTMS deviation
  scores alongside medical labels (ideal for downstream analysis).
- `outputs/exp4_medical_correlations.json` — structured summary of correlation
  coefficients, sample counts, and the best feature for each medical score.
- `outputs/exp4_medical_correlations.png` (and `.pdf`) — bar + scatter plot for
  the headline correlation identified in the JSON summary.

## Reference results (public sample dataset)

The bundled dataset is intentionally small; the clean runner still reproduces a
consistent ordering of correlations, albeit without statistical significance.
The table below highlights the strongest CTMS dimension for each medical score.

| Medical score | n (subjects) | Best CTMS feature | Pearson r | p-value |
| ------------- | ------------ | ----------------- | --------- | ------- |
| MoCA          | 30           | Circadian score   | 0.195     | 0.302   |
| ZBI           | 19           | Movement score    | 0.321     | 0.180   |
| DSS           | 15           | Social score      | 0.328     | 0.233   |
| FAS           | 15           | Movement score    | 0.337     | 0.219   |

To experiment with alternative cohorts or stricter filtering, edit
`config.yaml` (e.g., adjust the CN baseline IDs, change the minimum window
count, or swap the aggregation statistic from `mean` to `median`). Legacy
exploratory scripts remain in `../../exp4_medical/` for historical reference.
````