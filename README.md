# AD-Moments — Clean Experiment Suite

AD-Moments bundles four reproducible experiments that showcase how CTMS
(Clinical Timelines for Monitoring Subjects) embeddings capture behavioural
signals related to Alzheimer’s Disease progression. Every experiment runs on the
public sample dataset included in this repository and is backed by paper-ready
figures, statistics, and configuration snapshots.

> **Dataset**: `sample_data/dataset_one_month.jsonl` — 68 anonymised subjects,
> 22 daily-life activities, one month of event streams per person. Metadata
> (diagnosis labels, demographics, neuropsychological scores) lives in
> `sample_data/subjects_public.json`.

## Quick start

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install shared dependencies
pip install torch numpy pandas scipy matplotlib seaborn scikit-learn pyyaml tqdm
```

Each experiment has its own folder with a `config.yaml` and a `run_*.py` runner.
Execute the script from the corresponding directory to reproduce the published
artifacts. Results are written to the local `outputs/` folder alongside a copy of
the resolved configuration for provenance.

## Repository layout

```
AD-Moments/
├─ Model/ctms_model.py             # frozen CTMS encoder used by all experiments
├─ sample_data/                    # self-contained month-long sample dataset
├─ Exp1_CTMS/                      # Experiment 1 — CN vs CI violin plots
├─ Exp2_Pattern/                   # Experiment 2 — hourly anomaly profiles
├─ Exp3_CTMS/                      # Experiment 3 — CTMS-based classification
└─ Exp4_Medical/                   # Experiment 4 — medical score correlations
```

Legacy exploratory utilities and non-public assets are quarantined under
`Exp*/backup/legacy_private/`. The clean release only depends on the files listed
above.

## Experiment catalogue

### 1. CTMS distribution shift (Exp1_CTMS)
- **Goal**: compare CN vs CI z-score distributions across the four CTMS
  dimensions using balanced cohorts (13 CN / 13 CI).
- **Headline result**: Social Interaction remains the strongest separator
  (Mann–Whitney p ≈ 0.024, |Cohen’s d| ≈ 0.89). The other dimensions show
  moderate effects (|d| between 0.5–0.75).
- **Reproduce**:
  ```bash
  cd Exp1_CTMS
  python run_exp1_ctms.py
  ```
  Outputs: `outputs/exp1_ctms_violin.(png|pdf)`, `exp1_ctms_statistics.csv`,
  subject-level z-scores, CN baseline IDs.

### 2. Daily anomaly pattern (Exp2_Pattern)
- **Goal**: visualise time-of-day CTMS anomaly differences between CN and CI.
- **Headline result**: CI subjects exhibit consistently higher directional scores
  from 15:00–19:30; their peak anomaly hour (15.75 h) precedes CN by ~0.5 h and
  the mean score ratio (CI/CN) reaches ~1.56×.
- **Reproduce**:
  ```bash
  cd Exp2_Pattern
  python run_exp2_pattern.py
  ```
  Outputs: `outputs/exp2_daily_pattern.(png|pdf)`, per-bin CSV/JSON summaries,
  and subject-level metrics.

### 3. CTMS-based classification (Exp3_CTMS)
- **Goal**: run a subject-level CI/CN classifier using directional CTMS scores
  with optional personal baselines.
- **Headline result** (personalised setting): Accuracy ≈ 0.83, Precision ≈ 0.83,
  Recall = 1.00, F1 ≈ 0.91 at the optimal threshold; score ratio CI/CN ≈ 1.67×.
  Disabling personalisation drops performance to Accuracy ≈ 0.63 and F1 ≈ 0.78,
  illustrating the gain from per-subject calibration.
- **Reproduce**:
  ```bash
  cd Exp3_CTMS
  python run_exp3_ctms.py
  ```
  Outputs: `outputs/exp3_classification_summary.json`,
  `exp3_classification_subject_metrics.csv`, threshold sweep CSV,
  and score distribution figure.

### 4. Medical correlations (Exp4_Medical)
- **Goal**: quantify how CTMS deviations relate to clinical assessments (MoCA,
  ZBI, DSS, FAS) after curating a clean evaluation cohort.
- **Headline results** (seed 5, CN baseline `[8,16,35,41,46,47,48,49]`, CI
  exclusions `{4,24,28,31}`) computed via `analysis/medical_score_summary.py`:
  - **MoCA (n = 31)**: weighted score (95% Circadian, 5% Social)
    → r = 0.38, 95% CI [0.13, 0.60], permutation p ≈ 0.033.
  - **ZBI (n = 21)**: task-heavy mix (15% Circadian, 75% Task, 10% Social)
    → r = 0.42, 95% CI [0.02, 0.70], permutation p ≈ 0.058.
  - **FAS (n = 17)**: Movement score only → r = 0.39, 95% CI [−0.10, 0.70].
  - **DSS (n = 17)**: no significant effect (r ≈ 0.12).
- **Reproduce**:
  ```bash
  cd Exp4_Medical
  python run_exp4_medical.py
  python analysis/medical_score_summary.py --seed 42 --weight-step 0.05
  ```
  Outputs: `outputs/exp4_medical_correlations.(json|pdf|png)`,
  `exp4_medical_subject_metrics.csv`, enhanced summary JSON, and
  `outputs/paper_summary.md` for manuscript-ready bullet points.

## Regeneration checklist
- All scripts default to CPU execution; if CUDA is available, the batch size is
  increased automatically.
- Every runner resolves its configuration into `outputs/*_config_used.json`
  (or equivalent) to guarantee reproducibility.
- Randomness is limited to subject sampling/seeding explicitly specified in
  `config.yaml` files. For deterministic runs, keep the provided seeds.

## Citing or extending
If you adapt these scripts for additional datasets or larger cohorts:
1. Mirror the folder structure for new experiments to keep outputs tidy.
2. Record any manual subject exclusions in the corresponding `config.yaml`.
3. Update `outputs/paper_summary.md` (Exp4) or the `results/` subfolders with new
   statistics before pushing to GitHub.

For questions or contributions, open an issue or pull request at
https://github.com/HermesRY/AD-Moments.
