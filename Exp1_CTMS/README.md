# Experiment 1 — CTMS Violin (Clean)

This folder provides a clean, reproducible version of Experiment 1 to visualize distribution differences between CN and CI across the four CTMS dimensions.

- Data: `../sample_data/dataset_one_month.jsonl` (one month per subject, 68 subjects, 22 actions)
- Model: `../Model/ctms_model.py` (CTMS encoders)
- Script: `run_exp1_ctms.py`
- Config: `config.yaml` — controls split, filters, and plotting
- Outputs: `outputs/`

## What it does
1. Load one-month action sequences per subject.
2. Build CN baseline (mu, sigma) from a pre-selected training subset of CN subjects (see `config.yaml`).
3. Evaluate remaining CN + all CI, compute per-dimension z-scores.
4. Plot violin distributions for CN vs CI on each CTMS dimension and export statistics.

## Key knobs
- Training/testing split: specify `split_search.cn_train_ids` (balanced CN anon_ids). Exploratory search utilities live in `Exp1_backup/` for internal use.
- Subject filtering: enforce `min_windows`, optionally drop specific subjects and exclude outliers.
- Evaluation balance: require minimum test counts via `filter.min_cn_test` (default 2 for the sample subset) and `filter.min_ci_test`.
- CTMS weights: `ctms.alpha` controls how we report a combined AD z-score (for table only, violin is per-dimension).
- Windowing: `window.seq_len`, `window.stride`, `window.batch`, and `window.batch_cuda` (larger GPU batch if available).
- Progress & speed: encodings are cached once per subject; progress bars (via `tqdm`) show encoding status.

## Run
Just run the script; it will read `config.yaml` and write outputs under `outputs/`.

```bash
pip install torch numpy pandas scipy matplotlib seaborn pyyaml tqdm
python run_exp1_ctms.py
```

Artifacts:
- `outputs/exp1_ctms_violin.(png|pdf)` — violin figure
- `outputs/exp1_ctms_statistics.csv` — per-dimension stats (CN/CI mean/std, p-value, Cohen's d)
- `outputs/chosen_cn_train_ids.json` — selected CN IDs for the best split
- `outputs/exp1_config_used.json` — a copy of effective config used to run
- `outputs/exp1_subject_z_scores.csv` — subject-level z scores per dimension

## Notes
- The clean pipeline uses CTMS encoders as feature extractors without supervised training; it computes CN baselines directly from encoder norms as designed in the paper.
- If a subject has too few windows, it will be filtered to avoid noisy estimates.
- To reproduce the public figure, use the fixed CN IDs already listed in `config.yaml`. For custom split discovery, see `Exp1_backup/run_exp1_ctms_with_split_search.py`.
- When running on a GPU (e.g., RTX 4090) the script automatically uses the larger `batch_cuda` for faster encoding.
