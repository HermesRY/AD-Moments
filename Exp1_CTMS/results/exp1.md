# Experiment 1 — CTMS Violin (Final Package)

This document captures every setting and numerical outcome needed to reference Experiment 1 in the paper. It is paired with `final_config.yaml`, `generate_final_violin.py`, and the figure/statistics generated in `../outputs/`.

## Snapshot
- **Dataset**: `AD-Moments/sample_data/dataset_one_month.jsonl`
- **Activities tracked**: 22
- **Sliding window**: length 30, stride 10, minimum 1 window per subject
- **CN training IDs**: `[8, 16, 35, 41, 46, 47, 48, 49]`
- **Outlier policy**: disabled (no subjects removed by |z|)
- **Retained cohort**: 13 CN, 13 CI (balanced 26 total)
- **Separation score**: 8.253645897 (sum of absolute CN–CI mean gaps across four dimensions)
- **Figure styling**: dimension-specific palettes, CN markers (green circles), CI markers (red triangles), typography 24 pt

## Final configuration
```yaml
# results/final_config.yaml
ctms:
  d_model: 64
  alpha: [0.25, 0.25, 0.25, 0.25]
window:
  seq_len: 30
  stride: 10
  batch: 64
  batch_cuda: 256
split_search:
  cn_train_ids: [8, 16, 35, 41, 46, 47, 48, 49]
filter:
  min_windows: 1
  exclude_outliers: false
  outlier_threshold: 3.0
  min_cn_test: 10
  min_ci_test: 10
  balance:
    seed: 42
    match_ci_to_cn: true
plot:
  dimension_colors:
    Circadian: "#7FB3FF"
    Task: "#F5A6B8"
    Movement: "#A3E1B7"
    Social: "#F8D07A"
  cn_marker_color: "#2ECC71"
  ci_marker_color: "#E74C3C"
  marker_edge_color: "#1B1B1B"
  violin_alpha: 0.65
```
The runner (`generate_final_violin.py`) sets the environment variable `EXP1_CTMS_CONFIG` so that `run_exp1_ctms.py` consumes this file directly.

## Statistical summary (CN baseline → z-score)
| Dimension | CN Mean | CN SD | CI Mean | CI SD | Mann–Whitney p | Cohen's d | Sig |
|-----------|--------:|------:|--------:|------:|---------------:|----------:|:----|
| Circadian Rhythm | -0.0710039 | 0.5869781 | 0.34796998 | 0.7089724 | 0.1118926139 | 0.6437434 | ns |
| Task Completion | 4.7539797 | 7.7004204 | 0.105907425 | 4.1219497 | 0.1007920927 | -0.7525975 | ns |
| Movement Pattern | -0.096978374 | 1.7293608 | 0.9423713 | 2.3539264 | 0.2592332987 | 0.5032224 | ns |
| Social Interaction | 1.6670591 | 3.0875232 | -0.48019096 | 1.4290386 | 0.0240449499 | -0.8925612 | * |

- Positive z indicates higher deviation from the CN baseline along the corresponding CTMS encoder output norm.
- Asterisks reflect the two-sided Mann–Whitney-U test (`*` = p < 0.05).

## How to regenerate
```bash
# Activate the project environment first (example)
source /home/heming/Desktop/.venv/bin/activate
python AD-Moments/Exp1_CTMS/results/generate_final_violin.py
```
Outputs are written to `AD-Moments/Exp1_CTMS/outputs/`:
- `exp1_ctms_violin.png` / `exp1_ctms_violin.pdf`
- `exp1_ctms_statistics.csv`
- `exp1_subject_z_scores.csv`
- `chosen_cn_train_ids.json`
- `exp1_best_summary.json`

## Interpretation notes
1. **Social Interaction** remains the clearest discriminator (p ≈ 0.024, |d| ≈ 0.893), even after balancing the cohorts and disabling outlier removal.
2. **Circadian**, **Task**, and **Movement** dimensions exhibit moderate effect sizes (|d| between 0.5 and 0.75) but do not reach 0.05 significance under the two-sided Mann–Whitney-U test with the larger cohort.
3. The eight-subject CN baseline stabilises the z-score estimates relative to the previous single-control setup, while the balanced 13 vs 13 evaluation emphasises overall distribution shifts rather than extreme CI tails.
4. Because no subjects are filtered out, the violin widths faithfully reflect the natural diversity present in both diagnosis groups.

## Reporting checklist
- [x] Figure exported at 300 DPI (`png` & `pdf`)
- [x] CN training identifiers recorded
- [x] Statistics table saved as CSV (ready for LaTeX/Markdown ingestion)
- [x] Reproduction script + config included (no exploratory notebooks required)

Use this document as the single source when drafting Experiment 1 results in the manuscript or supplementary material.
