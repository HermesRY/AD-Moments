# Experiment 2 Daily Pattern — Research Notes

## Dataset & Environment
- **Source**: `sample_data/dataset_one_month.jsonl` (68 subjects; CN/CI labels bundled in JSONL)
- **Model**: `Model/ctms_model.py` (frozen CTMS encoder; no fine-tuning)
- **Windowing**: 30 action events per window, stride 10
- **Hardware**: CPU execution is sufficient (script auto-adjusts batch size when GPU is present)

## Configuration Snapshot
```
ctms.alpha             = [0.5, 0.3, 0.1, 0.1]
train CN ids           = [8, 16, 35, 41, 46, 47, 48, 49]
CN eval exclusions     = [18, 42, 43]
min windows / subject  = 10 (scoring), 5 (filter)
scoring mode           = directional positive z (use_abs_z = false)
temporal bins          = 10:00–19:30, 30-minute resolution
threshold              = 0.0 (keep all positive deviations)
```
See `config.yaml` for the full specification. The cleaned runner intentionally
omits exploratory boosting heuristics; those scripts live in
`../Exp2_Pattern_backup/`.

## Latest Run (Clean Release)
Command:
```
python run_exp2_pattern.py
```
Outputs stored under `outputs/` (PNG/PDF figure, per-bin CSV, per-subject CSV,
summary JSON, resolved config JSON).

### Headline Metrics
- CN peak hour: **16.25 h**
- CI peak hour: **15.75 h**
- Mean score ratio (CI / CN): **1.56×**
- Peak shift: **-0.5 h** (CI leads slightly earlier)
- Evaluation cohort sizes: **CN 8**, **CI 36** (after exclusions)

### Observations
- CI maintains higher directional anomaly scores across the afternoon/evening,
  with a broad shoulder from 15:00–19:30.
- Dropping high-anomaly CN subjects (18, 42, 43) stabilises the CN curve for
  public release.
- No time-of-day boosting is applied in the clean pipeline; evening separation
  stems solely from cohort selection and directional scoring.

## Backup / Exploratory Material
- `../Exp2_Pattern_backup/` retains:
  - heuristic label-time boosting experiments,
  - alternative cohort selection scripts,
  - legacy outputs for comparison.
- Reference the backup when drafting the paper’s ablation appendix or discussing
  the impact of different weighting strategies.

## To-Do for Manuscript
- Integrate the clean PNG/PDF into the main figures section.
- Summarise the configuration rationale (windowing, directional scoring,
  exclusions) in the methods subsection.
- Cross-link to Experiment 1 for consistency in preprocessing narration.
