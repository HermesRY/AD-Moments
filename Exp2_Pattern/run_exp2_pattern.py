#!/usr/bin/env python3
"""Experiment 2 (Daily Pattern) — clean publication runner.

This script reproduces the daily temporal anomaly pattern comparison between
cognitively normal (CN) and cognitively impaired (CI) cohorts using the
one-month sample dataset.

Pipeline overview:
1. Load unified dataset (``sample_data/dataset_one_month.jsonl``).
2. Use a fixed subset of CN subjects (from ``config.yaml``) to establish CTMS
   baseline statistics (per-dimension encoder norms).
3. Evaluate the remaining CN and CI subjects, compute per-window anomaly scores,
   and aggregate anomaly rates across hourly bins.
4. Plot CN vs CI daily anomaly patterns and export per-bin/per-subject metrics.

The runner reads ``config.yaml`` in the same folder and writes artifacts under
``outputs/``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Path setup and configuration
# ---------------------------------------------------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
AD_DIR = os.path.abspath(os.path.dirname(THIS_DIR))  # AD-Moments root
if AD_DIR not in os.sys.path:
    os.sys.path.insert(0, AD_DIR)

from Model.ctms_model import CTMSModel  # noqa: E402 (local import after sys.path tweak)

CONFIG_PATH = os.environ.get('EXP2_PATTERN_CONFIG', os.path.join(THIS_DIR, 'config.yaml'))
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.abspath(os.path.join(THIS_DIR, CFG['dataset_path']))
OUTPUT_DIR = os.path.join(THIS_DIR, CFG['outputs']['dir'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = int(CFG['ctms']['d_model'])
NUM_ACTIVITIES = int(CFG['num_activities'])
ALPHA = np.asarray(CFG['ctms'].get('alpha', [0.25, 0.25, 0.25, 0.25]), dtype=float)
ALPHA = ALPHA / ALPHA.sum()  # ensure normalization

WINDOW_CFG = CFG['window']
SEQ_LEN = int(WINDOW_CFG['seq_len'])
STRIDE = int(WINDOW_CFG['stride'])
BATCH = int(WINDOW_CFG['batch'])
if DEVICE.type == 'cuda':
    BATCH = int(WINDOW_CFG.get('batch_cuda', BATCH))

TIME_CFG = CFG['time_bins']
START_H = float(TIME_CFG['start_hour'])
END_H = float(TIME_CFG['end_hour'])
BIN_MIN = int(TIME_CFG['bin_minutes'])
BIN_WIDTH = BIN_MIN / 60.0
NUM_BINS = int(math.ceil((END_H - START_H) / BIN_WIDTH))
BIN_EDGES = np.asarray([START_H + i * BIN_WIDTH for i in range(NUM_BINS + 1)], dtype=float)
TIME_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0

ANOM_CFG = CFG['anomaly']
USE_ABS_Z = bool(ANOM_CFG.get('use_abs_z', True))
ANOM_THRESHOLD = float(ANOM_CFG.get('threshold', 2.0))
MIN_WINDOWS_PER_SUBJECT = int(ANOM_CFG.get('min_windows_per_subject', 0))

FILTER_CFG = CFG['filter']
MIN_WINDOWS = int(FILTER_CFG.get('min_windows', 0))
DROP_SUBJECTS = {
    'CN': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CN', [])},
    'CI': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CI', [])},
}
MIN_CN_EVAL = int(FILTER_CFG.get('min_cn_eval', 0))
MIN_CI_EVAL = int(FILTER_CFG.get('min_ci_eval', 0))
BALANCE_CFG = FILTER_CFG.get('balance', {}) or {}

TRAIN_IDS = [int(x) for x in CFG['split']['cn_train_ids']]

OUTPUTS_CFG = CFG['outputs']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_id(raw) -> int:
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ValueError(f"Subject anon_id {raw!r} is not an integer.")


def load_dataset(path: str) -> pd.DataFrame:
    data = []
    bad = 0
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            if 'sequence' in obj and isinstance(obj['sequence'], list):
                obj['sequence'] = sorted(obj['sequence'], key=lambda e: e.get('ts', 0))
            data.append(obj)
    if bad:
        print(f"[warn] skipped {bad} malformed lines from {os.path.basename(path)}")
    df = pd.DataFrame(data)
    return df[df['label'].isin(['CN', 'CI'])].copy()


def to_windows(events: List[Dict], seq_len: int, stride: int) -> List[Dict]:
    if len(events) < seq_len:
        return []
    timestamps = np.asarray([e['ts'] for e in events], dtype=np.int64)
    hours = (timestamps % 86400) / 3600.0
    actions = np.asarray([e['action_id'] for e in events], dtype=np.int64)

    windows = []
    for start in range(0, len(events) - seq_len + 1, stride):
        end = start + seq_len
        acts_slice = actions[start:end]
        hours_slice = hours[start:end]
        if acts_slice.size != seq_len:
            continue
        center_hour = float(hours_slice[hours_slice.size // 2])
        windows.append({
            'actions': acts_slice,
            'hours': hours_slice.astype(np.float32),
            'center_hour': center_hour,
        })
    return windows


@dataclass
class EncodedWindow:
    anon_id: int
    label: str
    center_hour: float
    norms: Dict[str, float]


def encode_windows(model: CTMSModel, subjects: Dict[int, Dict]) -> List[EncodedWindow]:
    """Encode all windows (per subject) and return per-window norms."""
    encoded: List[EncodedWindow] = []
    iterator = tqdm(subjects.items(), desc='Encoding windows', leave=False)
    for sid, info in iterator:
        windows = info['windows']
        if not windows:
            continue
        acts_batches = []
        hrs_batches = []
        centers = []
        for win in windows:
            acts_batches.append(win['actions'])
            hrs_batches.append(win['hours'])
            centers.append(win['center_hour'])
        acts_np = np.stack(acts_batches, axis=0)
        hrs_np = np.stack(hrs_batches, axis=0)
        centers_np = np.asarray(centers, dtype=np.float32)

        feats = {'circadian': [], 'task': [], 'movement': [], 'social': []}
        with torch.no_grad():
            for i in range(0, acts_np.shape[0], BATCH):
                batch_actions = torch.from_numpy(acts_np[i:i + BATCH]).to(DEVICE)
                batch_hours = torch.from_numpy(hrs_np[i:i + BATCH]).to(DEVICE)
                enc = model(batch_actions, batch_hours, return_encodings_only=True)
                feats['circadian'].append(enc['h_c'].detach().cpu().numpy())
                feats['task'].append(enc['h_t'].detach().cpu().numpy())
                feats['movement'].append(enc['h_m'].detach().cpu().numpy())
                feats['social'].append(enc['h_s'].detach().cpu().numpy())

        norm_arrays = {}
        for key, arrs in feats.items():
            concat = np.concatenate(arrs, axis=0)
            norm_arrays[key] = np.linalg.norm(concat, axis=1)

        for idx in range(norm_arrays['circadian'].shape[0]):
            encoded.append(EncodedWindow(
                anon_id=sid,
                label=info['label'],
                center_hour=float(centers_np[idx]),
                norms={
                    'Circadian': float(norm_arrays['circadian'][idx]),
                    'Task': float(norm_arrays['task'][idx]),
                    'Movement': float(norm_arrays['movement'][idx]),
                    'Social': float(norm_arrays['social'][idx]),
                }
            ))
    return encoded


# ---------------------------------------------------------------------------
# Load dataset and construct windows per subject
# ---------------------------------------------------------------------------
df = load_dataset(DATASET_PATH)
subject_windows: Dict[int, Dict] = {}
for row in df.itertuples(index=False):
    sid = normalize_id(getattr(row, 'anon_id'))
    label = getattr(row, 'label')
    if label not in ['CN', 'CI']:
        continue
    if sid in DROP_SUBJECTS.get(label, set()):
        continue
    seqs = to_windows(getattr(row, 'sequence'), SEQ_LEN, STRIDE)
    subject_windows[sid] = {
        'label': label,
        'windows': seqs,
    }

# Filter out subjects with too few windows
subject_windows = {
    sid: info for sid, info in subject_windows.items()
    if len(info['windows']) >= MIN_WINDOWS
}

if not set(TRAIN_IDS).issubset(subject_windows.keys()):
    missing = sorted(set(TRAIN_IDS) - subject_windows.keys())
    raise RuntimeError(
        f"Training CN IDs {missing} are missing after filtering; please adjust config or filters.")

# Split into training/evaluation cohorts
train_subjects = {sid: subject_windows[sid] for sid in TRAIN_IDS}
eval_subjects = {
    sid: info for sid, info in subject_windows.items()
    if not (info['label'] == 'CN' and sid in TRAIN_IDS)
}

# Optional balancing (down-sample CI/CN counts)
if BALANCE_CFG:
    rng = np.random.default_rng(BALANCE_CFG.get('seed', 42))
    max_cn = BALANCE_CFG.get('max_cn')
    max_ci = BALANCE_CFG.get('max_ci')
    grouped = {'CN': [], 'CI': []}
    for sid, info in eval_subjects.items():
        grouped[info['label']].append(sid)
    for label, limit in (('CN', max_cn), ('CI', max_ci)):
        ids = grouped[label]
        if limit is None or limit <= 0 or len(ids) <= limit:
            continue
        keep = set(rng.choice(ids, size=int(limit), replace=False))
        eval_subjects = {
            sid: info for sid, info in eval_subjects.items()
            if info['label'] != label or sid in keep
        }

# Ensure minimum evaluation cohort sizes
cn_eval = sum(1 for info in eval_subjects.values() if info['label'] == 'CN')
ci_eval = sum(1 for info in eval_subjects.values() if info['label'] == 'CI')
if cn_eval < MIN_CN_EVAL or ci_eval < MIN_CI_EVAL:
    raise RuntimeError(
        f"Not enough evaluation subjects after filtering — CN: {cn_eval}, CI: {ci_eval}"
    )

print(f"Loaded dataset: {len(df)} subjects -> train CN: {len(train_subjects)}, eval CN: {cn_eval}, eval CI: {ci_eval}")

# ---------------------------------------------------------------------------
# Encode windows with CTMS
# ---------------------------------------------------------------------------
model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES).to(DEVICE)
model.eval()

encoded_train = encode_windows(model, train_subjects)
encoded_eval = encode_windows(model, eval_subjects)
eval_windows_by_subject: Dict[int, List[EncodedWindow]] = {}
for win in encoded_eval:
    eval_windows_by_subject.setdefault(win.anon_id, []).append(win)

if not encoded_train:
    raise RuntimeError('Training CN windows are empty; cannot compute baseline statistics.')

# ---------------------------------------------------------------------------
# Baseline statistics from training CN windows
# ---------------------------------------------------------------------------
train_by_dim = {'Circadian': [], 'Task': [], 'Movement': [], 'Social': []}
for win in encoded_train:
    for dim, val in win.norms.items():
        train_by_dim[dim].append(val)

baseline_stats = {}
for dim, vals in train_by_dim.items():
    arr = np.asarray(vals, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        std = 1.0
    baseline_stats[dim] = {'mean': float(mean), 'std': float(std)}

# ---------------------------------------------------------------------------
# Score evaluation windows and aggregate per time bin
# ---------------------------------------------------------------------------
exposure_counts = {
    'CN': np.zeros(NUM_BINS, dtype=float),
    'CI': np.zeros(NUM_BINS, dtype=float),
}
anomaly_counts = {
    'CN': np.zeros(NUM_BINS, dtype=float),
    'CI': np.zeros(NUM_BINS, dtype=float),
}
score_sums = {
    'CN': np.zeros(NUM_BINS, dtype=float),
    'CI': np.zeros(NUM_BINS, dtype=float),
}

subject_metrics = {}

for sid, info in eval_subjects.items():
    label = info['label']
    windows = eval_windows_by_subject.get(sid, [])
    if len(windows) < MIN_WINDOWS_PER_SUBJECT:
        continue

    total_windows = len(windows)
    used_windows = 0
    anomaly_windows = 0
    score_total = 0.0

    for win in windows:
        hour = win.center_hour
        if hour < START_H or hour >= END_H:
            continue
        bin_idx = int((hour - START_H) / BIN_WIDTH)
        bin_idx = min(max(bin_idx, 0), NUM_BINS - 1)

        z_scores = []
        for dim in ['Circadian', 'Task', 'Movement', 'Social']:
            mu = baseline_stats[dim]['mean']
            sigma = baseline_stats[dim]['std']
            z = (win.norms[dim] - mu) / sigma
            if USE_ABS_Z:
                z = abs(z)
            z_scores.append(z)
        z_scores = np.asarray(z_scores, dtype=float)
        if not USE_ABS_Z:
            z_scores = np.maximum(z_scores, 0.0)
        combined = float(np.sum(ALPHA * z_scores))
        if not USE_ABS_Z:
            combined = max(0.0, combined)
        if not np.isfinite(combined):
            continue

        exposure_counts[label][bin_idx] += 1
        score_sums[label][bin_idx] += combined
        used_windows += 1
        score_total += combined

        if combined >= ANOM_THRESHOLD:
            anomaly_counts[label][bin_idx] += 1
            anomaly_windows += 1

    if used_windows == 0:
        continue

    subject_metrics[sid] = {
        'anon_id': sid,
        'label': label,
        'total_windows': total_windows,
        'used_windows': used_windows,
        'anomaly_windows': anomaly_windows,
        'mean_score': score_total / used_windows,
        'anomaly_rate': anomaly_windows / used_windows,
    }

# ---------------------------------------------------------------------------
# Compute per-bin statistics
# ---------------------------------------------------------------------------
records = []
for label in ['CN', 'CI']:
    exposures = exposure_counts[label]
    anomalies = anomaly_counts[label]
    scores = score_sums[label]
    for idx in range(NUM_BINS):
        exp = exposures[idx]
        anom = anomalies[idx]
        score = scores[idx]
        rate = anom / exp if exp > 0 else float('nan')
        mean_score = score / exp if exp > 0 else float('nan')
        records.append({
            'label': label,
            'bin_index': idx,
            'hour_center': TIME_CENTERS[idx],
            'exposures': exp,
            'anomalies': anom,
            'anomaly_rate': rate,
            'mean_score': mean_score,
        })

rates_df = pd.DataFrame(records)
rates_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUTS_CFG['rates_csv']), index=False)

subject_df = pd.DataFrame(subject_metrics.values()).sort_values(by=['label', 'anon_id'])
subject_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUTS_CFG['per_subject_csv']), index=False)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
cn_curve = rates_df[rates_df['label'] == 'CN']['mean_score'].to_numpy()
ci_curve = rates_df[rates_df['label'] == 'CI']['mean_score'].to_numpy()
cn_curve = np.nan_to_num(cn_curve, nan=0.0)
ci_curve = np.nan_to_num(ci_curve, nan=0.0)

fig, ax = plt.subplots(figsize=(14, 6))

cn_color = '#2ECC71'
ci_color = '#E74C3C'

ax.plot(TIME_CENTERS, cn_curve, color=cn_color, linewidth=3, marker='o', markersize=8,
        alpha=0.9, label='CN')
ax.plot(TIME_CENTERS, ci_curve, color=ci_color, linewidth=3, marker='s', markersize=8,
        alpha=0.9, label='CI')

ax.fill_between(
    TIME_CENTERS,
    cn_curve,
    ci_curve,
    where=ci_curve >= cn_curve,
    color=ci_color,
    alpha=0.18,
    interpolate=True,
    label='CI > CN'
)

ax.set_xlabel('Time of Day', fontsize=24, fontweight='bold')
ax.set_ylabel('Avg Anomalous Moments\nper Subject per Hour', fontsize=24, fontweight='bold')
ax.set_title('Daily Temporal Activity Anomaly Patterns', fontsize=24, fontweight='bold', pad=18)
ax.set_xlim(START_H, END_H)

tick_hours = np.arange(math.ceil(START_H), END_H + 1e-6, 2.0)
tick_hours = tick_hours[(tick_hours >= START_H) & (tick_hours <= END_H)]
tick_indices = [int(np.argmin(np.abs(TIME_CENTERS - th))) for th in tick_hours]
ax.set_xticks([TIME_CENTERS[idx] for idx in tick_indices])
tick_labels = [f"{int(th):02d}:00" for th in tick_hours]
ax.set_xticklabels(tick_labels, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.8)
ax.set_axisbelow(True)

ylim_top = max(np.max(cn_curve), np.max(ci_curve))
ylim_top = ylim_top * 1.15 if ylim_top > 0 else 1.0
ax.set_ylim(0, ylim_top)
label_y = ylim_top * 0.95
shade_def = [
    ('Morning', START_H, 12.0, '#FF6F00', 0.05),
    ('Afternoon', 12.0, 18.0, '#2E7D32', 0.05),
    ('Eve', 18.0, 19.5, '#5E35B1', 0.05),
]
for name, start, end, color, alpha_val in shade_def:
    ax.axvspan(start, end, color=color, alpha=alpha_val, zorder=-1)
    ax.text((start + end) / 2, label_y, name, fontsize=22, fontweight='bold',
            color=color, ha='center', va='top', alpha=0.7)

ax.legend(loc='lower left', fontsize=20, framealpha=0.95, fancybox=True, edgecolor='gray')
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, OUTPUTS_CFG['figure'])
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, OUTPUTS_CFG['figure_pdf']), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
summary = {}
for label, curve in [('CN', cn_curve), ('CI', ci_curve)]:
    valid = np.isfinite(curve)
    if valid.any():
        peak_idx = int(np.nanargmax(curve))
        summary[f'{label.lower()}_peak_hour'] = float(TIME_CENTERS[peak_idx])
        summary[f'{label.lower()}_mean_score'] = float(np.nanmean(curve))
    else:
        summary[f'{label.lower()}_peak_hour'] = None
        summary[f'{label.lower()}_mean_score'] = None

if summary['cn_mean_score'] and summary['cn_mean_score'] > 0:
    summary['ci_vs_cn_ratio'] = float(summary['ci_mean_score'] / summary['cn_mean_score'])
else:
    summary['ci_vs_cn_ratio'] = None

if summary['cn_peak_hour'] is not None and summary['ci_peak_hour'] is not None:
    summary['peak_shift_hours'] = float(summary['ci_peak_hour'] - summary['cn_peak_hour'])
else:
    summary['peak_shift_hours'] = None

summary['train_cn_subjects'] = sorted(TRAIN_IDS)
summary['eval_cn_subjects'] = sorted(int(sid) for sid, info in eval_subjects.items() if info['label'] == 'CN')
summary['eval_ci_subjects'] = sorted(int(sid) for sid, info in eval_subjects.items() if info['label'] == 'CI')

with open(os.path.join(OUTPUT_DIR, OUTPUTS_CFG['summary_json']), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, OUTPUTS_CFG['config_used_json']), 'w', encoding='utf-8') as f:
    json.dump(CFG, f, indent=2)

print("\n=== Experiment 2 (Daily Pattern) Summary ===")
print(json.dumps(summary, indent=2))
print(f"Figure saved to: {fig_path}")
print(f"Rates CSV: {os.path.join(OUTPUT_DIR, OUTPUTS_CFG['rates_csv'])}")
print(f"Subject metrics CSV: {os.path.join(OUTPUT_DIR, OUTPUTS_CFG['per_subject_csv'])}")
