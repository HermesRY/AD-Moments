#!/usr/bin/env python3
"""Experiment 4 (Medical Correlation) — clean publication runner.

This script computes subject-level CTMS deviation scores and quantifies how
strongly each behavioural dimension correlates with medical assessments such as
MoCA, ZBI, DSS, and FAS. It mirrors the structure of the other clean
experiments and runs end-to-end on the public sample dataset.

Pipeline outline:
1. Load the month-long action sequences (`sample_data/dataset_one_month.jsonl`)
   and the companion subject metadata (`sample_data/subjects_public.json`).
2. Build CTMS sliding windows per subject, filtering low-coverage cohorts.
3. Fit reference baseline statistics from a fixed CN cohort (defined in
   `config.yaml`).
4. Score every remaining subject by aggregating per-dimension CTMS deviations.
5. Join the scores with clinical assessments and report Pearson correlations,
   along with a publication-ready visualisation.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy import stats
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from Model.ctms_model import CTMSModel  # noqa: E402

CONFIG_PATH = os.environ.get('EXP4_MEDICAL_CONFIG', os.path.join(THIS_DIR, 'config.yaml'))
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.abspath(os.path.join(THIS_DIR, CFG['dataset_path']))
SUBJECT_METADATA_PATH = os.path.abspath(os.path.join(THIS_DIR, CFG['subject_metadata']))
OUTPUT_DIR = os.path.join(THIS_DIR, CFG['outputs']['dir'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_ACTIVITIES = int(CFG['num_activities'])
CTMS_CFG = CFG['ctms']
D_MODEL = int(CTMS_CFG['d_model'])
ALPHA = np.asarray(CTMS_CFG.get('alpha', [0.25, 0.25, 0.25, 0.25]), dtype=float)
ALPHA = ALPHA / ALPHA.sum()

WINDOW_CFG = CFG['window']
SEQ_LEN = int(WINDOW_CFG['seq_len'])
STRIDE = int(WINDOW_CFG['stride'])
BATCH = int(WINDOW_CFG.get('batch', 64))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    BATCH = int(WINDOW_CFG.get('batch_cuda', BATCH))

RANDOM_SEED = CFG.get('random_seed')
if RANDOM_SEED is not None:
    seed = int(RANDOM_SEED)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

FILTER_CFG = CFG['filter']
MIN_WINDOWS = int(FILTER_CFG.get('min_windows', 0))
MIN_SUBJECT_WINDOWS = int(FILTER_CFG.get('min_subject_windows', 0))
DROP_SUBJECTS = {
    'CN': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CN', [])},
    'CI': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CI', [])},
}
DAYTIME_ONLY = bool(FILTER_CFG.get('daytime_only', False))
DAYTIME_RANGE = FILTER_CFG.get('daytime_hours', [0.0, 24.0])
if not isinstance(DAYTIME_RANGE, (list, tuple)) or len(DAYTIME_RANGE) != 2:
    raise ValueError('`daytime_hours` must be a two-element list [start, end].')
DAYTIME_START = float(DAYTIME_RANGE[0])
DAYTIME_END = float(DAYTIME_RANGE[1])

SCORING_CFG = CFG['scoring']
USE_ABS_Z = bool(SCORING_CFG.get('use_abs_z', True))
SCORE_STATISTIC = SCORING_CFG.get('score_statistic', 'mean').lower()
if SCORE_STATISTIC not in {'mean', 'median'}:
    raise ValueError('`score_statistic` must be either "mean" or "median".')

MEDICAL_CFG = CFG['medical']
MEDICAL_SCORES = [str(s).lower() for s in MEDICAL_CFG.get('scores', [])]
MIN_SAMPLES = int(MEDICAL_CFG.get('min_samples', 0))

OUTPUT_CFG = CFG['outputs']
SUMMARY_JSON = os.path.join(OUTPUT_DIR, OUTPUT_CFG.get('summary_json', 'exp4_medical_correlations.json'))
SUBJECT_CSV = os.path.join(OUTPUT_DIR, OUTPUT_CFG.get('subject_csv', 'exp4_medical_subject_metrics.csv'))
FIG_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CFG.get('figure', 'exp4_medical_correlations.png'))
FIG_PDF_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CFG.get('figure_pdf', 'exp4_medical_correlations.pdf'))

BASELINE_IDS = [int(x) for x in CFG['split']['cn_baseline_ids']]

DIMS = ['Circadian', 'Task', 'Movement', 'Social']

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, 1):
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                continue
            if obj.get('label') not in {'CN', 'CI'}:
                continue
            seq = obj.get('sequence', [])
            if isinstance(seq, list):
                seq = sorted(seq, key=lambda item: item.get('ts', 0))
            obj['sequence'] = seq
            records.append(obj)
    return pd.DataFrame(records)


def load_subject_metadata(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    flattened: List[Dict] = []
    for entry in data:
        rec = {
            'anon_id': entry.get('anon_id'),
            'label_meta': entry.get('label'),
            'gender': entry.get('gender'),
            'age': entry.get('age'),
            'status': entry.get('status'),
        }
        scores = entry.get('scores', {}) or {}
        for key, value in scores.items():
            rec[str(key).lower()] = value
        flattened.append(rec)
    meta_df = pd.DataFrame(flattened)
    meta_df['anon_id'] = meta_df['anon_id'].astype(pd.Int64Dtype())
    return meta_df


def to_windows(events: Sequence[Dict], seq_len: int, stride: int) -> List[Dict]:
    if len(events) < seq_len:
        return []
    timestamps = np.asarray([e['ts'] for e in events], dtype=np.int64)
    hours = (timestamps % 86400) / 3600.0
    actions = np.asarray([e['action_id'] for e in events], dtype=np.int64)
    windows = []
    for start in range(0, len(events) - seq_len + 1, stride):
        end = start + seq_len
        acts_slice = actions[start:end]
        hrs_slice = hours[start:end]
        if acts_slice.size != seq_len:
            continue
        windows.append({
            'actions': acts_slice,
            'hours': hrs_slice.astype(np.float32),
        })
    return windows


def sort_windows_by_time(windows: List[Dict]) -> List[Dict]:
    if not windows:
        return windows
    indices = np.argsort([float(np.median(win['hours'])) for win in windows])
    return [windows[i] for i in indices]


@dataclass
class EncodedWindow:
    anon_id: int
    label: str
    embeddings: Dict[str, np.ndarray]
    mid_hour: float


def encode_windows(model: CTMSModel, subjects: Dict[int, Dict]) -> List[EncodedWindow]:
    encoded: List[EncodedWindow] = []
    iterator = tqdm(subjects.items(), desc='Encoding windows', leave=False)
    for sid, info in iterator:
        windows = info['windows']
        if not windows:
            continue
        acts_batches = np.stack([win['actions'] for win in windows], axis=0)
        hrs_batches = np.stack([win['hours'] for win in windows], axis=0)

        feats = {dim: [] for dim in DIMS}
        with torch.no_grad():
            for i in range(0, acts_batches.shape[0], BATCH):
                act_tensor = torch.from_numpy(acts_batches[i:i + BATCH]).to(DEVICE)
                hour_tensor = torch.from_numpy(hrs_batches[i:i + BATCH]).to(DEVICE)
                outputs = model(act_tensor, hour_tensor, return_encodings_only=True)
                feats['Circadian'].append(outputs['h_c'].detach().cpu().numpy())
                feats['Task'].append(outputs['h_t'].detach().cpu().numpy())
                feats['Movement'].append(outputs['h_m'].detach().cpu().numpy())
                feats['Social'].append(outputs['h_s'].detach().cpu().numpy())

        emb_arrays = {
            dim: np.concatenate(arrs, axis=0) if arrs else np.empty((0, D_MODEL))
            for dim, arrs in feats.items()
        }

        for idx in range(acts_batches.shape[0]):
            embeddings = {dim: emb_arrays[dim][idx] for dim in DIMS}
            encoded.append(EncodedWindow(
                anon_id=sid,
                label=info['label'],
                embeddings=embeddings,
                mid_hour=float(np.median(hrs_batches[idx])),
            ))
    return encoded


def compute_dim_stats(values: Iterable[np.ndarray]) -> Dict[str, np.ndarray]:
    arrs = [np.asarray(v, dtype=float) for v in values]
    if not arrs:
        return {
            'mean': np.zeros(D_MODEL, dtype=float),
            'std': np.ones(D_MODEL, dtype=float),
        }
    stacked = np.stack(arrs, axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return {
        'mean': stacked.mean(axis=0),
        'std': std,
    }


def compute_baseline(encoded_windows: List[EncodedWindow]) -> Dict[str, Dict[str, np.ndarray]]:
    storage = {dim: [] for dim in DIMS}
    for win in encoded_windows:
        for dim in DIMS:
            storage[dim].append(win.embeddings[dim])
    return {dim: compute_dim_stats(vals) for dim, vals in storage.items()}


def windows_by_subject(encoded_list: List[EncodedWindow]) -> Dict[int, List[EncodedWindow]]:
    per_subject: Dict[int, List[EncodedWindow]] = {}
    for win in encoded_list:
        per_subject.setdefault(win.anon_id, []).append(win)
    for sid, wins in per_subject.items():
        per_subject[sid] = sorted(wins, key=lambda w: w.mid_hour)
    return per_subject


def window_in_daytime(win: EncodedWindow) -> bool:
    if not DAYTIME_ONLY:
        return True
    return DAYTIME_START <= win.mid_hour <= DAYTIME_END


def aggregate_scores(values: Sequence[float]) -> float:
    if not values:
        return float('nan')
    arr = np.asarray(values, dtype=float)
    if SCORE_STATISTIC == 'median':
        return float(np.median(arr))
    return float(np.mean(arr))


def compute_subject_metrics(windows: List[EncodedWindow], baseline: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    if not windows:
        return {}
    per_dim_scores: Dict[str, List[float]] = {dim: [] for dim in DIMS}
    used_windows = 0
    for win in windows:
        if not window_in_daytime(win):
            continue
        used_windows += 1
        for dim in DIMS:
            mu = baseline[dim]['mean']
            sigma = baseline[dim]['std']
            emb = win.embeddings[dim]
            sigma_safe = np.where(np.abs(sigma) < 1e-6, 1.0, sigma)
            z_vec = (emb - mu) / sigma_safe
            if USE_ABS_Z:
                value = np.linalg.norm(z_vec)
            else:
                value = float(np.mean(z_vec))
            per_dim_scores[dim].append(float(value))

    if used_windows < MIN_SUBJECT_WINDOWS:
        return {}

    agg = {f'{dim.lower()}_score': aggregate_scores(vals) for dim, vals in per_dim_scores.items() if vals}
    if len(agg) != len(DIMS):
        return {}
    combined = float(np.dot(ALPHA, [agg[f'{dim.lower()}_score'] for dim in DIMS]))
    agg['combined_score'] = combined
    agg['n_windows'] = used_windows
    return agg


# ---------------------------------------------------------------------------
# Main experiment flow
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 80)
    print('EXPERIMENT 4 — MEDICAL CORRELATION (Clean Runner)')
    print('=' * 80)

    df = load_dataset(DATASET_PATH)
    if df.empty:
        raise RuntimeError('Dataset is empty or failed to load.')

    print(f'Loaded dataset with {len(df)} subjects.')

    subject_windows: Dict[int, Dict] = {}
    for row in df.itertuples(index=False):
        sid = int(getattr(row, 'anon_id'))
        label = getattr(row, 'label')
        if label not in {'CN', 'CI'}:
            continue
        if sid in DROP_SUBJECTS.get(label, set()):
            continue
        seqs = to_windows(getattr(row, 'sequence'), SEQ_LEN, STRIDE)
        seqs = sort_windows_by_time(seqs)
        if len(seqs) < MIN_WINDOWS:
            continue
        subject_windows[sid] = {
            'label': label,
            'windows': seqs,
        }

    if not subject_windows:
        raise RuntimeError('No subjects remain after window filtering.')

    baseline_subjects = {
        sid: info for sid, info in subject_windows.items()
        if sid in BASELINE_IDS and info['label'] == 'CN'
    }
    if len(baseline_subjects) != len(BASELINE_IDS):
        missing = sorted(set(BASELINE_IDS) - set(baseline_subjects.keys()))
        raise RuntimeError(f'Baseline CN subjects missing after filtering: {missing}')

    eval_subjects = {
        sid: info for sid, info in subject_windows.items()
        if sid not in BASELINE_IDS
    }

    print(
        f"Baseline CN subjects: {len(baseline_subjects)} | "
        f"Evaluation subjects: {len(eval_subjects)}"
    )

    model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES).to(DEVICE)
    model.eval()

    encoded_baseline = encode_windows(model, baseline_subjects)
    encoded_eval = encode_windows(model, eval_subjects)

    if not encoded_baseline:
        raise RuntimeError('Baseline windows empty; cannot compute statistics.')

    baseline_stats = compute_baseline(encoded_baseline)
    eval_windows = windows_by_subject(encoded_eval)

    subject_metrics: List[Dict[str, float]] = []
    for sid, wins in eval_windows.items():
        metrics = compute_subject_metrics(wins, baseline_stats)
        if not metrics:
            continue
        metrics['anon_id'] = sid
        metrics['label'] = wins[0].label if wins else subject_windows[sid]['label']
        subject_metrics.append(metrics)

    metrics_df = pd.DataFrame(subject_metrics)
    if metrics_df.empty:
        raise RuntimeError('No evaluation subjects produced valid metrics.')

    meta_df = load_subject_metadata(SUBJECT_METADATA_PATH)
    combined_df = metrics_df.merge(meta_df, on='anon_id', how='left', suffixes=('', '_meta'))

    combined_df.to_csv(SUBJECT_CSV, index=False)
    print(f'Wrote per-subject metrics to {os.path.relpath(SUBJECT_CSV, THIS_DIR)}')

    correlation_summary = {
        'baseline_cn_ids': BASELINE_IDS,
        'filters': {
            'min_windows_raw': MIN_WINDOWS,
            'min_windows_eval': MIN_SUBJECT_WINDOWS,
            'daytime_only': DAYTIME_ONLY,
            'daytime_range': DAYTIME_RANGE,
        },
        'alpha': ALPHA.tolist(),
        'medical_scores': {},
    }

    best_overall = None
    for score_name in MEDICAL_SCORES:
        if score_name not in combined_df.columns:
            continue
        feature_cols = [f'{dim.lower()}_score' for dim in DIMS] + ['combined_score']
        subset = combined_df[['anon_id', score_name] + feature_cols].copy()
        subset[score_name] = pd.to_numeric(subset[score_name], errors='coerce')
        for feature in feature_cols:
            subset[feature] = pd.to_numeric(subset[feature], errors='coerce')
        subset = subset.dropna()
        if len(subset) < max(MIN_SAMPLES, 3):
            continue

        corr_entries = {}
        for feature in feature_cols:
            values = subset[feature].to_numpy(dtype=float)
            target = subset[score_name].to_numpy(dtype=float)
            if values.size < 2 or np.allclose(values, values[0]):
                continue
            try:
                r_val, p_val = stats.pearsonr(values, target)
            except ValueError:
                continue
            corr_entries[feature] = {
                'r': float(r_val),
                'p': float(p_val),
            }

        if not corr_entries:
            continue

        best_feature = max(corr_entries.items(), key=lambda kv: abs(kv[1]['r']))
        feature_name, stats_dict = best_feature
        correlation_summary['medical_scores'][score_name] = {
            'n': int(len(subset)),
            'correlations': {
                feature: {'r': vals['r'], 'p': vals['p']}
                for feature, vals in corr_entries.items()
            },
            'best_feature': feature_name,
            'best_r': stats_dict['r'],
            'best_p': stats_dict['p'],
        }

        if (best_overall is None) or (abs(stats_dict['r']) > abs(best_overall['stats']['r'])):
            best_overall = {
                'score': score_name,
                'feature': feature_name,
                'stats': stats_dict,
                'data': subset[['anon_id', score_name, feature_name]].copy(),
                'correlations': corr_entries,
            }

    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(correlation_summary, f, indent=2)
    print(f'Correlation summary saved to {os.path.relpath(SUMMARY_JSON, THIS_DIR)}')

    if best_overall:
        plot_correlations(best_overall, FIG_PATH, FIG_PDF_PATH)
    else:
        print('No medical score met the minimum sample requirement for plotting.')

    print('\nTop correlation:')
    if best_overall:
        print(
            f"  Score: {best_overall['score']} | Feature: {best_overall['feature']} | "
            f"r = {best_overall['stats']['r']:.3f}, p = {best_overall['stats']['p']:.4f}"
        )
    else:
        print('  (none)')


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_correlations(best_info: Dict, png_path: str, pdf_path: str) -> None:
    score_name = best_info['score']
    feature = best_info['feature']
    stats_dict = best_info['stats']
    df = best_info['data'].rename(columns={feature: 'feature_value', score_name: 'score_value'})

    # Prepare bar data
    dim_labels = [f'{dim.lower()}_score' for dim in DIMS] + ['combined_score']
    bar_vals = []
    bar_labels = []
    bar_colors = ['#4A90E2', '#E89DAC', '#90D4A0', '#F4C96B', '#7E6BFF']
    for idx, dim in enumerate(dim_labels):
        entry = best_info['correlations'].get(dim)
        if entry:
            bar_vals.append(entry['r'])
            bar_labels.append(dim.replace('_score', '').title())
        else:
            bar_vals.append(0.0)
            bar_labels.append(dim.replace('_score', '').title())

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax0 = axes[0]
    bars = ax0.bar(range(len(bar_vals)), bar_vals, color=bar_colors[:len(bar_vals)], edgecolor='black', linewidth=1.2)
    ax0.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax0.set_xticks(range(len(bar_vals)))
    ax0.set_xticklabels(bar_labels, rotation=15)
    ax0.set_ylabel('Pearson r')
    ax0.set_title(f'{score_name.upper()} correlation per CTMS dimension (n={len(df)})')

    for idx, entry in enumerate(bar_vals):
        corr_entry = best_info['correlations'].get(dim_labels[idx])
        if corr_entry and corr_entry['p'] < 0.05:
            sig = '***' if corr_entry['p'] < 0.001 else ('**' if corr_entry['p'] < 0.01 else '*')
            y = entry + (0.03 if entry >= 0 else -0.03)
            ax0.text(idx, y, sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1 = axes[1]
    sns.regplot(
        data=df,
        x='feature_value',
        y='score_value',
        ax=ax1,
        scatter_kws={'s': 120, 'color': '#4A90E2', 'edgecolor': 'black'},
        line_kws={'color': 'red', 'linewidth': 2, 'alpha': 0.8},
    )
    ax1.set_xlabel(feature.replace('_score', '').title())
    ax1.set_ylabel(score_name.upper())
    sig = '***' if stats_dict['p'] < 0.001 else ('**' if stats_dict['p'] < 0.01 else ('*' if stats_dict['p'] < 0.05 else 'ns'))
    ax1.set_title(f"r = {stats_dict['r']:.3f}, p = {stats_dict['p']:.4f} ({sig})")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Figures saved to {os.path.relpath(png_path, THIS_DIR)} / {os.path.relpath(pdf_path, THIS_DIR)}')


if __name__ == '__main__':
    main()
