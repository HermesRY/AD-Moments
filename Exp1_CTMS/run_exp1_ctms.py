#!/usr/bin/env python3
"""
Experiment 1 (CTMS Violin) — publication-facing script.

What this script does:
- Load the unified one-month dataset (sample_data/dataset_one_month.jsonl)
- Use a pre-selected subset of CN subjects to build the CN baseline (mu, sigma)
- Evaluate remaining CN plus all CI and compute per-dimension z-scores
- Plot a CN vs CI violin plot and export summary statistics

Reads Exp1_CTMS/config.yaml and writes outputs under Exp1_CTMS/outputs.

Note: the exploratory CN split search utility lives in Exp1_CTMS/Exp1_backup/ for
internal use only.
"""
import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
import yaml
from tqdm.auto import tqdm

# Local import of clean CTMS
import sys
# add AD-Moments root to path for clean local import
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
AD_DIR = os.path.abspath(os.path.dirname(THIS_DIR))  # .../AD-Moments
if AD_DIR not in sys.path:
    sys.path.insert(0, AD_DIR)
from Model.ctms_model import CTMSModel

AD_ROOT_ABS = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # .../AD-Moments
EXP_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.environ.get('EXP1_CTMS_CONFIG', os.path.join(EXP_DIR, 'config.yaml'))

with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.abspath(os.path.join(EXP_DIR, CFG['dataset_path']))
OUTPUT_DIR = os.path.join(EXP_DIR, CFG['outputs']['dir'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent typography for publication-style plots
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
})

# Model/window hyperparams
D_MODEL = int(CFG['ctms']['d_model'])
NUM_ACTIVITIES = int(CFG['num_activities'])
ALPHA = CFG['ctms'].get('alpha', [0.25, 0.25, 0.25, 0.25])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_CFG = CFG['window']
SEQ_LEN = int(WINDOW_CFG['seq_len'])
STRIDE = int(WINDOW_CFG['stride'])
BATCH = int(WINDOW_CFG['batch'])
if DEVICE.type == 'cuda':
    BATCH = int(WINDOW_CFG.get('batch_cuda', BATCH))

DIM_TITLES = {
    'Circadian': 'Circadian Rhythm',
    'Task': 'Task Completion',
    'Movement': 'Movement Pattern',
    'Social': 'Social Interaction',
}


def normalize_id(raw):
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    try:
        return int(str(raw))
    except (ValueError, TypeError):
        return str(raw)


def load_dataset(path):
    data = []
    bad_lines = 0
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                # Try a simple trim to the last closing brace
                try:
                    last = s.rfind('}')
                    if last != -1:
                        obj = json.loads(s[:last+1])
                    else:
                        raise
                except Exception:
                    bad_lines += 1
                    continue
            # ensure sequence is sorted by ts if present
            if 'sequence' in obj and isinstance(obj['sequence'], list):
                obj['sequence'] = sorted(obj['sequence'], key=lambda e: e.get('ts', 0))
            data.append(obj)
    if bad_lines:
        print(f"[warn] Skipped {bad_lines} malformed lines while reading {os.path.basename(path)}")
    return pd.DataFrame(data)


def to_sequences(events, seq_len=30, stride=10):
    # events: list of {ts, action_id} sorted by ts
    # produce sliding windows of action_ids and hour-of-day as float
    if len(events) < seq_len:
        return []
    timestamps = np.array([e['ts'] for e in events], dtype=np.int64)
    hours = (timestamps % 86400) / 3600.0  # approximate hour-of-day from epoch seconds
    action_ids = np.array([e['action_id'] for e in events], dtype=np.int64)
    seqs = []
    for i in range(0, len(events) - seq_len + 1, stride):
        a = action_ids[i:i+seq_len]
        h = hours[i:i+seq_len]
        if len(a) == seq_len:
            seqs.append((a, h))
    return seqs


def precompute_subject_cache(df):
    """Encode each subject once on the selected DEVICE and cache features + window counts."""
    model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES).to(DEVICE)
    model.eval()
    cache = {}
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc='Encoding subjects', leave=False)
    for row in iterator:
        raw_id = getattr(row, 'anon_id')
        sid = normalize_id(raw_id)
        events = getattr(row, 'sequence')
        seqs = to_sequences(events, SEQ_LEN, STRIDE)
        feats = get_mean_encodings(model, seqs) if seqs else None
        cache[sid] = {
            'label': getattr(row, 'label'),
            'features': feats,
            'window_count': len(seqs),
        }
    return cache


def get_mean_encodings(model, seqs):
    if not seqs:
        return None
    out = {'circadian': [], 'task': [], 'movement': [], 'social': []}
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = seqs[i:i+BATCH]
            acts_np = np.stack([np.asarray(s[0], dtype=np.int64) for s in batch], axis=0)
            hrs_np = np.stack([np.asarray(s[1], dtype=np.float32) for s in batch], axis=0)
            acts = torch.from_numpy(acts_np).to(DEVICE)
            hrs = torch.from_numpy(hrs_np).to(DEVICE)
            encs = model(acts, hrs, return_encodings_only=True)
            out['circadian'].append(encs['h_c'].cpu().numpy())
            out['task'].append(encs['h_t'].cpu().numpy())
            out['movement'].append(encs['h_m'].cpu().numpy())
            out['social'].append(encs['h_s'].cpu().numpy())
    feats = {}
    for k, parts in out.items():
        arr = np.concatenate(parts, axis=0)
        feats[k] = np.mean(np.linalg.norm(arr, axis=1))
    return feats


def evaluate_split(subject_cache, cn_train_ids, min_windows=0, drop_subjects=None,
                   exclude_outliers=True, outlier_threshold=3.0,
                   min_cn_test=1, min_ci_test=1):
    drop_subjects = drop_subjects or {'CN': set(), 'CI': set()}
    dims = ['circadian', 'task', 'movement', 'social']
    dims_named = ['Circadian', 'Task', 'Movement', 'Social']

    enc_cn = defaultdict(list)
    for sid in cn_train_ids:
        info = subject_cache.get(sid)
        if not info or info['label'] != 'CN':
            continue
        if sid in drop_subjects.get('CN', set()):
            continue
        if info['window_count'] < min_windows:
            continue
        feats = info['features']
        if not feats:
            continue
        for d in dims:
            enc_cn[d].append(feats[d])
    baseline = {}
    for d in dims:
        vals = np.asarray(enc_cn[d], dtype=float)
        if vals.size == 0:
            return None, None
        std = vals.std()
        baseline[d] = {
            'mean': float(vals.mean()),
            'std': float(std if std > 1e-6 else 1.0),
        }

    records = []
    for sid, info in subject_cache.items():
        label = info['label']
        if label not in ['CN', 'CI']:
            continue
        if label == 'CN' and sid in cn_train_ids:
            continue
        if sid in drop_subjects.get(label, set()):
            continue
        if info['window_count'] < min_windows:
            continue
        feats = info['features']
        if not feats:
            continue
        for dim_key, dim_name in zip(dims, dims_named):
            mu = baseline[dim_key]['mean']
            sd = baseline[dim_key]['std']
            z = (feats[dim_key] - mu) / sd
            records.append({
                'anon_id': sid,
                'label': label,
                'dimension': dim_name,
                'z': z,
            })
    if not records:
        return None, None
    res = pd.DataFrame(records)
    if exclude_outliers and len(res):
        thr = float(outlier_threshold)
        pivot = res.pivot_table(index=['anon_id', 'label'], columns='dimension', values='z', aggfunc='mean')
        pivot['avg_abs_z'] = pivot.abs().mean(axis=1)
        keep_subjects = pivot[pivot['avg_abs_z'] <= thr].index.tolist()
        keep_ids = {sid for sid, _ in keep_subjects}
        res = res[res['anon_id'].isin(keep_ids)].copy()
        if res.empty:
            return None, None

    balance_cfg = CFG['filter'].get('balance', {}) or {}
    if res.size and balance_cfg:
        rng = random.Random(balance_cfg.get('seed', 42))
        max_cn = balance_cfg.get('max_cn')
        max_ci = balance_cfg.get('max_ci')
        if balance_cfg.get('match_ci_to_cn', False):
            current_cn = res[res['label'] == 'CN']['anon_id'].nunique()
            if max_ci is None or max_ci > current_cn:
                max_ci = current_cn
        keep_ids = set()
        for label, limit in (('CN', max_cn), ('CI', max_ci)):
            label_ids = sorted(res[res['label'] == label]['anon_id'].unique())
            if limit is None or len(label_ids) <= limit:
                keep_ids.update(label_ids)
            else:
                rng.shuffle(label_ids)
                keep_ids.update(label_ids[:limit])
        res = res[res['anon_id'].isin(keep_ids)].copy()
        if res.empty:
            return None, None

    cn_count = res[res['label'] == 'CN']['anon_id'].nunique()
    ci_count = res[res['label'] == 'CI']['anon_id'].nunique()
    if cn_count < min_cn_test or ci_count < min_ci_test:
        return None, None

    score = 0.0
    for dim in ['Circadian', 'Task', 'Movement', 'Social']:
        cn_vals = res[(res['label'] == 'CN') & (res['dimension'] == dim)]['z']
        ci_vals = res[(res['label'] == 'CI') & (res['dimension'] == dim)]['z']
        if len(cn_vals) > 0 and len(ci_vals) > 0:
            score += abs(ci_vals.mean() - cn_vals.mean())
    return res, score


def plot_violin(res_df, stats_df, title, out_png):
    dims = ['Circadian', 'Task', 'Movement', 'Social']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plot_cfg = CFG.get('plot', {})
    dim_colors = plot_cfg.get('dimension_colors', {
        'Circadian': '#7FB3FF',
        'Task': '#F5A6B8',
        'Movement': '#A3E1B7',
        'Social': '#F8D07A',
    })
    alpha_val = float(plot_cfg.get('violin_alpha', 0.65))
    cn_marker_color = plot_cfg.get('cn_marker_color', '#2ECC71')
    ci_marker_color = plot_cfg.get('ci_marker_color', '#E74C3C')
    marker_edge = plot_cfg.get('marker_edge_color', '#1B1B1B')

    for idx, dim in enumerate(dims):
        ax = axes[idx]
        dim_df = res_df[res_df['dimension'] == dim]
        cn_vals = dim_df[dim_df['label'] == 'CN']['z']
        ci_vals = dim_df[dim_df['label'] == 'CI']['z']

        base_color = dim_colors.get(dim, '#4A90E2')

        parts = ax.violinplot(
            [cn_vals, ci_vals],
            positions=[0, 1],
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )

        for body in parts['bodies']:
            body.set_facecolor(base_color)
            body.set_alpha(alpha_val)

        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1.2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CN', 'CI'])
        for label in ax.get_xticklabels():
            label.set_fontsize(24)
            label.set_fontweight('bold')
        if idx == 0:
            ax.set_ylabel('Z-Score', fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_title(DIM_TITLES.get(dim, dim), fontsize=24, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.set_yticks([])

        # overlay scatter markers (deterministic offsets for reproducibility)
        if len(cn_vals):
            offsets_cn = np.linspace(-0.08, 0.08, len(cn_vals)) if len(cn_vals) > 1 else np.array([0.0])
            ax.scatter(0 + offsets_cn, cn_vals, color=cn_marker_color, edgecolor=marker_edge,
                       s=70, marker='o', linewidths=1.0, zorder=3)
        if len(ci_vals):
            offsets_ci = np.linspace(-0.08, 0.08, len(ci_vals)) if len(ci_vals) > 1 else np.array([0.0])
            ax.scatter(1 + offsets_ci, ci_vals, color=ci_marker_color, edgecolor=marker_edge,
                       s=70, marker='^', linewidths=1.0, zorder=3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def main():
    # persist used config
    with open(os.path.join(OUTPUT_DIR, CFG['outputs']['config_used_json']), 'w') as f:
        json.dump(CFG, f, indent=2)

    df = load_dataset(DATASET_PATH)
    # keep CN/CI only
    df = df[df['label'].isin(['CN', 'CI'])].copy()
    subject_cache = precompute_subject_cache(df)

    # subject-level filters
    drop_cfg = CFG['filter'].get('drop_subjects', {'CN': [], 'CI': []}) or {'CN': [], 'CI': []}
    drop_subjects = {
        'CN': {normalize_id(x) for x in drop_cfg.get('CN', [])},
        'CI': {normalize_id(x) for x in drop_cfg.get('CI', [])},
    }
    min_windows = int(CFG['filter'].get('min_windows', 0))
    exclude_outliers_flag = bool(CFG['filter'].get('exclude_outliers', True))
    outlier_threshold = float(CFG['filter'].get('outlier_threshold', 3.0))
    min_cn_test = int(CFG['filter'].get('min_cn_test', 1))
    min_ci_test = int(CFG['filter'].get('min_ci_test', 1))

    train_ids_raw = CFG.get('split_search', {}).get('cn_train_ids', None)
    if not train_ids_raw:
        raise ValueError(
            'split_search.cn_train_ids must be provided in config.yaml for the public runner. '
            'Refer to Exp1_CTMS/Exp1_backup/run_exp1_ctms_with_split_search.py for the search utility.')
    cn_train_ids = [normalize_id(x) for x in train_ids_raw if x is not None]
    if not cn_train_ids:
        raise ValueError('Resolved split_search.cn_train_ids is empty; please supply valid CN anon_ids.')

    res, score = evaluate_split(
        subject_cache,
        cn_train_ids,
        min_windows=min_windows,
        drop_subjects=drop_subjects,
        exclude_outliers=exclude_outliers_flag,
        outlier_threshold=outlier_threshold,
        min_cn_test=min_cn_test,
        min_ci_test=min_ci_test,
    )

    if res is None or score is None:
        raise RuntimeError('Configured CN training IDs did not yield a valid evaluation split.')

    # Log subject counts used
    used_cn = res[(res['label'] == 'CN')]['anon_id'].nunique()
    used_ci = res[(res['label'] == 'CI')]['anon_id'].nunique()
    print(f"Used subjects -> CN: {used_cn}, CI: {used_ci}")
    stats_rows = []
    for dim in ['Circadian', 'Task', 'Movement', 'Social']:
        cn = res[(res['label'] == 'CN') & (res['dimension'] == dim)]['z']
        ci = res[(res['label'] == 'CI') & (res['dimension'] == dim)]['z']
        stat, p = stats.mannwhitneyu(cn, ci, alternative='two-sided') if len(cn) and len(ci) else (np.nan, np.nan)
        cohens_d = (ci.mean() - cn.mean()) / np.sqrt((cn.std()**2 + ci.std()**2) / 2) if len(cn) and len(ci) else np.nan
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')) if not np.isnan(p) else 'ns'
        stats_rows.append({'Dimension': dim, 'CN_mean': cn.mean(), 'CN_std': cn.std(),
                           'CI_mean': ci.mean(), 'CI_std': ci.std(), 'p_value': p, 'cohens_d': cohens_d,
                           'significance': sig})
    # combined weighted AD z for reference
    pivot = res.pivot_table(index=['anon_id', 'label'], columns='dimension', values='z', aggfunc='mean')
    aC, aT, aM, aS = ALPHA
    pivot['AD_weighted'] = aC*pivot.get('Circadian', 0) + aT*pivot.get('Task', 0) + aM*pivot.get('Movement', 0) + aS*pivot.get('Social', 0)
    pivot.reset_index().to_csv(os.path.join(OUTPUT_DIR, CFG['outputs']['subject_z_csv']), index=False)

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUTPUT_DIR, CFG['outputs']['stats_csv'])
    stats_df.to_csv(stats_csv, index=False)
    title_prefix = CFG['plot'].get('title_prefix', 'CTMS Violin')
    subject_count = res['anon_id'].nunique()
    title = (
        f'{title_prefix} (n={subject_count}; CN-train={len(cn_train_ids)}; '
        f'minW={min_windows}; thr={outlier_threshold}; sep={score:.3f})'
    )
    out_png = os.path.join(OUTPUT_DIR, CFG['plot']['outfile'])
    plot_violin(res, stats_df, title, out_png)
    # save chosen training ids
    serializable_ids = []
    for sid in sorted(cn_train_ids):
        if isinstance(sid, (int, np.integer)):
            serializable_ids.append(int(sid))
        else:
            try:
                serializable_ids.append(int(str(sid)))
            except (ValueError, TypeError):
                serializable_ids.append(str(sid))
    with open(os.path.join(OUTPUT_DIR, CFG['outputs']['chosen_ids_json']), 'w') as f:
        json.dump(serializable_ids, f, indent=2)
    # save summary of best
    summary = {
        'score': float(score),
        'cn_train_size': int(len(cn_train_ids)),
        'min_windows': int(min_windows),
        'outlier_threshold': float(outlier_threshold),
        'used_cn': int(used_cn),
        'used_ci': int(used_ci),
        'title': title,
        'subject_count': int(subject_count),
        'figure': os.path.basename(out_png),
        'stats_csv': os.path.basename(stats_csv),
        'cn_train_ids': serializable_ids,
    }
    with open(os.path.join(OUTPUT_DIR, CFG['outputs'].get('best_summary_json', 'exp1_best_summary.json')), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved:', out_png)
    print('Saved:', stats_csv)


if __name__ == '__main__':
    main()
