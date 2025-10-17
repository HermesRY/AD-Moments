#!/usr/bin/env python3
"""Experiment 3 (CTMS Classification) — clean publication runner.

This script evaluates the ability of CTMS-derived anomaly scores to separate
cognitively normal (CN) and cognitively impaired (CI) cohorts.

Pipeline overview:
1. Load the unified sample dataset (`sample_data/dataset_one_month.jsonl`).
2. Use a fixed subset of CN subjects (from `config.yaml`) to build global CTMS
   baselines (per-dimension encoder norms).
3. For every remaining CN and CI subject, derive directional CTMS anomaly
   scores. Optionally compute a personal baseline using the early portion of
   each subject's windows.
4. Aggregate subject-level scores and perform a threshold sweep to maximise
   F1-score, reporting sensitivity/specificity and the CI/CN ratio.

Outputs include publication-ready metrics, a subject-level CSV, and an optional
score distribution plot.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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

from Model.ctms_model import CTMSModel  # noqa: E402

CONFIG_PATH = os.environ.get('EXP3_CTMS_CONFIG', os.path.join(THIS_DIR, 'config.yaml'))
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.abspath(os.path.join(THIS_DIR, CFG['dataset_path']))
OUTPUT_DIR = os.path.join(THIS_DIR, CFG['outputs']['dir'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = int(CFG['ctms']['d_model'])
NUM_ACTIVITIES = int(CFG['num_activities'])
ALPHA = np.asarray(CFG['ctms'].get('alpha', [0.25, 0.25, 0.25, 0.25]), dtype=float)
ALPHA = ALPHA / ALPHA.sum()

WINDOW_CFG = CFG['window']
SEQ_LEN = int(WINDOW_CFG['seq_len'])
STRIDE = int(WINDOW_CFG['stride'])
BATCH = int(WINDOW_CFG['batch'])
if DEVICE.type == 'cuda':
    BATCH = int(WINDOW_CFG.get('batch_cuda', BATCH))

SCORING_CFG = CFG['scoring']
USE_ABS_Z = bool(SCORING_CFG.get('use_abs_z', False))
PERSONAL_FRAC = float(SCORING_CFG.get('personal_fraction', 0.0))
PERSONAL_MIN_WINDOWS = int(SCORING_CFG.get('personal_min_windows', 0))
MIN_WINDOWS_PER_SUBJECT = int(SCORING_CFG.get('min_windows_per_subject', 0))
CLASS_THRESHOLD_TYPE = SCORING_CFG.get('classification_feature', 'mean_score')
WINDOW_THRESHOLD = float(SCORING_CFG.get('window_threshold', 1.5))
DAYTIME_ONLY = bool(SCORING_CFG.get('daytime_only', False))
DAYTIME_RANGE = SCORING_CFG.get('daytime_hours', [0.0, 24.0])
if not isinstance(DAYTIME_RANGE, (list, tuple)) or len(DAYTIME_RANGE) != 2:
    raise ValueError('`daytime_hours` must be a two-element list [start, end].')
DAYTIME_START = float(DAYTIME_RANGE[0])
DAYTIME_END = float(DAYTIME_RANGE[1])

CLASS_CFG = CFG['classification']
THRESHOLD_PERCENTILES = CLASS_CFG.get('threshold_percentiles', list(range(30, 90, 5)))
SUMMARY_JSON = os.path.join(OUTPUT_DIR, CLASS_CFG.get('summary_json', 'exp3_classification_summary.json'))
PER_SUBJECT_CSV = os.path.join(OUTPUT_DIR, CLASS_CFG.get('per_subject_csv', 'exp3_classification_subject_metrics.csv'))
FIG_PATH = os.path.join(OUTPUT_DIR, CLASS_CFG.get('figure', 'exp3_score_distribution.png'))
THRESHOLD_CSV = os.path.join(OUTPUT_DIR, CLASS_CFG.get('threshold_metrics_csv', 'exp3_threshold_metrics.csv'))

FILTER_CFG = CFG['filter']
MIN_WINDOWS = int(FILTER_CFG.get('min_windows', 0))
DROP_SUBJECTS = {
    'CN': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CN', [])},
    'CI': {int(x) for x in FILTER_CFG.get('drop_subjects', {}).get('CI', [])},
}
MIN_CN_EVAL = int(FILTER_CFG.get('min_cn_eval', 0))
MIN_CI_EVAL = int(FILTER_CFG.get('min_ci_eval', 0))

TRAIN_IDS = [int(x) for x in CFG['split']['cn_train_ids']]

DIMS = ['Circadian', 'Task', 'Movement', 'Social']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_id(raw) -> int:
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Subject anon_id {raw!r} is not an integer") from exc


def load_dataset(path: str) -> pd.DataFrame:
    data: List[Dict] = []
    bad = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, 1):
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
        windows.append({
            'actions': acts_slice,
            'hours': hours_slice.astype(np.float32),
        })
    return windows


def sort_windows_by_time(windows: List[Dict]) -> List[Dict]:
    """Ensure windows follow temporal order based on their median timestamp."""
    if not windows:
        return windows
    indices = np.argsort([float(win['hours'][len(win['hours']) // 2]) for win in windows])
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
        acts_batches = []
        hrs_batches = []
        for win in windows:
            acts_batches.append(win['actions'])
            hrs_batches.append(win['hours'])
        acts_np = np.stack(acts_batches, axis=0)
        hrs_np = np.stack(hrs_batches, axis=0)

        feats = {dim: [] for dim in DIMS}
        with torch.no_grad():
            for i in range(0, acts_np.shape[0], BATCH):
                batch_actions = torch.from_numpy(acts_np[i:i + BATCH]).to(DEVICE)
                batch_hours = torch.from_numpy(hrs_np[i:i + BATCH]).to(DEVICE)
                enc = model(batch_actions, batch_hours, return_encodings_only=True)
                feats['Circadian'].append(enc['h_c'].detach().cpu().numpy())
                feats['Task'].append(enc['h_t'].detach().cpu().numpy())
                feats['Movement'].append(enc['h_m'].detach().cpu().numpy())
                feats['Social'].append(enc['h_s'].detach().cpu().numpy())

        emb_arrays = {dim: np.concatenate(arrs, axis=0) if arrs else np.empty((0, D_MODEL)) for dim, arrs in feats.items()}

        for idx in range(acts_np.shape[0]):
            embeddings = {dim: emb_arrays[dim][idx] for dim in DIMS}
            encoded.append(EncodedWindow(
                anon_id=sid,
                label=info['label'],
                embeddings=embeddings,
                mid_hour=float(np.median(hrs_np[idx])),
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


def compute_baseline(encoded_windows: List[EncodedWindow]) -> Dict[str, Dict[str, float]]:
    storage = {dim: [] for dim in DIMS}
    for win in encoded_windows:
        for dim in DIMS:
            storage[dim].append(win.embeddings[dim])
    return {dim: compute_dim_stats(vals) for dim, vals in storage.items()}


def subject_windows_from_encoded(encoded_list: List[EncodedWindow]) -> Dict[int, List[EncodedWindow]]:
    per_subject: Dict[int, List[EncodedWindow]] = {}
    for win in encoded_list:
        per_subject.setdefault(win.anon_id, []).append(win)
    return per_subject


def window_in_daytime(win: EncodedWindow) -> bool:
    if not DAYTIME_ONLY:
        return True
    return DAYTIME_START <= win.mid_hour <= DAYTIME_END


def derive_personal_stats(
    windows: List[EncodedWindow],
    fraction: float,
    min_windows: int,
    global_stats: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], int]:
    if fraction <= 0 or not windows:
        return global_stats, 0
    baseline_count = max(int(len(windows) * fraction), min_windows)
    baseline_count = min(baseline_count, len(windows))
    if baseline_count < min_windows:
        return global_stats, 0
    storage = {dim: [] for dim in DIMS}
    for win in windows[:baseline_count]:
        for dim in DIMS:
            storage[dim].append(win.embeddings[dim])
    stats = {dim: compute_dim_stats(vals) for dim, vals in storage.items()}
    return stats, baseline_count


def score_windows(windows: List[EncodedWindow], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    scores = []
    for win in windows:
        zs = []
        for dim in DIMS:
            mu = stats[dim]['mean']
            sigma = stats[dim]['std']
            emb = win.embeddings[dim]
            diff = emb - mu
            numerator = np.linalg.norm(diff)
            denom = np.linalg.norm(sigma) + 1e-8
            z = numerator / denom if denom > 0 else 0.0
            if USE_ABS_Z:
                z = abs(z)
            zs.append(z)
        combined = float(np.sum(ALPHA * np.asarray(zs, dtype=float)))
        scores.append(combined)
    return np.asarray(scores, dtype=float)


def threshold_sweep(values: np.ndarray, labels: np.ndarray, percentiles: List[int]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    best: Dict[str, float] | None = None
    history: List[Dict[str, float]] = []
    unique_percentiles = sorted({p for p in percentiles if 0 < p < 100})
    for perc in unique_percentiles:
        thresh = float(np.percentile(values, perc))
        preds = (values >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
        entry = {
            'percentile': perc,
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
        history.append(entry)
        if best is None or f1 > best['f1']:
            best = entry
    if best is None:
        raise RuntimeError('Threshold sweep failed to evaluate any candidate thresholds.')
    return best, history


# ---------------------------------------------------------------------------
# Load dataset and construct windows
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
    seqs = sort_windows_by_time(seqs)
    subject_windows[sid] = {
        'label': label,
        'windows': seqs,
    }

subject_windows = {
    sid: info for sid, info in subject_windows.items()
    if len(info['windows']) >= MIN_WINDOWS
}

if not set(TRAIN_IDS).issubset(subject_windows.keys()):
    missing = sorted(set(TRAIN_IDS) - subject_windows.keys())
    raise RuntimeError(f"Training CN IDs {missing} missing after filtering; adjust config or filters.")

train_subjects = {sid: subject_windows[sid] for sid in TRAIN_IDS}

eval_subjects = {
    sid: info for sid, info in subject_windows.items()
    if not (info['label'] == 'CN' and sid in TRAIN_IDS)
}

cn_eval = sum(1 for info in eval_subjects.values() if info['label'] == 'CN')
ci_eval = sum(1 for info in eval_subjects.values() if info['label'] == 'CI')
if cn_eval < MIN_CN_EVAL or ci_eval < MIN_CI_EVAL:
    raise RuntimeError(f"Not enough evaluation subjects — CN: {cn_eval}, CI: {ci_eval}")

print(
    f"Loaded dataset: {len(df)} subjects -> train CN: {len(train_subjects)}, "
    f"eval CN: {cn_eval}, eval CI: {ci_eval}"
)

# ---------------------------------------------------------------------------
# Encode windows with CTMS and compute baselines
# ---------------------------------------------------------------------------

model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES).to(DEVICE)
model.eval()

encoded_train = encode_windows(model, train_subjects)
encoded_eval = encode_windows(model, eval_subjects)

if not encoded_train:
    raise RuntimeError('Training CN windows empty; cannot compute baseline statistics.')

global_baseline = compute_baseline(encoded_train)

eval_windows_by_subject = subject_windows_from_encoded(encoded_eval)

# ---------------------------------------------------------------------------
# Score subjects
# ---------------------------------------------------------------------------

subject_records = []

for sid, info in eval_subjects.items():
    label = info['label']
    windows = eval_windows_by_subject.get(sid, [])
    if len(windows) < MIN_WINDOWS_PER_SUBJECT:
        continue

    stats, baseline_windows = derive_personal_stats(
        windows,
        PERSONAL_FRAC,
        PERSONAL_MIN_WINDOWS,
        global_baseline,
    )
    baseline_type = 'personal' if baseline_windows >= PERSONAL_MIN_WINDOWS else 'global'
    eval_candidates = windows[baseline_windows:] if baseline_type == 'personal' else windows
    eval_windows = [win for win in eval_candidates if window_in_daytime(win)]

    if len(eval_windows) < max(MIN_WINDOWS_PER_SUBJECT, 1):
        stats = global_baseline
        baseline_type = 'global'
        baseline_windows = 0
        eval_candidates = windows
        eval_windows = [win for win in eval_candidates if window_in_daytime(win)]

    if len(eval_windows) < max(MIN_WINDOWS_PER_SUBJECT, 1):
        continue

    scores = score_windows(eval_windows, stats)
    if scores.size == 0:
        continue

    anomaly_flags = scores >= WINDOW_THRESHOLD
    anomaly_count = int(anomaly_flags.sum())
    anomaly_rate = float(anomaly_count / scores.size) if scores.size else 0.0

    subject_records.append({
        'anon_id': int(sid),
        'label': label,
        'baseline_type': baseline_type,
        'baseline_windows': baseline_windows,
        'eval_windows': len(eval_windows),
        'total_windows': len(windows),
        'mean_score': float(np.mean(scores)),
        'median_score': float(np.median(scores)),
        'std_score': float(np.std(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores)),
        'anomaly_count': anomaly_count,
        'anomaly_rate': anomaly_rate,
    })

if not subject_records:
    raise RuntimeError('No evaluation subjects satisfied window requirements.')

# ---------------------------------------------------------------------------
# Classification threshold search
# ---------------------------------------------------------------------------

supported_features = {'mean_score', 'median_score', 'anomaly_count', 'anomaly_rate'}
if CLASS_THRESHOLD_TYPE not in supported_features:
    raise ValueError(f"Unsupported classification feature: {CLASS_THRESHOLD_TYPE}")


def extract_feature(record: Dict[str, float]) -> float:
    if CLASS_THRESHOLD_TYPE == 'mean_score':
        return record['mean_score']
    if CLASS_THRESHOLD_TYPE == 'median_score':
        return record['median_score']
    if CLASS_THRESHOLD_TYPE == 'anomaly_count':
        return float(record['anomaly_count'])
    if CLASS_THRESHOLD_TYPE == 'anomaly_rate':
        return record['anomaly_rate']
    raise ValueError(f"Unsupported classification feature: {CLASS_THRESHOLD_TYPE}")


score_feature = np.array([extract_feature(rec) for rec in subject_records], dtype=float)
for rec, feat in zip(subject_records, score_feature):
    rec['score_feature'] = float(feat)
labels = np.array([1 if rec['label'] == 'CI' else 0 for rec in subject_records], dtype=int)

best, threshold_history = threshold_sweep(score_feature, labels, THRESHOLD_PERCENTILES)

preds = (score_feature >= best['threshold']).astype(int)

cn_scores = score_feature[labels == 0]
ci_scores = score_feature[labels == 1]
ci_cn_ratio = (
    float(ci_scores.mean() / cn_scores.mean())
    if cn_scores.size and cn_scores.mean() > 0
    else None
)

summary = {
    'classification_feature': CLASS_THRESHOLD_TYPE,
    'anomaly_threshold': WINDOW_THRESHOLD,
    'threshold_percentile': best['percentile'],
    'threshold_value': best['threshold'],
    'precision': best['precision'],
    'recall': best['recall'],
    'f1': best['f1'],
    'specificity': best['specificity'],
    'accuracy': best['accuracy'],
    'sensitivity': best['recall'],
    'confusion_matrix': {'tn': best['tn'], 'fp': best['fp'], 'fn': best['fn'], 'tp': best['tp']},
    'ci_feature_mean': float(ci_scores.mean()) if ci_scores.size else None,
    'ci_feature_std': float(ci_scores.std()) if ci_scores.size else None,
    'cn_feature_mean': float(cn_scores.mean()) if cn_scores.size else None,
    'cn_feature_std': float(cn_scores.std()) if cn_scores.size else None,
    'ci_cn_ratio': ci_cn_ratio,
    'train_cn_subjects': sorted(int(x) for x in TRAIN_IDS),
    'eval_cn_subjects': sorted(int(rec['anon_id']) for rec in subject_records if rec['label'] == 'CN'),
    'eval_ci_subjects': sorted(int(rec['anon_id']) for rec in subject_records if rec['label'] == 'CI'),
}

with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

threshold_df = pd.DataFrame(threshold_history).sort_values('percentile')
threshold_df.to_csv(THRESHOLD_CSV, index=False)

# ---------------------------------------------------------------------------
# Persist subject-level metrics
# ---------------------------------------------------------------------------

subject_df = pd.DataFrame(subject_records)
subject_df['prediction'] = preds
subject_df['score_feature'] = score_feature
subject_df.to_csv(PER_SUBJECT_CSV, index=False)

# ---------------------------------------------------------------------------
# Optional visualization
# ---------------------------------------------------------------------------

if CLASS_THRESHOLD_TYPE == 'mean_score':
    feature_label = 'Mean CTMS anomaly score'
elif CLASS_THRESHOLD_TYPE == 'median_score':
    feature_label = 'Median CTMS anomaly score'
elif CLASS_THRESHOLD_TYPE == 'anomaly_count':
    feature_label = 'Anomalous window count'
elif CLASS_THRESHOLD_TYPE == 'anomaly_rate':
    feature_label = 'Anomalous window rate'
else:
    feature_label = 'CTMS feature'

plt.figure(figsize=(10, 6))
positions = [0, 1]
colors = ['#2E7D32', '#C0392B']
plt.boxplot(
    [cn_scores, ci_scores],
    positions=positions,
    widths=0.5,
    patch_artist=True,
    boxprops={'facecolor': '#F5F5F5', 'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 2},
    whiskerprops={'linewidth': 1.2},
)
plt.scatter(
    np.zeros_like(cn_scores) + positions[0],
    cn_scores,
    color=colors[0],
    alpha=0.7,
    label='CN',
    s=60,
)
plt.scatter(
    np.zeros_like(ci_scores) + positions[1],
    ci_scores,
    color=colors[1],
    alpha=0.7,
    label='CI',
    s=60,
)
plt.axhline(
    best['threshold'],
    color='black',
    linestyle='--',
    linewidth=2,
    label=f"Threshold ({best['threshold']:.2f})",
)
plt.xticks(positions, ['CN', 'CI'])
plt.ylabel(feature_label, fontweight='bold')
plt.title('Subject-level CTMS Scores', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print('\n=== Experiment 3 (CTMS Classification) Summary ===')
print(json.dumps(summary, indent=2))
print(f"Per-subject metrics: {PER_SUBJECT_CSV}")
print(f"Summary JSON: {SUMMARY_JSON}")
print(f"Score distribution figure: {FIG_PATH}")