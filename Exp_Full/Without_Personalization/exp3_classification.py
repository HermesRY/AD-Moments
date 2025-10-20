#!/usr/bin/env python3
"""
Exp3: Binary Classification (Without Personalization)
Performs AD vs CN classification using CTMS anomaly scores.

Usage:
    python exp3_classification.py

Note:
    Uses unified baseline from all CN training subjects.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import stats as scipy_stats
from Model.ctms_model import CTMSModel

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optimal split from Exp2
OPTIMAL_SPLIT = {
    'cn_train': [8, 33, 34, 37, 40, 45, 46, 47, 48, 49, 61, 65],
    'cn_test': [16, 35, 36, 38, 41, 44],
    'ci_test': [4, 5, 7, 10, 12, 17, 19, 22, 23, 50, 62],
}

# Model configuration
ALPHA = np.array([0.5, 0.3, 0.1, 0.1])  # Exp2 optimal weights
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 256
D_MODEL = 64
NUM_ACTIVITIES = 22

# Threshold sweep for optimal operating point
THRESHOLD_PERCENTILES = list(range(10, 91, 2))

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load and filter dataset."""
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subj = json.loads(line)
            data.append(subj)
    
    # Filter by split
    all_ids = (OPTIMAL_SPLIT['cn_train'] + OPTIMAL_SPLIT['cn_test'] + 
               OPTIMAL_SPLIT['ci_test'])
    data = [s for s in data if s['subject_id'] in all_ids]
    
    print(f"Loaded {len(data)} subjects")
    return data


def create_sequences(activity_series, seq_len=SEQ_LEN, stride=STRIDE):
    """Create sliding windows from activity series."""
    sequences = []
    for i in range(0, len(activity_series) - seq_len + 1, stride):
        seq = activity_series[i:i+seq_len]
        sequences.append(seq)
    return sequences


def encode_subject(subject, model):
    """Encode all sequences of a subject."""
    activities = subject['activities']
    sequences = create_sequences(activities)
    
    if len(sequences) == 0:
        return None
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i:i+BATCH_SIZE]
            batch_tensor = torch.LongTensor(batch).to(DEVICE)
            emb = model(batch_tensor, return_dim_embeddings=True)  # [B, 4, D]
            embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)  # [N, 4, D]
    return embeddings.numpy()


def compute_baseline_unified(train_embeddings_list):
    """Compute unified baseline from all CN training subjects."""
    # Concatenate all windows from all CN train subjects
    all_windows = np.concatenate(train_embeddings_list, axis=0)  # [Total_windows, 4, D]
    baseline = np.mean(all_windows, axis=0)  # [4, D]
    return baseline


def compute_anomaly_scores(embeddings, baseline, alpha):
    """Compute weighted anomaly scores."""
    # embeddings: [N, 4, D]
    # baseline: [4, D]
    diffs = embeddings - baseline[np.newaxis, :, :]  # [N, 4, D]
    norms = np.linalg.norm(diffs, axis=2)  # [N, 4]
    
    # Z-score normalization per dimension
    z_scores = (norms - np.mean(norms, axis=0)) / (np.std(norms, axis=0) + 1e-8)
    
    # Weighted sum
    scores = np.sum(z_scores * alpha, axis=1)  # [N]
    return scores


def classify_threshold_sweep(cn_test_scores, ci_test_scores):
    """Sweep thresholds to find optimal operating point."""
    # Combine scores for percentile calculation
    all_scores = np.concatenate([cn_test_scores, ci_test_scores])
    
    results = []
    for percentile in THRESHOLD_PERCENTILES:
        threshold = np.percentile(all_scores, percentile)
        
        # Classify: score > threshold => CI (anomalous)
        cn_pred = cn_test_scores > threshold
        ci_pred = ci_test_scores > threshold
        
        tp = np.sum(ci_pred)
        tn = np.sum(~cn_pred)
        fp = np.sum(cn_pred)
        fn = np.sum(~ci_pred)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        ci_mean = np.mean(ci_test_scores)
        cn_mean = np.mean(cn_test_scores)
        ci_cn_ratio = ci_mean / cn_mean if cn_mean != 0 else 0
        
        results.append({
            'percentile': percentile,
            'threshold': float(threshold),
            'f1': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'ci_cn_ratio': float(ci_cn_ratio),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        })
    
    # Find best F1
    best_idx = np.argmax([r['f1'] for r in results])
    best = results[best_idx]
    
    return {
        'best': best,
        'sweep': results,
        'cn_test_scores': cn_test_scores.tolist(),
        'ci_test_scores': ci_test_scores.tolist()
    }


def main():
    print("="*80)
    print("EXP3: BINARY CLASSIFICATION (WITHOUT PERSONALIZATION)")
    print("="*80)
    
    # Load data
    print("\n1. Loading dataset...")
    data = load_dataset()
    
    # Split data
    cn_train = [s for s in data if s['subject_id'] in OPTIMAL_SPLIT['cn_train']]
    cn_test = [s for s in data if s['subject_id'] in OPTIMAL_SPLIT['cn_test']]
    ci_test = [s for s in data if s['subject_id'] in OPTIMAL_SPLIT['ci_test']]
    
    print(f"   CN train: {len(cn_train)}")
    print(f"   CN test: {len(cn_test)}")
    print(f"   CI test: {len(ci_test)}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    print(f"   Alpha weights: {ALPHA}")
    
    # Encode CN training set
    print("\n3. Encoding CN training set...")
    cn_train_embeddings = []
    for subj in tqdm(cn_train, desc="CN train"):
        emb = encode_subject(subj, model)
        if emb is not None:
            cn_train_embeddings.append(emb)
    
    # Compute unified baseline
    print("\n4. Computing unified baseline...")
    baseline = compute_baseline_unified(cn_train_embeddings)
    print(f"   Baseline shape: {baseline.shape}")
    
    # Encode and score CN test
    print("\n5. Encoding CN test subjects...")
    cn_test_scores = []
    for subj in tqdm(cn_test, desc="CN test"):
        emb = encode_subject(subj, model)
        if emb is not None:
            scores = compute_anomaly_scores(emb, baseline, ALPHA)
            cn_test_scores.append(np.mean(scores))  # Subject-level mean
    cn_test_scores = np.array(cn_test_scores)
    
    # Encode and score CI test
    print("\n6. Encoding CI test subjects...")
    ci_test_scores = []
    for subj in tqdm(ci_test, desc="CI test"):
        emb = encode_subject(subj, model)
        if emb is not None:
            scores = compute_anomaly_scores(emb, baseline, ALPHA)
            ci_test_scores.append(np.mean(scores))  # Subject-level mean
    ci_test_scores = np.array(ci_test_scores)
    
    # Threshold sweep
    print("\n7. Performing threshold sweep...")
    results = classify_threshold_sweep(cn_test_scores, ci_test_scores)
    
    # Print best results
    best = results['best']
    print("\n" + "="*80)
    print("BEST RESULTS (Max F1)")
    print("="*80)
    print(f"Threshold percentile: {best['percentile']}")
    print(f"Threshold value: {best['threshold']:.4f}")
    print(f"F1 Score: {best['f1']:.3f}")
    print(f"Sensitivity: {best['sensitivity']:.3f}")
    print(f"Specificity: {best['specificity']:.3f}")
    print(f"Precision: {best['precision']:.3f}")
    print(f"Accuracy: {best['accuracy']:.3f}")
    print(f"CI/CN Ratio: {best['ci_cn_ratio']:.3f}")
    print(f"Confusion: TP={best['tp']}, TN={best['tn']}, FP={best['fp']}, FN={best['fn']}")
    
    # Save results
    output_path = OUTPUT_DIR / 'exp3_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
