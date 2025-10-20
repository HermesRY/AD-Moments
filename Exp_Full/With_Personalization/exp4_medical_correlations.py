#!/usr/bin/env python3
"""
Exp4: Medical Correlations (Without Personalization)
Analyzes correlations between CTMS dimensions and medical assessments.

Usage:
    python exp4_medical_correlations.py

Note:
    Requires subjects_public.json with MoCA, ZBI, DSS, FAS scores.
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
import pandas as pd
from tqdm import tqdm
from scipy import stats
from Model.ctms_model import CTMSModel

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
SUBJECTS_PATH = PROJECT_ROOT / 'sample_data' / 'subjects_public.json'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optimal split from Exp2
OPTIMAL_SPLIT = {
    'cn_train': [8, 33, 34, 37, 40, 45, 46, 47, 48, 49, 61, 65],
    'cn_test': [16, 35, 36, 38, 41, 44],
    'ci_test': [4, 5, 7, 10, 12, 17, 19, 22, 23, 50, 62],
}

# Model configuration
ALPHA = np.array([0.5, 0.3, 0.1, 0.1])
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 256
D_MODEL = 64
NUM_ACTIVITIES = 22

# Medical scores to correlate with
MEDICAL_SCORES = ['moca', 'zbi', 'dss', 'fas']
DIMENSION_NAMES = ['Circadian', 'Task', 'Movement', 'Social']

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load activity data and subject metadata."""
    # Load activities
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subj = json.loads(line)
            data.append(subj)
    
    # Load medical assessments
    with open(SUBJECTS_PATH, 'r') as f:
        subjects_meta = json.load(f)
    
    # Create lookup
    meta_lookup = {s['subject_id']: s for s in subjects_meta}
    
    # Filter by split
    all_ids = (OPTIMAL_SPLIT['cn_train'] + OPTIMAL_SPLIT['cn_test'] + 
               OPTIMAL_SPLIT['ci_test'])
    data = [s for s in data if s['subject_id'] in all_ids]
    
    print(f"Loaded {len(data)} subjects with medical data")
    return data, meta_lookup


def create_sequences(activity_series, seq_len=SEQ_LEN, stride=STRIDE):
    """Create sliding windows."""
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
    all_windows = np.concatenate(train_embeddings_list, axis=0)  # [Total, 4, D]
    baseline_mean = np.mean(all_windows, axis=0)  # [4, D]
    baseline_std = np.std(all_windows, axis=0)  # [4, D]
    return baseline_mean, baseline_std


def compute_dimension_scores(embeddings, baseline_mean, baseline_std):
    """Compute per-dimension deviation scores."""
    # embeddings: [N, 4, D]
    # baseline_mean, baseline_std: [4, D]
    diffs = embeddings - baseline_mean[np.newaxis, :, :]  # [N, 4, D]
    z_scores = diffs / (baseline_std[np.newaxis, :, :] + 1e-8)  # [N, 4, D]
    
    # Mean absolute z-score per dimension
    dim_scores = np.mean(np.abs(z_scores), axis=(0, 2))  # [4]
    return dim_scores


def correlate_with_medical(subject_scores_df, meta_lookup):
    """Compute correlations between dimension scores and medical assessments."""
    # Add medical scores to dataframe
    for score_name in MEDICAL_SCORES:
        subject_scores_df[score_name] = subject_scores_df['subject_id'].apply(
            lambda sid: meta_lookup.get(sid, {}).get(score_name, np.nan)
        )
    
    # Remove subjects without medical data
    valid_mask = subject_scores_df[MEDICAL_SCORES].notna().all(axis=1)
    df_valid = subject_scores_df[valid_mask].copy()
    
    print(f"   Valid subjects for correlation: {len(df_valid)}")
    
    # Compute correlations
    correlations = {}
    for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
        dim_col = f'dim_{dim_idx}'
        correlations[dim_name] = {}
        
        for score_name in MEDICAL_SCORES:
            r, p = stats.pearsonr(df_valid[dim_col], df_valid[score_name])
            correlations[dim_name][score_name] = {
                'r': float(r),
                'p': float(p),
                'n': len(df_valid)
            }
    
    return correlations, df_valid


def main():
    print("="*80)
    print("EXP4: MEDICAL CORRELATIONS (WITHOUT PERSONALIZATION)")
    print("="*80)
    
    # Load data
    print("\n1. Loading dataset and medical assessments...")
    data, meta_lookup = load_dataset()
    
    # Split data
    cn_train = [s for s in data if s['subject_id'] in OPTIMAL_SPLIT['cn_train']]
    test_subjects = [s for s in data if s['subject_id'] in 
                     OPTIMAL_SPLIT['cn_test'] + OPTIMAL_SPLIT['ci_test']]
    
    print(f"   CN train: {len(cn_train)}")
    print(f"   Test subjects: {len(test_subjects)}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    
    # Encode CN training set
    print("\n3. Encoding CN training set...")
    cn_train_embeddings = []
    for subj in tqdm(cn_train, desc="CN train"):
        emb = encode_subject(subj, model)
        if emb is not None:
            cn_train_embeddings.append(emb)
    
    # Compute baseline statistics
    print("\n4. Computing baseline statistics...")
    baseline_mean, baseline_std = compute_baseline_unified(cn_train_embeddings)
    print(f"   Baseline shape: {baseline_mean.shape}")
    
    # Encode test subjects and compute dimension scores
    print("\n5. Computing dimension scores for test subjects...")
    subject_scores = []
    for subj in tqdm(test_subjects, desc="Test subjects"):
        emb = encode_subject(subj, model)
        if emb is not None:
            dim_scores = compute_dimension_scores(emb, baseline_mean, baseline_std)
            subject_scores.append({
                'subject_id': subj['subject_id'],
                'label': subj['label'],
                **{f'dim_{i}': float(dim_scores[i]) for i in range(4)}
            })
    
    subject_scores_df = pd.DataFrame(subject_scores)
    print(f"   Computed scores for {len(subject_scores_df)} subjects")
    
    # Correlate with medical assessments
    print("\n6. Computing correlations with medical assessments...")
    correlations, df_valid = correlate_with_medical(subject_scores_df, meta_lookup)
    
    # Print significant correlations
    print("\n" + "="*80)
    print("SIGNIFICANT CORRELATIONS (p < 0.05)")
    print("="*80)
    sig_count = 0
    for dim_name in DIMENSION_NAMES:
        for score_name in MEDICAL_SCORES:
            corr = correlations[dim_name][score_name]
            if corr['p'] < 0.05:
                sig_count += 1
                print(f"{dim_name} - {score_name.upper()}: "
                      f"r={corr['r']:.3f}, p={corr['p']:.4f}{'*' if corr['p']<0.05 else ''}")
    
    if sig_count == 0:
        print("No significant correlations found (p < 0.05)")
    
    # Save results
    output_path = OUTPUT_DIR / 'exp4_correlations.json'
    results = {
        'correlations': correlations,
        'significant_count': sig_count,
        'total_tests': len(DIMENSION_NAMES) * len(MEDICAL_SCORES),
        'n_subjects': len(df_valid),
        'dimension_names': DIMENSION_NAMES,
        'medical_scores': MEDICAL_SCORES
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
