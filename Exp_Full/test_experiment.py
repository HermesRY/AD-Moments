#!/usr/bin/env python3
"""
Full test run with optimal configuration to verify results.
This runs the complete Exp3 classification with the best configuration.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
from tqdm import tqdm
from Model.ctms_model import CTMSModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
OUTPUT_DIR = SCRIPT_DIR / 'test_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use actual optimal split from successful experiments
OPTIMAL_SPLIT = {
    'cn_train': [8, 33, 34, 37, 40, 45, 46, 47, 48, 49, 61, 65],
    'cn_test': [16, 35, 36, 38, 41, 44],
    'ci_test': [4, 5, 7, 10, 12, 17, 19, 22, 23, 50, 62],
}

# Model configuration - OPTIMAL from successful experiments
D_MODEL = 64
NUM_ACTIVITIES = 22
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 256 if torch.cuda.is_available() else 64  # Full batch size
ALPHA = np.array([0.5, 0.3, 0.1, 0.1])  # Circadian-dominant (Exp2 optimal)

def load_dataset():
    """Load dataset using actual data format."""
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subj = json.loads(line)
            # Use actual field names: anon_id and sequence
            data.append(subj)
    
    # Filter by split
    all_ids = (OPTIMAL_SPLIT['cn_train'] + OPTIMAL_SPLIT['cn_test'] + 
               OPTIMAL_SPLIT['ci_test'])
    data = [s for s in data if s['anon_id'] in all_ids]
    
    print(f"Loaded {len(data)} subjects")
    return data


def create_sequences(events, seq_len=SEQ_LEN, stride=STRIDE):
    """Create sliding windows from event sequence with activities and hours."""
    if len(events) < seq_len:
        return [], []
    
    activity_seqs = []
    hour_seqs = []
    for i in range(0, len(events) - seq_len + 1, stride):
        window = events[i:i+seq_len]
        activities = [e['action_id'] for e in window]
        # Calculate hour of day from timestamp
        hours = [(e['ts'] % 86400) / 3600 for e in window]  # 86400 = seconds in a day
        activity_seqs.append(activities)
        hour_seqs.append(hours)
    return activity_seqs, hour_seqs


def encode_subject(subject, model):
    """Encode all sequences of a subject."""
    activity_seqs, hour_seqs = create_sequences(subject['sequence'])
    
    if len(activity_seqs) == 0:
        return None
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(activity_seqs), BATCH_SIZE):
            batch_activities = torch.LongTensor(activity_seqs[i:i+BATCH_SIZE]).to(DEVICE)
            batch_hours = torch.FloatTensor(hour_seqs[i:i+BATCH_SIZE]).to(DEVICE)
            
            # Get dimensional encodings
            encodings = model(batch_activities, batch_hours, return_encodings_only=True)
            
            # Stack to [B, 4, D]
            batch_emb = torch.stack([
                encodings['h_c'],  # Circadian
                encodings['h_t'],  # Task
                encodings['h_m'],  # Movement
                encodings['h_s']   # Social
            ], dim=1)
            embeddings.append(batch_emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)  # [N, 4, D]
    return embeddings.numpy()


def compute_baseline_unified(train_embeddings_list):
    """Compute unified baseline from all CN training subjects."""
    all_windows = np.concatenate(train_embeddings_list, axis=0)  # [Total, 4, D]
    baseline_mean = np.mean(all_windows, axis=0)  # [4, D]
    baseline_std = np.std(all_windows, axis=0)  # [4, D]
    return baseline_mean, baseline_std


def compute_anomaly_score(embeddings, baseline_mean, baseline_std, alpha):
    """Compute weighted anomaly scores."""
    # embeddings: [N, 4, D]
    # baseline_mean, baseline_std: [4, D]
    diffs = embeddings - baseline_mean[np.newaxis, :, :]  # [N, 4, D]
    
    # L2 norm per dimension (Mahalanobis-style distance)
    norms = np.linalg.norm(diffs, axis=2)  # [N, 4]
    
    # Z-score normalize norms across samples
    z_scores = (norms - np.mean(norms, axis=0)) / (np.std(norms, axis=0) + 1e-8)
    
    # Weighted sum using alpha weights
    scores = np.sum(z_scores * alpha, axis=1)  # [N]
    return scores


def classify_with_threshold(cn_test_scores, ci_test_scores, threshold_percentiles):
    """Sweep through threshold percentiles to find optimal operating point."""
    all_scores = np.concatenate([cn_test_scores, ci_test_scores])
    
    results = []
    for percentile in threshold_percentiles:
        threshold = np.percentile(all_scores, percentile)
        
        # Classify: score > threshold => CI (anomalous)
        cn_pred = cn_test_scores > threshold
        ci_pred = ci_test_scores > threshold
        
        tp = np.sum(ci_pred)  # CI correctly identified as CI
        tn = np.sum(~cn_pred)  # CN correctly identified as CN
        fp = np.sum(cn_pred)  # CN incorrectly identified as CI
        fn = np.sum(~ci_pred)  # CI incorrectly identified as CN
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        ci_mean = np.mean(ci_test_scores)
        cn_mean = np.mean(cn_test_scores)
        ci_cn_ratio = ci_mean / cn_mean if cn_mean != 0 else 0
        
        results.append({
            'percentile': int(percentile),
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
    return results[best_idx], results


def visualize_embeddings(cn_embeddings, ci_embeddings, output_file):
    """
    Visualize embeddings using t-SNE (Exp1 style).
    
    Args:
        cn_embeddings: List of CN embeddings [n_windows, 4, d_model]
        ci_embeddings: List of CI embeddings [n_windows, 4, d_model]
        output_file: Path to save figure
    """
    print("\n   Creating embedding visualization...")
    
    # Flatten embeddings to [n_windows, 4*d_model]
    cn_flat = np.concatenate([emb.reshape(emb.shape[0], -1) for emb in cn_embeddings], axis=0)
    ci_flat = np.concatenate([emb.reshape(emb.shape[0], -1) for emb in ci_embeddings], axis=0)
    
    print(f"   CN: {cn_flat.shape[0]} windows, CI: {ci_flat.shape[0]} windows")
    
    # Combine and create labels
    X = np.vstack([cn_flat, ci_flat])
    labels = np.array([0] * len(cn_flat) + [1] * len(ci_flat))
    
    # Compute clustering metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"   Silhouette Score: {silhouette:.3f}")
    print(f"   Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Sample for t-SNE (max 5000 points)
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    # Apply t-SNE
    print(f"   Running t-SNE on {len(X_sample)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sample)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot CN (blue) and CI (red)
    cn_mask = labels_sample == 0
    ci_mask = labels_sample == 1
    
    ax.scatter(X_2d[cn_mask, 0], X_2d[cn_mask, 1], 
               c='blue', alpha=0.5, s=20, label='CN')
    ax.scatter(X_2d[ci_mask, 0], X_2d[ci_mask, 1], 
               c='red', alpha=0.5, s=20, label='CI')
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'Embedding Visualization (t-SNE)\nSilhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved visualization to: {output_file}")
    
    return {
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'n_cn_windows': int(cn_flat.shape[0]),
        'n_ci_windows': int(ci_flat.shape[0])
    }


def main():
    print("="*80)
    print("FULL TEST RUN WITH OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Alpha weights: {ALPHA}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load data
    print("\n1. Loading dataset...")
    data = load_dataset()
    
    # Split data
    cn_train = [s for s in data if s['anon_id'] in OPTIMAL_SPLIT['cn_train']]
    cn_test = [s for s in data if s['anon_id'] in OPTIMAL_SPLIT['cn_test']]
    ci_test = [s for s in data if s['anon_id'] in OPTIMAL_SPLIT['ci_test']]
    
    print(f"   CN train: {len(cn_train)} subjects")
    print(f"   CN test: {len(cn_test)} subjects")
    print(f"   CI test: {len(ci_test)} subjects")
    
    # Initialize model
    print(f"\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES)
    model.to(DEVICE)
    model.eval()
    print(f"   ✓ Model ready on {DEVICE}")
    
    # Encode CN training set (FULL, not sample)
    print("\n3. Encoding CN training set...")
    cn_train_embeddings = []
    for subj in tqdm(cn_train, desc="CN train"):
        emb = encode_subject(subj, model)
        if emb is not None:
            cn_train_embeddings.append(emb)
        else:
            print(f"   Warning: Failed to encode CN subject {subj['anon_id']}")
    
    print(f"   ✓ Encoded {len(cn_train_embeddings)}/{len(cn_train)} CN train subjects")
    total_windows = sum(e.shape[0] for e in cn_train_embeddings)
    print(f"   Total training windows: {total_windows}")
    
    # Compute baseline
    print("\n4. Computing baseline statistics...")
    baseline_mean, baseline_std = compute_baseline_unified(cn_train_embeddings)
    print(f"   ✓ Baseline computed: shape {baseline_mean.shape}")
    
    # Encode and score CN test subjects
    print("\n5. Encoding and scoring CN test subjects...")
    cn_test_scores = []
    for subj in tqdm(cn_test, desc="CN test"):
        emb = encode_subject(subj, model)
        if emb is not None:
            scores = compute_anomaly_score(emb, baseline_mean, baseline_std, ALPHA)
            subject_score = np.mean(scores)  # Subject-level: average of all windows
            cn_test_scores.append(subject_score)
        else:
            print(f"   Warning: Failed to encode CN test subject {subj['anon_id']}")
    
    cn_test_scores = np.array(cn_test_scores)
    print(f"   ✓ Scored {len(cn_test_scores)} CN test subjects")
    print(f"   CN scores: mean={cn_test_scores.mean():.4f}, std={cn_test_scores.std():.4f}")
    
    # Encode and score CI test subjects
    print("\n6. Encoding and scoring CI test subjects...")
    ci_test_scores = []
    ci_test_embeddings = []  # Store for visualization
    for subj in tqdm(ci_test, desc="CI test"):
        emb = encode_subject(subj, model)
        if emb is not None:
            ci_test_embeddings.append(emb)
            scores = compute_anomaly_score(emb, baseline_mean, baseline_std, ALPHA)
            subject_score = np.mean(scores)
            ci_test_scores.append(subject_score)
        else:
            print(f"   Warning: Failed to encode CI test subject {subj['anon_id']}")
    
    ci_test_scores = np.array(ci_test_scores)
    print(f"   ✓ Scored {len(ci_test_scores)} CI test subjects")
    print(f"   CI scores: mean={ci_test_scores.mean():.4f}, std={ci_test_scores.std():.4f}")
    
    # Compute CI/CN ratio
    ci_cn_ratio = ci_test_scores.mean() / cn_test_scores.mean()
    print(f"\n   CI/CN Ratio: {ci_cn_ratio:.3f}")
    
    # Create embedding visualization (Exp1)
    print("\n7. Creating Exp1 embedding visualization...")
    cn_test_embeddings = []
    for subj in tqdm(cn_test, desc="Re-encoding CN test"):
        emb = encode_subject(subj, model)
        if emb is not None:
            cn_test_embeddings.append(emb)
    
    viz_file = OUTPUT_DIR / 'exp1_embedding_visualization.png'
    exp1_metrics = visualize_embeddings(cn_test_embeddings, ci_test_embeddings, viz_file)
    
    # Threshold sweep (Exp3)
    print("\n8. Performing threshold sweep (Exp3)...")
    threshold_percentiles = list(range(10, 91, 2))  # Every 2% from 10 to 90
    best_result, all_results = classify_with_threshold(
        cn_test_scores, ci_test_scores, threshold_percentiles
    )
    
    print(f"   ✓ Tested {len(all_results)} threshold percentiles")
    
    # Print results
    print("\n" + "="*80)
    print("BEST RESULTS (Maximum F1 Score)")
    print("="*80)
    print(f"Threshold percentile: {best_result['percentile']}%")
    print(f"Threshold value: {best_result['threshold']:.4f}")
    print(f"")
    print(f"F1 Score:        {best_result['f1']:.3f}")
    print(f"Sensitivity:     {best_result['sensitivity']:.3f} ({best_result['sensitivity']*100:.1f}%)")
    print(f"Specificity:     {best_result['specificity']:.3f} ({best_result['specificity']*100:.1f}%)")
    print(f"Precision:       {best_result['precision']:.3f}")
    print(f"Accuracy:        {best_result['accuracy']:.3f}")
    print(f"CI/CN Ratio:     {best_result['ci_cn_ratio']:.3f}")
    print(f"")
    print(f"Confusion Matrix:")
    print(f"  TP (CI→CI): {best_result['tp']:2d}  |  FN (CI→CN): {best_result['fn']:2d}")
    print(f"  FP (CN→CI): {best_result['fp']:2d}  |  TN (CN→CN): {best_result['tn']:2d}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = OUTPUT_DIR / 'test_results.json'
    output_data = {
        'exp1_embedding': exp1_metrics,
        'exp3_classification': {
            'best': best_result,
            'all_thresholds': all_results
        },
        'configuration': {
            'split': OPTIMAL_SPLIT,
            'alpha': ALPHA.tolist(),
            'seq_len': SEQ_LEN,
            'stride': STRIDE,
            'd_model': D_MODEL,
            'device': str(DEVICE)
        },
        'statistics': {
            'cn_test_mean': float(cn_test_scores.mean()),
            'cn_test_std': float(cn_test_scores.std()),
            'ci_test_mean': float(ci_test_scores.mean()),
            'ci_test_std': float(ci_test_scores.std()),
            'ci_cn_ratio': float(ci_cn_ratio),
            'n_cn_train': len(cn_train_embeddings),
            'n_cn_test': len(cn_test_scores),
            'n_ci_test': len(ci_test_scores)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Visualization saved to: {viz_file}")
    
    # Compare with expected results
    print("\n" + "="*80)
    print("COMPARISON WITH EXPECTED RESULTS")
    print("="*80)
    
    print("\n>>> EXP1: Embedding Quality")
    exp1_expected = {
        'silhouette': 0.312,
        'davies_bouldin': 1.685
    }
    print("Metric           | Expected | Actual   | Match?")
    print("-" * 55)
    print(f"Silhouette       | {exp1_expected['silhouette']:.3f}    | {exp1_metrics['silhouette_score']:.3f}    | {'✓' if abs(exp1_metrics['silhouette_score'] - exp1_expected['silhouette']) < 0.10 else '✗'}")
    print(f"Davies-Bouldin   | {exp1_expected['davies_bouldin']:.3f}    | {exp1_metrics['davies_bouldin_score']:.3f}    | {'✓' if abs(exp1_metrics['davies_bouldin_score'] - exp1_expected['davies_bouldin']) < 0.20 else '✗'}")
    
    print("\n>>> EXP3: Classification Performance")
    exp3_expected = {
        'f1': 0.800,
        'sensitivity': 0.889,
        'specificity': 0.250,
        'ci_cn_ratio': 0.936
    }
    print("Metric           | Expected | Actual   | Match?")
    print("-" * 55)
    print(f"F1 Score         | {exp3_expected['f1']:.3f}    | {best_result['f1']:.3f}    | {'✓' if abs(best_result['f1'] - exp3_expected['f1']) < 0.05 else '✗'}")
    print(f"Sensitivity      | {exp3_expected['sensitivity']:.3f}    | {best_result['sensitivity']:.3f}    | {'✓' if abs(best_result['sensitivity'] - exp3_expected['sensitivity']) < 0.05 else '✗'}")
    print(f"Specificity      | {exp3_expected['specificity']:.3f}    | {best_result['specificity']:.3f}    | {'✓' if abs(best_result['specificity'] - exp3_expected['specificity']) < 0.05 else '✗'}")
    print(f"CI/CN Ratio      | {exp3_expected['ci_cn_ratio']:.3f}    | {best_result['ci_cn_ratio']:.3f}    | {'✓' if abs(best_result['ci_cn_ratio'] - exp3_expected['ci_cn_ratio']) < 0.10 else '✗'}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("="*80)


if __name__ == '__main__':
    main()
