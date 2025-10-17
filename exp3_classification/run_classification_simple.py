"""
Version 5 - Experiment 3: CI/CN Classification Performance (Simplified)
Using anomaly counts from exp2_weekly results
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def load_exp2_results():
    """Load anomaly detection results from exp2"""
    print("📂 Loading exp2 results...")
    
    with open('../exp2_weekly/outputs/subject_moments_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Extract anomaly counts and labels
    subjects = []
    anomaly_counts = []
    labels = []
    
    for subject_id, data in summary.items():
        subjects.append(subject_id)
        anomaly_counts.append(data['num_anomalies'])
        labels.append(1 if data['label'] == 'CI' else 0)
    
    print(f"✓ Loaded {len(subjects)} subjects")
    print(f"  CN: {sum(1 for l in labels if l == 0)}")
    print(f"  CI: {sum(1 for l in labels if l == 1)}")
    
    return np.array(anomaly_counts), np.array(labels), subjects

def find_optimal_threshold(counts, labels):
    """Find optimal threshold for classification"""
    thresholds = np.percentile(counts, np.arange(10, 90, 5))
    
    best_f1 = 0
    best_threshold = 0
    best_results = None
    
    for thresh in thresholds:
        preds = (counts > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            
            best_results = {
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'confusion_matrix': cm.tolist(),
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
    
    return best_results

def compute_anomaly_statistics(counts, labels):
    """Compute CN/CI anomaly statistics"""
    cn_counts = counts[labels == 0]
    ci_counts = counts[labels == 1]
    
    stats = {
        'cn_mean': float(np.mean(cn_counts)),
        'cn_std': float(np.std(cn_counts)),
        'cn_median': float(np.median(cn_counts)),
        'ci_mean': float(np.mean(ci_counts)),
        'ci_std': float(np.std(ci_counts)),
        'ci_median': float(np.median(ci_counts)),
        'ci_cn_ratio': float(np.mean(ci_counts) / np.mean(cn_counts)) if np.mean(cn_counts) > 0 else 0
    }
    
    return stats

def main():
    print("="*80)
    print("VERSION 5 - EXPERIMENT 3: CI/CN CLASSIFICATION PERFORMANCE")
    print("="*80)
    print()
    
    # Load anomaly counts from exp2
    counts, labels, subjects = load_exp2_results()
    
    # Find optimal threshold
    print("\n🔍 Finding optimal classification threshold...")
    results = find_optimal_threshold(counts, labels)
    
    print(f"\n✓ Optimal threshold: {results['threshold']:.1f} anomalies")
    print(f"\nClassification Performance:")
    print(f"  Precision:   {results['precision']:.3f}")
    print(f"  Recall:      {results['recall']:.3f}")
    print(f"  F1-Score:    {results['f1']:.3f}")
    print(f"  Sensitivity: {results['sensitivity']:.3f}")
    print(f"  Specificity: {results['specificity']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"           CN    CI")
    print(f"Actual CN  {results['tn']}    {results['fp']}")
    print(f"Actual CI  {results['fn']}    {results['tp']}")
    
    # Compute anomaly statistics
    print("\n📊 Computing anomaly statistics...")
    stats = compute_anomaly_statistics(counts, labels)
    
    print(f"\nAnomaly Counts:")
    print(f"  CN: {stats['cn_mean']:.1f}±{stats['cn_std']:.1f} (median: {stats['cn_median']:.1f})")
    print(f"  CI: {stats['ci_mean']:.1f}±{stats['ci_std']:.1f} (median: {stats['ci_median']:.1f})")
    print(f"  CI/CN Ratio: {stats['ci_cn_ratio']:.2f}×")
    
    # Save results
    all_results = {
        'classification': results,
        'statistics': stats,
        'subjects': {
            subj: {'anomaly_count': int(count), 'label': 'CI' if label == 1 else 'CN'}
            for subj, count, label in zip(subjects, counts, labels)
        }
    }
    
    with open('outputs/classification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to outputs/classification_results.json")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()
