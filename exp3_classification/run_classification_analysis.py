"""
Version 5 - Experiment 3: CI/CN Classification Performance
Goal: Evaluate classification accuracy using CTMS-detected anomalous moments
Compare: Baseline vs CTMS without personalization vs CTMS with personalization
"""

import sys
import os
sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import torch
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from ctms_model import CTMSModel

def load_data():
    """Load dataset and labels"""
    print("📂 Loading data...")
    
    with open('/Users/hermes/Desktop/AD-Moments/New_Code/Data/processed_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    label_df = pd.read_csv('/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv')
    
    print(f"✓ Loaded {len(dataset['subjects'])} subjects")
    print(f"✓ Label mapping: {len(label_df)} entries")
    
    return dataset, label_df

def normalize_subject_id(subject_id):
    """Normalize subject ID format"""
    return subject_id.lower().replace('-', '')

def load_model(model_path):
    """Load trained CTMS model"""
    print(f"🔧 Loading model from {model_path}...")
    
    device = torch.device('cpu')
    model = CTMSModel(d_model=128, num_activities=5)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    alpha_weights = checkpoint.get('alpha_weights', [0.29, 0.30, 0.23, 0.18])
    
    print(f"✓ Model loaded (device: {device})")
    print(f"✓ Alpha weights: {alpha_weights}")
    
    return model, alpha_weights, device

def compute_baseline_statistics(dataset, label_df, cn_subjects):
    """
    Baseline method: Statistical aggregation (like ADMarker)
    Compute simple statistics: activity counts, transitions, durations
    """
    print("📊 Computing baseline statistics (ADMarker-style)...")
    
    features = []
    labels = []
    subject_ids = []
    
    for subject_id in cn_subjects + [s for s in dataset['subjects'].keys() if s not in cn_subjects]:
        norm_id = normalize_subject_id(subject_id)
        label_row = label_df[label_df['subject_id'] == norm_id]
        
        if label_row.empty:
            continue
        
        label = label_row.iloc[0]['label']
        if label not in ['CN', 'CI']:
            continue
        
        subject_df = dataset['subjects'][subject_id]['data']
        
        # Compute simple statistics
        total_frames = len(subject_df)
        activity_counts = subject_df['basic_category'].value_counts()
        
        # Transitions
        transitions = (subject_df['basic_category'].shift() != subject_df['basic_category']).sum()
        
        # Average duration per activity
        avg_duration = total_frames / max(len(activity_counts), 1)
        
        # Hour distribution variance
        hour_var = subject_df['hour'].var()
        
        feature_vec = [
            total_frames,
            len(activity_counts),
            transitions,
            avg_duration,
            hour_var,
            activity_counts.get('Lying', 0),
            activity_counts.get('Standing', 0),
            activity_counts.get('Daily_Activity', 0),
            activity_counts.get('Other', 0),
            activity_counts.get('Unknown', 0)
        ]
        
        features.append(feature_vec)
        labels.append(1 if label == 'CI' else 0)
        subject_ids.append(subject_id)
    
    return np.array(features), np.array(labels), subject_ids

def detect_anomalies_no_personalization(model, dataset, label_df, cn_baseline, alpha_weights, device, threshold=1.5):
    """
    CTMS without personalization: Use global CN baseline for all subjects
    """
    print("🔍 Detecting anomalies (no personalization)...")
    
    anomaly_counts = []
    labels = []
    subject_ids = []
    
    for subject_id in dataset['subjects'].keys():
        norm_id = normalize_subject_id(subject_id)
        label_row = label_df[label_df['subject_id'] == norm_id]
        
        if label_row.empty:
            continue
        
        label = label_row.iloc[0]['label']
        if label not in ['CN', 'CI']:
            continue
        
        subject_df = dataset['subjects'][subject_id]['data']
        
        # Extract sequences
        anomaly_count = 0
        for i in range(0, len(subject_df) - 30, 10):
            seq = subject_df.iloc[i:i+30]
            
            # Filter daytime
            mid_idx = i + 15
            hour = subject_df.iloc[mid_idx]['hour']
            if not (6.5 <= hour <= 19.5):
                continue
            
            activity_ids = seq['basic_category'].map({
                'Lying': 0, 'Standing': 1, 'Daily_Activity': 2,
                'Other': 3, 'Unknown': 4
            }).fillna(4).values
            
            hours = seq['hour'].values
            
            # Convert to tensors
            activity_tensor = torch.from_numpy(np.array([activity_ids], dtype=np.int64)).to(device)
            hour_tensor = torch.from_numpy(np.array([hours], dtype=np.float32)).to(device)
            
            with torch.no_grad():
                h_c, h_t, h_m, h_s = model(activity_tensor, hour_tensor, return_encodings_only=True)
                
                encodings = [h_c[0], h_t[0], h_m[0], h_s[0]]
                
                # Compute anomaly scores
                scores = []
                for j, enc in enumerate(encodings):
                    dim_names = ['circadian', 'task', 'movement', 'social']
                    mean = cn_baseline[f'{dim_names[j]}_mean']
                    std = cn_baseline[f'{dim_names[j]}_std']
                    
                    diff = torch.norm(enc - mean)
                    score = diff / (torch.norm(std) + 1e-8)
                    scores.append(score.item())
                
                # Combined score
                combined = sum(alpha * s for alpha, s in zip(alpha_weights, scores))
                
                if combined > threshold:
                    anomaly_count += 1
        
        anomaly_counts.append(anomaly_count)
        labels.append(1 if label == 'CI' else 0)
        subject_ids.append(subject_id)
    
    return np.array(anomaly_counts).reshape(-1, 1), np.array(labels), subject_ids

def detect_anomalies_with_personalization(model, dataset, label_df, cn_baseline, alpha_weights, device, threshold=1.5):
    """
    CTMS with personalization: Compute subject-specific baseline from first week
    """
    print("🔍 Detecting anomalies (with personalization)...")
    
    anomaly_counts = []
    labels = []
    subject_ids = []
    
    for subject_id in dataset['subjects'].keys():
        norm_id = normalize_subject_id(subject_id)
        label_row = label_df[label_df['subject_id'] == norm_id]
        
        if label_row.empty:
            continue
        
        label = label_row.iloc[0]['label']
        if label not in ['CN', 'CI']:
            continue
        
        subject_df = dataset['subjects'][subject_id]['data']
        
        # Use first 20% as personal baseline
        split_idx = int(len(subject_df) * 0.2)
        baseline_df = subject_df.iloc[:split_idx]
        test_df = subject_df.iloc[split_idx:]
        
        # Compute personal baseline
        personal_baseline = {}
        encodings_list = {k: [] for k in ['circadian', 'task', 'movement', 'social']}
        
        for i in range(0, len(baseline_df) - 30, 20):
            seq = baseline_df.iloc[i:i+30]
            
            activity_ids = seq['basic_category'].map({
                'Lying': 0, 'Standing': 1, 'Daily_Activity': 2,
                'Other': 3, 'Unknown': 4
            }).fillna(4).values
            
            hours = seq['hour'].values
            
            activity_tensor = torch.from_numpy(np.array([activity_ids], dtype=np.int64)).to(device)
            hour_tensor = torch.from_numpy(np.array([hours], dtype=np.float32)).to(device)
            
            with torch.no_grad():
                h_c, h_t, h_m, h_s = model(activity_tensor, hour_tensor, return_encodings_only=True)
                
                encodings_list['circadian'].append(h_c[0].cpu().numpy())
                encodings_list['task'].append(h_t[0].cpu().numpy())
                encodings_list['movement'].append(h_m[0].cpu().numpy())
                encodings_list['social'].append(h_s[0].cpu().numpy())
        
        # Compute mean and std for personal baseline
        for dim_name in ['circadian', 'task', 'movement', 'social']:
            if len(encodings_list[dim_name]) > 0:
                encs = np.array(encodings_list[dim_name])
                personal_baseline[f'{dim_name}_mean'] = torch.tensor(np.mean(encs, axis=0)).to(device)
                personal_baseline[f'{dim_name}_std'] = torch.tensor(np.std(encs, axis=0) + 1e-8).to(device)
            else:
                # Fallback to CN baseline
                personal_baseline[f'{dim_name}_mean'] = cn_baseline[f'{dim_name}_mean']
                personal_baseline[f'{dim_name}_std'] = cn_baseline[f'{dim_name}_std']
        
        # Detect anomalies in test set
        anomaly_count = 0
        for i in range(0, len(test_df) - 30, 10):
            seq = test_df.iloc[i:i+30]
            
            # Filter daytime
            mid_idx = split_idx + i + 15
            if mid_idx >= len(subject_df):
                continue
            hour = subject_df.iloc[mid_idx]['hour']
            if not (6.5 <= hour <= 19.5):
                continue
            
            activity_ids = seq['basic_category'].map({
                'Lying': 0, 'Standing': 1, 'Daily_Activity': 2,
                'Other': 3, 'Unknown': 4
            }).fillna(4).values
            
            hours = seq['hour'].values
            
            activity_tensor = torch.from_numpy(np.array([activity_ids], dtype=np.int64)).to(device)
            hour_tensor = torch.from_numpy(np.array([hours], dtype=np.float32)).to(device)
            
            with torch.no_grad():
                h_c, h_t, h_m, h_s = model(activity_tensor, hour_tensor, return_encodings_only=True)
                
                encodings = [h_c[0], h_t[0], h_m[0], h_s[0]]
                
                # Compute anomaly scores using personal baseline
                scores = []
                for j, enc in enumerate(encodings):
                    dim_names = ['circadian', 'task', 'movement', 'social']
                    mean = personal_baseline[f'{dim_names[j]}_mean']
                    std = personal_baseline[f'{dim_names[j]}_std']
                    
                    diff = torch.norm(enc - mean)
                    score = diff / (torch.norm(std) + 1e-8)
                    scores.append(score.item())
                
                combined = sum(alpha * s for alpha, s in zip(alpha_weights, scores))
                
                if combined > threshold:
                    anomaly_count += 1
        
        anomaly_counts.append(anomaly_count)
        labels.append(1 if label == 'CI' else 0)
        subject_ids.append(subject_id)
    
    return np.array(anomaly_counts).reshape(-1, 1), np.array(labels), subject_ids

def simple_classifier(features, labels, threshold):
    """
    Simple threshold-based classifier
    Predict CI if feature value > threshold
    """
    predictions = (features.flatten() > threshold).astype(int)
    return predictions

def find_optimal_threshold(features, labels):
    """Find optimal threshold for classification"""
    feature_values = features.flatten()
    thresholds = np.percentile(feature_values, np.arange(10, 90, 5))
    
    best_f1 = 0
    best_threshold = 0
    
    for thresh in thresholds:
        preds = simple_classifier(features, labels, thresh)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold

def evaluate_method(features, labels, subject_ids, method_name):
    """Evaluate classification performance"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*60}")
    
    # Find optimal threshold
    threshold = find_optimal_threshold(features, labels)
    print(f"Optimal threshold: {threshold:.2f}")
    
    # Make predictions
    predictions = simple_classifier(features, labels, threshold)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Compute CI/CN anomaly rates
    cn_features = features[labels == 0].flatten()
    ci_features = features[labels == 1].flatten()
    
    cn_mean = np.mean(cn_features)
    cn_std = np.std(cn_features)
    ci_mean = np.mean(ci_features)
    ci_std = np.std(ci_features)
    ratio = ci_mean / cn_mean if cn_mean > 0 else 0
    
    results = {
        'method': method_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'threshold': threshold,
        'confusion_matrix': cm.tolist(),
        'cn_mean': cn_mean,
        'cn_std': cn_std,
        'ci_mean': ci_mean,
        'ci_std': ci_std,
        'ci_cn_ratio': ratio
    }
    
    print(f"\nResults:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"\nAnomaly Rates:")
    print(f"  CN: {cn_mean:.1f}±{cn_std:.1f}")
    print(f"  CI: {ci_mean:.1f}±{ci_std:.1f}")
    print(f"  CI/CN Ratio: {ratio:.2f}×")
    
    return results

def main():
    print("="*80)
    print("VERSION 5 - EXPERIMENT 3: CI/CN CLASSIFICATION PERFORMANCE")
    print("="*80)
    print()
    
    # Load data
    dataset, label_df = load_data()
    
    # Load model
    model_path = '/Users/hermes/Desktop/AD-Moments/ctms_model_best.pth'
    model, alpha_weights, device = load_model(model_path)
    
    # Get CN subjects for baseline
    cn_subject_ids = []
    for _, row in label_df.iterrows():
        if row['label'] == 'CN':
            for subj_id in dataset['subjects'].keys():
                if normalize_subject_id(subj_id) == row['subject_id']:
                    cn_subject_ids.append(subj_id)
                    break
    
    print(f"\n📋 Using {len(cn_subject_ids)} CN subjects for baseline")
    
    # Compute CN baseline (for CTMS methods)
    print("\n📊 Computing CN baseline...")
    cn_baseline = {}
    encodings_list = {k: [] for k in ['circadian', 'task', 'movement', 'social']}
    
    for subject_id in cn_subject_ids:
        subject_df = dataset['subjects'][subject_id]['data']
        
        for i in range(0, len(subject_df) - 30, 20):
            seq = subject_df.iloc[i:i+30]
            
            activity_ids = seq['basic_category'].map({
                'Lying': 0, 'Standing': 1, 'Daily_Activity': 2,
                'Other': 3, 'Unknown': 4
            }).fillna(4).values
            
            hours = seq['hour'].values
            
            activity_tensor = torch.from_numpy(np.array([activity_ids], dtype=np.int64)).to(device)
            hour_tensor = torch.from_numpy(np.array([hours], dtype=np.float32)).to(device)
            
            with torch.no_grad():
                encodings_dict = model(activity_tensor, hour_tensor, return_encodings_only=True)
                
                encodings_list['circadian'].append(encodings_dict['h_c'][0].cpu().numpy())
                encodings_list['task'].append(encodings_dict['h_t'][0].cpu().numpy())
                encodings_list['movement'].append(encodings_dict['h_m'][0].cpu().numpy())
                encodings_list['social'].append(encodings_dict['h_s'][0].cpu().numpy())
    
    for dim_name in ['circadian', 'task', 'movement', 'social']:
        encs = np.array(encodings_list[dim_name])
        cn_baseline[f'{dim_name}_mean'] = torch.tensor(np.mean(encs, axis=0)).to(device)
        cn_baseline[f'{dim_name}_std'] = torch.tensor(np.std(encs, axis=0) + 1e-8).to(device)
    
    print("✓ CN baseline computed")
    
    # Method 1: Baseline (statistical aggregation)
    features_baseline, labels_baseline, subj_baseline = compute_baseline_statistics(
        dataset, label_df, cn_subject_ids
    )
    # Use total transitions as feature for baseline
    results_baseline = evaluate_method(
        features_baseline[:, 2:3],  # transitions column
        labels_baseline,
        subj_baseline,
        "Baseline (Statistical Aggregation)"
    )
    
    # Method 2: CTMS without personalization
    features_no_pers, labels_no_pers, subj_no_pers = detect_anomalies_no_personalization(
        model, dataset, label_df, cn_baseline, alpha_weights, device, threshold=1.2
    )
    results_no_pers = evaluate_method(
        features_no_pers,
        labels_no_pers,
        subj_no_pers,
        "CTMS (No Personalization)"
    )
    
    # Method 3: CTMS with personalization
    features_with_pers, labels_with_pers, subj_with_pers = detect_anomalies_with_personalization(
        model, dataset, label_df, cn_baseline, alpha_weights, device, threshold=1.2
    )
    results_with_pers = evaluate_method(
        features_with_pers,
        labels_with_pers,
        subj_with_pers,
        "CTMS (With Personalization)"
    )
    
    # Save results
    all_results = {
        'baseline': results_baseline,
        'no_personalization': results_no_pers,
        'with_personalization': results_with_pers
    }
    
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/classification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/classification_results.json")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
