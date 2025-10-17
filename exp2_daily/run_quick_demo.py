"""
Quick test version - processes only 10 CN and 10 CI subjects for demo
"""
import sys
sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import os
os.environ['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output

import torch
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

from ctms_model import CTMSModel

print("Starting quick demo...", flush=True)

# Load config
config = {
    "model_path": "/Users/hermes/Desktop/AD-Moments/ctms_model_best.pth",
    "alpha_weights": [0.29, 0.30, 0.23, 0.18],
    "threshold_multiplier": 2.0,  # Lower threshold for more detections
    "max_cn": 5,  # Only process 5 CN
    "max_ci": 10,  # Only process 10 CI
}

print(f"Config: {config}", flush=True)

# Load data
print("Loading data...", flush=True)
with open('/Users/hermes/Desktop/AD-Moments/New_Code/Data/processed_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
label_df = pd.read_csv('/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv')

# Load model
print("Loading model...", flush=True)
device = torch.device('cpu')
model = CTMSModel(d_model=128, num_activities=5)
checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Normalize subject IDs
def normalize_id(sid):
    return sid.lower().replace('-', '')

# Find subjects
dataset_id_map = {normalize_id(s): s for s in dataset['subjects'].keys()}
label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)

cn_subjects = []
ci_subjects = []

for norm_id, raw_id in dataset_id_map.items():
    label_row = label_df[label_df['normalized_id'] == norm_id]
    if len(label_row) > 0:
        label = label_row['label'].values[0]
        if label == 'CN' and len(cn_subjects) < config['max_cn']:
            cn_subjects.append(raw_id)
        elif label == 'CI' and len(ci_subjects) < config['max_ci']:
            ci_subjects.append(raw_id)
    
    if len(cn_subjects) >= config['max_cn'] and len(ci_subjects) >= config['max_ci']:
        break

print(f"Selected: CN={len(cn_subjects)}, CI={len(ci_subjects)}", flush=True)

# Compute baseline (simplified)
print("Computing baseline...", flush=True)
category_map = {'Lying': 0, 'Standing': 1, 'Daily_Activity': 2, 'Other': 3, 'Unknown': 4}

all_encodings = {'circadian': [], 'task': [], 'movement': [], 'social': []}

with torch.no_grad():
    for subject_id in cn_subjects[:3]:  # Only use 3 for baseline
        subject_df = dataset['subjects'][subject_id]['data']
        
        # Take only first 5 sequences
        for i in range(0, min(len(subject_df)-30, 50), 10):
            seq = subject_df.iloc[i:i+30]
            if len(seq) == 30:
                activity_ids = torch.tensor([[category_map.get(cat, 4) for cat in seq['basic_category'].values]], 
                                           dtype=torch.long).to(device)
                hours = torch.tensor([seq['hour'].values.astype(np.float32)], dtype=torch.float32).to(device)
                
                encodings = model(activity_ids, hours, return_encodings_only=True)
                all_encodings['circadian'].append(encodings['h_c'].cpu().numpy())
                all_encodings['task'].append(encodings['h_t'].cpu().numpy())
                all_encodings['movement'].append(encodings['h_m'].cpu().numpy())
                all_encodings['social'].append(encodings['h_s'].cpu().numpy())

baseline = {}
for dim in ['circadian', 'task', 'movement', 'social']:
    if len(all_encodings[dim]) > 0:
        encodings = np.concatenate(all_encodings[dim], axis=0)
        baseline[f'{dim}_mean'] = np.mean(encodings, axis=0)
        baseline[f'{dim}_std'] = np.std(encodings, axis=0)

print("Baseline computed!", flush=True)

# Detect anomalies
print("Detecting anomalies...", flush=True)
all_subjects = cn_subjects + ci_subjects
subject_moments = {}

with torch.no_grad():
    for idx, subject_id in enumerate(all_subjects):
        print(f"  Processing {idx+1}/{len(all_subjects)}: {subject_id}", flush=True)
        
        # Get label
        norm_id = normalize_id(subject_id)
        label_row = label_df[label_df['normalized_id'] == norm_id]
        label = label_row['label'].values[0]
        
        subject_df = dataset['subjects'][subject_id]['data']
        moments = []
        
        # Process fewer sequences
        for i in range(0, len(subject_df) - 30, 20):  # Larger stride
            seq = subject_df.iloc[i:i+30]
            if len(seq) != 30:
                continue
            
            mid_idx = i + 15
            timestamp_str = subject_df.iloc[mid_idx]['timestamp']
            hour = subject_df.iloc[mid_idx]['hour']
            
            try:
                dt = datetime.strptime(str(timestamp_str), '%Y-%m-%d_%H-%M-%S')
                minute = dt.minute
            except:
                continue
            
            # Only daytime
            if hour < 6 or (hour == 6 and minute < 30) or hour >= 20 or (hour == 19 and minute >= 30):
                continue
            
            # Get encoding
            activity_ids_seq = [category_map.get(cat, 4) for cat in seq['basic_category'].values]
            activity_ids = torch.tensor([activity_ids_seq], dtype=torch.long).to(device)
            hours = torch.tensor([seq['hour'].values.astype(np.float32)], dtype=torch.float32).to(device)
            
            encodings = model(activity_ids, hours, return_encodings_only=True)
            
            # Compute anomaly score
            dim_scores = []
            for dim, enc_key in zip(['circadian', 'task', 'movement', 'social'], ['h_c', 'h_t', 'h_m', 'h_s']):
                enc = encodings[enc_key].cpu().numpy()[0]
                mean = baseline[f'{dim}_mean']
                std = baseline[f'{dim}_std']
                score = np.linalg.norm(enc - mean) / (np.linalg.norm(std) + 1e-6)
                dim_scores.append(score)
            
            combined_score = sum(a * s for a, s in zip(config['alpha_weights'], dim_scores))
            
            if combined_score > config['threshold_multiplier']:
                moments.append({
                    'timestamp': dt.timestamp(),
                    'score': combined_score,
                    'hour': hour,
                    'minute': minute
                })
        
        if len(moments) > 0:
            subject_moments[subject_id] = {'label': label, 'moments': moments}
            print(f"    Found {len(moments)} anomalous moments", flush=True)

print(f"\nDetected anomalies in {len(subject_moments)} subjects", flush=True)

# Create visualization
print("Creating visualization...", flush=True)
os.makedirs('outputs', exist_ok=True)

cn_moments = []
ci_moments = []
cn_count = 0
ci_count = 0

for subject_id, data in subject_moments.items():
    if data['label'] == 'CN':
        cn_moments.extend(data['moments'])
        cn_count += 1
    else:
        ci_moments.extend(data['moments'])
        ci_count += 1

print(f"CN: {cn_count} subjects, {len(cn_moments)} moments", flush=True)
print(f"CI: {ci_count} subjects, {len(ci_moments)} moments", flush=True)

# Create simple heatmap
hour_bins = np.arange(6.5, 20.0, 0.5)
cn_grid = np.zeros((len(hour_bins), 7))
ci_grid = np.zeros((len(hour_bins), 7))

for moment in cn_moments:
    dt = datetime.fromtimestamp(moment['timestamp'])
    hour_decimal = dt.hour + dt.minute / 60.0
    day = dt.weekday()
    bin_idx = np.searchsorted(hour_bins, hour_decimal, side='right') - 1
    if 0 <= bin_idx < len(hour_bins):
        cn_grid[bin_idx, day] += 1

for moment in ci_moments:
    dt = datetime.fromtimestamp(moment['timestamp'])
    hour_decimal = dt.hour + dt.minute / 60.0
    day = dt.weekday()
    bin_idx = np.searchsorted(hour_bins, hour_decimal, side='right') - 1
    if 0 <= bin_idx < len(hour_bins):
        ci_grid[bin_idx, day] += 1

if cn_count > 0:
    cn_grid /= cn_count
if ci_count > 0:
    ci_grid /= ci_count

# Plot
from matplotlib.colors import LinearSegmentedColormap
colors = ['#ffffff', '#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', 
          '#26a69a', '#009688', '#00796b', '#004d40', '#00251a']
cmap = LinearSegmentedColormap.from_list('teal', colors)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

vmax = max(np.max(cn_grid), np.max(ci_grid))

im1 = axes[0].imshow(cn_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
axes[0].set_title('CN - Anomalous Moments (Demo)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Hour')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(ci_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
axes[1].set_title('CI - Anomalous Moments (Demo)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Hour')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('outputs/weekly_demo.png', dpi=300, bbox_inches='tight')
print(f"Saved: outputs/weekly_demo.png", flush=True)

print("\n✅ DEMO COMPLETE!", flush=True)
if ci_count > 0 and cn_count > 0:
    ratio = (len(ci_moments)/ci_count) / (len(cn_moments)/cn_count)
    print(f"CI/CN ratio: {ratio:.2f}×", flush=True)
