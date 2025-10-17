"""
Experiment 2: Daily Temporal Pattern Analysis
Unified runner script with correct paths
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats

# Import CTMS model
from models.ctms_model import CTMSModel

print("=" * 80)
print("EXPERIMENT 2: DAILY TEMPORAL PATTERN ANALYSIS")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "../../../ctms_model_medium.pth"
DATA_PATH = "../../Data/processed_dataset.pkl"
LABELS_PATH = "../../Data/subject_label_mapping_with_scores.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model config
D_MODEL = 64
NUM_ACTIVITIES = 22
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 32

# Time bins (6:30 AM - 7:30 PM, 30-minute bins)
START_HOUR = 6.5
END_HOUR = 19.5
NUM_BINS = 27

# Visualization
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

print(f"\n📂 Loading data...")
print(f"  Model: {MODEL_PATH}")
print(f"  Dataset: {DATA_PATH}")
print(f"  Labels: {LABELS_PATH}")
print(f"  Time range: {START_HOUR:.1f} - {END_HOUR:.1f} ({NUM_BINS} bins)")

# ============================================================================
# Load Data
# ============================================================================
with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

label_df = pd.read_csv(LABELS_PATH)

print(f"\n✅ Data loaded")
print(f"  Subjects in dataset: {len(dataset['subjects'])}")
print(f"  Subjects in labels: {len(label_df)}")

# ============================================================================
# Load Model
# ============================================================================
print(f"\n🔧 Loading model...")
model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES)
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
device = torch.device('cpu')
model = model.to(device)
print(f"✅ Model loaded on {device}")

# ============================================================================
# Helper Functions
# ============================================================================
def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

def extract_sequences_by_hour(subject_data, hour_min, hour_max, seq_len=30, stride=10):
    """Extract sequences within specific hour range"""
    # Filter to hour range
    hour_data = subject_data[(subject_data['hour'] >= hour_min) & (subject_data['hour'] < hour_max)]
    
    if len(hour_data) < seq_len:
        return []
    
    sequences = []
    for i in range(0, len(hour_data) - seq_len, stride):
        seq = hour_data.iloc[i:i+seq_len]
        if len(seq) != seq_len:
            continue
        
        action_labels = np.clip(seq['action_label'].values.astype(int), 0, NUM_ACTIVITIES-1)
        hours = seq['hour'].values.astype(float)
        
        sequences.append({
            'action_labels': action_labels,
            'hours': hours
        })
    
    return sequences

def get_encoding_magnitude(sequences, model, device, batch_size=32):
    """Get average encoding magnitude from sequences"""
    if not sequences or len(sequences) < 5:  # Min 5 sequences per bin
        return None
    
    all_mags = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            action_ids = torch.from_numpy(np.array([s['action_labels'] for s in batch])).long().to(device)
            hours = torch.from_numpy(np.array([s['hours'] for s in batch])).float().to(device)
            
            outputs = model(action_ids, hours, return_encodings_only=True)
            
            # Compute magnitude as L2 norm across all dimensions
            h_c = outputs['h_c'].cpu().numpy()
            h_t = outputs['h_t'].cpu().numpy()
            h_m = outputs['h_m'].cpu().numpy()
            h_s = outputs['h_s'].cpu().numpy()
            
            # Average magnitude
            mag = np.mean([
                np.linalg.norm(h_c, axis=1).mean(),
                np.linalg.norm(h_t, axis=1).mean(),
                np.linalg.norm(h_m, axis=1).mean(),
                np.linalg.norm(h_s, axis=1).mean()
            ])
            
            all_mags.append(mag)
    
    return np.mean(all_mags)

# ============================================================================
# Extract Temporal Patterns
# ============================================================================
print(f"\n🔄 Extracting temporal patterns...")

label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
dataset_ids = {normalize_id(sid): sid for sid in dataset['subjects'].keys()}

# Create time bins
time_bins = np.linspace(START_HOUR, END_HOUR, NUM_BINS + 1)
time_centers = (time_bins[:-1] + time_bins[1:]) / 2

# Store patterns
patterns = {
    'CN': {f'bin_{i}': [] for i in range(NUM_BINS)},
    'CI': {f'bin_{i}': [] for i in range(NUM_BINS)}
}

for idx, row in label_df.iterrows():
    subject_id = row['subject_id']
    label = row['label']
    
    if label not in ['CN', 'CI']:
        continue
    
    norm_id = normalize_id(subject_id)
    if norm_id not in dataset_ids:
        continue
    
    orig_id = dataset_ids[norm_id]
    subject_data = dataset['subjects'][orig_id]['data']
    
    if len(subject_data) < SEQ_LEN:
        continue
    
    # Extract for each time bin
    bin_mags = []
    for bin_idx, (hour_start, hour_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
        sequences = extract_sequences_by_hour(subject_data, hour_start, hour_end, SEQ_LEN, STRIDE)
        mag = get_encoding_magnitude(sequences, model, device, BATCH_SIZE)
        
        if mag is not None:
            patterns[label][f'bin_{bin_idx}'].append(mag)
            bin_mags.append(mag)
    
    if bin_mags:
        print(f"  ✓ {subject_id} ({label}): {len(bin_mags)}/{NUM_BINS} bins")

# ============================================================================
# Compute Statistics
# ============================================================================
print(f"\n📊 Computing statistics...")

cn_pattern = []
cn_std = []
ci_pattern = []
ci_std = []

for i in range(NUM_BINS):
    cn_vals = patterns['CN'][f'bin_{i}']
    ci_vals = patterns['CI'][f'bin_{i}']
    
    cn_mean = np.mean(cn_vals) if cn_vals else np.nan
    cn_std_val = np.std(cn_vals) if cn_vals else np.nan
    ci_mean = np.mean(ci_vals) if ci_vals else np.nan
    ci_std_val = np.std(ci_vals) if ci_vals else np.nan
    
    cn_pattern.append(cn_mean)
    cn_std.append(cn_std_val)
    ci_pattern.append(ci_mean)
    ci_std.append(ci_std_val)

cn_pattern = np.array(cn_pattern)
cn_std = np.array(cn_std)
ci_pattern = np.array(ci_pattern)
ci_std = np.array(ci_std)

# Overall statistics
print(f"\n  CN: avg={np.nanmean(cn_pattern):.4f}, peak at {time_centers[np.nanargmax(cn_pattern)]:.1f}h")
print(f"  CI: avg={np.nanmean(ci_pattern):.4f}, peak at {time_centers[np.nanargmax(ci_pattern)]:.1f}h")
print(f"  CI/CN ratio: {np.nanmean(ci_pattern)/np.nanmean(cn_pattern):.2f}×")

# ============================================================================
# Visualization
# ============================================================================
print(f"\n🎨 Creating visualization...")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot lines with confidence intervals
ax.plot(time_centers, cn_pattern, 'o-', color='#4A90E2', linewidth=2.5, 
       markersize=8, label='CN', alpha=0.9)
ax.fill_between(time_centers, cn_pattern - cn_std, cn_pattern + cn_std, 
                color='#4A90E2', alpha=0.2)

ax.plot(time_centers, ci_pattern, 's-', color='#E89DAC', linewidth=2.5,
       markersize=8, label='CI', alpha=0.9)
ax.fill_between(time_centers, ci_pattern - ci_std, ci_pattern + ci_std,
                color='#E89DAC', alpha=0.2)

# Formatting
ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Encoding Magnitude', fontsize=14, fontweight='bold')
ax.set_title('Daily Temporal Pattern of CTMS Encodings', 
            fontsize=16, fontweight='bold')

# X-axis formatting
hour_labels = [f'{int(h)}:{int((h%1)*60):02d}' for h in time_centers[::3]]
ax.set_xticks(time_centers[::3])
ax.set_xticklabels(hour_labels, rotation=45, ha='right')

ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save
output_path = os.path.join(OUTPUT_DIR, 'daily_pattern.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Save data
data_path = os.path.join(OUTPUT_DIR, 'daily_pattern_data.npz')
np.savez(data_path,
        time_centers=time_centers,
        cn_pattern=cn_pattern,
        cn_std=cn_std,
        ci_pattern=ci_pattern,
        ci_std=ci_std)
print(f"  ✓ Saved: {data_path}")

print(f"\n" + "=" * 80)
print(f"✅ EXPERIMENT 2 COMPLETED")
print(f"=" * 80)
print(f"\nKey Findings:")
print(f"  CN Peak: {time_centers[np.nanargmax(cn_pattern)]:.1f}h")
print(f"  CI Peak: {time_centers[np.nanargmax(ci_pattern)]:.1f}h")
print(f"  Peak shift: {time_centers[np.nanargmax(ci_pattern)] - time_centers[np.nanargmax(cn_pattern)]:.1f}h earlier")
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {data_path}")
print(f"=" * 80)
