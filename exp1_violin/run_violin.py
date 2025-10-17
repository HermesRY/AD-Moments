"""
Experiment 1: Violin Plot Analysis
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
from scipy import stats
import pickle

# Import CTMS model
from models.ctms_model import CTMSModel

print("=" * 80)
print("EXPERIMENT 1: VIOLIN PLOT ANALYSIS")
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

# Visualization config
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300

print(f"\n📂 Loading data...")
print(f"  Model: {MODEL_PATH}")
print(f"  Dataset: {DATA_PATH}")
print(f"  Labels: {LABELS_PATH}")

# ============================================================================
# Load Data
# ============================================================================
with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

label_df = pd.read_csv(LABELS_PATH)

print(f"\n✅ Data loaded")
print(f"  Total subjects in dataset: {len(dataset['subjects'])}")
print(f"  Total subjects in labels: {len(label_df)}")

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
# Extract Encodings
# ============================================================================
def extract_sequences(subject_data, seq_len=30, stride=10):
    """Extract sequences from subject data"""
    sequences = []
    for i in range(0, len(subject_data) - seq_len, stride):
        seq = subject_data.iloc[i:i+seq_len]
        if len(seq) != seq_len:
            continue
        action_labels = np.clip(seq['action_label'].values.astype(int), 0, NUM_ACTIVITIES-1)
        hours = seq['hour'].values.astype(float)
        sequences.append({
            'action_labels': action_labels,
            'hours': hours
        })
    return sequences

def get_encodings(sequences, model, device, batch_size=32):
    """Get CTMS encodings from sequences"""
    if not sequences:
        return None
    
    all_encs = {
        'circadian': [],
        'task': [],
        'movement': [],
        'social': []
    }
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            action_ids = torch.from_numpy(np.array([s['action_labels'] for s in batch])).long().to(device)
            hours = torch.from_numpy(np.array([s['hours'] for s in batch])).float().to(device)
            
            outputs = model(action_ids, hours, return_encodings_only=True)
            
            all_encs['circadian'].append(outputs['h_c'].cpu().numpy())
            all_encs['task'].append(outputs['h_t'].cpu().numpy())
            all_encs['movement'].append(outputs['h_m'].cpu().numpy())
            all_encs['social'].append(outputs['h_s'].cpu().numpy())
    
    for dim in all_encs:
        all_encs[dim] = np.concatenate(all_encs[dim], axis=0)
    
    return all_encs

print(f"\n🔄 Extracting encodings...")

# Normalize subject IDs
def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
dataset_ids = {normalize_id(sid): sid for sid in dataset['subjects'].keys()}

# Extract for all subjects
all_subject_data = []

for idx, row in label_df.iterrows():
    subject_id = row['subject_id']
    label = row['label']
    
    norm_id = normalize_id(subject_id)
    if norm_id not in dataset_ids:
        continue
    
    orig_id = dataset_ids[norm_id]
    subject_data = dataset['subjects'][orig_id]['data']
    
    if len(subject_data) < SEQ_LEN:
        continue
    
    sequences = extract_sequences(subject_data, seq_len=SEQ_LEN, stride=STRIDE)
    if not sequences:
        continue
    
    encodings = get_encodings(sequences, model, device, batch_size=BATCH_SIZE)
    if encodings is None:
        continue
    
    # Compute mean encodings per dimension
    for dim in ['circadian', 'task', 'movement', 'social']:
        all_subject_data.append({
            'subject_id': subject_id,
            'label': label,
            'dimension': dim.capitalize(),
            'encoding': np.mean(np.linalg.norm(encodings[dim], axis=1))
        })
    
    print(f"  ✓ {subject_id} ({label}): {len(sequences)} sequences")

df_plot = pd.DataFrame(all_subject_data)

print(f"\n✅ Extracted encodings for {len(df_plot)//4} subjects")

# ============================================================================
# Compute Z-scores (normalized to CN baseline)
# ============================================================================
print(f"\n📊 Computing Z-scores...")

cn_data = df_plot[df_plot['label'] == 'CN']
ci_data = df_plot[df_plot['label'] == 'CI']

# Compute baseline from CN
baseline = {}
for dim in ['Circadian', 'Task', 'Movement', 'Social']:
    dim_data = cn_data[cn_data['dimension'] == dim]['encoding']
    baseline[dim] = {
        'mean': dim_data.mean(),
        'std': dim_data.std()
    }
    print(f"  {dim}: mean={baseline[dim]['mean']:.3f}, std={baseline[dim]['std']:.3f}")

# Apply Z-score normalization
df_plot['z_score'] = df_plot.apply(
    lambda row: (row['encoding'] - baseline[row['dimension']]['mean']) / baseline[row['dimension']]['std'],
    axis=1
)

# ============================================================================
# Statistical Testing
# ============================================================================
print(f"\n📈 Statistical testing...")

stats_results = []
for dim in ['Circadian', 'Task', 'Movement', 'Social']:
    cn_vals = cn_data[cn_data['dimension'] == dim]['z_score']
    ci_vals = ci_data[ci_data['dimension'] == dim]['z_score']
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(cn_vals, ci_vals, alternative='two-sided')
    
    # Effect size (Cohen's d)
    cohens_d = (ci_vals.mean() - cn_vals.mean()) / np.sqrt((cn_vals.std()**2 + ci_vals.std()**2) / 2)
    
    sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
    
    stats_results.append({
        'Dimension': dim,
        'CN_mean': cn_vals.mean(),
        'CN_std': cn_vals.std(),
        'CI_mean': ci_vals.mean(),
        'CI_std': ci_vals.std(),
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': sig
    })
    
    print(f"  {dim:12s}: CN={cn_vals.mean():>6.3f}±{cn_vals.std():.3f}, "
          f"CI={ci_vals.mean():>6.3f}±{ci_vals.std():.3f}, "
          f"p={p_value:.4f} {sig}")

stats_df = pd.DataFrame(stats_results)

# ============================================================================
# Create Violin Plot
# ============================================================================
print(f"\n🎨 Creating violin plot...")

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
colors = {'CN': '#4A90E2', 'CI': '#E89DAC'}

for idx, dim in enumerate(['Circadian', 'Task', 'Movement', 'Social']):
    ax = axes[idx]
    
    dim_data = df_plot[df_plot['dimension'] == dim]
    
    # Create violin plot
    parts = ax.violinplot(
        [dim_data[dim_data['label'] == 'CN']['z_score'],
         dim_data[dim_data['label'] == 'CI']['z_score']],
        positions=[0, 1],
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    # Color the violins
    for i, (pc, label) in enumerate(zip(parts['bodies'], ['CN', 'CI'])):
        pc.set_facecolor(colors[label])
        pc.set_alpha(0.6)
    
    # Styling
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CN', 'CI'], fontweight='bold')
    ax.set_ylabel('Z-Score' if idx == 0 else '', fontweight='bold')
    ax.set_title(dim, fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add p-value
    p_val = stats_df[stats_df['Dimension'] == dim]['p_value'].values[0]
    sig = stats_df[stats_df['Dimension'] == dim]['significance'].values[0]
    ax.text(0.5, ax.get_ylim()[1] * 0.95, f'p = {p_val:.4f} {sig}',
           ha='center', va='top', fontsize=12, fontweight='bold')

plt.suptitle(f'Four-Dimension CTMS Encoding Comparison (n={len(df_plot)//4})',
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_path = os.path.join(OUTPUT_DIR, 'violin_plots.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# Save Statistics
# ============================================================================
stats_path = os.path.join(OUTPUT_DIR, 'statistics.csv')
stats_df.to_csv(stats_path, index=False)
print(f"  ✓ Saved: {stats_path}")

print(f"\n" + "=" * 80)
print(f"✅ EXPERIMENT 1 COMPLETED")
print(f"=" * 80)
print(f"\nResults:")
print(stats_df.to_string(index=False))
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {stats_path}")
print(f"=" * 80)
