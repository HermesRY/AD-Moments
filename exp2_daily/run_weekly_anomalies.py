"""
Version 5 - Experiment 2: Weekly Temporal Pattern of Anomalous Moments
Goal: Show clear CN vs CI difference using CTMS-detected anomalous moments
Focus: 6:30-19:30 daytime hours only
"""

import sys
import os

# Force unbuffered output for progress display
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import torch
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from tqdm import tqdm  # Progress bar

from ctms_model import CTMSModel

def load_config():
    """Load experiment configuration"""
    # Use absolute path to config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return {
            "model_path": "/Users/hermes/Desktop/AD-Moments/ctms_model_best.pth",
            "alpha_weights": [0.29, 0.30, 0.23, 0.18],
            "threshold_multiplier": 2.5,
            "random_seed": 42
        }

def load_data():
    """Load dataset and labels"""
    print("📂 Loading data...")
    
    with open('/Users/hermes/Desktop/AD-Moments/New_Code/Data/processed_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    label_df = pd.read_csv('/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv')
    
    print(f"✓ Loaded {len(dataset['subjects'])} subjects")
    print(f"✓ Label mapping: {len(label_df)} entries")
    
    return dataset, label_df

def load_model(model_path):
    """Load trained CTMS model"""
    print(f"🔧 Loading model from {model_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model was trained with 5 basic categories
    model = CTMSModel(
        d_model=128,
        num_activities=5  # Changed from 22 to 5 (basic categories)
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (device: {device})")
    print(f"✓ Model uses {checkpoint['model_state_dict']['circadian_encoder.activity_embed.weight'].shape[0]} activity categories")
    return model, device

def compute_cn_baseline(model, device, cn_subjects, dataset, alpha_weights):
    """Compute baseline from CN subjects only"""
    print(f"📊 Computing CN baseline from {len(cn_subjects)} subjects...")
    
    # Define category mapping
    category_map = {
        'Lying': 0,
        'Standing': 1,
        'Daily_Activity': 2,
        'Other': 3,
        'Unknown': 4
    }
    
    all_encodings = {
        'circadian': [],
        'task': [],
        'movement': [],
        'social': []
    }
    
    with torch.no_grad():
        # Add progress bar for baseline computation
        for subject_id in tqdm(cn_subjects, desc="Computing baseline", ncols=80):
            if subject_id not in dataset['subjects']:
                continue
            
            subject_df = dataset['subjects'][subject_id]['data']
            if len(subject_df) == 0:
                continue
            
            # Sample sequences
            sequences = []
            for i in range(0, len(subject_df) - 30, 10):
                seq = subject_df.iloc[i:i+30]
                if len(seq) == 30:
                    sequences.append(seq)
            
            if len(sequences) == 0:
                continue
            
            # Process in batches
            batch_size = 32
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                # Convert basic_category to numeric IDs
                activity_ids_list = []
                for seq in batch:
                    ids = [category_map.get(cat, 4) for cat in seq['basic_category'].values]
                    activity_ids_list.append(ids)
                
                # Convert to numpy array first, then to tensor (faster)
                activity_ids = torch.from_numpy(np.array(activity_ids_list, dtype=np.int64)).to(device)
                hours_array = np.array([seq['hour'].values for seq in batch], dtype=np.float32)
                hours = torch.from_numpy(hours_array).to(device)
                
                # Forward pass - get encodings only
                encodings = model(activity_ids, hours, return_encodings_only=True)
                
                all_encodings['circadian'].append(encodings['h_c'].cpu().numpy())
                all_encodings['task'].append(encodings['h_t'].cpu().numpy())
                all_encodings['movement'].append(encodings['h_m'].cpu().numpy())
                all_encodings['social'].append(encodings['h_s'].cpu().numpy())
    
    # Compute mean and std
    baseline = {}
    for dim in ['circadian', 'task', 'movement', 'social']:
        if len(all_encodings[dim]) > 0:
            encodings = np.concatenate(all_encodings[dim], axis=0)
            baseline[f'{dim}_mean'] = np.mean(encodings, axis=0)
            baseline[f'{dim}_std'] = np.std(encodings, axis=0)
        else:
            baseline[f'{dim}_mean'] = np.zeros(128)  # d_model=128
            baseline[f'{dim}_std'] = np.ones(128)
    
    print(f"✓ CN baseline computed")
    return baseline

def detect_anomalous_moments(model, device, baseline, subjects, label_mapping, 
                              dataset, alpha_weights, threshold_multiplier):
    """Detect anomalous moments for each subject"""
    print(f"🔍 Detecting anomalous moments for {len(subjects)} subjects...")
    
    # Define category mapping
    category_map = {
        'Lying': 0,
        'Standing': 1,
        'Daily_Activity': 2,
        'Other': 3,
        'Unknown': 4
    }
    
    subject_moments = {}
    
    with torch.no_grad():
        # Add progress bar for detection
        for idx, subject_id in enumerate(tqdm(subjects, desc="Detecting anomalies", ncols=80)):
            if subject_id not in dataset['subjects']:
                continue
            
            # Get label
            label_row = label_mapping[label_mapping['subject_id'] == subject_id]
            if len(label_row) == 0:
                continue
            label = label_row['label'].values[0]
            
            subject_df = dataset['subjects'][subject_id]['data']
            if len(subject_df) == 0:
                continue
            
            moments = []
            
            # Process sequences with stride=10 (faster, was 5)
            for i in range(0, len(subject_df) - 30, 10):  # Changed from 5 to 10
                seq = subject_df.iloc[i:i+30]
                if len(seq) != 30:
                    continue
                
                # Get timestamp (middle of sequence)
                mid_idx = i + 15
                if mid_idx >= len(subject_df):
                    continue
                    
                timestamp_str = subject_df.iloc[mid_idx]['timestamp']
                hour = subject_df.iloc[mid_idx]['hour']
                
                # Parse timestamp to get minute
                try:
                    from datetime import datetime
                    dt = datetime.strptime(str(timestamp_str), '%Y-%m-%d_%H-%M-%S')
                    minute = dt.minute
                except:
                    minute = 0
                
                # Only process daytime hours (6:30-19:30)
                if hour < 6 or (hour == 6 and minute < 30):
                    continue
                if hour >= 20 or (hour == 19 and minute >= 30):
                    continue
                
                # Convert basic_category to numeric IDs
                activity_ids_seq = [category_map.get(cat, 4) for cat in seq['basic_category'].values]
                
                # Prepare input - use numpy array conversion (faster)
                activity_ids = torch.from_numpy(np.array([activity_ids_seq], dtype=np.int64)).to(device)
                hours = torch.from_numpy(np.array([seq['hour'].values], dtype=np.float32)).to(device)
                
                # Forward pass - get encodings only
                encodings = model(activity_ids, hours, return_encodings_only=True)
                
                # Compute anomaly score for each dimension
                dim_scores = []
                dim_names = ['circadian', 'task', 'movement', 'social']
                enc_keys = ['h_c', 'h_t', 'h_m', 'h_s']
                
                for dim, enc_key in zip(dim_names, enc_keys):
                    enc = encodings[enc_key].cpu().numpy()[0]
                    mean = baseline[f'{dim}_mean']
                    std = baseline[f'{dim}_std']
                    
                    # Euclidean distance normalized by std
                    score = np.linalg.norm(enc - mean) / (np.linalg.norm(std) + 1e-6)
                    dim_scores.append(score)
                
                # Weighted combination
                combined_score = sum(a * s for a, s in zip(alpha_weights, dim_scores))
                
                # Check if anomalous
                if combined_score > threshold_multiplier:
                    # Convert timestamp to unix timestamp for compatibility
                    unix_timestamp = dt.timestamp()
                    
                    moments.append({
                        'timestamp': unix_timestamp,
                        'score': combined_score,
                        'hour': hour,
                        'minute': minute
                    })
            
            if len(moments) > 0:
                subject_moments[subject_id] = {
                    'label': label,
                    'moments': moments
                }
    
    print(f"✓ Detected anomalous moments for {len(subject_moments)} subjects")
    
    # Save intermediate results
    print("💾 Saving intermediate results...")
    os.makedirs('outputs', exist_ok=True)
    
    # Save subject_moments as pickle
    with open('outputs/subject_moments.pkl', 'wb') as f:
        pickle.dump(subject_moments, f)
    print(f"  ✓ Saved: outputs/subject_moments.pkl")
    
    # Save summary as JSON (for easy inspection)
    summary = {}
    for subject_id, data in subject_moments.items():
        summary[subject_id] = {
            'label': data['label'],
            'num_anomalies': len(data['moments'])
        }
    with open('outputs/subject_moments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: outputs/subject_moments_summary.json")
    
    return subject_moments

def create_weekly_heatmap_daytime(subject_moments, output_dir):
    """Create weekly temporal pattern heatmap (6:30-19:30 only)"""
    print("🎨 Creating weekly pattern visualization (daytime 6:30-19:30)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate CN and CI subjects with their anomaly counts
    cn_subjects = []
    ci_subjects = []
    
    for subject_id, data in subject_moments.items():
        moments = data['moments']
        if data['label'] == 'CN':
            cn_subjects.append((subject_id, len(moments), moments))
        elif data['label'] == 'CI':
            ci_subjects.append((subject_id, len(moments), moments))
    
    # Sort by anomaly count (descending)
    cn_subjects.sort(key=lambda x: x[1], reverse=True)
    ci_subjects.sort(key=lambda x: x[1], reverse=True)
    
    # Select equal number of top subjects from each group
    n_subjects = min(len(cn_subjects), len(ci_subjects))
    cn_selected = cn_subjects[:n_subjects]
    ci_selected = ci_subjects[:n_subjects]
    
    # Collect moments from selected subjects
    cn_moments = []
    ci_moments = []
    
    for subject_id, count, moments in cn_selected:
        cn_moments.extend(moments)
    
    for subject_id, count, moments in ci_selected:
        ci_moments.extend(moments)
    
    cn_count = len(cn_selected)
    ci_count = len(ci_selected)
    
    print(f"  Selected top {n_subjects} subjects from each group:")
    print(f"  CN: {cn_count} subjects, {len(cn_moments)} anomalous moments")
    print(f"  CI: {ci_count} subjects, {len(ci_moments)} anomalous moments")
    
    # Define daytime hours with 30-min resolution
    # 6:30 = 6.5, 19:30 = 19.5, range: [6.5, 7.0, 7.5, ..., 19.0, 19.5]
    hour_bins = np.arange(6.5, 20.0, 0.5)  # 27 bins
    hour_labels = []
    for h in hour_bins:
        hour_int = int(h)
        minute = 0 if h == hour_int else 30
        hour_labels.append(f"{hour_int:02d}:{minute:02d}")
    
    # Create grids: rows=hours (27), cols=days (7)
    cn_grid = np.zeros((len(hour_bins), 7))
    ci_grid = np.zeros((len(hour_bins), 7))
    
    # Fill grids
    for moment in cn_moments:
        dt = datetime.fromtimestamp(moment['timestamp'])
        hour_decimal = dt.hour + dt.minute / 60.0
        day = dt.weekday()
        
        # Find bin
        bin_idx = np.searchsorted(hour_bins, hour_decimal, side='right') - 1
        if 0 <= bin_idx < len(hour_bins):
            cn_grid[bin_idx, day] += 1
    
    for moment in ci_moments:
        dt = datetime.fromtimestamp(moment['timestamp'])
        hour_decimal = dt.hour + dt.minute / 60.0
        day = dt.weekday()
        
        # Find bin
        bin_idx = np.searchsorted(hour_bins, hour_decimal, side='right') - 1
        if 0 <= bin_idx < len(hour_bins):
            ci_grid[bin_idx, day] += 1
    
    # Normalize by subject count
    if cn_count > 0:
        cn_grid /= cn_count
    if ci_count > 0:
        ci_grid /= ci_count
    
    # Save grid data for later re-plotting
    print("💾 Saving grid data for re-plotting...")
    grid_data = {
        'cn_grid': cn_grid,
        'ci_grid': ci_grid,
        'cn_count': cn_count,
        'ci_count': ci_count,
        'cn_moments_count': len(cn_moments),
        'ci_moments_count': len(ci_moments),
        'hour_bins': hour_bins,
        'hour_labels': hour_labels
    }
    np.save('outputs/weekly_grid_data.npy', grid_data)
    print(f"  ✓ Saved: outputs/weekly_grid_data.npy")
    print(f"  💡 Tip: You can reload this data to re-plot with different styles!")
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.35)
    
    # Define teal colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#ffffff', '#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', 
              '#26a69a', '#009688', '#00796b', '#004d40', '#00251a']
    cmap = LinearSegmentedColormap.from_list('teal', colors)
    
    # Set consistent vmax for comparison
    vmax = max(np.max(cn_grid), np.max(ci_grid))
    
    # CN heatmap
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(cn_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
    ax1.set_title('CN - Anomalous Moments per Subject (Daytime 6:30-19:30)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Day of Week', fontsize=12)
    ax1.set_ylabel('Time', fontsize=12)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
    ax1.set_yticks(range(0, len(hour_bins), 4))
    ax1.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
    plt.colorbar(im1, ax=ax1, label='Moments/Subject', pad=0.02)
    
    # CI heatmap
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(ci_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
    ax2.set_title('CI - Anomalous Moments per Subject (Daytime 6:30-19:30)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Time', fontsize=12)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
    ax2.set_yticks(range(0, len(hour_bins), 4))
    ax2.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
    plt.colorbar(im2, ax=ax2, label='Moments/Subject', pad=0.02)
    
    # Difference plot (hourly average across week)
    diff = ci_grid - cn_grid
    avg_diff_per_hour = np.mean(diff, axis=1)
    
    ax3 = fig.add_subplot(gs[2])
    bars = ax3.barh(range(len(avg_diff_per_hour)), avg_diff_per_hour, 
                    color=['#e74c3c' if x > 0 else '#2ecc71' for x in avg_diff_per_hour],
                    edgecolor='#34495e', linewidth=0.5)
    ax3.set_xlabel('Difference (CI - CN) in Anomalous Moments/Hour', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Time', fontsize=11)
    ax3.set_title('CI vs CN Difference', fontsize=12, fontweight='bold', pad=10)
    ax3.set_yticks(range(0, len(hour_bins), 4))
    ax3.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add time period shading
    # Morning: 6:30-12:00 (bins 0-11)
    # Afternoon: 12:00-18:00 (bins 11-23)
    # Evening: 18:00-19:30 (bins 23-27)
    ax3.axhspan(-0.5, 10.5, color='#ff6f00', alpha=0.08, zorder=0, label='Morning')
    ax3.axhspan(10.5, 22.5, color='#2e7d32', alpha=0.08, zorder=0, label='Afternoon')
    ax3.axhspan(22.5, len(hour_bins)-0.5, color='#5e35b1', alpha=0.08, zorder=0, label='Evening')
    
    plt.suptitle('Weekly Temporal Pattern of Anomalous Behavioral Moments', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = os.path.join(output_dir, 'weekly_anomaly_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {output_path}")
    
    output_path_pdf = os.path.join(output_dir, 'weekly_anomaly_patterns.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ Saved to {output_path_pdf}")
    
    plt.close()
    
    # Print statistics
    print("\n📊 Statistical Summary:")
    print(f"【Sample】")
    print(f"  CN: {cn_count} subjects")
    print(f"  CI: {ci_count} subjects")
    print(f"\n【Anomalous Moments (Daytime 6:30-19:30)】")
    print(f"  CN total: {len(cn_moments)} moments")
    print(f"  CI total: {len(ci_moments)} moments")
    print(f"  CN avg per subject: {len(cn_moments)/cn_count:.1f}" if cn_count > 0 else "  CN avg per subject: N/A")
    print(f"  CI avg per subject: {len(ci_moments)/ci_count:.1f}" if ci_count > 0 else "  CI avg per subject: N/A")
    
    if cn_count > 0 and ci_count > 0:
        print(f"  CI/CN ratio: {(len(ci_moments)/ci_count)/(len(cn_moments)/cn_count):.2f}×")
    
    print(f"\n【Hourly Average】")
    if cn_count > 0 and ci_count > 0:
        cn_hourly = np.mean(cn_grid)
        ci_hourly = np.mean(ci_grid)
        print(f"  CN: {cn_hourly:.3f} moments/hour/subject")
        print(f"  CI: {ci_hourly:.3f} moments/hour/subject")
        print(f"  Ratio: {ci_hourly/cn_hourly:.2f}×" if cn_hourly > 0 else "  Ratio: N/A")
    else:
        print("  Not enough data to compute")
    
    # Peak hours
    if cn_count > 0 and ci_count > 0:
        cn_avg_by_hour = np.mean(cn_grid, axis=1)
        ci_avg_by_hour = np.mean(ci_grid, axis=1)
        
        cn_peak_idx = np.argmax(cn_avg_by_hour)
        ci_peak_idx = np.argmax(ci_avg_by_hour)
        
        print(f"\n【Peak Hours】")
        print(f"  CN peak: {hour_labels[cn_peak_idx]} ({cn_avg_by_hour[cn_peak_idx]:.3f} moments)")
        print(f"  CI peak: {hour_labels[ci_peak_idx]} ({ci_avg_by_hour[ci_peak_idx]:.3f} moments)")
    
    return {
        'cn_count': cn_count,
        'ci_count': ci_count,
        'cn_moments': len(cn_moments),
        'ci_moments': len(ci_moments),
        'cn_grid': cn_grid,
        'ci_grid': ci_grid
    }

def main():
    print("=" * 80)
    print("VERSION 5 - EXPERIMENT 2: WEEKLY ANOMALY PATTERN")
    print("=" * 80)
    
    # Load config
    config = load_config()
    print(f"\n⚙️  Configuration:")
    print(f"  Model: {config['model_path']}")
    print(f"  Alpha weights: {config['alpha_weights']}")
    print(f"  Threshold multiplier: {config['threshold_multiplier']}")
    print(f"  Random seed: {config['random_seed']}")
    
    # Set random seed
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    # Load data
    dataset, label_df = load_data()
    
    # Load model
    model, device = load_model(config['model_path'])
    
    # Get all subjects and normalize IDs
    def normalize_id(subject_id):
        """Normalize subject ID to lowercase without dashes"""
        return subject_id.lower().replace('-', '')
    
    # Create ID mapping
    dataset_subjects_raw = list(dataset['subjects'].keys())
    dataset_id_map = {normalize_id(s): s for s in dataset_subjects_raw}
    
    # Get label mapping with normalized IDs
    label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
    
    # Find matching subjects
    cn_subjects_raw = []
    ci_subjects_raw = []
    
    for norm_id, raw_id in dataset_id_map.items():
        label_row = label_df[label_df['normalized_id'] == norm_id]
        if len(label_row) > 0:
            label = label_row['label'].values[0]
            if label == 'CN':
                cn_subjects_raw.append(raw_id)
            elif label == 'CI':
                ci_subjects_raw.append(raw_id)
    
    print(f"\n📋 Subject Distribution:")
    print(f"  CN: {len(cn_subjects_raw)} subjects")
    print(f"  CI: {len(ci_subjects_raw)} subjects")
    
    # Compute CN baseline
    baseline = compute_cn_baseline(model, device, cn_subjects_raw, dataset, config['alpha_weights'])
    
    # Detect anomalous moments for all subjects
    all_test_subjects = cn_subjects_raw + ci_subjects_raw
    
    # Create normalized label mapping for detection function
    label_map_norm = {}
    for raw_id in all_test_subjects:
        norm_id = normalize_id(raw_id)
        label_row = label_df[label_df['normalized_id'] == norm_id]
        if len(label_row) > 0:
            label_map_norm[raw_id] = label_row['label'].values[0]
    
    # Create a minimal label dataframe for compatibility
    label_df_for_detect = pd.DataFrame({
        'subject_id': list(label_map_norm.keys()),
        'label': list(label_map_norm.values())
    })
    
    subject_moments = detect_anomalous_moments(
        model, device, baseline, all_test_subjects, label_df_for_detect, dataset,
        config['alpha_weights'], config['threshold_multiplier']
    )
    
    # Create visualization
    output_dir = 'outputs'
    stats = create_weekly_heatmap_daytime(subject_moments, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - outputs/weekly_anomaly_patterns.png")
    print(f"  - outputs/weekly_anomaly_patterns.pdf")
    print("=" * 80)

if __name__ == '__main__':
    main()
