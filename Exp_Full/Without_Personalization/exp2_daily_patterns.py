"""
Exp2: Daily Pattern Analysis (Without Personalization)
Analyzes circadian rhythm patterns and computes CI/CN ratio.

Usage:
    python exp2_daily_patterns.py

Note:
    This script expects the CTMS model and dataset to be available in the parent directories.
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from models.ctms import CTMSModel

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Split configuration (80/20)
DROP_SUBJECTS = {'CN': [18, 42, 43], 'CI': []}
SPLIT_SEED = 0
TRAIN_RATIO = 0.8

# Model configuration
ALPHA = [0.5, 0.3, 0.1, 0.1]  # Circadian-dominant for Exp2
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 256

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_split_dataset():
    """Load dataset and perform train/test split."""
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subject = json.loads(line)
            label = subject['label']
            subject_id = subject['subject_id']
            
            if label == 'CN' and subject_id in DROP_SUBJECTS['CN']:
                continue
            if label == 'CI' and subject_id in DROP_SUBJECTS['CI']:
                continue
                
            data.append(subject)
    
    # Separate by label
    cn_subjects = [s for s in data if s['label'] == 'CN']
    ci_subjects = [s for s in data if s['label'] == 'CI']
    
    # Split CN into train/test
    np.random.seed(SPLIT_SEED)
    n_cn = len(cn_subjects)
    n_train = int(n_cn * TRAIN_RATIO)
    
    cn_indices = np.random.permutation(n_cn)
    cn_train = [cn_subjects[i] for i in cn_indices[:n_train]]
    cn_test = [cn_subjects[i] for i in cn_indices[n_train:]]
    
    return cn_train, cn_test, ci_subjects

def create_windows(time_series, seq_len, stride):
    """Create sliding windows from time series."""
    windows = []
    for i in range(0, len(time_series) - seq_len + 1, stride):
        windows.append(time_series[i:i+seq_len])
    return windows

def compute_baseline(cn_train, model):
    """Compute baseline statistics from CN training set."""
    all_norms = {
        'circadian': [],
        'task': [],
        'movement': [],
        'social': []
    }
    
    for subject in tqdm(cn_train, desc="Computing baseline"):
        # Create windows
        circ_windows = create_windows(subject['circadian_rhythm'], SEQ_LEN, STRIDE)
        task_windows = create_windows(subject['task_completion'], SEQ_LEN, STRIDE)
        move_windows = create_windows(subject['movement_patterns'], SEQ_LEN, STRIDE)
        soc_windows = create_windows(subject['social_interactions'], SEQ_LEN, STRIDE)
        
        n_windows = len(circ_windows)
        
        for i in range(0, n_windows, BATCH_SIZE):
            batch_circ = torch.FloatTensor(circ_windows[i:i+BATCH_SIZE]).to(DEVICE)
            batch_task = torch.FloatTensor(task_windows[i:i+BATCH_SIZE]).to(DEVICE)
            batch_move = torch.FloatTensor(move_windows[i:i+BATCH_SIZE]).to(DEVICE)
            batch_soc = torch.FloatTensor(soc_windows[i:i+BATCH_SIZE]).to(DEVICE)
            
            with torch.no_grad():
                circ_emb = model.circadian_encoder(batch_circ)
                task_emb = model.task_encoder(batch_task)
                move_emb = model.movement_encoder(batch_move)
                soc_emb = model.social_encoder(batch_soc)
                
                # Compute norms
                all_norms['circadian'].extend(torch.norm(circ_emb, dim=1).cpu().numpy())
                all_norms['task'].extend(torch.norm(task_emb, dim=1).cpu().numpy())
                all_norms['movement'].extend(torch.norm(move_emb, dim=1).cpu().numpy())
                all_norms['social'].extend(torch.norm(soc_emb, dim=1).cpu().numpy())
    
    # Compute mean and std
    baseline_stats = {}
    for dim in all_norms:
        norms = np.array(all_norms[dim])
        baseline_stats[dim] = {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms))
        }
    
    return baseline_stats

def compute_subject_score(subject, model, baseline_stats):
    """Compute anomaly score for a subject."""
    # Create windows
    circ_windows = create_windows(subject['circadian_rhythm'], SEQ_LEN, STRIDE)
    task_windows = create_windows(subject['task_completion'], SEQ_LEN, STRIDE)
    move_windows = create_windows(subject['movement_patterns'], SEQ_LEN, STRIDE)
    soc_windows = create_windows(subject['social_interactions'], SEQ_LEN, STRIDE)
    
    n_windows = len(circ_windows)
    window_scores = []
    
    for i in range(0, n_windows, BATCH_SIZE):
        batch_circ = torch.FloatTensor(circ_windows[i:i+BATCH_SIZE]).to(DEVICE)
        batch_task = torch.FloatTensor(task_windows[i:i+BATCH_SIZE]).to(DEVICE)
        batch_move = torch.FloatTensor(move_windows[i:i+BATCH_SIZE]).to(DEVICE)
        batch_soc = torch.FloatTensor(soc_windows[i:i+BATCH_SIZE]).to(DEVICE)
        
        with torch.no_grad():
            circ_emb = model.circadian_encoder(batch_circ)
            task_emb = model.task_encoder(batch_task)
            move_emb = model.movement_encoder(batch_move)
            soc_emb = model.social_encoder(batch_soc)
            
            # Compute norms
            circ_norms = torch.norm(circ_emb, dim=1).cpu().numpy()
            task_norms = torch.norm(task_emb, dim=1).cpu().numpy()
            move_norms = torch.norm(move_emb, dim=1).cpu().numpy()
            soc_norms = torch.norm(soc_emb, dim=1).cpu().numpy()
            
            # Compute z-scores
            z_circ = np.abs((circ_norms - baseline_stats['circadian']['mean']) / baseline_stats['circadian']['std'])
            z_task = np.abs((task_norms - baseline_stats['task']['mean']) / baseline_stats['task']['std'])
            z_move = np.abs((move_norms - baseline_stats['movement']['mean']) / baseline_stats['movement']['std'])
            z_soc = np.abs((soc_norms - baseline_stats['social']['mean']) / baseline_stats['social']['std'])
            
            # Weighted sum
            scores = (ALPHA[0] * z_circ + 
                     ALPHA[1] * z_task + 
                     ALPHA[2] * z_move + 
                     ALPHA[3] * z_soc)
            
            window_scores.extend(scores)
    
    return np.array(window_scores)

def analyze_daily_patterns(cn_test, ci_subjects, model, baseline_stats):
    """Analyze daily patterns and compute CI/CN ratio."""
    cn_scores = []
    ci_scores = []
    
    print("Computing CN test scores...")
    for subject in tqdm(cn_test):
        scores = compute_subject_score(subject, model, baseline_stats)
        cn_scores.extend(scores)
    
    print("Computing CI scores...")
    for subject in tqdm(ci_subjects):
        scores = compute_subject_score(subject, model, baseline_stats)
        ci_scores.extend(scores)
    
    cn_scores = np.array(cn_scores)
    ci_scores = np.array(ci_scores)
    
    # Compute statistics
    cn_mean = np.mean(cn_scores)
    ci_mean = np.mean(ci_scores)
    ci_cn_ratio = ci_mean / cn_mean if cn_mean > 0 else 0
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(ci_scores, cn_scores)
    
    return {
        'cn_mean': float(cn_mean),
        'cn_std': float(np.std(cn_scores)),
        'ci_mean': float(ci_mean),
        'ci_std': float(np.std(ci_scores)),
        'ci_cn_ratio': float(ci_cn_ratio),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cn_scores': cn_scores,
        'ci_scores': ci_scores
    }

def plot_daily_patterns(results, save_path):
    """Plot daily pattern distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution plot
    axes[0].hist(results['cn_scores'], bins=50, alpha=0.6, label=f'CN (n={len(results["cn_scores"])})', color='#2E86AB', density=True)
    axes[0].hist(results['ci_scores'], bins=50, alpha=0.6, label=f'CI (n={len(results["ci_scores"])})', color='#A23B72', density=True)
    axes[0].axvline(results['cn_mean'], color='#2E86AB', linestyle='--', linewidth=2, label=f'CN mean: {results["cn_mean"]:.3f}')
    axes[0].axvline(results['ci_mean'], color='#A23B72', linestyle='--', linewidth=2, label=f'CI mean: {results["ci_mean"]:.3f}')
    axes[0].set_xlabel('Anomaly Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Score Distribution (Without Personalization)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [results['cn_scores'], results['ci_scores']]
    bp = axes[1].boxplot(data_to_plot, labels=['CN', 'CI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    axes[1].set_ylabel('Anomaly Score', fontsize=12)
    axes[1].set_title(f'CI/CN Ratio: {results["ci_cn_ratio"]:.3f}×\np-value: {results["p_value"]:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("EXP2: DAILY PATTERN ANALYSIS (WITHOUT PERSONALIZATION)")
    print("="*80)
    
    # Load and split data
    print("\n1. Loading and splitting dataset...")
    cn_train, cn_test, ci_subjects = load_and_split_dataset()
    print(f"   CN train: {len(cn_train)}, CN test: {len(cn_test)}, CI: {len(ci_subjects)}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=64, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    print(f"   Alpha weights: {ALPHA}")
    
    # Compute baseline
    print("\n3. Computing baseline from CN train...")
    baseline_stats = compute_baseline(cn_train, model)
    print("   Baseline statistics:")
    for dim, stats in baseline_stats.items():
        print(f"     {dim}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Analyze patterns
    print("\n4. Analyzing daily patterns...")
    results = analyze_daily_patterns(cn_test, ci_subjects, model, baseline_stats)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"CN mean score: {results['cn_mean']:.4f} ± {results['cn_std']:.4f}")
    print(f"CI mean score: {results['ci_mean']:.4f} ± {results['ci_std']:.4f}")
    print(f"CI/CN ratio: {results['ci_cn_ratio']:.4f}×")
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    
    # Plot
    print("\n5. Generating visualization...")
    save_path = OUTPUT_DIR / 'exp2_daily_patterns.png'
    plot_daily_patterns(results, str(save_path))
    print(f"   Saved to: {save_path}")
    
    # Save metrics
    metrics = {k: v for k, v in results.items() if k not in ['cn_scores', 'ci_scores']}
    metrics_path = OUTPUT_DIR / 'exp2_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to: {metrics_path}")
    print("="*80)

if __name__ == '__main__':
    main()
