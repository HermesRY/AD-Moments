"""
Exp4: Medical Correlations (Without Personalization)
Correlates CTMS scores with medical assessment scores (MoCA, ZBI, FAS, DSS).
"""

import sys
sys.path.append('/home/heming/Desktop/AD-Moments-1/AD-Moments')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from models.ctms import CTMSModel

# Configuration
DATA_PATH = '/home/heming/Desktop/AD-Moments-1/AD-Moments/sample_data/dataset_one_month.jsonl'
OUTPUT_DIR = '/home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/Without_Personalization/outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DROP_SUBJECTS = {'CN': [18, 42, 43], 'CI': []}

# Model configuration
ALPHA = [0.5, 0.3, 0.1, 0.1]  # Use same alpha as other experiments
SEQ_LEN = 30
STRIDE = 10
BATCH_SIZE = 256

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ci_subjects():
    """Load only CI subjects with medical scores."""
    ci_subjects = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subject = json.loads(line)
            if subject['label'] == 'CI' and subject['subject_id'] not in DROP_SUBJECTS['CI']:
                # Check if medical scores exist in 'scores' field
                if 'scores' in subject:
                    scores = subject['scores']
                    if all(k in scores for k in ['moca', 'zbi', 'fas', 'dss']):
                        ci_subjects.append(subject)
    return ci_subjects

def create_windows(time_series, seq_len, stride):
    """Create sliding windows from time series."""
    windows = []
    for i in range(0, len(time_series) - seq_len + 1, stride):
        windows.append(time_series[i:i+seq_len])
    return windows

def compute_dimension_scores(subject, model):
    """Compute scores for each dimension separately."""
    circ_windows = create_windows(subject['circadian_rhythm'], SEQ_LEN, STRIDE)
    task_windows = create_windows(subject['task_completion'], SEQ_LEN, STRIDE)
    move_windows = create_windows(subject['movement_patterns'], SEQ_LEN, STRIDE)
    soc_windows = create_windows(subject['social_interactions'], SEQ_LEN, STRIDE)
    
    n_windows = len(circ_windows)
    
    circ_norms = []
    task_norms = []
    move_norms = []
    soc_norms = []
    
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
            
            circ_norms.extend(torch.norm(circ_emb, dim=1).cpu().numpy())
            task_norms.extend(torch.norm(task_emb, dim=1).cpu().numpy())
            move_norms.extend(torch.norm(move_emb, dim=1).cpu().numpy())
            soc_norms.extend(torch.norm(soc_emb, dim=1).cpu().numpy())
    
    # Return mean norms for each dimension
    return {
        'circadian': np.mean(circ_norms),
        'task': np.mean(task_norms),
        'movement': np.mean(move_norms),
        'social': np.mean(soc_norms)
    }

def analyze_correlations(ci_subjects, model):
    """Compute correlations between CTMS dimensions and medical scores."""
    # Collect data
    circadian_scores = []
    task_scores = []
    movement_scores = []
    social_scores = []
    
    moca_scores = []
    zbi_scores = []
    fas_scores = []
    dss_scores = []
    
    print("Computing dimension scores for CI subjects...")
    for subject in tqdm(ci_subjects):
        dim_scores = compute_dimension_scores(subject, model)
        
        circadian_scores.append(dim_scores['circadian'])
        task_scores.append(dim_scores['task'])
        movement_scores.append(dim_scores['movement'])
        social_scores.append(dim_scores['social'])
        
        scores = subject['scores']
        moca_scores.append(scores['moca'])
        zbi_scores.append(scores['zbi'])
        fas_scores.append(scores['fas'])
        dss_scores.append(scores['dss'])
    
    # Convert to arrays
    circadian_scores = np.array(circadian_scores)
    task_scores = np.array(task_scores)
    movement_scores = np.array(movement_scores)
    social_scores = np.array(social_scores)
    
    moca_scores = np.array(moca_scores)
    zbi_scores = np.array(zbi_scores)
    fas_scores = np.array(fas_scores)
    dss_scores = np.array(dss_scores)
    
    # Compute correlations
    correlations = {}
    
    for dim_name, dim_scores in [
        ('Circadian', circadian_scores),
        ('Task', task_scores),
        ('Movement', movement_scores),
        ('Social', social_scores)
    ]:
        correlations[dim_name] = {}
        
        for med_name, med_scores in [
            ('MoCA', moca_scores),
            ('ZBI', zbi_scores),
            ('FAS', fas_scores),
            ('DSS', dss_scores)
        ]:
            r, p = stats.pearsonr(dim_scores, med_scores)
            correlations[dim_name][med_name] = {
                'r': float(r),
                'p': float(p),
                'significant': p < 0.05
            }
    
    return correlations, {
        'circadian': circadian_scores,
        'task': task_scores,
        'movement': movement_scores,
        'social': social_scores,
        'moca': moca_scores,
        'zbi': zbi_scores,
        'fas': fas_scores,
        'dss': dss_scores
    }

def plot_correlations(correlations, data, save_path):
    """Plot correlation heatmap."""
    # Create correlation matrix
    dims = ['Circadian', 'Task', 'Movement', 'Social']
    meds = ['MoCA', 'ZBI', 'FAS', 'DSS']
    
    corr_matrix = np.zeros((len(dims), len(meds)))
    p_matrix = np.zeros((len(dims), len(meds)))
    
    for i, dim in enumerate(dims):
        for j, med in enumerate(meds):
            corr_matrix[i, j] = correlations[dim][med]['r']
            p_matrix[i, j] = correlations[dim][med]['p']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotations with significance stars
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(len(dims)):
        for j in range(len(meds)):
            r = corr_matrix[i, j]
            p = p_matrix[i, j]
            if p < 0.001:
                annot[i, j] = f'{r:.3f}***'
            elif p < 0.01:
                annot[i, j] = f'{r:.3f}**'
            elif p < 0.05:
                annot[i, j] = f'{r:.3f}*'
            else:
                annot[i, j] = f'{r:.3f}'
    
    sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='coolwarm', center=0,
                xticklabels=meds, yticklabels=dims, ax=ax,
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'})
    
    ax.set_title('CTMS-Medical Correlations (Without Personalization)\n* p<0.05, ** p<0.01, *** p<0.001',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Medical Assessment', fontsize=12)
    ax.set_ylabel('CTMS Dimension', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("EXP4: MEDICAL CORRELATIONS (WITHOUT PERSONALIZATION)")
    print("="*80)
    
    # Load CI subjects
    print("\n1. Loading CI subjects with medical scores...")
    ci_subjects = load_ci_subjects()
    print(f"   Total CI subjects: {len(ci_subjects)}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=64, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    print(f"   Alpha weights: {ALPHA}")
    
    # Analyze correlations
    print("\n3. Computing correlations...")
    correlations, data = analyze_correlations(ci_subjects, model)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for dim in ['Circadian', 'Task', 'Movement', 'Social']:
        print(f"\n{dim}:")
        for med in ['MoCA', 'ZBI', 'FAS', 'NPI']:
            r = correlations[dim][med]['r']
            p = correlations[dim][med]['p']
            sig = '*' if p < 0.05 else ''
            print(f"  vs {med}: r={r:.3f}, p={p:.4f} {sig}")
    
    # Plot
    print("\n4. Generating visualization...")
    save_path = os.path.join(OUTPUT_DIR, 'exp4_medical_correlations.png')
    plot_correlations(correlations, data, save_path)
    print(f"   Saved to: {save_path}")
    
    # Save correlations
    corr_path = os.path.join(OUTPUT_DIR, 'exp4_correlations.json')
    with open(corr_path, 'w') as f:
        json.dump(correlations, f, indent=2)
    print(f"   Correlations saved to: {corr_path}")
    print("="*80)

if __name__ == '__main__':
    main()
