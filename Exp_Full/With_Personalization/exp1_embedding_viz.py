"""
Exp1: CTMS Embedding Visualization (With Personalization)
Generates UMAP plots of CTMS embeddings to visualize CN vs CI separation.

Usage:
    python exp1_embedding_viz.py

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
from sklearn.manifold import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
from models.ctms import CTMSModel

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Split configuration (70/30 for With Personalization)
DROP_SUBJECTS = {'CN': [18, 42, 43], 'CI': []}
SPLIT_SEED = 1  # Different seed
TRAIN_RATIO = 0.7

# Model configuration - can be different from Without Personalization
ALPHA = [0.7, 0.15, 0.1, 0.05]  # Circadian-heavy for personalization
SEQ_LEN = 30
STRIDE = 10
PERSONAL_RATIO = 0.2  # Use first 20% for personal baseline

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    """Load and filter dataset."""
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
    return data

def create_windows(time_series, seq_len, stride):
    """Create sliding windows."""
    windows = []
    for i in range(0, len(time_series) - seq_len + 1, stride):
        windows.append(time_series[i:i+seq_len])
    return windows

def compute_personal_baseline(subject, model):
    """Compute personalized baseline from first 20% of subject's data."""
    circ_windows = create_windows(subject['circadian_rhythm'], SEQ_LEN, STRIDE)
    task_windows = create_windows(subject['task_completion'], SEQ_LEN, STRIDE)
    move_windows = create_windows(subject['movement_patterns'], SEQ_LEN, STRIDE)
    soc_windows = create_windows(subject['social_interactions'], SEQ_LEN, STRIDE)
    
    n_windows = len(circ_windows)
    n_baseline = max(1, int(n_windows * PERSONAL_RATIO))
    
    # Use first 20% for baseline
    circ_baseline = torch.FloatTensor(circ_windows[:n_baseline]).to(DEVICE)
    task_baseline = torch.FloatTensor(task_windows[:n_baseline]).to(DEVICE)
    move_baseline = torch.FloatTensor(move_windows[:n_baseline]).to(DEVICE)
    soc_baseline = torch.FloatTensor(soc_windows[:n_baseline]).to(DEVICE)
    
    with torch.no_grad():
        circ_emb = model.circadian_encoder(circ_baseline)
        task_emb = model.task_encoder(task_baseline)
        move_emb = model.movement_encoder(move_baseline)
        soc_emb = model.social_encoder(soc_baseline)
        
        circ_norms = torch.norm(circ_emb, dim=1).cpu().numpy()
        task_norms = torch.norm(task_emb, dim=1).cpu().numpy()
        move_norms = torch.norm(move_emb, dim=1).cpu().numpy()
        soc_norms = torch.norm(soc_emb, dim=1).cpu().numpy()
    
    return {
        'circadian': {'mean': np.mean(circ_norms), 'std': np.std(circ_norms)},
        'task': {'mean': np.mean(task_norms), 'std': np.std(task_norms)},
        'movement': {'mean': np.mean(move_norms), 'std': np.std(move_norms)},
        'social': {'mean': np.mean(soc_norms), 'std': np.std(soc_norms)}
    }

def encode_subjects_personalized(data, model):
    """Encode subjects using personalized baselines."""
    embeddings_list = []
    labels_list = []
    subject_ids = []
    
    for subject in tqdm(data, desc="Encoding subjects (personalized)"):
        # Compute personal baseline
        personal_baseline = compute_personal_baseline(subject, model)
        
        # Prepare full sequence
        circadian = torch.FloatTensor(subject['circadian_rhythm']).unsqueeze(0).to(DEVICE)
        task = torch.FloatTensor(subject['task_completion']).unsqueeze(0).to(DEVICE)
        movement = torch.FloatTensor(subject['movement_patterns']).unsqueeze(0).to(DEVICE)
        social = torch.FloatTensor(subject['social_interactions']).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Get embeddings
            circ_emb = model.circadian_encoder(circadian).cpu().numpy()
            task_emb = model.task_encoder(task).cpu().numpy()
            move_emb = model.movement_encoder(movement).cpu().numpy()
            soc_emb = model.social_encoder(social).cpu().numpy()
            
            # Compute personalized z-scores
            circ_z = abs((np.linalg.norm(circ_emb) - personal_baseline['circadian']['mean']) / 
                        (personal_baseline['circadian']['std'] + 1e-8))
            task_z = abs((np.linalg.norm(task_emb) - personal_baseline['task']['mean']) / 
                        (personal_baseline['task']['std'] + 1e-8))
            move_z = abs((np.linalg.norm(move_emb) - personal_baseline['movement']['mean']) / 
                        (personal_baseline['movement']['std'] + 1e-8))
            soc_z = abs((np.linalg.norm(soc_emb) - personal_baseline['social']['mean']) / 
                       (personal_baseline['social']['std'] + 1e-8))
            
            # Weighted combination
            embedding = np.array([circ_z, task_z, move_z, soc_z])
            
            embeddings_list.append(embedding)
            labels_list.append(1 if subject['label'] == 'CI' else 0)
            subject_ids.append(subject['subject_id'])
    
    return np.array(embeddings_list), np.array(labels_list), subject_ids

def plot_umap(embeddings, labels, subject_ids, save_path):
    """Generate UMAP visualization without individual markers."""
    # Fit UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Calculate metrics
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    for label, color, name in [(0, '#2E86AB', 'CN'), (1, '#A23B72', 'CI')]:
        mask = labels == label
        points = embedding_2d[mask]
        
        # Plot density contours
        from scipy.stats import gaussian_kde
        if len(points) > 3:
            xy = points.T
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = xy[0][idx], xy[1][idx], z[idx]
            plt.scatter(x, y, c=z, s=50, alpha=0.6, cmap='viridis' if label == 0 else 'plasma', 
                       label=f'{name} (n={len(points)})')
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(f'CTMS Embedding Space (With Personalization)\nSilhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'n_cn': int((labels == 0).sum()),
        'n_ci': int((labels == 1).sum())
    }

def main():
    print("="*80)
    print("EXP1: CTMS EMBEDDING VISUALIZATION (WITH PERSONALIZATION)")
    print("="*80)
    
    # Load data
    print("\n1. Loading dataset...")
    data = load_dataset()
    print(f"   Total subjects: {len(data)}")
    cn_count = sum(1 for s in data if s['label'] == 'CN')
    ci_count = sum(1 for s in data if s['label'] == 'CI')
    print(f"   CN: {cn_count}, CI: {ci_count}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=64, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    print(f"   Alpha weights: {ALPHA}")
    print(f"   Personal baseline: first {int(PERSONAL_RATIO*100)}% of data")
    
    # Encode subjects with personalization
    print("\n3. Encoding subjects with personalized baselines...")
    embeddings, labels, subject_ids = encode_subjects_personalized(data, model)
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Generate UMAP visualization
    print("\n4. Generating UMAP visualization...")
    save_path = OUTPUT_DIR / 'exp1_umap_embedding.png'
    metrics = plot_umap(embeddings, labels, subject_ids, str(save_path))
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    print(f"CN subjects: {metrics['n_cn']}")
    print(f"CI subjects: {metrics['n_ci']}")
    print(f"\nVisualization saved to: {save_path}")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / 'exp1_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    print("="*80)

if __name__ == '__main__':
    main()
