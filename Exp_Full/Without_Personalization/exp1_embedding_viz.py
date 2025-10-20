"""
Exp1: CTMS Embedding Visualization (Without Personalization)
Generates UMAP plots of CTMS embeddings to visualize CN vs CI separation.

Usage:
    python exp1_embedding_viz.py

Note:
    This script expects the CTMS model and dataset to be available in the parent directories.
    Adjust the paths below if your project structure differs.
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
try:
    from umap import UMAP
except ImportError:
    print("Warning: UMAP not installed. Install with: pip install umap-learn")
    UMAP = None
from sklearn.metrics import silhouette_score, davies_bouldin_score
from Model.ctms_model import CTMSModel

# Configuration
DATA_PATH = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Split configuration (80/20 for Without Personalization)
DROP_SUBJECTS = {'CN': [18, 42, 43], 'CI': []}
SPLIT_SEED = 0
TRAIN_RATIO = 0.8

# Model configuration
ALPHA = [0.5, 0.3, 0.1, 0.1]  # Exp2 optimal weights

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
            
            # Apply subject filtering
            if label == 'CN' and subject_id in DROP_SUBJECTS['CN']:
                continue
            if label == 'CI' and subject_id in DROP_SUBJECTS['CI']:
                continue
                
            data.append(subject)
    return data

def encode_subjects(data, model):
    """Encode all subjects and extract embeddings."""
    embeddings_list = []
    labels_list = []
    subject_ids = []
    
    for subject in tqdm(data, desc="Encoding subjects"):
        # Prepare input
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
            
            # Combine with alpha weights
            embedding = (ALPHA[0] * circ_emb + 
                        ALPHA[1] * task_emb + 
                        ALPHA[2] * move_emb + 
                        ALPHA[3] * soc_emb)
            
            embeddings_list.append(embedding.squeeze())
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
    
    # Plot without individual markers (violin-style density)
    for label, color, name in [(0, '#2E86AB', 'CN'), (1, '#A23B72', 'CI')]:
        mask = labels == label
        points = embedding_2d[mask]
        
        # Plot density contours instead of points
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
    plt.title(f'CTMS Embedding Space (Without Personalization)\nSilhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}', 
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
    print("EXP1: CTMS EMBEDDING VISUALIZATION (WITHOUT PERSONALIZATION)")
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
    
    # Encode subjects
    print("\n3. Encoding subjects...")
    embeddings, labels, subject_ids = encode_subjects(data, model)
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
