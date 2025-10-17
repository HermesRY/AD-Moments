"""
Experiment 3: Classification Analysis  
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Import CTMS model
from models.ctms_model import CTMSModel

print("=" * 80)
print("EXPERIMENT 3: CLASSIFICATION ANALYSIS")
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

# Classification config
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

print(f"\n📂 Configuration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Data: {DATA_PATH}")
print(f"  Test size: {TEST_SIZE}")
print(f"  CV folds: {CV_FOLDS}")

# ============================================================================
# Load Data & Model
# ============================================================================
print(f"\n📂 Loading data...")

with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

label_df = pd.read_csv(LABELS_PATH)

print(f"  ✅ Loaded {len(dataset['subjects'])} subjects")

print(f"\n🔧 Loading model...")
model = CTMSModel(d_model=D_MODEL, num_activities=NUM_ACTIVITIES)
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
device = torch.device('cpu')
model = model.to(device)
print(f"  ✅ Model loaded")

# ============================================================================
# Extract Features
# ============================================================================
def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

def extract_sequences(subject_data, seq_len=30, stride=10):
    sequences = []
    for i in range(0, len(subject_data) - seq_len, stride):
        seq = subject_data.iloc[i:i+seq_len]
        if len(seq) != seq_len:
            continue
        action_labels = np.clip(seq['action_label'].values.astype(int), 0, NUM_ACTIVITIES-1)
        hours = seq['hour'].values.astype(float)
        sequences.append({'action_labels': action_labels, 'hours': hours})
    return sequences

def get_mean_encoding(sequences, model, device, batch_size=32):
    if not sequences:
        return None
    
    all_encs = {'circadian': [], 'task': [], 'movement': [], 'social': []}
    
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
    
    # Compute mean magnitude per dimension
    features = []
    for dim in ['circadian', 'task', 'movement', 'social']:
        enc = np.concatenate(all_encs[dim], axis=0)
        features.append(np.mean(np.linalg.norm(enc, axis=1)))
    
    return np.array(features)

print(f"\n🔄 Extracting features...")

label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
dataset_ids = {normalize_id(sid): sid for sid in dataset['subjects'].keys()}

X = []
y = []
subject_ids = []

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
    
    sequences = extract_sequences(subject_data, SEQ_LEN, STRIDE)
    features = get_mean_encoding(sequences, model, device, BATCH_SIZE)
    
    if features is not None:
        X.append(features)
        y.append(0 if label == 'CN' else 1)
        subject_ids.append(subject_id)
        print(f"  ✓ {subject_id} ({label})")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Extracted features for {len(X)} subjects")
print(f"  CN: {(y == 0).sum()}, CI: {(y == 1).sum()}")

# ============================================================================
# Train-Test Split
# ============================================================================
print(f"\n📊 Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")

# ============================================================================
# Train Classifiers
# ============================================================================
print(f"\n🤖 Training classifiers...")

# Define models
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
}

results = {}

for name, clf in classifiers.items():
    print(f"\n  Training {name}...")
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=CV_FOLDS, scoring='roc_auc')
    
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"    Accuracy: {acc:.3f}")
    print(f"    AUC: {auc:.3f}")
    print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================================
# Visualizations
# ============================================================================
print(f"\n🎨 Creating visualizations...")

# 1. Performance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Metrics bar plot
metrics_data = []
for name, res in results.items():
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        metrics_data.append({
            'Classifier': name,
            'Metric': metric.upper(),
            'Score': res[metric]
        })

df_metrics = pd.DataFrame(metrics_data)

ax = axes[0]
metrics_order = ['ACCURACY', 'PRECISION', 'RECALL', 'F1', 'AUC']
for i, metric in enumerate(metrics_order):
    metric_data = df_metrics[df_metrics['Metric'] == metric]
    x = np.arange(len(classifiers)) + i * (len(classifiers) + 1)
    ax.bar(x, metric_data['Score'], width=0.8, label=metric)

ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Classification Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks([])
ax.set_ylim([0, 1.0])
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)

# CV scores
ax = axes[1]
cv_means = [results[name]['cv_mean'] for name in classifiers.keys()]
cv_stds = [results[name]['cv_std'] for name in classifiers.keys()]

x = np.arange(len(classifiers))
ax.bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color=['#4A90E2', '#E89DAC', '#90D4A0'])
ax.set_xticks(x)
ax.set_xticklabels(classifiers.keys(), rotation=45, ha='right')
ax.set_ylabel(f'{CV_FOLDS}-Fold CV AUC', fontweight='bold')
ax.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'classification_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(OUTPUT_DIR, 'results_summary.csv'))

print(f"\n" + "=" * 80)
print(f"✅ EXPERIMENT 3 COMPLETED")
print(f"=" * 80)
print(f"\nResults Summary:")
print(results_df[['accuracy', 'precision', 'recall', 'f1', 'auc']].to_string())
print(f"\n" + "=" * 80)
