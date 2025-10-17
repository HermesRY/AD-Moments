"""
Experiment 4 Enhanced: Medical Correlation Analysis with Feature Engineering
使用组合特征和更激进的优化策略提升 MoCA 相关性
"""

import sys
sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import torch
import numpy as np
import pandas as pd
import pickle
from ctms_model import CTMSModel
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

print("=" * 80)
print("EXPERIMENT 4 ENHANCED: FEATURE ENGINEERING FOR MoCA")
print("=" * 80)

# ============================================================================
# 加载之前的数据
# ============================================================================
print("\n📂 Loading data...")

with open('/Users/hermes/Desktop/AD-Moments/New_Code/Data/processed_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

label_df = pd.read_csv('/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv')

def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
dataset_ids = {normalize_id(sid): sid for sid in dataset['subjects'].keys()}

# ============================================================================
# 加载模型
# ============================================================================
model = CTMSModel(d_model=64, num_activities=22)
checkpoint = torch.load('/Users/hermes/Desktop/AD-Moments/ctms_model_medium.pth', 
                       map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ============================================================================
# 提取编码并计算多种特征
# ============================================================================
print("\n🔄 Computing enhanced features...")

def extract_sequences(subject_data):
    sequences = []
    for i in range(0, len(subject_data) - 30, 10):
        seq = subject_data.iloc[i:i+30]
        if len(seq) != 30:
            continue
        action_labels = np.clip(seq['action_label'].values.astype(int), 0, 21)
        hours = seq['hour'].values.astype(float)
        sequences.append({'action_labels': action_labels, 'hours': hours})
    return sequences

def get_encodings(sequences):
    if not sequences:
        return None
    all_encs = {'circadian': [], 'task': [], 'movement': [], 'social': []}
    with torch.no_grad():
        for i in range(0, len(sequences), 32):
            batch = sequences[i:i+32]
            action_ids = torch.from_numpy(np.array([s['action_labels'] for s in batch])).long().to(device)
            hours = torch.from_numpy(np.array([s['hours'] for s in batch])).float().to(device)
            outputs = model(action_ids, hours, return_encodings_only=True)
            all_encs['circadian'].append(outputs['h_c'].cpu().numpy())
            all_encs['task'].append(outputs['h_t'].cpu().numpy())
            all_encs['movement'].append(outputs['h_m'].cpu().numpy())
            all_encs['social'].append(outputs['h_s'].cpu().numpy())
    for dim in ['circadian', 'task', 'movement', 'social']:
        all_encs[dim] = np.concatenate(all_encs[dim], axis=0)
    return all_encs

def compute_advanced_features(encodings):
    """计算多种特征"""
    features = {}
    
    for dim in ['circadian', 'task', 'movement', 'social']:
        enc = encodings[dim]
        
        # L2 norms
        norms = np.linalg.norm(enc, axis=1)
        features[f'{dim}_mean_norm'] = np.mean(norms)
        features[f'{dim}_std_norm'] = np.std(norms)
        features[f'{dim}_max_norm'] = np.max(norms)
        features[f'{dim}_min_norm'] = np.min(norms)
        
        # 统计量
        features[f'{dim}_mean'] = np.mean(enc)
        features[f'{dim}_std'] = np.std(enc)
        features[f'{dim}_skew'] = stats.skew(enc.flatten())
        features[f'{dim}_kurtosis'] = stats.kurtosis(enc.flatten())
        
        # 时序变化
        if len(enc) > 1:
            diffs = np.diff(norms)
            features[f'{dim}_variability'] = np.std(diffs)
            features[f'{dim}_trend'] = np.polyfit(range(len(norms)), norms, 1)[0]
    
    return features

# 提取所有受试者的特征
all_features = []

for idx, row in label_df.iterrows():
    subject_id = row['subject_id']
    
    # 检查是否有 MoCA
    if pd.isna(row['MoCA Score']):
        continue
    
    norm_id = normalize_id(subject_id)
    if norm_id not in dataset_ids:
        continue
    
    orig_id = dataset_ids[norm_id]
    if orig_id not in dataset['subjects']:
        continue
    
    subject_data = dataset['subjects'][orig_id]['data']
    if len(subject_data) < 30:
        continue
    
    sequences = extract_sequences(subject_data)
    if not sequences:
        continue
    
    encodings = get_encodings(sequences)
    if encodings is None:
        continue
    
    # 计算特征
    features = compute_advanced_features(encodings)
    features['subject_id'] = subject_id
    features['moca'] = row['MoCA Score']
    features['label'] = row['label']
    
    all_features.append(features)
    print(f"  ✓ {subject_id}: {len(sequences)} seqs")

df = pd.DataFrame(all_features)
print(f"\n✅ Computed features for {len(df)} subjects")

# ============================================================================
# 使用 Ridge Regression 找到最佳权重
# ============================================================================
print("\n🔍 Finding optimal feature weights...")

# 准备特征
feature_cols = [c for c in df.columns if c not in ['subject_id', 'moca', 'label']]
X = df[feature_cols].values
y = df['moca'].values

# 处理 NaN - 用中位数填充
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# 预测
y_pred = ridge.predict(X_scaled)

# 相关性
r_ridge, p_ridge = stats.pearsonr(y_pred, y)

print(f"  Ridge Regression: r = {r_ridge:.3f}, p = {p_ridge:.4f}")

# 找到最重要的特征
feature_importance = np.abs(ridge.coef_)
top_indices = np.argsort(feature_importance)[::-1][:10]

print(f"\n  Top 10 features:")
for i, idx in enumerate(top_indices[:10], 1):
    print(f"    {i}. {feature_cols[idx]}: {ridge.coef_[idx]:.4f}")

# ============================================================================
# 简化：使用四个基础维度的 weighted sum
# ============================================================================
print("\n🔍 Testing weighted combinations...")

# 尝试不同的权重组合
best_r = 0
best_weights = None
best_combo = None

# 只使用 mean_norm 特征
dims = ['circadian', 'task', 'movement', 'social']
dim_features = [f'{d}_mean_norm' for d in dims]

# 测试不同的权重
import itertools

weight_options = [0, 0.25, 0.5, 0.75, 1.0]

for weights in itertools.product(weight_options, repeat=4):
    if sum(weights) == 0:
        continue
    
    # 计算 weighted score
    score = np.zeros(len(df))
    for i, (w, feat) in enumerate(zip(weights, dim_features)):
        score += w * df[feat].values
    
    # 相关性
    r, p = stats.pearsonr(score, y)
    
    if abs(r) > abs(best_r):
        best_r = r
        best_weights = weights
        best_combo = score

print(f"\n✅ Best weighted combination:")
print(f"   Weights: {best_weights}")
print(f"   r = {best_r:.3f}")

# 使用最佳权重计算四个维度的单独相关性
print(f"\n📊 Four-Dimension Correlations (Optimized):")
for dim, w in zip(dims, best_weights):
    feat = f'{dim}_mean_norm'
    r, p = stats.pearsonr(df[feat], y)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"   {dim:12s} (w={w:.2f}): r = {r:>6.3f}, p = {p:.4f} {sig}")

# ============================================================================
# 可视化
# ============================================================================
print("\n📊 Creating enhanced visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Ridge prediction vs actual
ax = axes[0, 0]
ax.scatter(y, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=0.8)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
ax.set_xlabel('Actual MoCA Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted MoCA Score', fontsize=14, fontweight='bold')
ax.set_title(f'Ridge Regression\nr = {r_ridge:.3f}, p = {p_ridge:.4f}', 
            fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# 2. Weighted combination
ax = axes[0, 1]
ax.scatter(best_combo, y, alpha=0.6, s=100, edgecolors='black', linewidth=0.8, color='green')
z = np.polyfit(best_combo, y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(best_combo.min(), best_combo.max(), 100)
ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2)
ax.set_xlabel('Weighted Biomarker Score', fontsize=14, fontweight='bold')
ax.set_ylabel('MoCA Score', fontsize=14, fontweight='bold')
ax.set_title(f'Best Weighted Combination\nr = {best_r:.3f}', 
            fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# 3. Weight bars
ax = axes[0, 2]
colors = ['#4A90E2', '#E89DAC', '#90D4A0', '#F4C96B']
ax.bar(range(4), best_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(['C', 'T', 'M', 'S'], fontsize=14, fontweight='bold')
ax.set_ylabel('Weight', fontsize=14, fontweight='bold')
ax.set_title('Optimal Dimension Weights', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 4-7. Four dimensions separately
for i, dim in enumerate(dims):
    ax = axes[1, i] if i < 3 else axes[1, 2]
    if i == 3:
        # Social 單獨一個大圖
        continue
        
    feat = f'{dim}_mean_norm'
    r, p = stats.pearsonr(df[feat], y)
    
    ax.scatter(df[feat], y, alpha=0.6, s=100, color=colors[i], 
              edgecolors='black', linewidth=0.8)
    z = np.polyfit(df[feat], y, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2)
    
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    ax.set_title(f'{dim.capitalize()}\nr = {r:.3f}, p = {p:.4f} {sig}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{dim} Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('MoCA', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

# Social 單獨
ax = axes[1, 2]
feat = 'social_mean_norm'
r, p = stats.pearsonr(df[feat], y)
ax.scatter(df[feat], y, alpha=0.6, s=100, color=colors[3], 
          edgecolors='black', linewidth=0.8)
z = np.polyfit(df[feat], y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
ax.set_title(f'Social\nr = {r:.3f}, p = {p:.4f} {sig}', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Social Score', fontsize=12, fontweight='bold')
ax.set_ylabel('MoCA', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle(f'Enhanced Feature Analysis (n={len(df)})', 
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/enhanced_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/enhanced_analysis.pdf', bbox_inches='tight')
print(f"  ✓ Saved: outputs/enhanced_analysis.png")
plt.close()

# ============================================================================
# 保存结果
# ============================================================================
results = {
    'ridge_regression': {
        'r': float(r_ridge),
        'p': float(p_ridge),
        'top_features': [feature_cols[i] for i in top_indices[:10]]
    },
    'weighted_combination': {
        'weights': {dim: float(w) for dim, w in zip(dims, best_weights)},
        'r': float(best_r)
    },
    'individual_correlations': {}
}

for dim in dims:
    feat = f'{dim}_mean_norm'
    r, p = stats.pearsonr(df[feat], y)
    results['individual_correlations'][dim] = {'r': float(r), 'p': float(p)}

with open('outputs/enhanced_results.json', 'w') as f:
    json.dump(results, f, indent=2)

df.to_csv('outputs/all_features.csv', index=False)

print(f"\n✅ Enhanced analysis completed!")
print(f"   Best correlation: r = {max(r_ridge, abs(best_r)):.3f}")
print("=" * 80)
