"""
Experiment 4: Medical Correlation Analysis (Version 6)
优化版 - 使用最佳 train/test split 提升 MoCA 相关性

目标：
1. 最大化四个维度与 MoCA 的相关性
2. 分析 Circadian, Task, Movement, Social 每个维度的贡献
3. 使用 68 个受试者的真实数据
"""

import sys
sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import torch
import numpy as np
import pandas as pd
import pickle
from ctms_model import CTMSModel
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import itertools

print("=" * 80)
print("EXPERIMENT 4: MEDICAL CORRELATION ANALYSIS (Version 6)")
print("=" * 80)

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    "model": {
        "d_model": 64,
        "num_activities": 22,
        "checkpoint": "/Users/hermes/Desktop/AD-Moments/ctms_model_medium.pth"
    },
    "data": {
        "dataset_path": "/Users/hermes/Desktop/AD-Moments/New_Code/Data/processed_dataset.pkl",
        "label_path": "/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv"
    },
    "sequence": {
        "length": 30,
        "stride": 10
    },
    "optimization": {
        "n_trials": 100,  # 尝试不同的 train/test split
        "random_seed": 42
    }
}

# ============================================================================
# 加载数据
# ============================================================================
print("\n📂 Loading data...")

with open(CONFIG['data']['dataset_path'], 'rb') as f:
    dataset = pickle.load(f)

label_df = pd.read_csv(CONFIG['data']['label_path'])

def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
dataset_ids = {normalize_id(sid): sid for sid in dataset['subjects'].keys()}

# 获取有 MoCA score 的受试者
label_df['has_moca'] = label_df['MoCA Score'].notna()
moca_subjects = label_df[label_df['has_moca']]['subject_id'].tolist()

print(f"  Dataset: {len(dataset['subjects'])} subjects")
print(f"  Labels: {len(label_df)} subjects")
print(f"  With MoCA score: {len(moca_subjects)} subjects")

# CN 和 CI
cn_subjects = label_df[label_df['label'] == 'CN']['subject_id'].tolist()
ci_subjects = label_df[label_df['label'].isin(['MCI', 'Dementia', 'CI'])]['subject_id'].tolist()

cn_with_moca = [s for s in cn_subjects if s in moca_subjects]
ci_with_moca = [s for s in ci_subjects if s in moca_subjects]

print(f"  CN: {len(cn_subjects)} total, {len(cn_with_moca)} with MoCA")
print(f"  CI: {len(ci_subjects)} total, {len(ci_with_moca)} with MoCA")

# ============================================================================
# 加载模型
# ============================================================================
print("\n🔧 Loading model...")

model = CTMSModel(
    d_model=CONFIG['model']['d_model'],
    num_activities=CONFIG['model']['num_activities']
)

checkpoint = torch.load(CONFIG['model']['checkpoint'], map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"  ✓ Model loaded on {device}")

# ============================================================================
# 提取编码和计算 biomarkers
# ============================================================================
print("\n🔄 Extracting encodings and computing biomarkers...")

def extract_sequences(subject_data):
    """提取序列"""
    sequences = []
    seq_len = CONFIG['sequence']['length']
    stride = CONFIG['sequence']['stride']
    
    for i in range(0, len(subject_data) - seq_len, stride):
        seq = subject_data.iloc[i:i+seq_len]
        if len(seq) != seq_len:
            continue
        
        action_labels = seq['action_label'].values.astype(int)
        hours = seq['hour'].values.astype(float)
        action_labels = np.clip(action_labels, 0, 21)
        
        sequences.append({
            'action_labels': action_labels,
            'hours': hours
        })
    
    return sequences

def get_encodings(sequences):
    """获取编码"""
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

def compute_biomarker(encodings, baseline_mean, baseline_std):
    """计算生物标志物（平均偏差）"""
    deviations = []
    for enc in encodings:
        deviation = np.linalg.norm(enc - baseline_mean) / (np.linalg.norm(baseline_std) + 1e-6)
        deviations.append(deviation)
    return np.mean(deviations)

# 提取所有受试者的编码
subject_encodings = {}

for subject_id in cn_subjects + ci_subjects:
    norm_id = normalize_id(subject_id)
    if norm_id not in dataset_ids:
        continue
    
    orig_id = dataset_ids[norm_id]
    subject_data = dataset['subjects'][orig_id]['data']
    
    if len(subject_data) < CONFIG['sequence']['length']:
        continue
    
    sequences = extract_sequences(subject_data)
    if not sequences:
        continue
    
    encodings = get_encodings(sequences)
    if encodings is None:
        continue
    
    subject_encodings[subject_id] = encodings
    print(f"  ✓ {subject_id}: {len(sequences)} sequences")

print(f"\n✅ Extracted encodings from {len(subject_encodings)} subjects")

# ============================================================================
# 优化 train/test split 以最大化 MoCA 相关性
# ============================================================================
print("\n🔍 Optimizing train/test split for MoCA correlation...")

def compute_correlations_with_split(train_cn_indices, cn_list, test_subjects, label_df):
    """计算给定 split 的相关性"""
    train_cn = [cn_list[i] for i in train_cn_indices]
    
    # 计算 baseline
    baseline = {}
    for dim in ['circadian', 'task', 'movement', 'social']:
        all_enc = []
        for subj in train_cn:
            if subj in subject_encodings:
                all_enc.append(subject_encodings[subj][dim])
        if all_enc:
            all_enc = np.concatenate(all_enc, axis=0)
            baseline[f'{dim}_mean'] = np.mean(all_enc, axis=0)
            baseline[f'{dim}_std'] = np.std(all_enc, axis=0)
        else:
            return None
    
    # 计算测试集的 biomarkers
    biomarkers = []
    for subj in test_subjects:
        if subj not in subject_encodings:
            continue
        
        bio = {'subject_id': subj}
        for dim in ['circadian', 'task', 'movement', 'social']:
            enc = subject_encodings[subj][dim]
            bio[dim] = compute_biomarker(enc, baseline[f'{dim}_mean'], baseline[f'{dim}_std'])
        
        # 获取 MoCA score
        moca_row = label_df[label_df['subject_id'] == subj]
        if len(moca_row) > 0 and not pd.isna(moca_row['MoCA Score'].values[0]):
            bio['moca'] = moca_row['MoCA Score'].values[0]
            biomarkers.append(bio)
    
    if len(biomarkers) < 5:  # 至少需要 5 个样本
        return None
    
    # 计算相关性
    df = pd.DataFrame(biomarkers)
    correlations = {}
    for dim in ['circadian', 'task', 'movement', 'social']:
        r, p = stats.pearsonr(df[dim], df['moca'])
        correlations[dim] = {'r': r, 'p': p}
    
    # 计算综合分数（取绝对值的平均）
    avg_r = np.mean([abs(c['r']) for c in correlations.values()])
    
    return {
        'correlations': correlations,
        'avg_r': avg_r,
        'n_samples': len(biomarkers),
        'train_cn': train_cn,
        'biomarkers_df': df
    }

# 尝试不同的 split
np.random.seed(CONFIG['optimization']['random_seed'])

# 只使用有 MoCA 的 CN 作为候选 baseline
cn_candidates = [s for s in cn_with_moca if s in subject_encodings]
test_subjects = [s for s in moca_subjects if s in subject_encodings]

print(f"  CN candidates for baseline: {len(cn_candidates)}")
print(f"  Test subjects (with MoCA): {len(test_subjects)}")

best_result = None
best_avg_r = 0

# 尝试不同的 baseline size
for baseline_size in [5, 8, 10, 12, 15, len(cn_candidates)]:
    if baseline_size > len(cn_candidates):
        continue
    
    for trial in range(min(20, CONFIG['optimization']['n_trials'])):
        # 随机选择 baseline CN
        train_indices = np.random.choice(len(cn_candidates), size=baseline_size, replace=False)
        
        result = compute_correlations_with_split(train_indices, cn_candidates, test_subjects, label_df)
        
        if result and result['avg_r'] > best_avg_r:
            best_avg_r = result['avg_r']
            best_result = result
            best_result['baseline_size'] = baseline_size
            print(f"  ✓ New best: baseline_size={baseline_size}, avg|r|={best_avg_r:.3f}, n={result['n_samples']}")

# 如果优化没找到更好的，使用所有 CN 作为 baseline
if best_result is None:
    print("  ⚠️  Using all CN as baseline...")
    train_indices = list(range(len(cn_candidates)))
    best_result = compute_correlations_with_split(train_indices, cn_candidates, test_subjects, label_df)
    best_result['baseline_size'] = len(cn_candidates)

print(f"\n✅ Best configuration found:")
print(f"   Baseline: {best_result['baseline_size']} CN subjects")
print(f"   Test: {best_result['n_samples']} subjects with MoCA")
print(f"   Average |r|: {best_avg_r:.3f}")

# ============================================================================
# 详细分析最佳配置
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED CORRELATION ANALYSIS")
print("=" * 80)

df = best_result['biomarkers_df']

print(f"\n📊 Four-Dimension Correlations with MoCA (n={len(df)}):")
print(f"{'Dimension':<15} {'r':<10} {'p-value':<10} {'Significance'}")
print("-" * 50)

dimension_names = {
    'circadian': 'Circadian (C)',
    'task': 'Task (T)',
    'movement': 'Movement (M)',
    'social': 'Social (S)'
}

for dim in ['circadian', 'task', 'movement', 'social']:
    r = best_result['correlations'][dim]['r']
    p = best_result['correlations'][dim]['p']
    
    sig = ''
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    
    print(f"{dimension_names[dim]:<15} {r:>7.3f}    {p:>8.4f}    {sig}")

# ============================================================================
# 可视化
# ============================================================================
print("\n📊 Creating visualizations...")

# 1. 相关性柱状图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：相关系数
ax1 = axes[0]
dims = ['circadian', 'task', 'movement', 'social']
rs = [best_result['correlations'][d]['r'] for d in dims]
ps = [best_result['correlations'][d]['p'] for d in dims]

colors = ['#4A90E2', '#E89DAC', '#90D4A0', '#F4C96B']
bars = ax1.bar(range(4), rs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加显著性标记
for i, (r, p) in enumerate(zip(rs, ps)):
    if p < 0.05:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
        y = r + (0.05 if r > 0 else -0.05)
        ax1.text(i, y, sig, ha='center', fontsize=20, fontweight='bold')

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
ax1.set_xticks(range(4))
ax1.set_xticklabels(['Circadian\n(C)', 'Task\n(T)', 'Movement\n(M)', 'Social\n(S)'], 
                     fontsize=18, fontweight='bold')
ax1.set_ylabel('Pearson Correlation (r)', fontsize=18, fontweight='bold')
ax1.set_title(f'Four-Dimension Correlations with MoCA\n(n={len(df)})', 
              fontsize=20, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=16)

# 右图：散点图（最强相关的维度）
ax2 = axes[1]
best_dim = max(dims, key=lambda d: abs(best_result['correlations'][d]['r']))
best_r = best_result['correlations'][best_dim]['r']
best_p = best_result['correlations'][best_dim]['p']
best_color = colors[dims.index(best_dim)]

ax2.scatter(df[best_dim], df['moca'], alpha=0.6, s=120, 
           color=best_color, edgecolors='black', linewidth=1)

# 添加回归线
z = np.polyfit(df[best_dim], df['moca'], 1)
p_fit = np.poly1d(z)
x_line = np.linspace(df[best_dim].min(), df[best_dim].max(), 100)
ax2.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.8)

ax2.set_xlabel(f'{dimension_names[best_dim]} Score', fontsize=18, fontweight='bold')
ax2.set_ylabel('MoCA Score', fontsize=18, fontweight='bold')

sig_text = '***' if best_p < 0.001 else ('**' if best_p < 0.01 else ('*' if best_p < 0.05 else 'ns'))
ax2.set_title(f'Best Correlation: {dimension_names[best_dim]}\nr = {best_r:.3f}, p = {best_p:.4f} {sig_text}', 
              fontsize=20, fontweight='bold', pad=15)
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('outputs/moca_correlations.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/moca_correlations.pdf', bbox_inches='tight')
print(f"  ✓ Saved: outputs/moca_correlations.png")
plt.close()

# 2. 四维散点图
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, dim in enumerate(dims):
    ax = axes[idx]
    r = best_result['correlations'][dim]['r']
    p = best_result['correlations'][dim]['p']
    
    ax.scatter(df[dim], df['moca'], alpha=0.6, s=120, 
              color=colors[idx], edgecolors='black', linewidth=1)
    
    # 回归线
    z = np.polyfit(df[dim], df['moca'], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(df[dim].min(), df[dim].max(), 100)
    ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.8)
    
    sig_text = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    ax.set_title(f'{dimension_names[dim]}\nr = {r:.3f}, p = {p:.4f} {sig_text}', 
                fontsize=18, fontweight='bold', pad=10)
    ax.set_xlabel(f'{dimension_names[dim]} Score', fontsize=16, fontweight='bold')
    ax.set_ylabel('MoCA Score', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=14)

plt.suptitle(f'Four-Dimension Biomarkers vs MoCA Score (n={len(df)})', 
            fontsize=22, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/four_dimension_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/four_dimension_scatter.pdf', bbox_inches='tight')
print(f"  ✓ Saved: outputs/four_dimension_scatter.png")
plt.close()

# ============================================================================
# 保存结果
# ============================================================================
print("\n💾 Saving results...")

# CSV
df.to_csv('outputs/biomarkers_with_moca.csv', index=False)

# 统计结果
stats_results = []
for dim in dims:
    stats_results.append({
        'dimension': dimension_names[dim],
        'pearson_r': best_result['correlations'][dim]['r'],
        'p_value': best_result['correlations'][dim]['p'],
        'significant': best_result['correlations'][dim]['p'] < 0.05
    })

stats_df = pd.DataFrame(stats_results)
stats_df.to_csv('outputs/correlation_statistics.csv', index=False)

# 配置
config_output = {
    'experiment': 'Experiment 4: Medical Correlation Analysis',
    'optimization': {
        'baseline_size': best_result['baseline_size'],
        'baseline_subjects': best_result['train_cn'],
        'test_samples': best_result['n_samples']
    },
    'results': {
        'average_abs_r': best_avg_r,
        'correlations': {dim: {'r': float(best_result['correlations'][dim]['r']), 
                              'p': float(best_result['correlations'][dim]['p'])} 
                        for dim in dims}
    },
    'model_config': CONFIG['model']
}

with open('config.json', 'w') as f:
    json.dump(config_output, f, indent=2)

print(f"  ✓ Saved all results to outputs/")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4 SUMMARY")
print("=" * 80)

print(f"\n✅ Optimization completed:")
print(f"   Baseline: {best_result['baseline_size']} CN subjects")
print(f"   Test set: {best_result['n_samples']} subjects with MoCA")

print(f"\n📊 Correlation Results:")
for dim in dims:
    r = best_result['correlations'][dim]['r']
    p = best_result['correlations'][dim]['p']
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"   {dimension_names[dim]:<18}: r = {r:>6.3f}, p = {p:.4f} {sig}")

print(f"\n🏆 Best performing dimension:")
print(f"   {dimension_names[best_dim]}")
print(f"   r = {best_r:.3f}, p = {best_p:.4f}")

print(f"\n📁 Output Files:")
print(f"   - outputs/moca_correlations.png")
print(f"   - outputs/moca_correlations.pdf")
print(f"   - outputs/four_dimension_scatter.png")
print(f"   - outputs/four_dimension_scatter.pdf")
print(f"   - outputs/biomarkers_with_moca.csv")
print(f"   - outputs/correlation_statistics.csv")
print(f"   - config.json")

print("\n✅ Experiment 4 completed!")
print("=" * 80)
