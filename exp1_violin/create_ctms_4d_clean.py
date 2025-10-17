#!/usr/bin/env python3
"""
创建简洁版4维度CTMS可视化 (Clean Version)
- 放大字体
- 移除标题
- 简化标签
- 移除所有统计文本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置随机种子
np.random.seed(42)

# 配置matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 24  # 增大基础字体
plt.rcParams['axes.labelsize'] = 24  # 增大轴标签
plt.rcParams['axes.titlesize'] = 24  # 增大标题
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['figure.dpi'] = 300

def load_data():
    """加载数据"""
    feature_file = 'outputs/subject_features_advanced.csv'
    
    print("📂 加载数据...")
    df = pd.read_csv(feature_file)
    
    # 过滤出CN和CI
    df = df[df['label'].isin(['CN', 'CI'])].copy()
    
    print(f"  ✓ {len(df)} 个受试者 (CN: {len(df[df['label']=='CN'])}, CI: {len(df[df['label']=='CI'])})")
    
    return df

def create_clean_4d_violin(df, output_path='outputs/ctms_four_dimensions_clean.png'):
    """创建简洁版4维度Violin图"""
    
    print("\n📊 生成简洁版4维度Violin图...")
    
    # 定义4个维度及其最佳特征
    dimensions = {
        'Circadian Rhythm': 'circadian_daytime_ratio',
        'Task Completion': 'task_simpson_diversity',
        'Movement Pattern': 'movement_out_of_view_ratio',
        'Social Interaction': 'social_short_gaps_ratio'
    }
    
    # 创建1x4横向子图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 颜色配置 - 使用之前的4种颜色
    dimension_colors = {
        'Circadian Rhythm': {'CN': '#4A90E2', 'CI': '#4A90E2'},  # 蓝色
        'Task Completion': {'CN': '#E89DAC', 'CI': '#E89DAC'},   # 粉色
        'Movement Pattern': {'CN': '#90D4A0', 'CI': '#90D4A0'},  # 绿色
        'Social Interaction': {'CN': '#F4C96B', 'CI': '#F4C96B'} # 橙色
    }
    
    for idx, (dim_name, feature_name) in enumerate(dimensions.items()):
        ax = axes[idx]
        colors = dimension_colors[dim_name]  # 获取当前维度的颜色
        base_color = colors['CN']  # 该维度的基础颜色
        
        # 提取数据
        cn_data = df[df['label'] == 'CN'][feature_name].values
        ci_data = df[df['label'] == 'CI'][feature_name].values
        
        # 准备绘图数据
        plot_data = pd.DataFrame({
            'Group': ['CN']*len(cn_data) + ['CI']*len(ci_data),
            'Value': np.concatenate([cn_data, ci_data])
        })
        
        # 绘制Violin图
        parts = ax.violinplot(
            [cn_data, ci_data],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # 设置Violin颜色 - 两组都用同一个颜色
        for pc in parts['bodies']:
            pc.set_facecolor(base_color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # 叠加Box图
        bp = ax.boxplot(
            [cn_data, ci_data],
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2.5),
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.8),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5)
        )
        
        # 叠加散点 - CN用绿色圆点，CI用红色三角形
        np.random.seed(42 + idx)
        # CN组：绿色圆点
        x_cn = np.random.normal(0, 0.04, size=len(cn_data))
        ax.scatter(x_cn, cn_data, alpha=0.7, s=100, color='#2ECC71', 
                  marker='o', edgecolors='black', linewidth=0.8, zorder=3, label='CN')
        
        # CI组：红色三角形
        x_ci = np.random.normal(1, 0.04, size=len(ci_data))
        ax.scatter(x_ci, ci_data, alpha=0.7, s=100, color='#E74C3C', 
                  marker='^', edgecolors='black', linewidth=0.8, zorder=3, label='CI')
        
        # 添加y=0的虚线参考线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # 设置标题
        ax.set_title(dim_name, fontsize=24, fontweight='bold', pad=15)
        
        # 设置X轴
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CN', 'CI'], fontsize=24, fontweight='bold')
        
        # 设置Y轴
        if idx == 0:  # 只在最左侧子图显示Y轴标签
            ax.set_ylabel('Normalized Score (Z-score)', fontsize=24, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # 隐藏Y轴刻度值，但保留刻度线
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=6, width=1.5)
        
        # 添加水平网格线
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
    
    # 调整布局
    plt.tight_layout(pad=1.5)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✅ 保存: {output_path}")
    plt.close()

def main():
    """主函数"""
    print("="*80)
    print("🎨 CTMS 4维度可视化 - 简洁版")
    print("="*80)
    
    # 加载数据
    df = load_data()
    
    # 生成简洁版Violin图
    create_clean_4d_violin(df)
    
    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80)
    print("\n📊 生成的图表:")
    print("    ctms_four_dimensions_clean.png ⭐⭐⭐⭐⭐")
    print("\n💡 特点:")
    print("    ✓ 大字体,易读")
    print("    ✓ 无标题,直接展示")
    print("    ✓ 简洁的轴标签")
    print("    ✓ 移除所有统计文本")
    print("    ✓ 纯净的数据可视化")
    
    print("\n🔍 查看图片:")
    print("    cd /Users/hermes/Desktop/AD-Moments/New_Code/Version4/exp1_tsne/outputs")
    print("    open ctms_four_dimensions_clean.png")

if __name__ == '__main__':
    main()
