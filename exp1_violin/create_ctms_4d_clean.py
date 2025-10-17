#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

np.random.seed(42)

# Matplotlib settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 24  
plt.rcParams['axes.labelsize'] = 24  
plt.rcParams['axes.titlesize'] = 24  
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load and filter subject feature data."""
    feature_file = 'outputs/subject_features_advanced.csv'
    
    df = pd.read_csv(feature_file)
    
    # Keep only CN and CI subjects
    df = df[df['label'].isin(['CN', 'CI'])].copy()
    
    print(f"  ✓ {len(df)} subjects loaded (CN: {len(df[df['label']=='CN'])}, CI: {len(df[df['label']=='CI'])})")
    
    return df


def create_clean_4d_violin(df, output_path='outputs/ctms_four_dimensions_clean.png'):
    """Create a clean 4-dimensional violin plot."""
    print("\n📊 Generating clean 4D Violin plot...")

    # Define the four dimensions and their representative features
    dimensions = {
        'Circadian Rhythm': 'circadian_daytime_ratio',
        'Task Completion': 'task_simpson_diversity',
        'Movement Pattern': 'movement_out_of_view_ratio',
        'Social Interaction': 'social_short_gaps_ratio'
    }
    
    # Create a 1x4 subplot layout
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Color settings
    dimension_colors = {
        'Circadian Rhythm': {'CN': '#4A90E2', 'CI': '#4A90E2'},  # Blue
        'Task Completion': {'CN': '#E89DAC', 'CI': '#E89DAC'},   # Pink
        'Movement Pattern': {'CN': '#90D4A0', 'CI': '#90D4A0'},  # Green
        'Social Interaction': {'CN': '#F4C96B', 'CI': '#F4C96B'} # Orange
    }
    
    for idx, (dim_name, feature_name) in enumerate(dimensions.items()):
        ax = axes[idx]
        colors = dimension_colors[dim_name]
        base_color = colors['CN']
        
        # Extract data for each group
        cn_data = df[df['label'] == 'CN'][feature_name].values
        ci_data = df[df['label'] == 'CI'][feature_name].values
        
        # Prepare for plotting
        plot_data = pd.DataFrame({
            'Group': ['CN'] * len(cn_data) + ['CI'] * len(ci_data),
            'Value': np.concatenate([cn_data, ci_data])
        })
        
        # Draw Violin plot
        parts = ax.violinplot(
            [cn_data, ci_data],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set Violin color
        for pc in parts['bodies']:
            pc.set_facecolor(base_color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay Boxplot
        ax.boxplot(
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
        
        # Add scatter points
        np.random.seed(42 + idx)
        # CN group: green circles
        x_cn = np.random.normal(0, 0.04, size=len(cn_data))
        ax.scatter(x_cn, cn_data, alpha=0.7, s=100, color='#2ECC71', 
                  marker='o', edgecolors='black', linewidth=0.8, zorder=3, label='CN')
        
        # CI group: red triangles
        x_ci = np.random.normal(1, 0.04, size=len(ci_data))
        ax.scatter(x_ci, ci_data, alpha=0.7, s=100, color='#E74C3C', 
                  marker='^', edgecolors='black', linewidth=0.8, zorder=3, label='CI')
        
        # Add horizontal reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Set title
        ax.set_title(dim_name, fontsize=24, fontweight='bold', pad=15)
        
        # X-axis setup
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CN', 'CI'], fontsize=24, fontweight='bold')
        
        # Y-axis setup
        if idx == 0:
            ax.set_ylabel('Normalized Score (Z-score)', fontsize=24, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # Hide Y tick labels but keep ticks
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=6, width=1.5)
        
        # Add horizontal grid lines
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # Beautify borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
    
    # Adjust layout
    plt.tight_layout(pad=1.5)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✅ Saved: {output_path}")
    plt.close()


def main():
    """Main function"""
    print("=" * 80)
    print("🎨 CTMS 4-Dimension Visualization - Clean Version")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Generate violin plot
    create_clean_4d_violin(df)


if __name__ == '__main__':
    main()
