"""
Generate comprehensive comparison table from all experiment results.

This script aggregates metrics from Without_Personalization and With_Personalization
experiments and creates side-by-side comparison tables in CSV and Markdown formats.

Usage:
    python generate_comparison_table.py

Outputs:
    - comparison_table.csv: Machine-readable tabular format
    - comparison_table.md: Human-readable markdown table
"""

import json
import pandas as pd
from pathlib import Path

# Use relative paths for portability
ROOT_DIR = Path(__file__).parent

def load_metrics(pipeline, exp_name):
    """Load metrics JSON for a specific experiment."""
    metrics_file = ROOT_DIR / pipeline / "outputs" / f"{exp_name}_metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except:
        return None

def format_value(value, format_type='float'):
    """Format value for table display."""
    if value is None:
        return "N/A"
    
    if format_type == 'float':
        return f"{value:.3f}"
    elif format_type == 'percent':
        return f"{value*100:.1f}%"
    elif format_type == 'ratio':
        return f"{value:.3f}×"
    elif format_type == 'pvalue':
        if value < 0.001:
            return "<0.001***"
        elif value < 0.01:
            return f"{value:.3f}**"
        elif value < 0.05:
            return f"{value:.3f}*"
        else:
            return f"{value:.3f}"
    else:
        return str(value)

def main():
    print("="*80)
    print("GENERATING COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    
    # Load all metrics
    pipelines = ["Without_Personalization", "With_Personalization"]
    experiments = ["exp1", "exp2", "exp3", "exp4"]
    
    all_metrics = {}
    for pipeline in pipelines:
        all_metrics[pipeline] = {}
        for exp in experiments:
            metrics = load_metrics(pipeline, exp)
            all_metrics[pipeline][exp] = metrics
    
    # Create comparison table
    rows = []
    
    # === EXP1: Embedding Visualization ===
    print("\n1. Processing Exp1 metrics...")
    for pipeline in pipelines:
        exp1 = all_metrics[pipeline].get('exp1')
        if exp1:
            rows.append({
                'Experiment': 'Exp1: Embedding',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'Silhouette Score',
                'Value': format_value(exp1.get('silhouette_score'), 'float')
            })
            rows.append({
                'Experiment': 'Exp1: Embedding',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'Davies-Bouldin Score',
                'Value': format_value(exp1.get('davies_bouldin_score'), 'float')
            })
    
    # === EXP2: Daily Patterns ===
    print("2. Processing Exp2 metrics...")
    for pipeline in pipelines:
        exp2 = all_metrics[pipeline].get('exp2')
        if exp2:
            rows.append({
                'Experiment': 'Exp2: Daily Patterns',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'CI/CN Ratio',
                'Value': format_value(exp2.get('ci_cn_ratio'), 'ratio')
            })
            rows.append({
                'Experiment': 'Exp2: Daily Patterns',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'P-value',
                'Value': format_value(exp2.get('p_value'), 'pvalue')
            })
            rows.append({
                'Experiment': 'Exp2: Daily Patterns',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'CN Mean Score',
                'Value': format_value(exp2.get('cn_mean'), 'float')
            })
            rows.append({
                'Experiment': 'Exp2: Daily Patterns',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'CI Mean Score',
                'Value': format_value(exp2.get('ci_mean'), 'float')
            })
    
    # === EXP3: Classification ===
    print("3. Processing Exp3 metrics...")
    for pipeline in pipelines:
        exp3 = all_metrics[pipeline].get('exp3')
        if exp3:
            rows.append({
                'Experiment': 'Exp3: Classification',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'F1 Score',
                'Value': format_value(exp3.get('f1'), 'float')
            })
            rows.append({
                'Experiment': 'Exp3: Classification',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'Sensitivity',
                'Value': format_value(exp3.get('sensitivity'), 'percent')
            })
            rows.append({
                'Experiment': 'Exp3: Classification',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'Specificity',
                'Value': format_value(exp3.get('specificity'), 'percent')
            })
            rows.append({
                'Experiment': 'Exp3: Classification',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'Precision',
                'Value': format_value(exp3.get('precision'), 'percent')
            })
            rows.append({
                'Experiment': 'Exp3: Classification',
                'Pipeline': pipeline.replace('_', ' '),
                'Metric': 'CI/CN Ratio',
                'Value': format_value(exp3.get('ci_cn_ratio'), 'ratio')
            })
    
    # === EXP4: Medical Correlations ===
    print("4. Processing Exp4 correlations...")
    for pipeline in pipelines:
        exp4_file = ROOT_DIR / pipeline / "outputs" / "exp4_correlations.json"
        if exp4_file.exists():
            with open(exp4_file, 'r') as f:
                exp4 = json.load(f)
            
            # Report significant correlations only
            for dim in ['Circadian', 'Task', 'Movement', 'Social']:
                for med in ['MoCA', 'ZBI', 'FAS', 'NPI']:
                    corr_data = exp4.get(dim, {}).get(med, {})
                    if corr_data.get('significant', False):
                        rows.append({
                            'Experiment': 'Exp4: Medical Corr',
                            'Pipeline': pipeline.replace('_', ' '),
                            'Metric': f'{dim} vs {med}',
                            'Value': f"r={format_value(corr_data['r'], 'float')}, {format_value(corr_data['p'], 'pvalue')}"
                        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_csv = ROOT_DIR / "comparison_table.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Comparison table saved to: {output_csv}")
    
    # Create pivot table for better visualization
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Group by experiment
    for exp in df['Experiment'].unique():
        print(f"\n{exp}:")
        exp_df = df[df['Experiment'] == exp]
        
        # Pivot to show side-by-side comparison
        pivot = exp_df.pivot_table(
            index='Metric',
            columns='Pipeline',
            values='Value',
            aggfunc='first'
        )
        print(pivot.to_string())
    
    # Save nicely formatted table
    output_md = ROOT_DIR / "comparison_table.md"
    with open(output_md, 'w') as f:
        f.write("# CTMS Experiment Results Comparison\n\n")
        
        for exp in df['Experiment'].unique():
            f.write(f"## {exp}\n\n")
            exp_df = df[df['Experiment'] == exp]
            pivot = exp_df.pivot_table(
                index='Metric',
                columns='Pipeline',
                values='Value',
                aggfunc='first'
            )
            f.write(pivot.to_markdown())
            f.write("\n\n")
    
    print(f"\n✓ Markdown table saved to: {output_md}")
    print("="*80)

if __name__ == '__main__':
    main()
