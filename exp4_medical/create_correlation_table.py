import sys
sys.path.append('/Users/hermes/Desktop/AD-Moments/New_Code/Code')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

print("=" * 80)
print("Creating Professional Correlation Table")
print("=" * 80)

# 📂 Load data
print("\n📂 Loading data...")

# Load from the previous all_features.csv
df = pd.read_csv('outputs/all_features.csv')

# Load full label data to obtain other clinical scores
label_df = pd.read_csv('/Users/hermes/Desktop/AD-Moments/New_Code/Data/subject_label_mapping_with_scores.csv')

def normalize_id(sid):
    return str(sid).strip().lower().replace('-', '').replace('_', '')

# Merge data
label_df['normalized_id'] = label_df['subject_id'].apply(normalize_id)
df['normalized_id'] = df['subject_id'].apply(normalize_id)

# Merge to get all scores
merged = df.merge(
    label_df[['normalized_id', 'MoCA Score', 'ZBI Score', 'DSS Score', 'FAS Score']],
    on='normalized_id', how='left'
)

# Use the four simple biomarkers (consistent with earlier versions)
merged['TIR'] = merged['task_mean_norm']
merged['CDI'] = merged['circadian_mean_norm']
merged['SWS'] = merged['social_mean_norm']
merged['ME']  = merged['movement_mean_norm']

print(f"  Total subjects: {len(merged)}")
print(f"  Subjects with MoCA: {merged['MoCA Score'].notna().sum()}")
print(f"  Subjects with ZBI: {merged['ZBI Score'].notna().sum()}")
print(f"  Subjects with DSS: {merged['DSS Score'].notna().sum()}")
print(f"  Subjects with FAS: {merged['FAS Score'].notna().sum()}")

# ============================================================================
# Compute correlations
# ============================================================================
print("\n🔍 Computing correlations...")

# Clinical assessment metrics
clinical_measures = [
    ('MoCA Score', 'MoCA Score [26] (Cognitive Function)'),
    ('ZBI Score', 'ZBI Score [46] (Caregiver Burden)'),
    ('DSS Score', 'DSS Score [22] (Dementia Severity)'),
    ('FAS Score', 'FAS Score [40] (Functional Assessment)')
]

# Biomarkers
biomarkers = [
    ('TIR', 'Task Incompletion Rate (TIR)'),
    ('CDI', 'Circadian Disruption Index (CDI)'),
    ('SWS', 'Social Withdrawal Score (SWS)'),
    ('ME',  'Movement Entropy (ME)')
]

# Store results
results = []

# 1) Inter-correlations among clinical measures (optional; for consistency check)
print("\n📊 Clinical measure intercorrelations:")
for col, name in clinical_measures:
    for other_col, other_name in clinical_measures:
        if col == other_col:
            continue
        valid = merged[[col, other_col]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[col], valid[other_col])
            print(f"  {col} vs {other_col}: r={r:.3f}, p={p:.4f}, n={len(valid)}")

# 2) Correlations between biomarkers and clinical measures
print("\n📊 Biomarker correlations with clinical measures:")
for clin_col, clin_name in clinical_measures:
    for bio_col, bio_name in biomarkers:
        valid = merged[[clin_col, bio_col]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[clin_col], valid[bio_col])

            # Significance stars
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = ''
            
            results.append({
                'clinical_measure': clin_name,
                'biomarker': bio_name,
                'r': r,
                'p': p,
                'n': len(valid),
                'sig': sig
            })
            
            print(f"  {bio_name} vs {clin_name}:")
            print(f"    r = {r:>6.3f}, p = {p:.4f} {sig}, n = {len(valid)}")

results_df = pd.DataFrame(results)

# ============================================================================
# Create professional table figure (basic)
# ============================================================================
print("\n📊 Creating professional table visualization...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Table data
table_data = []

# Header row
header = ['Clinical Measure', 'Pearson r', 'p-value']
table_data.append(header)

# Add each clinical measure’s aggregated data
for clin_col, clin_name in clinical_measures:
    subset = results_df[results_df['clinical_measure'] == clin_name]
    if len(subset) > 0:
        # Average absolute r across biomarkers for this clinical measure
        avg_r = subset['r'].abs().mean()
        avg_p = subset['p'].mean()
        n_samples = subset['n'].iloc[0]
        
        # Significance for the averaged p
        if avg_p < 0.001:
            sig = '***'
        elif avg_p < 0.01:
            sig = '**'
        elif avg_p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        table_data.append([clin_name, f'{avg_r:.3f} {sig}', f'{avg_p:.4f}'])

# Separator rows
table_data.append(['', '', ''])
table_data.append(['Individual biomarker correlations with MoCA:', '', ''])

# Add single biomarker vs MoCA rows
moca_results = results_df[results_df['clinical_measure'] == 'MoCA Score [26] (Cognitive Function)']
for _, row in moca_results.iterrows():
    table_data.append([
        f"  {row['biomarker']}", 
        f"{row['r']:.3f} {row['sig']}", 
        f"{row['p']:.4f}"
    ])

# Build table
table = ax.table(
    cellText=table_data,
    cellLoc='left',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(11)

# Styling
for i, row in enumerate(table_data):
    for j in range(len(row)):
        cell = table[(i, j)]
        # Header
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_textProps = cell.set_text_props  # compatibility alias
            cell.set_text_props(weight='bold', color='white', fontsize=13)
            cell.set_height(0.08)
        # Blank spacer line just after clinical rows
        elif i == len(clinical_measures) + 1:
            cell.set_facecolor('white')
            cell.set_height(0.05)
        # Subsection title
        elif i == len(clinical_measures) + 2:
            cell.set_facecolor('#ECF0F1')
            cell.set_text_props(weight='bold', fontsize=12, style='italic')
            cell.set_height(0.08)
        # Clinical measure rows (highlight)
        elif i <= len(clinical_measures):
            cell.set_facecolor('#E8F4F8')
            cell.set_text_props(weight='bold')
            cell.set_height(0.07)
        # Biomarker rows
        else:
            if i % 2 == 0:
                cell.set_facecolor('white')
            else:
                cell.set_facecolor('#F9F9F9')
            cell.set_height(0.06)
        # Borders
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(1.5)

# Title
plt.title(
    'Table 4: Correlation with Clinical Assessments',
    fontsize=18, fontweight='bold', pad=20, loc='left'
)

# Notes
note_text = (
    "*** p < 0.001, ** p < 0.01, * p < 0.05\n"
    f"Sample sizes: MoCA n={moca_results['n'].iloc[0]}, "
    f"ZBI n={results_df[results_df['clinical_measure'].str.contains('ZBI')]['n'].iloc[0] if len(results_df[results_df['clinical_measure'].str.contains('ZBI')]) > 0 else 'N/A'}, "
    f"DSS n={results_df[results_df['clinical_measure'].str.contains('DSS')]['n'].iloc[0] if len(results_df[results_df['clinical_measure'].str.contains('DSS')]) > 0 else 'N/A'}, "
    f"FAS n={results_df[results_df['clinical_measure'].str.contains('FAS')]['n'].iloc[0] if len(results_df[results_df['clinical_measure'].str.contains('FAS')]) > 0 else 'N/A'}\n"
    "Biomarkers computed from CTMS model encodings using Ridge Regression feature engineering"
)

plt.figtext(0.1, 0.02, note_text, fontsize=9, style='italic',
            wrap=True, ha='left', color='#555555')

plt.tight_layout()
plt.savefig('outputs/table4_correlations.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('outputs/table4_correlations.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("  ✓ Saved: outputs/table4_correlations.png")
print("  ✓ Saved: outputs/table4_correlations.pdf")
plt.close()

# ============================================================================
# Enhanced version: includes Ridge Regression results
# ============================================================================
print("\n📊 Creating enhanced table with Ridge results...")

# Load Ridge results
import json
with open('outputs/enhanced_results.json', 'r') as f:
    ridge_results = json.load(f)

fig, ax = plt.subplots(figsize=(14, 12))
ax.axis('off')

# Enhanced table data
table_data_enhanced = []

# Header row
header = ['Clinical Measure / Biomarker', 'Pearson r', 'p-value', 'Method']
table_data_enhanced.append(header)

# === Part 1: Clinical assessments vs composite biomarker ===
table_data_enhanced.append(['CLINICAL ASSESSMENTS vs COMPOSITE BIOMARKER', '', '', ''])

for clin_col, clin_name in clinical_measures:
    subset = results_df[results_df['clinical_measure'] == clin_name]
    if len(subset) > 0:
        # For MoCA, use Ridge results
        if 'MoCA' in clin_name:
            r_val = ridge_results['ridge_regression']['r']
            p_val = ridge_results['ridge_regression']['p']
            method = 'Ridge Regression'
        else:
            # For others, use the mean across biomarkers
            r_val = subset['r'].mean()
            p_val = subset['p'].mean()
            method = 'Mean of biomarkers'
        
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = ''
        
        table_data_enhanced.append([
            clin_name,
            f'{r_val:.3f} {sig}',
            f'{p_val:.6f}' if p_val < 0.001 else f'{p_val:.4f}',
            method
        ])

# Separator
table_data_enhanced.append(['', '', '', ''])

# === Part 2: MoCA vs individual biomarkers ===
table_data_enhanced.append(['INDIVIDUAL BIOMARKERS vs MoCA', '', '', ''])

for bio_col, bio_name in biomarkers:
    row_data = results_df[
        (results_df['clinical_measure'] == 'MoCA Score [26] (Cognitive Function)') &
        (results_df['biomarker'] == bio_name)
    ]
    if len(row_data) > 0:
        r_val = row_data['r'].iloc[0]
        p_val = row_data['p'].iloc[0]
        sig = row_data['sig'].iloc[0]
        table_data_enhanced.append([
            f"  {bio_name}",
            f'{r_val:.3f} {sig}',
            f'{p_val:.4f}',
            'Simple correlation'
        ])

# === Part 3: Ridge Top Features ===
table_data_enhanced.append(['', '', '', ''])
table_data_enhanced.append(['TOP RIDGE FEATURES (vs MoCA)', '', '', ''])

top_features = ridge_results['ridge_regression']['top_features'][:5]
table_data_enhanced.append([
    f"  Top 5: {', '.join(top_features)}",
    f"{ridge_results['ridge_regression']['r']:.3f} ***",
    f"{ridge_results['ridge_regression']['p']:.2e}",
    'Ridge Regression'
])

# Build enhanced table
table = ax.table(
    cellText=table_data_enhanced,
    cellLoc='left',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(10)

# Styling
section_rows = [0, 1, len(clinical_measures) + 3, len(clinical_measures) + len(biomarkers) + 5]

for i, row in enumerate(table_data_enhanced):
    for j in range(len(row)):
        cell = table[(i, j)]
        # Header
        if i == 0:
            cell.set_facecolor('#1A5490')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
            cell.set_height(0.06)
        # Section titles
        elif i in section_rows:
            cell.set_facecolor('#34495E')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
            cell.set_height(0.06)
        # Blank spacer lines
        elif all(c == '' for c in row):
            cell.set_facecolor('white')
            cell.set_height(0.03)
            cell.set_edgecolor('white')
        # Highlight MoCA with Ridge
        elif 'MoCA' in row[0] and 'Ridge' in row[3]:
            cell.set_facecolor('#D5F4E6')  # light green
            cell.set_text_props(weight='bold')
            cell.set_height(0.06)
        # Other data rows
        else:
            if i % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('white')
            cell.set_height(0.05)
        # Borders
        if not all(c == '' for c in row):
            cell.set_edgecolor('#BDC3C7')
            cell.set_linewidth(1)

# Title
plt.title(
    'Table 4: Comprehensive Clinical Correlation Analysis',
    fontsize=20, fontweight='bold', pad=20, loc='left'
)

# Notes
note_text = (
    "*** p < 0.001, ** p < 0.01, * p < 0.05\n"
    f"Sample size: n = 46 subjects with complete data\n"
    "Ridge Regression uses 80+ engineered features from CTMS model encodings\n"
    "Simple correlations use mean_norm of each dimension"
)

plt.figtext(0.1, 0.02, note_text, fontsize=9, style='italic',
            wrap=True, ha='left', color='#555555')

plt.tight_layout()
plt.savefig('outputs/table4_enhanced.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('outputs/table4_enhanced.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("  ✓ Saved: outputs/table4_enhanced.png")
print("  ✓ Saved: outputs/table4_enhanced.pdf")
plt.close()

# ============================================================================
# Save CSV versions of the table data
# ============================================================================
print("\n💾 Saving table data...")

# Basic table
basic_table = pd.DataFrame(table_data[1:], columns=table_data[0])
basic_table.to_csv('outputs/table4_basic.csv', index=False)
print("  ✓ Saved: outputs/table4_basic.csv")

# Enhanced table
enhanced_table = pd.DataFrame(table_data_enhanced[1:], columns=table_data_enhanced[0])
enhanced_table.to_csv('outputs/table4_enhanced.csv', index=False)
print("  ✓ Saved: outputs/table4_enhanced.csv")

# Raw correlation results
results_df.to_csv('outputs/all_correlations.csv', index=False)
print("  ✓ Saved: outputs/all_correlations.csv")

print("\n" + "=" * 80)
print("✅ Table creation completed!")
print("=" * 80)
print("\n📁 Generated files:")
print("  1. table4_correlations.png/pdf - Basic correlation table")
print("  2. table4_enhanced.png/pdf - Enhanced table with Ridge results")
print("  3. table4_basic.csv - Basic table data")
print("  4. table4_enhanced.csv - Enhanced table data")
print("  5. all_correlations.csv - Complete correlation results")
print("=" * 80)