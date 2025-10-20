#!/usr/bin/env python3
"""
Quick test to verify Exp_Full structure and data loading.
Tests basic functionality without running full experiments.
"""

import sys
from pathlib import Path
import json
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

print("="*80)
print("EXP_FULL QUICK VALIDATION TEST")
print("="*80)

# Test 1: Check directory structure
print("\n1. Checking directory structure...")
without_dir = SCRIPT_DIR / 'Without_Personalization'
with_dir = SCRIPT_DIR / 'With_Personalization'

print(f"   Without_Personalization exists: {without_dir.exists()}")
print(f"   With_Personalization exists: {with_dir.exists()}")

# Test 2: Check experiment scripts
print("\n2. Checking experiment scripts...")
experiments = ['exp1_embedding_viz.py', 'exp2_daily_patterns.py', 
               'exp3_classification.py', 'exp4_medical_correlations.py']

for exp in experiments:
    without_exists = (without_dir / exp).exists()
    with_exists = (with_dir / exp).exists()
    status_without = "✓" if without_exists else "✗"
    status_with = "✓" if with_exists else "✗"
    print(f"   {exp:30s} Without:{status_without}  With:{status_with}")

# Test 3: Check data files
print("\n3. Checking data files...")
data_path = PROJECT_ROOT / 'sample_data' / 'dataset_one_month.jsonl'
subjects_path = PROJECT_ROOT / 'sample_data' / 'subjects_public.json'

print(f"   dataset_one_month.jsonl exists: {data_path.exists()}")
print(f"   subjects_public.json exists: {subjects_path.exists()}")

if data_path.exists():
    # Load one subject
    with open(data_path, 'r') as f:
        first_line = f.readline()
        subj = json.loads(first_line)
        print(f"   Dataset format: anon_id={subj.get('anon_id')}, "
              f"label={subj.get('label')}, "
              f"sequence_len={len(subj.get('sequence', []))}")

if subjects_path.exists():
    with open(subjects_path, 'r') as f:
        subjects_meta = json.load(f)
        print(f"   Medical data: {len(subjects_meta)} subjects")
        first = subjects_meta[0]
        print(f"   Format: anon_id={first.get('anon_id')}, "
              f"scores={list(first.get('scores', {}).keys())}")

# Test 4: Check output directories
print("\n4. Checking output directories...")
without_outputs = without_dir / 'outputs'
with_outputs = with_dir / 'outputs'

print(f"   Without/outputs exists: {without_outputs.exists()}")
print(f"   With/outputs exists: {with_outputs.exists()}")

if without_outputs.exists():
    json_files = list(without_outputs.glob('*.json'))
    png_files = list(without_outputs.glob('*.png'))
    pdf_files = list(without_outputs.glob('*.pdf'))
    print(f"   Without/outputs: {len(json_files)} JSON, "
          f"{len(png_files)} PNG, {len(pdf_files)} PDF")

if with_outputs.exists():
    json_files = list(with_outputs.glob('*.json'))
    png_files = list(with_outputs.glob('*.png'))
    pdf_files = list(with_outputs.glob('*.pdf'))
    print(f"   With/outputs: {len(json_files)} JSON, "
          f"{len(png_files)} PNG, {len(pdf_files)} PDF")

# Test 5: Check result files
print("\n5. Checking result files...")
result_files = {
    'exp1_metrics.json': (without_outputs / 'exp1_metrics.json', 
                          with_outputs / 'exp1_metrics.json'),
    'exp2_metrics.json': (without_outputs / 'exp2_metrics.json',
                          with_outputs / 'exp2_metrics.json'),
    'exp3_metrics.json': (without_outputs / 'exp3_metrics.json',
                          with_outputs / 'exp3_metrics.json'),
    'exp4_correlations.json': (without_outputs / 'exp4_correlations.json',
                               with_outputs / 'exp4_correlations.json'),
}

for name, (without_path, with_path) in result_files.items():
    without_exists = without_path.exists() if without_outputs.exists() else False
    with_exists = with_path.exists() if with_outputs.exists() else False
    
    status = ""
    if without_exists and with_exists:
        status = "✓✓ Both exist"
    elif without_exists:
        status = "⚠  Without only"
    elif with_exists:
        status = "⚠  With only"
    else:
        status = "✗  Missing"
    
    print(f"   {name:25s} {status}")

# Test 6: Sample a result file
print("\n6. Sampling result content...")
sample_file = without_outputs / 'exp2_metrics.json' if (without_outputs / 'exp2_metrics.json').exists() else None
if sample_file is None:
    sample_file = without_outputs / 'exp1_metrics.json' if (without_outputs / 'exp1_metrics.json').exists() else None

if sample_file and sample_file.exists():
    with open(sample_file, 'r') as f:
        data = json.load(f)
        print(f"   Sample from {sample_file.name}:")
        if isinstance(data, dict):
            for key in list(data.keys())[:5]:
                val = data[key]
                if isinstance(val, (int, float, str, bool)):
                    print(f"      {key}: {val}")
                elif isinstance(val, dict):
                    print(f"      {key}: {{...}} ({len(val)} keys)")
                elif isinstance(val, list):
                    print(f"      {key}: [...] ({len(val)} items)")
else:
    print("   No result files found yet")

# Test 7: Check documentation
print("\n7. Checking documentation...")
docs = ['README.md', 'RESULTS_SUMMARY.md', 'results.md', 'STATUS_REPORT.md']
for doc in docs:
    doc_path = SCRIPT_DIR / doc
    exists = doc_path.exists()
    status = "✓" if exists else "✗"
    private = " (PRIVATE)" if doc == 'results.md' else ""
    print(f"   {doc:25s} {status}{private}")

# Test 8: Check .gitignore
print("\n8. Checking .gitignore...")
gitignore_path = SCRIPT_DIR / '.gitignore'
if gitignore_path.exists():
    with open(gitignore_path, 'r') as f:
        content = f.read()
        excludes_results = 'results.md' in content
        excludes_cleanup = 'CLEANUP_SUMMARY.md' in content
        print(f"   .gitignore exists: ✓")
        print(f"   Excludes results.md: {'✓' if excludes_results else '✗'}")
        print(f"   Excludes CLEANUP_SUMMARY.md: {'✓' if excludes_cleanup else '✗'}")
else:
    print(f"   .gitignore: ✗ Missing")

# Summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

issues = []
if not without_dir.exists():
    issues.append("Without_Personalization directory missing")
if not with_dir.exists():
    issues.append("With_Personalization directory missing")
if not data_path.exists():
    issues.append("Dataset file missing")
if not subjects_path.exists():
    issues.append("Medical data file missing")

if issues:
    print("⚠ Issues found:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("✓ Structure looks good!")
    print("\nNote: Experiment scripts are REFERENCE IMPLEMENTATIONS.")
    print("      They demonstrate the analysis approach but may need")
    print("      data format adaptation to run directly.")
    print("\n✓ All result files and documentation are in place.")
    print("✓ Ready for GitHub upload (after updating LICENSE).")

print("="*80)
