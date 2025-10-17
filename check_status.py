#!/usr/bin/env python3
"""
Quick validation script for Version 7
Checks only essential components
"""

import os
import sys

print("=" * 70)
print("Version 7 - Quick Status Check")
print("=" * 70)
print()

# Check 1: Directory structure
print("[1/5] Directory Structure")
required_dirs = [
    'models',
    'exp1_violin',
    'exp2_daily', 
    'exp3_classification',
    'exp4_medical',
    'docs'
]

all_dirs_ok = True
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"  ✅ {dir_name}/")
    else:
        print(f"  ❌ {dir_name}/ - MISSING")
        all_dirs_ok = False

print()

# Check 2: Model file
print("[2/5] Model File")
model_paths = [
    '../../ctms_model_medium.pth',
    '../../../ctms_model_medium.pth'
]

model_found = False
for path in model_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ Found: {path} ({size:.1f} MB)")
        model_found = True
        break

if not model_found:
    print(f"  ⚠️  Model not found (will need to download)")

print()

# Check 3: Data files
print("[3/5] Data Files")
data_paths = [
    ('../Data/processed_dataset.pkl', 'Dataset'),
    ('../../Data/processed_dataset.pkl', 'Dataset (alt)'),
]

data_found = False
for path, name in data_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {name}: {path} ({size:.1f} MB)")
        data_found = True
        break

if not data_found:
    print(f"  ⚠️  Data not found (optional for demo)")

print()

# Check 4: Experiment scripts
print("[4/5] Experiment Scripts")
scripts = [
    'exp1_violin/run_violin.py',
    'exp2_daily/run_daily_pattern.py',
    'exp3_classification/run_classification.py',
    'exp4_medical/run_enhanced.py'
]

all_scripts_ok = True
for script in scripts:
    if os.path.exists(script):
        print(f"  ✅ {script}")
    else:
        print(f"  ❌ {script} - MISSING")
        all_scripts_ok = False

print()

# Check 5: Documentation
print("[5/5] Documentation")
docs = ['README.md', 'QUICKSTART.md', 'requirements.txt', 'config.yaml']

all_docs_ok = True
for doc in docs:
    if os.path.exists(doc):
        size_kb = os.path.getsize(doc) / 1024
        print(f"  ✅ {doc} ({size_kb:.0f} KB)")
    else:
        print(f"  ❌ {doc} - MISSING")
        all_docs_ok = False

print()
print("=" * 70)

# Summary
if all_dirs_ok and all_scripts_ok and all_docs_ok:
    print("✅ Version 7 structure is COMPLETE")
    print()
    print("Ready for GitHub publication!")
    print()
    print("Next steps:")
    print("  1. Run quick_setup.sh to install dependencies (optional)")
    print("  2. Test an experiment: cd exp4_medical && python run_enhanced.py")
    print("  3. Initialize Git: git init && git add . && git commit")
else:
    print("⚠️  Some files are missing - check above")

print("=" * 70)
