#!/usr/bin/env python3
"""
Version 7 Setup and Validation Script
Tests that all components are correctly installed and configured
"""

import sys
import os

print("=" * 80)
print("CTMS Activity Pattern Analysis - Setup Validation")
print("Version 7.0.0")
print("=" * 80)

# ============================================================================
# Check Python Version
# ============================================================================
print("\n[1/8] Checking Python version...")
if sys.version_info < (3, 8):
    print(f"  ❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    print(f"  ⚠️  Python 3.8+ required")
    sys.exit(1)
else:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# ============================================================================
# Check Dependencies
# ============================================================================
print("\n[2/8] Checking dependencies...")

required_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'scikit-learn',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn'
}

missing_packages = []

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - NOT INSTALLED")
        missing_packages.append(name)

if missing_packages:
    print(f"\n  ⚠️  Missing packages: {', '.join(missing_packages)}")
    print(f"  Install with: pip install {' '.join([p.lower() for p in missing_packages])}")
    sys.exit(1)

# ============================================================================
# Check Directory Structure
# ============================================================================
print("\n[3/8] Checking directory structure...")

required_dirs = [
    'models',
    'exp1_violin',
    'exp2_daily',
    'exp3_classification',
    'exp4_medical',
    'docs'
]

for dir_name in required_dirs:
    if os.path.isdir(dir_name):
        print(f"  ✅ {dir_name}/")
    else:
        print(f"  ❌ {dir_name}/ - NOT FOUND")

# ============================================================================
# Check Model File
# ============================================================================
print("\n[4/8] Checking CTMS model...")

model_paths = [
    '../../../ctms_model_medium.pth',
    '../../ctms_model_medium.pth',
    '../ctms_model_medium.pth',
    'ctms_model_medium.pth',
    'models/ctms_model_medium.pth'
]

model_found = False
model_location = None

for path in model_paths:
    if os.path.exists(path):
        model_found = True
        model_location = path
        break

if model_found:
    size_mb = os.path.getsize(model_location) / (1024 * 1024)
    print(f"  ✅ Model found: {model_location}")
    print(f"     Size: {size_mb:.1f} MB")
else:
    print(f"  ❌ Model file not found")
    print(f"  ⚠️  Please download ctms_model_medium.pth")
    print(f"     Expected location: ../../../ctms_model_medium.pth")

# ============================================================================
# Check CTMS Model Implementation
# ============================================================================
print("\n[5/8] Checking CTMS model implementation...")

if os.path.exists('models/ctms_model.py'):
    print(f"  ✅ models/ctms_model.py")
    
    try:
        sys.path.insert(0, 'models')
        from ctms_model import CTMSModel
        
        # Try to instantiate
        model = CTMSModel(d_model=64, num_activities=22)
        print(f"  ✅ CTMSModel can be instantiated")
        print(f"     d_model=64, num_activities=22")
        
    except Exception as e:
        print(f"  ❌ Error importing CTMSModel: {e}")
else:
    print(f"  ❌ models/ctms_model.py - NOT FOUND")

# ============================================================================
# Check Data Files
# ============================================================================
print("\n[6/8] Checking data files...")

# Try multiple possible data locations
data_paths = {
    'Dataset': [
        '../Data/processed_dataset.pkl',
        '../../Data/processed_dataset.pkl',
        '../../../Data/processed_dataset.pkl'
    ],
    'Labels': [
        '../Data/subject_label_mapping_with_scores.csv',
        '../../Data/subject_label_mapping_with_scores.csv',
        '../../../Data/subject_label_mapping_with_scores.csv'
    ]
}

data_ok = True
found_paths = {}

for name, paths in data_paths.items():
    found = False
    for path in paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {name}: {path}")
            print(f"     Size: {size_mb:.1f} MB")
            found_paths[name] = path
            found = True
            break
    
    if not found:
        print(f"  ⚠️  {name} - NOT FOUND")
        print(f"     (Optional for demo, required for full experiments)")
        data_ok = False

# ============================================================================
# Check Experiment Scripts
# ============================================================================
print("\n[7/8] Checking experiment scripts...")

experiment_scripts = {
    'Experiment 1': 'exp1_violin/run_violin.py',
    'Experiment 2': 'exp2_daily/run_daily_pattern.py',
    'Experiment 3': 'exp3_classification/run_classification.py',
    'Experiment 4': 'exp4_medical/run_enhanced.py'
}

for name, path in experiment_scripts.items():
    if os.path.exists(path):
        print(f"  ✅ {name}: {path}")
    else:
        print(f"  ❌ {name}: {path} - NOT FOUND")

# ============================================================================
# Check Configuration
# ============================================================================
print("\n[8/8] Checking configuration...")

if os.path.exists('config.yaml'):
    print(f"  ✅ config.yaml found")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✅ Configuration valid")
        print(f"     Model: d_model={config['model']['d_model']}, num_activities={config['model']['num_activities']}")
    except:
        print(f"  ⚠️  PyYAML not installed (optional)")
else:
    print(f"  ⚠️  config.yaml not found (optional)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

issues = []

if missing_packages:
    issues.append("Missing Python packages")

if not model_found:
    issues.append("CTMS model file not found")

if not data_ok:
    issues.append("Data files not found (optional for demo)")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    
    print("\n📝 Next steps:")
    if missing_packages:
        print(f"  1. Install packages: pip install -r requirements.txt")
    if not model_found:
        print(f"  2. Download model: ctms_model_medium.pth")
    if not data_ok:
        print(f"  3. Prepare data (see docs/DATA_FORMAT.md)")
    
    print("\n  Then run this script again to validate.")
    
else:
    print("\n✅ All checks passed!")
    print("\n🚀 Ready to run experiments:")
    print("  - Individual: cd exp1_violin && python run_violin.py")
    print("  - All at once: bash run_all_experiments.sh")
    
    print("\n📚 Documentation:")
    print("  - README.md - Full project documentation")
    print("  - docs/DATA_FORMAT.md - Data format specifications")
    print("  - config.yaml - Configuration settings")

print("\n" + "=" * 80)
