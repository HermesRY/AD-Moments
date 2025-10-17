# Quick Start Guide

Get up and running with CTMS Activity Pattern Analysis in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- 2GB disk space

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/[YourUsername]/AD-Moments.git
cd AD-Moments/New_Code/Version7
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model
Download `ctms_model_medium.pth` from the [releases page](https://github.com/[YourUsername]/AD-Moments/releases) and place it in the parent directory:

```
AD-Moments/
├── ctms_model_medium.pth  ← Place here
└── New_Code/
    └── Version7/
```

### 5. Validate Setup
```bash
python setup_validate.py
```

You should see: ✅ All checks passed!

## Running Experiments

### Quick Demo (All Experiments)
```bash
bash run_all_experiments.sh
```

Results will be saved in `all_results/`

### Individual Experiments

#### Experiment 1: Violin Plots
```bash
cd exp1_violin
python run_violin.py
# Output: outputs/violin_plots.png
```

#### Experiment 2: Daily Patterns
```bash
cd exp2_daily  
python run_daily_pattern.py
# Output: outputs/daily_pattern.png
```

#### Experiment 3: Classification
```bash
cd exp3_classification
python run_classification.py
# Output: outputs/classification_results.png
```

#### Experiment 4: Medical Correlation ⭐
```bash
cd exp4_medical
python run_enhanced.py
# Output: outputs/enhanced_analysis.png (r=0.701***)
```

## Expected Results

### Experiment 1
- Task dimension shows trend towards significance (p=0.0604)
- Visual comparison of CN vs CI distributions

### Experiment 2
- CN peak activity: ~14:00 (afternoon)
- CI peak activity: ~09:00 (morning)
- Temporal pattern shift visualization

### Experiment 3
- Best classifier: XGBoost with ~75% accuracy
- AUC ~0.79

### Experiment 4 ⭐
- **Ridge Regression: r = 0.701, p < 0.0001***
- Movement dimension contributes 60% of top features
- Social dimension contributes 30%

## Data Requirements

### Your Own Data
If using your own data, ensure it follows the format:

```
Data/
├── processed_dataset.pkl          # Activity sequences
└── subject_label_mapping_with_scores.csv  # Labels & scores
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for detailed specifications.

### Demo Data
A small demo dataset with 10 subjects is available in `demo_data/` (if provided).

## Troubleshooting

### "Model file not found"
**Solution**: Ensure `ctms_model_medium.pth` is in the correct location (3 levels up from Version7)

### "Data files not found"
**Solution**: 
- Check data is in `../Data/` directory
- Or modify paths in experiment scripts

### "ModuleNotFoundError"
**Solution**: 
```bash
pip install -r requirements.txt
```

### ImportError for ctms_model
**Solution**: Ensure you're running from within experiment directories:
```bash
cd exp1_violin  # Must be in experiment directory
python run_violin.py
```

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Understand data format**: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
3. **Customize configuration**: Edit `config.yaml`
4. **Explore results**: Check `all_results/` or `exp*/outputs/`

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Open an issue on GitHub
- **Questions**: Contact the authors

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python setup_validate.py` | Validate installation |
| `bash run_all_experiments.sh` | Run all experiments |
| `cd expN && python run_*.py` | Run individual experiment |
| `ls all_results/` | View all results |
| `open all_results/exp4_enhanced.png` | View best result (Exp 4) |

---

**Ready to analyze activity patterns? Start with Experiment 4 for the strongest results!** 🚀

```bash
cd exp4_medical
python run_enhanced.py
open outputs/enhanced_analysis.png
```

You should see: **r = 0.701, p < 0.0001*** 🎉
