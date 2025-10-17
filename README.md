# CTMS-Based Activity Pattern Analysis for Cognitive Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive framework for analyzing daily activity patterns using Circadian-Task-Movement-Social (CTMS) model encodings to assess cognitive function in older adults.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This repository contains the implementation of a multi-dimensional activity pattern analysis framework that leverages the **CTMS (Circadian-Task-Movement-Social)** model to extract behavioral encodings from daily activity data. The framework enables **non-invasive cognitive assessment** through analysis of routine activities.

### Research Highlights

✨ **Strong Clinical Correlation**: Achieved **r = 0.701 (p < 0.0001)*** correlation with MoCA cognitive scores using Ridge Regression on engineered features

✨ **Multi-Dimensional Analysis**: Comprehensive evaluation across four behavioral dimensions:
- 🌙 **Circadian**: Daily rhythm patterns
- 📋 **Task**: Activity completion patterns  
- 🏃 **Movement**: Physical activity patterns
- 👥 **Social**: Social engagement patterns

✨ **Robust Methodology**: Validated through multiple experimental approaches including violin plots, temporal analysis, classification, and correlation studies

---

## 🌟 Key Features

### Experiment 1: Violin Plot Analysis
- **Objective**: Compare distribution differences between cognitively normal (CN) and cognitively impaired (CI) groups across four CTMS dimensions
- **Method**: Statistical testing with violin plots for visual comparison
- **Key Finding**: Task dimension shows near-significant difference (p = 0.0604)

### Experiment 2: Daily Temporal Pattern Analysis  
- **Objective**: Analyze hourly activity patterns throughout the day (6:30 AM - 7:30 PM)
- **Method**: Time-binned encoding analysis with 27 time windows
- **Key Finding**: CI group peaks earlier (9:00 AM) vs CN group (2:00 PM)

### Experiment 3: Classification Analysis
- **Objective**: Discriminate CN from CI using CTMS encodings
- **Method**: Multiple classifiers (Logistic Regression, SVM, Random Forest, XGBoost)
- **Key Finding**: Achieves up to 75% accuracy with ensemble methods

### Experiment 4: Clinical Correlation Analysis ⭐
- **Objective**: Predict cognitive scores (MoCA, ZBI, DSS, FAS) from CTMS features
- **Method**: Ridge Regression with 80+ engineered features
- **Key Finding**: **r = 0.701 (p < 0.0001)*** for MoCA prediction, with Movement dimension contributing 60% of top features

---

## 💻 System Requirements

### Dependencies
```
torch >= 1.9.0
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/[YourUsername]/AD-Moments.git
cd AD-Moments/New_Code/Version7
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
Download `ctms_model_medium.pth` from [releases](https://github.com/[YourUsername]/AD-Moments/releases) and place it in the project root directory.

### Step 5: Prepare Data
Ensure your data follows the required format:
```
Data/
├── processed_dataset.pkl          # Activity sequences
└── subject_label_mapping_with_scores.csv  # Clinical labels & scores
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for detailed specifications.

---

## ⚡ Quick Start

### Run All Experiments
```bash
# From Version7 directory
bash run_all_experiments.sh
```

### Run Individual Experiments

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

#### Experiment 4: Clinical Correlation
```bash
cd exp4_medical
python run_enhanced.py
# Output: outputs/enhanced_analysis.png
```

---

## 🔬 Experiments

### Detailed Experiment Descriptions

#### 1️⃣ Violin Plot Analysis (`exp1_violin/`)

**Purpose**: Visualize and test distribution differences between CN and CI groups

**Method**:
1. Extract CTMS encodings from activity sequences (seq_len=30, stride=10)
2. Compute z-scores normalized to CN baseline
3. Statistical testing using Mann-Whitney U test
4. Visualization with violin plots

**Key Parameters**:
- Model: `ctms_model_medium.pth` (d_model=64, num_activities=22)
- Subjects: 57 total (21 CN, 36 CI)
- Sequences: ~100-400 per subject

**Output Files**:
- `violin_plots.png`: Main visualization
- `statistics.csv`: Statistical test results

**Results**:
| Dimension | CN Mean ± SD | CI Mean ± SD | p-value | Significance |
|-----------|-------------|-------------|---------|--------------|
| Circadian | 0.000 ± 1.000 | 0.318 ± 0.431 | 0.1078 | ns |
| **Task** | 0.000 ± 1.000 | 0.354 ± 0.330 | **0.0604** | Trend |
| Movement | 0.000 ± 1.000 | 0.212 ± 0.878 | 0.4151 | ns |
| Social | 0.000 ± 1.000 | -0.059 ± 0.732 | 0.8036 | ns |

---

#### 2️⃣ Daily Temporal Pattern Analysis (`exp2_daily/`)

**Purpose**: Analyze hourly variation in activity patterns

**Method**:
1. Segment data into 27 hourly bins (6:30 AM - 7:30 PM)
2. Extract encodings for each time bin
3. Compute mean encoding magnitude per hour
4. Compare CN vs CI temporal profiles

**Key Parameters**:
- Time range: 6:30 - 19:30 (13 hours)
- Bin size: 30 minutes
- Minimum sequences per bin: 5

**Output Files**:
- `daily_pattern.png`: Temporal curve with confidence intervals
- `daily_pattern_data.npz`: Raw pattern data

**Results**:
- **CN Peak**: 14:00 (afternoon activity)
- **CI Peak**: 09:00 (morning activity)  
- **CI/CN Ratio**: 1.02× (minimal overall difference but shifted timing)

---

#### 3️⃣ Classification Analysis (`exp3_classification/`)

**Purpose**: Evaluate CN vs CI discrimination capability

**Method**:
1. Feature extraction: Mean encodings across all sequences per subject
2. Train-test split: 80/20 stratified
3. Multiple classifiers:
   - Logistic Regression (L2 penalty)
   - Support Vector Machine (RBF kernel)
   - Random Forest (100 trees)
   - XGBoost (gradient boosting)
4. 5-fold cross-validation
5. Performance metrics: Accuracy, Precision, Recall, F1, AUC

**Key Parameters**:
- Features: 4-dimensional CTMS encodings
- Imbalance handling: Class weights
- Hyperparameter tuning: GridSearchCV

**Output Files**:
- `classification_results.png`: Performance comparison
- `confusion_matrices.png`: Confusion matrices for all models
- `feature_importance.png`: Feature importance ranking
- `results_summary.csv`: Detailed metrics

**Results**:
| Classifier | Accuracy | Precision | Recall | F1 Score | AUC |
|-----------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.68 | 0.72 | 0.65 | 0.68 | 0.71 |
| SVM (RBF) | 0.70 | 0.73 | 0.68 | 0.70 | 0.74 |
| Random Forest | 0.73 | 0.75 | 0.70 | 0.72 | 0.76 |
| **XGBoost** | **0.75** | **0.77** | **0.72** | **0.74** | **0.79** |

---

#### 4️⃣ Clinical Correlation Analysis (`exp4_medical/`) ⭐

**Purpose**: Quantify relationship between CTMS features and clinical assessments

**Method**:
1. **Feature Engineering**: Extract 80+ features per subject
   - Norm features: mean, std, max, min of L2 norms
   - Statistical features: mean, std, skewness, kurtosis
   - Temporal features: variability, trend
2. **Ridge Regression**: L2-regularized linear model (α=1.0)
3. **Correlation Analysis**: Pearson correlation with clinical scores
4. **Feature Importance**: Identify top predictive features

**Key Parameters**:
- Features: 80+ (40 per dimension × 4 dimensions)
- Subjects: 48 with complete MoCA scores
- Regularization: Ridge (α=1.0)
- Imputation: Median strategy for missing values

**Output Files**:
- `enhanced_analysis.png`: 6-panel comprehensive analysis
- `table4_enhanced.png`: Professional correlation table
- `all_features.csv`: Complete feature matrix
- `enhanced_results.json`: Detailed statistical results

**Results**:

**Overall Performance**:
- **MoCA Correlation**: r = 0.701, p = 5.76×10⁻⁸ ***
- **Explained Variance**: R² = 49.1%
- **Sample Size**: n = 48

**Top 10 Predictive Features**:
| Rank | Feature | Coefficient | Dimension | Interpretation |
|------|---------|-------------|-----------|----------------|
| 1 | social_min_norm | -5.77 | Social | Lower social activity → Lower cognition |
| 2 | movement_mean | -5.30 | Movement | Reduced movement → Lower cognition |
| 3 | social_max_norm | -4.22 | Social | Narrower social range → Lower cognition |
| 4 | circadian_max_norm | +4.04 | Circadian | Stronger rhythm → Higher cognition |
| 5 | movement_std | -3.82 | Movement | Less variability → Lower cognition |
| 6 | movement_mean_norm | -3.50 | Movement | Lower activity level → Lower cognition |
| 7 | movement_std_norm | -3.50 | Movement | Reduced diversity → Lower cognition |
| 8 | movement_trend | -3.50 | Movement | Declining trend → Lower cognition |
| 9 | social_trend | -3.33 | Social | Declining social → Lower cognition |
| 10 | movement_skew | +3.20 | Movement | Distribution asymmetry |

**Dimension Contributions**:
- **Movement**: 60% (6/10 features) - **Primary predictor**
- **Social**: 30% (3/10 features) - **Important complement**
- **Circadian**: 10% (1/10 features) - **Secondary role**
- **Task**: 0% (0/10 features) - Not significant in this cohort

**Clinical Assessments Correlation**:
| Assessment | Correlation with MoCA | Sample Size | Interpretation |
|------------|----------------------|-------------|----------------|
| ZBI (Caregiver Burden) | r = -0.582** | n = 30 | Higher burden with lower cognition |
| DSS (Dementia Severity) | r = 0.005 ns | n = 23 | No significant association |
| FAS (Functional Assessment) | r = -0.218 ns | n = 23 | Weak negative association |

---

## 📊 Results

### Summary of Main Findings

#### Scientific Discoveries

1. **Multi-dimensional Integration is Essential**
   - Single dimensions: r < 0.15 (all non-significant)
   - Combined features: r = 0.70*** (highly significant)
   - **Implication**: Cognitive function requires multi-system assessment

2. **Movement Patterns as Core Biomarker**
   - 60% of top features from Movement dimension
   - **Key insight**: Activity **diversity** > absolute activity level
   - **Supports**: Cognitive Reserve Theory

3. **Social Engagement Matters**
   - 30% of top features from Social dimension
   - Both **range** (min/max) and **dynamics** (trend) important
   - **Supports**: Social Engagement Hypothesis

4. **Temporal Dynamics are Predictive**
   - Trend and variability features enter top 10
   - **Implication**: Monitor **trajectories** not just snapshots

5. **Circadian Regularity has Protective Role**
   - Positive coefficient for circadian_max_norm
   - **Implication**: Maintaining daily routine beneficial

### Performance Comparison

| Approach | Features | MoCA Correlation | p-value | Improvement |
|----------|----------|-----------------|---------|-------------|
| Baseline (Z-score) | 4 | r = 0.174 | p = 0.247 ns | - |
| Simple Weighted | 4 | r = 0.150 | p = 0.320 ns | -14% |
| **Ridge Regression** | **80+** | **r = 0.701** | **p < 0.0001*** | **+303%** |

---

## 📁 Repository Structure

```
Version7/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_all_experiments.sh            # Master script to run all experiments
├── setup.py                          # Package setup
│
├── docs/                             # Documentation
│   ├── DATA_FORMAT.md               # Data format specifications
│   ├── MODEL_ARCHITECTURE.md        # CTMS model details
│   └── TROUBLESHOOTING.md           # Common issues & solutions
│
├── models/                           # Model definitions
│   ├── ctms_model.py                # CTMS model implementation
│   └── ctms_model_medium.pth        # Pre-trained weights (download separately)
│
├── utils/                            # Shared utilities
│   ├── data_loader.py               # Data loading functions
│   ├── feature_extractor.py         # Feature engineering
│   └── visualization.py             # Plotting utilities
│
├── exp1_violin/                      # Experiment 1: Violin plots
│   ├── README.md                    # Experiment documentation
│   ├── run_violin.py                # Main script
│   └── outputs/                     # Generated results
│
├── exp2_daily/                       # Experiment 2: Daily patterns
│   ├── README.md                    
│   ├── run_daily_pattern.py         
│   └── outputs/                     
│
├── exp3_classification/              # Experiment 3: Classification
│   ├── README.md                    
│   ├── run_classification.py        
│   └── outputs/                     
│
├── exp4_medical/                     # Experiment 4: Clinical correlation
│   ├── README.md                    
│   ├── run_enhanced.py              # Ridge regression analysis
│   ├── create_correlation_table.py  # Generate publication tables
│   └── outputs/                     
│
└── tests/                            # Unit tests
    ├── test_data_loader.py          
    ├── test_feature_extractor.py    
    └── test_models.py               
```

---

## 🔧 Configuration

### Global Configuration File: `config.yaml`

```yaml
# Model Configuration
model:
  checkpoint: "models/ctms_model_medium.pth"
  d_model: 64
  num_activities: 22
  device: "cpu"  # or "cuda"

# Data Configuration
data:
  dataset_path: "Data/processed_dataset.pkl"
  labels_path: "Data/subject_label_mapping_with_scores.csv"
  
# Sequence Extraction
sequences:
  seq_len: 30
  stride: 10
  batch_size: 32

# Experiment-Specific Settings
exp1_violin:
  min_sequences: 10
  alpha: 0.05

exp2_daily:
  start_hour: 6.5
  end_hour: 19.5
  num_bins: 27
  
exp3_classification:
  test_size: 0.2
  cv_folds: 5
  random_state: 42

exp4_medical:
  ridge_alpha: 1.0
  imputer_strategy: "median"
```

---

## 🧪 Testing

Run unit tests to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_feature_extractor.py -v

# Generate coverage report
python -m pytest --cov=. --cov-report=html
```

---

## 📖 Documentation

Detailed documentation for each component:

- **[Data Format Guide](docs/DATA_FORMAT.md)**: Input data specifications
- **[Model Architecture](docs/MODEL_ARCHITECTURE.md)**: CTMS model details
- **[API Reference](docs/API.md)**: Function and class documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
flake8 .
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
