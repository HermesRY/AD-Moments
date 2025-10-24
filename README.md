# AD-Moments: CTMS Model Implementation

This branch contains the implementation and experimental results of the CTMS (Circadian-Task-Movement-Social) temporal model for Alzheimer's disease detection.

## 📁 Repository Structure

```
Moments/
├── models/
│   ├── ctms_model.py              # Complete CTMS model implementation
│   ├── ctms_data.py               # Dataset and data loading utilities
│   ├── usage_example.py           # Basic usage example
│   ├── test_CTMS_model.ipynb      # Full training and evaluation notebook
│   ├── main.tex                   # Paper LaTeX source
│   ├── best_ctms_model.pt         # Trained model checkpoint (62.7 MB)
│   ├── baseline_stats.pt          # CN baseline statistics
│   └── *.png                      # Visualization results
└── sample_data/
    ├── best_month_sequences.jsonl # Activity sequences
    ├── subjects_public.json       # Participant labels (CN/CI)
    ├── action_map.json            # Activity type mappings
    └── summary.json               # Dataset statistics
```

## 🎯 Key Features

### CTMS Model Components

1. **Circadian Encoder (Transformer)** - Lines 316-322 in main.tex
   - Captures daily rhythm disruptions
   - Outputs: CDI (Circadian Disruption Index)

2. **Task Completion Encoder (BiLSTM + Attention)** - Lines 324-330
   - Analyzes goal-directed activity patterns
   - Outputs: TIR (Task Incompletion Rate) via DTW similarity

3. **Movement Pattern Encoder (GAT)** - Lines 341-345
   - Models activity transition graphs
   - Outputs: ME (Movement Entropy)

4. **Social Interaction Encoder (CNN1D)** - Lines 347-349
   - Tracks social engagement patterns
   - Outputs: SWS (Social Withdrawal Score)

5. **Multi-Head Attention Fusion** - Lines 351-356
   - Adaptive dimensional weight learning
   - Personalization mechanism

### Behavioral Metrics

- **CDI**: Circadian Disruption Index (Jensen-Shannon divergence)
- **TIR**: Task Incompletion Rate (DTW-based similarity)
- **ME**: Movement Entropy (transition probability entropy)
- **SWS**: Social Withdrawal Score (engagement decline)

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision
pip install torch-geometric
pip install fastdtw scipy numpy pandas matplotlib seaborn tqdm
```

### Basic Usage

```python
from ctms_model import CTMSModel
import torch

# Initialize model
model = CTMSModel(
    d_model=128,
    num_activities=21,
    num_task_templates=20
)

# Forward pass
activity_ids = torch.randint(0, 21, (32, 100))  # [batch, seq_len]
timestamps = torch.randint(1675350536, 1675436936, (32, 100))

outputs = model(activity_ids, timestamps)

# Outputs include:
# - h_c, h_t, h_m, h_s: Dimensional encodings [32, 128]
# - cdi, tir, me, sws: Behavioral metrics [32]
# - alpha: Fusion weights [32, 4]
# - output: Classification logits [32]
```

### Training and Evaluation

See `test_CTMS_model.ipynb` for complete pipeline:
1. Data loading and preprocessing
2. Model training (CN vs CI classification)
3. CN baseline statistics computation
4. Anomaly score calculation
5. Visualization and statistical analysis

## 📊 Experimental Results

### Model Performance
- **Validation Accuracy**: ~85%
- **Sensitivity**: ~84.6%
- **Specificity**: ~85%
- **AUC**: 0.94

### CN vs CI Comparison
- CI participants show **1.61× higher** AD manifestation episodes
- All dimensional scores significantly different (p < 0.001)
- Strong correlation with clinical assessments (MoCA scores)

### Key Findings
1. **Circadian Dimension**: CI shows increased nighttime activity and irregular patterns
2. **Task Dimension**: Higher incompletion rates in CI group
3. **Movement Dimension**: Increased entropy and disorganized transitions
4. **Social Dimension**: Reduced engagement duration and frequency

## 📝 Files Description

### Model Files
- `ctms_model.py`: Complete model with all encoders, fusion, and anomaly detection
- `ctms_data.py`: Dataset class for loading JSONL sequences
- `usage_example.py`: Minimal working example

### Experiment Files
- `test_CTMS_model.ipynb`: Full experimental pipeline
- `best_ctms_model.pt`: Trained model checkpoint
- `baseline_stats.pt`: CN participant baseline statistics

### Visualization Files
- `training_history.png`: Training/validation curves
- `cn_vs_ci_comparison.png`: Box plots of anomaly scores
- `behavioral_metrics_comparison.png`: Violin plots of CDI/TIR/ME/SWS
- `roc_curve.png`: ROC analysis

### Data Files
- `best_month_sequences.jsonl`: Activity sequences (one sample per line)
- `subjects_public.json`: Participant metadata and labels
- `action_map.json`: Activity ID to name mapping

## 🔬 Paper Correspondence

This implementation follows the paper "AD-Moments" Section 2.4 exactly:
- **Line 273-276**: CDI computation
- **Line 283-289**: TIR with DTW
- **Line 295-297**: Movement Entropy
- **Line 303-305**: Social Withdrawal Score
- **Line 316-356**: CTMS encoders and fusion

All line numbers refer to `main.tex`.

## ⚠️ Notes

1. **Large File Warning**: `best_ctms_model.pt` is 62.7 MB. Consider using Git LFS if you plan to frequently update model files.

2. **Data Privacy**: Sample data in `sample_data/` is anonymized and contains only public information.

3. **Reproducibility**: Random seeds are set in the notebook for reproducible results.

## 📧 Contact

For questions about the implementation:
- Heming Fu - heming.fu@stonybrook.edu
- GitHub: @HermesRY

## 📄 License

See main repository LICENSE file.

## 🙏 Acknowledgments

This work is part of the AD-Moments project for early Alzheimer's disease detection using temporal digital biomarkers.

---

**Last Updated**: October 24, 2025  
**Branch**: `ctms-implementation`  
**Commit**: Initial implementation with full experimental pipeline
