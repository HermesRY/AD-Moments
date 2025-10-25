# AD-Moments

**Leveraging LLM to Grasp Personalized Behavioral Anomaly Highlights for Alzheimer's Monitoring**

This repository contains the source code and experimental data for the AD-Moments project, which implements CTMS (Circadian-Task-Movement-Social) temporal encoders with multi-scale behavioral analysis for early Alzheimer's disease detection.

---

## 📖 Overview

AD-Moments is the first system designed to identify **personalized temporal digital biomarkers** indicative of Alzheimer's disease from activities of daily living (ADL). The system combines:

- **Multi-scale Temporal Analysis**: Captures behavioral patterns across three temporal scales
  - **Micro-scale (5-30 secs)**: Immediate activity transitions indicating cognitive lapses
  - **Meso-scale (1-10 mins)**: Task completion logic and executive dysfunction
  - **Macro-scale (hours-days)**: Circadian rhythm stability and daily patterns

- **Four-Dimensional CTMS Behavioral Space**:
  - **C**ircadian: Daily rhythm regularity
  - **T**ask: Goal-directed activity completion
  - **M**ovement: Spatial navigation patterns
  - **S**ocial: Interaction engagement quality

- **LLM-Guided Personalization**: Medical literature-augmented LLM refines anomaly boundaries and adapts model weights based on individual variability

---

## 🗂️ Repository Structure

```
AD-Moments/
├── models/                          # Core implementation
│   ├── ctms_model.py               # Main CTMS model implementation
│   ├── ctms_model_gpu.py           # GPU-accelerated version
│   ├── ctms_data.py                # Data loading and preprocessing
│   ├── main.tex                    # Paper manuscript
│   │
│   ├── optimized_CTMS_search_gpu.ipynb     # Hyperparameter search
│   ├── CTMS_train_with_optimal_config.ipynb # Training with optimal config
│   │
│   ├── best_with_config.json              # Best sample config with personalization
│   ├── best_without_config.json           # Best sample config without personalization
│   ├── best_configs_summary.json          # Summary of all configs
│   │
│   ├── trained_model_with_personalization.pt    # Trained model
│   ├── trained_model_without_personalization.pt
│   │
│   └── abnormalTS/                 # Detected anomalous timestamps per subject
│       ├── summary.json
│       └── *_abnormal_timestamps.json
│
├── sample_data/                    # Sample dataset
│   ├── sequences.jsonl            # Activity sequences
│   ├── subjects.json              # Subject labels and metadata
│   ├── action_map.json            # Activity ID mappings
│   └── summary.json               # Dataset statistics
│
├── figures                         # Paper figures (if included)
│
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas scikit-learn tqdm
pip install fastdtw scipy
```

### Basic Usage

```python
from models.ctms_model import CTMSModel
import torch

# Initialize model
model = CTMSModel(
    d_model=128,
    num_activities=21,
    num_task_templates=20,
    num_fusion_heads=4
)

# Prepare input
# activity_ids: [batch_size, seq_len] - activity type indices (0-20)
# timestamps: [batch_size, seq_len] - Unix timestamps
activity_ids = torch.randint(0, 21, (4, 100))
timestamps = torch.randint(0, 86400*30, (4, 100))

# Forward pass
outputs = model(activity_ids, timestamps)

# Access multi-scale encodings
h_c = outputs['h_c']  # Circadian encoding (macro-scale)
h_t = outputs['h_t']  # Task encoding (meso-scale)
h_m = outputs['h_m']  # Movement encoding (micro-scale)
h_s = outputs['h_s']  # Social encoding (cross-scale)

# Access behavioral metrics
cdi = outputs['cdi']  # Circadian Disruption Index
tir = outputs['tir']  # Task Incompletion Rate
me = outputs['me']    # Movement Entropy
sws = outputs['sws']  # Social Withdrawal Score

# Access adaptive fusion weights
alpha = outputs['alpha']  # [batch, 4] = [α_c, α_t, α_m, α_s]
```

### GPU-Accelerated Version

For faster inference on large datasets:

```python
from models.ctms_model_gpu import CTMSModelGPU

model = CTMSModelGPU(
    d_model=128,
    num_activities=21,
    num_task_templates=20,
    use_fast_similarity=True  # Use pooled-cosine instead of DTW
).cuda()

outputs = model(activity_ids.cuda(), timestamps.cuda())
```

---

## 📊 CTMS Architecture

### Four-Dimensional Encoders

| Encoder | Temporal Scale | Architecture | Output Metric |
|---------|---------------|--------------|---------------|
| **Circadian (C)** | Macro (hours-days) | 3-layer Transformer | CDI (Jensen-Shannon divergence) |
| **Task (T)** | Meso (1-10 mins) | BiLSTM + Cross-Attention | TIR (DTW-based similarity) |
| **Movement (M)** | Micro (5-30 secs) | Graph Attention Network | ME (Transition entropy) |
| **Social (S)** | Cross-scale | 1D CNN + Statistics | SWS (Withdrawal score) |

### Multi-Head Attention Fusion

```
H = [h_c; h_t; h_m; h_s] ∈ R^{4×d}
H_attn, A = MultiHeadAttn(H, H, H)
α_d = (1/H)∑_h (1/4)∑_q A_{h,q,d}
h_fused = ∑_d α_d × H_attn[d]
```

### Personalization

```python
# Update fusion weights based on LLM recommendations
model.update_user_alpha(
    user_id="patient_001",
    alpha_llm=torch.tensor([0.4, 0.3, 0.2, 0.1]),  # [α_c, α_t, α_m, α_s]
    learning_rate=0.1
)

# Use personalized weights in forward pass
outputs = model(activity_ids, timestamps, user_ids=["patient_001"])
```

---

## 🔬 Experiments

### Training with Optimal Configuration

See `models/CTMS_train_with_optimal_config.ipynb` for the complete training pipeline:

1. Load optimal hyperparameters from `best_with_config.json`
2. Train CTMS model on cognitively normal (CN) participants
3. Evaluate on test set with cognitively impaired (CI) participants
4. Compute performance metrics and visualizations

### Hyperparameter Search

See `models/optimized_CTMS_search_gpu.ipynb` for:

- Grid search over learning rates, batch sizes, model dimensions
- Cross-validation on CN participants
- Fixed false positive rate (FPR) evaluation
- Configuration export for reproducibility

## 📝 Data Format

### Input Sequences (`sequences.jsonl`)

```json
{
  "anon_id": 1,
  "month": "2024-01",
  "sequence": [
    {"ts": 1704067200, "action_id": 5},
    {"ts": 1704067230, "action_id": 8},
    ...
  ]
}
```

### Subject Labels (`subjects.json`)

```json
{
  "anon_id": 1,
  "label": "CN",
  "label_binary": 0,
  "age": 72,
  "gender": "F",
  "moca_score": 28
}
```

### Activity Mapping (`action_map.json`)

21 activities covering:
- **BADL** (Basic ADL): Walk, Stand, Sit, Sleep, Eat, Drink, etc.
- **IADL** (Instrumental ADL): Clean, Cook, Groom, etc.
- **SI** (Social Interaction): Phone, Talk, etc.

---

## 🔧 Model Components

### 1. Circadian Encoder (Macro-Scale)

```python
class CircadianEncoder(nn.Module):
    """
    Temporal Scale: MACRO-SCALE (hours-days)
    - Processes 24-hour activity distributions
    - Uses 3-layer Transformer with positional encodings
    - Outputs CDI via Jensen-Shannon divergence
    """
```

**Paper Reference:** Lines 314-318 (main.tex)

### 2. Task Completion Encoder (Meso-Scale)

```python
class TaskCompletionEncoder(nn.Module):
    """
    Temporal Scale: MESO-SCALE (1-10 minutes)
    - Sliding window of size 20 (~10 mins)
    - BiLSTM encoding + Cross-attention with task templates
    - Outputs TIR via DTW similarity
    """
```

**Paper Reference:** Lines 321-325 (main.tex)

### 3. Movement Pattern Encoder (Micro-Scale)

```python
class MovementPatternEncoder(nn.Module):
    """
    Temporal Scale: MICRO-SCALE (5-30 seconds)
    - 21-node activity transition graph
    - Graph Attention Network (GAT)
    - Outputs ME via transition entropy
    """
```

**Paper Reference:** Lines 338-341 (main.tex)

### 4. Social Interaction Encoder (Cross-Scale)

```python
class SocialInteractionEncoder(nn.Module):
    """
    Temporal Scale: CROSS-SCALE (minutes to hours)
    - 1D CNN + statistical features
    - Captures duration, frequency, response time
    - Outputs SWS via withdrawal scoring
    """
```

**Paper Reference:** Line 344 (main.tex)

---

## 📚 Documentation

- **`models/TEMPORAL_SCALE_MAPPING.md`**: Detailed technical documentation on temporal scale implementation
- **`models/main.tex`**: Full paper manuscript with all equations and references
- **Code Comments**: All classes and functions include comprehensive docstrings with paper references

---

## 🎯 Key Features

- ✅ **Multi-scale temporal analysis** (micro/meso/macro)
- ✅ **Four-dimensional behavioral space** (CTMS)
- ✅ **GPU-accelerated inference** with fast similarity approximation
- ✅ **Personalized fusion weights** via LLM guidance
- ✅ **DTW-based task similarity** for precise pattern matching
- ✅ **Graph-based movement modeling** for transition patterns
- ✅ **Comprehensive behavioral metrics** (CDI, TIR, ME, SWS)
- ✅ **Reproducible experiments** with config export

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fu2025admoments,
  title={AD-Moments: Leveraging LLM to Grasp Personalized Behavioral Anomaly Highlights for Alzheimer's Monitoring},
  author={Fu, Heming and Chen, Hongkai and Lin, Shan and Xing, Guoliang},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

## 📌 Notes

### Sample Data

The `sample_data/` directory contains anonymized sample sequences for demonstration purposes. For full dataset access, please contact the authors.

### Model Weights

Pre-trained models are provided in:
- `models/trained_model_with_personalization.pt`
- `models/trained_model_without_personalization.pt`

Load with:
```python
model = CTMSModel(...)
model.load_state_dict(torch.load('models/trained_model_with_personalization.pt'))
```

### Performance Optimization

For large-scale experiments (>1000 sequences):
1. Use `CTMSModelGPU` with `use_fast_similarity=True`
2. Enable mixed precision training with `torch.cuda.amp`
3. Batch sequences of similar length for efficient padding

---

**Last Updated:** 2025-10-25
