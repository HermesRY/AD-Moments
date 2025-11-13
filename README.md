# Temporal Digital Biomarkers (TDB) for Alzheimer's Disease Detection

This repository contains a novel **Temporal Digital Biomarker (TDB) system** for early Alzheimer's Disease detection using passive sensor data.

## 📁 Repository Structure

```
.
.
├── Fixed/                           #Fixed Weight (Default)
│   ├── tdb_system.ipynb
│   ├── anomalous_time_stamps_export
│   ├── heatmap.png
│
├── LLM (Gemini)/                    #Fixed & LLM Adjusted Weights Comparison
│   ├── tdb_system_LLM.ipynb
│   └── anomalous_time_stamps_export
│   └── llm_logs for weights adjustment
│   └── medical reports
│
├── VideoLM/                         #TODO
│   └── videolm_eval.ipynb
│
├── sample_data/
│   ├── subjects.json
│   ├── action.json
│   ├── sequences.jsonl
│
├── .gitignore
└── README.md
```

## 🎯 Key Features

- **6 Temporal Behavioral Metrics** grounded in AD neuroscience
- **Multi-scale Analysis** (1-hour, 6-hour, 15-hour windows)
- **Transparent, Interpretable Scoring System**
- **Rigorous Train/Test Validation** (70/30 stratified split)

## 📊 Performance

- **Test Accuracy:** 76.19%
- **Sensitivity:** 92.31% (CI detection)
- **Specificity:** 50.00% (CN detection)
- **AUC-ROC:** 0.577

## 🚀 Getting Started

1. Clone the repository
2. Open `Fixed/tdb_system.ipynb` in Jupyter
3. Run all cells to reproduce the analysis

## 📄 Dataset

- **Population:** 68 subjects (25 CN, 20 MCI, 23 AD)
- **Data Type:** Timestamped action sequences (21 categories)
- **Source:** Passive depth-camera monitoring

## 📖 Citation

If you use this code or dataset, please cite our work.

## 📧 Contact

For questions or collaborations, please open an issue.
