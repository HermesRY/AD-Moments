# Data Format Specification

This document describes the required data format for the CTMS Activity Pattern Analysis framework.

---

## Overview

The framework requires two main data files:
1. **Activity Dataset** (`processed_dataset.pkl`) - Time-series activity sequences
2. **Subject Labels** (`subject_label_mapping_with_scores.csv`) - Clinical labels and scores

---

## 1. Activity Dataset (`processed_dataset.pkl`)

### Format
Python pickle file containing a dictionary with the following structure:

```python
{
    'subjects': {
        'subject_id_1': {
            'data': pandas.DataFrame,
            'label': str  # 'CN' or 'CI'
        },
        'subject_id_2': {
            'data': pandas.DataFrame,
            'label': str
        },
        ...
    }
}
```

### DataFrame Columns

Each subject's `data` DataFrame must contain the following columns:

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `action_label` | int | Activity type ID | 0-21 (22 classes) |
| `hour` | float | Hour of day | 0.0-23.99 |
| `day_of_week` | int | Day of week | 0-6 (Mon-Sun) |
| `timestamp` | datetime | Event timestamp | ISO format (optional) |

### Activity Classes (action_label)

The 22 activity classes should follow this encoding:

| ID | Activity | Category |
|----|----------|----------|
| 0 | Lying | Rest |
| 1 | Sitting | Sedentary |
| 2 | Standing | Static |
| 3 | Walking | Movement |
| 4 | Running | Movement |
| 5 | Turning | Movement |
| 6 | Eating | ADL |
| 7 | Drinking | ADL |
| 8 | Cooking | ADL |
| 9 | Cleaning | ADL |
| 10 | Washing | ADL |
| 11 | Toileting | ADL |
| 12 | Reading | Cognitive |
| 13 | Writing | Cognitive |
| 14 | Watching TV | Leisure |
| 15 | Listening Music | Leisure |
| 16 | Talking | Social |
| 17 | Phone Call | Social |
| 18 | Visiting | Social |
| 19 | Shopping | IADL |
| 20 | Gardening | IADL |
| 21 | Other | Misc |

### Example

```python
import pandas as pd

# Example data for one subject
subject_data = pd.DataFrame({
    'action_label': [0, 1, 3, 6, 7, 3, 2, ...],
    'hour': [7.5, 8.0, 8.5, 12.0, 12.25, 15.0, 18.5, ...],
    'day_of_week': [0, 0, 0, 0, 0, 0, 0, ...],
    'timestamp': ['2024-01-01 07:30:00', '2024-01-01 08:00:00', ...]
})

# Full dataset structure
dataset = {
    'subjects': {
        'NX001': {
            'data': subject_data,
            'label': 'CN'
        },
        'NX002': {
            'data': subject_data_2,
            'label': 'CI'
        }
    }
}

# Save as pickle
import pickle
with open('processed_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

---

## 2. Subject Labels (`subject_label_mapping_with_scores.csv`)

### Format
CSV file with subject-level information and clinical scores.

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `subject_id` | str | Unique subject identifier | "NX001", "V023" |
| `label` | str | Diagnostic label | "CN", "CI" |

### Optional Columns (Clinical Scores)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `MoCA Score` | float | Montreal Cognitive Assessment | 0-30 |
| `ZBI Score` | float | Zarit Burden Interview | 0-88 |
| `DSS Score` | float | Dementia Severity Scale | 0-60 |
| `FAS Score` | float | Functional Assessment Score | 0-100 |
| `age` | int | Subject age | years |
| `gender` | str | Subject gender | "M", "F" |

### Example

```csv
subject_id,label,MoCA Score,ZBI Score,DSS Score,FAS Score,age,gender
NX001,CN,28.0,15.0,8.0,92.0,68,F
NX002,CI,18.0,45.0,35.0,55.0,75,M
NX003,CN,27.0,12.0,5.0,95.0,70,F
V023,CI,15.0,52.0,42.0,48.0,78,M
```

---

## Data Quality Requirements

### Minimum Requirements

1. **Sample Size**
   - Minimum 20 subjects total
   - At least 10 subjects per group (CN/CI)
   - Recommended: 50+ subjects for robust analysis

2. **Data Completeness**
   - Each subject must have at least 10 sequences (seq_len=30)
   - Recommended: 100+ sequences per subject
   - Activity sequences should cover multiple days

3. **Clinical Scores**
   - For Experiment 4, at least 30 subjects with MoCA scores
   - Other scores optional but recommended

### Data Quality Checks

Run the provided validation script before analysis:

```bash
python utils/validate_data.py --dataset Data/processed_dataset.pkl \
                               --labels Data/subject_label_mapping_with_scores.csv
```

This will check for:
- Missing values
- Invalid activity labels (outside 0-21)
- Invalid hour values (outside 0-23.99)
- Subject ID mismatches
- Minimum sequence requirements
- Clinical score ranges

---

## Data Preprocessing

### Recommended Preprocessing Steps

1. **Remove Outliers**
   ```python
   # Remove extreme hour values
   data = data[(data['hour'] >= 0) & (data['hour'] < 24)]
   
   # Clip action labels to valid range
   data['action_label'] = data['action_label'].clip(0, 21)
   ```

2. **Handle Missing Values**
   ```python
   # Drop rows with missing critical columns
   data = data.dropna(subset=['action_label', 'hour'])
   
   # Fill day_of_week if missing
   data['day_of_week'] = data['day_of_week'].fillna(0)
   ```

3. **Sort by Time**
   ```python
   # Ensure chronological order
   data = data.sort_values('timestamp')
   ```

4. **Normalize Subject IDs**
   ```python
   # Ensure consistent casing
   subject_id = subject_id.strip().lower()
   ```

---

## Creating Your Own Dataset

### Step 1: Collect Raw Data

Collect activity data from sensors or wearables. Each record should have:
- Timestamp
- Activity type
- (Optional) Location, duration, etc.

### Step 2: Label Activities

Map your activities to the 22-class schema. You may need to:
- Combine similar activities
- Split complex activities
- Add "Other" (class 21) for unclassifiable events

### Step 3: Extract Temporal Features

```python
import pandas as pd

# Parse timestamps
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract hour (0-23.99)
data['hour'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60

# Extract day of week (0=Monday, 6=Sunday)
data['day_of_week'] = data['timestamp'].dt.dayofweek
```

### Step 4: Create Subject Dictionary

```python
dataset = {'subjects': {}}

for subject_id in subject_ids:
    subject_data = data[data['subject_id'] == subject_id]
    
    dataset['subjects'][subject_id] = {
        'data': subject_data[['action_label', 'hour', 'day_of_week']],
        'label': get_label(subject_id)  # 'CN' or 'CI'
    }
```

### Step 5: Save and Validate

```python
import pickle

# Save dataset
with open('processed_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# Validate
!python utils/validate_data.py --dataset processed_dataset.pkl --labels labels.csv
```

---

## Troubleshooting

### Common Issues

**Issue**: "Subject not found in dataset"
- **Cause**: Subject ID mismatch between dataset and labels CSV
- **Solution**: Normalize IDs (lowercase, no spaces/special chars)

**Issue**: "Insufficient sequences for subject"
- **Cause**: Subject has too few data points
- **Solution**: Collect more data or remove subject from analysis

**Issue**: "Invalid action_label values"
- **Cause**: Action labels outside 0-21 range
- **Solution**: Remap or clip to valid range

**Issue**: "Hour values out of range"
- **Cause**: Hour > 24 or < 0
- **Solution**: Use `hour % 24` to wrap around

---

## Example Datasets

### Demo Dataset
A small demo dataset is provided in `demo_data/`:
- 10 subjects (5 CN, 5 CI)
- 3 days of data per subject
- ~200 sequences per subject

Load with:
```python
import pickle
with open('demo_data/processed_dataset.pkl', 'rb') as f:
    demo_data = pickle.load(f)
```

### Full Dataset (if available)
Contact the authors for access to the full research dataset:
- 68 subjects (28 CN, 40 CI)
- 7-30 days per subject
- Clinical scores included

---

## Data Privacy & Ethics

### Important Considerations

1. **De-identification**: All subject IDs must be anonymized codes
2. **Consent**: Ensure IRB approval and informed consent
3. **Security**: Store data securely with encryption
4. **Sharing**: Follow your institution's data sharing policies

### Recommended Practices

- Use coded IDs (e.g., "NX001" instead of names)
- Remove or hash any personally identifiable information
- Store data separately from code repository
- Include only aggregated results in publications

---

## References

For more information on activity recognition data formats:
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [CASAS Smart Home Dataset](https://casas.wsu.edu/datasets/)
- [ActiGraph Data Format](https://actigraphcorp.com/)

---

**Questions?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.
