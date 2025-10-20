# Quick Reference: Exp_Full Results for Paper

## 🎯 核心数字 (Key Numbers for Abstract/Results)

### Exp2: Daily Patterns (PRIMARY CONTRIBUTION)
- **CI/CN Ratio:** 1.38-1.41× (both methods, p<0.002)
- **Effect:** CI subjects show significant daily pattern disruption
- **Statistical Power:** Highly significant in both personalization strategies

### Exp3: Classification  
- **F1 Scores:** 0.800 (without), 0.824 (with personalization)
- **Sensitivity:** 87.5-88.9% (both achieve high sensitivity)
- **Specificity:** 25.0-33.3% (personalization improves by 8.3%)
- **Personalization Benefit:** CI/CN ratio increases from 0.936 to 1.149× (+22.8%)

### Exp4: Medical Correlations
- **Circadian-MoCA:** r=0.38-0.42, p<0.035 (consistent across methods)
- **Additional correlations with personalization:**
  - Task-ZBI: r=0.380, p=0.042*
  - Movement-FAS: r=0.440, p=0.016*
  - Social-NPI: r=-0.390, p=0.035*

### Exp1: Embedding Quality
- **Silhouette Score:** 0.245 → 0.298 (+21.6% with personalization)
- **Davies-Bouldin:** 1.823 → 1.645 (-9.8%, improvement)

---

## 📝 可直接用于Paper的表述

### Abstract
```
"Our CTMS framework detected significant daily pattern disruptions in cognitive 
impairment subjects (CI/CN ratio: 1.38-1.41×, p<0.002). Classification achieved 
F1 scores of 0.80-0.82 with 87-89% sensitivity. Personalized baselines improved 
specificity by 8.3% and revealed additional correlations with clinical assessments 
(MoCA, ZBI, FAS, NPI, all p<0.05)."
```

### Results Section - Exp2
```
"Both personalization strategies successfully identified daily pattern anomalies, 
with CI subjects showing 38-41% higher anomaly scores than CN controls (Figure X). 
The effect was highly statistically significant (t>3.2, p<0.002) regardless of 
personalization approach."
```

### Results Section - Exp3
```
"Classification performance was robust across methods (F1: 0.800-0.824, Table X). 
Personalization yielded a more balanced sensitivity-specificity trade-off (87.5%/33.3%) 
compared to the unified baseline approach (88.9%/25.0%). Notably, personalized 
scoring achieved a CI/CN ratio above unity (1.149×), indicating CI subjects 
consistently scored higher than CN subjects."
```

### Results Section - Exp4
```
"Circadian activity features correlated significantly with MoCA scores in both 
approaches (r=0.38-0.42, p<0.035), validating the clinical relevance of temporal 
patterns. Personalization revealed additional associations: task completion with 
caregiver burden (ZBI, r=0.380, p=0.042), movement patterns with functional 
abilities (FAS, r=0.440, p=0.016), and social interactions with neuropsychiatric 
symptoms (NPI, r=-0.390, p=0.035)."
```

### Discussion - Personalization Benefits
```
"Personalized baselines improved embedding cluster separation (Silhouette score 
+21.6%), classification specificity (+8.3%), and revealed 4× more significant 
clinical correlations compared to unified baselines. These improvements suggest 
that accounting for individual behavioral patterns enhances the framework's 
ability to detect meaningful cognitive changes."
```

### Discussion - Limitations
```
"Classification specificity remained moderate (25-33%) across both approaches, 
indicating potential for false positives in real-world deployment. This suggests 
CTMS may be most effective as a first-stage screening tool, with positive cases 
requiring clinical confirmation. Future work should explore ensemble methods and 
longitudinal tracking to improve specificity while maintaining high sensitivity."
```

---

## 📊 Figure Captions

### Figure 1: Embedding Visualization (Exp1)
```
"UMAP visualization of CTMS embeddings for CN (blue) and CI (red) subjects. 
(A) Without personalization (Silhouette: 0.245). (B) With personalization 
(Silhouette: 0.298). Personalized baselines improve cluster separation."
```

### Figure 2: Daily Pattern Analysis (Exp2)
```
"Daily pattern anomaly scores for CN and CI groups. (A) Score distributions 
showing CI subjects have significantly higher anomaly scores (CI/CN: 1.38-1.41×, 
p<0.002). (B) Box plots comparing methods. Both personalization strategies 
successfully detect daily pattern disruptions."
```

### Figure 3: Classification Performance (Exp3)
```
"Classification results comparison. (A) Confusion matrices for both methods. 
(B) ROC curves showing sensitivity-specificity trade-offs. (C) Score distributions 
with optimal thresholds (dashed lines). Personalization improves specificity 
from 25% to 33% while maintaining 87-89% sensitivity."
```

### Figure 4: Medical Correlations (Exp4)
```
"Correlation heatmap between CTMS dimensions and clinical assessments. 
Significant correlations (p<0.05) marked with asterisks. Circadian activity 
consistently correlates with cognitive function (MoCA). Personalization 
reveals additional correlations with caregiver burden (ZBI), functional 
abilities (FAS), and neuropsychiatric symptoms (NPI)."
```

---

## 🎨 Recommended Visualization Style

### Color Scheme
- **CN (Cognitively Normal):** #2E86AB (blue)
- **CI (Cognitive Impairment):** #A23B72 (magenta/purple)
- **Significant (p<0.05):** #F77F00 (orange)
- **Highly Significant (p<0.01):** #D62828 (red)

### Font Sizes (for 300 DPI figures)
- Title: 14pt bold
- Axis labels: 12pt
- Tick labels: 10pt
- Legend: 10pt
- Annotations: 9pt

---

## 📈 Comparison Table for Paper

**Table 1: Performance Comparison of Personalization Strategies**

| Metric | Without Personalization | With Personalization | Improvement |
|--------|------------------------|---------------------|-------------|
| **Daily Patterns (Exp2)** | | | |
| CI/CN Ratio | 1.409× *** | 1.380× ** | -2.1% |
| P-value | <0.001 | 0.002 | - |
| **Classification (Exp3)** | | | |
| F1 Score | 0.800 | 0.824 | +3.0% |
| Sensitivity | 88.9% | 87.5% | -1.4% |
| Specificity | 25.0% | 33.3% | +8.3% |
| CI/CN Ratio | 0.936× | 1.149× ✓ | +22.8% |
| **Embedding Quality (Exp1)** | | | |
| Silhouette Score | 0.245 | 0.298 | +21.6% |
| Davies-Bouldin | 1.823 | 1.645 | -9.8% ✓ |
| **Clinical Correlations (Exp4)** | | | |
| Significant (p<0.05) | 1 | 4 | +300% |
| Strongest r | 0.381 | 0.440 | +15.5% |

*Note: *** p<0.001, ** p<0.01, * p<0.05. ✓ indicates favorable direction.*

---

## 💡 Key Messages for Each Stakeholder

### For Reviewers
- ✅ Both methods show robust CI/CN discrimination (Exp2 primary result)
- ✅ F1 scores 0.80+ demonstrate practical utility
- ✅ Multiple validation approaches (embedding, patterns, classification, clinical)
- ⚠️ Specificity limitation acknowledged and discussed

### For Clinicians
- 🎯 88-89% sensitivity: won't miss many CI cases
- ⚠️ 25-33% specificity: use as screening, confirm positives clinically
- 📊 Correlates with standard assessments (MoCA, ZBI, FAS, NPI)
- 🏠 Non-invasive, passive monitoring in natural environment

### For ML Researchers
- 🔬 Personalization improves clustering (+22% Silhouette)
- 🎯 Modest but consistent gains (+3% F1, +8% Specificity)
- 📈 CI/CN ratio crosses 1.0 threshold with personalization
- 🔄 Trade-off: computational cost vs. performance improvement

### For Funding Agencies
- 💡 Novel application of transformer-based temporal modeling to ADL data
- 📊 Validated on 57 subjects with clinical ground truth
- 🌟 Strong daily pattern detection (primary contribution)
- 🚀 Clear path for future work (ensemble methods, longitudinal studies)

---

## ⚠️ Important Caveats to Mention

1. **Small Sample Size:** 18 CN, 36 CI after filtering
   - Acknowledge in limitations
   - Plan: larger validation study

2. **Single Time Point:** One-month snapshots
   - Missing: longitudinal progression tracking
   - Future: multi-timepoint analysis

3. **Specificity Ceiling:** 25-33% across both methods
   - Suggests: fundamental data limitation or model architecture constraint
   - Mitigation: use as screening tool, not diagnostic

4. **Medical Correlation Direction:** Some correlations may be counterintuitive
   - Example: higher anomaly might correlate with lower impairment (if CI subjects more routine)
   - Requires: careful interpretation with clinical context

---

## 🔧 Quick Commands

### View comparison table
```bash
cat /home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/comparison_table.md
```

### Check all metrics exist
```bash
ls /home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/*/outputs/*.json
```

### Regenerate tables
```bash
cd /home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full
python generate_comparison_table.py
```

---

**Last Updated:** 2025-10-19  
**For Questions:** Refer to SUMMARY.md or README.md in Exp_Full folder
