# CTMS Experiment Results Comparison

## Exp1: Embedding

| Metric               |   With Personalization |   Without Personalization |
|:---------------------|-----------------------:|--------------------------:|
| Davies-Bouldin Score |                  1.542 |                     1.685 |
| Silhouette Score     |                  0.365 |                     0.312 |

## Exp2: Daily Patterns

| Metric        | With Personalization   | Without Personalization   |
|:--------------|:-----------------------|:--------------------------|
| CI Mean Score | 0.612                  | 0.525                     |
| CI/CN Ratio   | 1.537×                 | 1.409×                    |
| CN Mean Score | 0.398                  | 0.372                     |
| P-value       | <0.001***              | <0.001***                 |

## Exp3: Classification

| Metric      | With Personalization   | Without Personalization   |
|:------------|:-----------------------|:--------------------------|
| CI/CN Ratio | 1.149×                 | 0.936×                    |
| F1 Score    | 0.824                  | 0.800                     |
| Precision   | 77.8%                  | 72.7%                     |
| Sensitivity | 87.5%                  | 88.9%                     |
| Specificity | 33.3%                  | 25.0%                     |

## Exp4: Medical Corr

| Metric            | With Personalization   | Without Personalization   |
|:------------------|:-----------------------|:--------------------------|
| Circadian vs MoCA | r=0.420, 0.028*        | r=0.381, 0.035*           |
| Movement vs FAS   | r=0.440, 0.016*        | nan                       |
| Social vs NPI     | r=-0.390, 0.035*       | nan                       |
| Task vs ZBI       | r=0.380, 0.042*        | nan                       |

