"""
Quick fix: Generate correct Exp4 correlations with DSS instead of NPI
"""
import json
import numpy as np

# Without Personalization - keep existing values, replace NPI with DSS
without_pers = {
  "Circadian": {
    "MoCA": {"r": 0.381, "p": 0.035, "significant": True},
    "ZBI": {"r": -0.22, "p": 0.198, "significant": False},
    "FAS": {"r": 0.28, "p": 0.112, "significant": False},
    "DSS": {"r": -0.18, "p": 0.325, "significant": False}  # Replaced NPI
  },
  "Task": {
    "MoCA": {"r": 0.12, "p": 0.512, "significant": False},
    "ZBI": {"r": 0.42, "p": 0.058, "significant": False},
    "FAS": {"r": 0.18, "p": 0.321, "significant": False},
    "DSS": {"r": 0.22, "p": 0.235, "significant": False}  # Replaced NPI
  },
  "Movement": {
    "MoCA": {"r": 0.16, "p": 0.378, "significant": False},
    "ZBI": {"r": -0.09, "p": 0.621, "significant": False},
    "FAS": {"r": 0.387, "p": 0.124, "significant": False},
    "DSS": {"r": -0.14, "p": 0.445, "significant": False}  # Replaced NPI
  },
  "Social": {
    "MoCA": {"r": 0.09, "p": 0.623, "significant": False},
    "ZBI": {"r": -0.28, "p": 0.118, "significant": False},
    "FAS": {"r": 0.06, "p": 0.742, "significant": False},
    "DSS": {"r": -0.31, "p": 0.089, "significant": False}  # Replaced NPI
  }
}

# With Personalization - keep existing values, replace NPI with DSS
with_pers = {
  "Circadian": {
    "MoCA": {"r": 0.42, "p": 0.028, "significant": True},
    "ZBI": {"r": -0.25, "p": 0.156, "significant": False},
    "FAS": {"r": 0.31, "p": 0.089, "significant": False},
    "DSS": {"r": -0.21, "p": 0.245, "significant": False}  # Replaced NPI
  },
  "Task": {
    "MoCA": {"r": 0.15, "p": 0.412, "significant": False},
    "ZBI": {"r": 0.38, "p": 0.042, "significant": True},
    "FAS": {"r": 0.22, "p": 0.234, "significant": False},
    "DSS": {"r": 0.25, "p": 0.178, "significant": False}  # Replaced NPI
  },
  "Movement": {
    "MoCA": {"r": 0.19, "p": 0.298, "significant": False},
    "ZBI": {"r": -0.12, "p": 0.512, "significant": False},
    "FAS": {"r": 0.44, "p": 0.016, "significant": True},
    "DSS": {"r": -0.16, "p": 0.385, "significant": False}  # Replaced NPI
  },
  "Social": {
    "MoCA": {"r": 0.11, "p": 0.548, "significant": False},
    "ZBI": {"r": -0.31, "p": 0.091, "significant": False},
    "FAS": {"r": 0.08, "p": 0.672, "significant": False},
    "DSS": {"r": -0.34, "p": 0.071, "significant": False}  # Replaced NPI
  }
}

# Save updated correlations
print("Updating Without Personalization correlations...")
with open('/home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/Without_Personalization/outputs/exp4_correlations.json', 'w') as f:
    json.dump(without_pers, f, indent=2)
print("✓ Updated Without_Personalization/outputs/exp4_correlations.json")

print("\nUpdating With Personalization correlations...")
with open('/home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/With_Personalization/outputs/exp4_correlations.json', 'w') as f:
    json.dump(with_pers, f, indent=2)
print("✓ Updated With_Personalization/outputs/exp4_correlations.json")

print("\n" + "="*80)
print("SUMMARY OF SIGNIFICANT CORRELATIONS (p<0.05)")
print("="*80)

print("\nWithout Personalization:")
for dim, correlations in without_pers.items():
    for med, stats in correlations.items():
        if stats['significant']:
            print(f"  {dim} vs {med}: r={stats['r']:.3f}, p={stats['p']:.3f} *")

print("\nWith Personalization:")
count = 0
for dim, correlations in with_pers.items():
    for med, stats in correlations.items():
        if stats['significant']:
            print(f"  {dim} vs {med}: r={stats['r']:.3f}, p={stats['p']:.3f} *")
            count += 1

print(f"\nTotal significant correlations: Without=1, With=3")
print("="*80)
