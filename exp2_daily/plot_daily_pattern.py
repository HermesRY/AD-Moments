"""
Plot daily temporal pattern (averaged across all days)
Horizontal: Time 6:30-19:30
Vertical: Average anomalous moments per subject per hour
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json

print("📊 Loading saved grid data...")

# Load the saved data
data = np.load('outputs/weekly_grid_data.npy', allow_pickle=True).item()

cn_grid = data['cn_grid']  # Shape: (27 time bins, 7 days)
ci_grid = data['ci_grid']
cn_count = data['cn_count']
ci_count = data['ci_count']
hour_bins = data['hour_bins']
hour_labels = data['hour_labels']

print(f"✓ Loaded data:")
print(f"  CN: {cn_count} subjects")
print(f"  CI: {ci_count} subjects")
print(f"  Time bins: {len(hour_bins)} (6:30-19:30)")

# Average across all days (collapse the weekly dimension)
cn_daily = np.mean(cn_grid, axis=1)  # Shape: (27,) - average over 7 days
ci_daily = np.mean(ci_grid, axis=1)

# Create time axis (in hours)
time_hours = hour_bins

print("\n🎨 Creating daily pattern visualization...")

# Create figure with clean style
fig, ax = plt.subplots(figsize=(14, 6))

# Define color scheme: CN green, CI light red
cn_color = '#2ecc71'  # Green for CN
ci_color = '#e74c3c'  # Light red for CI

# Plot CN and CI curves
ax.plot(time_hours, cn_daily, color=cn_color, linewidth=3, 
        label='CN', marker='o', markersize=8, alpha=0.8)
ax.plot(time_hours, ci_daily, color=ci_color, linewidth=3, 
        label='CI', marker='s', markersize=8, alpha=0.8)

# Fill area between curves to show difference
ax.fill_between(time_hours, cn_daily, ci_daily, 
                where=(ci_daily >= cn_daily), 
                color=ci_color, alpha=0.15, 
                interpolate=True, label='CI > CN')

# Styling
ax.set_xlabel('Time of Day', fontsize=24, fontweight='bold')
ax.set_ylabel('Avg Anomalous Moments\nper Subject per Hour', fontsize=24, fontweight='bold')
ax.set_title('Daily Temporal Activity Anomaly Patterns', 
             fontsize=24, fontweight='bold', pad=20)

# Set x-axis limits to exactly 6:30-19:30
ax.set_xlim(6.5, 19.5)

# Set x-axis ticks (every 2 hours)
tick_indices = range(0, len(hour_labels), 4)  # Every 2 hours
ax.set_xticks([time_hours[i] for i in tick_indices])
ax.set_xticklabels([hour_labels[i] for i in tick_indices], fontsize=24)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Remove y-axis tick labels (keep ticks but hide numbers)
ax.set_yticklabels([])

# Legend - move to lower left
ax.legend(loc='lower left', fontsize=24, framealpha=0.95, 
          edgecolor='gray', fancybox=True)

# Add time period shading (subtle)
# Morning: 6:30-12:00
ax.axvspan(6.5, 12.0, color='#ff6f00', alpha=0.05, zorder=0)
ax.text(9.25, ax.get_ylim()[1]*0.95, 'Morning', 
        ha='center', va='top', fontsize=24, color='#ff6f00', fontweight='bold', alpha=0.6)

# Afternoon: 12:00-18:00
ax.axvspan(12.0, 18.0, color='#2e7d32', alpha=0.05, zorder=0)
ax.text(15.0, ax.get_ylim()[1]*0.95, 'Afternoon', 
        ha='center', va='top', fontsize=24, color='#2e7d32', fontweight='bold', alpha=0.6)

# Evening: 18:00-19:30
ax.axvspan(18.0, 19.5, color='#5e35b1', alpha=0.05, zorder=0)
ax.text(18.75, ax.get_ylim()[1]*0.95, 'Eve', 
        ha='center', va='top', fontsize=24, color='#5e35b1', fontweight='bold', alpha=0.6)

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

# Tight layout
plt.tight_layout()

# Save
output_path = 'outputs/daily_pattern.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

output_path_pdf = 'outputs/daily_pattern.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"✅ Saved: {output_path_pdf}")

plt.close()

# Print statistics
print("\n📊 Daily Pattern Statistics:")
print(f"  CN avg moments/hour: {np.mean(cn_daily):.4f}")
print(f"  CI avg moments/hour: {np.mean(ci_daily):.4f}")
if np.mean(cn_daily) > 0:
    print(f"  CI/CN ratio: {np.mean(ci_daily)/np.mean(cn_daily):.2f}×")

cn_peak_idx = np.argmax(cn_daily)
ci_peak_idx = np.argmax(ci_daily)
print(f"\n  CN peak time: {hour_labels[cn_peak_idx]} ({cn_daily[cn_peak_idx]:.4f})")
print(f"  CI peak time: {hour_labels[ci_peak_idx]} ({ci_daily[ci_peak_idx]:.4f})")

print("\n✨ Done!")
