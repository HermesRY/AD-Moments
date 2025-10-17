"""
Re-plot the weekly anomaly pattern from saved grid data
Use this to quickly adjust visualization style without re-running the model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os

print("📊 Loading saved grid data...")

# Load the saved data
data = np.load('outputs/weekly_grid_data.npy', allow_pickle=True).item()

cn_grid = data['cn_grid']
ci_grid = data['ci_grid']
cn_count = data['cn_count']
ci_count = data['ci_count']
cn_moments_count = data['cn_moments_count']
ci_moments_count = data['ci_moments_count']
hour_bins = data['hour_bins']
hour_labels = data['hour_labels']

print(f"✓ Loaded data:")
print(f"  CN: {cn_count} subjects, {cn_moments_count} anomalous moments")
print(f"  CI: {ci_count} subjects, {ci_moments_count} anomalous moments")

# Create figure
print("\n🎨 Creating visualization...")

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.35)

# Define teal colormap (you can easily change this!)
colors = ['#ffffff', '#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', 
          '#26a69a', '#009688', '#00796b', '#004d40', '#00251a']
cmap = LinearSegmentedColormap.from_list('teal', colors)

# Set consistent vmax for comparison
vmax = max(np.max(cn_grid), np.max(ci_grid))

# CN heatmap
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(cn_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
ax1.set_title('CN - Anomalous Moments per Subject (Daytime 6:30-19:30)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Day of Week', fontsize=12)
ax1.set_ylabel('Time', fontsize=12)
ax1.set_xticks(range(7))
ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
ax1.set_yticks(range(0, len(hour_bins), 4))
ax1.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
plt.colorbar(im1, ax=ax1, label='Moments/Subject', pad=0.02)

# CI heatmap
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(ci_grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
ax2.set_title('CI - Anomalous Moments per Subject (Daytime 6:30-19:30)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Day of Week', fontsize=12)
ax2.set_ylabel('Time', fontsize=12)
ax2.set_xticks(range(7))
ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
ax2.set_yticks(range(0, len(hour_bins), 4))
ax2.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
plt.colorbar(im2, ax=ax2, label='Moments/Subject', pad=0.02)

# Difference plot (hourly average across week)
diff = ci_grid - cn_grid
avg_diff_per_hour = np.mean(diff, axis=1)

ax3 = fig.add_subplot(gs[2])
bars = ax3.barh(range(len(avg_diff_per_hour)), avg_diff_per_hour, 
                color=['#e74c3c' if x > 0 else '#2ecc71' for x in avg_diff_per_hour],
                edgecolor='#34495e', linewidth=0.5)
ax3.set_xlabel('Difference (CI - CN) in Anomalous Moments/Hour', fontsize=11, fontweight='bold')
ax3.set_ylabel('Time', fontsize=11)
ax3.set_title('CI vs CN Difference', fontsize=12, fontweight='bold', pad=10)
ax3.set_yticks(range(0, len(hour_bins), 4))
ax3.set_yticklabels([hour_labels[i] for i in range(0, len(hour_bins), 4)], fontsize=10)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
ax3.grid(True, alpha=0.3, axis='x')

# Add time period shading
# Morning: 6:30-12:00 (bins 0-11)
# Afternoon: 12:00-18:00 (bins 11-23)
# Evening: 18:00-19:30 (bins 23-27)
ax3.axhspan(-0.5, 10.5, color='#ff6f00', alpha=0.08, zorder=0, label='Morning')
ax3.axhspan(10.5, 22.5, color='#2e7d32', alpha=0.08, zorder=0, label='Afternoon')
ax3.axhspan(22.5, len(hour_bins)-0.5, color='#5e35b1', alpha=0.08, zorder=0, label='Evening')

plt.suptitle('Weekly Temporal Pattern of Anomalous Behavioral Moments', 
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_path = 'outputs/weekly_anomaly_patterns.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

output_path_pdf = 'outputs/weekly_anomaly_patterns.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"✅ Saved: {output_path_pdf}")

plt.close()

# Print statistics
print("\n📊 Statistics:")
print(f"  CN avg moments/hour: {np.mean(cn_grid):.3f}")
print(f"  CI avg moments/hour: {np.mean(ci_grid):.3f}")
if np.mean(cn_grid) > 0:
    print(f"  CI/CN ratio: {np.mean(ci_grid)/np.mean(cn_grid):.2f}×")

print("\n✨ Done! You can edit this script to change colors, fonts, layout, etc.")
print("   Just re-run: python replot_weekly.py")
