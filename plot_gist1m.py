#!/usr/bin/env python3
"""Plot GIST 1M: Projection filter vs Random sampling (mc sweep)."""
import matplotlib.pyplot as plt
import numpy as np

# ── GIST 1M Data ──

# MC sweep (no filter, varying mc)
mc_labels = ['mc=25', 'mc=30', 'mc=35', 'mc=40']
mc_recall  = [0.5207, 0.5490, 0.5713, 0.5894]
mc_dist_M  = [1250.9, 1364.9, 1463.7, 1551.5]
mc_time    = [871.8,  942.0,  1003.6, 1056.7]

# Projection filter (mc=40, m=32, varying pτ)
pf_labels = ['pτ=0.99', 'pτ=0.95', 'pτ=0.90', 'pτ=0.85', 'pτ=0.80']
pf_recall  = [0.5879, 0.5820, 0.5720, 0.5603, 0.5470]
pf_dist_M  = [1425.5, 1265.7, 1143.0, 1048.4, 968.3]
pf_time    = [1058.1, 962.1,  885.8,  829.0,  778.5]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Plot 1: Recall vs Time ──
ax = axes[0]
ax.plot(mc_time, mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2, label='No filter (vary mc)', zorder=5)
ax.plot(pf_time, pf_recall, 's-', color='#F44336', markersize=9, linewidth=2, label='Proj filter (mc=40, vary pτ)', zorder=5)

for i, lbl in enumerate(mc_labels):
    ax.annotate(lbl, (mc_time[i], mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
for i, lbl in enumerate(pf_labels):
    ax.annotate(lbl, (pf_time[i], pf_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')

ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

# ── Plot 2: Recall vs Distance Computations ──
ax = axes[1]
ax.plot(mc_dist_M, mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2, label='No filter (vary mc)', zorder=5)
ax.plot(pf_dist_M, pf_recall, 's-', color='#F44336', markersize=9, linewidth=2, label='Proj filter (mc=40, vary pτ)', zorder=5)

for i, lbl in enumerate(mc_labels):
    ax.annotate(lbl, (mc_dist_M[i], mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
for i, lbl in enumerate(pf_labels):
    ax.annotate(lbl, (pf_dist_M[i], pf_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')

ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Dist Comps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/gist1m_filter_vs_sampling.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist1m_filter_vs_sampling.pdf', bbox_inches='tight')
print("Saved plots/gist1m_filter_vs_sampling.png and .pdf")


# ── Percentage change plots (relative to mc=40 no-filter baseline) ──
baseline_recall = 0.5894
baseline_dist = 1551.5
baseline_time = 1056.7

# MC sweep: % change from baseline
mc_recall_pct = [(r - baseline_recall) / baseline_recall * 100 for r in mc_recall]
mc_dist_pct   = [(d - baseline_dist) / baseline_dist * 100 for d in mc_dist_M]
mc_time_pct   = [(t - baseline_time) / baseline_time * 100 for t in mc_time]

# Filter: % change from baseline
pf_recall_pct = [(r - baseline_recall) / baseline_recall * 100 for r in pf_recall]
pf_dist_pct   = [(d - baseline_dist) / baseline_dist * 100 for d in pf_dist_M]
pf_time_pct   = [(t - baseline_time) / baseline_time * 100 for t in pf_time]

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5))

# ── Plot 1: % Recall Loss vs % Time Saved ──
ax = axes2[0]
ax.scatter([-t for t in mc_time_pct], [-r for r in mc_recall_pct],
           c='#2196F3', marker='o', s=100, zorder=5, label='No filter (vary mc)')
ax.scatter([-t for t in pf_time_pct], [-r for r in pf_recall_pct],
           c='#F44336', marker='s', s=100, zorder=5, label='Proj filter (vary pτ)')

for i, lbl in enumerate(mc_labels):
    ax.annotate(lbl, (-mc_time_pct[i], -mc_recall_pct[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color='#1565C0')
for i, lbl in enumerate(pf_labels):
    ax.annotate(lbl, (-pf_time_pct[i], -pf_recall_pct[i]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color='#C62828')

ax.set_xlabel('Time Saved (%)', fontsize=11)
ax.set_ylabel('Recall Loss (%)', fontsize=11)
ax.set_title('GIST 1M: Time Saved vs Recall Cost', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
# Add reference point for baseline
ax.scatter([0], [0], c='black', marker='*', s=200, zorder=6, label='Baseline (mc=40)')
ax.legend(fontsize=8)

# ── Plot 2: % Dist Comps Saved vs % Recall Loss ──
ax = axes2[1]
ax.scatter([-d for d in mc_dist_pct], [-r for r in mc_recall_pct],
           c='#2196F3', marker='o', s=100, zorder=5, label='No filter (vary mc)')
ax.scatter([-d for d in pf_dist_pct], [-r for r in pf_recall_pct],
           c='#F44336', marker='s', s=100, zorder=5, label='Proj filter (vary pτ)')

for i, lbl in enumerate(mc_labels):
    ax.annotate(lbl, (-mc_dist_pct[i], -mc_recall_pct[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color='#1565C0')
for i, lbl in enumerate(pf_labels):
    ax.annotate(lbl, (-pf_dist_pct[i], -pf_recall_pct[i]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color='#C62828')

ax.set_xlabel('Dist Comps Saved (%)', fontsize=11)
ax.set_ylabel('Recall Loss (%)', fontsize=11)
ax.set_title('GIST 1M: Dist Saved vs Recall Cost', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.scatter([0], [0], c='black', marker='*', s=200, zorder=6)
ax.legend(fontsize=8)

# ── Plot 3: Bar chart — efficiency ratio (% time saved per % recall lost) ──
ax = axes2[2]

# Skip baseline (0/0), compute ratio for each config
configs = []
ratios = []
colors = []

for i, lbl in enumerate(mc_labels):
    rl = -mc_recall_pct[i]
    ts = -mc_time_pct[i]
    if rl > 0.1:  # skip near-zero recall loss
        configs.append(lbl)
        ratios.append(ts / rl)
        colors.append('#2196F3')

for i, lbl in enumerate(pf_labels):
    rl = -pf_recall_pct[i]
    ts = -pf_time_pct[i]
    if rl > 0.1:
        configs.append(lbl)
        ratios.append(ts / rl)
        colors.append('#F44336')

x = np.arange(len(configs))
bars = ax.bar(x, ratios, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('% Time Saved / % Recall Lost', fontsize=11)
ax.set_title('GIST 1M: Efficiency Ratio\n(higher = better tradeoff)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f}x', ha='center', fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#2196F3', label='No filter (vary mc)'),
                   Patch(facecolor='#F44336', label='Proj filter (vary pτ)')],
          fontsize=8)

plt.tight_layout()
plt.savefig('plots/gist1m_percentage.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist1m_percentage.pdf', bbox_inches='tight')
print("Saved plots/gist1m_percentage.png and .pdf")