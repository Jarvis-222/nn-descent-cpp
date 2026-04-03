#!/usr/bin/env python3
"""Comprehensive plots for all datasets: GIST 100K, GIST 1M, SIFT 1M."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# DATA
# ============================================================

# --- GIST 100K (m=32, mc=40, averaged runs 2-5) ---
gist100k_labels   = ['No filter', 'pτ=0.99', 'pτ=0.95', 'pτ=0.90', 'pτ=0.80']
gist100k_recall   = [0.7832, 0.7831, 0.7755, 0.7626, 0.7330]
gist100k_dist_M   = [126.0, 116.0, 103.3, 93.4, 79.3]
gist100k_time     = [68.51, 64.90, 59.52, 54.73, 48.96]

# GIST 100K mc sweep (no filter)
gist100k_mc_labels  = ['mc=10', 'mc=15', 'mc=25', 'mc=30', 'mc=35', 'mc=40']
gist100k_mc_recall  = [0.5375, 0.6361, 0.7283, 0.7520, 0.7700, 0.7832]
gist100k_mc_dist_M  = [59.5, 77.4, 104.2, 112.7, 119.8, 126.0]
gist100k_mc_time    = [38.88, 58.22, 80.24, 64.80, 69.48, 68.51]

# --- GIST 1M (m=32, mc=40, single runs) ---
# Filter configs (mc=40)
gist1m_pf_labels  = ['pτ=0.99', 'pτ=0.95', 'pτ=0.90', 'pτ=0.85', 'pτ=0.80']
gist1m_pf_recall  = [0.5879, 0.5820, 0.5720, 0.5603, 0.5470]
gist1m_pf_dist_M  = [1425.5, 1265.7, 1143.0, 1048.4, 968.3]
gist1m_pf_time    = [1058.1, 962.1, 885.8, 829.0, 778.5]

# MC sweep (no filter) — mc=40 from user's first paste
gist1m_mc_labels  = ['mc=25', 'mc=30', 'mc=35', 'mc=40']
gist1m_mc_recall  = [0.5207, 0.5490, 0.5713, 0.5894]
gist1m_mc_dist_M  = [1250.9, 1364.9, 1463.7, 1551.5]
gist1m_mc_time    = [871.8, 942.0, 1003.6, 1056.7]

# --- SIFT 1M (m=32, mc=40, 5 runs averaged) ---
sift1m_labels   = ['No filter', 'pτ=0.99', 'pτ=0.95', 'pτ=0.90', 'pτ=0.80']
sift1m_recall   = [0.8641, 0.8633, 0.8589, 0.8513, 0.8315]
sift1m_dist_M   = [1536.2, 1233.5, 1043.4, 920.8, 764.5]
# Averaged times (5 runs each)
sift1m_time     = [
    np.mean([219.19, 219.30, 219.78, 219.55, 219.33]),  # no filter
    np.mean([246.03, 245.51, 245.63, 245.33, 245.08]),  # 0.99
    np.mean([230.49, 230.41, 230.56, 231.26, 231.58]),  # 0.95
    np.mean([219.45, 219.42, 219.52, 219.25, 219.55]),  # 0.90
    np.mean([206.31, 204.64, 204.76, 204.45, 204.74]),  # 0.80
]

# ============================================================
# FIGURE 1: GIST 1M — Filter vs Sampling (2 panels)
# ============================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes1[0]
ax.plot(gist1m_mc_time, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(gist1m_pf_time, gist1m_pf_recall, 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary pτ)', zorder=5)
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_time[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
for i, lbl in enumerate(gist1m_pf_labels):
    ax.annotate(lbl, (gist1m_pf_time[i], gist1m_pf_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

ax = axes1[1]
ax.plot(gist1m_mc_dist_M, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(gist1m_pf_dist_M, gist1m_pf_recall, 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary pτ)', zorder=5)
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_dist_M[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
for i, lbl in enumerate(gist1m_pf_labels):
    ax.annotate(lbl, (gist1m_pf_dist_M[i], gist1m_pf_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')
ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Dist Comps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/gist1m_filter_vs_sampling.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist1m_filter_vs_sampling.pdf', bbox_inches='tight')
print("Saved plots/gist1m_filter_vs_sampling")

# ============================================================
# FIGURE 2: SIFT 1M — Recall vs Time & Dist Comps
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes2[0]
ax.plot(sift1m_time, sift1m_recall, 's-', color='#F44336', markersize=9, linewidth=2, zorder=5)
for i, lbl in enumerate(sift1m_labels):
    ax.annotate(lbl, (sift1m_time[i], sift1m_recall[i]), textcoords="offset points",
                xytext=(8, -12 if i > 0 else 8), fontsize=9)
ax.set_xlabel('Wall-clock Time (s) — avg of 5 runs', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Time (m=32)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(sift1m_dist_M, sift1m_recall, 's-', color='#F44336', markersize=9, linewidth=2, zorder=5)
for i, lbl in enumerate(sift1m_labels):
    ax.annotate(lbl, (sift1m_dist_M[i], sift1m_recall[i]), textcoords="offset points",
                xytext=(8, -12 if i > 0 else 8), fontsize=9)
ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Dist Comps (m=32)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/sift1m_results.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/sift1m_results.pdf', bbox_inches='tight')
print("Saved plots/sift1m_results")

# ============================================================
# FIGURE 3: Bar chart — % Speedup & % Recall Loss across all datasets
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

configs = ['pτ=0.99', 'pτ=0.95', 'pτ=0.90', 'pτ=0.80']
x = np.arange(len(configs))
width = 0.25

# --- Speedup bars ---
ax = axes3[0]

# GIST 100K speedup (% faster than no-filter)
g100k_speedup = [(gist100k_time[0] - gist100k_time[i+1]) / gist100k_time[0] * 100
                 for i in range(4)]
# GIST 1M speedup
g1m_base_time = gist1m_mc_time[-1]  # mc=40 no filter
g1m_speedup = [(g1m_base_time - gist1m_pf_time[i]) / g1m_base_time * 100
               for i in range(4)]  # skip pτ=0.85
g1m_speedup_sel = [g1m_speedup[0], g1m_speedup[1], g1m_speedup[2], g1m_speedup[4] if len(g1m_speedup) > 4 else g1m_speedup[3]]
# Fix: only use matching configs
g1m_speedup_4 = []
gist1m_pf_configs = [0.99, 0.95, 0.90, 0.80]  # indices 0, 1, 2, 4
gist1m_pf_idx = [0, 1, 2, 4]
for idx in gist1m_pf_idx:
    g1m_speedup_4.append((g1m_base_time - gist1m_pf_time[idx]) / g1m_base_time * 100)

# SIFT 1M speedup
s1m_speedup = [(sift1m_time[0] - sift1m_time[i+1]) / sift1m_time[0] * 100
               for i in range(4)]

bars1 = ax.bar(x - width, g100k_speedup, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax.bar(x, g1m_speedup_4, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax.bar(x + width, s1m_speedup, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                    ha='center', fontsize=7, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h - 2.5, f'{h:.1f}%',
                    ha='center', fontsize=7, fontweight='bold', color='red')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('Time Saved (%)', fontsize=11)
ax.set_title('Wall-clock Speedup vs No Filter', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)

# --- Recall loss bars ---
ax = axes3[1]

g100k_rloss = [(gist100k_recall[0] - gist100k_recall[i+1]) / gist100k_recall[0] * 100
               for i in range(4)]
g1m_rloss_4 = []
g1m_base_recall = gist1m_mc_recall[-1]
for idx in gist1m_pf_idx:
    g1m_rloss_4.append((g1m_base_recall - gist1m_pf_recall[idx]) / g1m_base_recall * 100)

s1m_rloss = [(sift1m_recall[0] - sift1m_recall[i+1]) / sift1m_recall[0] * 100
             for i in range(4)]

bars1 = ax.bar(x - width, g100k_rloss, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax.bar(x, g1m_rloss_4, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax.bar(x + width, s1m_rloss, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f}%',
                ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('Recall Loss (%)', fontsize=11)
ax.set_title('Recall Loss vs No Filter', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/bar_speedup_recall_all.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/bar_speedup_recall_all.pdf', bbox_inches='tight')
print("Saved plots/bar_speedup_recall_all")

# ============================================================
# FIGURE 4: % Dist Comps Saved across all datasets
# ============================================================
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5.5))

g100k_dsaved = [(gist100k_dist_M[0] - gist100k_dist_M[i+1]) / gist100k_dist_M[0] * 100
                for i in range(4)]
g1m_dsaved_4 = []
g1m_base_dist = gist1m_mc_dist_M[-1]
for idx in gist1m_pf_idx:
    g1m_dsaved_4.append((g1m_base_dist - gist1m_pf_dist_M[idx]) / g1m_base_dist * 100)

s1m_dsaved = [(sift1m_dist_M[0] - sift1m_dist_M[i+1]) / sift1m_dist_M[0] * 100
              for i in range(4)]

bars1 = ax4.bar(x - width, g100k_dsaved, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax4.bar(x, g1m_dsaved_4, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax4.bar(x + width, s1m_dsaved, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                 ha='center', fontsize=8, fontweight='bold')

ax4.set_xticks(x)
ax4.set_xticklabels(configs, fontsize=10)
ax4.set_ylabel('Distance Computations Saved (%)', fontsize=11)
ax4.set_title('Dist Comps Reduction vs No Filter', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=10)

plt.tight_layout()
plt.savefig('plots/bar_dist_saved_all.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/bar_dist_saved_all.pdf', bbox_inches='tight')
print("Saved plots/bar_dist_saved_all")

# ============================================================
# FIGURE 5: GIST 100K vs GIST 1M side-by-side Recall-Time curves
# ============================================================
fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes5[0]
ax.plot(gist100k_time, gist100k_recall, 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter (vary pτ)', zorder=5)
ax.plot(gist100k_mc_time[2:], gist100k_mc_recall[2:], 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
for i, lbl in enumerate(gist100k_labels):
    ax.annotate(lbl, (gist100k_time[i], gist100k_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')
for i, lbl in enumerate(gist100k_mc_labels[2:]):
    ax.annotate(lbl, (gist100k_mc_time[i+2], gist100k_mc_recall[i+2]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 100K (960-dim)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

ax = axes5[1]
ax.plot(gist1m_pf_time, gist1m_pf_recall, 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter (vary pτ)', zorder=5)
ax.plot(gist1m_mc_time, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
for i, lbl in enumerate(gist1m_pf_labels):
    ax.annotate(lbl, (gist1m_pf_time[i], gist1m_pf_recall[i]), textcoords="offset points",
                xytext=(-10, -15), fontsize=8, color='#C62828')
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_time[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=8, color='#1565C0')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M (960-dim)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.suptitle('Projection Filter vs Random Sampling — GIST Dataset', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/gist_100k_vs_1m.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist_100k_vs_1m.pdf', bbox_inches='tight')
print("Saved plots/gist_100k_vs_1m")

# ============================================================
# FIGURE 6: All 3 datasets — Recall vs Time overlay
# ============================================================
fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))

# Normalize time to % of baseline for fair comparison
g100k_time_pct = [t / gist100k_time[0] * 100 for t in gist100k_time]
g1m_time_pct = [t / gist1m_pf_time[0] * 100 for t in gist1m_pf_time]  # pτ=0.99 is closest to baseline
# Use mc=40 baseline for gist1m
g1m_time_pct = [t / gist1m_mc_time[-1] * 100 for t in gist1m_pf_time]
s1m_time_pct = [t / sift1m_time[0] * 100 for t in sift1m_time]

# Recall as % of baseline
g100k_recall_pct = [r / gist100k_recall[0] * 100 for r in gist100k_recall]
g1m_recall_pct = [r / gist1m_mc_recall[-1] * 100 for r in gist1m_pf_recall]
s1m_recall_pct = [r / sift1m_recall[0] * 100 for r in sift1m_recall]

ax6.plot(g100k_time_pct, g100k_recall_pct, 'o-', color='#4CAF50', markersize=9, linewidth=2,
         label='GIST 100K (960-dim)', zorder=5)
ax6.plot(g1m_time_pct, g1m_recall_pct, 's-', color='#2196F3', markersize=9, linewidth=2,
         label='GIST 1M (960-dim)', zorder=5)
ax6.plot(s1m_time_pct, s1m_recall_pct, 'D-', color='#FF9800', markersize=9, linewidth=2,
         label='SIFT 1M (128-dim)', zorder=5)

for i, lbl in enumerate(gist100k_labels):
    ax6.annotate(lbl, (g100k_time_pct[i], g100k_recall_pct[i]), textcoords="offset points",
                 xytext=(8, 6), fontsize=7, color='#2E7D32')
for i, lbl in enumerate(gist1m_pf_labels):
    ax6.annotate(lbl, (g1m_time_pct[i], g1m_recall_pct[i]), textcoords="offset points",
                 xytext=(8, -12), fontsize=7, color='#1565C0')
for i, lbl in enumerate(sift1m_labels):
    ax6.annotate(lbl, (s1m_time_pct[i], s1m_recall_pct[i]), textcoords="offset points",
                 xytext=(8, 6), fontsize=7, color='#E65100')

ax6.set_xlabel('Time (% of no-filter baseline)', fontsize=11)
ax6.set_ylabel('Recall (% of no-filter baseline)', fontsize=11)
ax6.set_title('Projection Filter: Normalized Recall-Time Tradeoff', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)
ax6.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/all_datasets_normalized.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/all_datasets_normalized.pdf', bbox_inches='tight')
print("Saved plots/all_datasets_normalized")

# ============================================================
# Print summary table
# ============================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print("\nSIFT 1M (128-dim, K=10, mc=40, m=32) — avg of 5 runs:")
print(f"{'Config':<15} {'Recall':>8} {'Dist (M)':>10} {'Time (s)':>10} {'Speedup':>10} {'Recall Loss':>12}")
for i, lbl in enumerate(sift1m_labels):
    sp = (sift1m_time[0] - sift1m_time[i]) / sift1m_time[0] * 100
    rl = (sift1m_recall[0] - sift1m_recall[i]) / sift1m_recall[0] * 100
    print(f"{lbl:<15} {sift1m_recall[i]:>8.4f} {sift1m_dist_M[i]:>10.1f} {sift1m_time[i]:>10.1f} {sp:>9.1f}% {rl:>11.2f}%")

print(f"\nGIST 1M (960-dim, K=10, mc=40, m=32) — single runs:")
print(f"{'Config':<15} {'Recall':>8} {'Dist (M)':>10} {'Time (s)':>10} {'Speedup':>10} {'Recall Loss':>12}")
all_g1m = [('No filter', gist1m_mc_recall[-1], gist1m_mc_dist_M[-1], gist1m_mc_time[-1])]
for i, lbl in enumerate(gist1m_pf_labels):
    all_g1m.append((lbl, gist1m_pf_recall[i], gist1m_pf_dist_M[i], gist1m_pf_time[i]))
for lbl, r, d, t in all_g1m:
    sp = (gist1m_mc_time[-1] - t) / gist1m_mc_time[-1] * 100
    rl = (gist1m_mc_recall[-1] - r) / gist1m_mc_recall[-1] * 100
    print(f"{lbl:<15} {r:>8.4f} {d:>10.1f} {t:>10.1f} {sp:>9.1f}% {rl:>11.2f}%")

print(f"\nGIST 100K (960-dim, K=10, mc=40, m=32) — avg of runs 2-5:")
print(f"{'Config':<15} {'Recall':>8} {'Dist (M)':>10} {'Time (s)':>10} {'Speedup':>10} {'Recall Loss':>12}")
for i, lbl in enumerate(gist100k_labels):
    sp = (gist100k_time[0] - gist100k_time[i]) / gist100k_time[0] * 100
    rl = (gist100k_recall[0] - gist100k_recall[i]) / gist100k_recall[0] * 100
    print(f"{lbl:<15} {gist100k_recall[i]:>8.4f} {gist100k_dist_M[i]:>10.1f} {gist100k_time[i]:>10.1f} {sp:>9.1f}% {rl:>11.2f}%")