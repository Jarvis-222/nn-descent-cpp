#!/usr/bin/env python3
"""Comprehensive plots for all datasets: GIST 100K, GIST 1M, SIFT 1M."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# DATA
# ============================================================

# --- GIST 100K (m=32, mc=40, averaged runs 2-5) ---
gist100k_labels   = ['No filter', 'p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.80']
gist100k_recall   = [0.7832, 0.7831, 0.7755, 0.7626, 0.7330]
gist100k_dist_M   = [126.0, 116.0, 103.3, 93.4, 79.3]
gist100k_time     = [68.51, 64.90, 59.52, 54.73, 48.96]

# GIST 100K mc sweep (no filter)
gist100k_mc_labels  = ['mc=10', 'mc=15', 'mc=25', 'mc=30', 'mc=35', 'mc=40']
gist100k_mc_recall  = [0.5375, 0.6361, 0.7283, 0.7520, 0.7700, 0.7832]
gist100k_mc_dist_M  = [59.5, 77.4, 104.2, 112.7, 119.8, 126.0]
gist100k_mc_time    = [38.88, 58.22, 80.24, 64.80, 69.48, 68.51]

# --- SIFT 1M Random Init (m=32, mc=40) ---
sift1m_rand_labels  = ['No filter', 'p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.85',
                        'p\u03c4=0.80', 'p\u03c4=0.75', 'p\u03c4=0.70', 'p\u03c4=0.65', 'p\u03c4=0.60']
sift1m_rand_recall  = [0.8641, 0.8633, 0.8589, 0.8513, 0.8423, 0.8315, 0.8192, 0.8067, 0.7922, 0.7762]
sift1m_rand_dist_M  = [1536.2, 1233.5, 1043.4, 920.8, 834.1, 764.5, 705.8, 654.5, 608.3, 566.0]
sift1m_rand_time    = [210.4, 230.9, 218.4, 208.9, 201.0, 194.9, 189.8, 186.5, 182.2, 177.9]

# --- SIFT 1M RP-Tree Init (m=32, mc=40) ---
sift1m_rpt_labels   = ['No filter', 'p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.85',
                        'p\u03c4=0.80', 'p\u03c4=0.75', 'p\u03c4=0.70', 'p\u03c4=0.65', 'p\u03c4=0.60']
sift1m_rpt_recall   = [0.8653, 0.8647, 0.8609, 0.8547, 0.8472, 0.8387, 0.8291, 0.8188, 0.8074, 0.7952]
sift1m_rpt_dist_M   = [653.7, 591.7, 534.6, 495.4, 466.6, 443.0, 422.5, 404.1, 387.4, 371.9]
sift1m_rpt_time     = [121.9, 132.0, 128.1, 124.8, 122.2, 120.3, 118.0, 115.9, 114.0, 112.2]

# --- SIFT 1M MC Sweep (Random init, no filter) ---
sift1m_mc_labels  = ['mc=10', 'mc=15', 'mc=20', 'mc=25', 'mc=30', 'mc=35']
sift1m_mc_recall  = [0.7291, 0.8070, 0.8367, 0.8499, 0.8573, 0.8613]
sift1m_mc_dist_M  = [893.1, 1125.6, 1293.6, 1398.8, 1462.9, 1506.2]
sift1m_mc_time    = [166.4, 177.9, 189.8, 199.4, 204.4, 208.6]

# --- GIST 1M Random Init (m=32, mc=40) ---
gist1m_rand_labels  = ['No filter', 'p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.85',
                        'p\u03c4=0.80', 'p\u03c4=0.75', 'p\u03c4=0.70', 'p\u03c4=0.65', 'p\u03c4=0.60']
gist1m_rand_recall  = [0.5894, 0.5879, 0.5820, 0.5720, 0.5603, 0.5470, 0.5331, 0.5171, 0.5010, 0.4827]
gist1m_rand_dist_M  = [1551.5, 1425.5, 1265.7, 1143.0, 1048.4, 968.3, 898.0, 834.3, 775.7, 720.9]
gist1m_rand_time    = [1045.8, 1033.1, 942.0, 865.0, 808.6, 762.1, 718.9, 680.7, 645.0, 611.5]

# --- GIST 1M RP-Tree Init (m=32, mc=40) ---
gist1m_rpt_labels   = ['No filter', 'p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.85',
                        'p\u03c4=0.80', 'p\u03c4=0.75', 'p\u03c4=0.70', 'p\u03c4=0.65', 'p\u03c4=0.60']
gist1m_rpt_recall   = [0.6466, 0.6457, 0.6399, 0.6307, 0.6200, 0.6082, 0.5949, 0.5806, 0.5652, 0.5486]
gist1m_rpt_dist_M   = [909.6, 859.6, 785.2, 725.3, 677.8, 637.0, 600.3, 566.8, 535.6, 506.3]
gist1m_rpt_time     = [806.8, 816.1, 772.6, 736.7, 708.6, 684.3, 662.2, 641.0, 622.2, 603.0]

# --- GIST 1M MC Sweep (Random init, no filter) ---
gist1m_mc_labels  = ['mc=10', 'mc=15', 'mc=20', 'mc=25', 'mc=35']
gist1m_mc_recall  = [0.3300, 0.4243, 0.4820, 0.5207, 0.5713]
gist1m_mc_dist_M  = [688.2, 914.5, 1106.0, 1250.9, 1463.7]
gist1m_mc_time    = [780.4, 1005.5, 1217.4, 871.8, 1003.6]

# ============================================================
# Helpers
# ============================================================
# Common pτ subset for bar charts
bar_configs = ['p\u03c4=0.99', 'p\u03c4=0.95', 'p\u03c4=0.90', 'p\u03c4=0.80']
bar_idx = [1, 2, 3, 5]  # indices into the 10-element label arrays

# ============================================================
# FIGURE 1: SIFT 1M — Random vs RP-Tree Init (2 panels)
# ============================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes1[0]
ax.plot(sift1m_rand_time, sift1m_rand_recall, 'o-', color='#F44336', markersize=7, linewidth=2,
        label='Random init', zorder=5)
ax.plot(sift1m_rpt_time, sift1m_rpt_recall, 's-', color='#2196F3', markersize=7, linewidth=2,
        label='RP-Tree init', zorder=5)
for i in [0, 2, 5, 9]:
    ax.annotate(sift1m_rand_labels[i], (sift1m_rand_time[i], sift1m_rand_recall[i]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
    ax.annotate(sift1m_rpt_labels[i], (sift1m_rpt_time[i], sift1m_rpt_recall[i]),
                textcoords="offset points", xytext=(-10, 8), fontsize=7, color='#1565C0')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax = axes1[1]
ax.plot(sift1m_rand_dist_M, sift1m_rand_recall, 'o-', color='#F44336', markersize=7, linewidth=2,
        label='Random init', zorder=5)
ax.plot(sift1m_rpt_dist_M, sift1m_rpt_recall, 's-', color='#2196F3', markersize=7, linewidth=2,
        label='RP-Tree init', zorder=5)
for i in [0, 2, 5, 9]:
    ax.annotate(sift1m_rand_labels[i], (sift1m_rand_dist_M[i], sift1m_rand_recall[i]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
    ax.annotate(sift1m_rpt_labels[i], (sift1m_rpt_dist_M[i], sift1m_rpt_recall[i]),
                textcoords="offset points", xytext=(-10, 8), fontsize=7, color='#1565C0')
ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Dist Comps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/sift1m_random_vs_rptree.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/sift1m_random_vs_rptree.pdf', bbox_inches='tight')
print("Saved plots/sift1m_random_vs_rptree")

# ============================================================
# FIGURE 2: GIST 1M — Random vs RP-Tree Init (2 panels)
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes2[0]
ax.plot(gist1m_rand_time, gist1m_rand_recall, 'o-', color='#F44336', markersize=7, linewidth=2,
        label='Random init', zorder=5)
ax.plot(gist1m_rpt_time, gist1m_rpt_recall, 's-', color='#2196F3', markersize=7, linewidth=2,
        label='RP-Tree init', zorder=5)
for i in [0, 2, 5, 9]:
    ax.annotate(gist1m_rand_labels[i], (gist1m_rand_time[i], gist1m_rand_recall[i]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
    ax.annotate(gist1m_rpt_labels[i], (gist1m_rpt_time[i], gist1m_rpt_recall[i]),
                textcoords="offset points", xytext=(-10, 8), fontsize=7, color='#1565C0')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax = axes2[1]
ax.plot(gist1m_rand_dist_M, gist1m_rand_recall, 'o-', color='#F44336', markersize=7, linewidth=2,
        label='Random init', zorder=5)
ax.plot(gist1m_rpt_dist_M, gist1m_rpt_recall, 's-', color='#2196F3', markersize=7, linewidth=2,
        label='RP-Tree init', zorder=5)
for i in [0, 2, 5, 9]:
    ax.annotate(gist1m_rand_labels[i], (gist1m_rand_dist_M[i], gist1m_rand_recall[i]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
    ax.annotate(gist1m_rpt_labels[i], (gist1m_rpt_dist_M[i], gist1m_rpt_recall[i]),
                textcoords="offset points", xytext=(-10, 8), fontsize=7, color='#1565C0')
ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Dist Comps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/gist1m_random_vs_rptree.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist1m_random_vs_rptree.pdf', bbox_inches='tight')
print("Saved plots/gist1m_random_vs_rptree")

# ============================================================
# FIGURE 3: SIFT 1M — Filter vs Sampling (mc sweep vs pτ sweep)
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes3[0]
ax.plot(sift1m_mc_time, sift1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(sift1m_rand_time[1:], sift1m_rand_recall[1:], 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary p\u03c4)', zorder=5)
for i, lbl in enumerate(sift1m_mc_labels):
    ax.annotate(lbl, (sift1m_mc_time[i], sift1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=7, color='#1565C0')
for i in range(0, 9, 2):
    ax.annotate(sift1m_rand_labels[i+1], (sift1m_rand_time[i+1], sift1m_rand_recall[i+1]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

ax = axes3[1]
ax.plot(sift1m_mc_dist_M, sift1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(sift1m_rand_dist_M[1:], sift1m_rand_recall[1:], 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary p\u03c4)', zorder=5)
for i, lbl in enumerate(sift1m_mc_labels):
    ax.annotate(lbl, (sift1m_mc_dist_M[i], sift1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=7, color='#1565C0')
for i in range(0, 9, 2):
    ax.annotate(sift1m_rand_labels[i+1], (sift1m_rand_dist_M[i+1], sift1m_rand_recall[i+1]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
ax.set_xlabel('Distance Computations (millions)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('SIFT 1M: Recall vs Dist Comps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/sift1m_filter_vs_sampling.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/sift1m_filter_vs_sampling.pdf', bbox_inches='tight')
print("Saved plots/sift1m_filter_vs_sampling")

# ============================================================
# FIGURE 4: GIST 1M — Filter vs Sampling (mc sweep vs pτ sweep)
# ============================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes4[0]
ax.plot(gist1m_mc_time, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(gist1m_rand_time[1:], gist1m_rand_recall[1:], 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary p\u03c4)', zorder=5)
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_time[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=7, color='#1565C0')
for i in range(0, 9, 2):
    ax.annotate(gist1m_rand_labels[i+1], (gist1m_rand_time[i+1], gist1m_rand_recall[i+1]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M: Recall vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

ax = axes4[1]
ax.plot(gist1m_mc_dist_M, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
ax.plot(gist1m_rand_dist_M[1:], gist1m_rand_recall[1:], 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter mc=40 (vary p\u03c4)', zorder=5)
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_dist_M[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=7, color='#1565C0')
for i in range(0, 9, 2):
    ax.annotate(gist1m_rand_labels[i+1], (gist1m_rand_dist_M[i+1], gist1m_rand_recall[i+1]),
                textcoords="offset points", xytext=(-10, -14), fontsize=7, color='#C62828')
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
# FIGURE 5: Bar chart — % Speedup & % Recall Loss (Random init)
# ============================================================
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(bar_configs))
width = 0.25

# --- Speedup bars ---
ax = axes5[0]

g100k_speedup = [(gist100k_time[0] - gist100k_time[i+1]) / gist100k_time[0] * 100
                 for i in range(4)]
# GIST 1M random: indices 1,2,3,5 from the 10-element arrays
g1m_speedup = [(gist1m_rand_time[0] - gist1m_rand_time[i]) / gist1m_rand_time[0] * 100
               for i in bar_idx]
s1m_speedup = [(sift1m_rand_time[0] - sift1m_rand_time[i]) / sift1m_rand_time[0] * 100
               for i in bar_idx]

bars1 = ax.bar(x - width, g100k_speedup, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax.bar(x, g1m_speedup, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax.bar(x + width, s1m_speedup, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

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
ax.set_xticklabels(bar_configs, fontsize=10)
ax.set_ylabel('Time Saved (%)', fontsize=11)
ax.set_title('Wall-clock Speedup vs No Filter (Random Init)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)

# --- Recall loss bars ---
ax = axes5[1]

g100k_rloss = [(gist100k_recall[0] - gist100k_recall[i+1]) / gist100k_recall[0] * 100
               for i in range(4)]
g1m_rloss = [(gist1m_rand_recall[0] - gist1m_rand_recall[i]) / gist1m_rand_recall[0] * 100
             for i in bar_idx]
s1m_rloss = [(sift1m_rand_recall[0] - sift1m_rand_recall[i]) / sift1m_rand_recall[0] * 100
             for i in bar_idx]

bars1 = ax.bar(x - width, g100k_rloss, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax.bar(x, g1m_rloss, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax.bar(x + width, s1m_rloss, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f}%',
                ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(bar_configs, fontsize=10)
ax.set_ylabel('Recall Loss (%)', fontsize=11)
ax.set_title('Recall Loss vs No Filter (Random Init)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/bar_speedup_recall_random.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/bar_speedup_recall_random.pdf', bbox_inches='tight')
print("Saved plots/bar_speedup_recall_random")

# ============================================================
# FIGURE 6: Bar chart — % Speedup & % Recall Loss (RP-Tree init)
# ============================================================
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 6))

x2 = np.arange(len(bar_configs))
width2 = 0.30

ax = axes6[0]
g1m_rpt_speedup = [(gist1m_rpt_time[0] - gist1m_rpt_time[i]) / gist1m_rpt_time[0] * 100
                   for i in bar_idx]
s1m_rpt_speedup = [(sift1m_rpt_time[0] - sift1m_rpt_time[i]) / sift1m_rpt_time[0] * 100
                   for i in bar_idx]

bars1 = ax.bar(x2 - width2/2, g1m_rpt_speedup, width2, color='#2196F3', label='GIST 1M', edgecolor='white')
bars2 = ax.bar(x2 + width2/2, s1m_rpt_speedup, width2, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                    ha='center', fontsize=8, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h - 2.5, f'{h:.1f}%',
                    ha='center', fontsize=8, fontweight='bold', color='red')

ax.set_xticks(x2)
ax.set_xticklabels(bar_configs, fontsize=10)
ax.set_ylabel('Time Saved (%)', fontsize=11)
ax.set_title('Wall-clock Speedup vs No Filter (RP-Tree Init)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)

ax = axes6[1]
g1m_rpt_rloss = [(gist1m_rpt_recall[0] - gist1m_rpt_recall[i]) / gist1m_rpt_recall[0] * 100
                 for i in bar_idx]
s1m_rpt_rloss = [(sift1m_rpt_recall[0] - sift1m_rpt_recall[i]) / sift1m_rpt_recall[0] * 100
                 for i in bar_idx]

bars1 = ax.bar(x2 - width2/2, g1m_rpt_rloss, width2, color='#2196F3', label='GIST 1M', edgecolor='white')
bars2 = ax.bar(x2 + width2/2, s1m_rpt_rloss, width2, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f}%',
                ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels(bar_configs, fontsize=10)
ax.set_ylabel('Recall Loss (%)', fontsize=11)
ax.set_title('Recall Loss vs No Filter (RP-Tree Init)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/bar_speedup_recall_rptree.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/bar_speedup_recall_rptree.pdf', bbox_inches='tight')
print("Saved plots/bar_speedup_recall_rptree")

# ============================================================
# FIGURE 7: % Dist Comps Saved (Random init, all 3 datasets)
# ============================================================
fig7, ax7 = plt.subplots(1, 1, figsize=(8, 5.5))

g100k_dsaved = [(gist100k_dist_M[0] - gist100k_dist_M[i+1]) / gist100k_dist_M[0] * 100
                for i in range(4)]
g1m_dsaved = [(gist1m_rand_dist_M[0] - gist1m_rand_dist_M[i]) / gist1m_rand_dist_M[0] * 100
              for i in bar_idx]
s1m_dsaved = [(sift1m_rand_dist_M[0] - sift1m_rand_dist_M[i]) / sift1m_rand_dist_M[0] * 100
              for i in bar_idx]

bars1 = ax7.bar(x - width, g100k_dsaved, width, color='#4CAF50', label='GIST 100K', edgecolor='white')
bars2 = ax7.bar(x, g1m_dsaved, width, color='#2196F3', label='GIST 1M', edgecolor='white')
bars3 = ax7.bar(x + width, s1m_dsaved, width, color='#FF9800', label='SIFT 1M', edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                 ha='center', fontsize=8, fontweight='bold')

ax7.set_xticks(x)
ax7.set_xticklabels(bar_configs, fontsize=10)
ax7.set_ylabel('Distance Computations Saved (%)', fontsize=11)
ax7.set_title('Dist Comps Reduction vs No Filter (Random Init)', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
ax7.legend(fontsize=10)

plt.tight_layout()
plt.savefig('plots/bar_dist_saved_all.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/bar_dist_saved_all.pdf', bbox_inches='tight')
print("Saved plots/bar_dist_saved_all")

# ============================================================
# FIGURE 8: GIST 100K vs GIST 1M side-by-side Recall-Time
# ============================================================
fig8, axes8 = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes8[0]
ax.plot(gist100k_time, gist100k_recall, 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter (vary p\u03c4)', zorder=5)
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

ax = axes8[1]
ax.plot(gist1m_rand_time[1:], gist1m_rand_recall[1:], 's-', color='#F44336', markersize=9, linewidth=2,
        label='Proj filter (vary p\u03c4)', zorder=5)
ax.plot(gist1m_mc_time, gist1m_mc_recall, 'o-', color='#2196F3', markersize=9, linewidth=2,
        label='No filter (vary mc)', zorder=5)
for i in range(0, 9, 2):
    ax.annotate(gist1m_rand_labels[i+1], (gist1m_rand_time[i+1], gist1m_rand_recall[i+1]),
                textcoords="offset points", xytext=(-10, -15), fontsize=7, color='#C62828')
for i, lbl in enumerate(gist1m_mc_labels):
    ax.annotate(lbl, (gist1m_mc_time[i], gist1m_mc_recall[i]), textcoords="offset points",
                xytext=(-10, 10), fontsize=7, color='#1565C0')
ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('GIST 1M (960-dim)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.suptitle('Projection Filter vs Random Sampling \u2014 GIST Dataset', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/gist_100k_vs_1m.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/gist_100k_vs_1m.pdf', bbox_inches='tight')
print("Saved plots/gist_100k_vs_1m")

# ============================================================
# FIGURE 9: Normalized Recall-Time overlay (Random init, all datasets)
# ============================================================
fig9, ax9 = plt.subplots(1, 1, figsize=(10, 6))

# Normalize to no-filter baseline (index 0)
g100k_time_pct   = [t / gist100k_time[0] * 100 for t in gist100k_time]
g100k_recall_pct = [r / gist100k_recall[0] * 100 for r in gist100k_recall]

# For GIST 1M and SIFT 1M, use the full pτ sweep (including no-filter)
g1m_time_pct     = [t / gist1m_rand_time[0] * 100 for t in gist1m_rand_time]
g1m_recall_pct   = [r / gist1m_rand_recall[0] * 100 for r in gist1m_rand_recall]

s1m_time_pct     = [t / sift1m_rand_time[0] * 100 for t in sift1m_rand_time]
s1m_recall_pct   = [r / sift1m_rand_recall[0] * 100 for r in sift1m_rand_recall]

ax9.plot(g100k_time_pct, g100k_recall_pct, 'o-', color='#4CAF50', markersize=9, linewidth=2,
         label='GIST 100K (960-dim)', zorder=5)
ax9.plot(g1m_time_pct, g1m_recall_pct, 's-', color='#2196F3', markersize=9, linewidth=2,
         label='GIST 1M (960-dim)', zorder=5)
ax9.plot(s1m_time_pct, s1m_recall_pct, 'D-', color='#FF9800', markersize=9, linewidth=2,
         label='SIFT 1M (128-dim)', zorder=5)

for i, lbl in enumerate(gist100k_labels):
    ax9.annotate(lbl, (g100k_time_pct[i], g100k_recall_pct[i]), textcoords="offset points",
                 xytext=(8, 6), fontsize=7, color='#2E7D32')
for i in [0, 2, 5, 9]:
    ax9.annotate(gist1m_rand_labels[i], (g1m_time_pct[i], g1m_recall_pct[i]),
                 textcoords="offset points", xytext=(8, -12), fontsize=7, color='#1565C0')
for i in [0, 2, 5, 9]:
    ax9.annotate(sift1m_rand_labels[i], (s1m_time_pct[i], s1m_recall_pct[i]),
                 textcoords="offset points", xytext=(8, 6), fontsize=7, color='#E65100')

ax9.set_xlabel('Time (% of no-filter baseline)', fontsize=11)
ax9.set_ylabel('Recall (% of no-filter baseline)', fontsize=11)
ax9.set_title('Projection Filter: Normalized Recall-Time Tradeoff (Random Init)', fontsize=13, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend(fontsize=10)
ax9.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax9.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/all_datasets_normalized.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/all_datasets_normalized.pdf', bbox_inches='tight')
print("Saved plots/all_datasets_normalized")

# ============================================================
# Print summary tables
# ============================================================
print("\n" + "="*90)
print("SUMMARY TABLES")
print("="*90)

def print_table(title, labels, recall, dist_M, time, base_idx=0):
    print(f"\n{title}")
    print(f"{'Config':<15} {'Recall':>8} {'Dist (M)':>10} {'Time (s)':>10} {'Speedup':>10} {'Recall Loss':>12} {'Dist Saved':>12}")
    for i, lbl in enumerate(labels):
        sp = (time[base_idx] - time[i]) / time[base_idx] * 100
        rl = (recall[base_idx] - recall[i]) / recall[base_idx] * 100
        ds = (dist_M[base_idx] - dist_M[i]) / dist_M[base_idx] * 100
        print(f"{lbl:<15} {recall[i]:>8.4f} {dist_M[i]:>10.1f} {time[i]:>10.1f} {sp:>9.1f}% {rl:>11.2f}% {ds:>11.1f}%")

print_table("SIFT 1M Random Init (128-dim, K=10, mc=40, m=32):",
            sift1m_rand_labels, sift1m_rand_recall, sift1m_rand_dist_M, sift1m_rand_time)

print_table("SIFT 1M RP-Tree Init (128-dim, K=10, mc=40, m=32):",
            sift1m_rpt_labels, sift1m_rpt_recall, sift1m_rpt_dist_M, sift1m_rpt_time)

print_table("GIST 1M Random Init (960-dim, K=10, mc=40, m=32):",
            gist1m_rand_labels, gist1m_rand_recall, gist1m_rand_dist_M, gist1m_rand_time)

print_table("GIST 1M RP-Tree Init (960-dim, K=10, mc=40, m=32):",
            gist1m_rpt_labels, gist1m_rpt_recall, gist1m_rpt_dist_M, gist1m_rpt_time)

print_table("GIST 100K (960-dim, K=10, mc=40, m=32) \u2014 avg of runs 2-5:",
            gist100k_labels, gist100k_recall, gist100k_dist_M, gist100k_time)
