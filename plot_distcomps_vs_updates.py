"""
Plot waste analysis for GIST 1M baseline (no filter).
Sub-chart 1: % of distance computations that produce a graph update (Random vs RP-Tree)
Sub-chart 2: Cumulative distance computations vs recall (diminishing returns)
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

# ─── Parse CSV section from result file ──────────────────────────────────

def parse_result_file(path):
    iters, dist_comps, updates, recall = [], [], [], []
    in_csv = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("iter,dist_comps"):
                in_csv = True
                continue
            if in_csv and line and line[0].isdigit():
                parts = line.split(",")
                iters.append(int(parts[0]))
                dist_comps.append(int(parts[1]))
                updates.append(int(parts[4]))
                recall.append(float(parts[6]))
            elif in_csv and not line:
                break
    return (np.array(iters), np.array(dist_comps),
            np.array(updates), np.array(recall))


# ─── Load data ───────────────────────────────────────────────────────────

r_iters, r_dist, r_upd, r_recall = parse_result_file("results/gist1m_nofilter_r1.txt")
t_iters, t_dist, t_upd, t_recall = parse_result_file("results/gist1m_rptree_nofilter_search.txt")

# ─── Plot ────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# =====================================================================
# Sub-chart 1: % of dist comps that produce an update
# =====================================================================
r_pct = (r_dist - r_upd) / r_dist * 100
t_pct = (t_dist - t_upd) / t_dist * 100

max_iter = max(len(r_iters), len(t_iters))
x = np.arange(1, max_iter + 1)
width = 0.35

# Pad RP-Tree with NaN for missing iterations
t_pct_padded = np.full(max_iter, np.nan)
t_pct_padded[:len(t_pct)] = t_pct
r_pct_padded = np.full(max_iter, np.nan)
r_pct_padded[:len(r_pct)] = r_pct

ax1.bar(x - width/2, r_pct_padded, width, color="#4C72B0", alpha=0.85,
        label="Random Init", zorder=3)
ax1.bar(x + width/2, t_pct_padded, width, color="#DD8452", alpha=0.85,
        label="RP-Tree Init", zorder=3)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Wasted Distance Computations (%)")
ax1.set_title("Fraction of Distance Computations\nthat Do Not Produce a Graph Update", fontsize=12)
ax1.set_xticks(x)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.2, axis='y', zorder=0)
ax1.set_ylim(bottom=0)

# Annotate the last iteration values
for iters_arr, pct_arr, offset, color in [
    (r_iters, r_pct, -width/2, "#4C72B0"),
    (t_iters, t_pct, +width/2, "#DD8452"),
]:
    last = len(pct_arr) - 1
    ax1.annotate(f"{pct_arr[last]:.1f}%",
                 xy=(iters_arr[last] + offset, pct_arr[last]),
                 xytext=(0, 8), textcoords="offset points",
                 fontsize=7, color=color, ha='center', fontweight='bold')

# =====================================================================
# Sub-chart 2: Cumulative dist comps vs recall (iteration-only, no init)
# =====================================================================
r_cumul = np.cumsum(r_dist)
t_cumul = np.cumsum(t_dist)

ax2.plot(r_cumul / 1e6, r_recall, color="#4C72B0", marker='o', markersize=5,
         linewidth=2, label="Random Init", zorder=5)
ax2.plot(t_cumul / 1e6, t_recall, color="#DD8452", marker='s', markersize=5,
         linewidth=2, label="RP-Tree Init", zorder=5)

# Shade the diminishing returns region for Random
# Find where recall reaches 90% of final
r_final = r_recall[-1]
threshold_idx = np.searchsorted(r_recall, 0.9 * r_final)
if threshold_idx < len(r_cumul):
    ax2.axvspan(r_cumul[threshold_idx] / 1e6, r_cumul[-1] / 1e6,
                alpha=0.08, color='red', zorder=0)
    ax2.annotate("Diminishing returns",
                 xy=((r_cumul[threshold_idx] + r_cumul[-1]) / 2 / 1e6, r_recall[threshold_idx]),
                 xytext=(0, -30), textcoords="offset points",
                 fontsize=9, color="#C44E52", ha='center', style='italic',
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.2))

# Annotate iteration numbers on markers
for iters_arr, cumul, recall, color in [
    (r_iters, r_cumul, r_recall, "#4C72B0"),
    (t_iters, t_cumul, t_recall, "#DD8452"),
]:
    for i in [0, len(iters_arr)//2, -1]:
        ax2.annotate(f"iter {iters_arr[i]}",
                     xy=(cumul[i] / 1e6, recall[i]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=7, color=color)

ax2.set_xlabel("Cumulative Distance Computations (millions)")
ax2.set_ylabel("Recall@10")
ax2.set_title("Recall vs Cumulative Distance Computations\n(Iteration Phase Only)", fontsize=12)
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(True, alpha=0.2, zorder=0)

fig.suptitle("GIST 1M (d=960) — Wasted Computation in NN-Descent Without Filtering",
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig("plots/gist1m_distcomps_vs_updates.png", dpi=200, bbox_inches="tight")
plt.savefig("plots/gist1m_distcomps_vs_updates.pdf", bbox_inches="tight")
print("Saved: plots/gist1m_distcomps_vs_updates.{png,pdf}")
