"""Fig 2 — Projection filtering vs sampling reduction (mc sweep).
2x2 grid: rows = {SIFT 1M, GIST 1M}, cols = {Recall vs Time, Recall vs Dist Comps}.
Each panel overlays the mc-reduction curve (no filter, vary mc) and the filter
sweep curve (mc=40, vary pτ).  Matches the style of the uploaded fig2 reference.
Double-column IEEE figure.
"""
import os
import matplotlib.pyplot as plt
from parse import collect_filter_sweep, collect_mc_sweep
from style import DOUBLE

OUT = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.0))

PANELS = [
    ("sift1m", "SIFT 1M ($d{=}128$)"),
    ("gist1m", "GIST 1M ($d{=}960$)"),
]


def plot_row(row_axes, tag, dataset_title):
    ax_time, ax_dist = row_axes

    # --- MC sweep (no filter, vary mc) ---
    mc_runs = collect_mc_sweep(tag)
    mc_time = [b["total_time"] for b in mc_runs]
    mc_dist = [b["total_dist"] / 1e6 for b in mc_runs]
    mc_rec = [b["recall"] for b in mc_runs]

    ax_time.plot(mc_time, mc_rec, 'o-', color='#2171b5', linewidth=1.3,
                 markersize=4, label='No filter (vary $mc$)', zorder=5)
    ax_dist.plot(mc_dist, mc_rec, 'o-', color='#2171b5', linewidth=1.3,
                 markersize=4, label='No filter (vary $mc$)', zorder=5)
    for b in mc_runs:
        ax_time.annotate(f'mc={b["mc"]}',
                         (b["total_time"], b["recall"]),
                         textcoords="offset points", xytext=(-6, 6),
                         fontsize=5.5, color='#08306b')
        ax_dist.annotate(f'mc={b["mc"]}',
                         (b["total_dist"] / 1e6, b["recall"]),
                         textcoords="offset points", xytext=(-6, 6),
                         fontsize=5.5, color='#08306b')

    # --- Filter sweep (mc=40, vary pτ) ---
    builds = collect_filter_sweep(tag, "random")
    filt = [b for b in builds if b["pt"] is not None and b["recall"] > 0.0]
    filt.sort(key=lambda b: b["pt"])
    f_time = [b["total_time"] for b in filt]
    f_dist = [b["total_dist"] / 1e6 for b in filt]
    f_rec = [b["recall"] for b in filt]

    ax_time.plot(f_time, f_rec, 's-', color='#cc4c02', linewidth=1.3,
                 markersize=4, label=r'Proj filter $mc{=}40$ (vary $p_\tau$)',
                 zorder=5)
    ax_dist.plot(f_dist, f_rec, 's-', color='#cc4c02', linewidth=1.3,
                 markersize=4, label=r'Proj filter $mc{=}40$ (vary $p_\tau$)',
                 zorder=5)
    for i, b in enumerate(filt):
        if i % 2 == 0:
            ax_time.annotate(fr'$p_\tau$={b["pt"]:.2f}',
                             (b["total_time"], b["recall"]),
                             textcoords="offset points", xytext=(-6, -10),
                             fontsize=5.5, color='#7f2704')
            ax_dist.annotate(fr'$p_\tau$={b["pt"]:.2f}',
                             (b["total_dist"] / 1e6, b["recall"]),
                             textcoords="offset points", xytext=(-6, -10),
                             fontsize=5.5, color='#7f2704')

    ax_time.set_title(f"{dataset_title}: Recall vs Time")
    ax_time.set_xlabel("Wall-clock time (s)")
    ax_time.set_ylabel("Construction recall@10")
    ax_time.legend(loc="lower right", fontsize=6, framealpha=0.9)

    ax_dist.set_title(f"{dataset_title}: Recall vs Dist Comps")
    ax_dist.set_xlabel("Distance computations (millions)")
    ax_dist.set_ylabel("Construction recall@10")
    ax_dist.legend(loc="lower right", fontsize=6, framealpha=0.9)


for i, (tag, title) in enumerate(PANELS):
    plot_row(axes[i], tag, title)

plt.tight_layout()
out = os.path.join(OUT, "fig2_filtering_vs_sampling")
plt.savefig(out + ".pdf")
plt.savefig(out + ".png", dpi=300)
print(f"Saved: {out}.{{pdf,png}}")
