"""Fig 4 — Distance-computation savings from the projection filter.
(a) % dist comps saved vs pτ, random init (SIFT 1M + GIST 1M grouped bars).
(b) Same for RP-Tree init.
Double-column IEEE figure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from parse import collect_filter_sweep
from style import DOUBLE

OUT = os.path.dirname(os.path.abspath(__file__))

PT_ORDER = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

DATASETS = [
    ("gist1m", "GIST 1M", "#2171b5"),
    ("sift1m", "SIFT 1M", "#cc4c02"),
]

fig, (ax_rand, ax_rpt) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8), sharey=True)

x = np.arange(len(PT_ORDER))
width = 0.36


def plot_bars(ax, init, title):
    for i, (tag, label, color) in enumerate(DATASETS):
        builds = collect_filter_sweep(tag, init)
        base = next((b for b in builds if b["pt"] is None), None)
        if base is None:
            continue
        run_by_pt = {b["pt"]: b for b in builds if b["pt"] is not None}
        savings = []
        for pt in PT_ORDER:
            b = run_by_pt.get(pt)
            if b is None:
                savings.append(np.nan)
            else:
                savings.append(100.0 * (1.0 - b["total_dist"] / base["total_dist"]))
        offset = (-width / 2) if i == 0 else (width / 2)
        hatch = "" if init == "random" else "///"
        ax.bar(x + offset, savings, width, color=color, edgecolor="black",
               linewidth=0.3, hatch=hatch, label=label)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in PT_ORDER])
    ax.set_xlabel(r"$p_\tau$")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6)


plot_bars(ax_rand, "random", "(a) Random init")
plot_bars(ax_rpt, "rptree", "(b) RP-Tree init")

ax_rand.set_ylabel("Distance computations saved (%)")

plt.tight_layout()
out = os.path.join(OUT, "fig4_dist_comps")
plt.savefig(out + ".pdf")
plt.savefig(out + ".png", dpi=300)
print(f"Saved: {out}.{{pdf,png}}")