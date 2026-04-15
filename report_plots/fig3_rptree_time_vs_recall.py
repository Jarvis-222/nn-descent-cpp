"""Fig 3 — Same layout as Fig 1 but adds RP-Tree initialization.
(a) Wall-clock time vs construction recall for Random and RP-Tree init, on
    SIFT 1M and GIST 1M (4 curves total).
(b) Percentage wall-clock savings from the filter, 4 bars per p_tau.
No-filter baselines appear first in the legend; legend sits outside.
Double-column IEEE figure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from parse import collect_filter_sweep
from style import DOUBLE

OUT = os.path.dirname(os.path.abspath(__file__))

PT_ORDER = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

SERIES = [
    # (tag, init, dataset_label, init_label, color, marker)
    ("gist1m", "random", "GIST 1M", "Random",  "#2171b5", "o"),
    ("gist1m", "rptree", "GIST 1M", "RP-Tree", "#08306b", "D"),
    ("sift1m", "random", "SIFT 1M", "Random",  "#cc4c02", "s"),
    ("sift1m", "rptree", "SIFT 1M", "RP-Tree", "#7f2704", "^"),
]


def baseline_and_runs(tag, init):
    """Return (baseline, all pτ runs, pτ runs with construction recall)."""
    builds = collect_filter_sweep(tag, init)
    base = next((b for b in builds if b["pt"] is None), None)
    all_runs = [b for b in builds if b["pt"] is not None]
    all_runs.sort(key=lambda b: b["pt"])
    with_recall = [b for b in all_runs if b["recall"] > 0.0]
    return base, all_runs, with_recall


fig, (ax_line, ax_bar) = plt.subplots(
    1, 2, figsize=(DOUBLE, 3.2), gridspec_kw={"width_ratios": [1.1, 1.4]}
)

# ---- Panel (a): time vs recall ---------------------------------------------
handles, labels = [], []

# No-filter first
for tag, init, ds, ilabel, color, marker in SERIES:
    base, _, _ = baseline_and_runs(tag, init)
    if base is None:
        continue
    h = ax_line.scatter([base["recall"]], [base["total_time"]],
                        marker="*", s=70, color=color,
                        edgecolor="black", linewidth=0.6, zorder=5)
    handles.append(h)
    labels.append(f"{ds} · {ilabel} — no filter")

for tag, init, ds, ilabel, color, marker in SERIES:
    _, _, runs = baseline_and_runs(tag, init)
    if not runs:
        continue
    xs = [b["recall"] for b in runs]
    ys = [b["total_time"] for b in runs]
    ls = "-" if init == "random" else "--"
    h, = ax_line.plot(xs, ys, marker=marker, color=color, linewidth=1.2,
                      markersize=4, linestyle=ls)
    handles.append(h)
    labels.append(f"{ds} · {ilabel} — filter sweep")

ax_line.set_xlabel("Construction recall@10")
ax_line.set_ylabel("Wall-clock construction time (s)")
ax_line.set_title("(a) Time vs. recall")

# ---- Panel (b): grouped savings bars ---------------------------------------
x = np.arange(len(PT_ORDER))
n = len(SERIES)
width = 0.8 / n

for i, (tag, init, ds, ilabel, color, _) in enumerate(SERIES):
    base, _, with_recall = baseline_and_runs(tag, init)
    if base is None:
        continue
    # Only show bars where we also have recall, so line and bar panels
    # stay consistent per series.
    run_by_pt = {b["pt"]: b for b in with_recall}
    savings = []
    for pt in PT_ORDER:
        b = run_by_pt.get(pt)
        if b is None:
            savings.append(np.nan)
        else:
            savings.append(100.0 * (1.0 - b["total_time"] / base["total_time"]))
    offset = (i - (n - 1) / 2) * width
    hatch = "" if init == "random" else "///"
    ax_bar.bar(x + offset, savings, width, color=color, edgecolor="black",
               linewidth=0.3, hatch=hatch, label=f"{ds} · {ilabel}")

ax_bar.axhline(0, color="gray", linewidth=0.5)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{p:.2f}" for p in PT_ORDER])
ax_bar.set_xlabel(r"$p_\tau$")
ax_bar.set_ylabel("Wall-clock savings vs. no filter (%)")
ax_bar.set_title("(b) Filter time savings")
ax_bar.legend(loc="upper left", framealpha=0.9, fontsize=6)

# Shared legend for panel (a) below the figure (4 no-filter + 4 sweeps = 8)
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=6.5)

plt.tight_layout(rect=(0, 0.09, 1, 1))
out = os.path.join(OUT, "fig3_rptree_time_vs_recall")
plt.savefig(out + ".pdf")
plt.savefig(out + ".png", dpi=300)
print(f"Saved: {out}.{{pdf,png}}")
