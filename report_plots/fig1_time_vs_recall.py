"""Fig 1 — Two panels, random init only.
(a) Wall-clock construction time vs construction recall, SIFT 1M + GIST 1M.
(b) Percentage wall-clock savings from the filter, grouped by p_tau and dataset.
No-filter is the first legend entry; legend sits outside the axes.
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
    ("gist1m", "GIST 1M ($d{=}960$)", "#2171b5", "o"),
    ("sift1m", "SIFT 1M ($d{=}128$)", "#cc4c02", "s"),
]


def baseline_and_runs(tag, init):
    """Return (no-filter baseline, all pτ runs, pτ runs with recall).

    Bar chart can use every run (only timing needed); the line plot needs
    configs that also have a construction recall.
    """
    builds = collect_filter_sweep(tag, init)
    base = next((b for b in builds if b["pt"] is None), None)
    all_runs = [b for b in builds if b["pt"] is not None]
    all_runs.sort(key=lambda b: b["pt"])
    with_recall = [b for b in all_runs if b["recall"] > 0.0]
    return base, all_runs, with_recall


fig, (ax_line, ax_bar) = plt.subplots(
    1, 2, figsize=(DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.1, 1.3]}
)

# ---- Panel (a): time vs recall ---------------------------------------------
handles, labels = [], []

# No-filter first in legend
for tag, title, color, marker in DATASETS:
    base, _, _ = baseline_and_runs(tag, "random")
    if base is None:
        continue
    h = ax_line.scatter([base["recall"]], [base["total_time"]],
                        marker="*", s=70, color=color,
                        edgecolor="black", linewidth=0.6, zorder=5)
    handles.append(h)
    labels.append(f"{title} — no filter")

for tag, title, color, marker in DATASETS:
    _, _, runs = baseline_and_runs(tag, "random")
    if not runs:
        continue
    xs = [b["recall"] for b in runs]
    ys = [b["total_time"] for b in runs]
    h, = ax_line.plot(xs, ys, marker=marker, color=color, linewidth=1.3,
                      markersize=4, linestyle="-")
    handles.append(h)
    labels.append(f"{title} — filter sweep")

ax_line.set_xlabel("Construction recall@10")
ax_line.set_ylabel("Wall-clock construction time (s)")
ax_line.set_title("(a) Time vs. recall")

# ---- Panel (b): % wall-clock savings grouped bars --------------------------
x = np.arange(len(PT_ORDER))
width = 0.36

for i, (tag, title, color, _) in enumerate(DATASETS):
    base, _, with_recall = baseline_and_runs(tag, "random")
    if base is None:
        continue
    # Only include bars for configs where construction recall was measured,
    # so the line panel (a) and bar panel (b) show the same set of pτ.
    run_by_pt = {b["pt"]: b for b in with_recall}
    savings = []
    for pt in PT_ORDER:
        b = run_by_pt.get(pt)
        if b is None:
            savings.append(np.nan)
        else:
            savings.append(100.0 * (1.0 - b["total_time"] / base["total_time"]))
    offset = (-width / 2) if i == 0 else (width / 2)
    ax_bar.bar(x + offset, savings, width, color=color, edgecolor="black",
               linewidth=0.3, label=title)

ax_bar.axhline(0, color="gray", linewidth=0.5)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{p:.2f}" for p in PT_ORDER])
ax_bar.set_xlabel(r"$p_\tau$")
ax_bar.set_ylabel("Wall-clock savings vs. no filter (%)")
ax_bar.set_title("(b) Filter time savings")
ax_bar.legend(loc="upper left", framealpha=0.9)

# Shared legend for panel (a) below the figure
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7)

plt.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(OUT, "fig1_time_vs_recall")
plt.savefig(out + ".pdf")
plt.savefig(out + ".png", dpi=300)
print(f"Saved: {out}.{{pdf,png}}")
