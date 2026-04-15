"""Fig 5: Downstream search recall@10 vs QPS on GIST 1M graphs
built under varying pτ. Random-init panel and RP-Tree-init panel
side-by-side show the Pareto curves collapse — construction-recall
loss does not propagate to search quality.
Double-column.
"""
import os
import matplotlib.pyplot as plt
from parse import collect_search_sweep
from style import DOUBLE, PT_COLORS, pt_label

OUT = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)

panels = [
    ("random", "Random init", "o", "-"),
    ("rptree", "RP-Tree init", "s", "--"),
]

keep = [None, 0.99, 0.95, 0.90, 0.80, 0.70, 0.60]

for ax, (init, title, marker, ls) in zip(axes, panels):
    runs = collect_search_sweep("gist1m", init)
    for r in runs:
        if r["pt"] not in keep:
            continue
        recs = [row[1] for row in r["rows"]]
        qps = [row[2] for row in r["rows"]]
        c = PT_COLORS.get(r["pt"], "gray")
        lw = 1.5 if r["pt"] is None else 0.9
        ax.plot(recs, qps, marker=marker, color=c, linestyle=ls,
                linewidth=lw, markersize=3, label=pt_label(r["pt"]))
    ax.set_title(f"GIST 1M — {title}")
    ax.set_xlabel("Search recall@10")
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=2, fontsize=6,
              framealpha=0.9, handlelength=1.6)

axes[0].set_ylabel("QPS")

plt.tight_layout()
out = os.path.join(OUT, "fig5_search")
plt.savefig(out + ".pdf")
plt.savefig(out + ".png", dpi=300)
print(f"Saved: {out}.{{pdf,png}}")
