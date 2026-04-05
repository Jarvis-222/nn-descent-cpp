#!/usr/bin/env python3
"""Plot GIST 1M construction tradeoffs for Random vs RP-tree init."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


FILES = {
    "Random": [
        "results/gist1m_random_nofilter_search.txt",
        "results/gist1m_random_pt095_search.txt",
        "results/gist1m_random_pt090_search.txt",
        "results/gist1m_random_pt080_search.txt",
        "results/gist1m_random_pt070_search.txt",
    ],
    "RP-tree": [
        "results/gist1m_rptree_nofilter_search.txt",
        "results/gist1m_rptree_pt090_search.txt",
        "results/gist1m_rptree_pt080_search.txt",
        "results/gist1m_rptree_pt070_search.txt",
    ],
}

OUT_PNG = "plots/gist1m_init_compare.png"
OUT_PDF = "plots/gist1m_init_compare.pdf"


def parse_report(path: str):
    text = Path(path).read_text()

    filter_match = re.search(r"filter=(.+)", text)
    summary_match = re.search(
        r"total_dist=(\d+)\s+total_filtered=(\d+)\s*\n"
        r"total_time=([0-9.]+)s final_recall=([0-9.]+)",
        text,
    )
    if not filter_match or not summary_match:
        raise ValueError(f"Could not parse {path}")

    filter_str = filter_match.group(1).strip()
    if filter_str == "OFF":
        label = "No filter"
        ptau = None
    else:
        ptau_match = re.search(r"pτ=([0-9.]+)", filter_str)
        ptau = float(ptau_match.group(1)) if ptau_match else None
        label = f"pτ={ptau:.2f}" if ptau is not None else filter_str

    return {
        "label": label,
        "ptau": ptau,
        "dist_m": int(summary_match.group(1)) / 1e6,
        "time_sec": float(summary_match.group(3)),
        "recall": float(summary_match.group(4)),
        "path": path,
    }


def sort_key(row):
    if row["ptau"] is None:
        return 2.0
    return row["ptau"]


def main():
    series = {name: [parse_report(path) for path in paths] for name, paths in FILES.items()}
    for name in series:
        series[name].sort(key=sort_key, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    colors = {"Random": "#d95f02", "RP-tree": "#1b9e77"}
    markers = {"Random": "o", "RP-tree": "s"}

    for name, rows in series.items():
        recalls = [r["recall"] for r in rows]
        times = [r["time_sec"] for r in rows]
        dists = [r["dist_m"] for r in rows]

        axes[0].plot(
            recalls, times,
            marker=markers[name], linewidth=2.2, markersize=8,
            color=colors[name], label=name,
        )
        axes[1].plot(
            recalls, dists,
            marker=markers[name], linewidth=2.2, markersize=8,
            color=colors[name], label=name,
        )

        for row in rows:
            axes[0].annotate(
                row["label"], (row["recall"], row["time_sec"]),
                textcoords="offset points", xytext=(7, 6), fontsize=8, color=colors[name]
            )
            axes[1].annotate(
                row["label"], (row["recall"], row["dist_m"]),
                textcoords="offset points", xytext=(7, 6), fontsize=8, color=colors[name]
            )

    axes[0].set_title("GIST 1M: Construction Time vs Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Construction Recall@10")
    axes[0].set_ylabel("Construction Time (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("GIST 1M: Dist Comps vs Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Construction Recall@10")
    axes[1].set_ylabel("Distance Computations (millions)")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)

    plt.suptitle(
        "Random vs RP-tree Initialization on GIST 1M\n"
        "Projection-filter sweep using saved construction reports",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
