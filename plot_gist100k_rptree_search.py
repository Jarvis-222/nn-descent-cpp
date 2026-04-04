#!/usr/bin/env python3
"""Plot GIST 100K RP-tree-init search impact across projection-filter settings."""

import csv
from collections import OrderedDict

import matplotlib.pyplot as plt


CSV_PATH = "results/gist100k_rptree_search_sweep.csv"
OUT_PNG = "plots/gist100k_rptree_search_vs_ef.png"
OUT_PDF = "plots/gist100k_rptree_search_vs_ef.pdf"


def load_rows():
    series = OrderedDict()
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["config"]
            if key not in series:
                series[key] = {
                    "construction_recall": float(row["construction_recall"]),
                    "ef": [],
                    "search_recall": [],
                    "qps": [],
                }
            series[key]["ef"].append(int(row["ef"]))
            series[key]["search_recall"].append(float(row["search_recall"]))
            series[key]["qps"].append(float(row["qps"]))
    return series


def main():
    data = load_rows()

    colors = {
        "No filter": "#1f77b4",
        "pτ=0.99": "#2ca02c",
        "pτ=0.95": "#ff7f0e",
        "pτ=0.90": "#d62728",
        "pτ=0.80": "#9467bd",
    }
    markers = {
        "No filter": "o",
        "pτ=0.99": "s",
        "pτ=0.95": "^",
        "pτ=0.90": "D",
        "pτ=0.80": "P",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for config, vals in data.items():
        label = f"{config} (graph={vals['construction_recall']:.4f})"
        axes[0].plot(
            vals["ef"], vals["search_recall"],
            marker=markers.get(config, "o"),
            color=colors.get(config),
            linewidth=2.2,
            markersize=7,
            label=label,
        )
        axes[1].plot(
            vals["ef"], vals["qps"],
            marker=markers.get(config, "o"),
            color=colors.get(config),
            linewidth=2.2,
            markersize=7,
            label=label,
        )

    axes[0].set_title("GIST 100K RP-Tree: Search Recall vs ef", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("ef")
    axes[0].set_ylabel("Search Recall@10")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("GIST 100K RP-Tree: QPS vs ef", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("ef")
    axes[1].set_ylabel("QPS")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)

    plt.suptitle(
        "Projection Filtering Impact on Downstream Search\n"
        "RP-tree init, 1K held-out GIST queries, shared search algorithm",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
