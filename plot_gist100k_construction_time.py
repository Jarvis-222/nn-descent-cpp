#!/usr/bin/env python3
"""Plot GIST 100K construction time for random vs RP-tree init."""

import re

import matplotlib.pyplot as plt


REPORTS = {
    "Random": {
        "No filter": "results/gist100k_random_search_baseline.txt",
        "pτ=0.99": "results/gist100k_random_proj_pt099_search.txt",
        "pτ=0.95": "results/gist100k_random_proj_pt095_search.txt",
        "pτ=0.90": "results/gist100k_random_proj_pt090_search.txt",
        "pτ=0.80": "results/gist100k_random_proj_pt080_search.txt",
    },
    "RP-tree": {
        "No filter": "results/gist100k_rptree_nofilter_search.txt",
        "pτ=0.99": "results/gist100k_rptree_pt099_search.txt",
        "pτ=0.95": "results/gist100k_rptree_pt095_search.txt",
        "pτ=0.90": "results/gist100k_rptree_pt090_search.txt",
        "pτ=0.80": "results/gist100k_rptree_pt080_search.txt",
    },
}

ORDER = ["No filter", "pτ=0.99", "pτ=0.95", "pτ=0.90", "pτ=0.80"]
OUT_PNG = "plots/gist100k_construction_time_compare.png"
OUT_PDF = "plots/gist100k_construction_time_compare.pdf"


def extract_total_time(path):
    with open(path) as f:
        text = f.read()
    match = re.search(r"total_time=([0-9.]+)s", text)
    if not match:
        raise ValueError(f"Could not find total_time in {path}")
    return float(match.group(1))


def main():
    x = list(range(len(ORDER)))
    random_times = [extract_total_time(REPORTS["Random"][cfg]) for cfg in ORDER]
    rptree_times = [extract_total_time(REPORTS["RP-tree"][cfg]) for cfg in ORDER]

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x, random_times, "o-", linewidth=2.2, markersize=7, color="#1f77b4", label="Random init")
    plt.plot(x, rptree_times, "s-", linewidth=2.2, markersize=7, color="#d62728", label="RP-tree init")

    plt.xticks(x, ORDER)
    plt.ylabel("Construction Time (s)")
    plt.xlabel("Filter Setting")
    plt.title("GIST 100K: Construction Time\nRandom vs RP-tree init", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
