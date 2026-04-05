#!/usr/bin/env python3
"""Plot search recall/QPS curves from a saved GIST 1M search log."""

from collections import OrderedDict
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt


def parse_sections(text: str):
    runs = []
    lines = text.splitlines()
    current = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("n=") and "k=" in stripped:
            if current and current.get("rows"):
                runs.append(current)
            current = {"rows": []}
        elif current is None:
            continue
        elif stripped.startswith("init="):
            current["init"] = stripped.replace("init=", "", 1)
        elif stripped.startswith("filter="):
            current["filter"] = stripped.replace("filter=", "", 1)
        elif stripped.startswith("ef,recall@"):
            current["header_seen"] = True
        elif current.get("header_seen") and re.match(r"^\d+,", stripped):
            ef, recall, qps, dist, time_sec = stripped.split(",")
            current["rows"].append({
                "ef": int(ef),
                "recall": float(recall),
                "qps": float(qps),
                "dist_comps": int(dist),
                "time_sec": float(time_sec),
            })

    if current and current.get("rows"):
        runs.append(current)
    return runs


def run_label(run):
    filt = run.get("filter", "unknown")
    if filt == "OFF":
        return f'{run.get("init", "Unknown")} | No filter'
    ptau_match = re.search(r"pτ=([0-9.]+)", filt)
    if ptau_match:
        return f'{run.get("init", "Unknown")} | pτ={float(ptau_match.group(1)):.2f}'
    return f'{run.get("init", "Unknown")} | {filt}'


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_gist1m_search_log.py <search-log.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()
    runs = parse_sections(text)
    if not runs:
        raise SystemExit(f"No search sections found in {path}")

    grouped = OrderedDict()
    for run in runs:
        grouped[run_label(run)] = run["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]

    for i, (label, rows) in enumerate(grouped.items()):
        rows = sorted(rows, key=lambda r: r["ef"])
        ef = [r["ef"] for r in rows]
        recall = [r["recall"] for r in rows]
        qps = [r["qps"] for r in rows]

        axes[0].plot(ef, recall, marker=markers[i % len(markers)], linewidth=2.2,
                     markersize=7, color=palette[i % len(palette)], label=label)
        axes[1].plot(ef, qps, marker=markers[i % len(markers)], linewidth=2.2,
                     markersize=7, color=palette[i % len(palette)], label=label)

    axes[0].set_title("GIST 1M: Search Recall vs ef", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("ef")
    axes[0].set_ylabel("Search Recall@10")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("GIST 1M: QPS vs ef", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("ef")
    axes[1].set_ylabel("QPS")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)

    out_png = "plots/gist1m_search_from_log.png"
    out_pdf = "plots/gist1m_search_from_log.pdf"
    plt.suptitle(
        f"GIST 1M Search Comparison\nParsed from {path.name}",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
