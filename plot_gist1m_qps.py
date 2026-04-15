"""
GIST 1M: Recall@10 vs QPS — Random and RP-Tree init on one plot.
"""
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

LOG_FILE = "search_all.log"

def parse_search_log(path):
    results = {}
    current_graph = None
    in_csv = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"graph=(.+)", line)
            if m:
                current_graph = m.group(1)
                results[current_graph] = []
                in_csv = False
                continue
            if line.startswith("ef,recall@"):
                in_csv = True
                continue
            if in_csv and current_graph and re.match(r"\d+,", line):
                parts = line.split(",")
                ef = int(parts[0])
                recall = float(parts[1])
                qps = float(parts[2])
                dist_comps = int(parts[3])
                results[current_graph].append((ef, recall, qps, dist_comps))
            if line == "Done!":
                in_csv = False
    return results

results = parse_search_log(LOG_FILE)

def extract_pt(name):
    m = re.search(r"pt(\d+)", name)
    if m:
        return int(m.group(1)) / 100.0
    if "nofilter" in name:
        return None
    return None

# Select key configs to keep the plot readable
key_random = ["gist1m_random_nofilter", "gist1m_random_pt095",
              "gist1m_random_pt090", "gist1m_random_pt080",
              "gist1m_random_pt070", "gist1m_random_pt060"]

key_rptree = ["gist1m_rptree_pt099", "gist1m_rptree_pt095",
              "gist1m_rptree_pt090", "gist1m_rptree_pt080",
              "gist1m_rptree_pt070", "gist1m_rptree_pt060"]

fig, ax = plt.subplots(figsize=(9, 6))

# Color palette
random_colors = {
    None:  "#2c3e50",  # nofilter - dark
    0.99:  "#2980b9",
    0.95:  "#3498db",
    0.90:  "#1abc9c",
    0.85:  "#16a085",
    0.80:  "#27ae60",
    0.75:  "#2ecc71",
    0.70:  "#f39c12",
    0.65:  "#e67e22",
    0.60:  "#e74c3c",
}

rptree_colors = {
    None:  "#2c3e50",
    0.99:  "#8e44ad",
    0.95:  "#9b59b6",
    0.90:  "#c0392b",
    0.85:  "#d35400",
    0.80:  "#e74c3c",
    0.75:  "#e67e22",
    0.70:  "#f39c12",
    0.65:  "#f1c40f",
    0.60:  "#e74c3c",
}

# Plot Random init
for name in key_random:
    if name not in results:
        continue
    data = results[name]
    recalls = [d[1] for d in data]
    qps_vals = [d[2] for d in data]
    pt = extract_pt(name)
    color = random_colors.get(pt, "gray")
    label_str = "No filter" if pt is None else f"pτ={pt:.2f}"
    lw = 2.5 if pt is None else 1.5
    ax.plot(recalls, qps_vals, marker="o", markersize=6, label=f"Random — {label_str}",
            color=color, linewidth=lw, linestyle="-")

# Plot RP-Tree init
for name in key_rptree:
    if name not in results:
        continue
    data = results[name]
    recalls = [d[1] for d in data]
    qps_vals = [d[2] for d in data]
    pt = extract_pt(name)
    color = rptree_colors.get(pt, "gray")
    label_str = f"pτ={pt:.2f}" if pt else "No filter"
    ax.plot(recalls, qps_vals, marker="s", markersize=6, label=f"RP-Tree — {label_str}",
            color=color, linewidth=1.5, linestyle="--")

ax.set_xlabel("Recall@10", fontsize=13)
ax.set_ylabel("Queries Per Second (QPS)", fontsize=13)
ax.set_title("GIST 1M — Search Performance: Recall@10 vs QPS", fontsize=14)
ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/gist1m_recall_vs_qps.png", dpi=200, bbox_inches="tight")
plt.savefig("plots/gist1m_recall_vs_qps.pdf", bbox_inches="tight")
print("Saved: plots/gist1m_recall_vs_qps.{png,pdf}")
