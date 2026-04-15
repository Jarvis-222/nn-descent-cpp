"""Render all report figures."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "fig1_time_vs_recall.py",
    "fig3_rptree_time_vs_recall.py",
    "fig5_search.py",
]
for s in scripts:
    print(f"--- {s} ---")
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
