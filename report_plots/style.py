"""IEEE-template-friendly matplotlib defaults."""
import matplotlib as mpl

# IEEE conference template column widths (inches)
COL = 3.487
DOUBLE = 7.16

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

# Color palette for pτ values (None = no-filter baseline)
PT_COLORS = {
    None:  "#1a1a1a",
    0.99:  "#08306b",
    0.95:  "#2171b5",
    0.90:  "#4292c6",
    0.85:  "#41ab5d",
    0.80:  "#238b45",
    0.75:  "#fe9929",
    0.70:  "#ec7014",
    0.65:  "#cc4c02",
    0.60:  "#a50f15",
}


def pt_label(pt):
    return "no filter" if pt is None else fr"$p_\tau{{=}}{pt:.2f}$"
