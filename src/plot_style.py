"""One consistent, quiet chart style for notebooks and the app."""
import matplotlib as mpl

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # fixed series order
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def apply_style():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,  # grid behind the data
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
            "figure.dpi": 110,
            "legend.frameon": False,
        }
    )
