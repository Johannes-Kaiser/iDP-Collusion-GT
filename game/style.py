"""Shared matplotlib styling matching Kaiser et al. (SaTML 2026) figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


PRIMARY_ORANGE = "#fd8c3b"
PRIMARY_BLUE = "#08306b"
ACCENT_GREY = "#4d4d4d"


def apply_paper_style():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def column_size(rows: int = 1, aspect: float = 0.5):
    """Return a (w, h) figsize for an IEEEtran single-column figure."""
    width = 3.4
    return (width, width * aspect * rows)


def two_column_size(rows: int = 1, aspect: float = 0.36):
    """Two-column wide figure."""
    width = 7.1
    return (width, width * aspect * rows)


def orange_palette(n: int):
    cmap = cm.get_cmap("Oranges")
    return [cmap(x) for x in np.linspace(0.35, 0.95, n)]


def blue_palette(n: int):
    cmap = cm.get_cmap("Blues")
    return [cmap(x) for x in np.linspace(0.35, 0.95, n)]
