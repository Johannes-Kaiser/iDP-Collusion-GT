"""Re-render E1 figure: drop the duplicated naive-A1 panel, show BR direction.

Uses the cached E1.pkl, so it is fast and independent of the running sweep.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    two_column_size,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache"

apply_paper_style()
with open(CACHE_DIR / "E1.pkl", "rb") as f:
    data = pickle.load(f)
actions = data["actions"]
U_aw = data["U_aw"]
U_nv = data["U_nv"]
A_aw = data["A_aw"]
S_aw = data["S_aw"]

K = len(actions)

fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.32),
                          gridspec_kw=dict(width_ratios=[1, 1, 1]))

panels = [
    (axes[0], U_aw[:, :, 0], r"Aware regime: $u_1$", "Blues"),
    (axes[1], U_nv[:, :, 0], r"Naive regime: $u_1$", "Blues"),
    (axes[2], A_aw[:, :, 0], r"Realised $A_1$", "Oranges"),
]
for ax, M, title, cmap in panels:
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"{a:g}" for a in actions])
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{a:g}" for a in actions])
    ax.set_xlabel(r"$\varepsilon_2$ (other)")
    ax.set_ylabel(r"$\varepsilon_1$ (own)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay best response of player 1: argmax over rows for each column
    if title.startswith("Aware") or title.startswith("Naive"):
        best_row = np.argmax(M, axis=0)
        ax.plot(range(K), best_row, "o-", color="white",
                markeredgecolor="black", markersize=4, linewidth=1.2,
                label="BR$_1$")
        ax.legend(loc="lower right", fontsize=6, frameon=True,
                  framealpha=0.85, edgecolor="none")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_E1_payoff_grids.pdf")
plt.savefig(FIG_DIR / "fig_E1_payoff_grids.svg")
print("wrote", FIG_DIR / "fig_E1_payoff_grids.pdf")
