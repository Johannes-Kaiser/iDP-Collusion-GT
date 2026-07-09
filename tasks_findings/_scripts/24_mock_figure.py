"""T24 -- mock preview of the paper figure fig_H1_n2_measured_NE.

Runs on the login node in <5 s. Renders the three-panel layout with SYNTHETIC
LiRA advantages generated from the PLD upper bound minus a random shrinkage
(the "privacy-onion" gap reported in the companion paper 2601.12922v1
Section 5). This is a design preview only; the real figure is produced from
_cache_H1/H1_summary.pkl after Slurm completes.

Output: tasks_findings/_scripts/_out/fig_H1_n2_preview.pdf (+ .svg).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(REPO))

from game.style import apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, two_column_size  # noqa: E402


EPS_GRID = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
K = len(EPS_GRID)
OUT = Path(__file__).with_suffix("").parent / "_out"


def _synthetic_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Analytical A^PLD: decreasing in eps, silo-symmetric structure with the
    # C1 asymmetry -- larger eps -> larger A.
    A_pld = np.zeros((K, K, 2))
    for i, e1 in enumerate(EPS_GRID):
        for j, e2 in enumerate(EPS_GRID):
            A_pld[i, j, 0] = 0.5 * (1 - np.exp(-e1 / 8))
            A_pld[i, j, 1] = 0.5 * (1 - np.exp(-e2 / 8))
    # LiRA shrinkage ~ privacy onion: A_meas <= A_pld with per-cell noise.
    shrink = 0.55 + 0.15 * rng.random((K, K, 2))
    noise = 0.02 * rng.standard_normal((K, K, 2))
    A_meas = np.clip(A_pld * shrink + noise, 0.0, A_pld)
    # V: roughly increasing in max(eps_1, eps_2) with a low-eps plateau.
    V = 0.55 + 0.15 * np.tanh(np.log2(np.maximum(EPS_GRID[:, None],
                                                  EPS_GRID[None, :])) / 3)
    return A_pld, A_meas, V


def _find_ne(U1: np.ndarray, U2: np.ndarray):
    ne = []
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            if (U1[i, j] >= U1[:, j].max() - 1e-9
                    and U2[i, j] >= U2[i, :].max() - 1e-9):
                ne.append((i, j))
    return ne


def _heatmap(ax, U, title, ne):
    im = ax.imshow(U, origin="lower", cmap="viridis", aspect="equal")
    ax.set_xticks(range(K), [f"{e:g}" for e in EPS_GRID])
    ax.set_yticks(range(K), [f"{e:g}" for e in EPS_GRID])
    ax.set_xlabel(r"$\varepsilon_2$")
    ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(title)
    ax.grid(False)
    for i, j in ne:
        ax.plot(j, i, marker="s", markerfacecolor="none",
                markeredgecolor=PRIMARY_ORANGE, markeredgewidth=2.0,
                markersize=14)
    return im


def main():
    apply_paper_style()
    A_pld, A_meas, V = _synthetic_data(seed=42)
    U1_pld = V - A_pld[..., 0]
    U2_pld = V - A_pld[..., 1]
    U1_meas = V - A_meas[..., 0]
    U2_meas = V - A_meas[..., 1]
    ne_pld = _find_ne(U1_pld, U2_pld)
    ne_meas = _find_ne(U1_meas, U2_meas)

    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.32))
    im0 = _heatmap(axes[0], U1_pld,
                   r"Analytical $u_1$ (PLD)", ne_pld)
    im1 = _heatmap(axes[1], U1_meas,
                   r"Measured $u_1$ (LiRA)", ne_meas)
    fig.colorbar(im1, ax=axes[:2].tolist(), fraction=0.03, pad=0.02, shrink=0.85)

    # Panel C: A_measured vs A_analytical scatter, both silos.
    ax = axes[2]
    ax.plot([0, 0.5], [0, 0.5], ls="--", color="gray", lw=1)
    ax.scatter(A_pld[..., 0].ravel(), A_meas[..., 0].ravel(),
               s=18, color=PRIMARY_BLUE, alpha=0.75, label="silo 1")
    ax.scatter(A_pld[..., 1].ravel(), A_meas[..., 1].ravel(),
               s=18, color=PRIMARY_ORANGE, alpha=0.75, label="silo 2")
    ax.set_xlabel(r"$A^{\mathrm{PLD}}$")
    ax.set_ylabel(r"$A^{\mathrm{LiRA}}$")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_aspect("equal")
    ax.set_title("Privacy-onion gap")
    ax.legend(loc="upper left")

    fig.suptitle("H1 preview (mock LiRA scores) -- fig_H1_n2_measured_NE", y=1.02)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_H1_n2_preview.pdf")
    fig.savefig(OUT / "fig_H1_n2_preview.svg")
    plt.close(fig)
    print(f"Preview written to {OUT / 'fig_H1_n2_preview.pdf'}")
    print(f"Mock NE (analytical): {ne_pld}")
    print(f"Mock NE (measured):   {ne_meas}")


if __name__ == "__main__":
    main()
