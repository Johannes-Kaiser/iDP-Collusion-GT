"""T25 -- mock preview of the paper figure fig_C9_federated_toy.

Runs on the login node in <5 s. Renders the three-panel layout with
SYNTHETIC federated LiRA advantages generated from the PLD upper bound
with per-silo shrinkage (the "privacy-onion" gap in the companion paper
2601.12922v1 Section 5). This is a design preview only; the real figure
is produced from _cache_T25/T25_summary.pkl after Slurm completes.

Layout mirrors the T25 spec (findings doc Section 6):
  (a) Per-silo LiRA ROC on a representative feasible profile.
  (b) Measured A_g vs eps_g heatmap slice (fix eps_3 at mid, sweep i, j).
  (c) NE search on the 3x3x3 (eps_1, eps_2, eps_3) cube -- three 3x3
      face heatmaps, one per silo-3 action, with NE cells outlined.

Output: tasks_findings/_scripts/_out/fig_C9_federated_toy_preview.pdf (+ .svg).
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(REPO))

from game.style import (  # noqa: E402
    apply_paper_style,
    PRIMARY_BLUE,
    PRIMARY_ORANGE,
    two_column_size,
)

EPS_GRID = np.array([2.0, 8.0, 32.0])
K = len(EPS_GRID)
N_PLAYERS = 3
OUT = Path(__file__).with_suffix("").parent / "_out"


def _synthetic_data(seed: int = 42):
    """Build synthetic (A_pld, A_meas, V) tensors on the 3x3x3 grid."""
    rng = np.random.default_rng(seed)
    # A_g depends only on its own silo's eps -- this is the analytical form
    # (mia_advantage_subsampled_gaussian depends on (sigma, p_g)); the mock
    # preserves that structure. Cross-silo dependence enters only through V.
    A_pld = np.zeros((K, K, K, N_PLAYERS))
    IDX = list(enumerate(EPS_GRID))
    for (i, e1), (j, e2), (kk, e3) in product(IDX, IDX, IDX):
        A_pld[i, j, kk, 0] = 0.5 * (1 - np.exp(-e1 / 8))
        A_pld[i, j, kk, 1] = 0.5 * (1 - np.exp(-e2 / 8))
        A_pld[i, j, kk, 2] = 0.5 * (1 - np.exp(-e3 / 8))
    # Privacy-onion shrink is per-silo-per-own-eps only (not per joint cell),
    # matching the LiRA structure at large n_shadows where cross-silo cells
    # collapse to the same measured value at fixed own-eps.
    shrink_1 = 0.5 + 0.15 * rng.random(K)
    shrink_2 = 0.5 + 0.15 * rng.random(K)
    shrink_3 = 0.5 + 0.15 * rng.random(K)
    A_meas = np.zeros_like(A_pld)
    for (i, e1), (j, e2), (kk, e3) in product(IDX, IDX, IDX):
        A_meas[i, j, kk, 0] = A_pld[i, j, kk, 0] * shrink_1[i]
        A_meas[i, j, kk, 1] = A_pld[i, j, kk, 1] * shrink_2[j]
        A_meas[i, j, kk, 2] = A_pld[i, j, kk, 2] * shrink_3[kk]
    # V is roughly increasing in max(eps_g) (worse privacy -> more utility).
    V = np.zeros((K, K, K))
    for (i, e1), (j, e2), (kk, e3) in product(IDX, IDX, IDX):
        V[i, j, kk] = 0.55 + 0.15 * np.tanh(np.log2(max(e1, e2, e3)) / 3)
    return A_pld, A_meas, V


def _find_ne_cube(U: np.ndarray):
    ne = []
    for i, j, kk in product(range(K), range(K), range(K)):
        if (U[i, j, kk, 0] >= U[:, j, kk, 0].max() - 1e-9
                and U[i, j, kk, 1] >= U[i, :, kk, 1].max() - 1e-9
                and U[i, j, kk, 2] >= U[i, j, :, 2].max() - 1e-9):
            ne.append((i, j, kk))
    return ne


def _roc_from_advantage(A: float, n_pts: int = 200):
    """Toy per-silo ROC that hits advantage A at its Youden point.

    Uses a Gaussian-mean-shift IN vs OUT with delta = 2 * Phi^{-1}((1+A)/2).
    Not a fit to real data -- purely for the layout preview.
    """
    from scipy import stats
    d = 2 * stats.norm.ppf((1 + A) / 2)
    alpha = np.linspace(0, 1, n_pts)
    tpr = stats.norm.cdf(stats.norm.ppf(alpha) + d)
    return alpha, tpr


def main():
    apply_paper_style()
    A_pld, A_meas, V = _synthetic_data(seed=42)
    U_meas = np.stack([V - A_meas[..., g] for g in range(N_PLAYERS)], axis=-1)
    ne_meas = _find_ne_cube(U_meas)
    print(f"Mock measured NE cells (i,j,k): {ne_meas}")

    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.32))

    # Panel A: per-silo ROC on the mid profile (i=j=k=1 -> all eps=8).
    ax = axes[0]
    mid = (1, 1, 1)
    for g, colour in enumerate([PRIMARY_BLUE, PRIMARY_ORANGE, "#4c9c4c"]):
        A = float(A_meas[mid + (g,)])
        alpha, tpr = _roc_from_advantage(A)
        ax.plot(alpha, tpr, color=colour, lw=1.5,
                label=rf"silo {g+1} ($A={A:.3f}$)")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(r"Per-silo LiRA ROC, $\boldsymbol{\varepsilon}=(8,8,8)$")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Panel B: measured A_1 heatmap over (eps_1, eps_2) at fixed eps_3 = 8.
    ax = axes[1]
    slab = A_meas[:, :, 1, 0]  # A_1 at eps_3 index = 1 (=8).
    im = ax.imshow(slab, origin="lower", cmap="viridis", aspect="equal")
    ax.set_xticks(range(K), [f"{e:g}" for e in EPS_GRID])
    ax.set_yticks(range(K), [f"{e:g}" for e in EPS_GRID])
    ax.set_xlabel(r"$\varepsilon_2$")
    ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(r"Measured $A_1$ ($\varepsilon_3 = 8$)")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    # Panel C: NE on the 3x3x3 cube -- three 3x3 face-heatmaps of u_1 with
    # NE cells outlined in orange.
    ax = axes[2]
    face_utils = np.stack([U_meas[:, :, kk, 0] for kk in range(K)], axis=0)
    im = ax.imshow(np.hstack(list(face_utils)), origin="lower",
                   cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(K * K),
                  [f"{e:g}" for e in np.tile(EPS_GRID, K)], fontsize=6)
    ax.set_yticks(range(K), [f"{e:g}" for e in EPS_GRID])
    for kk in range(K):
        ax.axvline(kk * K - 0.5, color="white", lw=0.6)
        ax.text(kk * K + 1, K - 0.15, rf"$\varepsilon_3={EPS_GRID[kk]:g}$",
                ha="center", va="top", color="white", fontsize=7)
    for i, j, kk in ne_meas:
        ax.plot(kk * K + j, i, marker="s", markerfacecolor="none",
                markeredgecolor=PRIMARY_ORANGE, markeredgewidth=1.5,
                markersize=10)
    ax.set_xlabel(r"$\varepsilon_2$ (three panels: $\varepsilon_3 \in \{2,8,32\}$)")
    ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(r"NE on the $3^3$ cube ($u_1$ shown)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle("C9 preview (mock LiRA scores) -- fig_C9_federated_toy", y=1.02)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_C9_federated_toy_preview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_C9_federated_toy_preview.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Preview written to {OUT / 'fig_C9_federated_toy_preview.pdf'}")


if __name__ == "__main__":
    main()
