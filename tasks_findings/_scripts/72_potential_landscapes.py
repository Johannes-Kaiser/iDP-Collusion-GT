"""Potential landscapes of the limit game at increasing T
(nummeric_eval.md Sec. 26): Phi(eps1, eps2) = F_T(mu(eps1)+mu(eps2))
- lam [g(mu(eps1)) + g(mu(eps2))] on the action box, lam = 1.

Every best-response step climbs this surface (exact potential), so
the topography IS the dynamics: peaks = stable equilibria, ridges =
paths, and the evolution with T shows the story — ceiling peak at
small T, asymmetric peaks + symmetric structure at the paper's T,
floor peak at very large T.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import game_experiments.gdp_limit_game as g

LAM = 1.0
ES = np.geomspace(0.5, 64.0, 160)
MU = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                      1e-6, 60.0, xtol=1e-12) for e in ES])
GC = 2 * norm.cdf(MU / 2) - 1


def F(s, T):
    L = np.log1p((s / g.PSUM) ** 2 / T)
    return L / (1.0 + L)


TS = [10, 390, 6240, 100000]
fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.9), sharey=True)
for ax, T in zip(axes, TS):
    S = MU[:, None] + MU[None, :]
    PHI = F(S, T) - LAM * (GC[:, None] + GC[None, :])
    im = ax.pcolormesh(ES, ES, PHI.T, cmap="viridis", shading="auto")
    # contours to show the topography
    ax.contour(ES, ES, PHI.T, levels=14, colors="white", linewidths=0.4,
               alpha=0.6)
    # global max
    i, j = np.unravel_index(np.argmax(PHI), PHI.shape)
    ax.plot(ES[i], ES[j], "*", color="red", ms=16, mec="white", mew=0.8,
            zorder=6)
    if i != j:
        ax.plot(ES[j], ES[i], "*", color="red", ms=16, mec="white",
                mew=0.8, zorder=6)
    # symmetric stationary point (1-D along the diagonal of Phi)
    diag = np.array([F(2 * m, T) - 2 * LAM * gc for m, gc in zip(MU, GC)])
    k = int(np.argmax(diag))
    if 0 < k < len(ES) - 1:
        ax.plot(ES[k], ES[k], "o", color="orange", ms=10, mec="white",
                mew=0.8, zorder=6)
    ax.plot([0.5, 64], [0.5, 64], color="white", lw=0.5, alpha=0.5)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("$\\varepsilon_1$")
    ax.set_title(f"$T = {T:,}$", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.85)
axes[0].set_ylabel("$\\varepsilon_2$")
axes[0].plot([], [], "*", color="red", ms=12, mec="0.3", label="global peak of $\\Phi$")
axes[0].plot([], [], "o", color="orange", ms=9, mec="0.3",
             label="best symmetric point")
axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("The potential landscape $\\Phi$ at increasing composition depth ($\\lambda=1$; limit game — "
             "every best-response step climbs this surface)", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/potential_landscapes.png",
            dpi=150, bbox_inches="tight")
print("wrote potential_landscapes.png")
for T in TS:
    S = MU[:, None] + MU[None, :]
    PHI = F(S, T) - LAM * (GC[:, None] + GC[None, :])
    i, j = np.unravel_index(np.argmax(PHI), PHI.shape)
    diag = np.array([F(2 * m, T) - 2 * LAM * gc for m, gc in zip(MU, GC)])
    k = int(np.argmax(diag))
    print(f"T={T}: global peak ({ES[i]:.3f}, {ES[j]:.3f}), "
          f"best symmetric {ES[k]:.3f}, Phi gap {PHI[i,j]-diag[k]:.5f}")
