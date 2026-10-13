"""Landscape heatmaps with the ridge-engineering fixes applied
(T = 390 geometry, limit game): baseline saddle vs progressive
pricing (gamma > gamma* = 1.036) vs raised floor vs both.
The weight is rebalanced per panel so the symmetric stationary point
stays at eps* = 2.548 (lam_gamma = F'(2m*) / (gamma g^{gamma-1} g')).
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

A_CONST = g.PSUM ** 2
T = 390.0


def mu_of(e):
    return brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA, 1e-6, 60.0)


def Fv(S):
    L = np.log1p(S * S / (A_CONST * T))
    return L / (1 + L)


def Fp(s):
    L = math.log1p(s * s / (A_CONST * T))
    return (2 * s / (A_CONST * T + s * s)) / (1 + L) ** 2


M_STAR = mu_of(2.548)
G_STAR = 2 * norm.cdf(M_STAR / 2) - 1

PANELS = [
    dict(gam=1.0, floor=0.5, title="baseline: $\\gamma=1$, floor $0.5$\n(the saddle world)"),
    dict(gam=1.05, floor=0.5, title="progressive pricing $\\gamma=1.05>\\gamma^*$\n(same floor)"),
    dict(gam=1.0, floor=2.0, title="raised floor $\\varepsilon_{min}=2$\n(linear pricing)"),
    dict(gam=1.05, floor=1.5, title="both: $\\gamma=1.05$, floor $1.5$"),
]

fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.9))
for ax, P in zip(axes, PANELS):
    gam, floor = P["gam"], P["floor"]
    lam = Fp(2 * M_STAR) / (gam * G_STAR ** (gam - 1) * norm.pdf(M_STAR / 2))
    ES = np.geomspace(floor, 64.0, 150)
    MU = np.array([mu_of(float(e)) for e in ES])
    GC = 2 * norm.cdf(MU / 2) - 1
    PHI = Fv(MU[:, None] + MU[None, :]) - lam * (
        np.power(GC, gam)[:, None] + np.power(GC, gam)[None, :])
    im = ax.pcolormesh(ES, ES, PHI.T, cmap="viridis", shading="auto")
    ax.contour(ES, ES, PHI.T, levels=14, colors="white", linewidths=0.4,
               alpha=0.6)
    i, j = np.unravel_index(np.argmax(PHI), PHI.shape)
    ax.plot(ES[i], ES[j], "*", color="red", ms=15, mec="white", mew=0.8,
            zorder=6)
    if i != j:
        ax.plot(ES[j], ES[i], "*", color="red", ms=15, mec="white",
                mew=0.8, zorder=6)
    ax.plot(2.548, 2.548, "o", color="orange", ms=9, mec="white", mew=0.9,
            zorder=6)
    sym_is_global = abs(ES[i] - ES[j]) < 1e-9 and abs(ES[i] - 2.548) < 0.2
    ax.set_title(P["title"] + ("\nmoderation IS the global peak"
                               if sym_is_global else ""), fontsize=9.5)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("$\\varepsilon_1$")
axes[0].set_ylabel("$\\varepsilon_2$")
axes[0].plot([], [], "*", color="red", ms=11, mec="0.3", label="global peak of $\\Phi$")
axes[0].plot([], [], "o", color="orange", ms=8, mec="0.3",
             label="moderation point (2.548, 2.548)")
axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("The landscape under the fixes ($T=390$; weight rebalanced per panel so the moderation point stays put)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/fixed_landscapes.png",
            dpi=150, bbox_inches="tight")
print("wrote fixed_landscapes.png")
for P in PANELS:
    gam, floor = P["gam"], P["floor"]
    lam = Fp(2 * M_STAR) / (gam * G_STAR ** (gam - 1) * norm.pdf(M_STAR / 2))
    ES = np.geomspace(floor, 64.0, 150)
    MU = np.array([mu_of(float(e)) for e in ES])
    GC = 2 * norm.cdf(MU / 2) - 1
    PHI = Fv(MU[:, None] + MU[None, :]) - lam * (
        np.power(GC, gam)[:, None] + np.power(GC, gam)[None, :])
    i, j = np.unravel_index(np.argmax(PHI), PHI.shape)
    k = int(np.argmin(np.abs(ES - 2.548)))
    print(f"gamma={gam} floor={floor}: global peak ({ES[i]:.3f},{ES[j]:.3f}), "
          f"Phi(peak)-Phi(sym) = {PHI[i,j]-PHI[k,k]:.5f}, lam_rebal={lam:.4f}")
