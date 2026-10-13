"""The arc from the analytic solution to the numeric Renyi-DP game
(nummeric_eval.md Sec. 27): the potential game's near-flat ridge,
computed in closed form, should BE the deployed game's near-Nash
band — the region where the RDP-accounted instance shows its NE
segments, its plateau, and its orbits.

Both panels show REGRET: the most any single player can gain by
deviating. Left: the limit game (closed forms, lam = 1). Right: the
deployed game (K=200 RDP cache, w = 1). Overlay: the analytic
near-NE band (regret <= 2e-3) drawn on the deployed map.

Model correspondence (project description, CLAUDE.md terms): players
= groups of |G_i| = 5000 records; mechanism = Poisson-subsampled
Gaussian, shared sigma, expected batch B = 128 (sum G_i p_i = B);
contracts (eps_i, delta = 1e-12); utility u_i = V(sigma) -
lam_i * A_i with V = 1/(1+sigma^2), A_i the record-level MIA
advantage of the T-fold composition; T = 390. The two panels differ
ONLY in accounting: exact limit profile vs 89-order RDP.
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
import pickle

import game_experiments.gdp_limit_game as g
from game.continuous import sweep_from_grid

T, LAM = 390.0, 1.0
A_CONST = g.PSUM ** 2

# ---- analytic regret map (limit game, closed forms) ----------------------
ES = np.geomspace(0.5, 64.0, 90)
MU = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                      1e-6, 60.0) for e in ES])
GC = 2 * norm.cdf(MU / 2) - 1


def F(s):
    L = np.log1p(s * s / (A_CONST * T))
    return L / (1.0 + L)


# u2 over the full (mu1, own mu) grid; own best response by dense scan
MU_D = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                        1e-6, 60.0) for e in np.geomspace(0.5, 64.0, 400)])
GC_D = 2 * norm.cdf(MU_D / 2) - 1

R_an = np.zeros((len(ES), len(ES)))
for i, m1 in enumerate(MU):
    # player 2's payoff along own choices given mu1
    best2 = np.max(F(m1 + MU_D) - LAM * GC_D)
    for j, m2 in enumerate(MU):
        u2 = F(m1 + m2) - LAM * GC[j]
        best1 = np.max(F(MU_D + m2) - LAM * GC_D)
        u1 = F(m1 + m2) - LAM * GC[i]
        R_an[i, j] = max(best1 - u1, best2 - u2)

# ---- deployed regret map (K=200 RDP cache) --------------------------------
mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])
sweep = sweep_from_grid(mech, LAM, LAM)
U, feas = sweep["U"], sweep["feasible"]
K = len(acts)
R_dep = np.full((K, K), np.nan)
colmax = np.where(feas, U[:, :, 0], -np.inf).max(axis=0)
rowmax = np.where(feas, U[:, :, 1], -np.inf).max(axis=1)
for i in range(K):
    for j in range(K):
        if feas[i, j]:
            R_dep[i, j] = max(colmax[j] - U[i, j, 0], rowmax[i] - U[i, j, 1])

# ---- stats: does the analytic band predict the deployed near-NE set? -----
from scipy.interpolate import RegularGridInterpolator
itp = RegularGridInterpolator((np.log(ES), np.log(ES)), R_an,
                              bounds_error=False, fill_value=np.nan)
pts = np.array([[math.log(a), math.log(b)] for a in acts for b in acts])
R_an_on_dep = itp(pts).reshape(K, K)
band = R_an_on_dep <= 2e-3
ok = np.isfinite(R_dep)
in_band = band & ok
print(f"analytic band (regret<=2e-3) covers {in_band.sum()} deployed cells "
      f"({100*in_band.sum()/ok.sum():.1f}% of feasible)")
print(f"deployed regret on the band: median {np.nanmedian(R_dep[in_band]):.4f}, "
      f"90th pct {np.nanpercentile(R_dep[in_band], 90):.4f}")
print(f"deployed regret off the band: median {np.nanmedian(R_dep[ok & ~band]):.4f}")
nes = [(i, j) for i in range(K) for j in range(K)
       if ok[i, j] and R_dep[i, j] < 1e-9]
frac = np.mean([band[i, j] for i, j in nes]) if nes else float("nan")
print(f"deployed exact NE cells: {len(nes)}; fraction inside analytic band: "
      f"{100*frac:.0f}%")

# ---- figure ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))
lev = np.array([0, 5e-4, 2e-3, 8e-3, 3e-2, 0.1, 0.3])
for ax, R, xs, title in ((ax1, R_an, ES, "analytic game (closed forms, exact profile)"),
                         (ax2, R_dep, acts, "deployed game (Renyi-DP accountant, K=200)")):
    im = ax.pcolormesh(xs, xs, np.clip(R, 1e-6, None).T,
                       norm=matplotlib.colors.LogNorm(vmin=3e-4, vmax=0.3),
                       cmap="magma_r", shading="auto")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("$\\varepsilon_1$")
    ax.set_title(title + "\nregret: what one player can gain by deviating",
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)
# overlay the analytic band contour on BOTH
X, Y = np.meshgrid(ES, ES, indexing="ij")
ax1.contour(ES, ES, R_an.T, levels=[2e-3], colors="cyan", linewidths=1.6)
ax2.contour(acts, acts, np.where(np.isfinite(R_an_on_dep), R_an_on_dep, 1).T,
            levels=[2e-3], colors="cyan", linewidths=1.6)
for i, j in nes:
    ax2.plot(acts[i], acts[j], ".", color="lime", ms=4)
ax1.set_ylabel("$\\varepsilon_2$")
ax2.plot([], [], ".", color="lime", ms=8, label="exact NE of the deployed game")
ax2.plot([], [], color="cyan", lw=1.6, label="analytic near-NE band (regret $\\leq 2{\\times}10^{-3}$)")
ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.suptitle("Two different arcs: the analytic ridge is a trade-off curve ($\\mu_1{+}\\mu_2$ const, "
             "falling), the deployed NE strings ride the participation walls (rising) "
             f"($T={T:g}$, $\\lambda={LAM:g}$)", fontsize=11.5)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/ridge_arc.png",
            dpi=150, bbox_inches="tight")
print("wrote ridge_arc.png")
