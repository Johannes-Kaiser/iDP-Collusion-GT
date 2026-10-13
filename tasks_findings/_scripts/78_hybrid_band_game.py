"""Hybrid band game (nummeric_eval.md Sec. 27 addendum): analytic
payoffs + the exact feasibility band, best-response play with wall
branches, NE where the BR bands overlap.

Construction: payoffs are the limit game's closed forms
u_i = F_T(mu(eps1)+mu(eps2)) - lam g(mu(eps_i)) (valid on the
interior); the feasibility band (participation walls) is taken from
the deployed K=200 cache, i.e. from the true accounting, because the
limit model misplaces the walls (Sec. 27 seam). Constrained BR: the
own-payoff argmax over the band; when the unconstrained optimum is
infeasible the reply rides the wall. Question: do the two players'
wall branches COINCIDE on an interval (the exact-game segment
mechanism), and does the resulting NE set align with the deployed
game's 42-NE string?
"""
import math
import pickle
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
from game.continuous import sweep_from_grid

T, LAM = 390.0, 1.0
A_CONST = g.PSUM ** 2

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])
K = len(acts)
feas = sweep_from_grid(mech, LAM, LAM)["feasible"]

MU = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                      1e-6, 60.0) for e in acts])
GC = 2 * norm.cdf(MU / 2) - 1


def F(s):
    L = np.log1p(s * s / (A_CONST * T))
    return L / (1.0 + L)


S = MU[:, None] + MU[None, :]
FS = F(S)
U1 = FS - LAM * GC[:, None]          # player 1's analytic payoff
U2 = FS - LAM * GC[None, :]

# constrained BR on the exact band
BR2 = np.full(K, -1, dtype=int)
BR1 = np.full(K, -1, dtype=int)
for i in range(K):
    js = np.where(feas[i, :])[0]
    if len(js):
        BR2[i] = js[np.argmax(U2[i, js])]
for j in range(K):
    is_ = np.where(feas[:, j])[0]
    if len(is_):
        BR1[j] = is_[np.argmax(U1[is_, j])]

# wall classification: is the BR at the band edge?
def edge_kind(i, j, axis):
    js = np.where(feas[i, :])[0] if axis == 2 else np.where(feas[:, j])[0]
    v = j if axis == 2 else i
    if v == js[0]:
        return "lo-wall"
    if v == js[-1]:
        return "hi-wall"
    return "interior"

kinds2 = [edge_kind(i, BR2[i], 2) if BR2[i] >= 0 else None for i in range(K)]
kinds1 = [edge_kind(BR1[j], j, 1) if BR1[j] >= 0 else None for j in range(K)]
from collections import Counter
print("BR2 branch census:", Counter(k for k in kinds2 if k))
print("BR1 branch census:", Counter(k for k in kinds1 if k))

# NE of the hybrid = mutual argmax
nes = [(i, j) for i in range(K) for j in range(K)
       if feas[i, j] and BR2[i] == j and BR1[j] == i]
print(f"hybrid NE: {len(nes)}")
if nes:
    on_wall = sum(1 for i, j in nes if edge_kind(i, j, 2) != "interior"
                  or edge_kind(i, j, 1) != "interior")
    print(f"  of which on a wall: {on_wall}")
    span = [(round(acts[i], 3), round(acts[j], 3)) for i, j in nes]
    print("  first/last:", span[0], span[-1])

# deployed exact NE for comparison
Ud = sweep_from_grid(mech, LAM, LAM)["U"]
R1 = np.where(feas, Ud[:, :, 0], -np.inf)
R2 = np.where(feas, Ud[:, :, 1], -np.inf)
nes_dep = [(i, j) for i in range(K) for j in range(K)
           if feas[i, j] and Ud[i, j, 0] >= R1[:, j].max() - 1e-12
           and Ud[i, j, 1] >= R2[i, :].max() - 1e-12]

fig, ax = plt.subplots(figsize=(7.8, 7.2))
ok2 = BR2 >= 0
ax.plot(acts[ok2], acts[BR2[ok2]], color="crimson", lw=1.5,
        label="analytic BR$_2$ on the exact band")
ok1 = BR1 >= 0
ax.plot(acts[BR1[ok1]], acts[ok1], color="royalblue", lw=1.5, ls="--",
        label="analytic BR$_1$ on the exact band")
# band edges
lo_edge = [acts[np.where(feas[i, :])[0][0]] if feas[i, :].any() else np.nan
           for i in range(K)]
hi_edge = [acts[np.where(feas[i, :])[0][-1]] if feas[i, :].any() else np.nan
           for i in range(K)]
ax.plot(acts, lo_edge, color="0.6", lw=0.8)
ax.plot(acts, hi_edge, color="0.6", lw=0.8, label="exact feasibility walls")
for i, j in nes:
    ax.plot(acts[i], acts[j], "*", color="goldenrod", ms=13, mec="0.25",
            mew=0.5, zorder=6)
for i, j in nes_dep:
    ax.plot(acts[i], acts[j], ".", color="lime", ms=5, zorder=5)
ax.plot([], [], "*", color="goldenrod", ms=11, mec="0.3",
        label=f"hybrid NE (n={len(nes)})")
ax.plot([], [], ".", color="lime", ms=8, label=f"deployed exact NE (n={len(nes_dep)})")
ax.plot([0.5, 64], [0.5, 64], color="0.85", lw=0.6)
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_xlim(0.45, 70), ax.set_ylim(0.45, 70)
ax.set_xlabel("$\\varepsilon_1$"), ax.set_ylabel("$\\varepsilon_2$")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("The hybrid band game: analytic payoffs + exact walls\n"
             "NE where the wall-riding BR branches coincide", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/hybrid_band_game.png",
            dpi=150, bbox_inches="tight")
print("wrote hybrid_band_game.png")
