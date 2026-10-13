"""The user's construction, taken seriously with tolerance: analytic
BRs constrained to the exact feasible band, NE where the bands
overlap. Strict argmax overlap is grid-fragile; the exact game's wall
strings are near-ties, so the fair test is regret-based:

  For every cell of the band compute the hybrid regret
  r(i,j) = max_k feasible U_own(k) - U_own(i,j) for both players.
  The hybrid near-NE set at tolerance tau is {r_max <= tau}.

Questions: (a) what regret do the deployed exact NEs carry under the
hybrid payoffs (per wing: tight player vs loose player)? (b) at what
tolerance do the wall segments join the hybrid near-NE set, and is
that tolerance small (construction works) or payoff-scale (it
doesn't)?
"""
import pickle
import sys
from pathlib import Path

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
sw = sweep_from_grid(mech, LAM, LAM)
feas, Ud = sw["feasible"], sw["U"]

MU = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                      1e-6, 60.0) for e in acts])
GC = 2 * norm.cdf(MU / 2) - 1


def F(s):
    L = np.log1p(s * s / (A_CONST * T))
    return L / (1.0 + L)


FS = F(MU[:, None] + MU[None, :])
U1 = FS - LAM * GC[:, None]
U2 = FS - LAM * GC[None, :]

# hybrid regrets on the band
NEG = -np.inf
c1 = np.where(feas, U1, NEG).max(axis=0)   # best U1 per column j
c2 = np.where(feas, U2, NEG).max(axis=1)   # best U2 per row i
R1 = np.where(feas, c1[None, :] - U1, np.nan)
R2 = np.where(feas, c2[:, None] - U2, np.nan)
Rmax = np.fmax(R1, R2)

# deployed exact NEs
m1 = np.where(feas, Ud[:, :, 0], NEG).max(axis=0)
m2 = np.where(feas, Ud[:, :, 1], NEG).max(axis=1)
nes = [(i, j) for i in range(K) for j in range(K)
       if feas[i, j] and Ud[i, j, 0] >= m1[j] - 1e-12
       and Ud[i, j, 1] >= m2[i] - 1e-12]

print(f"deployed exact NEs: {len(nes)}")
print(f"{'eps1':>8} {'eps2':>8} {'r_loose':>10} {'r_tight':>10} {'wing':>10}")
rows = []
for i, j in nes:
    e1, e2 = acts[i], acts[j]
    if e1 > e2:      # player 1 loose (rides high), player 2 tight
        rl, rt, wing = R1[i, j], R2[i, j], "lower"
    elif e2 > e1:
        rl, rt, wing = R2[i, j], R1[i, j], "upper"
    else:
        rl, rt, wing = R1[i, j], R2[i, j], "diag"
    rows.append((e1, e2, rl, rt, wing))
for e1, e2, rl, rt, wing in rows[:6] + rows[-6:]:
    print(f"{e1:8.3f} {e2:8.3f} {rl:10.2e} {rt:10.2e} {wing:>10}")

rls = np.array([r[2] for r in rows if r[4] != "diag"])
rts = np.array([r[3] for r in rows if r[4] != "diag"])
print(f"\nloose-player hybrid regret across the strings: "
      f"median {np.median(rls):.3e}, max {rls.max():.3e}")
print(f"tight-player hybrid regret across the strings: "
      f"median {np.median(rts):.3e}, max {rts.max():.3e}")

# how much of the deployed NE set enters the hybrid near-NE set, by tau
for tau in (1e-4, 5e-4, 2e-3, 5e-3, 2e-2):
    inset = sum(1 for i, j in nes if Rmax[i, j] <= tau)
    cells = np.sum(Rmax <= tau)
    print(f"tau = {tau:7.0e}: hybrid near-NE cells {cells:5d}, "
          f"deployed NEs captured {inset}/{len(nes)}")

# payoff scale for context
print(f"\npayoff scale: U1 spans [{np.nanmin(np.where(feas, U1, np.nan)):.3f}, "
      f"{np.nanmax(np.where(feas, U1, np.nan)):.3f}]")
