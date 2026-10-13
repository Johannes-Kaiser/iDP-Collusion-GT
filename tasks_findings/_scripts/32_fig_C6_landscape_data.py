"""Data tables for the landscape version of fig:C6 (paper Figure 4).

From the K=200 mechanism cache, exports to paper/figures/data/:

  c6l_frontier.dat      analytic participation frontier (eps1_min, eps2)
  c6l_infeas_left.dat   fill polygon: eps1 < f(eps2)  (tight-side wall)
  c6l_infeas_bot.dat    fill polygon: eps2 < f(eps1)  (mirror)
  c6l_br1_interior.dat  BR1 interior branch (smoothed global argmax)
  c6l_br1_chi.dat       BR1 upper-corner branch: eps1 = f^{-1}(eps2)
  c6l_br1_clo.dat       BR1 lower-corner branch: eps1 = f(eps2)
  c6l_ne_segment.dat    frontier near-equilibrium segment, el in the
                        span of the 42 grid-NE loose coordinates
  c6l_ne_sym.dat        the symmetric equilibrium point

BR1 is the value-maximal branch per opponent column of the cached
payoff u1 = V(sigma) - A_1, differentiated numerically along the own
axis after Savitzky-Golay smoothing (window 11, order 2) to remove the
accountant's Renyi-order sawtooth. BR2 is the mirror (symmetric game;
D2 == D1^T on the cache to machine precision).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.utility import value_fn  # noqa: E402

CACHE = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"
DATA = ROOT / "paper" / "figures" / "data"

grid = pickle.load(open(CACHE, "rb"))
a = grid["actions"]
x = np.log(a)
dx = x[1] - x[0]
K = len(a)
feas = grid["feasible"].astype(bool)
U1 = np.where(feas, value_fn(grid["sigma"]) - grid["A"][:, :, 0], np.nan)
fr = grid["frontier_eps_min"]                 # eps1_min given eps2 = a[j]

EPS_SYM = 2.9646                              # symmetric NE (grid + refinement)
NE_SEG = (5.19414, 10.794)                    # span of grid-NE loose coords


def br_column(j):
    """(x_br, kind) of the value-maximal branch of u1(., a[j])."""
    col = U1[:, j]
    ii = np.where(np.isfinite(col))[0]
    if len(ii) < 9:
        return None
    s = slice(ii[0], ii[-1] + 1)
    if np.isnan(col[s]).any():
        return None
    w = min(11, (len(col[s]) // 2) * 2 - 1)
    if w < 5:
        return None
    sm = savgol_filter(col[s], w, 2)
    d = savgol_filter(col[s], w, 2, deriv=1, delta=dx)
    xs = x[s]
    cands = [(xs[0], sm[0], "clo"), (xs[-1], sm[-1], "chi")]
    for k in range(len(d) - 1):
        if d[k] > 0 >= d[k + 1]:
            xz = xs[k] + dx * d[k] / (d[k] - d[k + 1])
            cands.append((xz, np.interp(xz, xs, sm), "interior"))
    return max(cands, key=lambda c: c[1])


kinds = [br_column(j) for j in range(K)]
# regime switches (value ties near the switch produce one-column blips;
# take the outermost interior columns as the branch boundaries)
interior_js = [j for j, r in enumerate(kinds)
               if r is not None and r[2] == "interior"]
j_lo, j_hi = interior_js[0], interior_js[-1]
# drop blips inside corner regimes: interior only on a contiguous run
run = [j for j in interior_js if kinds[max(j - 1, 0)] is not None]
# contiguity: keep the longest contiguous block
blocks, cur = [], [interior_js[0]]
for j in interior_js[1:]:
    if j == cur[-1] + 1:
        cur.append(j)
    else:
        blocks.append(cur)
        cur = [j]
blocks.append(cur)
main = max(blocks, key=len)
print(f"interior regime: eps2 in [{a[main[0]]:.4f}, {a[main[-1]]:.4f}]")

hdr = "eps1 eps2"
np.savetxt(DATA / "c6l_frontier.dat",
           np.column_stack([fr, a]), header=hdr, comments="")

# fill polygons (corners appended; pgfplots clips to the axis box)
left = np.vstack([np.column_stack([fr, a]),
                  [[0.4, 70], [0.4, 0.4]]])
np.savetxt(DATA / "c6l_infeas_left.dat", left, header=hdr, comments="")
bot = np.vstack([np.column_stack([a, fr]),
                 [[70, 0.4], [0.4, 0.4]]])
np.savetxt(DATA / "c6l_infeas_bot.dat", bot, header=hdr, comments="")

interior = np.array([[np.exp(kinds[j][0]), a[j]] for j in main])
np.savetxt(DATA / "c6l_br1_interior.dat", interior, header=hdr, comments="")

# corner branches on the analytic frontier / its mirror
chi_j = np.arange(0, main[0])                 # eps2 below interior regime
clo_j = np.arange(main[-1] + 1, K)            # eps2 above interior regime
f_inv = lambda e2: np.exp(np.interp(np.log(e2), np.log(fr), np.log(a)))
chi = np.column_stack([f_inv(a[chi_j]), a[chi_j]])
np.savetxt(DATA / "c6l_br1_chi.dat", chi, header=hdr, comments="")
clo = np.column_stack([fr[clo_j], a[clo_j]])
np.savetxt(DATA / "c6l_br1_clo.dat", clo, header=hdr, comments="")

seg_el = np.geomspace(NE_SEG[0], NE_SEG[1], 60)
seg_et = np.exp(np.interp(np.log(seg_el), np.log(a), np.log(fr)))
np.savetxt(DATA / "c6l_ne_segment.dat",
           np.column_stack([seg_et, seg_el]), header=hdr, comments="")
np.savetxt(DATA / "c6l_ne_sym.dat",
           np.array([[EPS_SYM, EPS_SYM]]), header=hdr, comments="")

print(f"BR1 branches: corner-hi {len(chi)} cols, interior {len(interior)}, "
      f"corner-lo {len(clo)} cols")
print(f"NE segment el in [{NE_SEG[0]:.2f}, {NE_SEG[1]:.2f}], "
      f"et in [{seg_et[0]:.4f}, {seg_et[-1]:.4f}]")
print("wrote c6l_*.dat")
