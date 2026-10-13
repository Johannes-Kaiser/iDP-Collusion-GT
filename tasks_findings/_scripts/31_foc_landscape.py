"""FOC landscape from the K=200 grid cache — no new solver calls.

Numerically differentiates u1 = V(sigma) - A_1 along the player's own
log-budget axis on the cached grid, extracts the zero set of the FOC
D1 = du1/dlog(eps1):

  * falling crossings (D1: + -> -)  = local-max ridge  (best-response branch)
  * rising crossings  (D1: - -> +)  = local-min valley (the trough)

Player 2's loci are the mirror (symmetric game, verified against
A[:,:,1]). Equilibrium candidates read off the landscape:

  * diagonal:  zero of D1(e,e) with own-curvature sign (peak vs trough)
  * interior:  intersections of the two max-ridges (h(t) = b1(b1(t)) - t)
  * frontier:  columns where D1 < 0 at the first feasible cell (corner
               argmax for the tight player) paired with the loose
               player's interior ridge.

Prints the zero points and writes a landscape figure
(paper/figures/data/foc_landscape.png + zero-loci .dat files for TikZ).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.utility import value_fn  # noqa: E402

CACHE = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"
OUTDIR = ROOT / "paper" / "figures" / "data" / "foc_landscape"
OUTDIR.mkdir(parents=True, exist_ok=True)

grid = pickle.load(open(CACHE, "rb"))
a = grid["actions"]
x = np.log(a)
dx = x[1] - x[0]                       # uniform in log
feas = grid["feasible"].astype(bool)
V = value_fn(grid["sigma"])
U1 = np.where(feas, V - grid["A"][:, :, 0], np.nan)   # [i=eps1, j=eps2]
U2 = np.where(feas, V - grid["A"][:, :, 1], np.nan)

# ---- own-axis derivative fields (central differences) ----------------
D1 = np.full_like(U1, np.nan)
D1[1:-1, :] = (U1[2:, :] - U1[:-2, :]) / (2 * dx)
D11 = np.full_like(U1, np.nan)
D11[1:-1, :] = (U1[2:, :] - 2 * U1[1:-1, :] + U1[:-2, :]) / dx**2
D2 = np.full_like(U2, np.nan)
D2[:, 1:-1] = (U2[:, 2:] - U2[:, :-2]) / (2 * dx)

# symmetry check: D2 should be the transpose-mirror of D1
m = np.isfinite(D1) & np.isfinite(D2.T)
print(f"symmetry check |D1 - D2^T|: max {np.nanmax(np.abs(D1 - D2.T)[m]):.2e} "
      f"over {m.sum()} cells")


def crossings(col: np.ndarray) -> tuple[list[float], list[float]]:
    """Interpolated zeros of a 1-D signal on x; (falling=maxima, rising)."""
    fall, rise = [], []
    for k in range(len(col) - 1):
        y0, y1 = col[k], col[k + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)) or y0 * y1 >= 0:
            continue
        xz = x[k] + dx * y0 / (y0 - y1)
        (fall if y0 > 0 else rise).append(float(xz))
    return fall, rise


# ---- BR1 loci: zeros of D1 along axis 0, per column j ----------------
max_pts, min_pts = [], []              # (x1, x2) in log coords
for j in range(len(a)):
    fall, rise = crossings(D1[:, j])
    max_pts += [(xz, x[j]) for xz in fall]
    min_pts += [(xz, x[j]) for xz in rise]
max_pts, min_pts = np.array(max_pts), np.array(min_pts)
print(f"BR1 ridge points: {len(max_pts)} maxima, {len(min_pts)} minima")

# ---- diagonal profile -------------------------------------------------
dvec = np.array([D1[i, i] for i in range(len(a))])
cvec = np.array([D11[i, i] for i in range(len(a))])
print("\ndiagonal zeros of D1(e,e) [own-FOC on the diagonal]:")
for k in range(len(a) - 1):
    y0, y1 = dvec[k], dvec[k + 1]
    if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 < 0:
        xz = x[k] + dx * y0 / (y0 - y1)
        cz = np.interp(xz, x[1:-1], cvec[1:-1])
        kind = "PEAK (symmetric NE candidate)" if cz < 0 else "trough"
        print(f"  eps = {np.exp(xz):8.4f}   d2u/dx2 = {cz:+.5f}   -> {kind}")

# ---- interior ridge intersections: x1 in R(x2) and x2 in R(x1) -------
# build per-column root lists for the max ridge
roots = {j: crossings(D1[:, j])[0] for j in range(len(a))}


def b_branches(j):
    return roots[j]


print("\ninterior NE candidates (ridge x ridge, off-diagonal):")
cands = []
for j in range(len(a)):
    for x1 in roots[j]:
        # player 2's ridge at eps1 = x1 (mirror: roots of column nearest x1)
        i = int(round((x1 - x[0]) / dx))
        if not (0 <= i < len(a)):
            continue
        for x2 in roots[i]:
            if abs(x2 - x[j]) < 0.6 * dx:          # x2 consistent with column j
                cands.append((x1, x[j]))
# dedupe to cell resolution
seen, uniq = set(), []
for x1, x2 in cands:
    key = (round(x1 / dx), round(x2 / dx))
    if key not in seen:
        seen.add(key)
        uniq.append((x1, x2))
for x1, x2 in sorted(uniq):
    tag = "  (diagonal)" if abs(x1 - x2) < dx else ""
    print(f"  (eps1, eps2) = ({np.exp(x1):8.4f}, {np.exp(x2):8.4f}){tag}")

# ---- frontier corners: tight player pinned at feasibility edge --------
print("\nfrontier-corner candidates (tight at edge, loose on ridge):")
corner = []
for j in range(len(a)):
    ii = np.where(feas[:, j])[0]
    if len(ii) == 0:
        continue
    i0 = ii[0]                                   # first feasible eps1
    # one-sided derivative into the interior
    if i0 + 2 < len(a) and feas[i0 + 1, j] and feas[i0 + 2, j]:
        d_edge = (-3 * U1[i0, j] + 4 * U1[i0 + 1, j] - U1[i0 + 2, j]) / (2 * dx)
        if d_edge < 0:                           # corner argmax for player 1
            # loose player's interior ridge at eps1 = a[i0]
            for x2 in crossings(D2[i0, :])[0]:
                if abs(x2 - x[j]) < 0.6 * dx:
                    corner.append((x[i0], x[j], d_edge))
for x1, x2, de in corner:
    print(f"  (eps_t, eps_l) = ({np.exp(x1):8.4f}, {np.exp(x2):8.4f})   "
          f"edge slope {de:+.4f}")

# ---- exports for TikZ / inspection ------------------------------------
np.savetxt(OUTDIR / "ridge1.dat", np.exp(max_pts), header="eps1 eps2")
np.savetxt(OUTDIR / "valley1.dat", np.exp(min_pts), header="eps1 eps2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6.5))
E1, E2 = np.meshgrid(a, a, indexing="ij")
ax.pcolormesh(E1, E2, np.where(np.isfinite(D1), np.sign(D1), np.nan),
              cmap="RdBu", vmin=-1.5, vmax=1.5, alpha=.35, shading="auto")
if len(max_pts):
    ax.plot(np.exp(max_pts[:, 0]), np.exp(max_pts[:, 1]), ".", ms=2.5,
            color="tab:blue", label=r"BR$_1$ ridge ($\partial_1 u_1 = 0$, max)")
    ax.plot(np.exp(max_pts[:, 1]), np.exp(max_pts[:, 0]), ".", ms=2.5,
            color="tab:red", label=r"BR$_2$ ridge (mirror)")
if len(min_pts):
    ax.plot(np.exp(min_pts[:, 0]), np.exp(min_pts[:, 1]), ".", ms=1.5,
            color="tab:blue", alpha=.35, label=r"valley (min)")
    ax.plot(np.exp(min_pts[:, 1]), np.exp(min_pts[:, 0]), ".", ms=1.5,
            color="tab:red", alpha=.35)
fr = grid.get("frontier_eps_min")
if fr is not None:
    ax.plot(fr, a, "k--", lw=.8, label="participation frontier")
    ax.plot(a, fr, "k--", lw=.8)
ax.plot(a, a, color="gray", lw=.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
ax.set_title("FOC landscape: sign of $\\partial u_1/\\partial\\log\\varepsilon_1$ "
             "and zero loci")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "foc_landscape.png", dpi=170)
print(f"\nwrote {OUTDIR}/foc_landscape.png and ridge/valley .dat files")
