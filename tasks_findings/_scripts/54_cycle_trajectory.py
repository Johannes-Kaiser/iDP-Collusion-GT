"""Best-response dynamics at lambda = 2 (T = 390, K = 200 cache):
staircase trajectories in (eps1, eps2) space funneling into the
3-cycle, with both best-response curves and the mixed-NE support
overlaid. Alternating dynamics (the campaign's Sec. 11 setting):
player 2 replies to eps1, then player 1 replies to eps2, ...
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import best_response_curve, sweep_from_grid

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
sweep = sweep_from_grid(mech, 2.0, 2.0)
acts = np.asarray(mech["actions"])
BR1 = best_response_curve(sweep, 0)   # BR1[j]: player 1's reply index to j
BR2 = best_response_curve(sweep, 1)   # BR2[i]: player 2's reply index to i


def trajectory(i0: int, j0: int, n: int = 40):
    """Alternating BR dynamics from (i0, j0); returns index path."""
    path = [(i0, j0)]
    i, j = i0, j0
    for k in range(n):
        j = int(BR2[i]); path.append((i, j))
        i = int(BR1[j]); path.append((i, j))
    return path


STARTS = [(0.6, 40.0), (40.0, 0.6), (8.0, 8.0), (0.55, 0.55),
          (30.0, 30.0), (2.0, 20.0), (55.0, 3.0), (1.0, 1.0)]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.95, len(STARTS)))

fig, (ax, axz) = plt.subplots(1, 2, figsize=(13.2, 5.8))
for a in (ax, axz):
    # BR curves
    ok2 = BR2 >= 0
    a.plot(acts[ok2], acts[BR2[ok2]], color="crimson", lw=1.6,
           label="BR$_2(\\varepsilon_1)$")
    ok1 = BR1 >= 0
    a.plot(acts[BR1[ok1]], acts[ok1], color="royalblue", lw=1.6, ls="--",
           label="BR$_1(\\varepsilon_2)$")
    a.plot([0.4, 70], [0.4, 70], color="0.75", lw=0.7, zorder=0)
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel("$\\varepsilon_1$")
    a.grid(alpha=0.25, which="both")

cycle_pts = set()
for (e10, e20), c in zip(STARTS, COLORS):
    i0 = int(np.argmin(np.abs(acts - e10)))
    j0 = int(np.argmin(np.abs(acts - e20)))
    path = trajectory(i0, j0)
    xs = acts[[p[0] for p in path]]
    ys = acts[[p[1] for p in path]]
    ax.plot(xs, ys, color=c, lw=1.1, alpha=0.8)
    ax.plot(xs[0], ys[0], "o", color=c, ms=6, mec="0.2", mew=0.5)
    for k in range(0, min(8, len(xs) - 1)):
        ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.0,
                                    alpha=0.85, shrinkA=0, shrinkB=0),
                    annotation_clip=True)
    axz.plot(xs[-14:], ys[-14:], color=c, lw=1.2, alpha=0.8)
    cycle_pts.update(path[-6:])

# the attractor
cyc = sorted(cycle_pts)
print("attractor states (eps1, eps2):",
      [(round(acts[i], 4), round(acts[j], 4)) for i, j in cyc])
for i, j in cyc:
    for a in (ax, axz):
        a.plot(acts[i], acts[j], "s", color="gold", ms=10, mec="0.2",
               mew=0.8, zorder=6)
# mixed NE support (both axes)
for v in (1.0140, 1.0647):
    axz.axvline(v, color="0.5", lw=0.6, ls=":")
    axz.axhline(v, color="0.5", lw=0.6, ls=":")

ax.set_xlim(0.4, 70), ax.set_ylim(0.4, 70)
ax.set_ylabel("$\\varepsilon_2$")
ax.set_title("Best-response dynamics, $\\lambda=2$ (T=390 cache): "
             "every start funnels into one loop", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
axz.set_xlim(0.93, 1.16), axz.set_ylim(0.93, 1.16)
axz.set_title("The attractor: the undercutting 3-cycle "
              "(gold squares; dotted = mixed-NE support)", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/cycle_trajectory.png",
            dpi=160, bbox_inches="tight")
print("wrote cycle_trajectory.png")
