"""Companion to Figure 11: the same lambda = 2 window seen at the
paper grid's resolution (K = 200 over the full box -> ~12 points in
the window), and an overlay of coarse vs fine best responses."""
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


def br_curves(mech):
    acts = np.asarray(mech["actions"])
    sweep = sweep_from_grid(mech, 2.0, 2.0)
    return acts, best_response_curve(sweep, 0), best_response_curve(sweep, 1)


coarse = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl", "rb"))
fine = pickle.load(open(ROOT / "tasks_findings/_scripts/55_cycle_zoom.pkl", "rb"))
ac, BR1c, BR2c = br_curves(coarse)
af, BR1f, BR2f = br_curves(fine)

W = (0.90, 1.20)
mc = (ac >= W[0]) & (ac <= W[1])

fig, (ax, axo) = plt.subplots(1, 2, figsize=(12.6, 5.6))

# left: coarse only, drawn as the steps it really is
ok = mc & (BR2c >= 0)
ax.step(ac[ok], ac[BR2c[ok]], where="mid", color="crimson", lw=1.8,
        label="BR$_2$ (paper grid)")
ok1 = (BR1c >= 0) & (ac[BR1c] >= W[0]) & (ac[BR1c] <= W[1])
ax.step(ac[BR1c[ok1 & mc]], ac[ok1 & mc], where="mid", color="royalblue",
        lw=1.8, ls="--", label="BR$_1$ (paper grid)")
for v in (1.014, 1.0391, 1.0647):
    ax.plot(v, v, "s", color="gold", mec="0.2", ms=0)  # anchor for grid pts
cyc = [(1.014, 1.0391), (1.0391, 1.0647), (1.0647, 1.014)]
for x, y in cyc:
    ax.plot(x, y, "s", color="gold", mec="0.2", ms=9, zorder=6)
ax.plot([], [], "s", color="gold", mec="0.2", ms=8, label="3-cycle states")
ax.plot(*zip(*(cyc + [cyc[0]])), color="olive", lw=1.0, alpha=0.6)
ax.set_title(f"paper grid: {mc.sum()} points in this window (2.5% cells)\n"
             "verdict then: no pure NE, a 3-cycle", fontsize=10)
ax.legend(loc="upper left", fontsize=9)

# right: overlay
okf = BR2f >= 0
axo.plot(af[okf], af[BR2f[okf]], color="crimson", lw=0.9, alpha=0.45)
okf1 = BR1f >= 0
axo.plot(af[BR1f[okf1]], af[okf1], color="royalblue", lw=0.9, ls="--",
         alpha=0.45)
axo.step(ac[ok], ac[BR2c[ok]], where="mid", color="crimson", lw=2.2)
axo.step(ac[BR1c[ok1 & mc]], ac[ok1 & mc], where="mid", color="royalblue",
         lw=2.2, ls="--")
for (x, y) in [(1.0090, 1.0116), (1.0116, 1.0090)]:
    axo.plot(x, y, "*", color="goldenrod", ms=15, mec="0.2", mew=0.6,
             zorder=6)
axo.set_title("overlay: bold steps = paper grid, faint = 16$\\times$ zoom\n"
              "stars = the near-tie NE pair the coarse grid cannot see",
              fontsize=10)

for a in (ax, axo):
    a.plot(W, W, color="0.75", lw=0.7, zorder=0)
    a.set_xlim(*W); a.set_ylim(*W)
    a.set_xlabel("$\\varepsilon_1$")
    a.grid(alpha=0.25)
ax.set_ylabel("$\\varepsilon_2$")
fig.suptitle("The same crossing region at both resolutions ($\\lambda=2$, T=390)",
             fontsize=11.5)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/zoom_br_lam2_compare.png",
            dpi=160, bbox_inches="tight")
print("wrote zoom_br_lam2_compare.png")
