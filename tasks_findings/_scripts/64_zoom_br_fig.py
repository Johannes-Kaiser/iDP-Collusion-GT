"""Best-response curves of both players on the 16x zoom grid
(lambda = 2, T = 390, K = 220 on [0.90, 1.20]^2): the plateau seen at
full resolution, with the near-tie pure NE pair and the orbit values
marked. Companion to the 20.2 execution log."""
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

mech = pickle.load(open(ROOT / "tasks_findings/_scripts/55_cycle_zoom.pkl", "rb"))
acts = np.asarray(mech["actions"])
sweep = sweep_from_grid(mech, 2.0, 2.0)
BR1 = best_response_curve(sweep, 0)
BR2 = best_response_curve(sweep, 1)

fig, (ax, axz) = plt.subplots(1, 2, figsize=(12.6, 5.6))
for a in (ax, axz):
    ok2 = BR2 >= 0
    a.plot(acts[ok2], acts[BR2[ok2]], color="crimson", lw=1.4,
           label="BR$_2(\\varepsilon_1)$")
    ok1 = BR1 >= 0
    a.plot(acts[BR1[ok1]], acts[ok1], color="royalblue", lw=1.4, ls="--",
           label="BR$_1(\\varepsilon_2)$")
    a.plot([acts[0], acts[-1]], [acts[0], acts[-1]], color="0.75", lw=0.7)
    for (x, y) in [(1.0090, 1.0116), (1.0116, 1.0090)]:
        a.plot(x, y, "*", color="goldenrod", ms=15, mec="0.2", mew=0.6,
               zorder=6)
    a.set_xlabel("$\\varepsilon_1$")
    a.grid(alpha=0.25)
# orbit values as faint guides in the zoom panel
for v in (1.0063, 1.0183, 1.0495, 1.0578):
    axz.axvline(v, color="0.6", lw=0.5, ls=":")
    axz.axhline(v, color="0.6", lw=0.5, ls=":")
ax.set_ylabel("$\\varepsilon_2$")
ax.set_xlim(acts[0], acts[-1]); ax.set_ylim(acts[0], acts[-1])
ax.legend(loc="upper left", fontsize=9)
ax.set_title("full zoom window ($16\\times$ the paper grid's resolution)",
             fontsize=10)
axz.set_xlim(0.99, 1.075); axz.set_ylim(0.99, 1.075)
axz.set_title("crossing region: stars = the near-tie pure NE pair\n"
              "(margins $\\leq 5{\\times}10^{-6}$, below the accountant "
              "sawtooth); dotted = orbit values", fontsize=9.5)
fig.suptitle("Best responses at $\\lambda=2$ on the fine grid: the curves interleave "
             "through a near-flat plateau — they graze, barely cross, and play orbits",
             fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/zoom_br_lam2.png",
            dpi=160, bbox_inches="tight")
print("wrote zoom_br_lam2.png")
