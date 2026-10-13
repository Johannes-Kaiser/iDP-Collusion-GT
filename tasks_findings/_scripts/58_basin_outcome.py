"""Outcome map of alternating best-response dynamics (K=200 cache,
T=390, lambda=1): basins collapse to a function of the first mover's
budget alone, so represent them in 1-D and show the *content* of each
outcome — where both players settle, which role the first mover gets,
and how long it takes — instead of a 2-D identity map.

Panels: (a) outcome map: first mover's resting budget (solid) and
responder's (open) vs the first mover's commitment; cycles drawn as
shaded min-max bands; class strip underneath. (b) rounds to attractor.
(c) compact phase portrait with one sample staircase per class.
lambda = 2 is one line of text (single global cycle) — noted on (a').
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import best_response_curve, sweep_from_grid

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])
K = len(acts)

CLASS_COLORS = {0: "#b5651d", 1: "#4878cf", 2: "#d62728", 3: "#9a9a9a"}
CLASS_NAMES = ["diagonal 3-cycle", "dominant wall pair", "symmetric NE",
               "other frontier-segment point"]


def analyze(lam):
    sweep = sweep_from_grid(mech, lam, lam)
    feas = sweep["feasible"]
    BR1 = best_response_curve(sweep, 0)
    BR2 = best_response_curve(sweep, 1)

    def step(i, j):
        j2 = int(BR2[i])
        return int(BR1[j2]), j2

    rows = []
    for i0 in range(K):
        if BR2[i0] < 0 or not feas[i0].any():
            rows.append(None)
            continue
        j0 = int(BR2[i0])
        seen, i, j = {}, i0, j0
        for t in range(200):
            if (i, j) in seen:
                cyc, (ii, jj) = [], (i, j)
                while True:
                    cyc.append((ii, jj))
                    ii, jj = step(ii, jj)
                    if (ii, jj) == (i, j):
                        break
                rows.append(dict(t=seen[(i, j)] + 1, cyc=cyc))
                break
            seen[(i, j)] = t
            i, j = step(i, j)
    return rows, BR1, BR2


def classify(cyc):
    if len(cyc) > 1:
        return 0
    (i, j), = cyc
    if i == j:
        return 2
    pair = tuple(sorted((round(acts[i], 2), round(acts[j], 2))))
    return 1 if pair == (1.61, 10.79) else 3


rows, BR1, BR2 = analyze(1.0)

fig = plt.figure(figsize=(13.6, 8.6))
gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 0.55, 1.2],
                      width_ratios=[2.1, 1.0], hspace=0.09, wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
axs = fig.add_subplot(gs[1, 0], sharex=ax)
axt = fig.add_subplot(gs[2, 0], sharex=ax)
axp = fig.add_subplot(gs[:, 1])

for i0, r in enumerate(rows):
    if r is None:
        continue
    c = classify(r["cyc"])
    col = CLASS_COLORS[c]
    x = acts[i0]
    if len(r["cyc"]) == 1:
        (i, j), = r["cyc"]
        ax.plot(x, acts[i], ".", color=col, ms=5)               # first mover
        ax.plot(x, acts[j], ".", color=col, ms=5, mfc="white",
                mew=1.1)                                        # responder
    else:
        vals = [acts[a] for s in r["cyc"] for a in s]
        ax.plot([x, x], [min(vals), max(vals)], color=col, lw=2.2,
                alpha=0.55, solid_capstyle="butt")
    axs.axvspan(acts[i0] / 1.0122, acts[i0] * 1.0122, color=col)
    axt.plot(x, r["t"], ".", color=col, ms=4)

ax.plot([0.45, 70], [0.45, 70], color="0.8", lw=0.7, zorder=0)
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_ylabel("resting budgets")
ax.set_xlim(0.45, 70)
ax.tick_params(labelbottom=False)
handles = [
    plt.Line2D([], [], color="0.2", marker=".", ls="", ms=9,
               label="first mover settles at (filled)"),
    plt.Line2D([], [], color="0.2", marker=".", ls="", ms=9, mfc="white",
               label="responder settles at (open)"),
    Patch(color=CLASS_COLORS[0], alpha=0.6,
          label="cycle: band of visited budgets"),
] + [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(4)]
ax.legend(handles=handles, loc="upper left", fontsize=7.6, ncol=2,
          framealpha=0.95)
ax.set_title("$\\lambda=1$: outcome vs the first mover's commitment "
             "(the responder's start is irrelevant)", fontsize=10)

axs.set_yticks([])
axs.set_ylabel("class", rotation=0, ha="right", va="center", fontsize=8)
axs.tick_params(labelbottom=False)

axt.set_ylabel("rounds")
axt.set_xlabel("first mover's committed budget $\\varepsilon_1$")
axt.set_ylim(0, 7)
axt.grid(alpha=0.25)

# phase portrait with one sample per class
axp.plot([0.45, 70], [0.45, 70], color="0.8", lw=0.7, zorder=0)
ok2 = BR2 >= 0
axp.plot(acts[ok2], acts[BR2[ok2]], color="crimson", lw=1.2, alpha=0.7)
ok1 = BR1 >= 0
axp.plot(acts[BR1[ok1]], acts[ok1], color="royalblue", lw=1.2, ls="--",
         alpha=0.7)
starts = {}
for i0, r in enumerate(rows):
    if r is None:
        continue
    starts.setdefault(classify(r["cyc"]), []).append(i0)
sweep = sweep_from_grid(mech, 1.0, 1.0)
BR1c, BR2c = BR1, BR2
for c, lst in sorted(starts.items()):
    i0 = lst[len(lst) // 3]
    j0 = K - 10
    path = [(i0, j0)]
    i, j = i0, j0
    for _ in range(14):
        j = int(BR2c[i]); path.append((i, j))
        i = int(BR1c[j]); path.append((i, j))
    xs = acts[[p[0] for p in path]]
    ys = acts[[p[1] for p in path]]
    axp.plot(xs, ys, color=CLASS_COLORS[c], lw=1.5, alpha=0.95)
    axp.plot(xs[0], ys[0], "o", color="white", mec=CLASS_COLORS[c], ms=7)
    axp.plot(xs[-1], ys[-1], "*", color=CLASS_COLORS[c], mec="0.2", ms=13)
    for k in range(min(len(xs) - 1, 8)):
        axp.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                     arrowprops=dict(arrowstyle="->", lw=1.0,
                                     color=CLASS_COLORS[c], alpha=0.9,
                                     shrinkA=0, shrinkB=0),
                     annotation_clip=True)
axp.set_xscale("log"), axp.set_yscale("log")
axp.set_xlim(0.45, 70), axp.set_ylim(0.45, 70)
axp.set_xlabel("$\\varepsilon_1$")
axp.set_ylabel("$\\varepsilon_2$")
axp.set_title("sample staircases, one per class\n(thin: BR curves)",
              fontsize=9.5)

fig.suptitle("Where alternating best-response play settles, as a function of the only thing that matters\n"
             "($\\lambda=2$ needs no map: a single global attractor, the undercutting cycle $\\{1.01, 1.04, 1.06\\}$, from every start)",
             y=0.995, fontsize=11)
fig.savefig(ROOT / "paper/figures/data/foc_landscape/basins.png",
            dpi=150, bbox_inches="tight")
print("wrote basins.png (outcome-map redesign)")
