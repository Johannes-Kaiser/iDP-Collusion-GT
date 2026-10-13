"""Basins of attraction of alternating best-response dynamics on the
K=200 cache (T=390 paper instance), lambda in {1, 2}, with attractors
grouped into semantic classes (legend) and one representative
staircase trajectory overlaid per class.

Classes: diagonal 3-cycles (mirror pair merged), dominant wall pair
(1.61, 10.79) + mirror, symmetric NE (2.965, 2.965), other
frontier-segment fixed points, undercutting cycle (lambda = 2).
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import best_response_curve, sweep_from_grid

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])
K = len(acts)

CLASS_COLORS = ["#b5651d", "#4878cf", "#d62728", "#c9c9c9"]
CLASS_NAMES = {
    1.0: ["diagonal 3-cycles (mirror pair)",
          "dominant wall pair $(1.61, 10.79)$ + mirror",
          "symmetric NE $(2.96, 2.96)$",
          "other frontier-segment points"],
    2.0: ["undercutting 3-cycle $\\{1.01, 1.04, 1.06\\}$", "", "", ""],
}


def classify(states) -> int:
    """Map an attractor (tuple of index states) to a semantic class."""
    vals = [(acts[i], acts[j]) for i, j in states]
    if len(states) > 1:
        return 0                                   # cycle
    (e1, e2), = vals
    if abs(e1 - e2) < 1e-9:
        return 2                                   # symmetric fixed point
    pair = tuple(sorted((round(e1, 2), round(e2, 2))))
    if pair == (1.61, 10.79):
        return 1                                   # dominant wall pair
    return 3                                       # other segment point


fig, axes = plt.subplots(2, 2, figsize=(13.8, 11.8))
for row, lam in enumerate((1.0, 2.0)):
    sweep = sweep_from_grid(mech, lam, lam)
    feas = sweep["feasible"]
    BR1 = best_response_curve(sweep, 0)
    BR2 = best_response_curve(sweep, 1)

    def step(i, j):
        j2 = int(BR2[i])
        return int(BR1[j2]), j2

    def run(i0, j0, keep_path=False):
        seen, path = {}, [(i0, j0)]
        i, j = i0, j0
        for t in range(200):
            if (i, j) in seen:
                cyc, (ii, jj) = [], (i, j)
                while True:
                    cyc.append((ii, jj))
                    ii, jj = step(ii, jj)
                    if (ii, jj) == (i, j):
                        break
                return tuple(sorted(cyc)), seen[(i, j)], path
            seen[(i, j)] = t
            j2 = int(BR2[i])
            if keep_path:
                path.append((i, j2))
            i2 = int(BR1[j2])
            if keep_path:
                path.append((i2, j2))
            i, j = i2, j2
        return None, np.nan, path

    label = np.full((K, K), -1, dtype=int)
    t_hit = np.full((K, K), np.nan)
    for i0 in range(K):
        for j0 in range(K):
            if not feas[i0, j0] or BR2[i0] < 0:
                continue
            key, t, _ = run(i0, j0)
            if key is not None:
                label[i0, j0] = classify(key)
                t_hit[i0, j0] = t

    tot = (label >= 0).sum()
    shares = [(c, 100 * (label == c).sum() / tot) for c in range(4)]
    print(f"lambda={lam}: shares " +
          ", ".join(f"{CLASS_NAMES[1.0][c]}: {s:.1f}%" for c, s in shares
                    if s > 0))

    axb, axt = axes[row]
    cmap = ListedColormap(CLASS_COLORS)
    m = np.ma.masked_where(label < 0, label)
    axb.pcolormesh(acts, acts, m.T, cmap=cmap, vmin=-0.5, vmax=3.5,
                   shading="auto")
    # representative trajectory per present class
    reps = []
    for c, s in shares:
        if s <= 0:
            continue
        cells = np.argwhere(label == c)
        # pick a start away from its attractor: highest eps2 in the class
        i0, j0 = cells[np.argmax(cells[:, 1])]
        reps.append((c, int(i0), int(j0)))
    for c, i0, j0 in reps:
        _, _, path = run(i0, j0, keep_path=True)
        xs = acts[[p[0] for p in path]]
        ys = acts[[p[1] for p in path]]
        axb.plot(xs, ys, color="black", lw=1.3, alpha=0.85)
        axb.plot(xs[0], ys[0], "o", color="white", mec="black", ms=7,
                 zorder=6)
        axb.plot(xs[-1], ys[-1], "*", color="gold", mec="black", ms=13,
                 zorder=6)
        for k in range(min(len(xs) - 1, 10)):
            axb.annotate("", xy=(xs[k + 1], ys[k + 1]),
                         xytext=(xs[k], ys[k]),
                         arrowprops=dict(arrowstyle="->", color="black",
                                         lw=1.1, alpha=0.9,
                                         shrinkA=0, shrinkB=0),
                         annotation_clip=True)
    handles = [Patch(color=CLASS_COLORS[c],
                     label=CLASS_NAMES[1.0][c] if lam == 1.0
                     else CLASS_NAMES[2.0][0])
               for c, s in shares if s > 0]
    handles += [plt.Line2D([], [], color="black", lw=1.3, marker="o",
                           mfc="white", mec="black",
                           label="sample trajectory (start ○ → attractor ★)")]
    axb.legend(handles=handles, loc="upper left", fontsize=8,
               framealpha=0.95)
    axb.set_title(f"$\\lambda={lam:g}$: attractor basins by class "
                  f"(stripes: only the first mover's $\\varepsilon_1$ "
                  f"matters)", fontsize=10)
    tm = np.ma.masked_invalid(t_hit)
    im = axt.pcolormesh(acts, acts, tm.T, cmap="magma", shading="auto")
    plt.colorbar(im, ax=axt, label="rounds to attractor")
    axt.set_title(f"$\\lambda={lam:g}$: time to attractor", fontsize=10)
    for a in (axb, axt):
        a.set_xscale("log"), a.set_yscale("log")
        a.set_xlabel("$\\varepsilon_1$ (start)")
        a.set_ylabel("$\\varepsilon_2$ (start)")
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/basins.png",
            dpi=150, bbox_inches="tight")
print("wrote basins.png")
