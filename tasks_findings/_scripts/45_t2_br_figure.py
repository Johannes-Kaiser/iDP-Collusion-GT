"""Four-panel T=2 exact best responses and equilibria (lambda = 0.05,
0.7, 2, 28), from 42_two_step_br.pkl and the exact endpoints of the
frontier segments (polished tie roots)."""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = pickle.load(open(ROOT / "tasks_findings/_scripts/42_two_step_br.pkl", "rb"))
cols = np.asarray(d["cols"])

EQ = {
    0.05: dict(ne=(64.0, 64.0), seg=None,
               title="$\\lambda=0.05$: ceiling NE $(64,64)$"),
    0.7: dict(ne=None, seg=(16.663898, 29.841699),
              title="$\\lambda=0.7$: segment $\\varepsilon_1\\in[16.66,29.84]$ (ceiling-clipped)"),
    2.0: dict(ne=None, seg=(3.295886, 17.345243),
              title="$\\lambda=2$: segment $\\varepsilon_1\\in[3.30,17.35]$"),
    28.0: dict(ne=(0.5, 0.5), seg=None,
               title="$\\lambda=28$: floor NE $(0.5,0.5)$"),
}

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
for ax, lam in zip(axes, d["lams"]):
    rows = d["per_lam"][lam]
    br = np.array([np.clip(r["br"], 0.5, 64.0) for r in rows])
    kind = [r["kind"] for r in rows]
    e2hi = np.array([r["e2_hi"] for r in rows])
    start = 0
    for k in range(1, len(cols) + 1):
        if k == len(cols) or kind[k] != kind[start]:
            ax.plot(cols[start:k], br[start:k], color="tab:blue", lw=1.8)
            ax.plot(br[start:k], cols[start:k], color="tab:orange",
                    lw=1.8, ls="--")
            start = k
    ax.plot([0.5, 64], [0.5, 64], color="0.6", lw=0.8, zorder=0)
    q = EQ[lam]
    if q["seg"] is not None:
        a, b = q["seg"]
        m = (cols >= a * 0.999) & (cols <= b * 1.001)
        xs, ys = cols[m], np.clip(e2hi[m], 0.5, 64.0)
        xs = np.concatenate([[a], xs, [b]])
        ys = np.concatenate([[np.interp(a, cols, e2hi)], ys,
                             [min(np.interp(b, cols, e2hi), 64.0)]])
        ax.plot(xs, ys, color="tab:green", lw=4, alpha=0.75, zorder=4)
        ax.plot(ys, xs, color="tab:green", lw=4, alpha=0.75, zorder=4)
        for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
            ax.plot([x, y], [y, x], "o", color="darkgreen", ms=5, zorder=5)
    if q["ne"] is not None:
        ax.plot(*q["ne"], "*", color="darkgreen", ms=16, zorder=5)
    ax.set_title(q["title"], fontsize=10)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.45, 70), ax.set_ylim(0.45, 70)
    ax.set_xlabel("$\\varepsilon_1$")
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("$\\varepsilon_2$")
fig.suptitle("T=2 exact best responses and equilibria (certified composition quadrature, "
             "$|G|=5000$, $B=128$)", y=1.02)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/t2_br_ne.png",
            dpi=160, bbox_inches="tight")
print("wrote t2_br_ne.png")
