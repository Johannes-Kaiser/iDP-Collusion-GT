"""Best-response curves (flipbook style: red BR2, blue BR1 mirror, gold
stars / green segments for equilibria) for the four interior-BR
settings: T=1 |G|=5000 lam=28; T=1 |G|=500 lam=3 and lam=4; T=2
|G|=5000 lam=12.6. Interior branches drawn thicker to stand out.

Inputs: 38_t1_lam_sweep.pkl (lam=28), 47_t1_lam_G500.pkl (lam=3,4, job
180141), 48_t2_br_lam126.pkl. Equilibria: exact floor pairs where the
tight side clips to the action floor; T=2 frontier segment from the
mutual-consistency scan (sample resolution).
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / "tasks_findings" / "_scripts"

d38 = pickle.load(open(SC / "38_t1_lam_sweep.pkl", "rb"))
d47 = pickle.load(open(SC / "47_t1_lam_G500.pkl", "rb"))
r48 = pickle.load(open(SC / "48_t2_br_lam126.pkl", "rb"))


def rows_of(d, lam):
    return d["cols_e1"], d["sweep"][lam]


PANELS = []
cols, rows = rows_of(d38, 28)
PANELS.append(dict(cols=np.asarray(cols), rows=rows,
                   title="$T=1$, $|G|=5000$, $\\lambda=28$",
                   stars=[(0.5, 9.768436)], seg=None))
for lam in (3, 4):
    cols, rows = rows_of(d47, lam)
    y = rows[0]["br"]          # interior BR against the floor opponent
    PANELS.append(dict(cols=np.asarray(cols), rows=rows,
                       title=f"$T=1$, $|G|=500$, $\\lambda={lam}$",
                       stars=[(0.5, y)], seg=None))
PANELS.append(dict(cols=np.array([r["e1"] for r in r48]), rows=r48,
                   title="$T=2$, $|G|=5000$, $\\lambda=12.6$",
                   stars=None, seg=(0.5315, 0.8530)))

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.5), sharey=True)
for ax, P in zip(axes, PANELS):
    cols = P["cols"]
    br = np.array([np.clip(r["br"], 0.5, 64.0) for r in P["rows"]])
    kind = [r["kind"] for r in P["rows"]]
    start = 0
    for k in range(1, len(cols) + 1):
        if k == len(cols) or kind[k] != kind[start]:
            lw = 3.0 if kind[start] == "interior" else 1.6
            ax.plot(cols[start:k], br[start:k], color="crimson", lw=lw)
            ax.plot(br[start:k], cols[start:k], color="royalblue",
                    lw=lw, ls="--")
            start = k
    ax.plot([0.5, 64], [0.5, 64], color="0.6", lw=0.8, zorder=0)
    if P["seg"] is not None:
        a, b = P["seg"]
        m = (cols >= a * 0.98) & (cols <= b * 1.02)
        e2hi = np.array([r["e2_hi"] for r in P["rows"]])
        xs, ys = cols[m], np.clip(e2hi[m], 0.5, 64.0)
        ax.plot(xs, ys, color="tab:green", lw=5, alpha=0.75, zorder=4)
        ax.plot(ys, xs, color="tab:green", lw=5, alpha=0.75, zorder=4)
        ax.annotate("NE segment (frontier,\nnot the interior branch)",
                    (ys[-1], xs[-1]), textcoords="offset points",
                    xytext=(10, -16), fontsize=8.5, color="darkgreen")
    if P["stars"]:
        for (x, y) in P["stars"]:
            ax.plot([x, y], [y, x], "*", color="goldenrod", ms=17,
                    mec="0.3", mew=0.5, zorder=6)
        x, y = P["stars"][0]
        ax.annotate(f"NE $({x}, {y:.2f})$ + mirror", (x, y),
                    textcoords="offset points", xytext=(12, 6),
                    fontsize=8.5, color="0.25")
    # mark the interior stretch of BR2 explicitly
    ii = [i for i, kk in enumerate(kind) if kk == "interior"]
    if ii:
        ax.annotate("interior branch", (cols[ii[len(ii)//2]],
                    br[ii[len(ii)//2]]), textcoords="offset points",
                    xytext=(4, -18), ha="left", fontsize=8.5,
                    color="crimson")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.45, 70), ax.set_ylim(0.45, 70)
    ax.set_xlabel("$\\varepsilon_1$")
    ax.set_title(P["title"], fontsize=10)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("$\\varepsilon_2$")
axes[0].plot([], [], color="crimson", lw=1.6, label="BR$_2(\\varepsilon_1)$")
axes[0].plot([], [], color="royalblue", lw=1.6, ls="--",
             label="BR$_1(\\varepsilon_2)$")
axes[0].plot([], [], color="crimson", lw=3.0, label="interior stretch")
axes[0].plot([], [], "*", color="goldenrod", ms=12, mec="0.3",
             label="pure NE")
axes[0].legend(loc="center right", fontsize=8)
fig.suptitle("Best-response curves in the interior-BR settings: the interior branch appears, "
             "but equilibria stay constraint-pinned (floor pairs / frontier segment)", y=1.02)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/interior_br_curves.png",
            dpi=160, bbox_inches="tight")
print("wrote interior_br_curves.png")
