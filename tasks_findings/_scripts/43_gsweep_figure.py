"""Three-panel comparison figure for the T=1 group-size sweep at
lam = 0.7 (B = 128 fixed): |G| = 500 / 5000 / 50000. BR curves (box-
clipped), the equilibrium continuum segment on the participation
frontier (with exact value-tie endpoints from 43_gsweep_endpoints.py),
and the ceiling NE where the segment disappears."""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / "tasks_findings" / "_scripts"
OUT = ROOT / "paper" / "figures" / "data" / "foc_landscape"

d500 = pickle.load(open(SC / "40_t1_br_G500.pkl", "rb"))
d50k = pickle.load(open(SC / "41_t1_br_G50000.pkl", "rb"))
dlam = pickle.load(open(SC / "39_t1_lam_low.pkl", "rb"))

SETTINGS = [
    dict(G=500, psum=0.256, cols=d500["cols"],
         rows=d500["rows"],
         seg=[[0.5, 12.866238], [11.943736, 27.852421]], ne=None),
    dict(G=5000, psum=0.0256, cols=dlam["cols_e1"],
         rows=[dict(r, e2_hi=s["hi"])
               for r, s in zip(dlam["sweep"][0.7], dlam["samples"])],
         seg=[[21.968170, 35.816079], [35.816079, 51.399867]], ne=None),
    dict(G=50000, psum=0.00256, cols=d50k["cols"],
         rows=d50k["rows"], seg=None, ne=(64.0, 64.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for ax, S in zip(axes, SETTINGS):
    cols = np.asarray(S["cols"], float)
    br = np.array([np.clip(r["br"], 0.5, 64.0) for r in S["rows"]])
    kind = [r["kind"] for r in S["rows"]]
    # split BR2 into contiguous same-kind runs to avoid jump artifacts
    start = 0
    lbl_done = set()
    for k in range(1, len(cols) + 1):
        if k == len(cols) or kind[k] != kind[start]:
            lab = None
            if "BR$_2$" not in lbl_done:
                lab, _ = "BR$_2(\\varepsilon_1)$", lbl_done.add("BR$_2$")
            ax.plot(cols[start:k], br[start:k], color="tab:blue",
                    lw=1.8, label=lab)
            ax.plot(br[start:k], cols[start:k], color="tab:orange",
                    lw=1.8, ls="--",
                    label="BR$_1(\\varepsilon_2)$"
                    if "BR$_1$" not in lbl_done and not lbl_done.add("BR$_1$")
                    else None)
            start = k
    ax.plot([0.5, 64], [0.5, 64], color="0.6", lw=0.8, zorder=0)
    if S["seg"] is not None:
        (a, fa), (b, fb) = S["seg"]
        m = (cols >= a * 0.999) & (cols <= b * 1.001)
        e2hi = np.array([r["e2_hi"] if "e2_hi" in r else np.nan
                         for r in S["rows"]])
        if np.isfinite(e2hi[m]).all() and m.sum() > 3:
            xs, ys = cols[m], np.clip(e2hi[m], 0.5, 64)
        else:
            xs, ys = np.array([a, b]), np.array([fa, fb])
        ax.plot(ys, xs, color="tab:green", lw=4, alpha=0.75, zorder=4)
        ax.plot(xs, ys, color="tab:green", lw=4, alpha=0.75, zorder=4,
                label="NE segment")
        for (x, y) in ((a, fa), (b, fb)):
            ax.plot([x, y], [y, x], "o", color="darkgreen", ms=5, zorder=5)
        ax.set_title(f"$|G|={S['G']}$, $\\bar p_\\Sigma={S['psum']}$: "
                     f"NE segment $\\varepsilon_1\\in[{a:.2f},{b:.2f}]$",
                     fontsize=10)
    if S["ne"] is not None:
        ax.plot(*S["ne"], "*", color="darkgreen", ms=16, zorder=5,
                label="NE (ceiling)")
        ax.set_title(f"$|G|={S['G']}$, $\\bar p_\\Sigma={S['psum']}$: "
                     f"unique NE $(64, 64)$", fontsize=10)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.45, 70), ax.set_ylim(0.45, 70)
    ax.set_xlabel("$\\varepsilon_1$")
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("$\\varepsilon_2$")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("T=1 exact best responses and equilibria at $\\lambda=0.7$, "
             "batch $B=128$ fixed, group size varied", y=1.0)
fig.tight_layout()
for ext in ("png",):
    fig.savefig(OUT / f"t1_gsweep_lam07.{ext}", dpi=160,
                bbox_inches="tight")
print("wrote", OUT / "t1_gsweep_lam07.png")
