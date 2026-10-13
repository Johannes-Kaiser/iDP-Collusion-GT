"""Own-budget slice decompositions u2(eps2 | eps1) = V - lam*Adv, with
the components plotted separately, for four settings with interior best
responses (T=1 lam=28; G=500 lam=3; T=2 lam=12.6; G=500 lam=4), on a
linear budget axis extended past the participation frontier to the
action ceiling (where the declared budget is slack and the outcome is
frozen). Companion of the corner-solution decomposition
(t1_slice_decomposition.png) and the basis of nummeric_eval.md Sec. 15.

T=1 data is recomputed exactly per group size in a subprocess (module
constants bind PEG_G at import); the T=2 column comes from the cached
components of 42_two_step_br.pkl plus fresh rate solves.
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tasks_findings" / "_scripts" / "46_slice_data"
DATA.mkdir(exist_ok=True)

T1 = r'''
import os, sys, json
import numpy as np
sys.path.insert(0, os.environ["PEG_ROOT"])
import game_experiments.single_step_br as br
import game_experiments.single_step_game as gm
e1 = float(os.environ["E1"])
lo, hi = br.feasible_interval(e1)
xs = np.geomspace(max(lo*(1+1e-9), 1e-4), hi*(1-1e-9), 140)
out = []
for e2 in xs:
    r = gm.solve(e1, float(e2))
    if r is None: continue
    s, p1, p2 = r
    out.append((float(e2), gm.value(s), gm.advantage(s, p2), p2))
json.dump(dict(data=out, cap=gm.PSUM-gm.PBAR, e1=e1, G=gm.G1),
          open(os.environ["OUT"], "w"))
'''


def collect_t1(tag: str, G: int, e1: float) -> Path:
    out = DATA / f"{tag}.json"
    if not out.exists():
        env = dict(os.environ, PEG_G=str(G), E1=str(e1), OUT=str(out),
                   PEG_ROOT=str(ROOT))
        subprocess.run([sys.executable, "-c", T1], env=env, check=True)
    return out


def collect_t2(tag: str, e1_target: float) -> Path:
    out = DATA / f"{tag}.json"
    if out.exists():
        return out
    import game_experiments.two_step_game as g2
    d = pickle.load(open(ROOT / "tasks_findings/_scripts/42_two_step_br.pkl", "rb"))
    cols = np.asarray(d["cols"])
    i = int(np.argmin(np.abs(cols - e1_target)))
    s = d["samples"][i]
    xs = np.asarray(s["xs"])
    V, A = np.asarray(s["comp"]).T[:2]
    fin = np.isfinite(V)
    xs, V, A = xs[fin], V[fin], A[fin]
    sub = np.unique(np.linspace(0, len(xs) - 1, 45).astype(int))
    p2s = [g2.solve(float(cols[i]), float(xs[j]))[2] for j in sub]
    p2 = np.interp(np.arange(len(xs)), sub, p2s)
    json.dump(dict(data=[(float(a), float(b), float(c), float(p))
                         for a, b, c, p in zip(xs, V, A, p2)],
                   cap=g2.PSUM - g2.PBAR, e1=float(cols[i]), G=5000),
              open(out, "w"))
    return out


# exact interior-BR locations (Brent-polished; 38/47/48 pickles)
PANELS = [
    (collect_t1("t1_g5000_e05", 5000, 0.5), 28.0, 9.768436,
     "$T=1$, $|G|=5000$, $\\lambda=28$\n(opponent at $\\varepsilon_1=0.5$)"),
    (collect_t1("t1_g500_e05", 500, 0.5), 3.0, 11.904468,
     "$T=1$, $|G|=500$, $\\lambda=3$\n(opponent at $\\varepsilon_1=0.5$)"),
    (collect_t2("t2_g5000_e13", 1.3), 12.6, 11.594644,
     "$T=2$, $|G|=5000$, $\\lambda=12.6$\n(opponent at $\\varepsilon_1=1.29$)"),
    (collect_t1("t1_g500_e05", 500, 0.5), 4.0, 8.786123,
     "$T=1$, $|G|=500$, $\\lambda=4$: deep interior\n(opponent at $\\varepsilon_1=0.5$)"),
]

fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.7))
for ax, (path, lam, peak, title) in zip(axes, PANELS):
    d = json.load(open(path))
    xs, V, A, p2 = np.asarray(d["data"]).T
    cap = d["cap"]
    u = V - lam * A
    base = V[0]
    ax.plot(xs, V - base, color="tab:blue", lw=2,
            label="value gain $V-V_{\\rm lo}$")
    ax.plot(xs, lam * A, color="tab:red", lw=2,
            label="privacy cost $\\lambda\\,\\mathrm{Adv}$")
    ax.plot(xs, u - base, color="black", lw=2.4,
            label="utility $u_2-u_{\\rm lo}$")
    xe = np.array([xs[-1], 64.0])
    for y, c in ((V[-1] - base, "tab:blue"), (lam * A[-1], "tab:red"),
                 (u[-1] - base, "black")):
        ax.plot(xe, [y, y], color=c, lw=1.6, ls=(0, (2, 2)), alpha=0.7)
    ax.axvline(xs[-1], color="0.55", lw=1.0)
    ax.text(xs[-1] + 1.2, 0.42, "participation frontier", rotation=90,
            transform=ax.get_xaxis_transform(), ha="left", va="center",
            fontsize=8.5, color="0.35")
    ax.text(63, 1.012, "budget slack →",
            transform=ax.get_xaxis_transform(), ha="right",
            fontsize=8.5, color="0.35")
    pu = float(np.interp(peak, xs, u))
    ax.plot(peak, pu - base, "^", color="darkgreen", ms=11, zorder=6)
    ax.annotate(f"interior BR\n$\\varepsilon_2={peak:.2f}$",
                (peak, pu - base), textcoords="offset points",
                xytext=(-8, 16), ha="right", fontsize=9, color="darkgreen")
    ax2 = ax.twinx()
    ax2.plot(xs, p2 / cap, color="0.5", ls=":", lw=1.8)
    ax2.plot(xe, [p2[-1] / cap] * 2, color="0.5", ls=":", lw=1.2, alpha=0.6)
    ax2.set_ylim(-0.05, 1.6)
    if ax is axes[-1]:
        ax2.set_ylabel("own rate $p_2$ / batch cap (dotted)", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")
    ax.set_xlim(-1, 66)
    ax.set_xlabel("$\\varepsilon_2$")
    ax.set_title(title, fontsize=10, pad=16)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("change vs low corner")
axes[0].legend(loc="center right", fontsize=8.5, framealpha=0.95)
fig.suptitle(
    "Slice decompositions with interior best responses (linear budget axis, "
    "extended to the action ceiling).\nRight panel: one weight step later "
    "($\\lambda=3\\to4$) the peak sits at $\\varepsilon_2=8.79$, a third inside "
    "the frontier at 12.87; the floor corner takes over at $\\lambda=5$.",
    y=1.06)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/interior_slice_decompositions.png",
            dpi=160, bbox_inches="tight")
print("wrote interior_slice_decompositions.png")
