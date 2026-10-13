"""Slide figures not available as paper figures, drawn from existing caches.

Style follows game/style.py (the paper figures fig_C5_surface / fig_C7_lambda_ratio).
No mechanism solves: every number is read from a cache.

    .venv/bin/python slides/make_figs.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from game.style import PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY, apply_paper_style

OUT = Path(__file__).resolve().parent / "assets"
CACHE = ROOT / "game_experiments" / "_cache_v2"
SCRIPTS = ROOT / "tasks_findings" / "_scripts"
apply_paper_style()
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ frontier NE
mech = pickle.load(open(CACHE / "mech_grid_K200_g5000x5000_b128_e5.pkl", "rb"))
c6 = pickle.load(open(CACHE / "C6_K200.pkl", "rb"))
acts, A, feas = c6["actions"], c6["sweep"]["A"], mech["feasible"]
nes = c6["nes"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.6))
E1, E2 = np.meshgrid(acts, acts, indexing="ij")
ax.contourf(E2, E1, (~feas).astype(float), levels=[0.5, 1.5], colors=["#e3e3e3"])
ax.plot([acts[0], acts[-1]], [acts[0], acts[-1]], "--", color="grey", lw=0.9)
off = [(i, j) for i, j in nes if i != j]
sym = [(i, j) for i, j in nes if i == j]
ax.scatter([acts[j] for i, j in off], [acts[i] for i, j in off], s=16,
           color=PRIMARY_ORANGE, edgecolor="white", lw=0.4, zorder=3,
           label=f"asymmetric NE ({len(off)})")
ax.scatter([acts[j] for i, j in sym], [acts[i] for i, j in sym], marker="o", s=70,
           facecolor="none", edgecolor=PRIMARY_BLUE, lw=1.4, zorder=4,
           label="diagonal grid point (near-tie, not robust)")
ax.set(xscale="log", yscale="log", xlabel=r"$\varepsilon_2$", ylabel=r"$\varepsilon_1$",
       xlim=(acts[0], acts[-1]), ylim=(acts[0], acts[-1]))
ax.text(0.62, 0.9, "infeasible:\nopponent below\nparticipation floor", transform=ax.transAxes,
        fontsize=8, color=ACCENT_GREY, va="top")
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
ax.set_title("Equilibria sit on the participation frontier")
pairs = sorted((acts[i], acts[j], A[i, j, 0], A[i, j, 1]) for i, j in off if i < j)
tight = np.array([p[0] for p in pairs])
bx.plot(tight, [p[3] for p in pairs], "o-", color=PRIMARY_ORANGE, ms=4,
        label=r"loose player $A_{\mathrm{PLD}}$")
bx.plot(tight, [p[2] for p in pairs], "o-", color=PRIMARY_BLUE, ms=4,
        label=r"tight player $A_{\mathrm{PLD}}$")
bx.set(yscale="log", xlabel=r"tight player's budget $\varepsilon_{\mathrm{tight}}$",
       ylabel=r"realised MIA advantage")
bx.set_title("Realised risk along the equilibrium segment")
bx.legend(loc="center right")
fig.tight_layout()
save(fig, "fig_frontier_ne")

# ------------------------------------------------------------------ T=390 moderation
d = pickle.load(open(SCRIPTS / "52b_t390_fixedpoint.pkl", "rb"))
d2 = pickle.load(open(SCRIPTS / "52c_t390_saddle.pkl", "rb"))
cols = d["columns"][390] + d2["columns"][390]
cols.sort(key=lambda c: c["e1"])
fig, ax = plt.subplots(figsize=(5.4, 3.6))
pal = [plt.cm.Oranges(x) for x in np.linspace(0.4, 0.95, len(cols))]
for c, col in zip(cols, pal):
    rows = [r for r in c["rows"] if r["ok"] and 0.5 <= r["e2"] <= 64]
    e2 = np.array([r["e2"] for r in rows])
    u = np.array([1 / (1 + r["sigma"] ** 2) - r["A2"] for r in rows])
    k = [n for n, r in enumerate(rows) if r.get("diag")][0]
    lw = 2.6 if abs(c["e1"] - 2.548) < 1e-3 else 1.4
    ax.plot(e2, u - u[k], "o-", color=col, ms=3.5, lw=lw,
            label=rf"$\varepsilon_1 = {c['e1']:.3f}$")
    ax.plot(e2[k], 0, "o", color=col, ms=8, mec=PRIMARY_BLUE, mew=1.2)
ax.axhline(0, color="grey", lw=0.8)
ax.set(xscale="log", xlabel=r"player 2's deviation budget $\varepsilon_2$",
       ylabel=r"$u_2(\varepsilon_1,\varepsilon_2) - u_2(\varepsilon_1,\varepsilon_1)$",
       ylim=(-0.06, 0.01))
ax.set_title(r"Exact accounting, $T=390$, $\lambda=1$")
ax.legend(loc="lower left", ncol=2)
save(fig, "fig_t390_moderation")

# ------------------------------------------------------------------ welfare
r = [1, 2, 4, 8]
planner = [0.9406, 0.9397, 0.9380, 0.9346]
equal = [0.8853, 0.6743, 0.3596, -0.0577]
best = [0.8585, 0.8577, 0.8561, 0.8530]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
x = np.arange(len(r))
w = 0.26
ax.bar(x - w, planner, w, color=PRIMARY_BLUE, label="planner optimum (asymmetric)")
ax.bar(x, best, w, color=PRIMARY_ORANGE, label="best Nash equilibrium")
ax.bar(x + w, equal, w, color="#c9c9c9", label="enforced equal budgets")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, [f"{v}" for v in r])
ax.set(xlabel=r"heterogeneity $r = \lambda_2/\lambda_1$", ylabel=r"welfare $u_1+u_2$")
ax.set_ylim(-0.12, 1.22)
ax.legend(loc="upper center", ncol=2, fontsize=8.5)
save(fig, "fig_welfare")

# ------------------------------------------------------------------ E1 populations
eq, pl = [], []
for f in sorted(glob.glob(str(ROOT / "game_experiments/_cache_flpop/E1_pop*.json"))):
    j = json.load(open(f))
    eq.append(sum(j["eq"]["u"]))
    pl.append(sum(j["planner"]["u"]))
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.scatter(pl, eq, s=16, color=PRIMARY_ORANGE, edgecolor="white", lw=0.4)
lo, hi = min(eq + pl) * 0.95, max(eq + pl) * 1.03
ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.9, label="equilibrium = planner")
ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=r"planner welfare $\sum_i u_i$",
       ylabel=r"equilibrium welfare $\sum_i u_i$")
ax.legend(loc="upper left")
ax.set_title(f"n = 20 consortia: {len(eq)}/{len(eq)} equilibria at the floor")
save(fig, "fig_e1_population")

# ------------------------------------------------------------------ sensitivity-based control
sc = pickle.load(open(ROOT / "game_experiments/_cache_scale/C1_K200.pkl", "rb"))
sa = sc["actions"]
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.plot([acts[0], acts[-1]], [acts[0], acts[-1]], "--", color="grey", lw=0.9)
ax.scatter([acts[j] for i, j in nes], [acts[i] for i, j in nes], s=14, marker="s",
           color=PRIMARY_BLUE, label=f"sampling-based iDP ({len(nes)} NE)")
ax.scatter([sa[j] for i, j in sc["nes"]], [sa[i] for i, j in sc["nes"]], s=260, marker="*",
           color=PRIMARY_ORANGE, edgecolor="white", label="sensitivity-based iDP (1 NE)")
ax.set(xscale="log", yscale="log", xlabel=r"$\varepsilon_2$", ylabel=r"$\varepsilon_1$",
       xlim=(acts[0], acts[-1]), ylim=(acts[0], acts[-1]))
ax.legend(loc="upper left")
ax.set_title(r"Same game, two iDP realisations ($K=200$, $\lambda=1$)")
save(fig, "fig_scale_ne")

# ------------------------------------------------------------------ V-shape ablation (post-fix)
shapes = [
    ("V1", r"$1/(1+\sigma^2)$", lambda s: 1 / (1 + s ** 2)),
    ("V2", r"$1/(1+\sigma)$", lambda s: 1 / (1 + s)),
    ("V3", r"$e^{-\sigma^2/2}$", lambda s: np.exp(-s ** 2 / 2)),
    ("V4", r"$e^{-\sigma}$", lambda s: np.exp(-s)),
    ("V5", r"$-\log(1+\sigma)$", lambda s: -np.log1p(s)),
    ("V6", r"$-\sigma$", lambda s: -s),
]
SIG, AM = mech["sigma"], mech["A"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot([acts[0], acts[-1]], [acts[0], acts[-1]], "-", color=ACCENT_GREY, lw=0.7, alpha=0.7,
        label=r"$\varepsilon_1=\varepsilon_2$")
pal = [plt.cm.viridis(x) for x in np.linspace(0.05, 0.85, len(shapes))]
pal[0] = PRIMARY_BLUE
mk = ["o", "s", "D", "^", "v", "P"]
rng = np.random.default_rng(0)
for k, (tag, lab, V) in enumerate(shapes):
    v = V(SIG)
    U1 = np.where(feas, v - AM[:, :, 0], -np.inf)
    U2 = np.where(feas, v - AM[:, :, 1], -np.inf)
    b1, b2 = np.argmax(U1, axis=0), np.argmax(U2, axis=1)
    ne = [(i, j) for i in range(len(acts)) for j in range(len(acts))
          if feas[i, j] and b1[j] == i and b2[i] == j]
    off_ = [(i, j) for i, j in ne if i != j]
    dg_ = [(i, j) for i, j in ne if i == j]
    pts = np.array([[acts[i], acts[j]] for i, j in off_]) * 10 ** rng.normal(0, 0.012, (len(off_), 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=26, marker=mk[k], facecolor=pal[k], edgecolor="white",
               lw=0.4, alpha=0.9, label=f"{tag}: {lab}", zorder=3 + k)
    if dg_:
        ax.scatter([acts[i] for i, j in dg_], [acts[j] for i, j in dg_], s=60, marker=mk[k],
                   facecolor="none", edgecolor=pal[k], lw=1.4, zorder=10)
ax.scatter([], [], s=60, marker="o", facecolor="none", edgecolor="grey", lw=1.4,
           label="hollow: diagonal grid point\n(near-tie, margin ≲ 1e-4)")
ax.set(xscale="log", yscale="log", xlim=(acts[0] * 0.9, acts[-1] * 1.1),
       ylim=(acts[0] * 0.9, acts[-1] * 1.1), xlabel=r"$\varepsilon_1$", ylabel=r"$\varepsilon_2$")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "fig_v_shape_postfix")

# ------------------------------------------------------------------ exact T=390 local slice
loc_path = SCRIPTS / "83_t390_local_slice.pkl"
if loc_path.exists():
    loc = pickle.load(open(loc_path, "rb"))
    coarse = {}
    for f in ("52_t390_ladder", "52b_t390_fixedpoint", "52c_t390_saddle"):
        for c in pickle.load(open(SCRIPTS / f"{f}.pkl", "rb"))["columns"][390]:
            coarse[round(c["e1"], 3)] = c["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for e1, col in zip(loc["e1s"], (PRIMARY_ORANGE, PRIMARY_BLUE)):
        rows = sorted([r for r in loc["rows"] if r["ok"] and abs(r["e1"] - e1) < 1e-9],
                      key=lambda r: r["p2"])
        d = [r for r in rows if r["diag"]][0]
        ud = 1 / (1 + d["sigma"] ** 2) - d["A2"]
        e2 = np.array([r["e2"] for r in rows])
        du = np.array([1 / (1 + r["sigma"] ** 2) - r["A2"] for r in rows]) - ud
        ax.plot(e2, du * 1e3, "o-", color=col, ms=4, label=rf"$\varepsilon_1 = {e1:.3f}$ (dense exact)")
        cr = [r for r in coarse.get(round(e1, 3), []) if r["ok"] and 0.5 * e1 < r["e2"] < 2 * e1]
        ax.plot([r["e2"] for r in cr], [(1 / (1 + r["sigma"] ** 2) - r["A2"] - ud) * 1e3 for r in cr],
                "s", ms=8, mfc="none", mec=col, mew=1.3)
        ax.plot(d["e2"], 0, "o", ms=10, mfc="white", mec=col, mew=2)
    ax.plot([], [], "s", ms=8, mfc="none", mec="grey", label="earlier 21-point samples")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xscale="log", xlabel=r"player 2's budget $\varepsilon_2$",
           ylabel=r"$u_2(\varepsilon_1,\varepsilon_2)-u_2(\varepsilon_1,\varepsilon_1)$  [$10^{-3}$]")
    ax.set_title(r"Exact accounting, $T=390$, $\lambda=1$")
    ticks = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig_t390_local_slice")
