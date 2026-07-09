"""
Generate a hero figure for T18 (H2 sensitivity-based iDP / Scale).

Produces `paper/figures/fig_h2_scale.pdf` / `.svg` using:
  - a real Scale-mode sweep on the K = 22 grid over eps in [0.5, 64];
  - the cached sampling-iDP C1 sweep at the same K = 22 grid
    (`game_experiments/_cache_v2/C1.pkl`, produced by
    `experiment_C1` in `game_experiments/run_continuous.py`).

Scale is much faster than sampling because under Scale
A_i = mia_adv(sigma_g_i, p_fixed, I) — the K^2 sweep reduces to K unique
advantage evaluations (one per epsilon value on the grid).
"""
from __future__ import annotations

import pickle, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.mechanism_scale import solve_mechanism_scale_cached, _sigma_for_eps
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    two_column_size,
)

apply_paper_style()

FIG_DIR = ROOT / "paper" / "figures"
CACHE_V2 = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0
GROUP_SIZES = (5000, 5000)
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)
K = 22            # match cached C1 grid
W = 1.0
P_FIXED = EXPECTED_BATCH / sum(GROUP_SIZES)

actions = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), K)


# ---- Scale sweep (fast: per-eps sigma cached; A only depends on eps_i) ----
print(f"[scale] pre-computing per-eps sigma and A for K={K}...", flush=True)
t0 = time.time()
sigma_of_eps = np.array([
    _sigma_for_eps(float(e), P_FIXED, STEPS, DELTA) for e in actions
])
A_of_eps = np.array([
    mia_advantage_subsampled_gaussian(float(s), P_FIXED, STEPS)
    for s in sigma_of_eps
])
print(f"[scale] per-eps done in {time.time()-t0:.1f}s "
      f"(sigma range [{sigma_of_eps.min():.3f},{sigma_of_eps.max():.3f}])", flush=True)

# sigma_scale on K x K grid = weighted harmonic mean of the two sigma values
inv_s = 0.5 * (1.0/sigma_of_eps[:, None] + 1.0/sigma_of_eps[None, :])
S_scale = 1.0 / inv_s
A_scale = np.zeros((K, K, 2))
A_scale[:, :, 0] = A_of_eps[:, None]                 # A_1 depends on eps_1 only
A_scale[:, :, 1] = A_of_eps[None, :]                 # A_2 depends on eps_2 only
feas_scale = np.ones((K, K), dtype=bool)

# Feasibility: c_g = c_baseline * sigma_scale/sigma_g; drop cells with extreme ratios.
ratio = sigma_of_eps[:, None] / sigma_of_eps[None, :]
extreme = (ratio > 1e3) | (ratio < 1e-3)
feas_scale &= ~extreme

U_scale = np.full((K, K, 2), np.nan)
for i in range(K):
    for j in range(K):
        if not feas_scale[i, j]:
            continue
        v = value_fn(float(S_scale[i, j]))
        U_scale[i, j, 0] = v - W * float(A_scale[i, j, 0])
        U_scale[i, j, 1] = v - W * float(A_scale[i, j, 1])


def find_nes(U, feas):
    K = U.shape[0]
    nes = []
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            col = U[:, j, 0].copy()
            col[~feas[:, j]] = -np.inf
            i_br = int(np.argmax(col))
            row = U[i, :, 1].copy()
            row[~feas[i, :]] = -np.inf
            j_br = int(np.argmax(row))
            if i_br == i and j_br == j:
                nes.append((i, j))
    return nes


nes_scale = find_nes(U_scale, feas_scale)
print(f"[scale] NEs: {len(nes_scale)} -- "
      f"eps = {[(round(actions[i],3), round(actions[j],3)) for (i,j) in nes_scale]}",
      flush=True)


# ---- Sampling-iDP C1 sweep: load from cache ----
print(f"[sampling] loading cached C1 (K={K}) from {CACHE_V2/'C1.pkl'}...", flush=True)
c1 = pickle.load(open(CACHE_V2 / "C1.pkl", "rb"))
c1_actions = c1["actions"]
assert len(c1_actions) == K
assert np.allclose(c1_actions, actions), \
    f"cached actions mismatch: {c1_actions} vs {actions}"
nes_samp = c1["nes"]
print(f"[sampling] NEs: {len(nes_samp)} -- "
      f"eps = {[(round(actions[i],3), round(actions[j],3)) for (i,j) in nes_samp]}",
      flush=True)


# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.32),
                          gridspec_kw=dict(wspace=0.45))
tick_pos = np.linspace(0, K-1, 6, dtype=int)
tick_lab = [f"{actions[i]:.1f}" for i in tick_pos]

# (a) sigma_scale heatmap
ax = axes[0]
Splot = np.where(feas_scale, S_scale, np.nan)
im = ax.imshow(Splot, origin="lower", aspect="auto", cmap="Purples",
                extent=[0, K-1, 0, K-1])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=0)
ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
ax.set_title(r"(a) Scale: $\sigma_\mathrm{scale}(\varepsilon_1,\varepsilon_2)$"
             "\n(shared parameter is still coupled)")

# (b) A_1 heatmap under Scale
ax = axes[1]
Aplot = np.where(feas_scale, A_scale[:, :, 0], np.nan)
im = ax.imshow(Aplot, origin="lower", aspect="auto", cmap="Oranges",
                extent=[0, K-1, 0, K-1])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=0)
ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
ax.set_title(r"(b) Scale: $A_1$"
             "\n(flat in $\\varepsilon_2$ — no externality on $A$)")

# (c) NE overlay
ax = axes[2]
ax.plot([EPS_MIN, EPS_MAX], [EPS_MIN, EPS_MAX], "--", color=ACCENT_GREY,
        lw=0.6, alpha=0.7, label="diagonal")
for k, (i, j) in enumerate(nes_samp):
    ax.loglog(actions[i], actions[j], "s", color=PRIMARY_BLUE,
              markersize=6, markeredgecolor="white", markeredgewidth=0.5,
              label="sampling-iDP NE" if k == 0 else None)
for k, (i, j) in enumerate(nes_scale):
    ax.loglog(actions[i], actions[j], "o", color=PRIMARY_ORANGE,
              markersize=6, markeredgecolor="white", markeredgewidth=0.5,
              label="Scale NE" if k == 0 else None)
ax.set_xlim(EPS_MIN * 0.8, EPS_MAX * 1.2)
ax.set_ylim(EPS_MIN * 0.8, EPS_MAX * 1.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
ax.set_title(r"(c) Pure NEs at $w_1{=}w_2{=}1$"
             "\n(Scale on diagonal, sampling off)")
ax.legend(frameon=False, fontsize=6, loc="lower right")

plt.tight_layout()
out_pdf = FIG_DIR / "fig_h2_scale.pdf"
out_svg = FIG_DIR / "fig_h2_scale.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
plt.close()
print(f"[figure] wrote {out_pdf}", flush=True)
print(f"[figure] wrote {out_svg}", flush=True)

# Diagnostic summary
diag_samp = sum(1 for (i, j) in nes_samp if i == j)
off_samp = len(nes_samp) - diag_samp
diag_scale = sum(1 for (i, j) in nes_scale if i == j)
off_scale = len(nes_scale) - diag_scale
print()
print("=== T18 / H2 diagnostic (K=22, w=1) ===", flush=True)
print(f"Sampling-iDP NEs: {len(nes_samp)} total, {diag_samp} on diagonal, {off_samp} off-diagonal", flush=True)
print(f"Scale NEs:        {len(nes_scale)} total, {diag_scale} on diagonal, {off_scale} off-diagonal", flush=True)
if off_scale == 0 and off_samp > 0:
    print("=> NEGATIVE outcome: abstract-condition claim narrows to sampling-based iDP", flush=True)
elif off_scale > 0:
    print("=> POSITIVE outcome: asymmetric-NE phenomenology reproduces under Scale", flush=True)
else:
    print("=> INCONCLUSIVE at this grid resolution", flush=True)
