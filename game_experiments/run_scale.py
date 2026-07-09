"""
T18 / H2 cross-mechanism check: rerun of the sampling-iDP experiments on
the sensitivity-based iDP realisation ("Scale") of Boenisch et al. (2023).

Utility: u_i(eps) = V(sigma_scale(eps)) - w_i * A_i^scale(eps)

Under Scale, A_i^scale = mia_adv(sigma_i, p=E_b/M, I) depends only on eps_i
(the abstract-condition assumption (iii) fails). The residual coupling
runs through the shared sigma_scale in the V term only. This module lays
out that computation in the same grid format as run_locus.py (so
sweep_from_grid / find_pure_NE_2player / best_response_curve work
unchanged), then generates the hero comparison figure
`paper/figures/fig_h2_scale.{pdf,svg}` at the requested K.

Cache layout mirrors _cache_v2:
  _cache_scale/mech_grid_K{K}_g{g1}x{g2}_b{B}_e{E}.pkl
  _cache_scale/C1_K{K}.pkl
  _cache_scale/C7_K{K}.pkl
  _cache_scale/C8_K{K}.pkl
"""
from __future__ import annotations

import os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))

from game.mechanism_scale import _sigma_for_eps
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn
from game.continuous import (
    sweep_from_grid, find_pure_NE_2player, best_response_curve,
)
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size, two_column_size, orange_palette, blue_palette,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_SCALE = ROOT / "game_experiments" / "_cache_scale"
CACHE_V2 = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_SCALE.mkdir(parents=True, exist_ok=True)

apply_paper_style()

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0
GROUP_SIZES = (5000, 5000)
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)
P_FIXED = EXPECTED_BATCH / sum(GROUP_SIZES)


# --------------------------------------------------------------------------
# Scale mech grid: A_i depends only on eps_i => K unique sigma-for-eps
# calls and K unique advantage evaluations; the K x K structure is analytic.
# --------------------------------------------------------------------------

def _sigma_and_A_of_eps(eps: float) -> tuple:
    s = _sigma_for_eps(eps, P_FIXED, STEPS, DELTA)
    a = mia_advantage_subsampled_gaussian(float(s), P_FIXED, STEPS)
    return s, a


def compute_scale_mech_grid(K: int, n_workers: int = 1) -> dict:
    """Return the Scale mechanism grid in the same schema as
    `precompute_mechanism_grid` from `game.continuous`:
        actions, sigma, A, feasible.
    """
    actions = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), K)
    t0 = time.time()
    if n_workers <= 1:
        pairs = [_sigma_and_A_of_eps(float(e)) for e in actions]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            pairs = list(pool.map(_sigma_and_A_of_eps, [float(e) for e in actions]))
    sigma_of_eps = np.array([p[0] for p in pairs])
    A_of_eps = np.array([p[1] for p in pairs])
    print(f"[scale-grid] K={K}: per-eps done in {time.time()-t0:.1f}s "
          f"(sigma range [{sigma_of_eps.min():.3f}, {sigma_of_eps.max():.3f}])",
          flush=True)

    # sigma_scale on K x K = (0.5 (1/s_i + 1/s_j))^{-1}  (equal group sizes)
    inv_s = 0.5 * (1.0 / sigma_of_eps[:, None] + 1.0 / sigma_of_eps[None, :])
    sigma = 1.0 / inv_s
    A = np.zeros((K, K, 2))
    A[:, :, 0] = A_of_eps[:, None]
    A[:, :, 1] = A_of_eps[None, :]

    # Feasibility: reject cells where c_g / c_baseline = sigma_scale/sigma_g
    # goes outside [1e-3, 1e3] (extreme clipping ratio).
    ratio = sigma_of_eps[:, None] / sigma_of_eps[None, :]  # sigma_g / sigma_scale-ish; use ratio between groups
    max_ratio = np.maximum(ratio, 1.0 / ratio)
    feasible = max_ratio <= 1e3

    return dict(
        actions=actions,
        sigma=sigma,
        A=A,
        feasible=feasible,
        sigma_of_eps=sigma_of_eps,
        A_of_eps=A_of_eps,
    )


def load_or_compute_scale_grid(K: int) -> dict:
    tag = (f"mech_grid_K{K}_g{GROUP_SIZES[0]}x{GROUP_SIZES[1]}"
           f"_b{EXPECTED_BATCH}_e{EPOCHS}")
    path = CACHE_SCALE / f"{tag}.pkl"
    if path.exists():
        print(f"[scale-grid] loading cached grid: {path.name}", flush=True)
        return pickle.load(open(path, "rb"))
    print(f"[scale-grid] computing K={K} with {_N_WORKERS} workers...", flush=True)
    grid = compute_scale_mech_grid(K, n_workers=_N_WORKERS)
    pickle.dump(grid, open(path, "wb"))
    return grid


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def experiment_C1_scale(K: int = 22) -> dict:
    print(f"[C1-scale] symmetric w=1, K={K}", flush=True)
    cache = CACHE_SCALE / f"C1_K{K}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    grid = load_or_compute_scale_grid(K)
    sweep = sweep_from_grid(grid, w1=1.0, w2=1.0)
    nes = find_pure_NE_2player(sweep)
    br1 = best_response_curve(sweep, player=0)
    br2 = best_response_curve(sweep, player=1)
    data = dict(sweep=sweep, nes=nes, br1=br1, br2=br2,
                actions=sweep["actions"], K=K,
                sigma_of_eps=grid["sigma_of_eps"],
                A_of_eps=grid["A_of_eps"])
    pickle.dump(data, open(cache, "wb"))
    print(f"  |NE|={len(nes)}", flush=True)
    return data


def experiment_C7_scale(K: int = 200) -> dict:
    print(f"[C7-scale] symmetric w sweep, K={K}", flush=True)
    cache = CACHE_SCALE / f"C7_K{K}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    grid = load_or_compute_scale_grid(K)
    ws = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for w in ws:
        sweep = sweep_from_grid(grid, w1=w, w2=w)
        nes = find_pure_NE_2player(sweep)
        rows.append({"w": w, "nes": nes,
                     "actions": sweep["actions"],
                     "U": sweep["U"], "A": sweep["A"],
                     "S": sweep["sigma"],
                     "feasible": sweep["feasible"]})
        print(f"  w={w}: |NE|={len(nes)}", flush=True)
    data = dict(rows=rows, K=K)
    pickle.dump(data, open(cache, "wb"))
    return data


def experiment_C8_scale(K: int = 200) -> dict:
    print(f"[C8-scale] asymmetric w_1/w_2 sweep, K={K}", flush=True)
    cache = CACHE_SCALE / f"C8_K{K}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    grid = load_or_compute_scale_grid(K)
    ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for r in ratios:
        sweep = sweep_from_grid(grid, w1=r, w2=1.0)
        nes = find_pure_NE_2player(sweep)
        rows.append({"ratio": r, "nes": nes,
                     "actions": sweep["actions"],
                     "U": sweep["U"], "A": sweep["A"],
                     "S": sweep["sigma"],
                     "feasible": sweep["feasible"]})
        print(f"  w_1/w_2={r}: |NE|={len(nes)}", flush=True)
    data = dict(rows=rows, K=K)
    pickle.dump(data, open(cache, "wb"))
    return data


# --------------------------------------------------------------------------
# Hero comparison figure at full K
# --------------------------------------------------------------------------

def _load_sampling_nes_symmetric_w1(K: int) -> tuple:
    """Load NEs from the sampling-iDP symmetric-w=1 sweep at grid size K.

    Preference: (1) `_cache_v2/C6_K{K}.pkl` (K=200 case),
                (2) `_cache_v2/C1.pkl` (K=22 case, used for the K=22 hero).
    Returns (actions, nes).
    """
    p_c6 = CACHE_V2 / f"C6_K{K}.pkl"
    if p_c6.exists():
        d = pickle.load(open(p_c6, "rb"))
        return np.asarray(d["actions"]), list(d["nes"])
    if K == 22:
        p = CACHE_V2 / "C1.pkl"
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return np.asarray(d["actions"]), list(d["nes"])
    return None, None


def make_hero_figure(K: int, samp_actions=None, samp_nes=None):
    grid = load_or_compute_scale_grid(K)
    c1 = experiment_C1_scale(K=K)
    actions = grid["actions"]
    sigma = grid["sigma"]
    A = grid["A"]
    feas = grid["feasible"]
    nes_scale = c1["nes"]

    if samp_actions is None or samp_nes is None:
        samp_actions, samp_nes = _load_sampling_nes_symmetric_w1(K)
    if samp_actions is None:
        # Fall back to K=22 sampling comparison.
        samp_actions, samp_nes = _load_sampling_nes_symmetric_w1(22)

    # Constrained layout resolves the colorbar/y-label collision cleanly;
    # do NOT combine with plt.tight_layout(). Aspect matches the H8 hero.
    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.36),
                              constrained_layout=True)

    X, Y = np.meshgrid(actions, actions, indexing="xy")

    def _style_heatmap_axis(ax):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(EPS_MIN, EPS_MAX); ax.set_ylim(EPS_MIN, EPS_MAX)
        ax.grid(False)  # drop the default paper-style dashed grid on heatmaps
        # A single set of major ticks at round eps values for readability.
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64])
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])

    # (a) sigma_scale heatmap -- pcolormesh with rasterization for a light
    # cell rendering at K=200 without JPEG-style artefacts in the PDF.
    ax = axes[0]
    Splot = np.where(feas, sigma, np.nan)
    im = ax.pcolormesh(X, Y, Splot, cmap="Purples", shading="nearest",
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\sigma_\mathrm{scale}$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style_heatmap_axis(ax)
    ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(r"(a) Scale: shared $\sigma_\mathrm{scale}(\varepsilon_1,\varepsilon_2)$")

    # (b) A_1 heatmap
    ax = axes[1]
    Aplot = np.where(feas, A[:, :, 0], np.nan)
    im = ax.pcolormesh(X, Y, Aplot, cmap="Oranges", shading="nearest",
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$A_1$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style_heatmap_axis(ax)
    ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(r"(b) Scale: $A_1$ flat in $\varepsilon_2$")

    # (c) NE overlay: sampling-iDP (K=200) vs Scale (K=K)
    ax = axes[2]
    ax.plot([EPS_MIN, EPS_MAX], [EPS_MIN, EPS_MAX], "--", color=ACCENT_GREY,
            lw=0.6, alpha=0.7, label=r"$\varepsilon_1{=}\varepsilon_2$",
            zorder=1)
    for k, (i, j) in enumerate(samp_nes):
        ax.loglog(samp_actions[i], samp_actions[j], "s", color=PRIMARY_BLUE,
                  markersize=3.6, markeredgecolor="white", markeredgewidth=0.25,
                  alpha=0.8, zorder=2,
                  label=f"sampling-iDP NE (K={len(samp_actions)})" if k == 0 else None)
    for k, (i, j) in enumerate(nes_scale):
        ax.loglog(actions[i], actions[j], "*", color=PRIMARY_ORANGE,
                  markersize=11.0, markeredgecolor="white", markeredgewidth=0.5,
                  zorder=3,
                  label=f"Scale NE (K={K})" if k == 0 else None)
    # Match the heatmap panels' axis window exactly.
    ax.set_xlim(EPS_MIN, EPS_MAX); ax.set_ylim(EPS_MIN, EPS_MAX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64])
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
    # Light dashed grid, matched to the paper style but subdued.
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.25)
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(r"(c) Pure NEs at $w_1{=}w_2{=}1$")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    out_pdf = FIG_DIR / "fig_h2_scale.pdf"
    out_svg = FIG_DIR / "fig_h2_scale.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close()
    print(f"[figure] wrote {out_pdf}", flush=True)
    print(f"[figure] wrote {out_svg}", flush=True)

    # Diagnostic
    def diag_split(nes, actions):
        on = sum(1 for (i, j) in nes if i == j)
        off = len(nes) - on
        # off-diagonal budget-ratio range
        ratios = [max(actions[i], actions[j]) / max(min(actions[i], actions[j]), 1e-30)
                  for (i, j) in nes if i != j]
        return on, off, ratios

    on_s, off_s, r_s = diag_split(samp_nes, samp_actions)
    on_sc, off_sc, r_sc = diag_split(nes_scale, actions)
    print()
    print(f"=== T18 / H2 diagnostic (K={K}, w=1) ===", flush=True)
    print(f"Sampling-iDP NEs (K={len(samp_actions)}): {len(samp_nes)} total, "
          f"{on_s} on diagonal, {off_s} off-diagonal, "
          f"max ratio = {max(r_s) if r_s else 1.0:.2f}", flush=True)
    print(f"Scale NEs        (K={K}): {len(nes_scale)} total, "
          f"{on_sc} on diagonal, {off_sc} off-diagonal, "
          f"max ratio = {max(r_sc) if r_sc else 1.0:.2f}", flush=True)
    return dict(nes_scale=nes_scale, nes_sampling=samp_nes,
                actions_scale=actions, actions_sampling=samp_actions,
                on_s=on_s, off_s=off_s, ratios_s=r_s,
                on_sc=on_sc, off_sc=off_sc, ratios_sc=r_sc)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of C1,C7,C8,HERO.")
    ap.add_argument("--K1", type=int, default=22)
    ap.add_argument("--K7", type=int, default=200)
    ap.add_argument("--K8", type=int, default=200)
    ap.add_argument("--Khero", type=int, default=200,
                    help="K for the fig_h2_scale hero figure.")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    RUNS = [
        ("C1",   lambda: experiment_C1_scale(K=args.K1)),
        ("C7",   lambda: experiment_C7_scale(K=args.K7)),
        ("C8",   lambda: experiment_C8_scale(K=args.K8)),
        ("HERO", lambda: make_hero_figure(K=args.Khero)),
    ]
    for name, fn in RUNS:
        if only is None or name in only:
            fn()
