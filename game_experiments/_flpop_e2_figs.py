"""Build the two E2 figures from cached learning/reference results.

  fig_flpop_learning   : full-width, 4 panels vs iteration -- V(sigma),
                         average eps, Gini of realized advantage, mean
                         instantaneous regret (log y). Horizontal reference
                         lines for uniform-best / planner / PEG-equilibrium.
                         Adaptive learning = mean +/- 1 SE band over seeds.
  fig_flpop_trajectory : single column, joint (eps_a, eps_b) budget trajectory
                         of the two most-interchangeable agents for one
                         representative seed + faint alternates; diagonal and
                         equilibrium point overlaid.

All quantities are level-2 (PLD-exact).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    two_column_size, column_size,
)
from game_experiments._flpop_e2_oracle import ACTIONS

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "game_experiments" / "_cache_flpop_e2"
FIG_DIR = ROOT / "paper" / "figures"

apply_paper_style()

_UNI = ACCENT_GREY
_PLAN = "#2ca25f"
_EQ = PRIMARY_BLUE
_LEARN = PRIMARY_ORANGE


def _load(tag: str):
    """``tag`` is the run tag from run_flpop_e2 (e.g. ``pop0_Eb256_ep10_...``)."""
    with open(CACHE_DIR / f"references_{tag}.pkl", "rb") as f:
        refs = pickle.load(f)
    with open(CACHE_DIR / f"learn_myopic_{tag}.pkl", "rb") as f:
        learn = pickle.load(f)
    return refs, learn


def _band(ax, x, arr, color, label):
    """arr: (n_points, n_seeds). Plot mean +/- 1 SE."""
    m = arr.mean(axis=1)
    se = arr.std(axis=1) / np.sqrt(arr.shape[1])
    ax.plot(x, m, "-", color=color, lw=1.6, label=label, zorder=5)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.22, lw=0, zorder=4)


def _hlines(ax, refs, key):
    for label, ref, color, ls in [
        ("uniform-best", refs["uniform_best"], _UNI, ":"),
        ("planner (approx.)", refs["planner"], _PLAN, "-."),
        ("PEG equilibrium", refs["equilibrium"], _EQ, "--"),
    ]:
        ax.axhline(ref[key], color=color, ls=ls, lw=1.1, alpha=0.9, zorder=2)


def build_learning(tag: str):
    refs, learn = _load(tag)
    x = learn["log_iters"]
    xr = learn["regret_iters"]
    m = learn["metrics"]

    fig, axes = plt.subplots(1, 4, figsize=two_column_size(rows=1, aspect=0.30))

    ax = axes[0]
    _band(ax, x, m["V"], _LEARN, "adaptive")
    _hlines(ax, refs, "V")
    ax.set_ylabel(r"model quality $V(\sigma)$")
    ax.set_title("(a) Accuracy proxy")

    ax = axes[1]
    _band(ax, x, m["avg_eps"], _LEARN, "adaptive")
    _hlines(ax, refs, "avg_eps")
    ax.set_ylabel(r"average budget $\bar\varepsilon$")
    ax.set_title("(b) Average budget")

    ax = axes[2]
    _band(ax, x, m["gini_A"], _LEARN, "adaptive")
    _hlines(ax, refs, "gini")
    ax.set_ylabel(r"Gini of realized advantage")
    ax.set_title("(c) Privacy inequality")

    ax = axes[3]
    reg = learn["regret"].copy()
    reg = np.clip(reg, 1e-4, None)
    _band(ax, xr, reg, _LEARN, "adaptive")
    ax.set_yscale("log")
    ax.set_ylabel(r"mean instantaneous regret")
    ax.set_title("(d) Regret")

    for ax in axes:
        ax.set_xlabel("iteration")

    # single shared legend for the reference lines
    handles = [
        plt.Line2D([], [], color=_LEARN, lw=1.6, label="adaptive (mean $\\pm$1 SE)"),
        plt.Line2D([], [], color=_EQ, ls="--", lw=1.1, label="PEG equilibrium"),
        plt.Line2D([], [], color=_PLAN, ls="-.", lw=1.1, label="planner (approx.)"),
        plt.Line2D([], [], color=_UNI, ls=":", lw=1.1, label="uniform-best"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    out = FIG_DIR / "fig_flpop_learning"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"[E2] wrote {out.with_suffix('.pdf').name}")


def pick_pair(refs):
    """Two most-interchangeable agents by (log w, log |G|) proximity."""
    w = np.asarray(refs["weights"], float)
    g = np.asarray(refs["group_sizes"], float)
    z = np.column_stack([np.log(w), np.log(g)])
    z = (z - z.mean(0)) / z.std(0)
    n = len(w)
    best, bd = None, np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(z[i] - z[j])
            if d < bd:
                bd = d; best = (i, j)
    return best, bd


def build_trajectory(tag: str):
    refs, learn = _load(tag)
    (a, b), dist = pick_pair(refs)
    sh = learn["state_hist"]           # (n_logpoints, n_seeds, n) idx
    eps_grid = np.asarray(ACTIONS)
    n_seeds = sh.shape[1]

    fig, ax = plt.subplots(figsize=column_size(rows=1, aspect=0.95))
    # faint alternate seeds
    alt = list(range(min(6, n_seeds)))
    rep = alt[0]
    jit = 1.03  # multiplicative jitter to separate overlapping paths
    rng = np.random.default_rng(0)
    for k, s in enumerate(alt):
        ea = eps_grid[sh[:, s, a]] * (jit ** rng.normal(0, 0.5, sh.shape[0]))
        eb = eps_grid[sh[:, s, b]] * (jit ** rng.normal(0, 0.5, sh.shape[0]))
        if s == rep:
            continue
        ax.plot(ea, eb, "-", color=ACCENT_GREY, lw=0.7, alpha=0.35, zorder=2)

    # representative seed, coloured by time
    ea = eps_grid[sh[:, rep, a]].astype(float)
    eb = eps_grid[sh[:, rep, b]].astype(float)
    ax.plot(ea, eb, "-", color=_LEARN, lw=1.4, alpha=0.6, zorder=3)
    sc = ax.scatter(ea, eb, c=learn["log_iters"], cmap="Oranges", s=14,
                    zorder=4, edgecolors="none")
    ax.scatter([ea[0]], [eb[0]], marker="o", s=45, facecolors="none",
               edgecolors=PRIMARY_BLUE, lw=1.3, zorder=6, label="start")

    # diagonal
    lo, hi = eps_grid.min() * 0.9, eps_grid.max() * 1.1
    ax.plot([lo, hi], [lo, hi], color=ACCENT_GREY, ls="--", lw=0.8, alpha=0.7,
            zorder=1)

    # equilibrium point(s)
    eq = refs["equilibrium"]["eps"]
    ax.scatter([eq[a]], [eq[b]], marker="*", s=140, color=_EQ, zorder=7,
               edgecolors="white", lw=0.5, label="PEG equilibrium")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(eps_grid); ax.set_yticks(eps_grid)
    ax.set_xticklabels([f"{int(e)}" for e in eps_grid])
    ax.set_yticklabels([f"{int(e)}" for e in eps_grid])
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(fr"budget of agent {a} $\varepsilon_{{{a}}}$")
    ax.set_ylabel(fr"budget of agent {b} $\varepsilon_{{{b}}}$")
    ax.set_title("Near-interchangeable agents settle into opposite roles")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("iteration")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "fig_flpop_trajectory"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"[E2] wrote {out.with_suffix('.pdf').name} "
          f"(pair=({a},{b}) z-dist={dist:.3f} "
          f"w=({refs['weights'][a]:.3f},{refs['weights'][b]:.3f}) "
          f"|G|=({refs['group_sizes'][a]},{refs['group_sizes'][b]}))")


def build_all(tag: str):
    build_learning(tag)
    build_trajectory(tag)


if __name__ == "__main__":
    import sys
    build_all(sys.argv[1])
