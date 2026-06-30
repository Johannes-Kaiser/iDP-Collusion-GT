"""
Demo simulation of the Bayesian Repeated PEG.

H is fixed at a specific type theta_H. M plays two strategies:
  (a) constant - M always plays eps_M = eps_M_const and a fixed n_M.
  (b) posterior-greedy Bayesian - M updates its posterior over a small
      type grid and picks (eps_M, n_M) to maximise expected A_H.

We report A_H over rounds and M's posterior dynamics for both strategies.
"""
from __future__ import annotations

import sys, time, pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.bayesian import (
    ThetaH, GameConfig, simulate,
    strategy_constant, strategy_posterior_greedy,
)
from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    two_column_size, column_size,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()


def main():
    # Setup
    EPS_MIN, EPS_MAX = 0.5, 64.0
    K_eps = 9
    eps_grid = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), K_eps)
    eps_M_grid = eps_grid  # M's candidate budgets
    expected_batch = 128
    n_H_true = 5000
    n_M = 5000
    M_total = n_H_true + n_M
    steps = int(5 * M_total / expected_batch)
    delta = 1e-12

    # Type set for H -- M does not know which one is true.
    types = [
        ThetaH(B=0.5, C=1.0, n=n_H_true),   # privacy-preferring
        ThetaH(B=1.0, C=1.0, n=n_H_true),   # balanced
        ThetaH(B=2.0, C=1.0, n=n_H_true),   # utility-preferring
    ]
    cfg = GameConfig(
        types=types, eps_grid=eps_grid,
        expected_batch=expected_batch, steps=steps, delta=delta,
        posterior_noise=0.4,
    )

    # H's true type (unknown to M)
    theta_H_true = types[2]   # utility-preferring

    T = 15

    cache = CACHE_DIR / "bayesian_demo.pkl"
    if cache.exists():
        sims = pickle.load(open(cache, "rb"))
    else:
        sims = {}
        print(f"True theta_H = {theta_H_true}")

        # Strategy A: constant low eps_M
        print("\n[A] M plays constant eps_M = 1 (naive attack)")
        t0 = time.time()
        sims["constant_low"] = simulate(
            theta_H_true,
            strategy_constant(eps_M=1.0, n_M=n_M),
            cfg, T=T, verbose=True,
        )
        print(f"  took {time.time()-t0:.0f}s")

        # Strategy B: posterior-greedy Bayesian
        print("\n[B] M plays posterior-greedy Bayesian")
        t0 = time.time()
        sims["bayesian"] = simulate(
            theta_H_true,
            strategy_posterior_greedy(eps_M_grid=eps_M_grid, n_M=n_M),
            cfg, T=T, verbose=True,
        )
        print(f"  took {time.time()-t0:.0f}s")

        # Strategy C: constant high eps_M
        print("\n[C] M plays constant eps_M = 64 (benign baseline)")
        t0 = time.time()
        sims["constant_high"] = simulate(
            theta_H_true,
            strategy_constant(eps_M=64.0, n_M=n_M),
            cfg, T=T, verbose=True,
        )
        print(f"  took {time.time()-t0:.0f}s")
        pickle.dump(sims, open(cache, "wb"))

    # ---------------------------------------------------------
    # Plot 1: A_H over rounds for each strategy.
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.4))

    ax = axes[0]
    for key, color, label, marker in [
        ("constant_low", PRIMARY_ORANGE, r"M constant $\varepsilon_M{=}1$", "o"),
        ("bayesian", PRIMARY_BLUE, "M Bayesian-greedy", "s"),
        ("constant_high", ACCENT_GREY, r"M constant $\varepsilon_M{=}64$", "^"),
    ]:
        hist = sims[key]["history"]
        ts = [h["t"] for h in hist]
        A_H = [h["A_H"] for h in hist]
        ax.plot(ts, A_H, marker + "-", color=color, label=label,
                markersize=4, lw=1.0)
    ax.set_xlabel("round $t$")
    ax.set_ylabel(r"target's MIA advantage $A_H$")
    ax.set_title("Per-round vulnerability of $H$")
    ax.legend(frameon=False, fontsize=6, loc="best")

    # ---------------------------------------------------------
    # Plot 2: posterior over theta_H under the Bayesian strategy.
    # ---------------------------------------------------------
    ax = axes[1]
    hist_B = sims["bayesian"]["history"]
    K = len(cfg.types)
    P = np.zeros((len(hist_B) + 1, K))
    P[0] = np.ones(K) / K
    for k, h in enumerate(hist_B):
        if "posterior_after" in h:
            P[k+1] = h["posterior_after"]
        else:
            P[k+1] = P[k]
    palette = ["#fd8c3b", "#08306b", "#0b9b4d"]
    for k, theta in enumerate(cfg.types):
        marker = "*" if theta == sims["bayesian"]["theta_H_true"] else ""
        lbl = f"$\\theta_{k+1}$=$(B{{=}}{theta.B:g}, C{{=}}{theta.C:g})$"
        if marker:
            lbl += " (true)"
        ax.plot(range(len(P)), P[:, k], "-o", color=palette[k % len(palette)],
                markersize=3, lw=1.0, label=lbl)
    ax.set_xlabel("round $t$")
    ax.set_ylabel("posterior $\\Pr[\\theta_H = \\theta_k]$")
    ax.set_title("M's posterior over $H$'s type")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=6, loc="best")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_bayesian.pdf")
    fig.savefig(FIG_DIR / "fig_bayesian.svg")
    plt.close()
    print(f"\n-> {FIG_DIR / 'fig_bayesian.pdf'}")


if __name__ == "__main__":
    main()
