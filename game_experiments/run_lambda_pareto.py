"""
lambda-Pareto sweep for the coalition (Task 14).

Coalition objective (game/game.py:216):
    obj(eps_vec) = lambda * A_target(eps_vec) + sum_{i in C} u_i(eps_vec)

The endpoints of interest:
  * lambda = 0        : the benign coalition (own-utility only).
  * lambda -> infty   : the target-vulnerability-maximising attack
                        (reproduces the k=4 row of C4).

Setup mirrors experiment_C4 (n=5, target=0, non-coalition eps=32) so the
lambda -> infty endpoint of the sweep coincides with the k=4 point of the
C4 cache.

Coalition action is restricted to the symmetric subspace
    eps_1 = eps_2 = eps_3 = eps_4 = eps_C
following the C4 template (Task 07's 2-player self-sorting result covers
lambda = infty only; asymmetric coalitions are out of scope here).

Sweep cost. Since the coalition objective differs across lambda but the
mechanism state depends only on the action vector, we evaluate every
candidate eps_vec exactly once and then compute the argmax over eps_C for
each lambda in a cheap Python loop. Total mechanism solves: |coal_grid|,
default 22.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import ContinuousGame
from game.utility import PlayerProfile, value_fn
from game.advantage import mia_advantage_subsampled_gaussian
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, ACCENT_GREY,
    column_size, two_column_size, orange_palette,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0

N = 5
TARGET = 0
COALITION = (1, 2, 3, 4)
NON_COAL_EPS = 32.0
GROUP_SIZE = 5000
EXPECTED_BATCH = 128
EPOCHS = 5


def make_game() -> ContinuousGame:
    M = GROUP_SIZE * N
    steps = int(EPOCHS * M / EXPECTED_BATCH)
    players = tuple(PlayerProfile(f"P{i}", w=1.0) for i in range(N))
    return ContinuousGame(
        players=players, group_sizes=tuple([GROUP_SIZE] * N),
        expected_batch_size=EXPECTED_BATCH, steps=steps, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX, K=2,
    )


def experiment_lambda_pareto(k_grid: int = 22, tag: str = "lambda_pareto"):
    print(f"[lambda_pareto] n={N}, target={TARGET}, coalition={COALITION}, K={k_grid}")
    cache = CACHE_DIR / f"{tag}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
        print(f"  loaded cache {cache.name}")
    else:
        game = make_game()
        coal_grid = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), k_grid)
        # Evaluate every candidate coalition action once; the mechanism state
        # depends only on eps_vec, not on lambda.
        eval_rows = []
        t0 = time.time()
        for j, c_act in enumerate(coal_grid):
            eps_vec = [NON_COAL_EPS] * N
            for ci in COALITION:
                eps_vec[ci] = float(c_act)
            ev = game.evaluate(tuple(eps_vec))
            if not ev["feasible"]:
                eval_rows.append(dict(eps_C=float(c_act), feasible=False))
                continue
            A_target = float(ev["advantages"][TARGET])
            U_C = float(sum(ev["utilities"][ci] for ci in COALITION))
            eval_rows.append(dict(
                eps_C=float(c_act), feasible=True,
                eps_vec=tuple(eps_vec),
                sigma=float(ev["sigma"]),
                A_target=A_target,
                U_coalition=U_C,
                utilities=tuple(float(u) for u in ev["utilities"]),
            ))
            dt = time.time() - t0
            print(f"  [{j+1}/{k_grid}] eps_C={c_act:.3f} sigma={ev['sigma']:.4f} "
                  f"A_t={A_target:.4f} U_C={U_C:.4f}  ({dt:.1f}s)")

        # Now sweep lambda: obj = lambda * A_target + U_C, per row.
        lam_grid = np.concatenate([[0.0], np.logspace(-2, 3, 13), [1e6]])
        rows = []
        feasible = [r for r in eval_rows if r["feasible"]]
        for lam in lam_grid:
            best = max(feasible, key=lambda r: lam * r["A_target"] + r["U_coalition"])
            rows.append(dict(
                lam=float(lam),
                eps_C_star=best["eps_C"],
                A_target=best["A_target"],
                U_coalition=best["U_coalition"],
                sigma=best["sigma"],
                eps_vec=best["eps_vec"],
                feasible=True,
            ))
            print(f"  lam={lam:12.4g}  eps_C*={best['eps_C']:6.3f}  "
                  f"A_t={best['A_target']:.4f}  U_C={best['U_coalition']:+.4f}")

        data = dict(
            rows=rows, eval_rows=eval_rows,
            lam_grid=lam_grid, coal_grid=coal_grid,
            n=N, target=TARGET, coalition=list(COALITION),
            non_coal_eps=NON_COAL_EPS,
        )
        pickle.dump(data, open(cache, "wb"))
        print(f"  cached -> {cache}")

    return data


def plot_lambda_pareto(data: dict, tag: str = "fig_lambda_pareto"):
    """Two-panel figure explaining why the coalition Pareto frontier collapses.

    Panel (a) plots the feasible slice of the coalition objective:
        U_C(eps_C)  and  A_t(eps_C)  vs eps_C.
    Both curves are strictly monotone-decreasing in eps_C on the entire
    feasible region, so the coalition's own optimum coincides with the
    attacker's target-maximising optimum.

    Panel (b) is the corresponding Pareto frontier of (U_C, A_t) as lambda
    varies: because argmax_{eps_C} [U_C + lambda A_t] is eps_C-invariant
    with respect to lambda, all 15 lambdas map to the same point.
    """
    eval_rows = [r for r in data["eval_rows"] if r["feasible"]]
    eval_rows.sort(key=lambda r: r["eps_C"])
    eps_grid = np.array([r["eps_C"] for r in eval_rows])
    U_grid = np.array([r["U_coalition"] for r in eval_rows])
    A_grid = np.array([r["A_target"] for r in eval_rows])

    rows = data["rows"]
    lam_grid = np.array(data["lam_grid"])
    U_lam = np.array([r["U_coalition"] for r in rows])
    A_lam = np.array([r["A_target"] for r in rows])

    palette = orange_palette(len(lam_grid))

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=two_column_size(aspect=0.32)
    )

    # ---------------- Panel (a): co-monotonicity ------------------
    ax_l.plot(eps_grid, U_grid, "o-", color=PRIMARY_ORANGE, lw=1.2, ms=4,
              label=r"$U_C = \sum_{i\in C} u_i$")
    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"symmetric coalition budget  $\varepsilon_C$")
    ax_l.set_ylabel(r"coalition own-utility  $U_C$", color=PRIMARY_ORANGE)
    ax_l.tick_params(axis="y", labelcolor=PRIMARY_ORANGE)

    ax_l2 = ax_l.twinx()
    ax_l2.plot(eps_grid, A_grid, "s--", color=ACCENT_GREY, lw=1.2, ms=4,
               label=r"$A_t$")
    ax_l2.set_ylabel(r"target harm  $A_t$", color=ACCENT_GREY)
    ax_l2.tick_params(axis="y", labelcolor=ACCENT_GREY)

    eps_star = rows[0]["eps_C_star"]
    ax_l.axvline(eps_star, color="k", lw=0.6, ls=":", alpha=0.5)
    ax_l.annotate(
        rf"$\varepsilon_C^\star = {eps_star:.2f}$",
        xy=(eps_star, U_grid.max()),
        xytext=(6, -8), textcoords="offset points", fontsize=7,
    )
    ax_l.set_title("(a) coalition slice: $U_C$ and $A_t$ co-monotone in $\\varepsilon_C$",
                   fontsize=8)

    # ---------------- Panel (b): degenerate frontier --------------
    # Show the feasible U_C-A_t curve as context, plus overlaid lambda points.
    ax_r.plot(U_grid, A_grid, "-", color=ACCENT_GREY, lw=0.8, alpha=0.55,
              zorder=1, label="feasible slice")
    ax_r.scatter(U_grid, A_grid, color=ACCENT_GREY, s=14, alpha=0.55, zorder=2)

    # Overlay all 15 lambda solutions with orange palette; they coincide.
    for uc, at, col in zip(U_lam, A_lam, palette):
        ax_r.scatter([uc], [at], color=col, s=44, zorder=4,
                     edgecolor="white", linewidth=0.6)

    # Annotate two attractive extremes on the feasible slice.
    ax_r.annotate(
        rf"benign optimum $=$ attack optimum" "\n"
        rf"$(U_C^\star, A_t^\star){{=}}({U_lam[0]:.2f}, {A_lam[0]:.2f})$",
        xy=(U_lam[0], A_lam[0]),
        xytext=(-8, -22), textcoords="offset points",
        fontsize=7, color=PRIMARY_ORANGE, ha="right",
        arrowprops=dict(arrowstyle="-", color=PRIMARY_ORANGE, lw=0.5),
    )
    ax_r.annotate(
        r"$\varepsilon_C{=}64$ (loose)",
        xy=(U_grid[-1], A_grid[-1]),
        xytext=(6, -2), textcoords="offset points",
        fontsize=6, color=ACCENT_GREY,
    )

    ax_r.set_xlabel(r"coalition own-utility  $U_C$")
    ax_r.set_ylabel(r"target harm  $A_t$")
    ax_r.set_title("(b) $\\lambda$-frontier collapses to one point", fontsize=8)

    fig.suptitle(
        rf"C4 extended: $n{{=}}{data['n']}$, $k{{=}}{len(data['coalition'])}$, "
        rf"target $\varepsilon^{{\rm tgt}}{{=}}{data['non_coal_eps']:g}$; "
        rf"$\lambda \in \{{0, 10^{{-2}}, \ldots, 10^6\}}$ (15 values)",
        fontsize=8, y=1.02,
    )

    plt.tight_layout()
    for ext in ("pdf", "svg"):
        out = FIG_DIR / f"{tag}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=22, help="coalition action grid size")
    p.add_argument("--tag", type=str, default="lambda_pareto")
    p.add_argument("--fig-tag", type=str, default="fig_lambda_pareto")
    p.add_argument("--fig-dir", type=str, default=None,
                   help="override figure directory (default: paper/figures)")
    args = p.parse_args()

    if args.fig_dir is not None:
        global FIG_DIR
        FIG_DIR = Path(args.fig_dir)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = experiment_lambda_pareto(k_grid=args.K, tag=args.tag)
    plot_lambda_pareto(data, tag=args.fig_tag)


if __name__ == "__main__":
    main()
