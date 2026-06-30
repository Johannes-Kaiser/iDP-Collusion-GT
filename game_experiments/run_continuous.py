"""
Continuous-action experiments for the rewritten Privacy Externality Game.

Utility:  u_i = B_i * V(sigma(eps_vec))  -  C_i * A_i(eps_vec)
          V(sigma) = 1 / (1 + sigma**2)   (Bassily 2014 / sigma^2-loss frame)
Action set:  log-spaced grid of K points in [eps_min, eps_max], default K=22.
Feasibility:  every group must have p_i >= 1e-5 in the mechanism solution.

Experiments:
  C1:  2-player payoff surface (utility, MIA advantage, BR overlay)
  C2:  asymmetric group sizes
  C3:  n-player BR dynamics (n in {3,4,5}, coarser grid)
  C4:  k-coalition Stackelberg attack
  C5:  privacy-externality 3D surface (target eps fixed, others' eps + proportion swept)
  C6:  effect of awareness (BELIEF asymmetry, not utility shape) -- discussed in paper
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import (
    ContinuousGame, sweep_2player_grid, find_pure_NE_2player,
    best_response_curve,
)
from game.mechanism import _MECH_CACHE
from game.utility import PlayerProfile
from game.advantage import mia_advantage_subsampled_gaussian
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
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
KGRID = 22       # used for the heavy C1 sweep (already cached)
KGRID_FAST = 14  # used for multi-ratio sweeps where many sweeps are repeated


def make_2player(group_sizes=(5000, 5000), B=(1.0, 1.0), C=(1.0, 1.0),
                  K=KGRID, eps_min=EPS_MIN, eps_max=EPS_MAX) -> ContinuousGame:
    M = sum(group_sizes)
    expected_batch = 128
    steps = int(5 * M / expected_batch)
    players = (PlayerProfile("P1", B=B[0], C=C[0]),
               PlayerProfile("P2", B=B[1], C=C[1]))
    return ContinuousGame(
        players=players, group_sizes=group_sizes,
        expected_batch_size=expected_batch, steps=steps, delta=DELTA,
        eps_min=eps_min, eps_max=eps_max, K=K,
    )


def make_n_player(n: int, group_size: int = 5000,
                   B: float = 1.0, C: float = 1.0,
                   K: int = 12, eps_min=EPS_MIN, eps_max=EPS_MAX) -> ContinuousGame:
    M = group_size * n
    expected_batch = 128
    steps = int(5 * M / expected_batch)
    players = tuple(PlayerProfile(f"P{i}", B=B, C=C) for i in range(n))
    return ContinuousGame(
        players=players, group_sizes=tuple([group_size] * n),
        expected_batch_size=expected_batch, steps=steps, delta=DELTA,
        eps_min=eps_min, eps_max=eps_max, K=K,
    )


# --------------------------------------------------------------------------
# C1: 2-player payoff surface (single utility, single privacy term)
# --------------------------------------------------------------------------
def experiment_C1():
    print("[C1] 2-player payoff surface (continuous action grid)")
    cache = CACHE_DIR / "C1.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        game = make_2player(K=KGRID)
        t0 = time.time()
        sweep = sweep_2player_grid(game)
        nes = find_pure_NE_2player(sweep)
        br1 = best_response_curve(sweep, player=0)
        br2 = best_response_curve(sweep, player=1)
        print(f"  computed in {time.time()-t0:.1f}s, |NE|={len(nes)}")
        data = dict(sweep=sweep, nes=nes, br1=br1, br2=br2,
                    actions=sweep["actions"])
        pickle.dump(data, open(cache, "wb"))
    actions = data["actions"]
    sweep = data["sweep"]
    U = sweep["U"]; A = sweep["A"]; feas = sweep["feasible"]

    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.32),
                              gridspec_kw=dict(wspace=0.35))
    # Mask infeasible
    U_plot = np.where(feas[:, :, None], U, np.nan)
    A_plot = np.where(feas[:, :, None], A, np.nan)
    tick_pos = np.linspace(0, len(actions)-1, 6, dtype=int)
    tick_lab = [f"{actions[i]:.1f}" for i in tick_pos]

    titles = [
        (U_plot[:, :, 0], r"$u_1(\varepsilon_1, \varepsilon_2)$", "Blues"),
        (data["br1"], "Best-response curves", None),
        (A_plot[:, :, 0], r"$A_1(\varepsilon_1, \varepsilon_2)$", "Oranges"),
    ]
    for ax, (M, title, cmap) in zip(axes, titles):
        if title == "Best-response curves":
            actions_vec = actions
            br1 = data["br1"]; br2 = data["br2"]
            mask1 = br1 >= 0
            mask2 = br2 >= 0
            ax.loglog(actions_vec[mask1],
                       actions_vec[br1[mask1]],
                       "o-", color=PRIMARY_BLUE, markersize=4, lw=1.2,
                       label=r"$\mathrm{BR}_1(\varepsilon_2)$")
            ax.loglog(actions_vec[br2[mask2]],
                       actions_vec[mask2],
                       "s-", color=PRIMARY_ORANGE, markersize=4, lw=1.2,
                       label=r"$\mathrm{BR}_2(\varepsilon_1)$")
            for (ni, nj) in data["nes"]:
                ax.scatter([actions_vec[ni]], [actions_vec[nj]],
                            marker="*", s=80, c="black",
                            edgecolors="white", linewidth=0.5,
                            zorder=5)
            ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
            ax.set_title("Best-response & pure NE")
            ax.legend(frameon=False)
            continue
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                        extent=[0, len(actions)-1, 0, len(actions)-1])
        # Overlay BR curve
        if title.startswith(r"$u_1"):
            br = data["br1"]
            ok = br >= 0
            ax.plot(np.where(ok)[0], br[ok], "o-", color="white",
                    markeredgecolor="black", markersize=3, linewidth=1.0,
                    label=r"$\mathrm{BR}_1$")
            ax.legend(loc="lower right", fontsize=6, frameon=True,
                      framealpha=0.85, edgecolor="none")
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
        ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C1_payoff.pdf")
    fig.savefig(FIG_DIR / "fig_C1_payoff.svg")
    plt.close()
    print(f"  -> fig_C1_payoff.pdf  (NEs: {[(actions[i],actions[j]) for (i,j) in data['nes']]})")
    return data


# --------------------------------------------------------------------------
# C2: asymmetric group sizes
# --------------------------------------------------------------------------
def experiment_C2():
    print("[C2] Asymmetric group sizes")
    cache = CACHE_DIR / "C2.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        size_ratios = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
        rows = []
        for r in size_ratios:
            # Keep total ~10000
            g1 = int(round(10_000 * r / (1 + r)))
            g2 = 10_000 - g1
            g1, g2 = max(g1, 200), max(g2, 200)
            game = make_2player(group_sizes=(g1, g2), K=KGRID_FAST)
            sweep = sweep_2player_grid(game)
            nes = find_pure_NE_2player(sweep)
            rows.append({"ratio": r, "g1": g1, "g2": g2,
                         "actions": sweep["actions"], "nes": nes,
                         "U": sweep["U"], "A": sweep["A"], "S": sweep["sigma"],
                         "feasible": sweep["feasible"]})
            print(f"  r={r}: |NE|={len(nes)} (e.g. {[(sweep['actions'][i],sweep['actions'][j]) for (i,j) in nes[:3]]})")
        data = dict(rows=rows)
        pickle.dump(data, open(cache, "wb"))

    rows = data["rows"]
    # Plot: NE epsilon and NE A as a function of the size ratio. If multiple
    # NEs exist, take the one with the largest realised gap |A_1 - A_2|.
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.36))
    ratios = []
    eps1, eps2 = [], []
    A1, A2 = [], []
    for r_data in rows:
        actions = r_data["actions"]
        nes = r_data["nes"]
        if not nes:
            continue
        best = max(nes,
                   key=lambda ij: abs(r_data["A"][ij[0], ij[1], 0] -
                                       r_data["A"][ij[0], ij[1], 1]))
        i, j = best
        ratios.append(r_data["ratio"])
        eps1.append(actions[i]); eps2.append(actions[j])
        A1.append(float(r_data["A"][i, j, 0]))
        A2.append(float(r_data["A"][i, j, 1]))
    axes[0].semilogx(ratios, eps1, "o-", color=PRIMARY_ORANGE,
                      label=r"$\varepsilon_1^*$ (sweep group)")
    axes[0].semilogx(ratios, eps2, "s-", color=PRIMARY_BLUE,
                      label=r"$\varepsilon_2^*$")
    axes[0].set_xlabel(r"size ratio $|G_1|/|G_2|$")
    axes[0].set_ylabel(r"equilibrium $\varepsilon$")
    axes[0].set_yscale("log")
    axes[0].set_title("Equilibrium budgets")
    axes[0].legend(frameon=False)
    axes[1].semilogx(ratios, A1, "o-", color=PRIMARY_ORANGE,
                      label=r"$A_1^*$ (sweep group)")
    axes[1].semilogx(ratios, A2, "s-", color=PRIMARY_BLUE,
                      label=r"$A_2^*$")
    axes[1].set_xlabel(r"size ratio $|G_1|/|G_2|$")
    axes[1].set_ylabel("realised MIA advantage")
    axes[1].set_title("Realised excess vulnerability")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C2_asymmetric.pdf")
    fig.savefig(FIG_DIR / "fig_C2_asymmetric.svg")
    plt.close()
    print(f"  -> fig_C2_asymmetric.pdf")
    return data


# --------------------------------------------------------------------------
# C4: k-coalition Stackelberg attack on a fixed target
# --------------------------------------------------------------------------
def experiment_C4():
    print("[C4] k-coalition Stackelberg attack (continuous)")
    cache = CACHE_DIR / "C4.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        n = 5
        target = 0
        non_coal_eps = 32.0
        # Coalition action set: same logspaced grid
        coal_grid = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), 14)
        rows = []
        for k in range(0, n):
            best = None
            for c_act in coal_grid:
                eps_vec = [non_coal_eps] * n
                for ci in range(1, 1 + k):
                    eps_vec[ci] = float(c_act)
                game = make_n_player(n=n, K=2)  # K=2 dummy, evaluator only
                ev = game.evaluate(tuple(eps_vec))
                if not ev["feasible"]:
                    continue
                A_t = float(ev["advantages"][target])
                if best is None or A_t > best["A_target"]:
                    best = dict(k=k, eps_C=float(c_act), eps_vec=tuple(eps_vec),
                                A_target=A_t, sigma=float(ev["sigma"]))
            if best is None:
                # Try if k=0 (no coalition)
                game = make_n_player(n=n, K=2)
                ev = game.evaluate(tuple([non_coal_eps]*n))
                best = dict(k=0, eps_C=None, eps_vec=tuple([non_coal_eps]*n),
                            A_target=float(ev["advantages"][target]),
                            sigma=float(ev["sigma"]))
            rows.append(best)
            print(f"  k={k}: eps_C={best['eps_C']}, A_target={best['A_target']:.3f}")
        data = dict(rows=rows, n=n, non_coal_eps=non_coal_eps,
                    coal_grid=coal_grid)
        pickle.dump(data, open(cache, "wb"))

    rows = data["rows"]
    fig, ax = plt.subplots(figsize=column_size(aspect=0.6))
    ks = [r["k"] for r in rows]
    At = [r["A_target"] for r in rows]
    epsC = [r["eps_C"] for r in rows]
    ax.plot(ks, At, "o-", color=PRIMARY_ORANGE, label="target MIA advantage")
    for k, a, e in zip(ks, At, epsC):
        if e is not None:
            ax.annotate(f"$\\varepsilon_C{{=}}{e:.1f}$", (k, a),
                        textcoords="offset points", xytext=(5, -10),
                        fontsize=6, color=ACCENT_GREY)
    ax.set_xlabel("coalition size $k$")
    ax.set_ylabel(r"$A_{\rm tgt}$ at attack optimum")
    ax.set_title(rf"$n={data['n']}$, target $\varepsilon^{{\rm tgt}}={data['non_coal_eps']:g}$")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C4_coalition.pdf")
    fig.savefig(FIG_DIR / "fig_C4_coalition.svg")
    plt.close()
    print(f"  -> fig_C4_coalition.pdf")
    return data


# --------------------------------------------------------------------------
# C5: privacy-externality surface (target eps fixed, others' eps + proportion)
# --------------------------------------------------------------------------
def experiment_C5():
    print("[C5] Privacy-externality surface")
    cache = CACHE_DIR / "C5.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        target_eps_grid = [4.0, 16.0]
        other_eps_grid = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), 12)
        proportions = [0.2, 0.4, 0.6, 0.8]
        total = 10_000
        expected_batch = 128
        from game.continuous import ContinuousGame
        rows = []
        for tg_eps in target_eps_grid:
            for ot_eps in other_eps_grid:
                for prop in proportions:
                    g_other = int(prop * total)
                    g_tgt = total - g_other
                    if min(g_tgt, g_other) < 100:
                        continue
                    M = g_tgt + g_other
                    steps = int(5 * M / expected_batch)
                    players = (PlayerProfile("T", 1, 1),
                                PlayerProfile("O", 1, 1))
                    g = ContinuousGame(
                        players=players,
                        group_sizes=(g_tgt, g_other),
                        expected_batch_size=expected_batch,
                        steps=steps, delta=DELTA,
                        eps_min=EPS_MIN, eps_max=EPS_MAX, K=2,
                    )
                    ev = g.evaluate((float(tg_eps), float(ot_eps)))
                    if not ev["feasible"]:
                        continue
                    rows.append({
                        "target_eps": float(tg_eps),
                        "other_eps": float(ot_eps),
                        "proportion": float(prop),
                        "A_target": float(ev["advantages"][0]),
                        "A_other": float(ev["advantages"][1]),
                        "sigma": float(ev["sigma"]),
                    })
        data = dict(rows=rows)
        pickle.dump(data, open(cache, "wb"))

    rows = data["rows"]
    targets = sorted(set(r["target_eps"] for r in rows))
    fig, axes = plt.subplots(1, len(targets),
                              figsize=two_column_size(rows=1, aspect=0.4),
                              sharey=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, tg in zip(axes, targets):
        proportions = sorted(set(r["proportion"] for r in rows if r["target_eps"] == tg))
        cmap = orange_palette(len(proportions))
        for ip, p in enumerate(proportions):
            xs = [r["other_eps"] for r in rows
                  if r["target_eps"] == tg and r["proportion"] == p]
            ys = [r["A_target"] for r in rows
                  if r["target_eps"] == tg and r["proportion"] == p]
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            ax.semilogx(xs, ys, "o-", color=cmap[ip], markersize=3,
                         label=f"prop={p:.1f}")
        ax.set_xlabel(r"others' budget $\varepsilon_O$")
        ax.set_ylabel(r"target's MIA advantage $A_{\rm tgt}$")
        ax.set_title(rf"target $\varepsilon_T = {tg:g}$")
        ax.legend(frameon=False, fontsize=6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C5_surface.pdf")
    fig.savefig(FIG_DIR / "fig_C5_surface.svg")
    plt.close()
    print(f"  -> fig_C5_surface.pdf")
    return data


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of C1,C2,C4,C5 to run.")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    EXPERIMENTS = [
        ("C1", experiment_C1), ("C2", experiment_C2),
        ("C4", experiment_C4), ("C5", experiment_C5),
    ]
    for name, fn in EXPERIMENTS:
        if only is None or name in only:
            fn()
