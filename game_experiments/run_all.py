"""
Run all experiments for the Privacy Externality Game paper and write figures
into paper/figures/. Designed to be re-runnable: caches mechanism solutions
in /tmp by pickled key, and reads existing caches on rerun.
"""
from __future__ import annotations

import os
import pickle
import time
import sys
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.game import make_n_player_game, Game
from game.utility import aware_player, naive_player, PlayerProfile
from game.solvers import (
    payoff_grid_2player, fictitious_play, coalition_attack_sweep,
)
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size, two_column_size, orange_palette, blue_palette,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()


# --------------------------------------------------------------------------
# Shared scenario configuration  (mirrors a typical small-data MIA study)
# --------------------------------------------------------------------------
DELTA = 1e-12
ACTIONS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
# A coarser grid used by the n>=3 experiments to keep the sweep tractable
# (action-space size = 4 vs 7 -> roughly 7x fewer multi-set profiles for n=5).
ACTIONS_COARSE = (2.0, 8.0, 32.0, 64.0)


def small_scenario(n: int, eps_max: float = 64.0,
                    awareness: str = "aware",
                    use_coarse: bool = False) -> Game:
    """Symmetric n-player scenario, 5k samples each, batch 128, 5 epochs."""
    group_size = 5000
    M = group_size * n
    expected_batch = 128
    steps = int(5 * M / expected_batch)
    if awareness == "aware":
        players = [aware_player(f"P{i}", B=1.0, C=1.0, eps_max=eps_max) for i in range(n)]
    elif awareness == "naive":
        players = [naive_player(f"P{i}", B=1.0, C=1.0, eps_max=eps_max) for i in range(n)]
    else:  # mixed: half aware, half naive
        players = []
        for i in range(n):
            if i % 2 == 0:
                players.append(aware_player(f"P{i}_A", B=1.0, C=1.0, eps_max=eps_max))
            else:
                players.append(naive_player(f"P{i}_N", B=1.0, C=1.0, eps_max=eps_max))
    return make_n_player_game(
        players=players, group_sizes=[group_size] * n,
        expected_batch_size=expected_batch, steps=steps,
        actions=(ACTIONS_COARSE if use_coarse else ACTIONS), delta=DELTA,
    )


# --------------------------------------------------------------------------
# E1: 2-player honest game, aware vs naive payoff heatmaps
# --------------------------------------------------------------------------
def experiment_E1():
    print("[E1] 2-player aware/naive payoff grids")
    out_path = FIG_DIR / "fig_E1_payoff_grids.pdf"
    out_path_svg = FIG_DIR / "fig_E1_payoff_grids.svg"
    cache = CACHE_DIR / "E1.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        game_aware = small_scenario(n=2, awareness="aware")
        game_naive = small_scenario(n=2, awareness="naive")
        t0 = time.time()
        U_aw, A_aw, S_aw = payoff_grid_2player(game_aware)
        U_nv, A_nv, S_nv = payoff_grid_2player(game_naive)
        print(f"  computed grids in {time.time()-t0:.1f}s")
        data = dict(
            U_aw=U_aw, A_aw=A_aw, S_aw=S_aw,
            U_nv=U_nv, A_nv=A_nv, S_nv=S_nv,
            actions=ACTIONS,
        )
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    actions = data["actions"]
    fig, axes = plt.subplots(2, 2, figsize=two_column_size(rows=1.6, aspect=0.5))
    titles = [
        ("Aware regime: $u_1$", data["U_aw"][:, :, 0]),
        ("Naive regime: $u_1$", data["U_nv"][:, :, 0]),
        ("Aware regime: $A_1$", data["A_aw"][:, :, 0]),
        ("Naive regime: $A_1$", data["A_nv"][:, :, 0]),
    ]
    for ax, (title, M) in zip(axes.flat, titles):
        cmap_name = "Oranges" if "$A_1$" in title else "Blues"
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap_name)
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels([f"{a:g}" for a in actions])
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels([f"{a:g}" for a in actions])
        ax.set_xlabel(r"$\varepsilon_2$ (other)")
        ax.set_ylabel(r"$\varepsilon_1$ (own)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# E2: 2-player asymmetric group sizes
# --------------------------------------------------------------------------
def experiment_E2():
    print("[E2] 2-player asymmetric group sizes")
    out_path = FIG_DIR / "fig_E2_asymmetric.pdf"
    out_path_svg = FIG_DIR / "fig_E2_asymmetric.svg"
    cache = CACHE_DIR / "E2.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        size_ratios = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
        rows = []
        for r in size_ratios:
            g1 = int(5000 * r / (1 + r) * 2)
            g2 = 10_000 - g1
            g1, g2 = max(g1, 100), max(g2, 100)
            from game.game import make_n_player_game
            players = [aware_player("P1", 1, 1, 64), aware_player("P2", 1, 1, 64)]
            expected_batch = 128
            steps = int(5 * (g1 + g2) / expected_batch)
            game = make_n_player_game(
                players=players, group_sizes=[g1, g2],
                expected_batch_size=expected_batch, steps=steps,
                actions=ACTIONS, delta=DELTA,
            )
            history = game.best_response_dynamics(start=(8.0, 8.0), max_iter=12)
            # take the average of the last 5 visited profiles as a stand-in for
            # NE (cycling games)
            tail = history[-5:]
            avg_eps = np.mean(np.array(tail), axis=0)
            # mean advantage over the cycle:
            advs = []
            for prof in tail:
                ev = game.evaluate(prof)
                if ev.feasible:
                    advs.append(ev.advantages)
            advs = np.mean(advs, axis=0) if advs else np.array([np.nan, np.nan])
            rows.append({
                "ratio": r, "g1": g1, "g2": g2,
                "avg_eps_1": float(avg_eps[0]),
                "avg_eps_2": float(avg_eps[1]),
                "A_1": float(advs[0]),
                "A_2": float(advs[1]),
                "history": history,
            })
            print(f"  r={r}: avg eps={avg_eps}, A={advs}")
        data = dict(rows=rows)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.36))
    ratios = [r["ratio"] for r in rows]
    eps1 = [r["avg_eps_1"] for r in rows]
    eps2 = [r["avg_eps_2"] for r in rows]
    A1 = [r["A_1"] for r in rows]
    A2 = [r["A_2"] for r in rows]
    axes[0].semilogx(ratios, eps1, "o-", color=PRIMARY_ORANGE,
                     label=r"Player 1 (sweep group)")
    axes[0].semilogx(ratios, eps2, "s-", color=PRIMARY_BLUE,
                     label=r"Player 2")
    axes[0].set_xlabel(r"size ratio $|G_1|/|G_2|$")
    axes[0].set_ylabel(r"BR-cycle average $\bar\varepsilon$")
    axes[0].legend(frameon=False)
    axes[0].set_title("Equilibrium budgets")
    axes[1].semilogx(ratios, A1, "o-", color=PRIMARY_ORANGE,
                     label=r"$A_1$ (sweep group)")
    axes[1].semilogx(ratios, A2, "s-", color=PRIMARY_BLUE,
                     label=r"$A_2$")
    axes[1].set_xlabel(r"size ratio $|G_1|/|G_2|$")
    axes[1].set_ylabel("MIA advantage at equilibrium")
    axes[1].legend(frameon=False)
    axes[1].set_title("Realised excess vulnerability")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# E3: n-player honest game equilibrium
# --------------------------------------------------------------------------
def experiment_E3():
    print("[E3] n-player honest game equilibrium")
    out_path = FIG_DIR / "fig_E3_nplayer.pdf"
    out_path_svg = FIG_DIR / "fig_E3_nplayer.svg"
    cache = CACHE_DIR / "E3.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        ns = [2, 3, 4, 5]
        rows = []
        for n in ns:
            game = small_scenario(n=n, awareness="aware", use_coarse=(n >= 4))
            t0 = time.time()
            history = game.best_response_dynamics(
                start=tuple([8.0] * n), max_iter=15,
            )
            tail = history[-min(5, len(history)):]
            avg_eps = np.mean(np.array(tail), axis=0).tolist()
            advs = []
            for prof in tail:
                ev = game.evaluate(prof)
                if ev.feasible:
                    advs.append(ev.advantages.tolist())
            advs_mean = np.mean(advs, axis=0).tolist() if advs else [np.nan] * n
            rows.append({
                "n": n, "avg_eps": avg_eps, "advs": advs_mean,
                "history": history,
            })
            print(f"  n={n}: avg_eps={avg_eps}, A={advs_mean}  ({time.time()-t0:.1f}s)")
        data = dict(rows=rows)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.36))
    ns = [r["n"] for r in rows]
    mean_eps = [np.mean(r["avg_eps"]) for r in rows]
    mean_A = [np.mean(r["advs"]) for r in rows]
    axes[0].plot(ns, mean_eps, "o-", color=PRIMARY_BLUE)
    axes[0].set_xlabel(r"number of players $n$")
    axes[0].set_ylabel(r"mean BR-cycle $\bar\varepsilon$")
    axes[0].set_title("Equilibrium privacy demand")
    axes[1].plot(ns, mean_A, "o-", color=PRIMARY_ORANGE)
    axes[1].set_xlabel(r"number of players $n$")
    axes[1].set_ylabel("mean MIA advantage")
    axes[1].set_title("Equilibrium excess vulnerability")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# E4: k-coalition Stackelberg attack on target
# --------------------------------------------------------------------------
def experiment_E4():
    print("[E4] k-coalition Stackelberg attack")
    out_path = FIG_DIR / "fig_E4_coalition.pdf"
    out_path_svg = FIG_DIR / "fig_E4_coalition.svg"
    cache = CACHE_DIR / "E4.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        n = 5
        game = small_scenario(n=n, awareness="aware", use_coarse=True)
        # The target plays at eps_target_fixed = 32 (large budget by assumption).
        # The coalition picks low eps; rest of (non-coalition) players also play 32.
        results = []
        target = 0
        non_coalition_action = 32.0
        for k in range(0, n):
            coalition = list(range(1, 1 + k))
            if not coalition:
                # baseline: everyone plays 32
                ev = game.evaluate(tuple([32.0] * n))
                results.append({
                    "k": 0,
                    "coalition_eps": [], "action_vec": tuple([32.0] * n),
                    "A_target": float(ev.advantages[target]),
                    "sigma": float(ev.sigma),
                })
                continue
            # adversarial: coalition all play eps_min
            best = None
            for c_act in ACTIONS:
                eps_vec = [32.0] * n  # default
                eps_vec[target] = 32.0
                for ci in coalition:
                    eps_vec[ci] = c_act
                ev = game.evaluate(tuple(eps_vec))
                if not ev.feasible:
                    continue
                A_t = float(ev.advantages[target])
                if best is None or A_t > best["A_target"]:
                    best = {
                        "k": k, "coalition_eps": [c_act] * k,
                        "action_vec": tuple(eps_vec),
                        "A_target": A_t, "sigma": float(ev.sigma),
                    }
            results.append(best)
            print(f"  k={k}: coalition_eps={best['coalition_eps']}, A_target={best['A_target']:.3f}")
        data = dict(results=results, n=n, non_coalition_action=non_coalition_action)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    results = data["results"]
    fig, ax = plt.subplots(figsize=column_size(aspect=0.6))
    ks = [r["k"] for r in results]
    A_t = [r["A_target"] for r in results]
    coal_eps = [(r["coalition_eps"][0] if r["coalition_eps"] else None) for r in results]
    ax.plot(ks, A_t, "o-", color=PRIMARY_ORANGE, label="Target MIA advantage")
    for k, A, e in zip(ks, A_t, coal_eps):
        if e is not None:
            ax.annotate(f"$\\varepsilon_C{{=}}{e:g}$", (k, A),
                        textcoords="offset points", xytext=(4, -10), fontsize=6,
                        color=ACCENT_GREY)
    ax.set_xlabel("coalition size $k$ (target excluded)")
    ax.set_ylabel("target MIA advantage at attack opt.")
    ax.set_title(f"$n={data['n']}$, target $\\varepsilon^{{tgt}}=32$")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# E5: naive-vs-aware regime comparison (price of awareness)
# --------------------------------------------------------------------------
def experiment_E5():
    print("[E5] naive vs aware regime comparison")
    out_path = FIG_DIR / "fig_E5_awareness.pdf"
    out_path_svg = FIG_DIR / "fig_E5_awareness.svg"
    cache = CACHE_DIR / "E5.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        rows = []
        for reg in ("aware", "naive"):
            for n in [2, 3, 4, 5]:
                game = small_scenario(n=n, awareness=reg, use_coarse=(n >= 4))
                history = game.best_response_dynamics(
                    start=tuple([8.0] * n), max_iter=15,
                )
                tail = history[-min(5, len(history)):]
                advs = []
                for prof in tail:
                    ev = game.evaluate(prof)
                    if ev.feasible:
                        advs.append(np.mean(ev.advantages))
                mean_A = float(np.mean(advs)) if advs else np.nan
                avg_eps = float(np.mean(np.array(tail)))
                rows.append({
                    "regime": reg, "n": n,
                    "mean_A": mean_A, "mean_eps": avg_eps,
                })
                print(f"  {reg} n={n}: mean A={mean_A:.3f}, mean eps={avg_eps:.2f}")
        data = dict(rows=rows)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.36))
    ns = sorted(set(r["n"] for r in rows))
    for reg, color, marker in [("aware", PRIMARY_BLUE, "o"),
                                 ("naive", PRIMARY_ORANGE, "s")]:
        eps_y = [next(r["mean_eps"] for r in rows if r["regime"] == reg and r["n"] == n) for n in ns]
        A_y = [next(r["mean_A"] for r in rows if r["regime"] == reg and r["n"] == n) for n in ns]
        axes[0].plot(ns, eps_y, marker + "-", color=color, label=reg)
        axes[1].plot(ns, A_y, marker + "-", color=color, label=reg)
    axes[0].set_xlabel("$n$ players"); axes[0].set_ylabel(r"mean BR-cycle $\bar\varepsilon$")
    axes[0].set_title("Equilibrium budget"); axes[0].legend(frameon=False)
    axes[1].set_xlabel("$n$ players"); axes[1].set_ylabel("mean MIA advantage")
    axes[1].set_title("Realised excess vulnerability"); axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# E6: privacy-externality 3D surface (vulnerability as a function of others'
# choices), analogous to Fig. 3 of Kaiser et al. 2026, but for the n-player game.
# --------------------------------------------------------------------------
def experiment_E6():
    print("[E6] privacy-externality surface (3D)")
    out_path = FIG_DIR / "fig_E6_surface.pdf"
    out_path_svg = FIG_DIR / "fig_E6_surface.svg"
    cache = CACHE_DIR / "E6.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        target_eps_grid = [4.0, 8.0, 16.0, 32.0]
        other_eps_grid = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        proportion_other = [0.2, 0.4, 0.6, 0.8]
        target_group_size = 2000
        total_size = 10_000
        # We construct a 2-group scenario: target group with target_eps, other
        # group with other_eps; sweep proportion.
        from game.game import make_n_player_game
        rows = []
        for tg_eps in target_eps_grid:
            for ot_eps in other_eps_grid:
                for prop in proportion_other:
                    g_other = int(prop * total_size)
                    g_target = total_size - g_other
                    if g_other < 100 or g_target < 100:
                        continue
                    players = [aware_player("Tgt", 1, 1, 64),
                               aware_player("Oth", 1, 1, 64)]
                    expected_batch = 128
                    steps = int(5 * total_size / expected_batch)
                    g = make_n_player_game(
                        players=players,
                        group_sizes=[g_target, g_other],
                        expected_batch_size=expected_batch, steps=steps,
                        actions=ACTIONS, delta=DELTA,
                    )
                    ev = g.evaluate((tg_eps, ot_eps))
                    if not ev.feasible:
                        continue
                    rows.append({
                        "target_eps": tg_eps, "other_eps": ot_eps,
                        "proportion_other": prop,
                        "A_target": float(ev.advantages[0]),
                        "A_other": float(ev.advantages[1]),
                        "sigma": float(ev.sigma),
                    })
        data = dict(rows=rows)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    rows = data["rows"]
    # Two-panel: target_eps = 4 and target_eps = 16
    target_picks = [4.0, 16.0]
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.4))
    for ax, tg_eps in zip(axes, target_picks):
        proportions = sorted(set(r["proportion_other"] for r in rows if r["target_eps"] == tg_eps))
        cmap = orange_palette(len(proportions))
        for ip, p in enumerate(proportions):
            row_set = sorted([r for r in rows if r["target_eps"] == tg_eps and r["proportion_other"] == p],
                              key=lambda r: r["other_eps"])
            xs = [r["other_eps"] for r in row_set]
            ys = [r["A_target"] for r in row_set]
            ax.semilogx(xs, ys, "o-", color=cmap[ip], label=f"prop={p:.1f}")
        ax.set_xlabel(r"others' budget $\varepsilon_O$")
        ax.set_ylabel(r"target's MIA advantage $A_{\rm tgt}$")
        ax.set_title(rf"target $\varepsilon_T = {tg_eps:g}$")
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path_svg)
    plt.close()
    print(f"  -> {out_path.name}")
    return data


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of E1..E6 to run.")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    EXPERIMENTS = [
        ("E1", experiment_E1), ("E2", experiment_E2),
        ("E3", experiment_E3), ("E4", experiment_E4),
        ("E5", experiment_E5), ("E6", experiment_E6),
    ]
    for name, fn in EXPERIMENTS:
        if only is None or name in only:
            fn()
