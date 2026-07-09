"""
H3: fictitious-play sweep of the symmetric n-player Privacy Externality Game
for n in {3, 5, 10, 50}. Strengthens the C1 asymmetric-NE claim and the C8
self-sorting claim by extending both beyond n=2.

Protocol.
    - Symmetric utility u_i(eps) = V(sigma(eps)) - w_i * A_i(eps),
      V(sigma) = 1 / (1 + sigma^2).
    - Two w-regimes: "symmetric" (w_i = 1 for all i) and "heterogeneous"
      (w_i = np.geomspace(0.25, 4, n) so that log(w_i) is uniform in a
      factor-16 window centred on 1).
    - K = |actions| grid, log-spaced.
    - For n <= 5: brute-force enumerate the K^n grid, keep every profile
      that is a pure NE (no single-player deviation strictly improves).
    - For n >= 10: Monte-Carlo fictitious play (Fudenberg--Levine 1998,
      "The Theory of Learning in Games", MIT Press, Ch. 2), with best
      response computed against `n_samples_per_round` i.i.d. joint samples
      of the other players' empirical mixed strategy. We run
      `n_seeds` random-init histories per configuration, discard a burn-in,
      and cluster the tail-round profiles by rounding to the K-grid and
      hashing on the *permutation-invariant multiset* of eps values.
    - The mechanism solve is cached multiset-symmetrically inside
      `game/mechanism.py::solve_mechanism_cached`, so all seeds within an
      (n, w-regime) share cache.

Statistics recorded for the H3 sweep:
    (a) number of pure NEs (n <= 5) or of distinct stationary multiset
        clusters (n >= 10),
    (b) median Gini coefficient of the equilibrium eps-vector,
    (c) minority and majority realised A at the modal cluster,
    (d) heterogeneous-w only: Kendall-tau between (w_i, eps_i) across
        players at each NE / cluster.

Output artefacts:
    - Raw cache: game_experiments/_cache_v2/H3_nplayer{,_preview}.pkl
    - Figure   : paper/figures/fig_h3_fp_nplayer.{pdf,svg}

Command-line interface:
    python -m game_experiments.run_nplayer \\
        --ns 3,5,10,50 --n-seeds 32 --n-rounds 400 --n-samples-per-round 64 \\
        --regimes symmetric,heterogeneous
    python -m game_experiments.run_nplayer --preview   # tiny login-node run
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.game import make_n_player_game, Game
from game.utility import PlayerProfile
from game.solvers import fictitious_play_mc
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    two_column_size, orange_palette, blue_palette,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()

# Base scenario shared with run_all.small_scenario.
DELTA = 1e-12
ACTIONS_FULL = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)     # K=7 grid used by H3
ACTIONS_COARSE = (2.0, 8.0, 32.0, 64.0)                    # K=4 preview grid
GROUP_SIZE = 5000
EXPECTED_BATCH = 128
EPOCHS = 5

# --preview-fast overrides for a very quick login-node sanity run: small
# groups + tiny step count so each mechanism solve is <2s. Not a
# scientifically valid H3 result -- only used to smoke-test the pipeline
# end-to-end and produce a real-data figure at reduced scale.
GROUP_SIZE_FAST = 500
EXPECTED_BATCH_FAST = 32
EPOCHS_FAST = 2


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------

def build_game(n: int, w_regime: str, actions: Sequence[float],
               fast: bool = False) -> Game:
    """Return an n-player Game with the requested w-regime."""
    if w_regime == "symmetric":
        weights = np.ones(n)
    elif w_regime == "heterogeneous":
        weights = np.geomspace(0.25, 4.0, n)
    else:
        raise ValueError(f"unknown w_regime {w_regime}")
    players = [PlayerProfile(name=f"P{i}", w=float(w)) for i, w in enumerate(weights)]
    if fast:
        gs = GROUP_SIZE_FAST; eb = EXPECTED_BATCH_FAST; ep = EPOCHS_FAST
    else:
        gs = GROUP_SIZE; eb = EXPECTED_BATCH; ep = EPOCHS
    M = gs * n
    steps = int(ep * M / eb)
    return make_n_player_game(
        players=players,
        group_sizes=[gs] * n,
        expected_batch_size=eb,
        steps=steps,
        actions=tuple(actions),
        delta=DELTA,
    )


# --------------------------------------------------------------------------
# Brute-force pure-NE enumeration on the K^n grid.
# --------------------------------------------------------------------------

def brute_force_pure_NE(game: Game) -> List[Tuple[float, ...]]:
    """Return every pure NE on the game's action grid. For n <= 5 only."""
    actions = list(game.cfg.actions)
    K = len(actions)
    n = game.n
    if K ** n > 200_000:
        raise RuntimeError(f"grid K^n={K**n} too large for brute force")
    equilibria = []
    for profile in itertools.product(actions, repeat=n):
        ev = game.evaluate(profile)
        if not ev.feasible:
            continue
        is_ne = True
        for i in range(n):
            u_here = float(ev.utilities[i])
            for ai in actions:
                if ai == profile[i]:
                    continue
                alt = list(profile)
                alt[i] = ai
                ev_alt = game.evaluate(alt)
                if not ev_alt.feasible:
                    continue
                if float(ev_alt.utilities[i]) > u_here + 1e-9:
                    is_ne = False
                    break
            if not is_ne:
                break
        if is_ne:
            equilibria.append(tuple(profile))
    return equilibria


# --------------------------------------------------------------------------
# Clustering the fictitious-play tail into permutation-invariant multisets.
# --------------------------------------------------------------------------

def _multiset_key(profile: Sequence[float]) -> Tuple[float, ...]:
    return tuple(sorted(float(x) for x in profile))


def cluster_tail(history: Sequence[Tuple[float, ...]], tail: int = 50) -> Counter:
    """Return a Counter over multiset keys observed in the last `tail`
    rounds of a fictitious-play history."""
    tail_slice = history[-tail:]
    return Counter(_multiset_key(p) for p in tail_slice)


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def gini(x: Sequence[float]) -> float:
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0 or np.sum(a) <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * a) / (n * np.sum(a))) - (n + 1) / n)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b in {-1, 1}. Returns 0 if all pairs are tied."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            if sx * sy > 0:
                conc += 1
            else:
                disc += 1
        if conc + disc == 0:
            continue
    total = conc + disc
    if total == 0:
        return 0.0
    return (conc - disc) / total


def summarise_equilibria(
    game: Game,
    equilibria: List[Tuple[float, ...]],
    w_regime: str,
) -> Dict:
    """Compute the H3 statistics from a list of equilibrium profiles."""
    n = game.n
    if not equilibria:
        return dict(
            n=n, w_regime=w_regime, count=0,
            median_gini=float("nan"),
            modal_profile=None, modal_A=None, A_min=float("nan"), A_max=float("nan"),
            kendall_rate=float("nan"),
        )
    ginis = [gini(profile) for profile in equilibria]
    counter = Counter(_multiset_key(p) for p in equilibria)
    modal_key, _ = counter.most_common(1)[0]
    # Retrieve a canonical profile (sorted) and evaluate for A_min/A_max.
    modal_profile = list(modal_key)
    ev = game.evaluate(modal_profile)
    A_vec = np.asarray(ev.advantages, dtype=float)
    weights = np.array([p.w for p in game.cfg.players])
    kendalls = []
    for p in equilibria:
        # Sort players by w ascending; compare with eps ordering.
        kendalls.append(kendall_tau(weights, np.asarray(p)))
    kendall_rate = float(np.mean([1.0 if k < 0 else 0.0 for k in kendalls]))
    return dict(
        n=n, w_regime=w_regime, count=len(equilibria),
        median_gini=float(np.median(ginis)),
        modal_profile=modal_profile,
        modal_A=A_vec.tolist(),
        A_min=float(A_vec.min()),
        A_max=float(A_vec.max()),
        kendall_median=float(np.median(kendalls)),
        kendall_rate=kendall_rate,
    )


# --------------------------------------------------------------------------
# Per-config run
# --------------------------------------------------------------------------

def run_config(
    n: int,
    w_regime: str,
    actions: Sequence[float],
    n_seeds: int,
    n_rounds: int,
    n_samples_per_round: int,
    brute_force_max_n: int = 5,
    fast: bool = False,
) -> Dict:
    """Run a single (n, w_regime) configuration. Returns a dict with
    raw outputs and summary."""
    print(f"[run] n={n} w={w_regime} K={len(actions)} fast={fast}", flush=True)
    game = build_game(n, w_regime, actions, fast=fast)
    t0 = time.time()
    if n <= brute_force_max_n:
        nes = brute_force_pure_NE(game)
        histories = None
        clusters = None
        mode = "brute"
    else:
        histories = []
        rng = np.random.default_rng(20260701)
        starts = rng.choice(actions, size=(n_seeds, n))
        for s in range(n_seeds):
            _, hist = fictitious_play_mc(
                game, n_rounds=n_rounds,
                n_samples_per_round=n_samples_per_round,
                start=list(starts[s]),
                seed=int(s),
            )
            histories.append(hist)
        # Cluster the last 50 rounds of every history.
        cluster_counter: Counter = Counter()
        tail = min(50, n_rounds)
        for hist in histories:
            cluster_counter.update(cluster_tail(hist, tail=tail))
        clusters = cluster_counter
        # For summarise_equilibria we treat every observation as an
        # "equilibrium sample" -- weight of a cluster equals its
        # observed frequency.
        nes = []
        for key, freq in cluster_counter.items():
            nes.extend([list(key)] * freq)
        mode = "mc_fp"
    summary = summarise_equilibria(game, nes, w_regime)
    summary["mode"] = mode
    summary["wall_seconds"] = time.time() - t0
    return dict(
        n=n, w_regime=w_regime,
        actions=list(actions),
        summary=summary,
        equilibria=nes if mode == "brute" else None,
        histories=histories,
        clusters=(dict(clusters) if clusters is not None else None),
    )


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def plot_h3_summary(all_results: List[Dict], out_path: Path) -> None:
    """Write the H3 summary figure (2 panels) to out_path (.pdf and .svg)."""
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.36))

    ax = axes[0]
    ax2 = ax.twinx()
    ax2.grid(False)
    for w_regime, colour, marker in [
        ("symmetric", PRIMARY_BLUE, "o"),
        ("heterogeneous", PRIMARY_ORANGE, "s"),
    ]:
        rows = sorted(
            [r for r in all_results if r["w_regime"] == w_regime],
            key=lambda r: r["n"],
        )
        if not rows:
            continue
        ns = [r["n"] for r in rows]
        counts = [r["summary"]["count"] for r in rows]
        # Count of distinct clusters, not repeated observations, for n >= 10.
        # For brute (n<=5) count == #pure NEs (which we want).
        # For MC-FP (n>=10) count is number of clustered samples; we plot
        # distinct-cluster count instead.
        cluster_counts = []
        for r in rows:
            if r["clusters"] is not None:
                cluster_counts.append(len(r["clusters"]))
            else:
                cluster_counts.append(r["summary"]["count"])
        ginis = [r["summary"]["median_gini"] for r in rows]
        ax.plot(ns, cluster_counts, marker + "-", color=colour,
                label=f"# NE / clusters ({w_regime})",
                markersize=5, lw=1.2)
        ax2.plot(ns, ginis, marker + "--", color=colour,
                 markersize=4, lw=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("number of players $n$")
    ax.set_ylabel("# pure NE ($n{\\leq}5$) / # tail clusters")
    ax2.set_ylabel(r"median Gini$(\varepsilon_1,\ldots,\varepsilon_n)$", color=ACCENT_GREY)
    ax2.tick_params(axis="y", colors=ACCENT_GREY)
    ax.set_title("Equilibrium multiplicity and asymmetry")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    ax = axes[1]
    for w_regime, colour, ls in [
        ("symmetric", PRIMARY_BLUE, "-"),
        ("heterogeneous", PRIMARY_ORANGE, "--"),
    ]:
        rows = sorted(
            [r for r in all_results if r["w_regime"] == w_regime],
            key=lambda r: r["n"],
        )
        if not rows:
            continue
        ns = [r["n"] for r in rows]
        A_mins = [r["summary"]["A_min"] for r in rows]
        A_maxs = [r["summary"]["A_max"] for r in rows]
        ax.plot(ns, A_mins, ls, color=colour, marker="v", markersize=4, lw=1.0,
                label=f"$A_{{\\min}}$ ({w_regime})")
        ax.plot(ns, A_maxs, ls, color=colour, marker="^", markersize=4, lw=1.0,
                label=f"$A_{{\\max}}$ ({w_regime})", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("number of players $n$")
    ax.set_ylabel("MIA advantage at modal cluster")
    ax.set_title("Modal-cluster minority vs. majority $A$")
    ax.legend(frameon=False, fontsize=6, loc="best")

    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default=None,
                        help="Comma-separated list of n values.")
    parser.add_argument("--regimes", default="symmetric,heterogeneous",
                        help="Comma-separated list of w regimes.")
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--n-rounds", type=int, default=400)
    parser.add_argument("--n-samples-per-round", type=int, default=64)
    parser.add_argument("--grid", choices=("full", "coarse"), default="full",
                        help="K=7 (full, paper) or K=4 (coarse, preview).")
    parser.add_argument("--brute-force-max-n", type=int, default=5)
    parser.add_argument("--preview", action="store_true",
                        help="Reduced-scale login-node preview: coarse grid, "
                             "fewer seeds/rounds, n<=10.")
    parser.add_argument("--fast", action="store_true",
                        help="Use tiny group size / step count so the mechanism "
                             "solve is <2s per multiset. Not a paper-quality "
                             "H3 run: only used for pipeline smoke-testing.")
    parser.add_argument("--out-tag", default="H3_nplayer",
                        help="Cache and figure tag (default H3_nplayer).")
    args = parser.parse_args()

    if args.preview:
        args.grid = "coarse"
        args.n_seeds = max(2, args.n_seeds // 8)
        args.n_rounds = min(args.n_rounds, 60)
        args.n_samples_per_round = min(args.n_samples_per_round, 8)
        if args.ns is None:
            args.ns = "3,5,10"
        args.out_tag = args.out_tag + "_preview"
    if args.ns is None:
        args.ns = "3,5,10,50"

    ns = [int(x) for x in args.ns.split(",")]
    regimes = [r.strip() for r in args.regimes.split(",")]
    actions = ACTIONS_COARSE if args.grid == "coarse" else ACTIONS_FULL

    print(f"[H3] ns={ns} regimes={regimes} grid={args.grid} "
          f"seeds={args.n_seeds} rounds={args.n_rounds} "
          f"samples={args.n_samples_per_round}", flush=True)

    cache_path = CACHE_DIR / f"{args.out_tag}.pkl"
    all_results: List[Dict] = []
    if cache_path.exists():
        print(f"[H3] loading cached results {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            all_results = pickle.load(f)
        already = {(r["n"], r["w_regime"]) for r in all_results}
    else:
        already = set()

    for n in ns:
        for regime in regimes:
            if (n, regime) in already:
                print(f"[H3] skip n={n} {regime} (cached)", flush=True)
                continue
            res = run_config(
                n=n, w_regime=regime, actions=actions,
                n_seeds=args.n_seeds, n_rounds=args.n_rounds,
                n_samples_per_round=args.n_samples_per_round,
                brute_force_max_n=args.brute_force_max_n,
                fast=args.fast,
            )
            all_results.append(res)
            with open(cache_path, "wb") as f:
                pickle.dump(all_results, f)
            print(f"  -> saved {cache_path.name} ({res['summary']['wall_seconds']:.1f}s)",
                  flush=True)

    # Emit figure.
    fig_stem = FIG_DIR / "fig_h3_fp_nplayer"
    plot_h3_summary(all_results, fig_stem)
    print(f"[H3] figure -> {fig_stem.with_suffix('.pdf').name}")

    # Emit a compact tex fragment reporting the summary numbers.
    tex_path = CACHE_DIR / f"{args.out_tag}_table.tex"
    with open(tex_path, "w") as f:
        f.write("% auto-generated by run_nplayer.py\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("$n$ & regime & \\# NE/clust. & median Gini & $A_{\\min}$ & $A_{\\max}$ & sort $\\tau<0$ rate \\\\\n")
        f.write("\\midrule\n")
        for r in sorted(all_results, key=lambda x: (x["n"], x["w_regime"])):
            s = r["summary"]
            f.write(f"{r['n']} & {r['w_regime']} & {s['count']} & "
                    f"{s['median_gini']:.3f} & {s['A_min']:.3f} & "
                    f"{s['A_max']:.3f} & {s['kendall_rate']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[H3] table -> {tex_path.name}")


if __name__ == "__main__":
    main()
