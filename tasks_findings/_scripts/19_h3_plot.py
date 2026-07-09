"""T19 -- H3 fictitious-play n-player finaliser (single-panel).

Loads `game_experiments/_cache_v2/H3_nplayer.pkl` (the K=7 Slurm sweep
at n in {3, 4, 5, 6, 7, 8, 10}) and emits

  * paper/figures/fig_h3_fp_nplayer.{pdf,svg}
  * game_experiments/_cache_v2/H3_nplayer_table.tex

Figure design (single panel, no title, style-matched to
`paper/figures/fig_h5_v_shape.{pdf,svg}`):

  * Left y-axis (linear, 0-1): sort-order compliance rate under
    heterogeneous w -- the fraction of pure NEs (n<=5) or of raw
    last-50-round FP profiles (n>=6) with Kendall tau(w_i, eps_i) < 0.
    A rate close to 1 across every n is the C8 (self-sorting) signature.
  * Right y-axis (log): the A_max/A_min envelope ratio across all
    observed profiles and all players, plotted separately for each
    regime. A ratio bounded away from 1 across every n is the C1
    (asymmetric-NE) signature.

Numerical rigour:
  * At n<=5 (brute-force pure NE): compute the A envelope over ALL
    pure NEs and all players; not just the modal cluster.
  * At n>=6 (Monte-Carlo fictitious play): recompute Kendall's tau on
    the raw last-50-round profiles from `histories`, which preserve
    player order. Clustering to permutation-invariant multisets first
    (as the driver's `plot_h3_summary` did) loses per-player identity
    and spuriously reads out as tau >= 0 always.

Reproduce:

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/19_h3_plot.py
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
)
from game_experiments.run_nplayer import build_game, ACTIONS_FULL

CACHE = ROOT / "game_experiments" / "_cache_v2" / "H3_nplayer.pkl"
STATS_CACHE = ROOT / "game_experiments" / "_cache_v2" / "H3_nplayer_stats.pkl"
FIG_PDF = ROOT / "paper" / "figures" / "fig_h3_fp_nplayer.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_h3_fp_nplayer.svg"
TABLE_TEX = ROOT / "game_experiments" / "_cache_v2" / "H3_nplayer_table.tex"
SUMMARY = Path(__file__).with_name("19_h3_summary.txt")

TAIL = 50   # last-50-round tail used for the FP stationary approximation


def gini(x: Sequence[float]) -> float:
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0 or np.sum(a) <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * a) / (n * np.sum(a)))
                 - (n + 1) / n)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
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
    total = conc + disc
    if total == 0:
        return 0.0
    return (conc - disc) / total


def stats_for_config(item: Dict) -> Dict:
    """Compute correct H3 summary stats for one (n, regime) cache entry."""
    n = item["n"]; regime = item["w_regime"]
    game = build_game(n, regime, ACTIONS_FULL, fast=False)
    weights = np.array([p.w for p in game.cfg.players])

    if item["equilibria"] is not None:  # brute-force n <= 5
        profiles: List[Tuple[float, ...]] = [tuple(e) for e in item["equilibria"]]
        weight_of: Dict[Tuple[float, ...], int] = {p: 1 for p in profiles}
        mode = "brute"
    else:                                # MC-FP n >= 6, use raw histories
        profiles = []
        for hist in item["histories"]:
            for p in hist[-TAIL:]:
                profiles.append(tuple(float(x) for x in p))
        weight_of = dict(Counter(profiles))
        profiles = list(weight_of.keys())
        mode = "mc_fp"

    A_min = np.inf; A_max = -np.inf
    gini_values: List[Tuple[float, int]] = []
    kendall_values: List[Tuple[float, int]] = []
    weights_list: List[int] = []
    for p in profiles:
        ev = game.evaluate(list(p))
        if not ev.feasible:
            continue
        A = np.asarray(ev.advantages, dtype=float)
        A_min = min(A_min, float(A.min()))
        A_max = max(A_max, float(A.max()))
        w_here = weight_of[p]
        gini_values.append((gini(list(p)), w_here))
        kendall_values.append((kendall_tau(weights, np.asarray(p)), w_here))
        weights_list.append(w_here)

    if not gini_values:
        return dict(n=n, regime=regime, count_distinct=0, count_obs=0, mode=mode,
                    A_min=float("nan"), A_max=float("nan"),
                    median_gini=float("nan"), kendall_median=float("nan"),
                    kendall_rate=float("nan"))

    def _wmedian(vs: List[Tuple[float, int]]) -> float:
        arr = sorted(vs, key=lambda t: t[0])
        total = sum(w for _, w in arr)
        half = total / 2
        cum = 0
        for v, w in arr:
            cum += w
            if cum >= half:
                return float(v)
        return float(arr[-1][0])

    med_gini = _wmedian(gini_values)
    med_kendall = _wmedian(kendall_values)
    total_obs = sum(w for _, w in kendall_values)
    neg_obs = sum(w for v, w in kendall_values if v < 0)
    kendall_rate = neg_obs / total_obs if total_obs > 0 else 0.0

    return dict(
        n=n, regime=regime,
        count_distinct=len(profiles),
        count_obs=int(sum(weights_list)),
        A_min=float(A_min), A_max=float(A_max),
        median_gini=float(med_gini),
        kendall_median=float(med_kendall),
        kendall_rate=float(kendall_rate),
        mode=mode,
    )


def emit_table(rows: List[Dict]) -> None:
    lines = ["% auto-generated by tasks_findings/_scripts/19_h3_plot.py",
             "\\begin{tabular}{llrrrrrr}",
             "\\toprule",
             ("$n$ & regime & mode & \\# distinct & median Gini & "
              "$A_{\\min}$ & $A_{\\max}$ & $\\tau<0$ rate \\\\"),
             "\\midrule"]
    order = {"symmetric": 0, "heterogeneous": 1}
    for r in sorted(rows, key=lambda x: (x["n"], order[x["regime"]])):
        lines.append(
            f"{r['n']} & {r['regime']} & {r['mode']} & "
            f"{r['count_distinct']} & {r['median_gini']:.3f} & "
            f"{r['A_min']:.3f} & {r['A_max']:.3f} & "
            f"{r['kendall_rate']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLE_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_TEX}")


def emit_figure(rows: List[Dict]) -> None:
    """Single-panel dual-axis figure. Style-matched to
    `paper/figures/fig_h5_v_shape.pdf`: figsize ~ (5.0, 3.4),
    apply_paper_style, no title, legend outside the panel on the right.

    Left y-axis (linear 0..1): sort-order compliance rate under the
    heterogeneous-w regime -- the C8 (self-sorting) signature.
    Right y-axis (log): A_max/A_min envelope ratio across all observed
    profiles, plotted for both regimes -- the C1 (asymmetric-NE) signature.
    A grey horizontal line at ratio=1 marks "no envelope width"
    (a discretisation artefact for sym-w brute-force at n<=5)."""
    apply_paper_style()

    def _by(regime: str):
        return sorted([r for r in rows if r["regime"] == regime],
                      key=lambda r: r["n"])

    rs_h = _by("heterogeneous")
    rs_s = _by("symmetric")

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
    ax2 = ax.twinx()
    ax2.grid(False)

    # --- Left axis: sort-order compliance rate (heterogeneous only) ---
    ns_h = [r["n"] for r in rs_h]
    rates_h = [r["kendall_rate"] for r in rs_h]
    line_sort, = ax.plot(
        ns_h, rates_h, "-", marker="s", markersize=5.5,
        color=PRIMARY_ORANGE, lw=1.6, zorder=4,
        markeredgecolor="white", markeredgewidth=0.5,
        label=r"sort-order rate (het. $w$)",
    )
    # 1.0 reference line ("perfect self-sorting").
    ax.axhline(1.0, color=ACCENT_GREY, lw=0.5, ls=":", alpha=0.6, zorder=1)

    # --- Right axis: A_max/A_min envelope ratio (both regimes, log) ---
    def _ratio(r):
        if not np.isfinite(r["A_min"]) or r["A_min"] <= 0:
            return np.nan
        return r["A_max"] / r["A_min"]

    ns_s = [r["n"] for r in rs_s]
    ratio_s = [_ratio(r) for r in rs_s]
    ratio_h = [_ratio(r) for r in rs_h]
    line_env_h, = ax2.plot(
        ns_h, ratio_h, "--", marker="D", markersize=4.5,
        color=PRIMARY_ORANGE, lw=1.2, alpha=0.75, zorder=3,
        markeredgecolor="white", markeredgewidth=0.4,
        label=r"$A$-envelope ratio (het. $w$)",
    )
    line_env_s, = ax2.plot(
        ns_s, ratio_s, "--", marker="o", markersize=4.5,
        color=PRIMARY_BLUE, lw=1.2, alpha=0.75, zorder=3,
        markeredgecolor="white", markeredgewidth=0.4,
        label=r"$A$-envelope ratio (sym. $w$)",
    )
    # 1.0 reference on right axis ("no envelope width").
    ax2.axhline(1.0, color=ACCENT_GREY, lw=0.5, ls=":", alpha=0.5, zorder=1)

    # --- Axes cosmetics ---
    all_ns = sorted(set(ns_h) | set(ns_s))
    ax.set_xlabel(r"number of players $n$")
    ax.set_ylabel(r"sort-order compliance rate $\Pr[\tau<0]$",
                  color=PRIMARY_ORANGE)
    ax.tick_params(axis="y", colors=PRIMARY_ORANGE)
    ax.spines["left"].set_color(PRIMARY_ORANGE)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(all_ns)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlim(min(all_ns) - 0.5, max(all_ns) + 0.5)

    ax2.set_yscale("log")
    ax2.set_ylabel(r"MIA-advantage envelope $A_{\max}/A_{\min}$",
                   color=ACCENT_GREY)
    ax2.tick_params(axis="y", colors=ACCENT_GREY)
    ax2.spines["right"].set_color(ACCENT_GREY)
    # Force a comfortable log range that clearly contains y=1 and all data.
    ratios = [r for r in ratio_s + ratio_h if np.isfinite(r) and r > 0]
    if ratios:
        rmax = max(ratios); rmin = min(ratios)
        # Include the ratio=1 baseline in view (sym-w brute-force at n<=5).
        rmin = min(rmin, 0.8)
        ax2.set_ylim(rmin / 1.2, rmax * 1.4)

    # Combined legend outside the panel on the right, matched to h5.
    handles = [line_sort, line_env_h, line_env_s]
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles, labels, frameon=False, fontsize=7,
        loc="center left", bbox_to_anchor=(1.14, 0.5),
        handletextpad=0.5, labelspacing=0.5, borderaxespad=0.0,
    )

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)
    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")


def main() -> None:
    force = "--force" in sys.argv
    if (not force and STATS_CACHE.exists()
            and STATS_CACHE.stat().st_mtime >= CACHE.stat().st_mtime):
        with open(STATS_CACHE, "rb") as f:
            rows = pickle.load(f)
        print(f"loaded stats cache {STATS_CACHE}")
    else:
        with open(CACHE, "rb") as f:
            items = pickle.load(f)
        rows = [stats_for_config(item) for item in items]
        with open(STATS_CACHE, "wb") as f:
            pickle.dump(rows, f)
        print(f"wrote stats cache {STATS_CACHE}")

    lines = ["# T19 / H3 finaliser summary",
             "# n, regime, mode, #distinct, #obs, A_min, A_max, "
             "median_Gini, tau<0 rate"]
    for r in sorted(rows, key=lambda x: (x["n"], x["regime"])):
        lines.append(
            f"  {r['n']:3d}  {r['regime']:14s}  {r['mode']:6s}  "
            f"{r['count_distinct']:5d}  {r['count_obs']:6d}  "
            f"{r['A_min']:.4f}  {r['A_max']:.4f}  "
            f"{r['median_gini']:.3f}  {r['kendall_rate']:.3f}"
        )
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {SUMMARY}")

    emit_table(rows)
    emit_figure(rows)


if __name__ == "__main__":
    main()
