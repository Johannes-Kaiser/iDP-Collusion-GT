"""T11 -- Single-panel slack (excess risk) visualisation.

Loads the K=200 mechanism-grid cache
(``game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl``,
produced by ``slurm_scripts/run_locus_K200.sh``) and produces

    paper/figures/fig_slack_panel.pdf
    paper/figures/fig_slack_panel.svg

as a **single-panel** intuition figure for the paper's slack
(excess-risk) decomposition.

Computation.  For a fixed anchor budget ``eps_i`` we sweep the other
player's budget ``eps_j`` over the K=200 grid.  Three curves are
plotted on the same y-axis:

    (1) the guarantee ceiling
            c_bar(eps_i) = (exp(eps_i) - 1 + 2*delta) / (exp(eps_i) + 1)
        — the post-processing-tight (eps_i, delta)-DP upper bound on
        the adversary's MIA advantage (Hayes 2024; Kaissis 2024).
    (2) the realised MIA advantage
            A_i(eps_i, eps_j)
        of the two-fold subsampled Gaussian at profile (eps_i, eps_j),
        computed by ``game.mechanism.solve_mechanism_cached`` and
        ``game.advantage.mia_advantage_subsampled_gaussian`` and read
        off the pre-computed K=200 grid.
    (3) the slack (the excess risk the guarantee left uncounted):
            slack_i(eps_i, eps_j) = c_bar(eps_i) - A_i(eps_i, eps_j)
                                  >= 0 by construction.

The gap between the ceiling and A_i is shaded — this shaded wedge *is*
``slack_i``.  A single explicit formula annotation on the plot ties the
visual to the definition of slack in ``paper/tex/03_game.tex`` and to
``prop:slack-monotonicity`` (T04) which lifts the sign of its
eps_j-derivative to a stand-alone proposition.

Design choice.  We use the eps_i = 4 anchor row: it is fully feasible
across the eps_j grid, has no isolated solver artefact, and delivers
the widest visual wedge (slack rises from 0.79 at eps_j=0.5 to 0.96 at
eps_j=64 while c_bar sits at 0.963 — i.e. the guarantee overpromises
by almost the whole adversarial channel when the other player is
loose).  A second anchor row (eps_i = 16) would clutter the intuition
figure and repeat the same story; its statistics remain in the
findings document as backing evidence.

Style-matched to ``paper/figures/fig_h8_brpeg_regret.pdf`` (paper
style, PRIMARY_BLUE / PRIMARY_ORANGE / ACCENT_GREY, legend frameon
False, single-column figure).

Reproduce (cache-hit-only; runs in <30 s on the login node):

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/11_slack_panel.py

Outputs:
  paper/figures/fig_slack_panel.pdf
  paper/figures/fig_slack_panel.svg
  game_experiments/_cache_v2/slack_panel.pkl
  tasks_findings/_scripts/11_slack_summary.txt
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    column_size,
)

CACHE_MECH = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"
CACHE_OUT = ROOT / "game_experiments" / "_cache_v2" / "slack_panel.pkl"
FIG_PDF = ROOT / "paper" / "figures" / "fig_slack_panel.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_slack_panel.svg"
SUMMARY = Path(__file__).with_name("11_slack_summary.txt")

DELTA = 1e-12
# Anchors kept in the summary for backing evidence.  Only the first
# anchor is drawn on the intuition figure.
EPS_I_ANCHORS = [4.0, 16.0]
EPS_I_FIGURE = 4.0


def c_bar(eps: np.ndarray | float, delta: float = DELTA) -> np.ndarray | float:
    return (np.exp(eps) - 1.0 + 2.0 * delta) / (np.exp(eps) + 1.0)


def isolated_outlier_mask(y: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """Flag single-cell mechanism-solver root-finding artefacts.

    A cell is flagged if it disagrees with both neighbours by more than
    ``thresh`` while the two neighbours agree with each other to within
    ``thresh``.  Kept for the summary of the eps_i=16 anchor row; the
    eps_i=4 anchor plotted on the figure has none.
    """
    n = len(y)
    out = np.zeros(n, dtype=bool)
    for k in range(1, n - 1):
        left, mid, right = y[k - 1], y[k], y[k + 1]
        if abs(left - right) < thresh and \
                (abs(mid - left) > thresh and abs(mid - right) > thresh):
            out[k] = True
    return out


def main() -> None:
    if not CACHE_MECH.exists():
        print(f"cache missing: {CACHE_MECH}", file=sys.stderr)
        sys.exit(1)
    grid = pickle.load(open(CACHE_MECH, "rb"))
    actions = grid["actions"]   # (K,)
    A_grid = grid["A"]          # (K, K, 2)
    feas = grid["feasible"]     # (K, K)
    K = len(actions)

    rows = []
    for eps_i_target in EPS_I_ANCHORS:
        i0 = int(np.argmin(np.abs(actions - eps_i_target)))
        eps_i_actual = float(actions[i0])
        A_row = A_grid[i0, :, 0].copy()
        feas_row = feas[i0, :].copy()
        cbar_val = float(c_bar(eps_i_actual))
        slack_row = cbar_val - A_row
        rows.append(dict(
            eps_i_target=float(eps_i_target),
            eps_i_actual=eps_i_actual,
            i0=i0,
            eps_j_grid=actions.copy(),
            A=A_row,
            slack=slack_row,
            c_bar=cbar_val,
            feasible=feas_row,
        ))

    payload = dict(
        eps_i_anchors=EPS_I_ANCHORS,
        rows=rows,
        delta=DELTA,
        K=K,
        actions=actions.copy(),
    )
    CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(payload, open(CACHE_OUT, "wb"))
    print(f"wrote {CACHE_OUT}")

    # -------- Summary (kept as backing evidence for both anchors) --------
    lines = ["# T11 slack panel summary (K=200)"]
    lines.append(f"# c_bar(eps) = (exp(eps) - 1 + 2 delta) / (exp(eps) + 1), delta = {DELTA:g}")
    lines.append(f"# slack_i(eps_i, eps_j) = c_bar(eps_i) - A_i(eps_i, eps_j)")
    lines.append(f"# grid file: {CACHE_MECH.name}")
    lines.append("")
    for r in rows:
        idx = np.where(r["feasible"])[0]
        s = r["slack"][idx]
        A = r["A"][idx]
        d_s = np.diff(s)
        n_pairs = len(d_s)
        n_nondec = int((d_s >= -1e-6).sum())
        worst_dec = float(d_s.min()) if len(d_s) else 0.0
        median_slack = float(np.median(s))
        out_iso = isolated_outlier_mask(A, thresh=0.05)
        n_iso = int(out_iso.sum())
        keep_pair = ~(out_iso[:-1] | out_iso[1:])
        n_pairs_clean = int(keep_pair.sum())
        n_nondec_clean = int(((d_s >= -1e-6) & keep_pair).sum())
        lines.append(
            f"eps_i (target)={r['eps_i_target']:g}  eps_i (grid)={r['eps_i_actual']:.4f}  "
            f"c_bar={r['c_bar']:.4f}  feasible={len(idx)}/{r['eps_j_grid'].size}  "
            f"slack in [{s.min():.4f}, {s.max():.4f}]  median={median_slack:.4f}  "
            f"A in [{A.min():.4f}, {A.max():.4f}]  "
            f"non-decreasing pairs: {n_nondec}/{n_pairs} raw / "
            f"{n_nondec_clean}/{n_pairs_clean} after excluding {n_iso} isolated solver artefact(s); "
            f"worst raw decrease: {worst_dec:.5f}"
        )
        assert (s >= -1e-9).all(), "slack < 0 on feasible cell"
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {SUMMARY}")

    # -------- Figure (single panel, eps_i = 4) --------
    fig_row = next(r for r in rows if r["eps_i_target"] == EPS_I_FIGURE)
    idx = np.where(fig_row["feasible"])[0]
    e_j = fig_row["eps_j_grid"][idx]
    A = fig_row["A"][idx]
    s = fig_row["slack"][idx]
    cbar_val = fig_row["c_bar"]
    eps_i_actual = fig_row["eps_i_actual"]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=column_size(aspect=0.85))

    # (1) Slack wedge: shaded region between A_i (bottom) and c_bar (top).
    #     This shaded area IS slack_i(eps_i, eps_j).
    ax.fill_between(e_j, A, cbar_val, color=PRIMARY_BLUE, alpha=0.14,
                    linewidth=0, zorder=1,
                    label=r"slack$_i$ (excess risk)")

    # (2) Guarantee ceiling.  Horizontal dashed grey line at c_bar.
    ax.axhline(cbar_val, color=ACCENT_GREY, lw=1.1, ls="--", zorder=3,
               label=rf"DP ceiling $\bar c(\varepsilon_i){{=}}{cbar_val:.3f}$")

    # (3) Realised MIA advantage A_i.
    ax.plot(e_j, A, "-", color=PRIMARY_ORANGE, lw=1.7, zorder=4,
            label=r"realised advantage $A_{\mathrm{PLD},i}$")
    ax.plot(e_j[::14], A[::14], "o", color=PRIMARY_ORANGE, markersize=2.6,
            markeredgecolor="white", markeredgewidth=0.4, zorder=5)

    # (4) One illustrative wedge-width indicator: a vertical double-headed
    #     arrow at eps_j=8, annotated with the numeric slack value.
    k = int(np.argmin(np.abs(e_j - 8.0)))
    ax.annotate(
        "", xy=(e_j[k], cbar_val), xytext=(e_j[k], A[k]),
        arrowprops=dict(arrowstyle="<->", color=PRIMARY_BLUE,
                        lw=1.1, shrinkA=0, shrinkB=0),
        zorder=6,
    )
    ax.text(e_j[k] * 1.18, (cbar_val + A[k]) / 2.0,
            rf"$\mathrm{{slack}}_i = {cbar_val - A[k]:.2f}$",
            fontsize=7.5, color=PRIMARY_BLUE, va="center", ha="left",
            zorder=7)

    # (5) Formula annotation: single centered callout at the top of the
    #     wedge, just under the DP ceiling.
    ax.text(0.5, 0.86,
            r"$\mathrm{slack}_i(\varepsilon_i,\varepsilon_j)\;=\;"
            r"\bar c(\varepsilon_i)\;-\;A_{\mathrm{PLD},i}(\varepsilon_i,\varepsilon_j)\;\geq\;0$",
            transform=ax.transAxes, fontsize=7.5, color=PRIMARY_BLUE,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=PRIMARY_BLUE, lw=0.4, alpha=0.95),
            zorder=8)

    ax.set_xscale("log")
    ax.set_xlim(0.5, 64.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"other player's privacy budget $\varepsilon_j$")
    ax.set_ylabel(r"MIA advantage against player $i$")
    # Legend inside the wedge on the left, where the wedge has the most
    # empty space (A_i is <=0.18 there; c_bar sits at 0.96).
    leg = ax.legend(
        frameon=True, fontsize=7, loc="center left",
        bbox_to_anchor=(0.02, 0.48),
        handlelength=2.0, borderaxespad=0.3, handletextpad=0.5,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)
    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")


if __name__ == "__main__":
    main()
