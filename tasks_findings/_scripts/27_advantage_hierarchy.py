"""T27 -- the three-level advantage hierarchy, on one measured instance.

Builds the conceptual companion figure for Sec. II-C of main2:

  A_{eps,delta}  >=  A_PLD  >~  A_LiRA

on the H1 profile (eps_1, eps_2) = (16, 16) of the measured Adult instance
(sigma = 0.8027, p = 0.01294, I = 4680 steps, delta = 1e-12):

  panel (a): privacy profiles delta(eps) -- the worst-case profile consistent
             with knowing ONLY the calibration point (Kairouz-Oh-Viswanath
             dominating pair), versus the actual PLD profile of the mechanism;
             both pass through the calibration point.
  panel (b): trade-off curves -- the single-point (eps,delta) trade-off,
             the exact PLD trade-off, and the pooled empirical LiRA ROC of
             the trained shadow models; the three advantages are the maximal
             vertical gaps to the no-leakage diagonal.

Outputs:
  tasks_findings/_scripts/27_hierarchy_summary.txt
  paper/figures/fig_advantage_hierarchy.{pdf,svg}
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp_accounting import pld as dp_pld
from game.advantage import trade_off_function_from_pld
from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    column_size,
)

PID = 18  # (16, 16)
PDIR = ROOT / "game_experiments" / "_cache_H1" / "n2_adult" / str(PID)
GRID = ROOT / "game_experiments" / "_cache_H1" / "mechanism_grid.pkl"
STEPS = 4680
DELTA0 = 1e-12
FIG_PDF = ROOT / "paper" / "figures" / "fig_advantage_hierarchy.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_advantage_hierarchy.svg"
SUMMARY = ROOT / "tasks_findings" / "_scripts" / "27_hierarchy_summary.txt"

# import the LiRA scoring helpers from the analysis script
spec = importlib.util.spec_from_file_location(
    "lira24", ROOT / "tasks_findings" / "_scripts" / "24_lira_analysis.py")
lira24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lira24)


def wc_profile(eps0: float, delta0: float, eps_grid: np.ndarray) -> np.ndarray:
    """delta(eps) of the Kairouz-Oh-Viswanath dominating pair for (eps0, delta0):
    the largest profile any single-point-(eps0,delta0) mechanism can have."""
    e0 = np.exp(eps0)
    P = np.array([delta0, (1 - delta0) * e0 / (1 + e0), (1 - delta0) / (1 + e0), 0.0])
    Q = np.array([0.0, (1 - delta0) / (1 + e0), (1 - delta0) * e0 / (1 + e0), delta0])
    return np.array([np.maximum(P - np.exp(e) * Q, 0.0).sum() for e in eps_grid])


def wc_tradeoff(eps0: float, delta0: float, alpha: np.ndarray) -> np.ndarray:
    """f_{eps0,delta0}(alpha): trade-off of the single-point guarantee."""
    e0 = np.exp(eps0)
    return np.maximum.reduce([
        np.zeros_like(alpha),
        1 - delta0 - e0 * alpha,
        (1 - delta0 - alpha) / e0,
    ])


def main():
    out = []
    say = lambda s: (print(s), out.append(s))

    grid = pickle.load(open(GRID, "rb"))
    prof = grid[PID]
    sigma, p = float(prof["sigma"]), float(prof["p1"])
    eps0 = float(prof["eps_realised"][0])
    say(f"# T27 advantage hierarchy on H1 profile {PID} "
        f"(eps=({prof['eps'][0]:g},{prof['eps'][1]:g}))")
    say(f"mechanism: sigma={sigma:.4f} p={p:.5f} I={STEPS} delta={DELTA0:g} "
        f"eps_realised={eps0:.4f}")

    # ---- Level 2: PLD ---------------------------------------------------
    pld_obj = dp_pld.privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma, sampling_prob=p,
        value_discretization_interval=1e-3, use_connect_dots=True,
    ).self_compose(STEPS)

    alpha = np.concatenate([
        np.geomspace(1e-8, 1e-2, 200), np.linspace(1e-2, 1 - 1e-4, 400)])
    f_pld = trade_off_function_from_pld(pld_obj, alpha)
    A_pld = float(np.max(1 - alpha - f_pld))

    # ---- Level 1: single calibration point ------------------------------
    f_wc = wc_tradeoff(eps0, DELTA0, alpha)
    A_wc = (np.exp(eps0) - 1 + 2 * DELTA0) / (np.exp(eps0) + 1)

    # ---- Level 3: pooled LiRA ROC on the trained shadow models ----------
    shadow_dirs = sorted(d for d in PDIR.iterdir()
                         if d.is_dir() and d.name.startswith("shadow"))
    logits = np.stack([np.load(d / "logits.npy") for d in shadow_dirs])
    if logits.ndim == 4:
        logits = logits.mean(axis=2)
    keep = np.stack([np.load(d / "keep.npy") for d in shadow_dirs])
    labels = np.load(PDIR / "labels.npy")
    phi = lira24._logit_score(logits, labels.astype(int))

    from scipy import stats
    S, N = phi.shape
    mu_out = np.array([
        np.median(phi[~keep[:, r], r]) if (~keep[:, r]).sum() >= 2 else 0.0
        for r in range(N)])
    pooled = np.concatenate([phi[~keep[:, r], r] - mu_out[r] for r in range(N)])
    sig_out = max(pooled.std(), 1e-6)
    scores = -stats.norm.logpdf(phi, loc=mu_out[None, :], scale=sig_out)
    y = keep.astype(int).ravel()
    s = scores.ravel()
    order = np.argsort(-s)
    y = y[order]
    tpr = np.cumsum(y) / y.sum()
    fpr = np.cumsum(1 - y) / (1 - y).sum()
    A_lira_pooled = float((tpr - fpr).max())
    auc = float(np.trapz(tpr, fpr))
    A_per_record = float(np.load(PDIR / "A_measured.npy").mean())

    say(f"A_(eps,delta) (worst case at contract) = {A_wc:.6f}")
    say(f"A_PLD (optimal adversary, modelled mechanism) = {A_pld:.4f}")
    say(f"A_LiRA pooled-ROC advantage = {A_lira_pooled:.4f} (AUC {auc:.4f})")
    say(f"A_LiRA mean per-record advantage (payoff-matrix statistic) = "
        f"{A_per_record:.4f}")

    # ---- figure ----------------------------------------------------------
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=column_size(rows=1, aspect=0.46),
                             gridspec_kw=dict(wspace=0.5))

    # (a) privacy profiles
    ax = axes[0]
    eps_grid = np.linspace(0, eps0 * 1.15, 300)
    d_wc = wc_profile(eps0, DELTA0, eps_grid)
    d_pld = np.array([pld_obj.get_delta_for_epsilon(float(e)) for e in eps_grid])
    floor = 1e-14
    ax.semilogy(eps_grid, np.maximum(d_wc, floor), "--", color=ACCENT_GREY,
                lw=1.1, label="admitted by $(\\varepsilon,\\delta)$ alone")
    ax.semilogy(eps_grid, np.maximum(d_pld, floor), "-", color=PRIMARY_BLUE,
                lw=1.3, label="actual PLD profile")
    ax.plot([eps0], [max(DELTA0, floor)], "o", color=PRIMARY_ORANGE,
            markersize=5, zorder=5, label="contract $(\\varepsilon,\\delta)$")
    ax.set_ylim(floor, 3)
    ax.set_xlim(-0.3, eps_grid[-1])
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\delta(\varepsilon)$")
    ax.set_title("(a) one point, two curves", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="lower left",
              borderaxespad=0.3, handlelength=1.6)

    # (b) trade-off curves + advantage gaps
    ax = axes[1]
    ax.plot([0, 1], [1, 0], "-", color="black", lw=0.7,
            label="no leakage")
    ax.plot(alpha, f_wc, "--", color=ACCENT_GREY, lw=1.1,
            label=r"$(\varepsilon,\delta)$ worst case")
    ax.plot(alpha, f_pld, "-", color=PRIMARY_BLUE, lw=1.3,
            label="PLD (optimal attack)")
    ax.plot(fpr, 1 - tpr, "-", color=PRIMARY_ORANGE, lw=1.1,
            label="LiRA (observed)")

    # advantage gap annotations (manual placement for legibility)
    # A_{eps,delta}: gap ~ full height, drawn slightly inside the axis
    a0 = 0.015
    ax.annotate("", xy=(a0, float(np.interp(a0, alpha, f_wc))),
                xytext=(a0, 1 - a0),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_GREY, lw=0.9))
    ax.text(0.045, 0.40, rf"$A_{{\varepsilon,\delta}} = {A_wc:.2f}$",
            color=ACCENT_GREY, fontsize=6.5, va="center", rotation=90)
    # A_PLD at its argmax
    k = int(np.argmax(1 - alpha - f_pld))
    a_k = float(alpha[k])
    ax.annotate("", xy=(a_k, f_pld[k]), xytext=(a_k, 1 - a_k),
                arrowprops=dict(arrowstyle="<->", color=PRIMARY_BLUE, lw=0.9))
    ax.text(a_k + 0.035, (f_pld[k] + 1 - a_k) / 2,
            rf"$A_{{\mathrm{{PLD}}}} = {A_pld:.2f}$",
            color=PRIMARY_BLUE, fontsize=6.5, va="center")
    # A_LiRA: gap invisibly small; label the curve with a leader
    ax.annotate(rf"$A_{{\mathrm{{LiRA}}}} = {A_lira_pooled:.2f}$",
                xy=(0.74, float(np.interp(0.74, fpr, 1 - tpr))),
                xytext=(0.55, 0.12), color=PRIMARY_ORANGE, fontsize=6.5,
                arrowprops=dict(arrowstyle="->", color=PRIMARY_ORANGE,
                                lw=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"type-I error $\alpha$")
    ax.set_ylabel(r"type-II error $T(\alpha)$")
    ax.set_title("(b) three advantages", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")

    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    fig.savefig(FIG_PDF.with_suffix(".png"), dpi=300)
    say(f"wrote {FIG_PDF}")
    SUMMARY.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
