"""T24 -- publication figure for H1 (measured LiRA NE on Adult, n=2).

Reads `_cache_H1/H1_summary.pkl` (from `24_lira_analysis.py --collect`) and
the per-record LiRA scores at a representative profile, and emits
`paper/figures/fig_H1_n2_measured_NE.{pdf,svg}` in the style of
`paper/figures/fig_h8_brpeg_regret.pdf`.

Three panels:
  (a) Aggregate LiRA ROC per silo at pid=24 (eps=(32,32)), the high-eps
      corner where the analytical A^PLD is largest, so the leakage signal
      LiRA is trying to detect is at its ceiling.
  (b) Measured advantage heatmap A^LiRA_1 with the measured pure NE
      cell(s) starred and the analytical PLD NE outlined.
  (c) A^LiRA vs A^PLD scatter across all 25 profiles x 2 silos with the
      y = x reference. The gap on this scatter is the empirical
      privacy-onion.

Run:
    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/24_paper_figure.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (  # noqa: E402
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    two_column_size,
)

CACHE = ROOT / "game_experiments" / "_cache_H1"
CACHE_PROFILES = CACHE / "n2_adult"
FIG_PDF = ROOT / "paper" / "figures" / "fig_H1_n2_measured_NE.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_H1_n2_measured_NE.svg"

# Colours consistent with fig_h8_brpeg_regret and other paper figures.
SILO_COLORS = (PRIMARY_BLUE, PRIMARY_ORANGE)
NE_MEAS_COLOR = PRIMARY_ORANGE
NE_PLD_COLOR = "#41ab5d"        # green -- distinct from the blue/orange scheme
DIAG_COLOR = ACCENT_GREY


def _logit_score(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    n_records = logits.shape[1]
    idx = np.arange(n_records)
    p_true = p[:, idx, labels[idx]]
    p_true = np.clip(p_true, 1e-6, 1.0 - 1e-6)
    return np.log(p_true) - np.log1p(-p_true)


def _profile_roc(pid: int, n_fpr_points: int = 400):
    """Aggregate offline-fixed-variance LiRA ROC per silo at one profile.

    Returns (fpr_grid, tpr_silo0, tpr_silo1, auc_per_silo).
    Uses the same scoring as `_lira_offline` in 24_lira_analysis.py: for each
    record fit an OUT Gaussian; per-target score = -log pdf(phi | OUT). Then
    pool records within each silo and sweep a single threshold.
    """
    pdir = CACHE_PROFILES / str(pid)
    shadow_dirs = sorted(d for d in pdir.iterdir() if d.is_dir())
    logits = np.stack([np.load(d / "logits.npy") for d in shadow_dirs])
    if logits.ndim == 4:
        logits = logits.mean(axis=2)
    keep = np.stack([np.load(d / "keep.npy") for d in shadow_dirs])
    labels = np.load(pdir / "labels.npy").astype(int)
    silo_id = np.load(pdir / "silo_id.npy")

    phi = _logit_score(logits, labels)
    n_shadows, n_records = phi.shape

    mu_out = np.array([
        np.median(phi[~keep[:, r], r]) if (~keep[:, r]).sum() >= 2 else 0.0
        for r in range(n_records)
    ])
    pooled = np.concatenate([
        phi[~keep[:, r], r] - mu_out[r] for r in range(n_records)
    ])
    sigma_out = max(pooled.std(), 1e-6)
    # -log pdf under N(mu_out, sigma_out) per (shadow, record).
    z = (phi - mu_out[None, :]) / sigma_out
    scores = 0.5 * z ** 2 + np.log(sigma_out) + 0.5 * np.log(2 * np.pi)

    fpr_grid = np.logspace(-4, 0, n_fpr_points)
    tprs = []
    aucs = []
    for g in (0, 1):
        sel = silo_id == g
        s = scores[:, sel].ravel()
        y = keep[:, sel].ravel().astype(int)
        order = np.argsort(-s)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / max(tp[-1], 1)
        fpr = fp / max(fp[-1], 1)
        # Interpolate onto fpr_grid.
        tpr_g = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
        tprs.append(tpr_g)
        # AUC via trapezoid on the (fpr, tpr) with 0/1 anchors.
        fpr_a = np.concatenate([[0.0], fpr, [1.0]])
        tpr_a = np.concatenate([[0.0], tpr, [1.0]])
        aucs.append(float(np.trapz(tpr_a, fpr_a)))
    return fpr_grid, tprs[0], tprs[1], np.array(aucs)


def _annotate_cell(ax, cell, marker, color, size, edge=1.2, label=None):
    i, j = cell
    ax.plot(j, i, marker=marker, markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=edge, markersize=size,
            linestyle="", label=label)


def main() -> None:
    with open(CACHE / "H1_summary.pkl", "rb") as f:
        S = pickle.load(f)

    A_meas = S["A_meas"]              # (K, K, 2)
    A_pld = S["A_pld"]
    V_meas = S["V_meas"]
    U1_meas = S["U1_meas"]
    U2_meas = S["U2_meas"]
    ne_meas = S["ne_meas"]
    ne_pld = S["ne_pld"]
    eps_grid = S["eps_grid"]
    K = len(eps_grid)

    # Representative profile: eps=(32, 32) at pid = 4*K+4 = 24, the high-eps
    # corner where LiRA has the strongest chance of resolving the leakage.
    rep_i, rep_j = K - 1, K - 1
    rep_pid = rep_i * K + rep_j
    fpr, tpr0, tpr1, aucs = _profile_roc(rep_pid)

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.36))

    # ---- Panel (a): aggregate LiRA ROC per silo at representative profile.
    ax = axes[0]
    ax.plot(fpr, tpr0, "-", color=SILO_COLORS[0], lw=1.4,
            label=r"silo 1 (AUC $= %.3f$)" % aucs[0])
    ax.plot(fpr, tpr1, "-", color=SILO_COLORS[1], lw=1.4,
            label=r"silo 2 (AUC $= %.3f$)" % aucs[1])
    ax.plot([1e-4, 1.0], [1e-4, 1.0], "--", color=DIAG_COLOR, lw=0.6,
            label="chance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(r"(a) LiRA ROC at $\varepsilon=(%g,%g)$" %
                 (eps_grid[rep_i], eps_grid[rep_j]))
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    # ---- Panel (b): A^LiRA_1 heatmap with NE overlays.
    ax = axes[1]
    im = ax.imshow(A_meas[..., 0], origin="lower", cmap="magma",
                   aspect="equal")
    ax.set_xticks(range(K), [f"{e:g}" for e in eps_grid])
    ax.set_yticks(range(K), [f"{e:g}" for e in eps_grid])
    ax.set_xlabel(r"$\varepsilon_2$")
    ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(r"(b) Measured $A_1^{\mathrm{LiRA}}$ and NE")
    for k, cell in enumerate(ne_meas):
        _annotate_cell(ax, cell, marker="*", color=NE_MEAS_COLOR,
                       size=13, edge=1.8,
                       label="measured NE" if k == 0 else None)
    for k, cell in enumerate(ne_pld):
        _annotate_cell(ax, cell, marker="s", color=NE_PLD_COLOR,
                       size=11, edge=1.6,
                       label="PLD NE" if k == 0 else None)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r"$A_1^{\mathrm{LiRA}}$")
    cb.outline.set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left",
              handletextpad=0.3, borderpad=0.3)

    # ---- Panel (c): A^LiRA vs A^PLD scatter (privacy-onion gap).
    ax = axes[2]
    x_max = float(np.nanmax(A_pld))
    ax.plot([0.0, x_max * 1.05], [0.0, x_max * 1.05], "--",
            color=DIAG_COLOR, lw=0.6, label=r"$y = x$")
    ax.scatter(A_pld[..., 0].ravel(), A_meas[..., 0].ravel(),
               s=20, color=SILO_COLORS[0], alpha=0.85, edgecolor="none",
               label="silo 1")
    ax.scatter(A_pld[..., 1].ravel(), A_meas[..., 1].ravel(),
               s=20, color=SILO_COLORS[1], alpha=0.85, edgecolor="none",
               label="silo 2")
    ax.set_xlim(0.0, x_max * 1.05)
    ax.set_ylim(0.0, x_max * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"analytical $A_{\mathrm{PLD}}$")
    ax.set_ylabel(r"measured $A^{\mathrm{LiRA}}$")
    ax.set_title(r"(c) Privacy-onion gap")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    # -------- Numeric summary --------
    # LiRA AUC per silo across all 25 profiles (used for the doc).
    per_profile_auc = np.zeros((K, K, 2))
    for i in range(K):
        for j in range(K):
            pid = i * K + j
            try:
                _, _, _, a = _profile_roc(pid, n_fpr_points=200)
                per_profile_auc[i, j] = a
            except FileNotFoundError:
                per_profile_auc[i, j] = np.nan

    # Shadow-model variance of A_j across records within silo.
    A_per_record_stderr = np.zeros((K, K, 2))
    for i in range(K):
        for j in range(K):
            pid = i * K + j
            arr = np.load(CACHE_PROFILES / str(pid) / "lira_scores.npz",
                          allow_pickle=True)
            A_per = arr["A_per_record"]
            silo = arr["silo_id"]
            for g in (0, 1):
                sel = silo == g
                A_per_record_stderr[i, j, g] = A_per[sel].std() / np.sqrt(sel.sum())

    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")
    print(f"measured NE cells (log-eps indices): {ne_meas}")
    print(f"PLD NE cells: {ne_pld}")
    for cell in ne_meas:
        i, j = cell
        print(
            f"  measured NE eps=({eps_grid[i]:g},{eps_grid[j]:g}) "
            f"A1={A_meas[i,j,0]:.4f}+/-{A_per_record_stderr[i,j,0]:.4f} "
            f"A2={A_meas[i,j,1]:.4f}+/-{A_per_record_stderr[i,j,1]:.4f} "
            f"V={V_meas[i,j]:.4f}"
        )
    for cell in ne_pld:
        i, j = cell
        print(
            f"  PLD NE eps=({eps_grid[i]:g},{eps_grid[j]:g}) "
            f"A1_pld={A_pld[i,j,0]:.4f} A2_pld={A_pld[i,j,1]:.4f}"
        )
    print(f"AUC at rep profile pid={rep_pid}: silo1={aucs[0]:.4f} silo2={aucs[1]:.4f}")
    print(f"AUC range across grid: silo1 [{np.nanmin(per_profile_auc[...,0]):.4f}, "
          f"{np.nanmax(per_profile_auc[...,0]):.4f}] "
          f"silo2 [{np.nanmin(per_profile_auc[...,1]):.4f}, "
          f"{np.nanmax(per_profile_auc[...,1]):.4f}]")
    print(f"A_meas range: silo1 [{np.nanmin(A_meas[...,0]):.4f}, "
          f"{np.nanmax(A_meas[...,0]):.4f}] "
          f"silo2 [{np.nanmin(A_meas[...,1]):.4f}, "
          f"{np.nanmax(A_meas[...,1]):.4f}]")
    print(f"A_pld range: silo1 [{np.nanmin(A_pld[...,0]):.4f}, "
          f"{np.nanmax(A_pld[...,0]):.4f}] "
          f"silo2 [{np.nanmin(A_pld[...,1]):.4f}, "
          f"{np.nanmax(A_pld[...,1]):.4f}]")
    print(f"V_meas range: [{np.nanmin(V_meas):.4f}, {np.nanmax(V_meas):.4f}]")

    # Save summary for the doc.
    summary = dict(
        ne_meas=ne_meas, ne_pld=ne_pld,
        A_meas=A_meas, A_pld=A_pld, V_meas=V_meas,
        A_per_record_stderr=A_per_record_stderr,
        per_profile_auc=per_profile_auc,
        rep_pid=rep_pid, rep_eps=(float(eps_grid[rep_i]), float(eps_grid[rep_j])),
        rep_auc=(float(aucs[0]), float(aucs[1])),
    )
    with open(CACHE / "H1_figure_summary.pkl", "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()
