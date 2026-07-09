"""T25 -- placeholder measurement-ceiling figure for the toy federated n=3 PEG.

Path (A) choice from the T25 finalize brief: no Slurm submission of the n=3
Adult-MLP federated array. Rationale documented in
`tasks_findings/25_toy_federated_n3_shadow_model.md`: T24's finalize
established that LiRA on Adult MLP saturates at the empirical noise floor
(A^LiRA in [0.107, 0.115], per-silo AUC in [0.501, 0.506]) across all 25
profiles of the 5x5 log-eps grid. Because the T25 pipeline reuses T24's
trainer, mechanism, and LiRA scoring verbatim on the same dataset, a
federated sibling on Adult would inherit the same noise ceiling and
produce no distinguishable per-silo signal. This figure documents the
ceiling honestly from the T24 evidence and marks Purchase-100 (not yet
wired into `experiment_scripts/mia/utils/utils_general.py`)
as the natural next-step benchmark.

Reads:
    game_experiments/_cache_H1/H1_summary.pkl         (T24 n=2 measured)
    game_experiments/_cache_H1/H1_figure_summary.pkl  (T24 per-profile AUC)
    game_experiments/_cache_T25/mechanism_grid.pkl    (T25 n=3 mechanism)

Writes:
    paper/figures/fig_C9_federated_measurement_ceiling.pdf
    paper/figures/fig_C9_federated_measurement_ceiling.svg
    game_experiments/_cache_T25/T25_ceiling_summary.pkl

Run:
    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/25_paper_figure.py
"""
from __future__ import annotations

import pickle
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (  # noqa: E402
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    two_column_size,
)
from game.advantage import mia_advantage_subsampled_gaussian  # noqa: E402

H1_CACHE = ROOT / "game_experiments" / "_cache_H1"
T25_CACHE = ROOT / "game_experiments" / "_cache_T25"
FIG_PDF = ROOT / "paper" / "figures" / "fig_C9_federated_measurement_ceiling.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_C9_federated_measurement_ceiling.svg"

SILO_COLORS = (PRIMARY_BLUE, PRIMARY_ORANGE)
CEILING_COLOR = "#c2185b"
DIAG_COLOR = ACCENT_GREY

# ---------------------------------------------------------------------------
# Analytical n=3 PLD payoff cube, from the T25 mechanism grid.
# ---------------------------------------------------------------------------
EPS_GRID_N3 = np.array([2.0, 8.0, 32.0])
K3 = len(EPS_GRID_N3)
# T25 mechanism cache stores STEPS implicitly via (sigma, p_g); the analytical
# advantage function is the same as game/advantage.py. STEPS = 60 * (30000/128).
STEPS_N3 = 60 * (30000 // 128)


def _pld_cube_n3():
    with open(T25_CACHE / "mechanism_grid.pkl", "rb") as f:
        grid = pickle.load(f)

    A = np.full((K3, K3, K3, 3), np.nan)
    V_proxy = np.full((K3, K3, K3), np.nan)
    feasible = np.zeros((K3, K3, K3), dtype=bool)
    for row in grid:
        i, j, k = row["ijk"]
        feasible[i, j, k] = row["feasible"]
        if not row["feasible"]:
            continue
        A[i, j, k, 0] = mia_advantage_subsampled_gaussian(row["sigma"], row["p1"], STEPS_N3)
        A[i, j, k, 1] = mia_advantage_subsampled_gaussian(row["sigma"], row["p2"], STEPS_N3)
        A[i, j, k, 2] = mia_advantage_subsampled_gaussian(row["sigma"], row["p3"], STEPS_N3)
        # V is silo-agnostic; proxy by 1 / (1 + sigma), monotone-decreasing in
        # sigma, purely for the NE search visualisation. The measured V for
        # Adult MLP in T24 sits in [0.804, 0.826] across the grid -- a narrow
        # band -- so the qualitative NE structure is A-driven; we make this
        # explicit by using a sigma-monotone proxy rather than a fitted V.
        V_proxy[i, j, k] = 1.0 / (1.0 + row["sigma"])
    return A, V_proxy, feasible


def _find_pure_ne_cube(U: np.ndarray) -> list:
    ne = []
    for i, j, k in product(range(K3), repeat=3):
        if np.isnan(U[i, j, k, 0]):
            continue
        col0 = U[:, j, k, 0]
        col1 = U[i, :, k, 1]
        col2 = U[i, j, :, 2]
        if not np.all(np.isnan(col0)) and U[i, j, k, 0] >= np.nanmax(col0) - 1e-9 \
                and not np.all(np.isnan(col1)) and U[i, j, k, 1] >= np.nanmax(col1) - 1e-9 \
                and not np.all(np.isnan(col2)) and U[i, j, k, 2] >= np.nanmax(col2) - 1e-9:
            ne.append((i, j, k))
    return ne


def main() -> None:
    # -------- T24 n=2 measured evidence for the ceiling narrative --------
    with open(H1_CACHE / "H1_summary.pkl", "rb") as f:
        S = pickle.load(f)
    with open(H1_CACHE / "H1_figure_summary.pkl", "rb") as f:
        Sf = pickle.load(f)
    A_meas = S["A_meas"]                    # (5, 5, 2) LiRA advantage
    A_pld = S["A_pld"]                      # (5, 5, 2) PLD upper bound
    per_profile_auc = Sf["per_profile_auc"]  # (5, 5, 2) per-silo LiRA AUC

    # -------- T25 n=3 analytical PLD cube --------
    A_pld_n3, V_proxy_n3, feasible_n3 = _pld_cube_n3()
    U_pld_n3 = np.stack([V_proxy_n3 - A_pld_n3[..., g] for g in range(3)], axis=-1)
    ne_pld_n3 = _find_pure_ne_cube(U_pld_n3)

    # If the pipeline were run and inherited the T24 ceiling
    # (mean(A^LiRA over T24) at each silo), the measured payoff cube would be
    # V_proxy - <A^LiRA>_g, i.e. an additive constant across all cells for
    # each silo. Best response = argmax over V (i.e. minimum sigma / highest
    # eps corner). Compute the predicted noise-ceiling NE for the caption.
    A_ceiling_est = np.array([A_meas[..., 0].mean(),
                              A_meas[..., 1].mean(),
                              A_meas[..., 1].mean()])   # replicate silo 2 -> silo 3
    U_ceiling_n3 = np.stack(
        [V_proxy_n3 - A_ceiling_est[g] * np.ones_like(V_proxy_n3)
         for g in range(3)], axis=-1
    )
    U_ceiling_n3[~feasible_n3] = np.nan
    ne_ceiling_n3 = _find_pure_ne_cube(U_ceiling_n3)

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.36))

    # ---- Panel (a): T24 n=2 measured LiRA vs PLD; the ceiling as a scatter.
    ax = axes[0]
    x_max = float(np.nanmax(A_pld))
    ax.plot([0.0, x_max * 1.05], [0.0, x_max * 1.05], "--",
            color=DIAG_COLOR, lw=0.7, label=r"$y=x$")
    # Draw the empirical noise ceiling as a horizontal band across all
    # profiles: [min, max] of measured A across silos.
    lo = float(min(A_meas[..., 0].min(), A_meas[..., 1].min()))
    hi = float(max(A_meas[..., 0].max(), A_meas[..., 1].max()))
    ax.axhspan(lo, hi, color=CEILING_COLOR, alpha=0.15,
               label=r"measurement ceiling $A^{\mathrm{LiRA}}\!\in\![%.3f,%.3f]$" % (lo, hi))
    ax.scatter(A_pld[..., 0].ravel(), A_meas[..., 0].ravel(),
               s=18, color=SILO_COLORS[0], alpha=0.85, edgecolor="none",
               label="silo 1 (n=2)")
    ax.scatter(A_pld[..., 1].ravel(), A_meas[..., 1].ravel(),
               s=18, color=SILO_COLORS[1], alpha=0.85, edgecolor="none",
               label="silo 2 (n=2)")
    ax.set_xlim(0.0, x_max * 1.05)
    ax.set_ylim(0.0, x_max * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"analytical $A^{\mathrm{PLD}}$")
    ax.set_ylabel(r"measured $A^{\mathrm{LiRA}}$")
    ax.set_title(r"(a) Adult MLP measurement ceiling ($n{=}2$)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    # ---- Panel (b): per-silo LiRA AUC distribution across the 25 profiles.
    ax = axes[1]
    auc_flat = per_profile_auc.reshape(-1, 2)
    bins = np.linspace(0.495, 0.510, 20)
    ax.hist(auc_flat[:, 0], bins=bins, color=SILO_COLORS[0], alpha=0.75,
            label=f"silo 1 (mean {auc_flat[:,0].mean():.4f})", edgecolor="white",
            linewidth=0.4)
    ax.hist(auc_flat[:, 1], bins=bins, color=SILO_COLORS[1], alpha=0.6,
            label=f"silo 2 (mean {auc_flat[:,1].mean():.4f})", edgecolor="white",
            linewidth=0.4)
    ax.axvline(0.5, ls="--", color=DIAG_COLOR, lw=0.7, label="chance")
    ax.set_xlabel(r"per-silo LiRA AUC")
    ax.set_ylabel(r"profiles (of $5{\times}5=25$)")
    ax.set_title(r"(b) LiRA AUC hugs chance across all profiles")
    ax.legend(frameon=False, fontsize=6, loc="upper right")

    # ---- Panel (c): analytical n=3 PLD NE cube -- where the signal would be,
    # if measurable. Three side-by-side heatmaps of A_1^PLD at each eps_3.
    ax = axes[2]
    ax.set_axis_off()
    subgrid = ax.inset_axes([0.0, 0.15, 1.0, 0.75])
    subgrid.set_axis_off()
    subgrid.set_title(r"(c) Analytical PLD $A_1^{\mathrm{PLD}}$ over $(\varepsilon_1,\varepsilon_2)$,"
                      r" per slab $\varepsilon_3$", pad=6)

    axc = [ax.inset_axes([0.02 + 0.32 * i, 0.20, 0.28, 0.60]) for i in range(3)]
    vmax = np.nanmax(A_pld_n3[..., 0])
    for s, e3 in enumerate(EPS_GRID_N3):
        slab = A_pld_n3[:, :, s, 0]  # (K3, K3)
        im = axc[s].imshow(slab, origin="lower", cmap="magma",
                           vmin=0.0, vmax=vmax, aspect="equal")
        axc[s].set_xticks(range(K3), [f"{e:g}" for e in EPS_GRID_N3], fontsize=6)
        axc[s].set_yticks(range(K3), [f"{e:g}" for e in EPS_GRID_N3], fontsize=6)
        axc[s].set_title(rf"$\varepsilon_3{{=}}{e3:g}$", fontsize=7, pad=2)
        # Star each PLD NE cell whose eps_3 matches this slab.
        for (i, j, k) in ne_pld_n3:
            if k == s:
                axc[s].plot(j, i, marker="s", markerfacecolor="none",
                            markeredgecolor="#41ab5d", markeredgewidth=1.6,
                            markersize=10, linestyle="")
        # Mark predicted ceiling NE (measured pipeline, if run) as star.
        for (i, j, k) in ne_ceiling_n3:
            if k == s:
                axc[s].plot(j, i, marker="*", markerfacecolor="none",
                            markeredgecolor=CEILING_COLOR, markeredgewidth=1.6,
                            markersize=12, linestyle="")
        # Grey-out infeasible cells with a hatched patch.
        for ii in range(K3):
            for jj in range(K3):
                if not feasible_n3[ii, jj, s]:
                    axc[s].add_patch(plt.Rectangle(
                        (jj - 0.5, ii - 0.5), 1, 1, hatch="////",
                        edgecolor="white", facecolor="none", linewidth=0.0
                    ))
    # Colorbar under the three panels.
    cax = ax.inset_axes([0.10, 0.06, 0.80, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$A_1^{\mathrm{PLD}}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    # -------- Numeric summary --------
    summary = dict(
        A_lira_range_silo=[float(A_meas[..., 0].min()), float(A_meas[..., 0].max()),
                           float(A_meas[..., 1].min()), float(A_meas[..., 1].max())],
        A_pld_range_silo=[float(A_pld[..., 0].min()), float(A_pld[..., 0].max()),
                          float(A_pld[..., 1].min()), float(A_pld[..., 1].max())],
        auc_range=[float(per_profile_auc[..., 0].min()),
                   float(per_profile_auc[..., 0].max()),
                   float(per_profile_auc[..., 1].min()),
                   float(per_profile_auc[..., 1].max())],
        A_ceiling_est=A_ceiling_est.tolist(),
        ne_pld_n3=ne_pld_n3,
        ne_ceiling_n3=ne_ceiling_n3,
        eps_grid_n3=EPS_GRID_N3.tolist(),
        feasible_n3_count=int(feasible_n3.sum()),
    )
    with open(T25_CACHE / "T25_ceiling_summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")
    print(f"A^LiRA range across n=2 grid: "
          f"silo1=[{summary['A_lira_range_silo'][0]:.4f}, {summary['A_lira_range_silo'][1]:.4f}] "
          f"silo2=[{summary['A_lira_range_silo'][2]:.4f}, {summary['A_lira_range_silo'][3]:.4f}]")
    print(f"A^PLD range across n=2 grid: "
          f"silo1=[{summary['A_pld_range_silo'][0]:.4f}, {summary['A_pld_range_silo'][1]:.4f}]")
    print(f"AUC range: silo1=[{summary['auc_range'][0]:.4f}, {summary['auc_range'][1]:.4f}] "
          f"silo2=[{summary['auc_range'][2]:.4f}, {summary['auc_range'][3]:.4f}]")
    print(f"analytical PLD NE (n=3, K=3): {ne_pld_n3}")
    print(f"predicted ceiling NE (n=3, K=3, if pipeline run at Adult noise floor): "
          f"{ne_ceiling_n3}")
    print(f"feasible n=3 cells: {summary['feasible_n3_count']}/27")


if __name__ == "__main__":
    main()
