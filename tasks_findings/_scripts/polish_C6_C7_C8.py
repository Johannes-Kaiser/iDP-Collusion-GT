"""Polish C6/C7/C8 K=200 figures.

Purpose
-------
The auto-regenerated K=200 figures (`fig_C6_locus_K200`, `fig_C7_w_ratio`,
`fig_C8_asymmetric_w`) suffered from three visual defects that made them
unreadable in the paper:

  * fig_C6 left panel: the discrete-argmax best-response curves were
    jagged/spiky, and 48 NE stars formed an illegible cloud on the
    diagonal.
  * fig_C7: 240 NE points across six weights piled up in the same
    log-log region with bleeding Blues-palette colors.
  * fig_C8: same problem with the Oranges palette across five ratios.

This script reads the real caches at

    game_experiments/_cache_v2/{C6,C7,C8}_K200.pkl

and produces the polished figures at

    paper/figures/fig_C6_locus_K200.{pdf,svg}
    paper/figures/fig_C7_w_ratio.{pdf,svg}
    paper/figures/fig_C8_asymmetric_w.{pdf,svg}

Design choices
--------------
1. BR curves: running median (window=7) on log(actions[BR_i(j)]) as a
   function of log(actions[j]). The median filter removes single-pixel
   ties in the argmax without distorting the monotone envelope.

2. NE clustering: grid-quantize NEs in log-log eps space into cells of
   0.15 decades and take per-cell centroids. This mirrors what a reader
   would perceive as "one NE" at print scale and matches the resolution
   of the K=200 grid (log-step is 0.021 decades, so a 0.15-decade cell
   fuses roughly 7 adjacent argmax cells -- the typical extent of a
   BR-intersection cluster).

3. Perceptually-uniform colormaps: `viridis` for C7's symmetric weight
   sweep (ordered by w), `cividis` for C8's asymmetric ratio (ordered
   by log2 of the ratio, centred at 0 = symmetric). Both are
   color-vision-safe and photocopy-robust.

4. Dispersion shading: 68% inertia region per weight, computed from the
   covariance of the log-log NE cloud and rendered as a semi-transparent
   ellipse. Falls back to a small circle if fewer than 3 NEs remain.

5. Style: `game.style.apply_paper_style` (matches `fig_h8_brpeg_regret`).
   Two-column width for C6 (two panels), single-column for C7/C8.

Reproduce
---------
    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \
        tasks_findings/_scripts/polish_C6_C7_C8.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.ndimage import median_filter

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size, two_column_size,
)

CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR = ROOT / "paper" / "figures"

apply_paper_style()

# --- Utilities -------------------------------------------------------------

def smooth_br(br_idx: np.ndarray, actions: np.ndarray, window: int = 7):
    """Return smoothed (x_eps, y_eps) for a BR curve.

    br_idx[j] is the best-response index against opponent action index j
    (-1 if unfeasible). We smooth log10(actions[br_idx[j]]) as a function
    of log10(actions[j]) via a running median. The median filter fuses
    single-pixel argmax jumps into the local monotone trend.
    """
    mask = br_idx >= 0
    if mask.sum() < 3:
        return np.array([]), np.array([])
    j = np.arange(len(br_idx))[mask]
    br = br_idx[mask]
    log_x = np.log10(actions[j])
    log_y = np.log10(actions[br])
    log_y_smooth = median_filter(log_y, size=window, mode="nearest")
    return 10 ** log_x, 10 ** log_y_smooth


def cluster_nes(nes, actions, tol_dec: float = 0.15):
    """Grid-quantize NE index tuples in log-log eps space.

    Returns (centroids_eps, counts) where centroids_eps has shape (m, 2)
    and each row is (mean_eps1, mean_eps2) of a log-log cell of side
    tol_dec decades.
    """
    if len(nes) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    log_pts = np.array([(np.log10(actions[i]), np.log10(actions[j]))
                        for (i, j) in nes])
    cells = {}
    for p in log_pts:
        key = (int(np.floor(p[0] / tol_dec)),
               int(np.floor(p[1] / tol_dec)))
        cells.setdefault(key, []).append(p)
    centroids_log = np.array([np.mean(v, axis=0) for v in cells.values()])
    counts = np.array([len(v) for v in cells.values()], dtype=int)
    return 10 ** centroids_log, counts


def dispersion_ellipse(pts_eps: np.ndarray, ax, color, alpha=0.15,
                       quantile: float = 0.68):
    """Draw a shaded 68% inertia ellipse in log-log axes."""
    if len(pts_eps) < 3:
        return
    log_pts = np.log10(pts_eps)
    mu = log_pts.mean(axis=0)
    cov = np.cov(log_pts.T)
    # Convert to visual ellipse. For a 2D Gaussian the 68% region is
    # a Mahalanobis radius of sqrt(chi2.ppf(0.68, df=2)) ~= 1.515.
    from scipy.stats import chi2
    r = float(np.sqrt(chi2.ppf(quantile, df=2)))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle_deg = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width_log = 2 * r * float(np.sqrt(max(vals[0], 1e-8)))
    height_log = 2 * r * float(np.sqrt(max(vals[1], 1e-8)))
    # Ellipse in eps space: since axes are log-log, we exploit the
    # log-log affine embedding by drawing an ellipse in log units on
    # linear axes... but the plot IS in log-log via loglog(). To draw
    # correctly we transform to data coordinates in log space by
    # rendering in log space using axes transforms. Simplest: draw a
    # polygon approximation.
    theta = np.linspace(0, 2 * np.pi, 128)
    unit = np.stack([np.cos(theta), np.sin(theta)])
    scale = np.array([[width_log / 2, 0], [0, height_log / 2]])
    R = np.array([
        [np.cos(np.radians(angle_deg)), -np.sin(np.radians(angle_deg))],
        [np.sin(np.radians(angle_deg)),  np.cos(np.radians(angle_deg))],
    ])
    log_ellipse = mu[:, None] + R @ scale @ unit
    eps_ellipse = 10 ** log_ellipse
    ax.fill(eps_ellipse[0], eps_ellipse[1], color=color, alpha=alpha,
            lw=0, zorder=2)


# --- C6 -------------------------------------------------------------------

def polish_C6():
    data = pickle.load(open(CACHE_DIR / "C6_K200.pkl", "rb"))
    actions = data["actions"]
    br1 = data["br1"]
    br2 = data["br2"]
    nes = data["nes"]
    K = data["K"]
    sweep = data["sweep"]

    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.42),
                              gridspec_kw=dict(width_ratios=[1, 1]))

    # ---- Panel (a): smoothed BR loci + clustered NE stars --------------
    ax = axes[0]
    x1, y1 = smooth_br(br1, actions, window=7)   # BR_1(eps_2): x=eps_1, y=eps_2
    y2_x, y2_y = smooth_br(br2, actions, window=7)  # BR_2(eps_1): x=eps_1, y=eps_2
    # For BR_2, the smoothing runs BR_2[j] over j=eps_1 index, giving
    # (log actions[j], log actions[BR_2[j]]) -> (eps_1, eps_2). smooth_br
    # returns (10**log_x, 10**log_y_smooth), i.e. (eps_1, eps_2) already.
    ax.loglog(x1, y1, "-", color=PRIMARY_BLUE, lw=1.4,
              label=r"$\varepsilon_1 = \mathrm{BR}_1(\varepsilon_2)$")
    ax.loglog(y2_x, y2_y, "-", color=PRIMARY_ORANGE, lw=1.4,
              label=r"$\varepsilon_2 = \mathrm{BR}_2(\varepsilon_1)$")
    ax.plot([actions[0], actions[-1]], [actions[0], actions[-1]],
            "--", color=ACCENT_GREY, lw=0.6, alpha=0.6, label="diagonal")
    centroids, counts = cluster_nes(nes, actions, tol_dec=0.15)
    # Marker size scales weakly with count (unit + sqrt).
    sizes = 45 + 14 * np.sqrt(counts)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=sizes,
               c="black", edgecolors="white", linewidth=0.5, zorder=10,
               label=f"pure NE (n={len(nes)}, {len(centroids)} clusters)")
    ax.set_xlim(actions[0], actions[-1])
    ax.set_ylim(actions[0], actions[-1])
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(rf"(a) BR loci and pure NEs ($K={K}$)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    # ---- Panel (b): NE density heatmap in log-log eps space -----------
    ax = axes[1]
    log_pts = np.array([(np.log10(actions[i]), np.log10(actions[j]))
                        for (i, j) in nes])
    nbins = 24
    log_lo = np.log10(actions[0])
    log_hi = np.log10(actions[-1])
    H, xe, ye = np.histogram2d(
        log_pts[:, 0], log_pts[:, 1],
        bins=nbins, range=[[log_lo, log_hi], [log_lo, log_hi]]
    )
    im = ax.imshow(H.T, origin="lower", extent=[log_lo, log_hi, log_lo, log_hi],
                    aspect="auto", cmap="viridis", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NE count per cell", fontsize=7)
    # Overlay smoothed BR curves and clustered centroids in log units.
    log_x1 = np.log10(x1)
    log_y1 = np.log10(y1)
    log_y2x = np.log10(y2_x)
    log_y2y = np.log10(y2_y)
    ax.plot(log_x1, log_y1, "-", color="white", lw=0.9, alpha=0.85)
    ax.plot(log_y2x, log_y2y, color="white", lw=0.9, alpha=0.85, ls="--")
    ax.plot([log_lo, log_hi], [log_lo, log_hi], ":", color="white", lw=0.6,
            alpha=0.6)
    log_centroids = np.log10(centroids)
    ax.scatter(log_centroids[:, 0], log_centroids[:, 1], marker="*",
                s=sizes * 0.9, c="white", edgecolors="black", linewidth=0.4,
                zorder=10)
    # Log-decade tick labels rendered as raw eps.
    tick_log = np.log10([0.5, 1, 2, 4, 8, 16, 32, 64])
    tick_lab = ["0.5", "1", "2", "4", "8", "16", "32", "64"]
    ax.set_xticks(tick_log); ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_log); ax.set_yticklabels(tick_lab)
    ax.set_xlim(log_lo, log_hi)
    ax.set_ylim(log_lo, log_hi)
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(r"(b) NE density (log$_{10}$ bins) + smoothed BR")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C6_locus_K200.pdf")
    fig.savefig(FIG_DIR / "fig_C6_locus_K200.svg")
    plt.close(fig)
    print(f"  wrote fig_C6_locus_K200  (|NE|={len(nes)}, "
          f"clusters={len(centroids)})")


# --- C7 -------------------------------------------------------------------

def polish_C7():
    data = pickle.load(open(CACHE_DIR / "C7_K200.pkl", "rb"))
    rows = data["rows"]
    K = data["K"]

    fig, ax = plt.subplots(figsize=column_size(aspect=0.85))
    cmap = plt.get_cmap("viridis")
    n = len(rows)
    colors = [cmap(0.10 + 0.80 * t) for t in np.linspace(0, 1, n)]

    for ip, r in enumerate(rows):
        actions = r["actions"]
        nes = r["nes"]
        if len(nes) == 0:
            continue
        pts_eps = np.array([(actions[i], actions[j]) for (i, j) in nes])
        centroids, counts = cluster_nes(nes, actions, tol_dec=0.15)
        # Dispersion ellipse over raw NE cloud in log-log.
        dispersion_ellipse(pts_eps, ax, colors[ip], alpha=0.15)
        # Cluster centroids as filled circles.
        ax.loglog(centroids[:, 0], centroids[:, 1], "o", color=colors[ip],
                   markersize=4.0, markeredgecolor="white",
                   markeredgewidth=0.4, alpha=0.95)
        # Overall centroid as a bold cross.
        c_log = np.log10(pts_eps).mean(axis=0)
        c_eps = 10 ** c_log
        ax.plot(c_eps[0], c_eps[1], "+", color=colors[ip], mew=1.4,
                 markersize=8, zorder=5)

    handles = [Line2D([0], [0], color=colors[ip], marker="o", lw=0,
                       markersize=5, label=f"$w={r['w']:g}$")
               for ip, r in enumerate(rows)]
    ax.legend(handles=handles, frameon=False, fontsize=6,
              loc="lower right", ncol=2)
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5,
            alpha=0.6, zorder=1)
    ax.set_xlim(0.5, 64)
    ax.set_ylim(0.5, 64)
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(r"NE locations across symmetric $w$ (68\% dispersion shaded)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C7_w_ratio.pdf")
    fig.savefig(FIG_DIR / "fig_C7_w_ratio.svg")
    plt.close(fig)
    print(f"  wrote fig_C7_w_ratio")


# --- C8 -------------------------------------------------------------------

def polish_C8():
    data = pickle.load(open(CACHE_DIR / "C8_K200.pkl", "rb"))
    rows = data["rows"]
    K = data["K"]

    fig, ax = plt.subplots(figsize=column_size(aspect=0.85))
    cmap = plt.get_cmap("cividis")
    n = len(rows)
    # Position ratios on a log2 axis centred on 1.
    log_ratios = np.array([np.log2(r["ratio"]) for r in rows], dtype=float)
    if log_ratios.max() > log_ratios.min():
        norm_pos = (log_ratios - log_ratios.min()) / (log_ratios.max() - log_ratios.min())
    else:
        norm_pos = np.linspace(0, 1, n)
    colors = [cmap(0.10 + 0.80 * t) for t in norm_pos]

    for ip, r in enumerate(rows):
        actions = r["actions"]
        nes = r["nes"]
        if len(nes) == 0:
            continue
        pts_eps = np.array([(actions[i], actions[j]) for (i, j) in nes])
        centroids, counts = cluster_nes(nes, actions, tol_dec=0.15)
        dispersion_ellipse(pts_eps, ax, colors[ip], alpha=0.15)
        ax.loglog(centroids[:, 0], centroids[:, 1], "s", color=colors[ip],
                   markersize=3.6, markeredgecolor="white",
                   markeredgewidth=0.4, alpha=0.95)
        c_log = np.log10(pts_eps).mean(axis=0)
        c_eps = 10 ** c_log
        ax.plot(c_eps[0], c_eps[1], "x", color=colors[ip], mew=1.4,
                 markersize=8, zorder=5)

    handles = [Line2D([0], [0], color=colors[ip], marker="s", lw=0,
                       markersize=5,
                       label=f"$w_1/w_2 = {r['ratio']:g}$")
               for ip, r in enumerate(rows)]
    ax.legend(handles=handles, frameon=False, fontsize=6,
              loc="lower right", ncol=2)
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5,
            alpha=0.6, zorder=1)
    ax.set_xlim(0.5, 64)
    ax.set_ylim(0.5, 64)
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(r"NE shift under asymmetric $w_1/w_2$")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_C8_asymmetric_w.pdf")
    fig.savefig(FIG_DIR / "fig_C8_asymmetric_w.svg")
    plt.close(fig)
    print(f"  wrote fig_C8_asymmetric_w")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    polish_C6()
    polish_C7()
    polish_C8()


if __name__ == "__main__":
    main()
