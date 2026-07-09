"""T15 -- H5 V-shape robustness rerun of C1/C6 (appendix ablation).

This is the **utility-approximation ablation** of the paper's shared
proxy V(sigma). The main body assumes a specific V; T15 confirms that
the qualitative NE structure (existence, off-diagonal placement,
externality direction) is invariant across the admissible-V regularity
class of T05 assumption A1-A3.

We reuse the K=200 mechanism-grid cache
`game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` --
the mechanism solve (sigma, per-group MIA advantage, feasibility) is
V-free by construction (`game/continuous.py`), so per-V cost collapses
to O(K^2) elementwise ops. We recompose player utilities under six
admissible V (each continuous, strictly monotone-decreasing, bounded on
the feasible sigma range [0.37, 3.65]), find pure NEs at symmetric
w=1, and report:

  * NE count per V,
  * NE-set Hausdorff distance from the baseline V_1(sigma)=1/(1+sigma^2)
    in log10(eps) space (both directions and the symmetric max),
  * externality-sign agreement rate: fraction of interior feasible
    edges on which sign(du_i/deps_j) matches the baseline.

The figure is a single scatter panel (no title, no bar charts) showing
NE locations under all six V. Robustness metrics go into the caption
and a booktabs table in the appendix.

Reproduce (fast; no Slurm, no mechanism solves):

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/15_v_shape.py

Outputs:
  paper/figures/fig_h5_v_shape.pdf
  paper/figures/fig_h5_v_shape.svg
  tasks_findings/_scripts/15_v_shape_summary.txt
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
    apply_paper_style, PRIMARY_BLUE, ACCENT_GREY, column_size,
)

CACHE = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"
FIG_PDF = ROOT / "paper" / "figures" / "fig_h5_v_shape.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_h5_v_shape.svg"
SUMMARY = Path(__file__).with_name("15_v_shape_summary.txt")


# ---------------------------------------------------------------------------
# Admissible V library (T05 assumption A1-A3: continuous, strictly monotone
# decreasing, bounded on the feasible sigma range S = [~0.37, ~3.65]).
#
# The six shapes span polynomial, rational, exponential, Gaussian, affine,
# and log decay: convex, concave, and neither-convex-nor-concave classes.
# ---------------------------------------------------------------------------
V_LIBRARY: list[tuple[str, str, callable]] = [
    ("V1", r"$1/(1+\sigma^2)$",  lambda s: 1.0 / (1.0 + s * s)),
    ("V2", r"$1/(1+\sigma)$",    lambda s: 1.0 / (1.0 + s)),
    ("V3", r"$e^{-\sigma^2/2}$", lambda s: np.exp(-0.5 * s * s)),
    ("V4", r"$e^{-\sigma}$",     lambda s: np.exp(-s)),
    ("V5", r"$-\log(1+\sigma)$", lambda s: -np.log1p(s)),
    ("V6", r"$-\sigma$",         lambda s: -s),
]


def compose_utility(V, sigma, A, feas):
    """u_i(eps_1, eps_2) = V(sigma(eps)) - w_i * A_i(eps), w=1."""
    Vgrid = np.where(feas, V(sigma), 0.0)
    U = np.stack([Vgrid - A[:, :, 0], Vgrid - A[:, :, 1]], axis=-1)
    U = np.where(feas[..., None], U, -np.inf)
    return U


def pure_ne(U, feas):
    K = U.shape[0]
    U0 = np.where(feas, U[:, :, 0], -np.inf)
    U1 = np.where(feas, U[:, :, 1], -np.inf)
    col_max = U0.max(axis=0)
    row_max = U1.max(axis=1)
    nes = []
    tol = 1e-12
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            if U0[i, j] >= col_max[j] - tol and U1[i, j] >= row_max[i] - tol:
                nes.append((i, j))
    return nes


def hausdorff_log_eps(nes_a, nes_b, actions):
    """Symmetric log-eps Hausdorff distance between two NE sets."""
    if not nes_a or not nes_b:
        return float("inf"), float("inf"), float("inf")
    la = np.log10(np.array([[actions[i], actions[j]] for (i, j) in nes_a]))
    lb = np.log10(np.array([[actions[i], actions[j]] for (i, j) in nes_b]))
    dAB = np.linalg.norm(la[:, None, :] - lb[None, :, :], axis=-1)
    to_b = dAB.min(axis=1)
    to_a = dAB.min(axis=0)
    return float(max(to_b.max(), to_a.max())), float(to_b.mean()), float(to_a.mean())


def externality_signs(U, feas):
    """Per-V externality-sign statistics."""
    U0 = U[:, :, 0]
    U1 = U[:, :, 1]
    d0j = np.diff(U0, axis=1)
    feas_j = feas[:, :-1] & feas[:, 1:] & np.isfinite(U0[:, :-1]) & np.isfinite(U0[:, 1:])
    d1i = np.diff(U1, axis=0)
    feas_i = feas[:-1, :] & feas[1:, :] & np.isfinite(U1[:-1, :]) & np.isfinite(U1[1:, :])
    frac0 = float((d0j[feas_j] >= 0).mean()) if feas_j.any() else float("nan")
    frac1 = float((d1i[feas_i] >= 0).mean()) if feas_i.any() else float("nan")
    return frac0, frac1


def externality_agreement(U_a, U_b, feas):
    """Agreement rate of sign(du_i/deps_j) between two V's, over
    feasible interior edges. Returns a single fraction."""
    def d_pairs(U):
        d0j = np.diff(U[:, :, 0], axis=1)
        d1i = np.diff(U[:, :, 1], axis=0)
        return d0j, d1i
    a0j, a1i = d_pairs(U_a)
    b0j, b1i = d_pairs(U_b)
    feas_j = feas[:, :-1] & feas[:, 1:]
    feas_i = feas[:-1, :] & feas[1:, :]
    fin_j = feas_j & np.isfinite(a0j) & np.isfinite(b0j)
    fin_i = feas_i & np.isfinite(a1i) & np.isfinite(b1i)
    agree = 0
    total = 0
    for arr_a, arr_b, mask in ((a0j, b0j, fin_j), (a1i, b1i, fin_i)):
        s_a = np.sign(arr_a[mask])
        s_b = np.sign(arr_b[mask])
        agree += int((s_a == s_b).sum())
        total += int(mask.sum())
    return (agree / total) if total else float("nan")


def _fmt_frac(x: float, digits: int = 5) -> str:
    return f"{x*100:.{digits}f}%"


def main() -> None:
    if not CACHE.exists():
        print(f"cache missing: {CACHE}", file=sys.stderr)
        sys.exit(1)

    with CACHE.open("rb") as f:
        grid = pickle.load(f)
    actions = grid["actions"]
    sigma = grid["sigma"]
    A = grid["A"]
    feas = grid["feasible"]
    K = len(actions)

    print(f"[T15] K={K}, eps in [{actions[0]:.3g}, {actions[-1]:.3g}]")
    print(f"[T15] feasible cells: {int(feas.sum())}/{feas.size}")
    print(f"[T15] sigma range on feasible: "
          f"[{sigma[feas].min():.3f}, {sigma[feas].max():.3f}]")

    per_V: dict[str, dict] = {}
    for tag, label, V in V_LIBRARY:
        U = compose_utility(V, sigma, A, feas)
        nes = pure_ne(U, feas)
        f0, f1 = externality_signs(U, feas)
        per_V[tag] = {
            "label": label, "U": U, "nes": nes,
            "ext0": f0, "ext1": f1,
        }
        n_asym = sum(1 for (i, j) in nes if i != j)
        n_sym = len(nes) - n_asym
        print(f"[T15] {tag}  {label:>30s}  #NE={len(nes):>3d}  "
              f"(asym={n_asym}, sym={n_sym})  "
              f"ext P1={f0*100:.4f}% ext P2={f1*100:.4f}%")

    nes_base = per_V["V1"]["nes"]
    haus: dict[str, tuple[float, float, float]] = {}
    for tag in per_V:
        if tag == "V1":
            haus[tag] = (0.0, 0.0, 0.0)
            continue
        haus[tag] = hausdorff_log_eps(per_V[tag]["nes"], nes_base, actions)

    U_base = per_V["V1"]["U"]
    agree = {"V1": 1.0}
    for tag in per_V:
        if tag == "V1":
            continue
        agree[tag] = externality_agreement(per_V[tag]["U"], U_base, feas)

    # Cross-V invariant: pairwise |ext P_i - ext P_1| max (to 5 decimals?).
    ext_all = [(per_V[t]["ext0"], per_V[t]["ext1"]) for t in per_V]
    ext_diff_max = max(
        max(abs(e0 - ext_all[0][0]), abs(e1 - ext_all[0][1]))
        for (e0, e1) in ext_all
    )
    ext_identical_5dp = ext_diff_max < 5e-6

    # --------------------------------------------------------------
    # Summary block.
    # --------------------------------------------------------------
    lines = [
        "# T15 / H5 V-shape robustness summary (appendix ablation)",
        f"# K={K}, w1=w2=1, symmetric cache, "
        f"sigma range [{sigma[feas].min():.3f}, {sigma[feas].max():.3f}]",
        "",
        f"# {'V':>4} {'shape':>32s} {'#NE':>4} {'asym':>4} {'sym':>4} "
        f"{'ext P1':>10} {'ext P2':>10}  "
        f"{'d_H':>7} {'mean(a->b)':>10} {'mean(b->a)':>10} {'agree(sign)':>11}",
    ]
    for tag, label, _ in V_LIBRARY:
        r = per_V[tag]
        h, ma, mb = haus[tag]
        n_asym = sum(1 for (i, j) in r["nes"] if i != j)
        n_sym = len(r["nes"]) - n_asym
        lines.append(
            f"  {tag:>4} {label:>32s} {len(r['nes']):>4d} {n_asym:>4d} {n_sym:>4d} "
            f"{_fmt_frac(r['ext0']):>10s} {_fmt_frac(r['ext1']):>10s}  "
            f"{h:>7.3f} {ma:>10.3f} {mb:>10.3f} {agree[tag]*100:>10.1f}%"
        )
    lines += [
        "",
        f"# max pairwise |ext P_i - ext P_1|  = {ext_diff_max:.2e}",
        f"# ext P_i identical to 5 decimals?   {ext_identical_5dp}",
        "",
        "# Notes",
        "# d_H, mean(a->b), mean(b->a) are in log10(eps) units",
        "# 'a' is the V under test; 'b' is the baseline V1 = 1/(1+s^2)",
        "# ext P1 = fraction of feasible edges with dU_1/deps_2 >= 0",
        "# ext P2 = fraction of feasible edges with dU_2/deps_1 >= 0",
        "# agree(sign) = fraction of feasible edges where "
        "sign(du_i/deps_j) matches V1",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {SUMMARY}")

    # ------------------------------------------------------------------
    # Figure. Single-panel scatter in (eps_1, eps_2), no title, no bars.
    # Style-matched to fig_h8_brpeg_regret: apply_paper_style, primary
    # colour anchoring, log-log axes, no title, legend inside the panel.
    # ------------------------------------------------------------------
    apply_paper_style()
    from matplotlib import cm
    palette = [cm.get_cmap("viridis")(x)
               for x in np.linspace(0.05, 0.85, len(V_LIBRARY))]
    palette[0] = PRIMARY_BLUE  # baseline V_1 anchor
    markers = ["o", "s", "D", "^", "v", "P"]

    # Slightly wider than a single column so the outside legend does not
    # squeeze the axes; still narrower than the two-column banner used by
    # main-body figures.
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.4))

    # Diagonal reference (identity eps_1 = eps_2).
    ax.plot([actions[0], actions[-1]], [actions[0], actions[-1]],
            "-", color=ACCENT_GREY, lw=0.6, alpha=0.7,
            label=r"$\varepsilon_1=\varepsilon_2$", zorder=1)

    # Slight log-space jitter to disambiguate NEs that coincide across V.
    rng = np.random.default_rng(0)
    for k, (tag, label, _) in enumerate(V_LIBRARY):
        nes = per_V[tag]["nes"]
        if not nes:
            continue
        pts = np.array([[actions[i], actions[j]] for (i, j) in nes], dtype=float)
        jitter = 10 ** (rng.normal(0, 0.015, size=pts.shape))
        pts = pts * jitter
        n_ne = len(nes)
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=24, marker=markers[k],
                   facecolor=palette[k], edgecolor="white", linewidth=0.4,
                   alpha=0.85,
                   label=f"{tag}: {label}  ($n{{=}}{n_ne}$)",
                   zorder=3 + k)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(actions[0] * 0.9, actions[-1] * 1.1)
    ax.set_ylim(actions[0] * 0.9, actions[-1] * 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\varepsilon_2$")

    ax.legend(frameon=False, fontsize=6.5,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              handletextpad=0.4, labelspacing=0.4, borderaxespad=0.0)

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)
    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")


if __name__ == "__main__":
    main()
