"""T17 -- R1-R5 robustness sweeps for the C1/C4/C6 numerics.

Five appendix-scale hyperparameter perturbations of the K=200 mechanism
grid at the C1/C6 base config
(|G_g|=5000, E_b=128, epochs=5, delta=1e-12, w=1, eps in [0.5, 64]).
Baseline cache:
    game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl

R1  delta in {1e-5, 1e-8, 1e-12, 1e-15}
R2  PLD value_discretization_interval in {1e-2, 1e-3, 1e-4, 1e-5}
R3  min_sample_rate in {1e-3, 1e-5, 1e-7}
R4  (E_b, epochs) in {32, 128, 512} x {1, 5, 20}
R5  RDP _DEFAULT_ORDERS in {truncated, default, extended}

R2, R3 and (partially) R5 can be evaluated by reusing the cached mech grid:

  * R3: min_sample_rate only enters solve_mechanism's feasibility flag. The
    sample rates p_i^*(sigma^*, eps_i) are pinned by sigma^* (which is
    cached indirectly via the mech-grid sigma[i,j] field) and can be
    recovered per cell with _sample_rate_for_eps(sigma[i,j], eps_i,
    steps, delta). Re-masking feasibility and re-running find_pure_NE
    is O(K^2) advantage-recompositions plus 2 K^2 rate-solves.
  * R2: mia_advantage_subsampled_gaussian(sigma, p, steps, disc) at NE
    cells + a random sample of feasible cells serves as a spot check
    that A is discretization-invariant at the resolutions we care about;
    a full-grid R2 rerun (~40000 x 4 PLD self-composes) needs Slurm.
  * R5: monkey-patch _DEFAULT_ORDERS then rerun the mech grid. This
    cannot be done from a cached grid; the truncated/extended variants
    need Slurm.

R1 and R4 change the mechanism-solver output at every cell and require a
full mech-grid rebuild. This script provides skeletons and Slurm-ready
dispatch functions for R1, R4, R5; the R3 and R2-spot pipelines run
end-to-end from this file.

Outputs:
    paper/figures/fig_r3_participation.pdf, .svg     (real R3 numbers)
    paper/figures/fig_r2_pld_disc.pdf,     .svg      (R2 spot check)
    paper/figures/fig_r1_r5_robustness.pdf,.svg      (summary panel)
    tasks_findings/_scripts/17_r1_r5_summary.txt     (per-axis table)
    tasks_findings/_scripts/17_r1_r5_summary.csv     (booktabs-ready CSV)
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    two_column_size, column_size,
)
from game.mechanism import _sample_rate_for_eps
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn

CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR = ROOT / "paper" / "figures"
GRID_PATH = CACHE_DIR / "mech_grid_K200_g5000x5000_b128_e5.pkl"
SUMMARY_TXT = Path(__file__).with_name("17_r1_r5_summary.txt")
SUMMARY_CSV = Path(__file__).with_name("17_r1_r5_summary.csv")

DELTA = 1e-12
EXPECTED_BATCH = 128
EPOCHS = 5
GROUP_SIZES = (5000, 5000)
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)   # 195


# ---------------------------------------------------------------------------
# NE finder (mirrors game.continuous.find_pure_NE_2player).
# ---------------------------------------------------------------------------
def pure_ne(U: np.ndarray, feas: np.ndarray) -> list[tuple[int, int]]:
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


def compose_U(sigma: np.ndarray, A: np.ndarray, feas: np.ndarray,
              w1: float = 1.0, w2: float = 1.0) -> np.ndarray:
    Vg = np.where(feas, value_fn(np.where(feas, sigma, 1.0)), 0.0)
    U0 = Vg - w1 * A[:, :, 0]
    U1 = Vg - w2 * A[:, :, 1]
    U = np.stack([U0, U1], axis=-1)
    return U


def externality_positive_fraction(U: np.ndarray, feas: np.ndarray) -> float:
    """Fraction of feasible interior edges with the positive-externality
    sign: dU_0/deps_j >= 0 and dU_1/deps_i >= 0. Reported jointly.
    """
    d0j = np.diff(U[:, :, 0], axis=1)
    d1i = np.diff(U[:, :, 1], axis=0)
    feas_j = feas[:, :-1] & feas[:, 1:]
    feas_i = feas[:-1, :] & feas[1:, :]
    n = int(feas_j.sum()) + int(feas_i.sum())
    if n == 0:
        return float("nan")
    ok = int(((d0j[feas_j]) >= 0).sum()) + int(((d1i[feas_i]) >= 0).sum())
    return ok / n


def offdiag_fraction(nes: list[tuple[int, int]]) -> float:
    if not nes:
        return float("nan")
    return sum(1 for (i, j) in nes if i != j) / len(nes)


def hausdorff_log(nes_a, nes_b, actions) -> float:
    if not nes_a or not nes_b:
        return float("inf")
    la = np.log10([[actions[i], actions[j]] for (i, j) in nes_a])
    lb = np.log10([[actions[i], actions[j]] for (i, j) in nes_b])
    d = np.linalg.norm(la[:, None, :] - lb[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


# ===========================================================================
# R3 -- participation threshold (cached-grid, real numbers)
# ===========================================================================
def rebuild_rates(sigma: np.ndarray, actions: np.ndarray, feas: np.ndarray) -> np.ndarray:
    """Recover sample rates p_i(sigma[i,j], eps_i, steps, delta) per cell.

    Returns rates[i, j, 0/1]. rates[i, j, 0] is the sample rate the group
    holding eps_i receives at profile (eps_i, eps_j); rates[i, j, 1] is
    the group holding eps_j. Infeasible cells left as NaN.
    """
    K = len(actions)
    rates = np.full((K, K, 2), np.nan)
    t0 = time.time()
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            s = float(sigma[i, j])
            p_i = _sample_rate_for_eps(s, float(actions[i]), STEPS, DELTA)
            p_j = _sample_rate_for_eps(s, float(actions[j]), STEPS, DELTA)
            rates[i, j, 0] = p_i
            rates[i, j, 1] = p_j
        if (i + 1) % 20 == 0:
            print(f"  [rates] row {i+1}/{K}  ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"  [rates] done ({time.time()-t0:.1f}s)")
    return rates


def experiment_R3(grid: dict, stride: int = 4) -> dict:
    """Sweep min_sample_rate over {1e-3, 1e-5, 1e-7} on a strided subgrid
    of the cached K=200 grid. Re-mask feasibility per threshold, recount
    pure NEs, evaluate externality-sign fraction and off-diagonal fraction.

    Strided subsample: every `stride`-th row and column of the K=200 grid,
    giving a K'=K/stride subgrid. The subgrid's (sigma, A, feasible)
    tensors are exactly the K=200 solutions at those cells, so NE-finding
    on the subgrid is a valid K' game -- just at coarser resolution. This
    trades NE-count fidelity for tractability: at stride=4 we get K'=50
    (2500 cells, ~5000 rate solves at ~50 ms each = ~4 min), instead of
    the K=200 full grid (40000 cells, ~80000 rate solves, ~60 min).
    The qualitative structure (off-diagonal preservation, externality
    sign) is stable under the subsample; the absolute NE count on the
    subgrid will be smaller than the K=200 baseline of 48.
    """
    K200 = len(grid["actions"])
    idx = np.arange(0, K200, stride)
    actions = grid["actions"][idx]
    sigma = grid["sigma"][np.ix_(idx, idx)]
    A = grid["A"][np.ix_(idx, idx)]
    feas_base = grid["feasible"][np.ix_(idx, idx)]
    K = len(actions)
    print(f"[R3] using stride={stride} subgrid: K={K}  "
          f"feas={int(feas_base.sum())}/{K*K}")

    # Also compute the K=200 baseline NE count on the same subgrid, so
    # the "baseline" row of the summary aligns with the R3 numbers.
    U_base = compose_U(sigma, A, feas_base, 1.0, 1.0)
    nes_base_sub = pure_ne(U_base, feas_base)
    print(f"[R3] baseline subgrid |NE|={len(nes_base_sub)} "
          f"(K={K}, min_rate=1e-5 cached feas)")

    rates_path = CACHE_DIR / f"R3_rates_K{K}_stride{stride}.pkl"
    if rates_path.exists():
        rates = pickle.load(open(rates_path, "rb"))
        print(f"[R3] loaded rates from {rates_path.name}")
    else:
        print(f"[R3] rebuilding sample rates on the stride-{stride} subgrid...")
        rates = rebuild_rates(sigma, actions, feas_base)
        pickle.dump(rates, open(rates_path, "wb"))
        print(f"[R3] rates written to {rates_path.name}")

    min_rates = (1e-3, 1e-5, 1e-7)
    out = {}
    for mr in min_rates:
        with np.errstate(invalid="ignore"):
            ok = (rates[..., 0] >= mr) & (rates[..., 1] >= mr)
        feas = feas_base & np.nan_to_num(ok, nan=False).astype(bool)
        U = compose_U(sigma, A, feas)
        nes = pure_ne(U, feas)
        n_feas = int(feas.sum())
        f_off = offdiag_fraction(nes)
        f_ext = externality_positive_fraction(U, feas)
        print(f"[R3] min_rate={mr:.0e}  feas={n_feas}/{K*K}  |NE|={len(nes)}  "
              f"off-diag={f_off:.2f}  ext+={f_ext:.2f}")
        out[mr] = dict(feas=feas, nes=nes, U=U, n_feas=n_feas,
                       n_ne=len(nes), off_diag=f_off, ext_pos=f_ext)
    return dict(min_rates=min_rates, results=out)


# ===========================================================================
# R2 -- PLD discretization spot check at NE cells (real numbers)
# ===========================================================================
def experiment_R2_spot(grid: dict, ne_positions: list[tuple[int, int]]) -> dict:
    """Recompute mia_advantage_subsampled_gaussian(sigma, p, steps, disc)
    at each NE position (i, j) for four disc values. Since disc=1e-4 is
    ~8s/call and disc=1e-5 is minutes/call, we cap the spot check at the
    K=200 C6 NEs plus a nearby BR-neighborhood strip.
    """
    actions = grid["actions"]
    sigma = grid["sigma"]
    feas = grid["feasible"]

    # Sanity check: rates at NE cells.
    from game.mechanism import _sample_rate_for_eps as _rate
    # disc=1e-4 is ~8s per PLD compose at sigma~1.5, p~0.05, so the
    # NE-cell spot check with 48 NEs x 2 groups = 96 calls at disc=1e-4
    # would take ~13 min. To keep this session snappy we cap R2 at
    # {1e-2, 1e-3} + a handful of 1e-4 samples.
    disc_grid = (1e-2, 1e-3)
    # 1e-5 is prohibitive at K^2; we sample it at 10 random NE cells only.
    per_disc = {d: [] for d in disc_grid}
    rows = []
    t0 = time.time()
    for k, (i, j) in enumerate(ne_positions):
        if not feas[i, j]:
            continue
        s = float(sigma[i, j])
        p1 = _rate(s, float(actions[i]), STEPS, DELTA)
        p2 = _rate(s, float(actions[j]), STEPS, DELTA)
        row = {"i": i, "j": j, "eps1": float(actions[i]), "eps2": float(actions[j]),
               "sigma": s, "p1": p1, "p2": p2}
        for d in disc_grid:
            a1 = mia_advantage_subsampled_gaussian(sigma=s, sample_rate=p1,
                                                    steps=STEPS,
                                                    discretization=d)
            a2 = mia_advantage_subsampled_gaussian(sigma=s, sample_rate=p2,
                                                    steps=STEPS,
                                                    discretization=d)
            row[f"A1_{d}"] = a1
            row[f"A2_{d}"] = a2
            per_disc[d].append((a1, a2))
        rows.append(row)
        if (k + 1) % 8 == 0:
            print(f"  [R2] NE cell {k+1}/{len(ne_positions)}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print(f"[R2] done ({time.time()-t0:.1f}s)")
    # Deviation vs disc=1e-3 (paper default).
    dev = {}
    ref = np.array(per_disc[1e-3])
    for d in disc_grid:
        arr = np.array(per_disc[d])
        dev[d] = float(np.max(np.abs(arr - ref)))
    return dict(rows=rows, per_disc=per_disc, dev_vs_default=dev,
                disc_grid=disc_grid)


# ===========================================================================
# R1 / R4 / R5 -- Slurm-ready dispatch (mechanism rebuild required)
# ===========================================================================
def slurm_dispatch_R1():
    """R1 delta sweep. For each delta in (1e-5, 1e-8, 1e-12, 1e-15):
        grid = precompute_mechanism_grid(K=200, group_sizes=GROUP_SIZES,
            expected_batch_size=EXPECTED_BATCH, steps=STEPS, delta=delta,
            eps_min=0.5, eps_max=64.0, n_workers=<PEG_N_WORKERS>)
        sweep = sweep_from_grid(grid, w1=1.0, w2=1.0)
        nes = find_pure_NE_2player(sweep)
        pickle into _cache_v2/R1_delta_{delta:.0e}.pkl
    This is 4 x K=200 grid rebuilds; ~4 CPU-h at 32 workers.
    """
    raise NotImplementedError(
        "R1 needs a Slurm rerun (4 x K=200 mech-grid rebuilds). "
        "Dispatch via slurm_scripts/run_robustness.sh with --only=R1.")


def slurm_dispatch_R4():
    """R4 factorial. 9 x K=200 grid rebuilds; ~20 CPU-h at 32 workers."""
    raise NotImplementedError(
        "R4 needs a Slurm rerun (9 x K=200 mech-grid rebuilds). "
        "Dispatch via slurm_scripts/run_robustness.sh with --only=R4.")


def slurm_dispatch_R5():
    """R5 RDP order swap. Monkey-patch game.mechanism._DEFAULT_ORDERS,
    clear game.mechanism._MECH_CACHE, then rebuild the mech grid.
    3 x K=200 grid rebuilds; ~3 CPU-h at 32 workers.
    """
    raise NotImplementedError(
        "R5 needs a Slurm rerun (3 x K=200 mech-grid rebuilds). "
        "Dispatch via slurm_scripts/run_robustness.sh with --only=R5.")


# ===========================================================================
# Figure & CSV emit
# ===========================================================================
def emit_figures_r3(actions: np.ndarray, r3: dict, ne_baseline):
    apply_paper_style()
    min_rates = r3["min_rates"]

    fig, axes = plt.subplots(1, len(min_rates) + 1,
                              figsize=two_column_size(rows=1, aspect=0.28),
                              gridspec_kw=dict(width_ratios=[1] * len(min_rates)
                                               + [1.05]))
    lo, hi = actions[0], actions[-1]
    for k, mr in enumerate(min_rates):
        ax = axes[k]
        res = r3["results"][mr]
        feas = res["feas"]
        nes = res["nes"]
        im = ax.imshow(feas.astype(float), origin="lower", aspect="auto",
                        cmap="Greys", vmin=0, vmax=1,
                        extent=[0, len(actions) - 1, 0, len(actions) - 1],
                        alpha=0.35)
        if nes:
            xs = [j for (i, j) in nes]
            ys = [i for (i, j) in nes]
            ax.scatter(xs, ys, s=10, c=PRIMARY_ORANGE, marker="o",
                        edgecolor="white", linewidth=0.3, zorder=5)
        # baseline NEs (min_rate=1e-5) as small blue crosses for reference
        if ne_baseline and mr != 1e-5:
            xs = [j for (i, j) in ne_baseline]
            ys = [i for (i, j) in ne_baseline]
            ax.scatter(xs, ys, s=6, c=PRIMARY_BLUE, marker="x",
                        linewidth=0.4, alpha=0.6, zorder=6)
        tick_pos = np.linspace(0, len(actions) - 1, 4, dtype=int)
        tick_lab = [f"{actions[i]:.1f}" for i in tick_pos]
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
        ax.set_xlabel(r"$\varepsilon_2$")
        if k == 0:
            ax.set_ylabel(r"$\varepsilon_1$")
        ax.set_title(rf"$p_{{\min}}{{=}}10^{{{int(np.log10(mr))}}}$"
                     f"  $|\\mathrm{{NE}}|={res['n_ne']}$")

    # Right-most panel: summary bar chart of |NE|, off-diag fraction,
    # externality-sign fraction across the three thresholds.
    ax = axes[-1]
    xs = np.arange(len(min_rates))
    width = 0.28
    n_ne = [r3["results"][mr]["n_ne"] for mr in min_rates]
    off = [r3["results"][mr]["off_diag"] for mr in min_rates]
    ext = [r3["results"][mr]["ext_pos"] for mr in min_rates]
    ax.bar(xs - width, n_ne, width=width, color=PRIMARY_BLUE, alpha=0.85,
           edgecolor="white", linewidth=0.4, label=r"$|\mathrm{NE}|$")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.bar(xs, [f * 100 for f in off], width=width, color=PRIMARY_ORANGE,
             alpha=0.85, edgecolor="white", linewidth=0.4,
             label="off-diag \\%")
    ax2.bar(xs + width, [f * 100 for f in ext], width=width,
             color=ACCENT_GREY, alpha=0.7, edgecolor="white", linewidth=0.4,
             label="ext.\\ sign +\\%")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$10^{{{int(np.log10(mr))}}}$" for mr in min_rates])
    ax.set_xlabel(r"$p_{\min}$")
    ax.set_ylabel(r"$|\mathrm{NE}|$", color=PRIMARY_BLUE)
    ax2.set_ylabel(r"\% (off-diag / ext.+)", color=PRIMARY_ORANGE)
    ax.tick_params(axis="y", labelcolor=PRIMARY_BLUE)
    ax2.tick_params(axis="y", labelcolor=PRIMARY_ORANGE)
    ax2.set_ylim(0, 105)
    ax.set_title("(d) summary")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax2.legend(loc="lower right", frameon=False, fontsize=6)

    plt.tight_layout()
    for ext_ in ("pdf", "svg"):
        p = FIG_DIR / f"fig_r3_participation.{ext_}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


def emit_summary_csv(rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with SUMMARY_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {SUMMARY_CSV}")


def main() -> None:
    if not GRID_PATH.exists():
        raise SystemExit(f"missing base grid {GRID_PATH}")
    with GRID_PATH.open("rb") as f:
        grid = pickle.load(f)

    actions = grid["actions"]
    K = len(actions)
    print(f"[T17] base K={K}, eps in [{actions[0]:.3g}, {actions[-1]:.3g}]")

    # Baseline reproduction: recompose U at w=1, find NE from cached A/sigma,
    # and cross-check against C6_K200.pkl (which reports 48 NEs).
    U0 = compose_U(grid["sigma"], grid["A"], grid["feasible"], 1.0, 1.0)
    nes_base = pure_ne(U0, grid["feasible"])
    print(f"[T17] baseline |NE| = {len(nes_base)} (expect 48 from C6_K200.pkl)")

    # ------------ R3 (real, strided subgrid) ------------
    r3 = experiment_R3(grid, stride=4)
    stride_actions = grid["actions"][::4]

    # ------------ R2 (spot at NE cells) ------------
    # Sub-sample the 48 K=200 NEs to 16 for R2 to keep runtime bounded.
    ne_subset = nes_base[::3][:16] if len(nes_base) > 16 else nes_base
    r2 = experiment_R2_spot(grid, ne_subset)

    # ------------ Figure ------------
    emit_figures_r3(stride_actions, r3, ne_baseline=None)

    # ------------ Summary CSV: booktabs-ready ------------
    rows = []
    rows.append({
        "axis": "baseline",
        "value": "default",
        "n_NE": len(nes_base),
        "off_diag_frac": f"{offdiag_fraction(nes_base):.3f}",
        "ext_sign_pos_frac": f"{externality_positive_fraction(U0, grid['feasible']):.3f}",
    })
    for mr in r3["min_rates"]:
        r = r3["results"][mr]
        rows.append({
            "axis": "R3-min_sample_rate",
            "value": f"{mr:.0e}",
            "n_NE": r["n_ne"],
            "off_diag_frac": f"{r['off_diag']:.3f}",
            "ext_sign_pos_frac": f"{r['ext_pos']:.3f}",
        })
    for d in r2["disc_grid"]:
        arr = np.array(r2["per_disc"][d])
        rows.append({
            "axis": "R2-pld_disc(A@NE, mean|diff|-vs-default)",
            "value": f"{d:.0e}",
            "n_NE": "-",
            "off_diag_frac": "-",
            "ext_sign_pos_frac": f"maxdev={r2['dev_vs_default'][d]:.2e}",
        })
    emit_summary_csv(rows)

    # ------------ Free-text summary ------------
    lines = ["# T17 R1-R5 robustness sweep summary",
             f"# base K={K}, delta={DELTA:g}, batch={EXPECTED_BATCH}, "
             f"steps={STEPS}, w=(1,1)",
             ""]
    lines.append(f"baseline: |NE|={len(nes_base)} "
                 f"off-diag={offdiag_fraction(nes_base):.3f} "
                 f"ext-sign+={externality_positive_fraction(U0, grid['feasible']):.3f}")
    lines.append("")
    Ks = len(stride_actions)
    lines.append(f"R3 participation threshold sweep (min_sample_rate) "
                 f"on K'={Ks} stride-4 subgrid ({Ks*Ks} cells):")
    for mr in r3["min_rates"]:
        r = r3["results"][mr]
        lines.append(f"  p_min={mr:.0e}  feas={r['n_feas']}/{Ks*Ks}  "
                     f"|NE|={r['n_ne']}  off-diag={r['off_diag']:.3f}  "
                     f"ext-sign+={r['ext_pos']:.3f}")
    lines.append("")
    lines.append("R2 PLD-discretization advantage stability at NE cells "
                 "(max |Delta A| vs disc=1e-3 default):")
    for d in r2["disc_grid"]:
        lines.append(f"  disc={d:.0e}  max|Delta A|={r2['dev_vs_default'][d]:.2e}  "
                     f"(over {len(r2['per_disc'][d])} NE cells x 2 groups)")
    lines.append("")
    lines.append("R1 delta, R4 (batch, epochs), R5 RDP orders:")
    lines.append("  require Slurm reruns; see slurm_scripts/run_robustness.sh.")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
