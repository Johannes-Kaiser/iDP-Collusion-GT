"""
Single best-response-dynamics trace of the symmetric 2-player Privacy
Externality Game, rendered as a multi-panel diagnostic figure plus
pgfplots-ready .dat logs.

Perfectly symmetric players: group_sizes=(5000,5000), expected_batch=128,
steps=int(5*10000/128)=390, delta=1e-12, w1=w2=1.0. Action set: K=22
log-spaced eps in [0.5, 64] (run_continuous.py convention).

Dynamics: both players start at the grid midpoint action (symmetric,
deliberately off-equilibrium). Sequential best response: step 1 = P1
best-responds holding eps2 fixed, step 2 = P2, alternating until neither
player changes their action. Only mechanism-feasible profiles are admissible.

Everything is computed fresh; NO pickled caches are loaded (the 2026-07-13
bisection fix invalidated all pre-fix caches).

Quantities per evaluated profile (eps1, eps2):
  sigma            shared noise multiplier (mechanism solver)
  p_i              per-player Poisson sampling rate
  A_cert,i         certified/contract advantage
                     = (e^{eps_i} - 1 + 2 delta) / (e^{eps_i} + 1)
                   (mechanism-agnostic worst case; paper's A_{eps,delta})
  A_PLD,i          realised PLD advantage (mia_advantage_subsampled_gaussian)
  u_i              V(sigma) - w_i * A_PLD,i
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import ContinuousGame
from game.utility import PlayerProfile, value_fn
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
DAT_DIR = FIG_DIR / "data" / "br_trace"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DAT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0


def a_cert(eps: float, delta: float = DELTA) -> float:
    """Certified/contract advantage A_{eps,delta}."""
    e = math.exp(eps)
    return (e - 1.0 + 2.0 * delta) / (e + 1.0)


def build_game(K: int, w: float = 1.0, batch: int = 128,
               min_rate: float = 1e-5) -> ContinuousGame:
    group_sizes = (5000, 5000)
    M = sum(group_sizes)
    steps = int(5 * M / batch)  # 5 epochs; 390 at the default batch 128
    players = (PlayerProfile("P1", w=w), PlayerProfile("P2", w=w))
    return ContinuousGame(
        players=players, group_sizes=group_sizes,
        expected_batch_size=batch, steps=steps, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX, K=K,
        min_sample_rate=min_rate,
    )


def eval_profile(game: ContinuousGame, eps1: float, eps2: float) -> dict:
    ev = game.evaluate((float(eps1), float(eps2)))
    sigma = ev["sigma"]
    p1, p2 = float(ev["rates"][0]), float(ev["rates"][1])
    A_pld1, A_pld2 = float(ev["advantages"][0]), float(ev["advantages"][1])
    Ac1, Ac2 = a_cert(eps1), a_cert(eps2)
    return dict(
        eps1=float(eps1), eps2=float(eps2), sigma=float(sigma),
        p1=p1, p2=p2, A_pld1=A_pld1, A_pld2=A_pld2,
        A_cert1=Ac1, A_cert2=Ac2,
        u1=float(ev["utilities"][0]), u2=float(ev["utilities"][1]),
        feasible=bool(ev["feasible"]),
    )


def sweep_mover(game: ContinuousGame, grid: np.ndarray, mover: int,
                other_eps: float) -> list[dict]:
    """Sweep the mover's candidate eps over the whole grid, holding the
    other player's eps fixed. mover in {0,1} (0=P1)."""
    rows = []
    for cand in grid:
        if mover == 0:
            rows.append(eval_profile(game, cand, other_eps))
        else:
            rows.append(eval_profile(game, other_eps, cand))
    return rows


def argmax_feasible(rows: list[dict], mover: int):
    """Return (best_index, tie_flag) maximising the mover's own utility over
    feasible candidates. tie_flag True if >1 candidate within 1e-9."""
    ukey = "u1" if mover == 0 else "u2"
    best_i, best_u = None, -np.inf
    for i, r in enumerate(rows):
        if not r["feasible"]:
            continue
        u = r[ukey]
        if u > best_u:
            best_u, best_i = u, i
    if best_i is None:
        return None, False
    ties = sum(1 for r in rows if r["feasible"] and abs(r[ukey] - best_u) < 1e-9)
    return best_i, ties > 1


def run_dynamics(game: ContinuousGame, grid: np.ndarray, max_steps: int = 12):
    """Sequential best-response from the symmetric grid-midpoint start.

    Returns a list of step records. Step 0 is the initial state (mover = the
    upcoming mover P1, swept for context, but no move applied). Steps >= 1
    apply the mover's argmax best response.
    """
    K = len(grid)
    mid = K // 2
    idx1, idx2 = mid, mid
    steps = []

    # Step 0: initial state. Sweep the upcoming mover (P1) for context.
    sweep0 = sweep_mover(game, grid, mover=0, other_eps=float(grid[idx2]))
    state0 = eval_profile(game, float(grid[idx1]), float(grid[idx2]))
    steps.append(dict(
        t=0, mover=0, moved=False, is_initial=True,
        idx1=idx1, idx2=idx2, prev_mover_idx=idx1,
        opt_idx=None, tie=False, sweep=sweep0, state=state0,
    ))

    stable_streak = 0
    t = 1
    mover = 0
    while t <= max_steps:
        other_eps = float(grid[idx2]) if mover == 0 else float(grid[idx1])
        sweep = sweep_mover(game, grid, mover=mover, other_eps=other_eps)
        opt_idx, tie = argmax_feasible(sweep, mover)
        prev_idx = idx1 if mover == 0 else idx2
        if opt_idx is None:
            # No feasible candidate for the mover; abort.
            steps.append(dict(
                t=t, mover=mover, moved=False, is_initial=False,
                idx1=idx1, idx2=idx2, prev_mover_idx=prev_idx,
                opt_idx=None, tie=False, sweep=sweep, state=None,
                aborted=True,
            ))
            break
        moved = opt_idx != prev_idx
        if mover == 0:
            idx1 = opt_idx
        else:
            idx2 = opt_idx
        state = eval_profile(game, float(grid[idx1]), float(grid[idx2]))
        steps.append(dict(
            t=t, mover=mover, moved=moved, is_initial=False,
            idx1=idx1, idx2=idx2, prev_mover_idx=prev_idx,
            opt_idx=opt_idx, tie=tie, sweep=sweep, state=state,
        ))
        stable_streak = stable_streak + 1 if not moved else 0
        # Converged once a full alternating round produced no change.
        if stable_streak >= 2:
            break
        mover = 1 - mover
        t += 1

    return steps, (idx1, idx2)


# --------------------------------------------------------------------------
# .dat logging
# --------------------------------------------------------------------------
def write_dat_files(steps: list[dict], grid: np.ndarray, dat_dir: Path = None):
    dat_dir = DAT_DIR if dat_dir is None else dat_dir
    dat_dir.mkdir(parents=True, exist_ok=True)
    for st in steps:
        t = st["t"]
        sweep = st["sweep"]
        mover = st["mover"]
        for player in (0, 1):
            fname = dat_dir / f"step{t}_p{player+1}.dat"
            with open(fname, "w") as f:
                f.write("# eps_candidate u V A_pld A_cert sigma p_rate feasible "
                        f"(step={t}, mover=P{mover+1}, this_row=P{player+1})\n")
                for cand, r in zip(grid, sweep):
                    if player == 0:
                        u, A_pld, A_cert, p = r["u1"], r["A_pld1"], r["A_cert1"], r["p1"]
                    else:
                        u, A_pld, A_cert, p = r["u2"], r["A_pld2"], r["A_cert2"], r["p2"]
                    f.write(f"{cand:.8g} {u:.8g} {value_fn(r['sigma']):.8g} "
                            f"{A_pld:.8g} {A_cert:.8g} "
                            f"{r['sigma']:.8g} {p:.8g} {int(r['feasible'])}\n")

    traj = dat_dir / "trajectory.dat"
    with open(traj, "w") as f:
        f.write("# step mover eps1 eps2 sigma V p1 p2 u1 u2 "
                "A_pld1 A_pld2 A_cert1 A_cert2\n")
        for st in steps:
            s = st["state"]
            if s is None:
                continue
            f.write(f"{st['t']} {st['mover']+1} "
                    f"{s['eps1']:.8g} {s['eps2']:.8g} {s['sigma']:.8g} "
                    f"{value_fn(s['sigma']):.8g} "
                    f"{s['p1']:.8g} {s['p2']:.8g} {s['u1']:.8g} {s['u2']:.8g} "
                    f"{s['A_pld1']:.8g} {s['A_pld2']:.8g} "
                    f"{s['A_cert1']:.8g} {s['A_cert2']:.8g}\n")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(steps: list[dict], grid: np.ndarray, w: float = 1.0,
                tag: str = ""):
    apply_paper_style()
    ncols = len(steps)
    nrows = 2
    col_w = 2.7
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(col_w * ncols, 5.4), squeeze=False,
    )

    U_C = "#2a9d8f"              # utility of the row's player (teal, left axis)
    V_C = PRIMARY_BLUE           # shared model value V(sigma) (right axis)
    A_PLD_C = PRIMARY_ORANGE     # realised PLD advantage (right axis)
    A_CERT_C = ACCENT_GREY       # certified/contract advantage (right axis)

    for ci, st in enumerate(steps):
        sweep = st["sweep"]
        mover = st["mover"]
        opt_idx = st["opt_idx"]
        prev_idx = st["prev_mover_idx"]
        feas = np.array([r["feasible"] for r in sweep])
        state = st["state"]

        for row_player in (0, 1):
            ax = axes[row_player][ci]
            axr = ax.twinx()

            if row_player == 0:
                u = np.array([r["u1"] for r in sweep])
                A_pld = np.array([r["A_pld1"] for r in sweep])
                A_cert = np.array([r["A_cert1"] for r in sweep])
            else:
                u = np.array([r["u2"] for r in sweep])
                A_pld = np.array([r["A_pld2"] for r in sweep])
                A_cert = np.array([r["A_cert2"] for r in sweep])

            # Grey-hatch infeasible candidate regions.
            for i, fe in enumerate(feas):
                if not fe:
                    lo = grid[i] / (grid[1] / grid[0]) ** 0.5 if i > 0 else grid[i] * 0.9
                    hi = grid[i] * (grid[1] / grid[0]) ** 0.5 if i < len(grid) - 1 else grid[i] * 1.1
                    ax.axvspan(lo, hi, color="0.85", alpha=0.5, hatch="///",
                               edgecolor="0.6", linewidth=0.0, zorder=0)

            V = np.array([value_fn(r["sigma"]) for r in sweep])
            um = np.where(feas, u, np.nan)
            apm = np.where(feas, A_pld, np.nan)
            Vm = np.where(feas, V, np.nan)

            l_u, = ax.plot(grid, um, "-", color=U_C, lw=1.6, zorder=4)
            l_v, = axr.plot(grid, Vm, "-", color=V_C, lw=1.3, zorder=3)
            l_apld, = axr.plot(grid, apm, "-", color=A_PLD_C, lw=1.3,
                               zorder=4)
            l_acert, = axr.plot(grid, A_cert, "--", color=A_CERT_C, lw=1.2,
                                zorder=3)

            # Non-mover: previous-step utility level as a dotted teal line.
            if row_player != mover and not st["is_initial"] and ci > 0:
                prev_state = steps[ci - 1]["state"]
                if prev_state is not None:
                    u_prev = prev_state["u1" if row_player == 0 else "u2"]
                    ax.axhline(u_prev, color=U_C, ls=":", lw=1.2, zorder=3)

            ax.set_xscale("log")
            axr.set_ylim(-0.03, 1.03)

            # Mover's optimal next choice (argmax over feasible utilities),
            # drawn only in the mover's own row.
            if (opt_idx is not None and not st["is_initial"]
                    and row_player == mover):
                ax.axvline(grid[opt_idx], color="crimson", ls="--", lw=1.3,
                           zorder=5)
                ax.plot([grid[opt_idx]], [um[opt_idx]], marker="*",
                        color="crimson", markersize=11, zorder=6,
                        markeredgecolor="white", markeredgewidth=0.5)
            # Row player's own eps before the step (dotted grey): the mover's
            # pre-move action in the mover's row, the held-fixed action in
            # the non-mover's row.
            if row_player == mover:
                own_idx = prev_idx
            else:
                own_idx = st["idx1"] if row_player == 0 else st["idx2"]
            ax.axvline(grid[own_idx], color="0.45", ls=":", lw=1.1, zorder=2)

            ax.set_xlim(grid[0] * 0.9, grid[-1] * 1.1)
            ax.tick_params(axis="y", labelcolor=U_C)

            # Left ylabel only on first column; right ylabel only on last.
            if ci == 0:
                ax.set_ylabel(rf"$u_{{{row_player+1}}}$", color=U_C)
            if ci == ncols - 1:
                axr.set_ylabel(r"$V(\sigma)$, advantage")
            else:
                axr.set_yticklabels([])
            if row_player == 1:
                ax.set_xlabel(rf"P{mover+1} candidate $\varepsilon_{{{mover+1}}}$")

            # Row label on the far left, outside.
            if ci == 0:
                ax.annotate(
                    f"Player {row_player+1}", xy=(-0.42, 0.5),
                    xycoords="axes fraction", rotation=90, va="center",
                    ha="center", fontsize=10, fontweight="bold",
                    color=(PRIMARY_BLUE if row_player == 0 else PRIMARY_ORANGE),
                )

        # Column title on the top axis.
        if st["is_initial"]:
            title = "Initial state"
        elif st.get("aborted"):
            title = f"Step {st['t']}: P{mover+1} (no feasible move)"
        else:
            verb = "best-responds" if st["moved"] else "stays (converged)"
            title = f"Step {st['t']}: P{mover+1} {verb}"
        axes[0][ci].set_title(title, fontsize=9)

        # Post-step state annotation below the bottom row.
        if state is not None:
            txt = (rf"$(\varepsilon_1,\varepsilon_2)=({state['eps1']:.2g},"
                   rf"{state['eps2']:.2g})$" "\n"
                   rf"$\sigma={state['sigma']:.2f}$" "\n"
                   rf"$u_1={state['u1']:.3f},\ A_1={state['A_pld1']:.3f}$" "\n"
                   rf"$u_2={state['u2']:.3f},\ A_2={state['A_pld2']:.3f}$")
        else:
            txt = "no feasible move"
        axes[1][ci].annotate(
            txt, xy=(0.5, -0.42), xycoords="axes fraction", ha="center",
            va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="0.96", ec="0.7", lw=0.5),
        )

    # Single shared legend.
    legend_elems = [
        Line2D([0], [0], color=U_C,
               label=r"utility $u_i = V - \lambda_i A_{\mathrm{PLD},i}$"
                     " (left axis)"),
        Line2D([0], [0], color=U_C, ls=":",
               label="non-mover's utility before the step"),
        Line2D([0], [0], color=V_C,
               label=r"model value $V(\sigma)$ (shared, right axis)"),
        Line2D([0], [0], color=PRIMARY_ORANGE,
               label=r"realised $A_{\mathrm{PLD},i}$ (vulnerability, right axis)"),
        Line2D([0], [0], color=ACCENT_GREY, ls="--",
               label=r"certified $A_{\varepsilon,\delta,i}$ (contract, right axis)"),
        Line2D([0], [0], color="crimson", ls="--", marker="*", markersize=9,
               label="mover's optimal next choice (mover's row)"),
        Line2D([0], [0], color="0.45", ls=":",
               label="row player's $\\varepsilon_i$ before the step"),
        Patch(facecolor="0.85", hatch="///", edgecolor="0.6",
              label="infeasible candidate"),
    ]
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
               fontsize=8)

    fig.suptitle(
        "Best-response dynamics of the symmetric 2-player Privacy Externality "
        f"Game, $\\lambda_1 = \\lambda_2 = {w:g}$\n"
        "(each column: the mover sweeps its budget; the non-mover's "
        "contract $A_{\\varepsilon,\\delta}$ is flat but its realised "
        "$A_{\\mathrm{PLD}}$ still moves — the externality)",
        fontsize=9.5, y=1.075,
    )
    fig.tight_layout(rect=(0.03, 0.06, 1.0, 0.95))
    fig.subplots_adjust(wspace=0.32, hspace=0.55)

    pdf = FIG_DIR / f"fig_br_trace{tag}.pdf"
    svg = FIG_DIR / f"fig_br_trace{tag}.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    png = FIG_DIR / f"fig_br_trace{tag}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return pdf, svg, png


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=22)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--w", type=float, default=1.0,
                    help="symmetric privacy weight lambda_1 = lambda_2")
    ap.add_argument("--batch", type=int, default=128,
                    help="expected batch size (steps scale as 5*M/batch)")
    ap.add_argument("--min-rate", type=float, default=1e-5,
                    help="participation floor for feasibility (0 disables)")
    args = ap.parse_args()

    tag = "" if args.w == 1.0 else f"_lam{args.w:g}"
    if args.K != 22:
        tag += f"_K{args.K}"
    if args.batch != 128:
        tag += f"_B{args.batch}"
    if args.min_rate != 1e-5:
        tag += "_nofloor" if args.min_rate == 0 else f"_mr{args.min_rate:g}"
    game = build_game(K=args.K, w=args.w, batch=args.batch,
                      min_rate=args.min_rate)
    grid = game.grid()
    print(f"[br_trace] K={args.K} grid eps in [{grid[0]:.3g},{grid[-1]:.3g}], "
          f"steps={game.steps}, batch={args.batch}, lambda={args.w:g}, "
          f"start idx={args.K // 2} (eps={grid[args.K // 2]:.4g})")

    steps, (idx1, idx2) = run_dynamics(game, grid, max_steps=args.max_steps)

    dat_dir = FIG_DIR / "data" / f"br_trace{tag}"
    write_dat_files(steps, grid, dat_dir=dat_dir)
    pdf, svg, png = make_figure(steps, grid, w=args.w, tag=tag)

    # Convergence report.
    print("\n[br_trace] TRAJECTORY")
    for st in steps:
        s = st["state"]
        if st["is_initial"]:
            print(f"  step {st['t']} INITIAL  "
                  f"eps=({grid[st['idx1']]:.3g},{grid[st['idx2']]:.3g})")
        elif s is not None:
            print(f"  step {st['t']} P{st['mover']+1} "
                  f"{'moved' if st['moved'] else 'stayed'} "
                  f"-> eps=({s['eps1']:.3g},{s['eps2']:.3g}) "
                  f"sigma={s['sigma']:.3f} "
                  f"u=({s['u1']:.4f},{s['u2']:.4f}) "
                  f"A_pld=({s['A_pld1']:.4f},{s['A_pld2']:.4f}) "
                  f"A_cert=({s['A_cert1']:.4f},{s['A_cert2']:.4f})"
                  f"{'  [TIE]' if st['tie'] else ''}")
        else:
            print(f"  step {st['t']} P{st['mover']+1} NO FEASIBLE MOVE")

    last = steps[-1]
    last_moved = last.get("moved", False)
    converged = (not last_moved) and (not last.get("aborted", False))
    fin = eval_profile(game, float(grid[idx1]), float(grid[idx2]))
    print(f"\n[br_trace] converged={converged} in {last['t']} steps")
    print(f"[br_trace] final (eps1*,eps2*)=({fin['eps1']:.4g},{fin['eps2']:.4g}) "
          f"symmetric={abs(fin['eps1']-fin['eps2'])<1e-9}")
    print(f"[br_trace] final sigma={fin['sigma']:.4f} "
          f"u=({fin['u1']:.4f},{fin['u2']:.4f}) "
          f"A_pld=({fin['A_pld1']:.4f},{fin['A_pld2']:.4f}) "
          f"A_cert=({fin['A_cert1']:.4f},{fin['A_cert2']:.4f})")
    print(f"[br_trace] figures: {pdf}\n           {svg}\n           {png}")
    print(f"[br_trace] dat dir: {dat_dir}")


if __name__ == "__main__":
    main()
