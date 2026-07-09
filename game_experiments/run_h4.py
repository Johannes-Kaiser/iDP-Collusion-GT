"""
H4: aware-target coalition Stackelberg.

Companion to C4 (`run_continuous.experiment_C4`). C4 pins the target at
eps_t = 32 and lets a coalition of size k pick eps_C to maximise the
target's realised MIA advantage A_t. H4 asks the sharper question: does
the attack survive when the target itself best-responds to the coalition?

Regime (mirrors C4 exactly except for the target rule):
  * n = 5 symmetric aware players (w_i = 1, group_size = 5000, batch 128,
    5 epochs, delta = 1e-12). Player 0 is the target.
  * Non-coalition non-target players pinned at eps = 32.
  * Coalition of size k in {1, 2, 3, 4}: all colluders play the same
    eps_C on the log-spaced grid of 14 points in [0.5, 64].
  * Target best-responds by picking eps_t on the same 14-point grid to
    maximise u_t = V(sigma) - w_t A_t.
  * Coalition picks eps_C to maximise the target's realised A_t after
    the target best-response (Stackelberg leader, lambda -> infinity in
    the coalition objective of Def. 5).

Outputs:
  * game_experiments/_cache_v2/H4.pkl  raw sweep results.
  * paper/figures/fig_h4_aware_target.{pdf,svg}  two-panel figure.
  * baseline curve is reloaded from _cache_v2/C4.pkl (passive target).

Parallelism: (k, eps_C, eps_t) tuples are farmed to a ProcessPoolExecutor
of size PEG_N_WORKERS.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import ContinuousGame
from game.utility import PlayerProfile
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()

# --- Scenario constants (mirror run_continuous.py C4) ------------------
DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0
N_PLAYERS = 5
GROUP_SIZE = 5000
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * GROUP_SIZE * N_PLAYERS / EXPECTED_BATCH)
NON_COAL_EPS = 32.0
TARGET = 0
COAL_GRID = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), 14)
TARGET_GRID = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), 14)
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))


def _make_game() -> ContinuousGame:
    players = tuple(PlayerProfile(f"P{i}", w=1.0) for i in range(N_PLAYERS))
    return ContinuousGame(
        players=players,
        group_sizes=tuple([GROUP_SIZE] * N_PLAYERS),
        expected_batch_size=EXPECTED_BATCH,
        steps=STEPS, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX, K=2,   # K=2 dummy: evaluator only
    )


def _eval_one(args):
    """Evaluate a single (k, eps_C, eps_t) point. Returns dict or None."""
    k, ic, ec, it, et = args
    game = _make_game()
    eps_vec = [NON_COAL_EPS] * N_PLAYERS
    eps_vec[TARGET] = float(et)
    for ci in range(1, 1 + k):
        eps_vec[ci] = float(ec)
    ev = game.evaluate(tuple(eps_vec))
    if not ev["feasible"]:
        return None
    return dict(
        k=k, ic=ic, ec=float(ec), it=it, et=float(et),
        A_target=float(ev["advantages"][TARGET]),
        u_target=float(ev["utilities"][TARGET]),
        A_coalition=float(np.mean([ev["advantages"][ci]
                                    for ci in range(1, 1 + k)])) if k > 0 else np.nan,
        sigma=float(ev["sigma"]),
    )


def sweep_aware(k_values=(1, 2, 3, 4)) -> dict:
    """For each k, sweep (eps_C, eps_t) grid; return per-k Stackelberg summary."""
    tasks = []
    for k in k_values:
        for ic, ec in enumerate(COAL_GRID):
            for it, et in enumerate(TARGET_GRID):
                tasks.append((k, ic, float(ec), it, float(et)))
    print(f"[H4] {len(tasks)} evaluations to compute on {N_WORKERS} workers...")

    t0 = time.time()
    if N_WORKERS <= 1:
        raw = [_eval_one(t) for t in tasks]
    else:
        raw = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            fut2i = {pool.submit(_eval_one, t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(fut2i):
                raw[fut2i[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(tasks)} done ({time.time() - t0:.0f}s elapsed)")
    print(f"[H4] sweep done in {time.time() - t0:.1f}s")

    # Aggregate per k:
    #   for each eps_C, find target's best eps_t (max u_target across feasible)
    #   then find coalition's best eps_C (max A_target after target BR)
    per_k = []
    for k in k_values:
        rows = [r for r in raw if r is not None and r["k"] == k]
        # Group by ic (coalition action index)
        target_br = {}   # ic -> dict(et, A_target, u_target)
        for r in rows:
            ic = r["ic"]
            if ic not in target_br or r["u_target"] > target_br[ic]["u_target"]:
                target_br[ic] = r
        # Coalition picks ic that maximises A_target
        if not target_br:
            per_k.append(dict(k=k, feasible=False))
            continue
        best_ic = max(target_br, key=lambda ic: target_br[ic]["A_target"])
        best = target_br[best_ic]
        # Also record the full curve for the appendix
        per_k.append(dict(
            k=k,
            best_eps_C=best["ec"],
            best_eps_t=best["et"],
            A_target=best["A_target"],
            u_target=best["u_target"],
            A_coalition=best["A_coalition"],
            sigma=best["sigma"],
            feasible=True,
            # per-eps_C target-BR curve, for supplementary annotation:
            target_br_curve=sorted(target_br.values(), key=lambda r: r["ec"]),
        ))
    return dict(per_k=per_k, raw=raw,
                coal_grid=COAL_GRID, target_grid=TARGET_GRID,
                n=N_PLAYERS, non_coal_eps=NON_COAL_EPS, target=TARGET)


def run() -> dict:
    cache = CACHE_DIR / "H4.pkl"
    if cache.exists():
        print(f"[H4] loading cached {cache.name}")
        return pickle.load(open(cache, "rb"))
    data = sweep_aware()
    pickle.dump(data, open(cache, "wb"))
    print(f"[H4] wrote {cache.name}")
    return data


# --- Figure ------------------------------------------------------------

def _load_passive() -> dict:
    """Reload C4 passive-target baseline for overlay."""
    p = CACHE_DIR / "C4.pkl"
    if not p.exists():
        return {}
    d = pickle.load(open(p, "rb"))
    return {r["k"]: r for r in d["rows"]}


def plot(data: dict):
    """Single-panel log-y plot of target advantage A_t vs coalition size k
    for the passive-target baseline and the aware-target Stackelberg
    equilibrium. All descriptive content (ratio scale, k=4 corner
    interpretation, k=0 baseline value) lives in the LaTeX caption.
    """
    per_k = [r for r in data["per_k"] if r.get("feasible")]
    if not per_k:
        print("[H4] no feasible aware-target rows; skipping plot")
        return
    passive = _load_passive()
    ks_aware = [r["k"] for r in per_k]
    A_aware = [r["A_target"] for r in per_k]

    # k=4 interior local optimum: exclude the two smallest coalition
    # actions and re-take the target-BR argmax over A_t. Only k=4 has a
    # boundary-vs-interior split; k<4 optima are already interior.
    interior_pts = []  # (k, A_t)
    for r in per_k:
        if r["k"] < 4:
            continue
        interior = [row for row in r["target_br_curve"]
                    if row["ec"] > COAL_GRID[1] * 1.01]
        if not interior:
            continue
        best = max(interior, key=lambda row: row["A_target"])
        interior_pts.append((r["k"], best["A_target"]))

    # Passive curve on k=0..4 (reloaded from C4).
    ks_curve = [0] + ks_aware
    A_passive_curve = [passive.get(k, {}).get("A_target", np.nan)
                       for k in ks_curve]

    fig, ax = plt.subplots(figsize=column_size(rows=1, aspect=0.75))

    ax.plot(ks_curve, A_passive_curve, "s--", color=PRIMARY_ORANGE,
            markersize=4.5, lw=1.1, markeredgewidth=0,
            label=r"passive target")
    ax.plot(ks_aware, A_aware, "o-", color=PRIMARY_BLUE,
            markersize=4.5, lw=1.3, markeredgewidth=0,
            label=r"aware target (Stackelberg)")

    # Hollow marker for k=4 interior local optimum. No legend entry --
    # the caption describes it.
    if interior_pts:
        pk = [p[0] for p in interior_pts]
        pA = [p[1] for p in interior_pts]
        ax.plot(pk, pA, "o", color=PRIMARY_BLUE, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.1)

    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(8e-3, 1.0)
    ax.set_xlim(-0.25, 4.25)
    ax.set_xlabel(r"coalition size $k$")
    ax.set_ylabel(r"target advantage $A_t^\star(k)$")
    ax.legend(frameon=False, loc="center right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_h4_aware_target.pdf")
    fig.savefig(FIG_DIR / "fig_h4_aware_target.svg")
    plt.close()
    print(f"[H4] wrote fig_h4_aware_target.pdf")

    # Log a compact summary usable as caption seed.
    for r in per_k:
        print(f"  k={r['k']:d}  eps_C*={r['best_eps_C']:.2f}  "
              f"eps_t*={r['best_eps_t']:.2f}  A_t={r['A_target']:.3f}  "
              f"passive-A_t={passive.get(r['k'], {}).get('A_target', float('nan')):.3f}")


if __name__ == "__main__":
    data = run()
    plot(data)
