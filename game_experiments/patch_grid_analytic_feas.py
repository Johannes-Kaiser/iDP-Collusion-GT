"""Replace the K=200 mechanism grid's numerical feasibility mask with the
analytic participation frontier, and solve the cells the analytic mask
reclassifies as feasible.

The participation frontier is exact at every point (not just where the
best response sits): if one player's rate is at the floor
``p_bar = 1e-5`` (game.mechanism.solve_mechanism, min_sample_rate), the
batch constraint pins the other player's rate at ``B/|G| - p_bar``, so
the noise ``sigma_bar(eps_loose)`` solves the loose player's budget
alone and the tight player's minimal budget is one accountant
evaluation at ``(sigma_bar, p_bar)``.  A cell ``(eps_i, eps_j)`` is
feasible iff ``eps_i >= f(eps_j)`` and ``eps_j >= f(eps_i)`` with ``f``
that frontier function.  The grid's numerical mask agrees with this
within half a grid cell but is ragged (985 interior holes at K=200)
because it inherits the batch-bisection tolerance; this script makes it
exact.

Cells that the analytic mask declares feasible but the grid never
solved (NaN sigma/A) are solved with the same per-cell pipeline as the
original sweep (game.continuous.ContinuousGame.evaluate), parallelised
over PEG_N_WORKERS processes.

The canonical cache is REPLACED in place (the previous file is kept as
``<name>.pre-analytic-feas.bak`` until downstream regeneration has been
verified, then should be deleted).  Downstream caches (C6/C7/C8,
wsweep, slack panel) must be deleted and regenerated afterwards --- the
NE sets can shift where argmax cells bordered mask holes.

Run through Slurm: slurm_scripts/run_patch_grid_feasA.sh
"""
from __future__ import annotations

import math
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.continuous import ContinuousGame
from game.mechanism import epsilon_rdp
from game.utility import PlayerProfile

CACHE = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"

GROUP_SIZES = (5000, 5000)
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)
DELTA = 1e-12
P_BAR = 1e-5  # solve_mechanism's min_sample_rate
P_LOOSE = EXPECTED_BATCH / GROUP_SIZES[0] - P_BAR

N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))


def frontier_eps_min(eps_loose: float) -> float:
    """Minimal feasible own budget against an opponent at eps_loose.

    Returns eps_min such that own participation binds (p = P_BAR) when
    the opponent fills the rest of the batch (p = P_LOOSE) within their
    budget.  Returns 0.0 when the opponent cannot fill the batch alone
    within eps_loose at any noise level (own rate then exceeds the
    floor automatically).
    """
    lo, hi = 1e-3, 1e5
    if epsilon_rdp(hi, P_LOOSE, STEPS, DELTA) > eps_loose:
        return 0.0
    if epsilon_rdp(lo, P_LOOSE, STEPS, DELTA) < eps_loose:
        lo = 1e-4  # opponent looser than the range needs; extend downward
        if epsilon_rdp(lo, P_LOOSE, STEPS, DELTA) < eps_loose:
            raise RuntimeError(f"sigma bracket failed at eps={eps_loose}")
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if epsilon_rdp(mid, P_LOOSE, STEPS, DELTA) > eps_loose:
            lo = mid
        else:
            hi = mid
    return epsilon_rdp(math.sqrt(lo * hi), P_BAR, STEPS, DELTA)


def _solve_cells(cells):
    game = ContinuousGame(
        players=(PlayerProfile("dummy1", w=1.0),
                 PlayerProfile("dummy2", w=1.0)),
        group_sizes=GROUP_SIZES,
        expected_batch_size=EXPECTED_BATCH,
        steps=STEPS, delta=DELTA,
        eps_min=0.5, eps_max=64.0, K=200,
    )
    actions = game.grid()
    out = []
    for (i, j) in cells:
        ev = game.evaluate((float(actions[i]), float(actions[j])))
        out.append((i, j, ev["sigma"], float(ev["advantages"][0]),
                    float(ev["advantages"][1])))
    return out


def main() -> None:
    grid = pickle.load(open(CACHE, "rb"))
    actions = grid["actions"]
    K = len(actions)
    A, S, feas_old = grid["A"], grid["sigma"], grid["feasible"]

    print(f"[frontier] computing f(eps) at all {K} actions...")
    t0 = time.time()
    f = np.array([frontier_eps_min(float(e)) for e in actions])
    print(f"[frontier] done in {time.time()-t0:.1f}s; "
          f"f(64)={f[-1]:.3f}, f(8)={f[int(np.argmin(np.abs(actions-8)))]:.3f}")

    feas_new = np.zeros((K, K), dtype=bool)
    for i in range(K):
        for j in range(K):
            feas_new[i, j] = (actions[i] >= f[j]) and (actions[j] >= f[i])

    gained = np.argwhere(feas_new & ~feas_old)
    lost = np.argwhere(~feas_new & feas_old)
    print(f"[mask] analytic-feasible: {int(feas_new.sum())} "
          f"(was {int(feas_old.sum())}); gained {len(gained)}, lost {len(lost)}")

    # solve gained cells (upper triangle only; mirror afterwards)
    todo = sorted({(min(i, j), max(i, j)) for i, j in gained
                   if not np.isfinite(S[min(i, j), max(i, j)])})
    print(f"[solve] {len(todo)} upper-triangle cells with {N_WORKERS} workers")
    t0 = time.time()
    chunks = [todo[k::N_WORKERS] for k in range(N_WORKERS)]
    chunks = [c for c in chunks if c]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(_solve_cells, c) for c in chunks]
        done = 0
        for fut in as_completed(futs):
            for (i, j, sig, a0, a1) in fut.result():
                S[i, j] = sig
                A[i, j, 0], A[i, j, 1] = a0, a1
                S[j, i] = sig
                A[j, i, 0], A[j, i, 1] = a1, a0
            done += 1
            print(f"[solve] chunk {done}/{len(chunks)} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    still_nan = int((~np.isfinite(S) & feas_new).sum())
    assert still_nan == 0, f"{still_nan} analytic-feasible cells still unsolved"

    grid["feasible"] = feas_new
    grid["frontier_eps_min"] = f
    bak = CACHE.with_suffix(".pkl.pre-analytic-feas.bak")
    shutil.copy(CACHE, bak)
    pickle.dump(grid, open(CACHE, "wb"))
    print(f"[cache] replaced {CACHE.name} (backup at {bak.name})")


if __name__ == "__main__":
    main()
