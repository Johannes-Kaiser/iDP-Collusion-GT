"""Dense-Renyi-order recompute of the K=200 mechanism grid.

The 89->761-order zoom comparison (55 vs 56, nummeric_eval.md Sec. 20.3
verdict) showed the deployed game's sawtooth is order-grid
discretization: payoffs shift by up to 8e-4, tooth amplitude shrinks
~3x, and the lambda=2 zoom-window cycle collapses onto pure NE. The
main K=200 cache is lambda-independent (it stores sigma and A), so one
dense recompute re-derives every lambda-dependent NE claim with the
finer accountant. Saved under a *_dense tag; the 89-order cache and
its consumers are untouched.

Run with PEG_RDP_DENSE=1 (game/mechanism.py switches the order grid).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import precompute_mechanism_grid

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0
GROUP_SIZES = (5000, 5000)
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)
K = 200

if __name__ == "__main__":
    if os.environ.get("PEG_RDP_DENSE", "0") != "1":
        sys.exit("refusing to run without PEG_RDP_DENSE=1 (would duplicate "
                 "the 89-order cache under a _dense name)")
    n_workers = int(os.environ.get("PEG_N_WORKERS", "1"))
    tag = (f"mech_grid_K{K}_g{GROUP_SIZES[0]}x{GROUP_SIZES[1]}"
           f"_b{EXPECTED_BATCH}_e{EPOCHS}_dense")
    path = CACHE_DIR / f"{tag}.pkl"
    if path.exists():
        sys.exit(f"{path.name} already exists; delete it first (Rule 2)")
    print(f"[mech-grid-dense] computing K={K} with {n_workers} workers...")
    t0 = time.time()
    grid = precompute_mechanism_grid(
        K=K, group_sizes=GROUP_SIZES,
        expected_batch_size=EXPECTED_BATCH,
        steps=STEPS, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX,
        n_workers=n_workers,
    )
    dt = time.time() - t0
    n_feasible = int(grid["feasible"].sum())
    print(f"[mech-grid-dense] done in {dt:.1f}s  "
          f"feasible cells: {n_feasible}/{K*K}")
    pickle.dump(grid, open(path, "wb"))
    print(f"wrote {path}")
