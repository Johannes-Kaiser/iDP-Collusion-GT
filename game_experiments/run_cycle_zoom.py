"""Super-resolution mechanism grid on the cycle window.

Recomputes the paper-instance mechanism (T = 390, RDP accounting,
|G| = 5000x5000, B = 128, delta = 1e-12) on a fine eps grid restricted
to the window where the lambda = 2 undercutting cycle lives, to test
whether the 3-cycle's period and amplitude are K = 200 grid artifacts
or properties of the (accountant-defined) game. The window BR is only
valid while argmaxes stay interior to the window — the analysis must
check for boundary hits.

Env: PEG_K (default 220), PEG_EMIN/PEG_EMAX (0.90 / 1.20),
PEG_N_WORKERS, PEG_OUT (55_cycle_zoom.pkl).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.continuous import precompute_mechanism_grid  # noqa: E402

K = int(os.environ.get("PEG_K", "220"))
EMIN = float(os.environ.get("PEG_EMIN", "0.90"))
EMAX = float(os.environ.get("PEG_EMAX", "1.20"))
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "55_cycle_zoom.pkl")

if __name__ == "__main__":
    grid = precompute_mechanism_grid(
        K=K, group_sizes=(5000, 5000), expected_batch_size=128,
        steps=390, delta=1e-12, eps_min=EMIN, eps_max=EMAX,
        n_workers=N_WORKERS)
    pickle.dump(grid, open(OUT, "wb"))
    print(f"wrote {OUT} (K={K}, window [{EMIN}, {EMAX}])")
