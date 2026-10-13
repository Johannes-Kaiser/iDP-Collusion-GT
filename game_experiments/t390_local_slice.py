"""Fine own-budget slice around the diagonal, exact ladder, T = 390.

The 21-point columns of run_t_grid (52/52b/52c) sample player 2's rate
p2 with a factor-1.5 step, so the nearest sampled deviations from the
diagonal sit at e2 ~ 0.8 e1 and ~ 1.6 e1. A spline through those columns
puts the diagonal in a convex trough (paper_audit, 2026-10-12). This
script settles it: for each e1 it evaluates u2 on a dense p2 grid inside
the two neighbouring samples, one exact ladder root per point,
parallelised over points.

Env: PEG_E1S (comma list), PEG_NPTS, PEG_N_WORKERS, PEG_NE, PEG_OUT.
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game_experiments.multi_step_game import PSUM, Ladder  # noqa: E402
from game_experiments import run_t_grid  # noqa: E402

T = 390
E1S = [float(x) for x in os.environ.get("PEG_E1S", "2.548,2.92").split(",")]
NPTS = int(os.environ.get("PEG_NPTS", "13"))
N_E = int(os.environ.get("PEG_NE", "6000"))
run_t_grid.N_E = N_E
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get("PEG_OUT", "83_t390_local_slice.pkl")
SIG0 = 1.17          # diagonal sigma at e1 ~ 2.5-2.9 (52c)


def point(task):
    e1, p2 = task
    sig = run_t_grid.sigma_for_budget(e1, PSUM - p2, T, SIG0)
    if sig is None:
        return dict(e1=e1, p2=p2, ok=False)
    lad = Ladder(sig, p2, T, n_e=N_E)
    r = dict(e1=e1, p2=float(p2), ok=True, sigma=float(sig),
             e2=float(lad.eps_budget()), A2=float(lad.adv()),
             diag=bool(abs(p2 - PSUM / 2) < 1e-15))
    print(f"e1={e1:.3f} p2={p2:.5f} e2={r['e2']:.4f} sigma={sig:.5f} A2={r['A2']:.6f}", flush=True)
    return r


def main():
    p2s = np.unique(np.append(np.geomspace(0.0112, 0.0169, NPTS), PSUM / 2))
    tasks = [(e1, float(p)) for e1 in E1S for p in p2s]
    with ProcessPoolExecutor(int(os.environ.get("PEG_N_WORKERS", "1"))) as pool:
        rows = list(pool.map(point, tasks))
    pickle.dump(dict(T=T, e1s=E1S, psum=PSUM, rows=rows), open(OUT, "wb"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
