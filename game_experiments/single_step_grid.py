"""Full K=200 grid of the closed-form single-step (T=1) PEG.

Per cell: exact mechanism map (machine-precision root-finds of the
closed-form profile), utility u2, and the exact FOC derivative
du2/deps2 assembled from closed-form partials. Player 1's fields follow
by symmetry (transpose). Writes
tasks_findings/_scripts/35_single_step_grid.pkl.
Run via slurm_scripts/run_single_step_grid.sh.
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

from game_experiments.single_step_game import (  # noqa: E402
    G1, G2, LAM, advantage, dE, solve, value,
)

K = 200
ACTIONS = np.geomspace(0.5, 64.0, K)
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
OUT = ROOT / "tasks_findings" / "_scripts" / "35_single_step_grid.pkl"


def cell(args):
    e1, e2 = args
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, p1, p2 = r
    u = value(sigma) - LAM * advantage(sigma, p2)
    Fs1, Fp1, _ = dE(sigma, p1)
    Fs2, Fp2, _ = dE(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    from scipy.stats import norm
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    As = -p2 * norm.pdf(0.5 / sigma) / sigma**2
    Ap = 2.0 * norm.cdf(0.5 / sigma) - 1.0
    D = Vp * ds - LAM * (As * ds + Ap * dp2)
    return sigma, p1, p2, u, D


def main() -> None:
    args = [(float(ACTIONS[i]), float(ACTIONS[j]))
            for i in range(K) for j in range(K)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        res = list(pool.map(cell, args, chunksize=100))
    sig = np.full((K, K), np.nan)
    P2 = np.full((K, K), np.nan)
    U2 = np.full((K, K), np.nan)
    D2 = np.full((K, K), np.nan)
    for k, r in enumerate(res):
        if r is None:
            continue
        i, j = divmod(k, K)
        sig[i, j], _, P2[i, j], U2[i, j], D2[i, j] = r
    feas = np.isfinite(U2)
    print(f"feasible cells: {feas.sum()} / {K*K}")
    pickle.dump(dict(actions=ACTIONS, sigma=sig, P2=P2, U2=U2, D2=D2,
                     feasible=feas),
                open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
