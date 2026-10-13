"""Exact lambda sweep of the single-step (T=1) PEG.

The closed-form FOC is linear in the privacy weight:
    du2/deps2 (eps2; lam) = Vterm(eps1, eps2) - lam * Advterm(eps1, eps2),
so one dense component sample per column serves every lam. Phase 1
samples (Vterm, Advterm, V, Adv) on N_SAMP points per column over the
box-clipped feasible interval. Phase 2, for each lam in LAMS: locate
sign changes of the reconstructed FOC, Brent-polish each root exactly,
value-compare interior maxima against the exact corner utilities, and
store the global best response per column. Equilibria per lam follow in
post-processing.

Writes tasks_findings/_scripts/38_t1_lam_sweep.pkl.
Run via slurm_scripts/run_t1_lam_sweep.sh.
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game_experiments.single_step_game import (  # noqa: E402
    G1, G2, advantage, dE, solve, value,
)
from game_experiments.single_step_br import feasible_interval  # noqa: E402

LAMS = [float(s) for s in os.environ.get(
    "PEG_LAMS", "1,2,4,8,12,16,20,24,26,28,30,32,36,40,50,80").split(",")]
N_COLS = int(os.environ.get("PEG_N_COLS", "300"))
N_SAMP = int(os.environ.get("PEG_N_SAMP", "140"))
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
LOACT, HIACT = 0.5, 64.0
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "38_t1_lam_sweep.pkl")

COLS = np.geomspace(LOACT, HIACT, N_COLS)


def components(e1: float, e2: float):
    """(Vterm, Advterm, V, Adv): FOC = Vterm - lam*Advterm; u = V - lam*Adv."""
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, p1, p2 = r
    Fs1, Fp1, _ = dE(sigma, p1)
    Fs2, Fp2, _ = dE(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    As = -p2 * norm.pdf(0.5 / sigma) / sigma**2
    Ap = 2.0 * norm.cdf(0.5 / sigma) - 1.0
    return (Vp * ds, As * ds + Ap * dp2, value(sigma), advantage(sigma, p2))


def sample_column(e1: float) -> dict:
    e2_lo, e2_hi = feasible_interval(e1)
    lo = max(e2_lo * (1 + 1e-9), LOACT)
    hi = min(e2_hi * (1 - 1e-9), HIACT)
    xs = np.geomspace(lo, hi, N_SAMP)
    comp = np.full((N_SAMP, 4), np.nan)
    for k, e2 in enumerate(xs):
        c = components(e1, float(e2))
        if c is not None:
            comp[k] = c
    return dict(e1=e1, lo=lo, hi=hi, xs=xs, comp=comp)


def br_for_lam(args):
    """Global BR of one sampled column at one lam, exact polish."""
    col, lam = args
    e1, xs, comp = col["e1"], col["xs"], col["comp"]
    D = comp[:, 0] - lam * comp[:, 1]
    U = comp[:, 2] - lam * comp[:, 3]

    def Dfun(e2):
        c = components(e1, e2)
        return c[0] - lam * c[1]

    cands = []
    for k in range(len(xs) - 1):
        if np.isfinite(D[k]) and np.isfinite(D[k + 1]) \
                and D[k] > 0 >= D[k + 1]:
            z = brentq(Dfun, xs[k], xs[k + 1], xtol=1e-12, rtol=8.9e-16)
            c = components(e1, z)
            cands.append((z, c[2] - lam * c[3], "interior"))
    if np.isfinite(U[0]):
        cands.append((xs[0], U[0], "corner_lo"))
    if np.isfinite(U[-1]):
        cands.append((xs[-1], U[-1], "corner_hi"))
    if not cands:
        return None
    z, u, kind = max(cands, key=lambda c: c[1])
    return dict(e1=e1, br=z, u=u, kind=kind,
                D_lo=float(D[np.isfinite(D)][0]),
                D_hi=float(D[np.isfinite(D)][-1]))


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        cols = list(pool.map(sample_column, [float(e) for e in COLS],
                             chunksize=2))
    print(f"phase 1 done: {len(cols)} columns x {N_SAMP} samples")

    sweep = {}
    for lam in LAMS:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            rows = list(pool.map(br_for_lam, [(c, float(lam)) for c in cols],
                                 chunksize=4))
        rows = [r for r in rows if r is not None]
        kinds = [r["kind"] for r in rows]
        sweep[lam] = rows
        print(f"lam={lam:5.1f}: corner_hi {kinds.count('corner_hi'):3d} "
              f"interior {kinds.count('interior'):3d} "
              f"corner_lo {kinds.count('corner_lo'):3d}", flush=True)

    pickle.dump(dict(lams=LAMS, cols_e1=COLS, sweep=sweep, samples=cols),
                open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
