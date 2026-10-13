"""Analytic best-response tracer for the single-step (T=1) PEG.

Per column (fixed eps1, N_COLS log-spaced): the feasible eps2 interval
from the closed-form frontier, a dense sample of the closed-form FOC
derivative, Brent-polished zeros (machine precision), and the exact
global best response as the value-argmax over {interior maxima, both
corners}. Everything is elementary; there is no grid error along the
column and the only numerics are monotone root-finds.

LAM is read from the environment (default 28: all three BR branches
present). Writes tasks_findings/_scripts/36_single_step_br.pkl.
Run via slurm_scripts/run_single_step_br.sh.
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import game_experiments.single_step_game as gm  # noqa: E402
from game_experiments.single_step_game import (  # noqa: E402
    G1, G2, PBAR, PSUM, advantage, dE, eps_budget, p_for_eps, solve, value,
)

LAM = float(os.environ.get("PEG_LAM", "28"))
gm.LAM = LAM
N_COLS = int(os.environ.get("PEG_N_COLS", "800"))
N_SAMP = int(os.environ.get("PEG_N_SAMP", "160"))
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "36_single_step_br.pkl")

COLS = np.geomspace(0.5, 64.0, N_COLS)


def sigma_for_p(e1: float, p: float) -> float:
    """sigma with E(sigma, p) = e1 (E decreasing in sigma)."""
    f = lambda s: eps_budget(s, p) - e1
    return brentq(f, 1e-3, 1e4, xtol=1e-13, rtol=8.9e-16)


def feasible_interval(e1: float):
    """[e2_lo, e2_hi]: player 2 at the floor / player 1 at the floor."""
    s_lo = sigma_for_p(e1, PSUM - PBAR)
    e2_lo = eps_budget(s_lo, PBAR)
    s_hi = sigma_for_p(e1, PBAR)
    e2_hi = eps_budget(s_hi, PSUM - PBAR)
    return e2_lo, e2_hi


def u2_and_D(e1: float, e2: float):
    r = solve(e1, e2)
    if r is None:
        return None, None
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
    return u, Vp * ds - LAM * (As * ds + Ap * dp2)


def trace_column(e1: float) -> dict:
    e2_lo, e2_hi = feasible_interval(e1)
    lo, hi = e2_lo * (1 + 1e-9), e2_hi * (1 - 1e-9)
    xs = np.geomspace(lo, hi, N_SAMP)
    Ds, Us = np.empty(N_SAMP), np.empty(N_SAMP)
    for k, e2 in enumerate(xs):
        u, D = u2_and_D(e1, float(e2))
        Us[k], Ds[k] = (np.nan, np.nan) if u is None else (u, D)
    zeros = []                                   # (e2, u, falling)
    Dfun = lambda e2: u2_and_D(e1, e2)[1]
    for k in range(N_SAMP - 1):
        if np.isfinite(Ds[k]) and np.isfinite(Ds[k + 1]) \
                and Ds[k] * Ds[k + 1] < 0:
            z = brentq(Dfun, xs[k], xs[k + 1], xtol=1e-12, rtol=8.9e-16)
            uz, _ = u2_and_D(e1, z)
            zeros.append((z, uz, bool(Ds[k] > 0)))
    cands = [(z, uz, "interior") for z, uz, fall in zeros if fall]
    u_lo, _ = u2_and_D(e1, lo)
    u_hi, _ = u2_and_D(e1, hi)
    if u_lo is not None:
        cands.append((lo, u_lo, "corner_lo"))
    if u_hi is not None:
        cands.append((hi, u_hi, "corner_hi"))
    br = max(cands, key=lambda c: c[1])
    return dict(e1=e1, e2_lo=e2_lo, e2_hi=e2_hi, zeros=zeros,
                br=br[0], br_u=br[1], kind=br[2],
                cands=cands, D_lo=float(Ds[np.isfinite(Ds)][0]),
                D_hi=float(Ds[np.isfinite(Ds)][-1]))


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        rows = list(pool.map(trace_column, [float(e) for e in COLS],
                             chunksize=4))
    kinds = [r["kind"] for r in rows]
    print(f"lam={LAM}: corner_hi {kinds.count('corner_hi')}, "
          f"interior {kinds.count('interior')}, "
          f"corner_lo {kinds.count('corner_lo')}")
    pickle.dump(dict(lam=LAM, cols=COLS, rows=rows), open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
