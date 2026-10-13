"""Best-response tracer for the two-step (T=2) PEG.

Phase 1: per column (fixed eps1), sample the FOC components along the
feasible eps2 interval. The FOC is linear in the privacy weight,
D(eps2; lam) = Vterm - lam * Advterm, so one component pass serves
every lam. Phase 2: for each lam in PEG_LAMS, locate sign changes of
D, polish them with Brent on the exact chain, and pick the global best
response by value comparison against both corners.

Accuracy: delta_2 by kink-refined composite Gauss-Legendre, certified
against mpmath to <= 3e-9 relative at the budget scale (see
two_step_game.py); every other step is a monotone Brent root-find.

Env: PEG_LAMS (comma list), PEG_N_COLS, PEG_N_SAMP, PEG_N_WORKERS,
PEG_OUT (default 42_two_step_br.pkl), PEG_G.
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

import game_experiments.two_step_game as gm  # noqa: E402
from game_experiments.two_step_game import (  # noqa: E402
    G1, G2, PBAR, PSUM, advantage_all, dE, eps_budget, solve, value,
)

LAMS = [float(x) for x in
        os.environ.get("PEG_LAMS", "0.05,0.7,2,28").split(",")]
N_COLS = int(os.environ.get("PEG_N_COLS", "160"))
N_SAMP = int(os.environ.get("PEG_N_SAMP", "100"))
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "42_two_step_br.pkl")

COLS = np.geomspace(0.5, 64.0, N_COLS)


def sigma_for_p(e1: float, p: float) -> float:
    f = lambda s: eps_budget(s, p) - e1
    return brentq(f, 1e-3, 1e4, xtol=1e-11, rtol=1e-12)


def feasible_interval(e1: float):
    s_lo = sigma_for_p(e1, PSUM - PBAR)
    e2_lo = eps_budget(s_lo, PBAR)
    s_hi = sigma_for_p(e1, PBAR)
    e2_hi = eps_budget(s_hi, PSUM - PBAR)
    return e2_lo, e2_hi


def components(e1: float, e2: float):
    """(V, Adv, Vterm, Advterm) so that u = V - lam*Adv and
    D = Vterm - lam*Advterm; None if infeasible."""
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, p1, p2 = r
    Fs1, Fp1, _ = dE(sigma, p1)
    Fs2, Fp2, _ = dE(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    A, As, Ap = advantage_all(sigma, p2)
    return value(sigma), A, Vp * ds, As * ds + Ap * dp2


def sample_column(e1: float) -> dict:
    e2_lo, e2_hi = feasible_interval(e1)
    lo, hi = e2_lo * (1 + 1e-9), e2_hi * (1 - 1e-9)
    xs = np.geomspace(lo, hi, N_SAMP)
    comp = np.full((N_SAMP, 4), np.nan)
    for k, e2 in enumerate(xs):
        c = components(e1, float(e2))
        if c is not None:
            comp[k] = c
    return dict(e1=e1, e2_lo=e2_lo, e2_hi=e2_hi, xs=xs, comp=comp)


def br_for_lam(col: dict, lam: float) -> dict:
    xs, comp = col["xs"], col["comp"]
    V, A, Vt, At = comp.T
    D = Vt - lam * At
    U = V - lam * A
    Dfun = lambda e2: (lambda c: c[2] - lam * c[3])(
        components(col["e1"], e2))
    Ufun = lambda e2: (lambda c: c[0] - lam * c[1])(
        components(col["e1"], e2))
    zeros = []
    for k in range(len(xs) - 1):
        if np.isfinite(D[k]) and np.isfinite(D[k + 1]) \
                and D[k] * D[k + 1] < 0:
            z = brentq(Dfun, xs[k], xs[k + 1], xtol=1e-9, rtol=1e-10)
            zeros.append((z, Ufun(z), bool(D[k] > 0)))
    cands = [(z, uz, "interior") for z, uz, fall in zeros if fall]
    fin = np.isfinite(U)
    if fin.any():
        ks = np.where(fin)[0]
        cands.append((float(xs[ks[0]]), float(U[ks[0]]), "corner_lo"))
        cands.append((float(xs[ks[-1]]), float(U[ks[-1]]), "corner_hi"))
    br = max(cands, key=lambda c: c[1])
    return dict(e1=col["e1"], e2_lo=col["e2_lo"], e2_hi=col["e2_hi"],
                zeros=zeros, br=br[0], br_u=br[1], kind=br[2],
                cands=cands,
                D_lo=float(D[fin][0]) if fin.any() else np.nan,
                D_hi=float(D[fin][-1]) if fin.any() else np.nan)


def polish_column(col: dict) -> dict:
    return {lam: br_for_lam(col, lam) for lam in LAMS}


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        cols = list(pool.map(sample_column, [float(e) for e in COLS],
                             chunksize=1))
        print("phase 1 done: sampled", len(cols), "columns", flush=True)
        polished = list(pool.map(polish_column, cols, chunksize=1))
    per_lam = {lam: [pc[lam] for pc in polished] for lam in LAMS}
    for lam in LAMS:
        kinds = [r["kind"] for r in per_lam[lam]]
        print(f"lam={lam}: corner_hi {kinds.count('corner_hi')}, "
              f"interior {kinds.count('interior')}, "
              f"corner_lo {kinds.count('corner_lo')}")
    pickle.dump(dict(lams=LAMS, cols=COLS, samples=cols,
                     per_lam=per_lam, G=G1), open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
