"""Exact slice derivative du2/deps2 on the full K=200 feasible grid.

Per feasible cell (eps1, eps2), using the cached sigma:
  1. recover p2 by inverting the budget function eps_acct(sigma, .) = eps2
     (p1 follows from the batch),
  2. partials Fs, Fp of eps_acct at (sigma, p_i) by central differences on
     the smooth Renyi branch,
  3. implicit-chain derivatives dsigma/deps2, dp2/deps2,
  4. advantage partials from a bivariate spline of Adv(sigma, p) built on
     a smooth log-log table (PLD evaluations, discretization 1e-4),
  5. assemble du2/deps2 = V' dsigma - lam (Adv_s dsigma + Adv_p dp2).

Writes tasks_findings/_scripts/34_exact_foc_grid.pkl with the derivative
field D2 (and D1 by symmetry: D1[i,j] = D2[j,i]), the p2 field, and the
Adv table. Run via slurm_scripts/run_exact_foc_grid.sh.
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

from game.advantage import mia_advantage_subsampled_gaussian as adv
from game.mechanism import epsilon_rdp
from game.utility import value_fn  # noqa: F401

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

CACHE = ROOT / "game_experiments" / "_cache_v2" / \
    "mech_grid_K200_g5000x5000_b128_e5.pkl"
OUT = ROOT / "tasks_findings" / "_scripts" / "34_exact_foc_grid.pkl"

G1 = G2 = 5000
B, T, DELTA = 128, 390, 1e-12
PSUM = B / G1                      # p1 + p2, exact from the batch
LAM = 1.0
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
N_SIG, N_P = 90, 45                # Adv table resolution


def p_for_eps(sigma: float, eps: float) -> float:
    """Invert eps_acct(sigma, p) = eps for p (strictly increasing)."""
    lo, hi = 1e-9, 1.0
    if epsilon_rdp(sigma, hi, T, DELTA) < eps:
        return 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if epsilon_rdp(sigma, mid, T, DELTA) < eps:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def dF(sigma: float, p: float, h_rel: float = 1e-4):
    hs, hp = sigma * h_rel, p * h_rel
    Fs = (epsilon_rdp(sigma + hs, p, T, DELTA)
          - epsilon_rdp(sigma - hs, p, T, DELTA)) / (2 * hs)
    Fp = (epsilon_rdp(sigma, p + hp, T, DELTA)
          - epsilon_rdp(sigma, p - hp, T, DELTA)) / (2 * hp)
    return Fs, Fp


def cell_p2(args):
    sigma, e2 = args
    return p_for_eps(sigma, e2)


def adv_at(args):
    s, p = args
    return adv(s, p, T, 1e-4)


def cell_derivative(args):
    """du2/deps2 at one feasible cell, given the spline lookups."""
    sigma, p2, As, Ap = args
    p1 = PSUM - p2
    if p1 <= 0 or p2 <= 0:
        return np.nan
    Fs1, Fp1 = dF(sigma, p1)
    Fs2, Fp2 = dF(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    return Vp * ds - LAM * (As * ds + Ap * dp2)


def main() -> None:
    grid = pickle.load(open(CACHE, "rb"))
    a = grid["actions"]
    K = len(a)
    feas = grid["feasible"].astype(bool)
    sig = grid["sigma"]
    cells = np.argwhere(feas)
    print(f"{len(cells)} feasible cells")

    # ---- 1. recover p2 per feasible cell (eps2 = a[j]) ---------------
    args = [(float(sig[i, j]), float(a[j])) for i, j in cells]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        p2s = list(pool.map(cell_p2, args, chunksize=64))
    P2 = np.full((K, K), np.nan)
    for (i, j), p2 in zip(cells, p2s):
        P2[i, j] = p2
    print("p2 field done")

    # ---- 2. Adv(sigma, p) table + spline ------------------------------
    smin, smax = np.nanmin(sig[feas]) * 0.9, np.nanmax(sig[feas]) * 1.1
    pmin = max(min(p2s) * 0.9, 1e-7)
    pmax = min(max(p2s) * 1.1, PSUM)
    sgrid = np.geomspace(smin, smax, N_SIG)
    pgrid = np.geomspace(pmin, pmax, N_P)
    tab_args = [(float(s), float(p)) for s in sgrid for p in pgrid]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        vals = list(pool.map(adv_at, tab_args, chunksize=8))
    A_tab = np.array(vals).reshape(N_SIG, N_P)
    print("Adv table done")

    from scipy.interpolate import RectBivariateSpline
    sp = RectBivariateSpline(np.log(sgrid), np.log(pgrid), A_tab, kx=3, ky=3)

    # ---- 3-5. assemble the derivative field ---------------------------
    der_args = []
    for (i, j), p2 in zip(cells, p2s):
        s = float(sig[i, j])
        ls, lp = np.log(s), np.log(p2)
        As = float(sp(ls, lp, dx=1)) / s          # dAdv/dsigma
        Ap = float(sp(ls, lp, dy=1)) / p2         # dAdv/dp
        der_args.append((s, float(p2), As, Ap))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ders = list(pool.map(cell_derivative, der_args, chunksize=64))
    D2 = np.full((K, K), np.nan)
    for (i, j), d in zip(cells, ders):
        D2[i, j] = d
    print("derivative field done")

    pickle.dump(dict(actions=a, D2=D2, P2=P2, feasible=feas,
                     adv_table=dict(sgrid=sgrid, pgrid=pgrid, A=A_tab)),
                open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
