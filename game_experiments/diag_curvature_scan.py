"""Classify the diagonal of the symmetric 2-player PEG by the sharp
second-order quantity

    Psi = (1/4s^2) [ sigma^2 V'' + 2 sigma V'
                     + lambda (2 p A_p + 2 p sigma A_{p sigma}
                               - p^2 A_pp - sigma^2 A_ss - 2 sigma A_s) ],

the exact own-second-derivative of u_1 along the own action at a
diagonal profile (derived from the aggregator form; verified with
sympy). Psi > 0 means the diagonal cell is a trough of own utility
(no symmetric equilibrium there); Psi < 0 a local max (symmetric
equilibrium possible where the FOC also holds).

Also reports, per diagonal cell, the discrete own-action deviation gain
from the fresh K=200 cache, and a row-monotonicity check of the
externality sign (share of grid rows where A_1 is non-increasing in
the opponent's budget).

Writes tasks_findings/_scripts/28_diag_curvature.pkl + a text summary.
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
from game.mechanism import solve_mechanism

STEPS, DELTA, LAM = 390, 1e-12, 1.0
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
CACHE = ROOT / "game_experiments" / "_cache_v2" / "C6_K200.pkl"
OUT = ROOT / "tasks_findings" / "_scripts" / "28_diag_curvature.pkl"


def V(s: float) -> float:
    return 1.0 / (1.0 + s * s)


def psi_at(eps: float, h: float = 5e-3):
    r = solve_mechanism([eps, eps], [5000, 5000], 128, STEPS, DELTA)
    p, s = float(r.sample_rates[0]), float(r.sigma)
    hp, hs = p * h, s * h
    A0 = adv(s, p, STEPS)
    Apl = adv(s, p + hp, STEPS); Ami = adv(s, p - hp, STEPS)
    Asl = adv(s + hs, p, STEPS); Asm = adv(s - hs, p, STEPS)
    A_p = (Apl - Ami) / (2 * hp)
    A_pp = (Apl - 2 * A0 + Ami) / hp**2
    A_ss = (Asl - 2 * A0 + Asm) / hs**2
    A_s = (Asl - Asm) / (2 * hs)
    A_ps = (adv(s + hs, p + hp, STEPS) - adv(s + hs, p - hp, STEPS)
            - adv(s - hs, p + hp, STEPS) + adv(s - hs, p - hp, STEPS)) \
        / (4 * hp * hs)
    vterm = s * s * (2.0 * (3 * s * s - 1) / (1 + s * s) ** 3) \
        + 2 * s * (-2 * s / (1 + s * s) ** 2)
    risk = LAM * (2 * p * A_p + 2 * p * s * A_ps - p * p * A_pp
                  - s * s * A_ss - 2 * s * A_s)
    bracket = A_p + s * A_ps - p * A_pp
    return dict(eps=eps, p=p, sigma=s, vterm=vterm, risk=risk,
                psi_num=vterm + risk, bracket=bracket)


def main() -> None:
    d = pickle.load(open(CACHE, "rb"))
    actions, U, feas, A = d["actions"], d["sweep"]["U"], \
        d["sweep"]["feasible"], d["sweep"]["A"]
    K = len(actions)

    # discrete own-action deviation gain at every feasible diagonal cell
    gains = np.full(K, np.nan)
    for m in range(K):
        c = U[:, m, 0].copy()
        c[~feas[:, m]] = -np.inf
        if np.isneginf(c[m]):
            continue
        gains[m] = c.max() - c[m]

    # externality sign: share of rows on which A_1(eps_1 fixed) is
    # non-increasing in eps_2 (interior cells only)
    mono = []
    for i in range(K):
        row = A[i, :, 0]
        ok = np.isfinite(row)
        if ok.sum() < 10:
            continue
        dif = np.diff(row[ok])
        mono.append(float((dif <= 1e-9).mean()))
    mono = np.array(mono)

    # Psi at every 4th diagonal action (50 points), parallel
    idx = list(range(0, K, 4))
    eps_list = [float(actions[i]) for i in idx]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        rows = list(pool.map(psi_at, eps_list))

    pickle.dump(dict(actions=actions, gains=gains, mono=mono,
                     psi_rows=rows), open(OUT, "wb"))
    neg = [r for r in rows if r["psi_num"] < 0]
    print(f"externality: mean share of non-increasing steps per row: "
          f"{mono.mean()*100:.1f}% (min {mono.min()*100:.1f}%)")
    print(f"diagonal cells with discrete gain < 1e-6: "
          f"{int((gains < 1e-6).sum())} of {int(np.isfinite(gains).sum())}")
    print(f"Psi < 0 at {len(neg)}/{len(rows)} sampled diagonal points:")
    for r in rows:
        flag = " <-- max window" if r["psi_num"] < 0 else ""
        print(f"  eps={r['eps']:7.3f} sigma={r['sigma']:6.3f} "
              f"vterm={r['vterm']:+.4f} risk={r['risk']:+.4f} "
              f"Psi={r['psi_num']:+.4f} bracket={r['bracket']:+.3f}{flag}")


if __name__ == "__main__":
    main()
