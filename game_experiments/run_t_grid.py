"""T-grid composition study on the exact ladder (multi_step_game).

For each T and each diagonal budget e1: parametrize player 2's column
in physical space by the rate p2 (p1 = PSUM - p2 fixed by the batch),
root-find the shared sigma from player 1's binding budget
E_T(sigma, p1) = e1 (warm-started along the column), then read player
2's realised budget e2 = E_T(sigma, p2) and advantage
A2 = delta_T(0; sigma, p2) off the same ladder. This avoids every
nested (eps1, eps2) inversion: one 1-D root per column point, each
evaluation one exact ladder.

The stored rows post-process into u2(e2 | e1) = V(sigma) - lam * A2
for ANY lam, giving the symmetric-deviation gap
  gap(T, lam; e1) = max_col u2 - u2(diagonal)
whose zero crossing in T is the onset of symmetric interior equilibria
(nummeric_eval.md Secs. 14-16).

Env: PEG_TS, PEG_NDIAG, PEG_NP, PEG_N_WORKERS, PEG_OUT, PEG_G.
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

from game_experiments.multi_step_game import (  # noqa: E402
    DELTA, PBAR, PSUM, Ladder,
)

TS = [int(x) for x in os.environ.get(
    "PEG_TS", "1,2,3,4,6,8,12,16,24,32,48").split(",")]
N_DIAG = int(os.environ.get("PEG_NDIAG", "12"))
N_P = int(os.environ.get("PEG_NP", "30"))
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
N_E = int(os.environ.get("PEG_NE", "3000"))
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "50_t_grid.pkl")

DIAG_ES = (np.array([float(x) for x in os.environ["PEG_DIAG_ES"].split(",")])
           if "PEG_DIAG_ES" in os.environ
           else np.geomspace(0.5, 64.0, N_DIAG))


def sigma_for_budget(e1: float, p1: float, T: int,
                     sig0: float | None = None) -> float | None:
    """sigma with E_T(sigma, p1) = e1; E decreasing in sigma."""
    f = lambda s: Ladder(s, p1, T, n_e=N_E).eps_budget() - e1
    if sig0 is not None:
        a, b = sig0 / 1.3, sig0 * 1.3
        fa, fb = f(a), f(b)
        for _ in range(40):
            if fa > 0 > fb:
                return brentq(f, a, b, rtol=1e-9)
            if fa <= 0:
                a /= 1.6
                fa = f(a)
            else:
                b *= 1.6
                fb = f(b)
    a, b = 1e-2, 1e3
    if f(a) <= 0 or f(b) >= 0:
        return None
    return brentq(f, a, b, rtol=1e-9)


def column(task) -> dict:
    T, e1 = task
    p_half = PSUM / 2.0
    p2s = np.unique(np.append(
        np.geomspace(PBAR, PSUM - PBAR, N_P), p_half))
    rows, sig_prev = [], None
    for p2 in p2s:
        p1 = PSUM - p2
        sig = sigma_for_budget(e1, p1, T, sig_prev)
        if sig is None:
            rows.append(dict(p2=float(p2), ok=False))
            continue
        sig_prev = sig
        lad2 = Ladder(sig, p2, T, n_e=N_E)
        rows.append(dict(p2=float(p2), ok=True, sigma=float(sig),
                         e2=float(lad2.eps_budget()),
                         A2=float(lad2.adv()),
                         diag=bool(abs(p2 - p_half) < 1e-15)))
    dg = [r for r in rows if r.get("diag")]
    chk = abs(dg[0]["e2"] - e1) / e1 if dg and dg[0]["ok"] else np.nan
    print(f"T={T} e1={e1:.3f}: {sum(r['ok'] for r in rows)}/{len(rows)} ok, "
          f"diag consistency {chk:.2e}", flush=True)
    return dict(T=T, e1=float(e1), rows=rows, diag_check=float(chk))


def main() -> None:
    tasks = [(T, float(e)) for T in TS for e in DIAG_ES]
    # large T first so the long poles start immediately
    tasks.sort(key=lambda t: -t[0])
    cols = []
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(column, t) for t in tasks]
        for f in as_completed(futs):
            cols.append(f.result())
            # checkpoint after every column so a timeout loses nothing
            out = {}
            for c in cols:
                out.setdefault(c["T"], []).append(c)
            pickle.dump(dict(ts=TS, diag_es=DIAG_ES, psum=PSUM,
                             delta=DELTA, columns=out,
                             complete=len(cols) == len(tasks)),
                        open(OUT, "wb"))
    print(f"wrote {OUT} ({len(cols)}/{len(tasks)} columns)")


if __name__ == "__main__":
    main()
