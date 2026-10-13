"""The PEG in the Gaussian-DP large-T limit: fully closed form.

By the f-DP central limit theorem (Dong, Roth, Su 2022; Bu, Dong,
Long, Su 2020), the T-fold composition of the (p, sigma)-subsampled
Gaussian converges to mu-GDP with

    mu(sigma, p, T) = p * sqrt( T * (e^{1/sigma^2} - 1) ),

and mu-GDP has the exact closed-form profile

    delta(eps; mu) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2),

with attacker advantage Adv = delta(0) = 2 Phi(mu/2) - 1. Every game
object is therefore elementary again, exactly as at T=1: budgets are
1-D Brent roots of the closed form, the mechanism map is the usual
two-budgets-plus-batch system, and the whole column/gap machinery of
run_t_grid runs in microseconds per evaluation. The CLT error is
O(1/sqrt(T)) (Berry-Esseen type), so this is the *analytic limit
model*: exact as T -> infinity, to be validated against the exact
ladder at overlapping T.

__main__ runs the symmetric-deviation-gap study over a grid of very
large T and writes tasks_findings/_scripts/53_gdp_gap.pkl.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Phi = norm.cdf

G1 = G2 = int(os.environ.get("PEG_G", "5000"))
B = 128
PSUM = B / G1
DELTA = 1e-12
PBAR = 1e-5


def mu_of(sigma: float, p: float, T: float) -> float:
    x = 1.0 / (sigma * sigma)
    lex = x if x > 700.0 else math.log(math.expm1(x))
    lmu = math.log(p) + 0.5 * (math.log(T) + lex)
    return math.exp(min(lmu, 690.0))


def delta_gdp(eps: float, mu: float) -> float:
    la = norm.logcdf(-eps / mu - mu / 2.0)
    tail = math.exp(eps + la) if eps + la > -745.0 else 0.0
    return Phi(-eps / mu + mu / 2.0) - tail


def eps_budget(sigma: float, p: float, T: float) -> float:
    """E(sigma, p, T): eps with delta_gdp = DELTA."""
    mu = mu_of(sigma, p, T)
    f = lambda e: delta_gdp(e, mu) - DELTA
    lo = 1e-9
    if f(lo) < 0:
        return lo
    # delta ~ Phi(-eps/mu + mu/2); cap: budgets beyond 1e12 are "infinite"
    hi = min(mu * (7.5 + mu / 2.0) + 1.0, 1e12)
    while f(hi) > 0:
        if hi >= 1e12:
            return 1e12
        hi = min(hi * 2.0, 1e12)
    return brentq(f, lo, hi, xtol=1e-13, rtol=8.9e-16)


def advantage(sigma: float, p: float, T: float) -> float:
    return 2.0 * Phi(mu_of(sigma, p, T) / 2.0) - 1.0


def value(sigma: float) -> float:
    return 1.0 / (1.0 + sigma * sigma)


def sigma_for_budget(e1: float, p1: float, T: float) -> float | None:
    f = lambda s: eps_budget(s, p1, T) - e1
    a, b = 1e-2, 1e5
    if f(a) <= 0 or f(b) >= 0:
        return None
    return brentq(f, a, b, rtol=1e-12)


def column(T: float, e1: float, n_p: int = 31):
    p_half = PSUM / 2.0
    p2s = np.unique(np.append(np.geomspace(PBAR, PSUM - PBAR, n_p), p_half))
    rows = []
    for p2 in p2s:
        sig = sigma_for_budget(e1, PSUM - p2, T)
        if sig is None:
            continue
        rows.append(dict(p2=float(p2), sigma=float(sig),
                         e2=float(eps_budget(sig, p2, T)),
                         A2=float(advantage(sig, p2, T)),
                         diag=bool(abs(p2 - p_half) < 1e-15)))
    return rows


def gap_analysis(T: float, lam: float, diag_es: np.ndarray):
    """min over symmetric candidates of (best deviation - stay)."""
    out = []
    for e1 in diag_es:
        rows = column(T, float(e1))
        dg = [r for r in rows if r["diag"]]
        if not dg:
            continue
        box = [r for r in rows if 0.5 <= r["e2"] <= 64.0] + dg
        us = [value(r["sigma"]) - lam * r["A2"] for r in box]
        ud = value(dg[0]["sigma"]) - lam * dg[0]["A2"]
        out.append((float(e1), float(max(us) - ud)))
    return out


if __name__ == "__main__":
    TS = [float(x) for x in os.environ.get(
        "PEG_TS_GDP", "390,1560,6240,25000,100000,400000").split(",")]
    LAMS = [float(x) for x in os.environ.get(
        "PEG_LAMS", "0.5,1,2,5").split(",")]
    DIAG_ES = np.geomspace(0.5, 64.0, int(os.environ.get("PEG_NDIAG", "12")))
    res = {}
    for T in TS:
        cols = {float(e1): column(T, float(e1)) for e1 in DIAG_ES}
        res[T] = cols
        for lam in LAMS:
            gaps = []
            for e1, rows in cols.items():
                dg = [r for r in rows if r["diag"]]
                if not dg:
                    continue
                box = [r for r in rows if 0.5 <= r["e2"] <= 64.0] + dg
                us = [value(r["sigma"]) - lam * r["A2"] for r in box]
                ud = value(dg[0]["sigma"]) - lam * dg[0]["A2"]
                gaps.append((e1, max(us) - ud))
            if gaps:
                e_min, g_min = min(gaps, key=lambda t: t[1])
                print(f"T={T:9.0f} lam={lam:4.1f}: gap_min={g_min:+.5f} "
                      f"at e*={e_min:.3f}", flush=True)
    out = ROOT / "tasks_findings" / "_scripts" / "53_gdp_gap.pkl"
    pickle.dump(dict(ts=TS, lams=LAMS, diag_es=DIAG_ES, columns=res),
                open(out, "wb"))
    print("wrote", out)
