"""The single-step (T=1) PEG: fully closed-form model.

At one composition step every object of the model
(tasks_findings/33_u2_analytic_model.md) is elementary. With
Phi/phi the standard normal CDF/PDF and the threshold

  x_eps = 1/2 + sigma^2 log((e^eps - 1 + p)/p)      (e^eps > 1 - p),

the exact privacy profile of the Poisson-subsampled Gaussian is

  delta(eps; sigma, p) = (1-p) Phi(-x/sigma) + p Phi((1-x)/sigma)
                         - e^eps Phi(-x/sigma),

its partials (hockey-stick boundary terms vanish) are

  d delta/d eps   = -e^eps Phi(-x/sigma)
  d delta/d p     =  Phi((1-x)/sigma) - Phi(-x/sigma)
  d delta/d sigma = (x/sigma^2) phi(x/sigma) (1 - p - e^eps)
                    + p ((x-1)/sigma^2) phi((x-1)/sigma),

and the advantage is Adv(sigma, p) = p (2 Phi(1/(2 sigma)) - 1) with
elementary partials. The budget function E(sigma, p) inverts delta in
eps; the mechanism map solves the two-budget + batch system. The only
numerics anywhere are Brent root-finds of strictly monotone elementary
functions (machine precision).

Utility: u_i = 1/(1+sigma^2) - lam * Adv(sigma, p_i).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Phi, phi = norm.cdf, norm.pdf

import os

G1 = G2 = int(os.environ.get("PEG_G", "5000"))
B = 128
PSUM = B / G1
DELTA = 1e-12
LAM = 1.0
PBAR = 1e-5


# ---- exact profile (log-safe) ------------------------------------------
def _log_t(eps: float, p: float) -> float:
    """log(e^eps - 1 + p), stable for large eps."""
    if eps > 30.0:
        return eps + math.log1p((p - 1.0) * math.exp(-eps))
    return math.log(math.exp(eps) - 1.0 + p)


def x_thresh(eps: float, sigma: float, p: float) -> float:
    return 0.5 + sigma * sigma * (_log_t(eps, p) - math.log(p))


def delta(eps: float, sigma: float, p: float) -> float:
    x = x_thresh(eps, sigma, p)
    la = norm.logcdf(-x / sigma)          # log Phi(-x/sigma)
    tail = math.exp(eps + la) if eps + la > -745.0 else 0.0
    lead = math.exp(la) if la > -745.0 else 0.0
    return (1.0 - p) * lead + p * Phi((1.0 - x) / sigma) - tail


def ddelta_deps(eps: float, sigma: float, p: float) -> float:
    x = x_thresh(eps, sigma, p)
    la = norm.logcdf(-x / sigma)
    return -(math.exp(eps + la) if eps + la > -745.0 else 0.0)


def ddelta_dp(eps: float, sigma: float, p: float) -> float:
    x = x_thresh(eps, sigma, p)
    return Phi((1.0 - x) / sigma) - Phi(-x / sigma)


def ddelta_dsigma(eps: float, sigma: float, p: float) -> float:
    x = x_thresh(eps, sigma, p)
    lphi = norm.logpdf(x / sigma)
    # (1 - p - e^eps) phi(x/sigma): the e^eps part log-safely
    t1 = (1.0 - p) * (math.exp(lphi) if lphi > -745.0 else 0.0)
    t1 -= math.exp(eps + lphi) if eps + lphi > -745.0 else 0.0
    return ((x / sigma**2) * t1
            + p * ((x - 1.0) / sigma**2) * phi((x - 1.0) / sigma))


# ---- budget function E(sigma, p) and its inverses ----------------------
def eps_budget(sigma: float, p: float) -> float:
    """E(sigma,p): the eps at which delta(eps) = DELTA."""
    f = lambda e: delta(e, sigma, p) - DELTA
    lo = 1e-6
    if f(lo) < 0:                     # already below target at eps ~ 0
        return lo
    # the root sits near x_eps ~ 1 + O(sigma); bracket adaptively
    hi = (0.5 + 8.0 * sigma) / (sigma * sigma) + 100.0
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16)


def p_for_eps(sigma: float, eps: float) -> float:
    """phi(sigma, eps): rate at which noise sigma exhausts exactly eps."""
    f = lambda p: delta(eps, sigma, p) - DELTA
    lo, hi = 1e-12, 1.0 - 1e-12
    if f(hi) < 0:
        return 1.0
    if f(lo) > 0:
        return 0.0
    return brentq(f, lo, hi, xtol=1e-16, rtol=8.9e-16)


def dE(sigma: float, p: float):
    """(dE/dsigma, dE/dp) by the implicit function theorem, closed form."""
    e = eps_budget(sigma, p)
    de = ddelta_deps(e, sigma, p)
    return (-ddelta_dsigma(e, sigma, p) / de,
            -ddelta_dp(e, sigma, p) / de, e)


# ---- mechanism map -----------------------------------------------------
def solve(e1: float, e2: float):
    """(sigma*, p1*, p2*) of the two-budget + batch system; None if the
    batch cannot be met or a floor is violated."""
    def excess(sigma):
        return (G1 * p_for_eps(sigma, e1) + G2 * p_for_eps(sigma, e2) - B)
    lo, hi = 1e-2, 1e3
    if excess(lo) > 0 or excess(hi) < 0:
        return None
    sigma = brentq(excess, lo, hi, xtol=1e-13, rtol=8.9e-16)
    p1, p2 = p_for_eps(sigma, e1), p_for_eps(sigma, e2)
    if p1 < PBAR or p2 < PBAR:
        return None
    return sigma, p1, p2


# ---- utility and its exact slice derivative ----------------------------
def value(sigma: float) -> float:
    return 1.0 / (1.0 + sigma * sigma)


def advantage(sigma: float, p: float) -> float:
    return p * (2.0 * Phi(0.5 / sigma) - 1.0)


def u2(e1: float, e2: float):
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, _, p2 = r
    return value(sigma) - LAM * advantage(sigma, p2)


def du2_deps2(e1: float, e2: float):
    """Exact FOC: closed-form partials, implicit-function-theorem chain."""
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, p1, p2 = r
    Fs1, Fp1, _ = dE(sigma, p1)
    Fs2, Fp2, _ = dE(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    As = -p2 * phi(0.5 / sigma) / sigma**2
    Ap = 2.0 * Phi(0.5 / sigma) - 1.0
    return Vp * ds - LAM * (As * ds + Ap * dp2)


if __name__ == "__main__":
    # sanity: closed-form chain vs direct finite difference of u2
    for (e1, e2) in [(3.0, 3.0), (2.0, 6.0), (8.0, 1.5)]:
        r = solve(e1, e2)
        if r is None:
            print(f"({e1}, {e2}): infeasible")
            continue
        sigma, p1, p2 = r
        # consistency: budgets reproduced to machine precision
        b1, b2 = eps_budget(sigma, p1), eps_budget(sigma, p2)
        h = e2 * 1e-6
        up, um = u2(e1, e2 + h), u2(e1, e2 - h)
        fd = (up - um) / (2 * h) if (up is not None and um is not None) else float("nan")
        an = du2_deps2(e1, e2)
        print(f"({e1:4.1f},{e2:4.1f}) sigma={sigma:.6f} p1={p1:.6f} "
              f"p2={p2:.6f} | eps check {abs(b1-e1):.1e}/{abs(b2-e2):.1e} "
              f"| dU analytic {an:+.8e} FD {fd:+.8e}")
