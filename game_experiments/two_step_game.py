"""The two-step (T=2) PEG: exact composition by quadrature.

The T-fold privacy profile of the Poisson-subsampled Gaussian obeys the
exact recursion

  delta_T(eps) = E_{X~P}[ delta_{T-1}(eps - g(X)) ],

where P = (1-p) N(0, sigma^2) + p N(1, sigma^2) is the dominating
mixture, g(x) = log((1-p) + p e^{(2x-1)/(2 sigma^2)}) is the per-step
privacy loss, and delta_1 is the closed form of the single-step game
(extended to all real eps: delta_1(eps) = 1 - e^eps for
eps <= log(1-p)). At T=2 the profile is therefore a single 1-D integral
of elementary functions against a Gaussian mixture — no privacy-loss
discretization grid anywhere. Partial derivatives in (eps, p, sigma)
differentiate under the integral into the closed-form delta_1 partials.

Quadrature: per mixture component, composite Gauss-Legendre in the
standardized variable z = (x - mu)/sigma on fixed panels over
[-13, 13], with an extra breakpoint at the integrand kink
(eps - g(x) = log(1-p), where delta_1 switches branch; delta_1 is C^1
there, so derivative-under-the-integral needs no boundary terms). The
truncated tail mass is < 1.3e-38. Accuracy is validated against
adaptive scipy.integrate.quad in __main__.

The advantage of the composed mechanism is Adv_2(sigma, p)
= delta_2(0; sigma, p) (total-variation distance of the pair), so the
same quadrature answers whether composition bends the advantage from
the exactly-linear-in-p single-step form.

Utility and mechanism map mirror single_step_game: two binding budgets
plus the batch constraint; u_i = 1/(1+sigma^2) - LAM * Adv_2(sigma, p_i).
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Phi, phi = norm.cdf, norm.pdf

G1 = G2 = int(os.environ.get("PEG_G", "5000"))
B = 128
PSUM = B / G1
DELTA = 1e-12
LAM = 1.0
PBAR = 1e-5

_ZB = np.array([-13.0, -8.0, -5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 8.0, 13.0])
_GLX, _GLW = np.polynomial.legendre.leggauss(32)


# ---- closed-form delta_1 and partials, vectorized over eps --------------
def _log_t_vec(e: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """(log(e^e - 1 + p), valid-mask); stable at both ends of e."""
    big = e > 30.0
    t = np.where(big, 1.0, np.expm1(np.minimum(e, 30.0)) + p)
    ok = big | (t > 0.0)
    lt = np.where(big,
                  e + np.log1p((p - 1.0) * np.exp(-np.maximum(e, 30.0))),
                  np.log(np.where(ok & ~big, t, 1.0)))
    return lt, ok


def delta1_vec(e: np.ndarray, sigma: float, p: float) -> np.ndarray:
    e = np.asarray(e, float)
    lt, ok = _log_t_vec(e, p)
    x = 0.5 + sigma * sigma * (lt - math.log(p))
    la = norm.logcdf(-x / sigma)
    d = ((1.0 - p) * np.exp(la) + p * Phi((1.0 - x) / sigma)
         - np.exp(np.minimum(e + la, 700.0)))
    return np.where(ok, d, -np.expm1(np.minimum(e, 0.0)))


def ddelta1_deps_vec(e: np.ndarray, sigma: float, p: float) -> np.ndarray:
    e = np.asarray(e, float)
    lt, ok = _log_t_vec(e, p)
    x = 0.5 + sigma * sigma * (lt - math.log(p))
    la = norm.logcdf(-x / sigma)
    return np.where(ok, -np.exp(np.minimum(e + la, 700.0)),
                    -np.exp(np.minimum(e, 0.0)))


def ddelta1_dp_vec(e: np.ndarray, sigma: float, p: float) -> np.ndarray:
    e = np.asarray(e, float)
    lt, ok = _log_t_vec(e, p)
    x = 0.5 + sigma * sigma * (lt - math.log(p))
    return np.where(ok, Phi((1.0 - x) / sigma) - Phi(-x / sigma), 0.0)


def ddelta1_dsigma_vec(e: np.ndarray, sigma: float, p: float) -> np.ndarray:
    e = np.asarray(e, float)
    lt, ok = _log_t_vec(e, p)
    x = 0.5 + sigma * sigma * (lt - math.log(p))
    lphi = norm.logpdf(x / sigma)
    t1 = (1.0 - p) * np.exp(lphi) - np.exp(np.minimum(e + lphi, 700.0))
    d = ((x / sigma**2) * t1
         + p * ((x - 1.0) / sigma**2) * phi((x - 1.0) / sigma))
    return np.where(ok, d, 0.0)


# ---- per-step privacy loss g(x) and partials -----------------------------
def _g_terms(x: np.ndarray, sigma: float, p: float):
    """g, dg/dp, dg/dsigma|_x, dg/dx — all log-safe."""
    a = (2.0 * x - 1.0) / (2.0 * sigma * sigma)
    g = np.logaddexp(math.log1p(-p), math.log(p) + a)
    g_p = np.exp(a - g) - np.exp(-g)          # (e^a - 1) e^{-g}
    pa = np.exp(math.log(p) + a - g)          # p e^a / e^g  in (0, 1]
    g_s = pa * (-2.0 * a / sigma)
    g_x = pa / (sigma * sigma)
    return g, g_p, g_s, g_x


def _kink_x(e: float, sigma: float, p: float):
    """x* with e - g(x*) = log(1-p), or None (delta_1 branch switch)."""
    c = e - math.log1p(-p)                    # need g(x*) = c
    if c <= math.log1p(-p):                   # below range of g
        return None
    # log(e^c - (1-p)) stable both for small and large c
    ls = c + math.log1p(-math.exp(math.log1p(-p) - c))
    return 0.5 + sigma * sigma * (ls - math.log(p))


def _component_nodes(mu: float, e: float, sigma: float, p: float):
    """Gauss-Legendre nodes/weights in z for one mixture component."""
    zb = _ZB
    xk = _kink_x(e, sigma, p)
    if xk is not None:
        zk = (xk - mu) / sigma
        extra = [z for z in (zk - 1.0, zk - 0.25, zk, zk + 0.25, zk + 1.0)
                 if -13.0 < z < 13.0]
        if extra:
            zb = np.unique(np.append(zb, extra))
    lo, hi = zb[:-1], zb[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    z = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
    w = (half[:, None] * _GLW[None, :]).ravel() * phi(z)
    return mu + sigma * z, w, z


def _delta2_all(e: float, sigma: float, p: float):
    """(delta_2, d/deps, d/dp, d/dsigma) in one pass."""
    vals = []
    for mu in (0.0, 1.0):
        x, w, z = _component_nodes(mu, e, sigma, p)
        g, g_p, g_s, g_x = _g_terms(x, sigma, p)
        arg = e - g
        h = delta1_vec(arg, sigma, p)
        h_e = ddelta1_deps_vec(arg, sigma, p)
        h_p = ddelta1_dp_vec(arg, sigma, p) - h_e * g_p
        # x = mu + sigma z, so at fixed z, dx/dsigma = z enters through g
        h_s = (ddelta1_dsigma_vec(arg, sigma, p)
               - h_e * (g_s + g_x * z))
        vals.append((w @ h, w @ h_e, w @ h_p, w @ h_s))
    (I0, E0, P0, S0), (I1, E1, P1, S1) = vals
    d = (1.0 - p) * I0 + p * I1
    de = (1.0 - p) * E0 + p * E1
    dp = (I1 - I0) + (1.0 - p) * P0 + p * P1
    ds = (1.0 - p) * S0 + p * S1
    return d, de, dp, ds


def delta2(e: float, sigma: float, p: float) -> float:
    return _delta2_all(e, sigma, p)[0]


# ---- budget function, mechanism map, utility -----------------------------
def eps_budget(sigma: float, p: float) -> float:
    """E_2(sigma, p): the eps at which delta_2(eps) = DELTA."""
    f = lambda e: delta2(e, sigma, p) - DELTA
    lo = 1e-6
    if f(lo) < 0:
        return lo
    hi = 2.0 * ((0.5 + 8.0 * sigma) / (sigma * sigma) + 100.0)
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16)


def p_for_eps(sigma: float, eps: float) -> float:
    f = lambda p: delta2(eps, sigma, p) - DELTA
    lo, hi = 1e-12, 1.0 - 1e-12
    if f(hi) < 0:
        return 1.0
    if f(lo) > 0:
        return 0.0
    return brentq(f, lo, hi, xtol=1e-16, rtol=8.9e-16)


def dE(sigma: float, p: float):
    """(dE/dsigma, dE/dp, E) by the implicit function theorem."""
    e = eps_budget(sigma, p)
    _, de, dp, ds = _delta2_all(e, sigma, p)
    return -ds / de, -dp / de, e


def solve(e1: float, e2: float):
    def excess(sigma):
        return (G1 * p_for_eps(sigma, e1)
                + G2 * p_for_eps(sigma, e2) - B)
    lo, hi = 1e-2, 1e3
    if excess(lo) > 0 or excess(hi) < 0:
        return None
    sigma = brentq(excess, lo, hi, xtol=1e-13, rtol=8.9e-16)
    p1, p2 = p_for_eps(sigma, e1), p_for_eps(sigma, e2)
    if p1 < PBAR or p2 < PBAR:
        return None
    return sigma, p1, p2


def value(sigma: float) -> float:
    return 1.0 / (1.0 + sigma * sigma)


def advantage(sigma: float, p: float) -> float:
    """Adv_2 = TV of the composed pair = delta_2(0)."""
    return delta2(0.0, sigma, p)


def advantage_all(sigma: float, p: float):
    """(Adv_2, dAdv/dsigma, dAdv/dp)."""
    d, _, dp, ds = _delta2_all(0.0, sigma, p)
    return d, ds, dp


def u2(e1: float, e2: float):
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, _, p2 = r
    return value(sigma) - LAM * advantage(sigma, p2)


def du2_deps2(e1: float, e2: float):
    """Exact FOC via the implicit-function-theorem chain."""
    r = solve(e1, e2)
    if r is None:
        return None
    sigma, p1, p2 = r
    Fs1, Fp1, _ = dE(sigma, p1)
    Fs2, Fp2, _ = dE(sigma, p2)
    ds = (G2 / Fp2) / (G1 * Fs1 / Fp1 + G2 * Fs2 / Fp2)
    dp2 = (1.0 - Fs2 * ds) / Fp2
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    _, As, Ap = advantage_all(sigma, p2)
    return Vp * ds - LAM * (As * ds + Ap * dp2)


if __name__ == "__main__":
    from scipy.integrate import quad

    def delta2_quad(e, sigma, p):
        def integrand(x):
            xa = np.array([x])
            g = _g_terms(xa, sigma, p)[0]
            f = ((1 - p) * phi(x / sigma) + p * phi((x - 1) / sigma)) / sigma
            return f * float(delta1_vec(e - g, sigma, p)[0])
        pts = [0.0, 1.0]
        xk = _kink_x(e, sigma, p)
        lo, hi = -13 * sigma, 1 + 13 * sigma
        if xk is not None and lo < xk < hi:
            pts.append(xk)
        v, err = quad(integrand, lo, hi, points=sorted(pts), limit=400,
                      epsabs=1e-17, epsrel=1e-13)
        return v, err

    print("== delta_2: composite GL vs adaptive quad ==")
    for (e, s, p) in [(0.0, 3.0, 0.0128), (2.0, 3.0, 0.0128),
                      (6.0, 1.0, 0.01), (0.5, 0.5, 0.05),
                      (20.0, 0.3, 0.001), (0.0, 10.0, 0.02)]:
        d = delta2(e, s, p)
        dq, err = delta2_quad(e, s, p)
        print(f"  e={e:5.1f} s={s:4.1f} p={p:7.4f}: GL={d:.15e} "
              f"quad={dq:.15e} rel={(d-dq)/max(abs(dq),1e-300):+.1e}")

    print("== partials vs central FD ==")
    e, s, p = 2.0, 3.0, 0.0128
    d, de_, dp_, ds_ = _delta2_all(e, s, p)
    h = 1e-6
    fde = (delta2(e + h, s, p) - delta2(e - h, s, p)) / (2 * h)
    fdp = (delta2(e, s, p + h * p) - delta2(e, s, p - h * p)) / (2 * h * p)
    fds = (delta2(e, s + h, p) - delta2(e, s - h, p)) / (2 * h)
    print(f"  de {de_:+.10e} fd {fde:+.10e}")
    print(f"  dp {dp_:+.10e} fd {fdp:+.10e}")
    print(f"  ds {ds_:+.10e} fd {fds:+.10e}")

    print("== composition monotonicity: delta_2 >= delta_1 ==")
    for (e, s, p) in [(2.0, 3.0, 0.0128), (6.0, 1.0, 0.01)]:
        d1 = float(delta1_vec(np.array([e]), s, p)[0])
        print(f"  e={e} s={s} p={p}: d1={d1:.3e} d2={delta2(e, s, p):.3e} "
              f"ok={delta2(e, s, p) >= d1}")

    print("== mechanism map + FOC chain vs FD ==")
    for (e1, e2) in [(3.0, 3.0), (2.0, 6.0), (8.0, 1.5)]:
        r = solve(e1, e2)
        if r is None:
            print(f"  ({e1},{e2}): infeasible")
            continue
        sg, p1, p2 = r
        b1, b2 = eps_budget(sg, p1), eps_budget(sg, p2)
        hh = e2 * 1e-6
        up, um = u2(e1, e2 + hh), u2(e1, e2 - hh)
        fd = (up - um) / (2 * hh)
        an = du2_deps2(e1, e2)
        print(f"  ({e1:4.1f},{e2:4.1f}) sigma={sg:.6f} p2={p2:.6f} | "
              f"eps chk {abs(b1-e1):.1e}/{abs(b2-e2):.1e} | "
              f"dU an {an:+.8e} FD {fd:+.8e}")

    print("== Adv_2(p) curvature at fixed sigma (linear at T=1) ==")
    s = 3.0
    ps = np.linspace(0.002, 0.024, 12)
    av = np.array([advantage(s, float(pp)) for pp in ps])
    d2 = np.diff(av, 2) / (ps[1] - ps[0]) ** 2
    print(f"  second differences: min={d2.min():+.3e} max={d2.max():+.3e}")
    print(f"  Adv_2/p ratio drift: {av[0]/ps[0]:.6f} -> {av[-1]/ps[-1]:.6f}")
