import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

Phi = norm.cdf
phi = norm.pdf
PSUM, DELTA, T, lam = 0.0256, 1e-12, 390.0, 1.0
a = PSUM * PSUM


def delta_gdp(eps, mu):
    la = norm.logcdf(-eps / mu - mu / 2.0)
    tail = math.exp(eps + la) if eps + la > -745.0 else 0.0
    return Phi(-eps / mu + mu / 2.0) - tail


mu_of_eps = lambda e: brentq(lambda m: delta_gdp(e, m) - DELTA, 1e-6, 50.0, xtol=1e-15)


def F(s):
    l = math.log1p(s * s / (a * T))
    return l / (1.0 + l)


def Fp(s):
    l = math.log1p(s * s / (a * T))
    return (2.0 * s / (a * T + s * s)) / (1.0 + l) ** 2


g = lambda m: 2.0 * Phi(m / 2.0) - 1.0
u = lambda m, nu: F(m + nu) - lam * g(m)
mu_lo, mu_hi = mu_of_eps(0.5), mu_of_eps(64.0)
mstar = brentq(lambda m: Fp(2 * m) - lam * phi(m / 2.0), 0.1, 1.0, xtol=1e-15)


def best_response(nu):
    f = lambda m: -u(m, nu)
    grid = np.linspace(mu_lo, mu_hi, 8001)
    i = int(np.argmin([f(m) for m in grid]))
    r = minimize_scalar(f, bounds=(grid[max(i-1,0)], grid[min(i+1,-1)]),
                        method="bounded", options={"xatol": 1e-14})
    cand = [(r.x, -r.fun), (mu_lo, u(mu_lo, nu)), (mu_hi, u(mu_hi, nu))]
    return max(cand, key=lambda t: t[1])[0]


# theoretical local slope
def Fpp(s):
    l = math.log1p(s * s / (a * T))
    lp = 2.0 * s / (a * T + s * s)
    lpp = 2.0 * (a * T - s * s) / (a * T + s * s) ** 2
    return (lpp * (1.0 + l) - 2.0 * lp * lp) / (1.0 + l) ** 3


gpp = lambda m: -(m / 4.0) * phi(m / 2.0)
slope = -Fpp(2 * mstar) / (Fpp(2 * mstar) - lam * gpp(mstar))
print(f"m* = {mstar:.8f}, theoretical BR slope = {slope:.6f}, slope^2 = {slope*slope:.6f}")

# numerical slope check
h = 1e-5
print(f"numerical slope = {(best_response(mstar+h)-best_response(mstar-h))/(2*h):.6f}")

for pert in [1e-4, 1e-2]:
    y = mstar + pert
    x = mstar
    print(f"\nalternating BR, y perturbed by {pert:.0e}:")
    dev0 = pert
    for k in range(400):
        x = best_response(y)
        y = best_response(x)
        d = abs(x - y)
        if k % 40 == 0 or d > 0.2:
            print(f"  round {k+1}: ({x:.7f}, {y:.7f})  anti-diag dev={x-y:+.3e}")
        if d > 0.2:
            print(f"  escaped symmetric point at round {k+1}")
            break
    print(f"  final: ({x:.6f}, {y:.6f})")
