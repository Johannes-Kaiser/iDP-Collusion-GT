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


def mu_of_eps(eps):
    return brentq(lambda m: delta_gdp(eps, m) - DELTA, 1e-6, 50.0, xtol=1e-14)


def eps_of_mu(mu):
    return brentq(lambda e: delta_gdp(e, mu) - DELTA, 1e-9, 1e6, xtol=1e-13)


def F(s):
    l = math.log1p(s * s / (a * T))
    return l / (1.0 + l)


def Fp(s):
    l = math.log1p(s * s / (a * T))
    return (2.0 * s / (a * T + s * s)) / (1.0 + l) ** 2


def g(m):
    return 2.0 * Phi(m / 2.0) - 1.0


mu_lo, mu_hi = mu_of_eps(0.5), mu_of_eps(64.0)
mstar = brentq(lambda m: Fp(2 * m) - lam * phi(m / 2.0), 0.1, 1.0, xtol=1e-14)
print(f"interior symmetric root m* = {mstar:.8f} (eps = {eps_of_mu(mstar):.5f})")

u = lambda m, nu: F(m + nu) - lam * g(m)


def best_response(nu, full=False):
    f = lambda m: -u(m, nu)
    grid = np.linspace(mu_lo, mu_hi, 8001)
    vals = np.array([f(m) for m in grid])
    i = int(np.argmin(vals))
    r = minimize_scalar(f, bounds=(grid[max(i-1,0)], grid[min(i+1,-1)]),
                        method="bounded", options={"xatol": 1e-13})
    cand = [(r.x, -r.fun), (mu_lo, u(mu_lo, nu)), (mu_hi, u(mu_hi, nu))]
    # also search other local basins: find all sign changes of derivative numerically
    b = max(cand, key=lambda t: t[1])
    return b if full else b[0]


# NE check at symmetric root: own-coordinate global max?
bm, bu = best_response(mstar, full=True)
print(f"BR(m*) = {bm:.8f}, u = {bu:.8f}; u(m*, m*) = {u(mstar, mstar):.8f}")
print(f"  u at corners given nu=m*: u(mu_lo)={u(mu_lo, mstar):.8f}, u(mu_hi)={u(mu_hi, mstar):.8f}")

# asymmetric candidate equilibrium
x1 = mu_lo
x2 = best_response(x1)
x1b = best_response(x2)
print(f"\nasymmetric pair: BR(mu_lo)={x2:.6f} (eps={eps_of_mu(x2):.4f}), BR back = {x1b:.6f} (mu_lo={mu_lo:.6f})")
print(f"  -> asymmetric NE at mu=({x1b:.6f},{x2:.6f}) = eps=({eps_of_mu(x1b):.4f},{eps_of_mu(x2):.4f})")

# alternating continuous BR dynamics from perturbed interior symmetric root
for pert in [1e-6, 1e-4, 1e-2]:
    x, y = mstar + pert, mstar
    print(f"\nalternating BR from (m*+{pert:.0e}, m*):")
    for k in range(30):
        x = best_response(y)
        y = best_response(x)
        d = abs(x - y)
        if k < 6 or (k + 1) % 5 == 0 or d > 0.3:
            print(f"  round {k+1}: ({x:.7f}, {y:.7f})  |x-mstar|={abs(x-mstar):.2e}")
        if abs(x - x1b) < 1e-6 and abs(y - x2) < 1e-6 or abs(y - x1b) < 1e-6 and abs(x - x2) < 1e-6:
            print(f"  -> reached asymmetric NE at round {k+1}")
            break

# grid-BR dynamics in eps coordinates (like the coded study)
for npts in [12, 31, 101]:
    eg = np.geomspace(0.5, 64.0, npts)
    mg = np.array([mu_of_eps(e) for e in eg])
    def br_grid(nu):
        vals = [u(m, nu) for m in mg]
        return int(np.argmax(vals))
    i, j = br_grid(mstar), br_grid(mstar)
    # start from several starts
    print(f"\ngrid BR (geomspace eps, {npts} pts), alternating, several starts:")
    for start in [0, npts // 2, npts - 1]:
        i, j = start, start
        hist = []
        for k in range(20):
            i = br_grid(mg[j])
            j = br_grid(mg[i])
            hist.append((i, j))
            if k > 1 and hist[-1] == hist[-2]:
                break
        print(f"  start eps={eg[start]:.3f}: end eps=({eg[i]:.4f}, {eg[j]:.4f}) after {k+1} rounds")
