import math
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

Phi = norm.cdf
phi = norm.pdf

PSUM = 0.0256
DELTA = 1e-12
T = 390.0
lam = 1.0
a = PSUM * PSUM  # PSUM^2


def delta_gdp(eps, mu):
    la = norm.logcdf(-eps / mu - mu / 2.0)
    tail = math.exp(eps + la) if eps + la > -745.0 else 0.0
    return Phi(-eps / mu + mu / 2.0) - tail


def mu_of_eps(eps):
    f = lambda m: delta_gdp(eps, m) - DELTA
    return brentq(f, 1e-6, 50.0, xtol=1e-14, rtol=8.9e-16)


def L(s, T=T):
    return math.log1p(s * s / (a * T))


def F(s, T=T):
    l = L(s, T)
    return l / (1.0 + l)


def Fp(s, T=T):
    l = L(s, T)
    lp = 2.0 * s / (a * T + s * s)
    return lp / (1.0 + l) ** 2


def Fpp(s, T=T):
    l = L(s, T)
    lp = 2.0 * s / (a * T + s * s)
    lpp = 2.0 * (a * T - s * s) / (a * T + s * s) ** 2
    return (lpp * (1.0 + l) - 2.0 * lp * lp) / (1.0 + l) ** 3


def g(m):
    return 2.0 * Phi(m / 2.0) - 1.0


def gp(m):
    return phi(m / 2.0)


def gpp(m):
    return -(m / 4.0) * phi(m / 2.0)


mu_lo = mu_of_eps(0.5)
mu_hi = mu_of_eps(64.0)
mu_243 = mu_of_eps(2.43)
print(f"mu(0.5)  = {mu_lo:.8f}")
print(f"mu(2.43) = {mu_243:.8f}")
print(f"mu(64)   = {mu_hi:.8f}")
print(f"aT = {a*T:.6f}, sqrt(aT) = {math.sqrt(a*T):.6f}")

# --- symmetric stationarity: h(m) = F'(2m) - lam*phi(m/2) = 0 ---
h = lambda m: Fp(2 * m) - lam * gp(m)
ms = np.linspace(mu_lo, mu_hi, 20001)
hv = np.array([h(m) for m in ms])
roots = []
for i in range(len(ms) - 1):
    if hv[i] == 0 or hv[i] * hv[i + 1] < 0:
        roots.append(brentq(h, ms[i], ms[i + 1], xtol=1e-14))
print("\nsymmetric stationary roots (mu, and corresponding eps):")


def eps_of_mu(mu):
    f = lambda e: delta_gdp(e, mu) - DELTA
    return brentq(f, 1e-9, 1e6, xtol=1e-13)


for m in roots:
    s = 2 * m
    lam_anti = -lam * gpp(m)                 # eigenvalue along (1,-1)
    lam_diag = 2 * Fpp(s) - lam * gpp(m)     # eigenvalue along (1,1)
    H11 = Fpp(s) - lam * gpp(m)
    slope = -Fpp(s) / (Fpp(s) - lam * gpp(m))
    print(f"  m = {m:.6f}  eps = {eps_of_mu(m):.5f}")
    print(f"    F'(2m)={Fp(s):.6e}  F''(2m)={Fpp(s):.6e}  g''(m)={gpp(m):.6e}")
    print(f"    eig(1,1)  = 2F''-lam g'' = {lam_diag:+.6e}")
    print(f"    eig(1,-1) = -lam g''     = {lam_anti:+.6e}")
    print(f"    H11 (own SOC, need <0)   = {H11:+.6e}")
    cls = "LOCAL MIN" if lam_diag > 0 and lam_anti > 0 else (
        "LOCAL MAX" if lam_diag < 0 and lam_anti < 0 else "SADDLE")
    print(f"    classification of Phi: {cls}")
    print(f"    BR slope dBR/dnu = {slope:+.6f}  (|slope|>1 -> BR-unstable)")

# --- utility landscape / BR map on the box (continuous, fine polish) ---
def best_response(nu):
    # maximize F(mu+nu) - lam*g(mu) over mu in [mu_lo, mu_hi]
    f = lambda m: -(F(m + nu) - lam * g(m))
    grid = np.linspace(mu_lo, mu_hi, 4001)
    vals = np.array([f(m) for m in grid])
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-12})
    # compare against box corners
    cand = [(r.x, -r.fun), (mu_lo, -f(mu_lo)), (mu_hi, -f(mu_hi))]
    return max(cand, key=lambda t: t[1])[0]


print("\nBR map samples (mu_j -> BR_i):")
for nu in [mu_lo, 0.2, 0.3, roots[0] if roots else 0.3, 0.5, 1.0, 2.0, mu_hi]:
    print(f"  BR({nu:.6f}) = {best_response(nu):.6f}")

# BR dynamics from near-symmetric start
print("\ncontinuous BR dynamics (alternating), start = symmetric root + 1e-3 perturbation:")
if roots:
    m0 = roots[-1]
    x, y = m0 + 1e-3, m0
    for k in range(25):
        x = best_response(y)
        y = best_response(x)
        if k < 8 or k % 5 == 0:
            print(f"  round {k+1}: mu = ({x:.6f}, {y:.6f})  eps=({eps_of_mu(x):.4f},{eps_of_mu(y):.4f})")

# simultaneous BR dynamics too
print("\ncontinuous BR dynamics (simultaneous), same start:")
if roots:
    x, y = m0 + 1e-3, m0
    for k in range(25):
        x, y = best_response(y), best_response(x)
        if k < 8 or k % 5 == 0:
            print(f"  round {k+1}: mu = ({x:.6f}, {y:.6f})")

# --- global max of Phi on box ---
def Phi_pot(m1, m2):
    return F(m1 + m2) - lam * g(m1) - lam * g(m2)


gg = np.linspace(mu_lo, mu_hi, 801)
best = None
for m1 in gg:
    for m2 in gg:
        v = Phi_pot(m1, m2)
        if best is None or v > best[0]:
            best = (v, m1, m2)
print(f"\ngrid global max of Phi: Phi={best[0]:.6f} at mu=({best[1]:.4f},{best[2]:.4f})"
      f" eps=({eps_of_mu(best[1]):.4f},{eps_of_mu(best[2]):.4f})")
if roots:
    for m in roots:
        print(f"Phi at symmetric root m={m:.4f}: {Phi_pot(m, m):.6f}")
print(f"Phi at (mu_lo,mu_lo) = {Phi_pot(mu_lo, mu_lo):.6f}")
print(f"Phi at (mu_hi,mu_hi) = {Phi_pot(mu_hi, mu_hi):.6f}")
print(f"Phi at (mu_lo,mu_hi) = {Phi_pot(mu_lo, mu_hi):.6f}")

# --- T0 thresholds ---
mu_max = mu_hi
phi_max = phi(mu_max / 2.0)
T0_old = 4.0 * mu_max / (lam * a * phi_max)
# tight: solve F_T'(2 mu_max) = lam phi(mu_max/2) for T (valid once F' increasing on box)
fT = lambda TT: Fp(2 * mu_max, TT) - lam * phi_max
T0_tight = brentq(fT, 1e3, 1e12, rtol=1e-12)
print(f"\nmu_max = {mu_max:.6f}, phi(mu_max/2) = {phi_max:.6e}")
print(f"T0_old  (4 mu_max/(lam a phi)) = {T0_old:.6e}")
print(f"T0_tight (root of F_T'(2mu_max)=lam phi) = {T0_tight:.6e}")
print(f"ratio tight/old = {T0_tight/T0_old:.4f}")
# check F' increasing on box at T0_tight: need sqrt(aT) >= 2 mu_max
print(f"sqrt(a*T0_tight) = {math.sqrt(a*T0_tight):.4f} vs 2 mu_max = {2*mu_max:.4f}")
# verify dominance at T0_tight*1.001 over whole box
TT = T0_tight * 1.001
worst = max(Fp(m1 + m2, TT) - lam * gp(m1)
            for m1 in np.linspace(mu_lo, mu_hi, 200)
            for m2 in np.linspace(mu_lo, mu_hi, 200))
print(f"worst du/dmu on box at 1.001*T0_tight: {worst:+.3e} (should be <0)")
