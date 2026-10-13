"""Ridge engineering (nummeric_eval.md Sec. 26 addendum 2): can the
moderation pass be shortened or flipped into a peak?

The ridge through the symmetric saddle is the anti-diagonal fiber
{(m+t, m-t)}: the shared value F is constant on it, so the fiber
profile is Phi(t) - Phi(0) = -lam [C(g(m+t)) + C(g(m-t)) - 2C(g(m))],
with C the exposure-price shape. Closed-form consequences:

(1) FLOOR: the fiber is truncated at t_max = m* - mu_lo; raising the
    action floor shortens the ridge and shrinks the payoff advantage
    of sliding off, DeltaPhi = lam[2C(g(m*)) - C(g(mu_lo)) -
    C(g(2m* - mu_lo))]; at eps_min -> eps* the ridge length -> 0.
    (The no-go theorem still forbids interior BR-stability at any
    floor — truncation bounds the damage, it does not flip the pass.)

(2) PROGRESSIVITY: with C(A) = A^gamma, the anti-diagonal curvature is
    e_minus = -lam[C'' g'^2 + C' g''] and flips sign (pass -> peak)
    when C''/C' > -g''/g'^2 = m/(4 phi(m/2)), i.e.

        gamma*(m) = 1 + g(m) * m / (4 phi(m/2)).

    The same condition is |BR'| < 1: sufficient progressivity makes
    the symmetric point a true potential peak AND best-response
    stable. Computed along the existence window.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import game_experiments.gdp_limit_game as g

A_CONST = g.PSUM ** 2
phi = norm.pdf


def mu_of(e):
    return brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA, 1e-6, 60.0)


def Fp(s, T):
    L = math.log1p(s * s / (A_CONST * T))
    return (2 * s / (A_CONST * T + s * s)) / (1 + L) ** 2


def gval(m):  return 2 * norm.cdf(m / 2) - 1


MU_LO0 = mu_of(0.5)
M_STAR = mu_of(2.548)
T0 = 390.0

# ---- (2) the progressivity threshold ------------------------------------
def gamma_star(m):
    return 1 + gval(m) * m / (4 * phi(m / 2))

print(f"gamma* at the T=390 saddle (m*={M_STAR:.4f}): {gamma_star(M_STAR):.4f}")
# gamma* along the balanced path / existence window: m*(T) from
# stationarity with lam = 1 is not needed — gamma* depends on m only;
# report over the m-range the window uses
for e in (1.0, 2.548, 5.0, 10.0):
    m = mu_of(e)
    print(f"  at eps={e}: m={m:.4f} -> gamma* = {gamma_star(m):.4f}")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (a) ridge profiles for different action floors (gamma = 1)
ax = axes[0]
lam = 1.0
ts = np.linspace(0, M_STAR - MU_LO0, 300)
for eps_min, col in ((0.5, "#c0504d"), (1.0, "#e2a76f"), (1.8, "#9bbb59"),
                     (2.4, "#4f81bd")):
    mu_lo = mu_of(eps_min)
    tmax = M_STAR - mu_lo
    tt = ts[ts <= tmax]
    prof = -lam * (gval(M_STAR + tt) + gval(M_STAR - tt) - 2 * gval(M_STAR))
    ax.plot(tt, prof * 1e3, color=col, lw=2,
            label=f"action floor $\\varepsilon_{{min}}={eps_min}$")
    ax.plot(tt[-1], prof[-1] * 1e3, "o", color=col, ms=7, mec="0.2")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("asymmetry $t$ along the ridge (in $\\mu$)")
ax.set_ylabel("$\\Phi$ gain over the saddle  ($\\times 10^{-3}$)")
ax.set_title("(a) raising the floor truncates the ridge:\nshorter slide, smaller prize for sliding",
             fontsize=10)
ax.legend(fontsize=8)

# (b) ridge profiles under progressive pricing (floor 0.5)
ax = axes[1]
tmax = M_STAR - MU_LO0
tt = np.linspace(0, tmax, 300)
gm = gval(M_STAR)
for gam, col in ((1.0, "#c0504d"), (1.02, "#e2a76f"), (1.036, "0.4"),
                 (1.08, "#4f81bd")):
    # rebalance lam so the stationary point stays at m* (lam cancels in shape)
    prof = -(np.power(gval(M_STAR + tt), gam) + np.power(gval(M_STAR - tt), gam)
             - 2 * gm ** gam) / (gam * gm ** (gam - 1))
    ax.plot(tt, prof * 1e3, color=col, lw=2,
            label=f"$\\gamma={gam}$" + ("  ($\\gamma^*$)" if gam == 1.036 else ""))
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("asymmetry $t$ along the ridge (in $\\mu$)")
ax.set_title("(b) progressive pricing bends the ridge down:\nabove $\\gamma^*$ the pass becomes a peak",
             fontsize=10)
ax.legend(fontsize=8)

# (c) gamma* over budget scale, and where the window lives
ax = axes[2]
es = np.geomspace(0.6, 20, 120)
gs = [gamma_star(mu_of(float(e))) for e in es]
ax.semilogx(es, gs, lw=2, color="#4f81bd")
ax.axvline(2.548, color="crimson", ls=":", lw=1.2,
           label="the T=390 moderation point")
ax.plot(2.548, gamma_star(M_STAR), "o", color="crimson", ms=8)
ax.annotate(f"$\\gamma^*={gamma_star(M_STAR):.3f}$:\nonly {100*(gamma_star(M_STAR)-1):.1f}% progressivity\nstabilizes moderation",
            (2.548, gamma_star(M_STAR)), textcoords="offset points",
            xytext=(14, -34), fontsize=9)
ax.set_xlabel("moderation point location $\\varepsilon^*$")
ax.set_ylabel("required progressivity $\\gamma^*$")
ax.set_title("(c) the stabilization price: how progressive must\nthe exposure cost be, by moderation level",
             fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("Ridge engineering: two interventions on the moderation saddle (limit game, $T=390$ geometry)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/ridge_engineering.png",
            dpi=150, bbox_inches="tight")
print("wrote ridge_engineering.png")

# BR-stability check under gamma just above gamma*
gam = 1.05
# e_minus sign: C''g'^2 + C'g'' with C = A^gam at A = g(m*)
A = gval(M_STAR); gp = phi(M_STAR / 2); gpp = -(M_STAR / 4) * phi(M_STAR / 2)
em = gam * (gam - 1) * A ** (gam - 2) * gp ** 2 + gam * A ** (gam - 1) * gpp
print(f"anti-diagonal curvature term at gamma=1.05: {em:+.5f} (positive = peak)")
EOF_UNUSED = None
