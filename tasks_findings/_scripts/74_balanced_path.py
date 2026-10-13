"""Balanced-path analysis of the full-utility potential game
(nummeric_eval.md Sec. 26 addendum).

The potential Phi = F_T(mu1+mu2) - lam[g(mu1)+g(mu2)] already contains
the model value (F_T = V(sigma(s))). New here: scale lam WITH T so the
symmetric stationary point stays at a fixed location eps* — the exact
"equivalent weight" path lam*(T) = F_T'(2 mu*) / phi(mu*/2) — and
study what survives along it.

Key algebra: at a symmetric stationary point, lam = F'/phi, so the
own-coordinate curvature is H11 = F''(2m) + (m/4) F'(2m) — LAMBDA
DROPS OUT. Interior symmetric NE can exist at m iff
psi_T(m) := F''(2m) + (m/4) F'(2m) < 0, a lambda-free window in
(T, m). Its upper edge gives T_max: beyond it, no weight scaling can
produce interior symmetric moderation inside the box. The BR slope is
|BR'| = A/(A - c) with A = -F''(2m), c = (m/4)F'(2m), again
lambda-free on the path.

Outputs: lam*(T), |BR'|(T), psi-window, T_max, and four balanced-path
landscapes.
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


def L(s, T):   return math.log1p(s * s / (A_CONST * T))
def Lp(s, T):  return 2 * s / (A_CONST * T + s * s)
def Lpp(s, T): return 2 * (A_CONST * T - s * s) / (A_CONST * T + s * s) ** 2
def Fp(s, T):  return Lp(s, T) / (1 + L(s, T)) ** 2
def Fpp(s, T):
    l = L(s, T)
    return (Lpp(s, T) * (1 + l) - 2 * Lp(s, T) ** 2) / (1 + l) ** 3


def psi(m, T):
    return Fpp(2 * m, T) + (m / 4) * Fp(2 * m, T)


MU_LO = brentq(lambda m: g.delta_gdp(0.5, m) - g.DELTA, 1e-6, 60)
MU_HI = brentq(lambda m: g.delta_gdp(64.0, m) - g.DELTA, 1e-6, 60)
MU_STAR = brentq(lambda m: g.delta_gdp(2.548, m) - g.DELTA, 1e-6, 60)
print(f"box in mu: [{MU_LO:.5f}, {MU_HI:.5f}]; mu* = mu(2.548) = {MU_STAR:.5f}")

# T_max: largest T with psi(m, T) < 0 for some m in the box
def psi_min(T):
    ms = np.linspace(MU_LO, MU_HI, 400)
    return min(psi(float(m), T) for m in ms)
T_max = brentq(lambda T: psi_min(T), 1e3, 1e7, xtol=10)
print(f"T_max (no interior symmetric NE beyond, at ANY lambda): {T_max:.4g}")

# balanced path holding the stationary point at mu*
print(f"\n{'T':>8} {'lam*(T)':>10} {'psi(mu*)':>11} {'|BR_prime|':>10}  NE exists at mu*?")
rows = []
for T in (390, 1500, 6000, 25000, 100000, 240000, 400000):
    lam = Fp(2 * MU_STAR, T) / norm.pdf(MU_STAR / 2)
    p = psi(MU_STAR, T)
    Aa = -Fpp(2 * MU_STAR, T)
    c = (MU_STAR / 4) * Fp(2 * MU_STAR, T)
    br = Aa / (Aa - c) if Aa > c else float("nan")
    rows.append((T, lam, p, abs(br) if np.isfinite(br) else np.nan))
    print(f"{T:>8} {lam:10.4f} {p:11.5f} {abs(br) if np.isfinite(br) else float('nan'):10.4f}  {p < 0}")

# empirical-equivalence comparison note
print(f"\nlam*(T) ratios: lam*(25000)/lam*(390) = "
      f"{rows[3][1]/rows[0][1]:.4f} (empirical regime-mix equivalence gave ~0.35)")

# ---- figure: balanced-path landscapes + the two curves -------------------
ES = np.geomspace(0.5, 64.0, 140)
MU = np.array([brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA,
                      1e-6, 60.0) for e in ES])
GC = 2 * norm.cdf(MU / 2) - 1

fig = plt.figure(figsize=(18.5, 8.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0], hspace=0.32)
TS = [390, 25000, 240000, 400000]
for k, T in enumerate(TS):
    lam = Fp(2 * MU_STAR, T) / norm.pdf(MU_STAR / 2)
    ax = fig.add_subplot(gs[0, k])
    S = MU[:, None] + MU[None, :]
    Lv = np.log1p(S * S / (A_CONST * T))
    PHI = Lv / (1 + Lv) - lam * (GC[:, None] + GC[None, :])
    im = ax.pcolormesh(ES, ES, PHI.T, cmap="viridis", shading="auto")
    ax.contour(ES, ES, PHI.T, levels=12, colors="white", linewidths=0.4,
               alpha=0.6)
    i, j = np.unravel_index(np.argmax(PHI), PHI.shape)
    ax.plot(ES[i], ES[j], "*", color="red", ms=14, mec="white", mew=0.8)
    if i != j:
        ax.plot(ES[j], ES[i], "*", color="red", ms=14, mec="white", mew=0.8)
    ax.plot(2.548, 2.548, "o", color="orange", ms=9, mec="white", mew=0.8)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_title(f"$T={T:,}$, $\\lambda^*={lam:.3f}$"
                 + ("  (past $T_{max}$)" if T > T_max else ""), fontsize=10)
    ax.set_xlabel("$\\varepsilon_1$")
    if k == 0:
        ax.set_ylabel("$\\varepsilon_2$")

ax1 = fig.add_subplot(gs[1, :2])
Ts = np.geomspace(390, 6e5, 60)
lams = [Fp(2 * MU_STAR, T) / norm.pdf(MU_STAR / 2) for T in Ts]
ax1.loglog(Ts, lams, lw=2, color="#4f81bd")
ax1.loglog(Ts, [lams[0] * (Ts[0] / T) for T in Ts], "--", color="0.6",
           label="$\\propto 1/T$ (large-T law)")
ax1.axvline(T_max, color="crimson", lw=1.2, ls=":")
ax1.set_xlabel("T"), ax1.set_ylabel("balanced weight $\\lambda^*(T)$")
ax1.set_title("the exact equivalent-weight path (stationary point pinned at $\\varepsilon^*=2.548$)",
              fontsize=10)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[1, 2:])
brs = []
for T in Ts:
    Aa = -Fpp(2 * MU_STAR, T)
    c = (MU_STAR / 4) * Fp(2 * MU_STAR, T)
    brs.append(Aa / (Aa - c) if Aa > c else np.nan)
ax2.semilogx(Ts, brs, lw=2, color="#c0504d")
ax2.axhline(1, color="0.6", lw=0.8)
ax2.axvline(T_max, color="crimson", lw=1.2, ls=":",
            label=f"$T_{{max}}\\approx{T_max:,.0f}$: interior moderation\nimpossible beyond (any $\\lambda$)")
ax2.set_xlabel("T"), ax2.set_ylabel("$|BR'|$ at the symmetric NE")
ax2.set_title("knife-edge sharpness along the balanced path (>1 always: the no-go theorem)",
              fontsize=10)
ax2.legend(fontsize=9)
fig.suptitle("Full-utility potential along the balanced path: $\\lambda$ rescaled with T so the moderation point stays put",
             fontsize=12.5)
fig.savefig(ROOT / "paper/figures/data/foc_landscape/balanced_path.png",
            dpi=150, bbox_inches="tight")
print("wrote balanced_path.png")
