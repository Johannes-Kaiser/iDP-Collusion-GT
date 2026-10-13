"""Numeric validation of the sharpened math (nummeric_eval.md Sec. 25).

Check 1 — rate-phase exponentiality (Lemma 25.1): at fixed noise and
contract delta, the participation solving the budget satisfies
p(eps) ~ (e^eps - 1) / (e^K - 1) with K = K(sigma, delta) constant.
Verified by computing p(eps) exactly and testing that
p(eps) / (e^eps - 1) is flat over the rate phase.

Check 2 — interior tangency identity (Prop. 25.4, general V): at an
interior best reply in the noise phase, |V'(sigma)| = lam_eff * p2 *
|c'(sigma)| with c(sigma) = 2 Phi(1/(2 sigma)) - 1 and, for quadratic
cost, lam_eff = 2 lam_q A. Tested at the three signal-A worlds (the
linear-cost interior exceptions across different V shapes).

Check 3 — temptation-hump decomposition (Sec. 25.6): gap = DV - DC
(value prize minus exposure bill of the best deviation from the
symmetric candidate); computed for the baseline twin at T = 1 and
T = 2 to identify which term drives the observed rise.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game_experiments.robust_game import Game

print("== Check 1: p(eps) proportional to (e^eps - 1) at fixed sigma ==")
g = Game(T=1, G=(5000, 5000))
for s in (0.9, 3.0):
    es = np.linspace(0.5, 6.0, 6)
    ratio = [g.p_for_eps(s, float(e)) / np.expm1(e) for e in es]
    ratio = np.array(ratio)
    print(f"  sigma={s}: p/(e^eps-1) varies "
          f"{ratio.min():.4e} .. {ratio.max():.4e} "
          f"(rel spread {(ratio.max()-ratio.min())/ratio.mean():.1%})")

print("\n== Check 2: tangency identity at the three interior worlds ==")
WORLDS = [
    dict(T=1, G=(5000, 5000), pbar=1e-5, box=(0.5, 64.0),
         V_form="inv_sq", cost_form="linear", lam=28.0),
    dict(T=1, G=(2000, 8000), pbar=1e-6, box=(0.5, 64.0),
         V_form="exp", cost_form="linear", lam=38.1),
    dict(T=1, G=(500, 500), pbar=1e-6, box=(1.0, 16.0),
         V_form="inv", cost_form="linear", lam=2.54),
]
def dV(g, s):
    if g.V_form == "inv_sq":
        return -2*s/(1+s*s)**2
    if g.V_form == "inv":
        return -1/(1+s)**2
    return -np.exp(-s)
def dc(s):
    return -norm.pdf(0.5/s)/(2*s*s) * 2      # d/ds [2 Phi(1/(2s)) - 1]

for w in WORLDS:
    g = Game(**w)
    lo, hi = g.box
    # find a column with an interior peak, then locate it finely
    found = False
    for e1 in np.geomspace(lo, hi, 9):
        try:
            f_lo, f_hi = g.feasible_e2(float(e1))
        except Exception:
            continue
        a, b = max(lo, f_lo*1.000001), min(hi, f_hi*0.999999)
        if not a < b:
            continue
        xs = np.geomspace(a, b, 80)
        us = []
        for e2 in xs:
            r = g.utilities([float(e1), float(e2)])
            us.append(r[0][1] if r else np.nan)
        us = np.array(us)
        fin = np.isfinite(us)
        if fin.sum() < 10:
            continue
        k = int(np.nanargmax(us))
        if 2 < k < fin.sum() - 3:
            e2s = float(xs[k])
            r = g.utilities([float(e1), e2s])
            _, sig, ps = r
            p2 = ps[1]
            lhs = abs(dV(g, sig))
            rhs = g.lam_i(1) * p2 * abs(dc(sig))
            print(f"  V={w['V_form']:6s} w={w['lam']:5g} G={w['G']}: interior peak "
                  f"e2={e2s:.3f} sigma={sig:.3f} p2={p2:.4f} | "
                  f"|V'|={lhs:.4e} vs lam*p2*|c'|={rhs:.4e} "
                  f"(ratio {lhs/rhs:.3f})")
            found = True
            break
    if not found:
        print(f"  V={w['V_form']} w={w['lam']}: no interior peak found at this sampling")

print("\n== Check 3: hump decomposition, baseline twin T=1 vs T=2 ==")
for T in (1, 2):
    g = Game(T=T, G=(5000, 5000), lam=0.7)
    # symmetric candidate: the least-tempting diagonal point over a scan
    best = None
    for e in np.geomspace(0.5, 64.0, 8):
        r = g.utilities([float(e), float(e)])
        if r is None:
            continue
        u_stay, sig0, ps0 = r[0][1], r[1], r[2]
        # best deviation of player 2 along the column
        f_lo, f_hi = g.feasible_e2(float(e))
        a, b = max(0.5, f_lo*1.000001), min(64.0, f_hi*0.999999)
        xs = np.geomspace(a, b, 24)
        dev = None
        for e2 in xs:
            rr = g.utilities([float(e), float(e2)])
            if rr and (dev is None or rr[0][1] > dev[0]):
                dev = (rr[0][1], rr[1], rr[2], float(e2))
        if dev is None:
            continue
        gap = dev[0] - u_stay
        if best is None or gap < best[0]:
            V0, A0 = g.value(sig0), g.advantage(sig0, ps0[1])
            V1, A1 = g.value(dev[1]), g.advantage(dev[1], dev[2][1])
            best = (gap, float(e), dev[3], V1-V0, g.lam_i(1)*(A1-A0))
    gap, e, e2, DV, DC = best
    print(f"  T={T}: least-tempting diag e={e:.3f}, best deviation e2={e2:.3f}: "
          f"gap={gap:.4f} = value prize {DV:.4f} - exposure bill {DC:.4f}")
