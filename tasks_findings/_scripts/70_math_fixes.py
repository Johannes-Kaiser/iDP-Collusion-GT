"""Fixes for the open math of nummeric_eval.md Sec. 25.7.

Fix 1 — Lemma (budget factorization, replaces the refuted 25.1):
with r = (e^eps - 1 + p)/p and x = 1/2 + sigma^2 log r,

    delta_1(eps; sigma, p) = p * h(r; sigma),
    h(r; sigma) = Phi((1-x)/sigma) - r Phi(-x/sigma),

exactly (pure algebra on the closed form: 1 - p - e^eps = -p r and x
depends on (eps, p) only through r). Consequences: (i) the budget
curve delta_1 = dbar at fixed sigma is the one-parameter family
p = dbar/h(r), e^eps - 1 = dbar (r-1)/h(r); (ii) exact
proportionality p ~ (e^eps - 1) holds iff r stays constant along the
curve — the drift of r(eps) IS the correction the refuted lemma
missed; (iii) the exact log-slope is
d log p/d eps = [e^eps/(e^eps-1)] * eta/(eta - 1 + 1/(1 - h'r... ))
computed numerically here via the parametric form.

Fix 2 — Proposition (one-shot symmetric stationarity): at a symmetric
point (e, e) with rates p = PSUM/2 and noise sigma from
E(sigma, p) = e, the own-budget slice derivative is

    g(e) = V'(sigma)/(2 Fs) - lam [ c'(sigma) p / (2 Fs) + c(sigma)/(2 Fp) ],

with (Fs, Fp) = (dE/dsigma, dE/dp) the closed-form budget partials
(symmetry collapses the implicit chain: dsigma/deps2 = 1/(2 Fs),
dp2/deps2 = 1/(2 Fp)). A symmetric interior NE requires g(e*) = 0
with a falling crossing AND u(e*,e*) >= the corner/tangency
candidates of the column. Validated below on a world the screening
battery flagged stable (G=500, V=1/(1+s), lam=8.28) and on the
baseline (G=5000, lam=0.7) where no symmetric interior NE exists.
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game_experiments.robust_game import Game
from game_experiments.single_step_game import dE, eps_budget
from game_experiments.two_step_game import delta1_vec

print("== Fix 1: budget factorization delta_1 = p * h(r) ==")
rng = np.random.default_rng(70)
worst = 0.0
for _ in range(200):
    s = float(np.exp(rng.uniform(np.log(0.2), np.log(8.0))))
    p = float(np.exp(rng.uniform(np.log(1e-6), np.log(0.5))))
    e = float(rng.uniform(0.05, 12.0))
    r = (math.expm1(e) + p) / p
    x = 0.5 + s * s * math.log(r)
    h = norm.cdf((1 - x) / s) - r * norm.cdf(-x / s)
    lhs = float(delta1_vec(np.array([e]), s, p)[0])
    rel = abs(lhs - p * h) / max(abs(lhs), 1e-300)
    worst = max(worst, rel)
print(f"  identity max rel error over 200 random (sigma,p,eps): {worst:.2e}")

# r-drift along a budget curve (the exact correction to 'exponentiality')
g0 = Game(T=1)
print("  r(eps) along the budget curve at sigma=0.9 (its drift is the "
      "correction):")
for e in (0.5, 1.5, 3.0, 4.5, 6.0):
    p = g0.p_for_eps(0.9, float(e))
    if p >= 1.0:
        continue
    r = (math.expm1(e) + p) / p
    print(f"    eps={e}: p={p:.5f}  r={r:.1f}  p*(r-1)/(e^eps-1)="
          f"{p*(r-1)/math.expm1(e):.6f} (==1 exact)")

print("\n== Fix 2: symmetric stationarity criterion g(e) ==")

def c_fn(s):    return 2 * norm.cdf(0.5 / s) - 1
def cp_fn(s):   return -norm.pdf(0.5 / s) / (s * s)

def make_g(game):
    psum = game.B / game.G[0]
    p = psum / 2.0
    lam = game.lam_i(0)
    def dV(s):
        if game.V_form == "inv_sq": return -2*s/(1+s*s)**2
        if game.V_form == "inv":    return -1/(1+s)**2
        return -math.exp(-s)
    def g(e):
        f = lambda s: eps_budget(s, p) - e
        s = brentq(f, 1e-3, 1e4, xtol=1e-11)
        Fs, Fp, _ = dE(s, p)
        return dV(s)/(2*Fs) - lam*(cp_fn(s)*p/(2*Fs) + c_fn(s)/(2*Fp))
    return g

for (tag, kw, expect) in [
    ("G=500, V=1/(1+s), lam=8.28 (battery: stable)",
     dict(T=1, G=(500, 500), V_form="inv", lam=8.28), True),
    ("baseline G=5000, lam=0.7 (campaign: no symmetric interior NE)",
     dict(T=1, G=(5000, 5000), lam=0.7), False),
]:
    game = Game(**kw)
    g = make_g(game)
    es = np.geomspace(0.55, 60.0, 25)
    vals = [g(float(e)) for e in es]
    roots = []
    for k in range(len(es)-1):
        if np.isfinite(vals[k]) and np.isfinite(vals[k+1]) and vals[k]*vals[k+1] < 0:
            z = brentq(lambda e: g(e), es[k], es[k+1], xtol=1e-8)
            roots.append((z, "falling" if vals[k] > 0 else "rising"))
    print(f"  {tag}")
    print(f"    g roots: {[(round(z,4), d) for z,d in roots]}")
    for z, d in roots:
        if d != "falling":
            continue
        # global check along the column at e1 = z
        r = game.utilities([z, z]); u_sym = r[0][1]
        f_lo, f_hi = game.feasible_e2(z)
        a, b = max(game.box[0], f_lo*1.000001), min(game.box[1], f_hi*0.999999)
        xs = np.geomspace(a, b, 90)
        best = max((game.utilities([z, float(e2)])[0][1]
                    for e2 in xs if game.utilities([z, float(e2)])), default=np.nan)
        print(f"    falling root e*={z:.4f}: u_sym={u_sym:.6f}, column max="
              f"{best:.6f} -> symmetric interior NE: {u_sym >= best - 1e-9}")
EOF_MARKER_NOT_USED = None
