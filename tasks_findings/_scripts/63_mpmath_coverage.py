"""Register P4: randomized mpmath adjudication of the T=2 quadrature.

40 random (sigma, p) draws; each tested at the advantage scale (e = 0)
and at its own budget (delta = 1e-12 root) — the two regimes the game
actually queries. Reports the max relative error per regime, replacing
the ~10 hand-picked points of the original certification.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mpmath as mp

import game_experiments.two_step_game as g2

mp.mp.dps = 40


def delta1_mp(e, sigma, p):
    e, sigma, p = mp.mpf(e), mp.mpf(sigma), mp.mpf(p)
    t = mp.expm1(e) + p
    if t <= 0:
        return -mp.expm1(e)
    x = mp.mpf("0.5") + sigma**2 * (mp.log(t) - mp.log(p))
    return ((1 - p) * mp.ncdf(-x / sigma) + p * mp.ncdf((1 - x) / sigma)
            - mp.e**e * mp.ncdf(-x / sigma))


def delta2_mp(e, sigma, p):
    e, sigma, p = mp.mpf(e), mp.mpf(sigma), mp.mpf(p)

    def h(x):
        a = (2 * x - 1) / (2 * sigma**2)
        g = mp.log((1 - p) + p * mp.e**a)
        f = (1 - p) * mp.npdf(x, 0, sigma) + p * mp.npdf(x, 1, sigma)
        return f * delta1_mp(e - g, sigma, p)

    pts = [-14 * sigma, 0, 1, 1 + 14 * sigma]
    xk = g2._kink_x(float(e), float(sigma), float(p))
    if xk is not None and pts[0] < xk < pts[-1]:
        pts = sorted(pts + [mp.mpf(xk)])
    return mp.quad(h, pts)


rng = np.random.default_rng(63)
worst = {"adv": (0.0, None), "budget": (0.0, None)}
for k in range(40):
    s = float(np.exp(rng.uniform(np.log(0.2), np.log(10.0))))
    p = float(np.exp(rng.uniform(np.log(1e-4), np.log(0.05))))
    for regime in ("adv", "budget"):
        e = 0.0 if regime == "adv" else g2.eps_budget(s, p)
        ref = delta2_mp(e, s, p)
        if abs(ref) < mp.mpf("1e-200"):
            continue
        rel = abs((mp.mpf(g2.delta2(e, s, p)) - ref) / ref)
        if rel > worst[regime][0]:
            worst[regime] = (float(rel), (round(s, 3), round(p, 5), round(e, 4)))
    if (k + 1) % 10 == 0:
        print(f"{k+1}/40: worst adv {worst['adv'][0]:.2e} at {worst['adv'][1]}, "
              f"worst budget {worst['budget'][0]:.2e} at {worst['budget'][1]}",
              flush=True)
print("FINAL:", worst)
