"""Evaluate the responsiveness condition of prop:no-symmetric-ne,

    T(p, sigma) = A_p + sigma * A_{p sigma} - p * A_{pp},

by central finite differences of the PLD advantage A(p, sigma) at the
diagonal operating points of eps in {2.9646, 10.28} (K=200 canonical
game). The proposition rules out a diagonal NE only where T > 0; the
surviving symmetric equilibrium at eps = 2.9646 should coincide with
T <= 0 there, and the confirmed trough at eps = 10.28 with T > 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.mechanism import solve_mechanism
from game.advantage import mia_advantage_subsampled_gaussian as adv

STEPS, DELTA = 390, 1e-12


def bracket(p: float, s: float, hp_rel: float = 5e-3, hs_rel: float = 5e-3):
    hp, hs = p * hp_rel, s * hs_rel
    A_p = (adv(s, p + hp, STEPS) - adv(s, p - hp, STEPS)) / (2 * hp)
    A_pp = (adv(s, p + hp, STEPS) - 2 * adv(s, p, STEPS)
            + adv(s, p - hp, STEPS)) / hp**2
    A_ps = (adv(s + hs, p + hp, STEPS) - adv(s + hs, p - hp, STEPS)
            - adv(s - hs, p + hp, STEPS) + adv(s - hs, p - hp, STEPS)) \
        / (4 * hp * hs)
    return A_p + s * A_ps - p * A_pp, A_p, A_ps, A_pp


def main() -> None:
    for eps in [2.9646, 10.28]:
        r = solve_mechanism([eps, eps], [5000, 5000], 128, STEPS, DELTA)
        p, s = float(r.sample_rates[0]), float(r.sigma)
        for h in [1e-2, 5e-3, 2e-3]:
            T, A_p, A_ps, A_pp = bracket(p, s, h, h)
            print(f"eps={eps}: p={p:.5f} sigma={s:.4f} h_rel={h:g}  "
                  f"T={T:+.4f}  (A_p={A_p:+.3f}, sig*A_ps={s*A_ps:+.3f}, "
                  f"p*A_pp={p*A_pp:+.3f})")


if __name__ == "__main__":
    main()
