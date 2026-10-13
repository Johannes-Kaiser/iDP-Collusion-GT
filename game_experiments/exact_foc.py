"""Exact slice derivative du2/deps2 of the PEG utility.

Implements Section 5 of tasks_findings/33_u2_analytic_model.md: solve the
mechanism once at (eps1, eps2), then assemble

  du2/deps2 = V'(sigma) dsigma/deps2
              - lambda * (dAdv/dsigma * dsigma/deps2
                          + dAdv/dp * dp2/deps2)

with the implicit-function-theorem derivatives

  dsigma/deps2 = (|G2|/Fp2) / (|G1| Fs1/Fp1 + |G2| Fs2/Fp2)
  dp2/deps2    = (1 - Fs2 * dsigma/deps2) / Fp2

where Fs_i, Fp_i are the partials of the budget function
F(sigma, p) = eps_acct(sigma, p) at (sigma*, p_i*). The partials of F and
Adv are evaluated by high-order central differences on the smooth pieces
(the model is analytic between Renyi-order switches).

Run as a module for a validation table: implicit-chain derivative vs a
direct central difference of u2 with fresh mechanism solves.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.advantage import mia_advantage_subsampled_gaussian as adv
from game.mechanism import epsilon_rdp, solve_mechanism
from game.utility import value_fn

G = [5000, 5000]
B, T, DELTA = 128, 390, 1e-12
LAM = 1.0


def solve(e1: float, e2: float):
    r = solve_mechanism([e1, e2], G, B, T, DELTA, batch_rtol=1e-6)
    if not r.feasible:
        raise ValueError(f"infeasible profile ({e1}, {e2})")
    return r.sigma, float(r.sample_rates[0]), float(r.sample_rates[1])


def dF(sigma: float, p: float, h_rel: float = 1e-4):
    """(dF/dsigma, dF/dp) of the budget function at (sigma, p)."""
    hs, hp = sigma * h_rel, p * h_rel
    Fs = (epsilon_rdp(sigma + hs, p, T, DELTA)
          - epsilon_rdp(sigma - hs, p, T, DELTA)) / (2 * hs)
    Fp = (epsilon_rdp(sigma, p + hp, T, DELTA)
          - epsilon_rdp(sigma, p - hp, T, DELTA)) / (2 * hp)
    return Fs, Fp


def dAdv(sigma: float, p: float, h_rel: float = 1e-2,
         discretization: float = 1e-4):
    """(dAdv/dsigma, dAdv/dp) by central differences on a finer PLD."""
    hs, hp = sigma * h_rel, p * h_rel
    As = (adv(sigma + hs, p, T, discretization)
          - adv(sigma - hs, p, T, discretization)) / (2 * hs)
    Ap = (adv(sigma, p + hp, T, discretization)
          - adv(sigma, p - hp, T, discretization)) / (2 * hp)
    return As, Ap


def du2_deps2(e1: float, e2: float, verbose: bool = False) -> float:
    sigma, p1, p2 = solve(e1, e2)
    Fs1, Fp1 = dF(sigma, p1)
    Fs2, Fp2 = dF(sigma, p2)
    ds_de2 = (G[1] / Fp2) / (G[0] * Fs1 / Fp1 + G[1] * Fs2 / Fp2)
    dp2_de2 = (1.0 - Fs2 * ds_de2) / Fp2
    As, Ap = dAdv(sigma, p2)
    Vp = -2.0 * sigma / (1.0 + sigma * sigma) ** 2
    out = Vp * ds_de2 - LAM * (As * ds_de2 + Ap * dp2_de2)
    if verbose:
        print(f"  sigma={sigma:.5f} p1={p1:.6f} p2={p2:.6f}")
        print(f"  dsigma/de2={ds_de2:+.5e} dp2/de2={dp2_de2:+.5e}")
        print(f"  V' term={Vp * ds_de2:+.5e}  "
              f"Adv terms={-LAM * (As * ds_de2 + Ap * dp2_de2):+.5e}")
    return out


def u2(e1: float, e2: float) -> float:
    sigma, _, p2 = solve(e1, e2)
    return value_fn(sigma) - LAM * adv(sigma, p2, T)


def du2_direct(e1: float, e2: float, h_rel: float = 5e-3) -> float:
    h = e2 * h_rel
    return (u2(e1, e2 + h) - u2(e1, e2 - h)) / (2 * h)


if __name__ == "__main__":
    pts = [(2.9646, 2.9646), (1.12, 7.13), (4.0, 2.0)]
    print(f"{'(eps1, eps2)':>18} {'implicit chain':>15} {'direct FD':>12}")
    for e1, e2 in pts:
        a = du2_deps2(e1, e2, verbose=True)
        b = du2_direct(e1, e2)
        print(f"({e1:7.4f},{e2:7.4f}) {a:+15.6e} {b:+12.6e}")
