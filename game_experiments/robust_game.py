"""Parametrized exact PEG for holistic robustness testing.

Every modeling assumption from nummeric_eval.md Sec. 23 is a
constructor knob instead of a module constant:

  T          composition steps (1 = closed form, 2 = certified
             quadrature; both exact, no accountant)
  G          tuple of group sizes (n players, n >= 2)
  B          expected batch size
  pbar       participation floor                       [register A4]
  delta      contract delta                            [register A5]
  box        action interval (eps_min, eps_max)        [register A5]
  V_form     value function shape                      [register A2]
             'inv_sq' 1/(1+s^2) | 'inv' 1/(1+s) | 'exp' e^{-s}
  cost_form  exposure disutility shape                 [register A2]
             'linear' lam*A | 'quad' lam*A^2
  lam        privacy weight (per player, scalar or tuple)

The privacy profile itself (subsampled Gaussian, delta_T(eps)) is the
one thing NOT varied — it is the object of study.

Only *value-level* quantities are computed (utilities at sampled
actions, corner ties, argmaxes); no FOC machinery, so the battery in
run_robustness.py is shape-agnostic in the cost/value forms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from game_experiments.two_step_game import delta1_vec, delta2


@dataclass
class Game:
    T: int = 1
    G: tuple = (5000, 5000)
    B: float = 128.0
    pbar: float = 1e-5
    delta: float = 1e-12
    box: tuple = (0.5, 64.0)
    V_form: str = "inv_sq"
    cost_form: str = "linear"
    lam: object = 0.7          # scalar or per-player tuple

    def lam_i(self, i: int) -> float:
        return self.lam[i] if isinstance(self.lam, (tuple, list)) else self.lam

    # -- exact privacy profile ---------------------------------------------
    def profile(self, e: float, s: float, p: float) -> float:
        if self.T == 1:
            return float(delta1_vec(np.array([e]), s, p)[0])
        return delta2(e, s, p)

    def eps_budget(self, s: float, p: float) -> float:
        f = lambda e: self.profile(e, s, p) - self.delta
        lo = 1e-9
        if f(lo) < 0:
            return lo
        hi = self.T * ((0.5 + 8.0 * s) / (s * s) + 100.0) * 1.5
        while f(hi) > 0:
            hi *= 2.0
        return brentq(f, lo, hi, xtol=1e-12, rtol=8.9e-16)

    def p_for_eps(self, s: float, e: float) -> float:
        f = lambda p: self.profile(e, s, p) - self.delta
        lo, hi = 1e-12, 1.0 - 1e-12
        if f(hi) < 0:
            return 1.0
        if f(lo) > 0:
            return 0.0
        return brentq(f, lo, hi, xtol=1e-15, rtol=8.9e-16)

    def solve(self, eps_vec):
        """(sigma, rates) with all budgets binding + batch; None if
        infeasible or any rate below pbar."""
        def excess(s):
            return sum(g * self.p_for_eps(s, e)
                       for g, e in zip(self.G, eps_vec)) - self.B
        lo, hi = 1e-2, 1e3
        if excess(lo) > 0 or excess(hi) < 0:
            return None
        s = brentq(excess, lo, hi, xtol=1e-11, rtol=1e-12)
        ps = [self.p_for_eps(s, e) for e in eps_vec]
        if any(p < self.pbar for p in ps):
            return None
        return s, ps

    # -- utility pieces ------------------------------------------------------
    def value(self, s: float) -> float:
        if self.V_form == "inv_sq":
            return 1.0 / (1.0 + s * s)
        if self.V_form == "inv":
            return 1.0 / (1.0 + s)
        if self.V_form == "exp":
            return math.exp(-s)
        raise ValueError(self.V_form)

    def advantage(self, s: float, p: float) -> float:
        if self.T == 1:
            return p * (2.0 * norm.cdf(0.5 / s) - 1.0)
        return delta2(0.0, s, p)

    def disutility(self, i: int, A: float) -> float:
        if self.cost_form == "linear":
            return self.lam_i(i) * A
        if self.cost_form == "quad":
            return self.lam_i(i) * A * A
        raise ValueError(self.cost_form)

    def utilities(self, eps_vec):
        r = self.solve(eps_vec)
        if r is None:
            return None
        s, ps = r
        V = self.value(s)
        return [V - self.disutility(i, self.advantage(s, p))
                for i, p in enumerate(ps)], s, ps

    # -- frontier of player j given the others' budgets ----------------------
    def feasible_e2(self, e1: float):
        """2-player: [e2_lo, e2_hi] given e1 (others analogous)."""
        psum = self.B / self.G[1]      # p1*G0/G1 accounting handled by roots
        def s_for(e, p_own_g0):
            f = lambda s: self.eps_budget(s, p_own_g0) - e
            return brentq(f, 1e-3, 1e4, xtol=1e-10, rtol=1e-11)
        # player 2 at floor: sigma from player 1 absorbing the batch
        p1_hi = (self.B - self.G[1] * self.pbar) / self.G[0]
        s_lo = s_for(e1, p1_hi)
        e2_lo = self.eps_budget(s_lo, self.pbar)
        s_hi = s_for(e1, self.pbar)
        p2_hi = (self.B - self.G[0] * self.pbar) / self.G[1]
        e2_hi = self.eps_budget(s_hi, p2_hi)
        return e2_lo, e2_hi
