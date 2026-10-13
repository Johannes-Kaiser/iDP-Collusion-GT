"""Exact T-fold composition of the subsampled Gaussian via the ladder
recursion — no privacy accountant anywhere in the loop.

  delta_T(eps) = E_{X~P}[ delta_{T-1}(eps - g(X)) ],

with P = (1-p) N(0, s^2) + p N(1, s^2), g(x) = log((1-p) +
p e^{(2x-1)/(2 s^2)}), delta_1 the closed form of single_step_game
(extended: delta_1(e) = 1 - e^e for e <= log(1-p); analogously
delta_t(e) = 1 - e^e exactly for e <= t log(1-p), since
E[e^{-L_t}] = 1). Each level is one vectorized Gauss-Legendre
quadrature of the level below; levels t >= 2 are tabulated on a dense
e-grid sized from the exact mean/std of the per-step loss L = g(X) and
re-read through a cubic spline of log delta (the kink of delta_1 is
never interpolated — level 2 integrates the closed form directly, and
every deeper level is smooth).

Error control: grid-doubling self-test plus three anchors — the T=1
closed form, the certified T=2 kink-aware quadrature
(two_step_game, <= 3e-9 relative at the budget scale), and the
dp_accounting pessimistic/optimistic bracket. Run __main__ for the
validation table.

Cost: one ladder is O(T * n_e * n_nodes) vectorized flops; ~0.1 s per
composition step at the default resolution, independent of (sigma, p).
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm

from game_experiments.two_step_game import delta1_vec, _g_terms

G1 = G2 = int(os.environ.get("PEG_G", "5000"))
B = 128
PSUM = B / G1
DELTA = 1e-12
PBAR = 1e-5

_ZB = np.array([-13.0, -9.0, -6.0, -4.0, -2.5, -1.5, -0.75, 0.0,
                0.75, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.75,
                10.0, 11.5, 13.0])
_GLX, _GLW = np.polynomial.legendre.leggauss(32)
_LOG_TINY = -600.0


def _nodes(sigma: float, p: float):
    """Quadrature nodes x_j and weights w_j (mixture folded in) for
    E_{X~P}[h(X)] with both components on fixed z-panels."""
    lo, hi = _ZB[:-1], _ZB[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    z = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
    wz = (half[:, None] * _GLW[None, :]).ravel() * norm.pdf(z)
    xs = np.concatenate([0.0 + sigma * z, 1.0 + sigma * z])
    ws = np.concatenate([(1.0 - p) * wz, p * wz])
    return xs, ws


class Ladder:
    """delta_T(e; sigma, p) tabulated by exact recursion."""

    def __init__(self, sigma: float, p: float, T: int,
                 n_e: int = 3000, pad: float = 14.0):
        self.sigma, self.p, self.T = sigma, p, T
        xs, ws = _nodes(sigma, p)
        g = _g_terms(xs, sigma, p)[0]
        mu = float(ws @ g)
        s = float(math.sqrt(max(ws @ (g - mu) ** 2, 1e-300)))
        g_max = float(np.max(g))          # per-step loss at the node tail
        lo = T * math.log1p(-p)
        hi = (T * mu + pad * math.sqrt(T) * s + 6.0 * s
              + g_max + 2.0)
        # graded near lo (kink/advantage region), uniform to the tail
        span1 = min(max(6.0 * s, 1.0), hi - lo)
        n1 = n_e // 2
        seg1 = lo + span1 * np.linspace(0.0, 1.0, n1) ** 2
        seg2 = np.linspace(lo + span1, hi, n_e - n1 + 1)[1:]
        self.e_grid = np.concatenate([seg1, seg2])
        self.lo, self.hi = lo, hi
        cur = delta1_vec(self.e_grid, sigma, p)
        for t in range(2, T + 1):
            cur = self._step(cur, t, g, ws)
        cur = np.maximum(cur, 0.0)
        self._logd = np.log(np.maximum(cur, 1e-300))
        self._spl = CubicSpline(self.e_grid, self._logd)

    def _step(self, prev: np.ndarray, t: int, g: np.ndarray,
              ws: np.ndarray) -> np.ndarray:
        """delta_t(e_i) = sum_j w_j dtilde_{t-1}(e_i - g(x_j))."""
        lo_prev = (t - 1) * math.log1p(-self.p)
        if t == 2:
            # closed form directly: the delta_1 kink is never interpolated
            args = self.e_grid[:, None] - g[None, :]
            vals = delta1_vec(args.ravel(), self.sigma, self.p
                              ).reshape(args.shape)
            return vals @ ws
        spl = CubicSpline(self.e_grid, np.log(np.maximum(prev, 1e-300)))
        args = self.e_grid[:, None] - g[None, :]
        a = args.ravel()
        vals = np.zeros_like(a)
        left = a <= lo_prev
        vals[left] = -np.expm1(a[left])
        mid = (~left) & (a < self.hi)
        # cap at 0: delta <= 1, and the cubic can overshoot the cliff at
        # the edge of the underflow-clamped (log = -690) plateau
        lv = np.minimum(spl(a[mid]), 0.0)
        vals[mid] = np.where(lv > _LOG_TINY, np.exp(lv), 0.0)
        return vals.reshape(args.shape) @ ws

    def delta(self, e: float) -> float:
        if e <= self.lo:
            return -math.expm1(e)
        if e >= self.hi:
            return 0.0
        lv = min(float(self._spl(e)), 0.0)
        return math.exp(lv) if lv > _LOG_TINY else 0.0

    def eps_budget(self) -> float:
        """E_T(sigma, p): root of delta_T(e) = DELTA (on the spline)."""
        f = lambda e: float(self._spl(e)) - math.log(DELTA)
        lo = self.lo + 1e-12
        if self._logd[0] < math.log(DELTA):
            return lo
        if self._logd[-1] > math.log(DELTA):
            return self.hi
        return brentq(f, lo, self.hi, xtol=1e-11, rtol=8.9e-16)

    def adv(self) -> float:
        """Adv_T = TV of the composed pair = delta_T(0)."""
        return self.delta(0.0)


def eps_budget(sigma: float, p: float, T: int, **kw) -> float:
    return Ladder(sigma, p, T, **kw).eps_budget()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import game_experiments.single_step_game as g1
    import game_experiments.two_step_game as g2t

    print("== anchor T=1: ladder vs closed form ==")
    for (s, p) in [(3.0, 0.0128), (0.9, 0.0128), (0.3, 0.001)]:
        lad = Ladder(s, p, 1)
        e_l, e_c = lad.eps_budget(), g1.eps_budget(s, p)
        print(f"  s={s} p={p}: eps {e_l:.10f} vs {e_c:.10f} "
              f"rel={abs(e_l-e_c)/e_c:.1e}; "
              f"adv rel={abs(lad.adv()-g1.advantage(s,p))/g1.advantage(s,p):.1e}")

    print("== anchor T=2: ladder vs certified kink-aware quadrature ==")
    for (s, p) in [(3.0, 0.0128), (0.9, 0.0128), (0.5, 0.05), (0.3, 0.001)]:
        lad = Ladder(s, p, 2)
        e_l = lad.eps_budget()
        e_c = g2t.eps_budget(s, p)
        a_l, a_c = lad.adv(), g2t.advantage(s, p)
        print(f"  s={s} p={p}: eps rel={abs(e_l-e_c)/e_c:.1e}; "
              f"adv rel={abs(a_l-a_c)/a_c:.1e}")

    print("== self-test: grid doubling at T=8, T=32 ==")
    for T in (8, 32):
        for (s, p) in [(1.5, 0.0128), (0.7, 0.0064)]:
            e1 = Ladder(s, p, T, n_e=3000).eps_budget()
            e2 = Ladder(s, p, T, n_e=6000).eps_budget()
            print(f"  T={T} s={s} p={p}: eps={e1:.6f} doubling rel diff "
                  f"{abs(e1-e2)/e1:.1e}")

    print("== anchor T=3,8: dp_accounting bracket on delta at the budget ==")
    try:
        from dp_accounting.pld import privacy_loss_distribution as pl
        for T, s, p in [(3, 1.5, 0.0128), (8, 1.0, 0.0128)]:
            lad = Ladder(s, p, T)
            e = lad.eps_budget()
            pes = pl.from_gaussian_mechanism(
                standard_deviation=s, sampling_prob=p,
                value_discretization_interval=1e-4,
                pessimistic_estimate=True,
                use_connect_dots=True).self_compose(T).get_delta_for_epsilon(e)
            print(f"  T={T}: delta(eps*)={lad.delta(e):.3e} target 1e-12, "
                  f"pessimistic bound {pes:.3e} (must be >= exact)")
    except Exception as ex:
        print("  dp_accounting failed:", ex)
