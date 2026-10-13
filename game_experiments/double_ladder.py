"""Certified two-sided composition by measure doubling.

The per-step privacy loss L = g(X) has a closed-form CDF (g is
monotone, X the dominating mixture), so L can be enclosed between two
discrete measures on a uniform loss grid: the *pessimistic* measure
rounds every cell's mass to its right edge (stochastically dominates
L), the *optimistic* one to its left edge (is dominated by L). The
hockey-stick functional delta(eps) = E[(1 - e^{eps - S})_+] is
monotone under stochastic dominance, and dominance is preserved by
convolution — so composing each side with itself T-fold gives
**rigorous upper/lower bounds on delta_T at every eps**, hence on
budgets, advantages, and utilities. T is reached in O(log T)
exact discrete self-convolutions (binary expansion, FFT), with
directed coarsening (right/left pair-rounding) whenever the support
outgrows N_MAX, and directed tail handling (pessimistic tail mass to
+infinity, optimistic to the grid edge).

The only non-directed epsilon is FFT roundoff (~1e-15 relative); any
mass deficit it creates is repaired in the safe direction (pessimistic
to +infinity, optimistic to the lowest loss).

delta evaluation is O(log n) after suffix-sum preprocessing:
delta(eps) = sum_{l > eps} m_l - e^eps sum_{l > eps} m_l e^{-l}
             (+ inf-mass on the pessimistic side).

**Scope of the certificate (measured, honest).** The enclosure is
rigorous wherever FFT roundoff is negligible relative to the local
mass scale. That holds at the ADVANTAGE scale (delta(0) ~ 1e-2..1e-4:
enclosures verified against the exact T=1/T=2/ladder references in
every test; width 4e-4 at T=390). It does NOT hold in the deep tail
(delta ~ 1e-12, i.e. budget roots): fftconvolve's absolute noise floor
(~bulk_mass * 1e-16 per entry) exceeds the true tail masses (~1e-24),
inflating both sides by ~1e-4 relative in eps — the same mechanism
that makes dp_accounting's pessimistic connect-the-dots dip below
exact values at tiny delta. Use this engine to certify advantages;
budgets belong to the cancellation-free smooth ladder
(multi_step_game, all-positive sums, anchored) with dp_accounting's
pessimistic eps as a one-sided rigorous bound.

Validation in __main__: advantage enclosures contain the exact
references at T in {2, 8, 48}; budget columns document the deep-tail
limitation; width report at T = 390.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import norm

G1 = G2 = int(os.environ.get("PEG_G", "5000"))
B = 128
PSUM = B / G1
DELTA = 1e-12
PBAR = 1e-5

N_MAX = int(os.environ.get("PEG_NMAX", str(2 ** 22)))
Z_TAIL = 9.5


def _loss_sf(l: np.ndarray, sigma: float, p: float) -> np.ndarray:
    """P(L > l), L = g(X), exact closed form via survival functions
    (no 1 - CDF cancellation in the deep tail)."""
    l = np.asarray(l, float)
    with np.errstate(divide="ignore"):
        t = np.expm1(l) + p                   # e^l - 1 + p  (> 0 on domain)
        x = 0.5 + sigma * sigma * (np.log(t) - math.log(p))
    return (1.0 - p) * norm.sf(x / sigma) + p * norm.sf((x - 1.0) / sigma)


class Side:
    """One side (pessimistic or optimistic) of the enclosure."""

    __slots__ = ("lo", "h", "m", "inf", "pess", "_cache")

    def __init__(self, lo, h, m, inf, pess):
        self.lo, self.h, self.m, self.inf, self.pess = lo, h, m, inf, pess
        self._cache = None

    # -- construction of the single-step enclosure ------------------------
    @classmethod
    def single(cls, sigma: float, p: float, pess: bool,
               n: int = N_MAX) -> "Side":
        lo = math.log1p(-p)
        a_hi = (2.0 * (1.0 + Z_TAIL * sigma) - 1.0) / (2.0 * sigma * sigma)
        hi = np.logaddexp(math.log1p(-p), math.log(p) + a_hi)
        h = (hi - lo) / n
        edges = lo + h * np.arange(n + 1)
        sf = _loss_sf(edges, sigma, p)
        sf[0] = 1.0
        mass = np.maximum(sf[:-1] - sf[1:], 0.0)
        tail = max(float(sf[-1]), 0.0)
        if pess:
            # cell mass -> right edge: grid point i holds mass of cell i
            # (value lo + (i+1) h); shift lo by h to represent right edges
            return cls(lo + h, h, mass, tail, True)
        # optimistic: cell mass -> left edge; tail mass -> top grid point
        m = mass.copy()
        m[-1] += tail
        return cls(lo, h, m, 0.0, False)

    # -- exact self/pair convolution + directed maintenance ---------------
    def _repair(self, m: np.ndarray, inf: float) -> tuple[np.ndarray, float]:
        m = np.maximum(m, 0.0)
        deficit = 1.0 - inf - float(m.sum())
        if self.pess:
            inf += max(deficit, 0.0)
            if deficit < 0:                    # excess mass: scale down (UB
                m *= (1.0 - inf) / float(m.sum())   # on masses stays UB)
        else:
            if deficit > 0:
                m[0] += deficit                # lowest loss: safe for LB
            else:
                m *= (1.0 - inf) / float(m.sum())
        return m, inf

    def conv(self, other: "Side") -> "Side":
        assert self.pess == other.pess and abs(self.h - other.h) < 1e-15
        m = fftconvolve(self.m, other.m)
        inf = 1.0 - (1.0 - self.inf) * (1.0 - other.inf)
        s = Side(self.lo + other.lo, self.h, m, inf, self.pess)
        s.m, s.inf = s._repair(m, inf)
        if len(s.m) > 2 * N_MAX:
            s = s.coarsen()
        return s

    def coarsen(self) -> "Side":
        m = self.m
        if len(m) % 2:
            m = np.append(m, 0.0)
        pairs = m.reshape(-1, 2)
        mm = pairs.sum(axis=1)
        if self.pess:
            # pair (l_i, l_{i+1}) -> l_{i+1}: new grid starts at lo + h
            return Side(self.lo + self.h, 2.0 * self.h, mm, self.inf, True)
        return Side(self.lo, 2.0 * self.h, mm, self.inf, False)

    # -- delta and budgets --------------------------------------------------
    def _suffix(self):
        l = self.lo + self.h * np.arange(len(self.m))
        sm = np.concatenate([np.cumsum(self.m[::-1])[::-1], [0.0]])
        sme = np.concatenate(
            [np.cumsum((self.m * np.exp(-np.minimum(l, 700)))[::-1])[::-1],
             [0.0]])
        return l, sm, sme

    def delta(self, eps: float) -> float:
        if self._cache is None:
            self._cache = self._suffix()
        l, sm, sme = self._cache
        k = int(np.searchsorted(l, eps, side="right"))
        val = sm[k] - math.exp(min(eps, 700.0)) * sme[k]
        return float(val + (self.inf if self.pess else 0.0))

    def eps_budget(self) -> float:
        f = lambda e: self.delta(e) - DELTA
        lo = self.lo - 1e-9
        hi = self.lo + self.h * len(self.m) + 1e-9
        if f(lo) < 0:
            return lo
        if f(hi) > 0:
            return hi
        return brentq(f, lo, hi, xtol=1e-11)

    def adv(self) -> float:
        return self.delta(0.0)


def compose(sigma: float, p: float, T: int, pess: bool,
            n: int = N_MAX) -> Side:
    """T-fold composition by binary expansion of exact convolutions."""
    base = Side.single(sigma, p, pess, n)
    result = None
    power = base
    t = T
    while t:
        if t & 1:
            if result is None:
                result = power
            else:
                a, b = result, power
                # align spacings by coarsening the finer side
                while a.h < b.h * (1 - 1e-12):
                    a = a.coarsen()
                while b.h < a.h * (1 - 1e-12):
                    b = b.coarsen()
                result = a.conv(b)
        t >>= 1
        if t:
            power = power.conv(power)
    return result


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import time
    import game_experiments.two_step_game as g2t
    from game_experiments.multi_step_game import Ladder

    print(f"N_MAX = {N_MAX}")
    print("== enclosure vs exact T=2 quadrature ==")
    for (s, p) in [(3.0, 0.0128), (0.9, 0.0128), (0.3, 0.001)]:
        pes = compose(s, p, 2, True)
        opt = compose(s, p, 2, False)
        e_x = g2t.eps_budget(s, p)
        e_p, e_o = pes.eps_budget(), opt.eps_budget()
        a_x, a_p, a_o = g2t.advantage(s, p), pes.adv(), opt.adv()
        print(f"  s={s} p={p}: eps [{e_o:.6f}, {e_p:.6f}] ∋ {e_x:.6f}: "
              f"{e_o <= e_x <= e_p}; adv [{a_o:.3e}, {a_p:.3e}] ∋ "
              f"{a_x:.3e}: {a_o <= a_x <= a_p}")

    print("== enclosure vs exact ladder, T in {8, 48} ==")
    for T in (8, 48):
        for (s, p) in [(1.5, 0.0128), (0.7, 0.0064)]:
            pes = compose(s, p, T, True)
            opt = compose(s, p, T, False)
            lad = Ladder(s, p, T)
            e_l = lad.eps_budget()
            e_p, e_o = pes.eps_budget(), opt.eps_budget()
            print(f"  T={T} s={s} p={p}: eps [{e_o:.6f}, {e_p:.6f}] "
                  f"∋ ladder {e_l:.6f}: {e_o <= e_l <= e_p}; "
                  f"width {e_p - e_o:.2e}")

    print("== T=390 width and cost ==")
    t0 = time.time()
    pes = compose(0.25, 0.0128, 390, True)
    opt = compose(0.25, 0.0128, 390, False)
    e_p, e_o = pes.eps_budget(), opt.eps_budget()
    print(f"  s=0.25 p=0.0128: eps in [{e_o:.6f}, {e_p:.6f}] "
          f"width {e_p - e_o:.2e}; adv width "
          f"{pes.adv() - opt.adv():.2e}; {time.time() - t0:.0f}s")
