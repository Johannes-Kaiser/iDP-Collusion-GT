"""Exact equilibrium-segment endpoints for the T=1 group-size sweep
(lam = 0.7, B = 128 fixed, |G| set via PEG_G).

The wall-segment equilibria live on the participation frontier
{(e1, f(e1))} with f = e2_hi. Mutuality requires the loose player's
column to be corner_hi and the tight player's column (at f(e1)) to be
corner_lo; the kind switch is a value tie u(corner_hi) = u(corner_lo),
polished here by Brent. The box [0.5, 64] clips the e1-range. When no
switch exists (all corner_hi) the only candidate is the ceiling
(64, 64), which is an NE iff f(64) >= 64.

Prints one JSON line. Run: PEG_G=... PEG_LAM=0.7 python -m ...
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.optimize import brentq

import game_experiments.single_step_br as br

G = br.gm.G1
LAM = br.LAM


def corner_utils(e1: float):
    e2_lo, e2_hi = br.feasible_interval(e1)
    u_lo, _ = br.u2_and_D(e1, e2_lo * (1 + 1e-9))
    u_hi, _ = br.u2_and_D(e1, e2_hi * (1 - 1e-9))
    return u_lo, u_hi, e2_lo, e2_hi


def tie(e1: float) -> float:
    u_lo, u_hi, _, _ = corner_utils(e1)
    return u_hi - u_lo


out = dict(G=G, lam=LAM, psum=br.PSUM)

t_floor, t_ceil = tie(0.5), tie(64.0)
_, _, lo05, f05 = corner_utils(0.5)
_, _, lo64, f64 = corner_utils(64.0)
out["f_05"], out["f_64"] = f05, f64
out["finv_of_ceiling_hits_box"] = f64 >= 64.0

if t_floor > 0 and t_ceil > 0:
    # all corner_hi: ceiling NE iff frontier exits the box above 64
    out["kind"] = "ceiling" if f64 >= 64.0 else "no_pure_NE"
    out["ne"] = [[64.0, 64.0]] if f64 >= 64.0 else []
elif t_floor > 0 > t_ceil:
    e1_star = brentq(tie, 0.5, 64.0, xtol=1e-10)
    _, _, finv_star, y_star = corner_utils(e1_star)
    # segment: e1 in [max(0.5, f^{-1}(e1*)), e1*]; f^{-1} = e2_lo by the
    # frontier involution, so the tight player's column is corner_lo
    # exactly when f(e1) >= e1*
    e1_low = max(0.5, finv_star)
    _, _, _, y_low = corner_utils(e1_low)
    out["kind"] = "frontier_segment"
    out["e1_star"] = e1_star
    out["finv_star"] = finv_star
    out["segment"] = [[e1_low, y_low], [e1_star, y_star]]
else:
    out["kind"] = "other"

print(json.dumps(out))
