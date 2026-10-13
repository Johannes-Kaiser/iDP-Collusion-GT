"""Do the participation walls (and hence the wall NE) exist inside the
action box under the ANALYTIC (limit) accounting?

In the limit model the per-step relation is mu_i = p_i * c(sigma),
c(sigma) = sqrt(T (e^{1/sigma^2} - 1)); the batch constraint
G(p1+p2) = B, i.e. p1 + p2 = PSUM, gives p_i = PSUM * mu_i / s with
s = mu1 + mu2. The participation frontier is p_i = pbar, i.e. the ray

    mu_i / s = pbar / PSUM.

Question: does this ray intersect the box eps in [0.5, 64]^2?
Compare with the exact game's wall (from the K=200 cache), which cuts
deep through the box.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import game_experiments.gdp_limit_game as g
from game.continuous import sweep_from_grid

PBAR = 1e-5
T = 390.0

mu = lambda e: brentq(lambda m: g.delta_gdp(float(e), m) - g.DELTA, 1e-6, 60.0)

m_lo, m_hi = mu(0.5), mu(64.0)
ratio_wall = PBAR / g.PSUM
print(f"mu(0.5) = {m_lo:.5f}, mu(64) = {m_hi:.5f}")
print(f"analytic wall ray: mu_i / s = pbar/PSUM = {ratio_wall:.2e}")

# worst case inside the box: own mu minimal, opponent mu maximal
worst = m_lo / (m_lo + m_hi)
print(f"minimum mu_i/s anywhere in the box = {worst:.4f}")
print(f"margin: worst case is {worst / ratio_wall:.0f}x ABOVE the wall")

# where WOULD the wall sit? opponent eps needed to push a player at
# eps=0.5 onto the wall: m_lo/(m_lo+m_opp) = ratio -> m_opp = m_lo(1/ratio - 1)
m_opp_needed = m_lo * (1.0 / ratio_wall - 1.0)
print(f"opponent mu needed to pin an eps=0.5 player to the floor: "
      f"{m_opp_needed:.1f} (box max is {m_hi:.2f})")

# exact game for contrast: fraction of the box that is infeasible
mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
feas = sweep_from_grid(mech, 1.0, 1.0)["feasible"]
print(f"\nexact game: {100 * (1 - feas.mean()):.1f}% of the K=200 box is "
      f"infeasible (the walls cut through it)")
print("analytic game: 0.0% infeasible (wall lies outside the box)")
