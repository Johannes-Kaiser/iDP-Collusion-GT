"""Local grid refinement at the surviving diagonal grid-NE (eps = 2.9646).

The K=200 canonical game has exactly one on-diagonal pure NE cell. The
trough of prop:no-symmetric-ne is second-order, so within one grid step
of the diagonal FOC candidate the deviation gain can fall below the
K=200 action resolution. This script densifies the own action locally
(25 log-spaced points over +-2 grid steps) at fixed opponent
eps_2 = 2.9646 and reports the fine-grid argmax and gain: if the max is
off-diagonal with positive gain, the diagonal cell is a discretisation
artifact of the K=200 step, not an equilibrium of the continuous game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.mechanism import solve_mechanism
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn

EPS2 = 2.9646
STEPS, DELTA = 390, 1e-12


def u1(e1: float) -> float:
    r = solve_mechanism([e1, EPS2], [5000, 5000], 128, STEPS, DELTA)
    a1 = mia_advantage_subsampled_gaussian(r.sigma, float(r.sample_rates[0]), STEPS)
    return value_fn(r.sigma) - a1


def main() -> None:
    # +-2 K=200 grid steps around the diagonal (step = 1.0247x)
    step = (64.0 / 0.5) ** (1.0 / 199.0)
    grid = EPS2 * step ** np.linspace(-2.0, 2.0, 25)
    vals = np.array([u1(float(e)) for e in grid])
    at = u1(EPS2)
    k = int(np.argmax(vals))
    print(f"u1 at diagonal ({EPS2}): {at:.7f}")
    print(f"fine-grid max: u1={vals[k]:.7f} at eps1={grid[k]:.4f} "
          f"(gain {vals[k]-at:+.2e})")
    for e, v in zip(grid, vals):
        print(f"  eps1={e:.4f}  u1={v:.7f}")


if __name__ == "__main__":
    main()
