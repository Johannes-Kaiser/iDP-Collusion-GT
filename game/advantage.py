"""
Theoretical MIA advantage for a subsampled Gaussian mechanism (composed `steps`
times). We use the trade-off-function formulation: MIA advantage =
max_alpha (1 - alpha - f(alpha)), where f is the trade-off function of the
mechanism.

We compute f via the PLD (privacy loss distribution) library: given a PLD object,
the trade-off function at FPR alpha equals f(alpha) = sup_eps (1 - delta(eps) -
alpha * e^eps), where (eps, delta(eps)) traces the privacy profile. Equivalently,
the MIA advantage equals the *total-variation* distance of the worst-case
adjacent output distributions, which is achievable by the optimal LR test.

For our purposes we use the closed-form:
  Adv(M) = max_eps  ( 1 - exp(-eps) ) / 2  ... NO -- this is wrong.

Correct: for an (eps, delta)-DP mechanism, the maximum MIA advantage over all
priors and decision rules is
  Adv = (1 - delta - exp(-eps)) / (1 + exp(-eps))  +  delta / (1 + exp(-eps))?
This is not tight for the *whole* trade-off function.

The cleanest robust route: use the dp_accounting PLD object's
`get_delta_for_epsilon(eps)` to trace the privacy profile, then compute
  Adv = sup_{eps >= 0} (1 - delta(eps) - exp(-eps)) / (1 + exp(-eps)) +  ...

Actually, the *tightest* MIA advantage from a privacy profile {(eps, delta(eps))}
is given by Kaissis et al. (2024) / Hayes et al.:
  Adv = sup_{eps>=0}  ( exp(eps) - 1 + 2 delta(eps) ) / ( exp(eps) + 1 )
       = sup_eps  ( 1 - 2 * beta_optimal(alpha=1/(1+e^eps)) )
This is the f-DP / trade-off-function representation.

Equivalently and exactly, the MIA advantage equals
  Adv = sup_alpha (1 - alpha - f(alpha))
with f obtained from the PLD.

This module implements `mia_advantage_from_pld` using a grid over alpha and the
formula  f(alpha) = max_eps  ( 1 - delta(eps) - alpha * e^eps )^+
(this is the trade-off function corresponding to a privacy profile, see
Balle, Barthe, Gaboardi "Privacy Profiles and Amplification by Subsampling").
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
from dp_accounting import pld


def trade_off_function_from_pld(privacy_loss_dist, alpha_grid: np.ndarray,
                                eps_grid: np.ndarray | None = None) -> np.ndarray:
    """Compute f(alpha) for the given PLD using the dual representation:

       f(alpha) = max_eps  [ 1 - delta(eps) - alpha * exp(eps) ]+,  for alpha in (0,1).

    Equivalently this is the type-II error at type-I error alpha for the worst-case
    likelihood-ratio test.
    """
    if eps_grid is None:
        eps_grid = np.concatenate([
            np.linspace(0.0, 1.0, 30, endpoint=False),
            np.linspace(1.0, 10.0, 50, endpoint=False),
            np.linspace(10.0, 80.0, 40),
        ])
    delta_grid = np.array([
        privacy_loss_dist.get_delta_for_epsilon(float(e)) for e in eps_grid
    ])
    # f(alpha) = max_eps [ 1 - delta - alpha exp(eps) ]+
    f_alpha = np.zeros_like(alpha_grid)
    for k, a in enumerate(alpha_grid):
        vals = 1.0 - delta_grid - a * np.exp(eps_grid)
        f_alpha[k] = max(float(vals.max()), 0.0)
    # Trade-off function is symmetric: f >= 1-alpha would mean "no privacy",
    # f <= 0 means "fully revealing". Clip:
    f_alpha = np.clip(f_alpha, 0.0, 1.0)
    return f_alpha


def mia_advantage_from_pld(privacy_loss_dist,
                            alpha_grid: np.ndarray | None = None) -> float:
    """MIA advantage = sup_alpha (1 - alpha - f(alpha))."""
    if alpha_grid is None:
        alpha_grid = np.linspace(1e-4, 1.0 - 1e-4, 500)
    f = trade_off_function_from_pld(privacy_loss_dist, alpha_grid)
    return float(np.max(1.0 - alpha_grid - f))


@lru_cache(maxsize=200_000)
def mia_advantage_subsampled_gaussian(sigma: float, sample_rate: float,
                                       steps: int,
                                       discretization: float = 1e-3) -> float:
    """MIA advantage of an `steps`-fold composition of the (sample_rate, sigma)-
    subsampled Gaussian mechanism (Poisson sampling)."""
    if sample_rate <= 0 or sigma <= 0 or steps <= 0:
        return 0.0
    pld_obj = pld.privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma,
        sampling_prob=sample_rate,
        value_discretization_interval=discretization,
        use_connect_dots=True,
    )
    pld_obj = pld_obj.self_compose(steps)
    return mia_advantage_from_pld(pld_obj)


if __name__ == "__main__":
    # quick sanity check
    for p, s in [(0.01, 1.0), (0.05, 1.0), (0.1, 1.0), (0.001, 1.0)]:
        a = mia_advantage_subsampled_gaussian(sigma=s, sample_rate=p, steps=1953)
        print(f"sigma={s}, p={p}, I=1953, MIA advantage = {a:.4f}")
