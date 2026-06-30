"""
Player utility functions for the Privacy Externality Game (PEG).

Each player i picks a per-group budget eps_i from a discrete action set. Given
the action profile, the sampling-based iDP mechanism determines (sigma, p_g).
The MIA advantage on group i is A_i = mia_advantage_subsampled_gaussian(p_g_i,
sigma, steps). We support two awareness regimes:

  * naive:  cost is the *promised* nominal budget,  c_naive(eps_i) = eps_i / eps_max.
  * aware:  cost is the *actual* MIA advantage A_i.

Plus a shared cooperation-incentive benefit V(sigma): higher sigma -> noisier
model -> lower benefit. We use V(sigma) = 1 / (1 + sigma) by default (monotone
decreasing in sigma, bounded in [0, 1], differentiable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class PlayerProfile:
    """Per-player parameters. `B` weights the cooperation benefit, `C_aware`
    weights the actual privacy cost A_i, `C_naive` weights the perceived
    nominal cost.  At least one of (B, C_aware, C_naive) should be non-zero."""
    name: str
    B: float = 1.0
    C_aware: float = 0.0
    C_naive: float = 0.0
    awareness: str = "aware"  # "aware" or "naive" or "mixed"
    eps_max: float = 64.0

    def utility(self, sigma: float, A_self: float, eps_self: float,
                value_fn: Callable[[float], float] | None = None) -> float:
        if value_fn is None:
            value_fn = default_value
        v = value_fn(sigma)
        c_naive = eps_self / max(self.eps_max, 1e-9)
        c_naive = 1.0 - c_naive  # higher eps -> less privacy cost perceived
        c_naive = max(0.0, 1.0 - eps_self / max(self.eps_max, 1e-9))
        # Note: 'cost' = privacy *lost*, so higher eps -> lower privacy ->
        # smaller cost-of-protection but larger cost-of-leakage.
        # We model the cost-of-leakage:
        #   naive perceives  loss = eps_self / eps_max   (so higher eps = more nominal loss).
        #   aware perceives  loss = A_self                (so eps externalities show up).
        nominal_loss = eps_self / max(self.eps_max, 1e-9)
        return (self.B * v
                - self.C_aware * A_self
                - self.C_naive * nominal_loss)


def default_value(sigma: float) -> float:
    """Cooperation benefit, monotone-decreasing in sigma."""
    return 1.0 / (1.0 + sigma)


# Predefined player archetypes
def aware_player(name: str, B: float = 1.0, C: float = 1.0,
                 eps_max: float = 64.0) -> PlayerProfile:
    return PlayerProfile(name=name, B=B, C_aware=C, C_naive=0.0,
                         awareness="aware", eps_max=eps_max)


def naive_player(name: str, B: float = 1.0, C: float = 1.0,
                 eps_max: float = 64.0) -> PlayerProfile:
    return PlayerProfile(name=name, B=B, C_aware=0.0, C_naive=C,
                         awareness="naive", eps_max=eps_max)


def attacker(name: str, lambda_target: float = 1.0,
             eps_max: float = 64.0) -> PlayerProfile:
    """An attacker only cares about a target's vulnerability. Their utility is
    +A_target. Built by setting B=0, C_aware=0, C_naive=0; the attacker's
    'utility' is supplied separately in coalition_utility below."""
    return PlayerProfile(name=name, B=0.0, C_aware=0.0, C_naive=0.0,
                         awareness="attacker", eps_max=eps_max)
