"""
Player utility for the Privacy Externality Game.

Canonical formulation (single utility term + single privacy term):

    u_i(eps_vec)  =  B_i * V( sigma(eps_vec) )  -  C_i * A_i(eps_vec)

* V(sigma) is the *shared* model-utility proxy. It is decreasing in the noise
  multiplier sigma of the sampling-based iDP mechanism.

  Choice of functional form. We use  V(sigma) = 1 / (1 + sigma**2)  motivated
  by the excess-empirical-risk bound for convex DP-ERM of Bassily, Smith and
  Thakurta (FOCS 2014), which proves that the excess empirical risk of
  DP-SGD on a Lipschitz strongly-convex loss scales as
       O( d * log(1/delta) / (n * eps)**2 )
  i.e. linearly in sigma**2 (since sigma_per_step ~ 1/eps under the moments
  accountant of Abadi et al. CCS 2016). The convex bound is the only setting
  with a closed-form utility theorem; we therefore use it as a *qualitative*
  proxy in our (potentially non-convex) game-theoretic analysis. Empirical
  studies of DP-SGD in deep learning (e.g. Tramer & Boneh, ICLR 2021) confirm
  the qualitative direction: test accuracy monotonically degrades with sigma.

* A_i is the realised MIA advantage on group i, computed exactly from the
  privacy profile of the I-fold (p_i, sigma)-subsampled Gaussian (Kaiser et
  al., SaTML 2026).

The naive vs aware distinction we previously made conflated *utility shape*
with *information asymmetry*. We drop it from the utility itself: there is a
single objective every rational player optimises. The discussion of what a
contract-only-aware ("naive") player would compute moves to a remark in the
paper.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerProfile:
    """Per-player weights for the canonical utility u_i = B_i V(sigma) - C_i A_i."""
    name: str
    B: float = 1.0       # utility weight on model quality
    C: float = 1.0       # cost weight on realised MIA advantage


def value_fn(sigma: float) -> float:
    """V(sigma) = 1 / (1 + sigma**2).

    Frame: motivated by the excess-empirical-risk bound for convex DP-ERM
    (Bassily, Smith, Thakurta, FOCS 2014). The convex bound scales as
    O(sigma**2 * d / n**2). Our value V is the inverse: large when sigma is
    small (low noise, high accuracy), small when sigma is large. We use it
    as a directional proxy; the empirical literature on DP-SGD in deep
    learning (Abadi et al., CCS 2016; Tramer & Boneh, ICLR 2021) confirms
    monotonic degradation of utility with sigma.
    """
    return 1.0 / (1.0 + sigma * sigma)


def utility(player: PlayerProfile, sigma: float, A_self: float) -> float:
    return player.B * value_fn(sigma) - player.C * A_self


# Convenience constructors retained for backward compatibility with existing
# experiment scripts.
def aware_player(name: str, B: float = 1.0, C: float = 1.0, eps_max: float = 64.0) -> PlayerProfile:
    return PlayerProfile(name=name, B=B, C=C)


def naive_player(name: str, B: float = 1.0, C: float = 1.0, eps_max: float = 64.0) -> PlayerProfile:
    # In the single-utility-term formulation there is no separate "naive"
    # archetype: every player has the same utility shape. The distinction
    # made in earlier drafts ("nominal cost only") has been removed.
    return PlayerProfile(name=name, B=B, C=C)
