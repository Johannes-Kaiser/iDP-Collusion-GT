"""
Player utility for the Privacy Externality Game.

    u_i(eps_vec)  =  V( sigma(eps_vec) )  -  w_i * A_i(eps_vec)

* V(sigma) is the *shared* model-utility proxy. It is decreasing in the
  noise multiplier sigma of the sampling-based iDP mechanism.

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

* w_i > 0 is the player's privacy weight: how much she values a unit of
  realised MIA advantage on her own data relative to a unit of model
  quality. Larger w_i means she weights privacy more; smaller w_i means
  she weights model quality more.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerProfile:
    """Per-player weight for the canonical utility u_i = V(sigma) - w_i A_i."""
    name: str
    w: float = 1.0


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
    return value_fn(sigma) - player.w * A_self


# Convenience constructors.
def aware_player(name: str, w: float = 1.0, eps_max: float = 64.0) -> PlayerProfile:
    return PlayerProfile(name=name, w=w)


def naive_player(name: str, w: float = 1.0, eps_max: float = 64.0) -> PlayerProfile:
    # There is no separate "naive" archetype at the utility level: every
    # player has the same utility shape. The distinction between contract-
    # only-aware and mechanism-aware players is one of information, not of
    # utility.
    return PlayerProfile(name=name, w=w)
