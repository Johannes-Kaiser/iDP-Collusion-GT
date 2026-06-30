"""
Bayesian Repeated Privacy-Externality Game (BR-PEG).

Two players, H (honest, fixed type theta_H) and M (malicious, knows the
mechanism but does not initially know theta_H). Per round:

  1. M picks (eps_M, n_M) from a policy.
  2. H best-responds with eps_H given its type theta_H = (B_H, C_H).
  3. Mechanism returns (sigma, p_H, p_M, A_H, A_M).
  4. M observes (eps_H, u_H) and updates a posterior over theta_H.
  5. M accrues  A_H ;  H accrues  u_H = B_H V(sigma) - C_H A_H.

We provide:
  * ContinuousGame-style mechanism wrappers,
  * H's best-response by argmax over a discrete grid,
  * M strategies: constant, posterior-greedy-Bayesian,
  * Simulator that runs T rounds.

For tractability the type set Theta_H is finite (a small grid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Callable, List

import numpy as np
import logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

from .mechanism import solve_mechanism_cached
from .advantage import mia_advantage_subsampled_gaussian
from .utility import PlayerProfile, value_fn


# -----------------------------------------------------------------------------
# Mechanism wrapper for arbitrary group sizes and a fixed eps vector.
# Group 0 = H, Group 1 = M.
# -----------------------------------------------------------------------------

def play_one(eps_H: float, eps_M: float, n_H: int, n_M: int,
              expected_batch: int, steps: int, delta: float) -> dict:
    """Single-round mechanism evaluation. Returns sigma, p, A for both
    groups, and feasibility flag."""
    res = solve_mechanism_cached(
        eps_target=[float(eps_H), float(eps_M)],
        group_sizes=[int(n_H), int(n_M)],
        expected_batch_size=int(expected_batch),
        steps=int(steps), delta=float(delta),
    )
    if not res.feasible:
        return dict(feasible=False, sigma=float(res.sigma), p=res.sample_rates,
                    A_H=0.0, A_M=0.0)
    A_H = mia_advantage_subsampled_gaussian(
        sigma=float(res.sigma), sample_rate=float(res.sample_rates[0]),
        steps=int(steps),
    )
    A_M = mia_advantage_subsampled_gaussian(
        sigma=float(res.sigma), sample_rate=float(res.sample_rates[1]),
        steps=int(steps),
    )
    return dict(feasible=True, sigma=float(res.sigma),
                p=res.sample_rates.copy(),
                A_H=float(A_H), A_M=float(A_M))


# -----------------------------------------------------------------------------
# H's best response. H is myopic: picks the eps_H that maximises one-shot u_H.
# -----------------------------------------------------------------------------

def H_best_response(B_H: float, C_H: float, eps_M: float, n_H: int, n_M: int,
                     eps_grid: np.ndarray, expected_batch: int, steps: int,
                     delta: float) -> Tuple[float, dict]:
    best_u = -np.inf
    best_eps = float(eps_grid[0])
    best_res = None
    for eps_H in eps_grid:
        res = play_one(float(eps_H), float(eps_M), int(n_H), int(n_M),
                        expected_batch, steps, delta)
        if not res["feasible"]:
            continue
        u_H = B_H * value_fn(res["sigma"]) - C_H * res["A_H"]
        if u_H > best_u:
            best_u = u_H
            best_eps = float(eps_H)
            best_res = res
    return best_eps, (best_res if best_res is not None else dict(feasible=False))


# -----------------------------------------------------------------------------
# Theta_H types and posterior bookkeeping.
# -----------------------------------------------------------------------------

@dataclass
class ThetaH:
    B: float
    C: float
    n: int

    def __repr__(self):
        return f"theta_H(B={self.B:g}, C={self.C:g}, n={self.n})"


@dataclass
class GameConfig:
    types: List[ThetaH]
    eps_grid: np.ndarray
    expected_batch: int
    steps: int
    delta: float = 1e-12
    posterior_noise: float = 0.05   # tolerance used when comparing observed u_H to model u_H


# -----------------------------------------------------------------------------
# M strategies.
# -----------------------------------------------------------------------------

def strategy_constant(eps_M: float, n_M: int):
    def pol(history, posterior, cfg):
        return float(eps_M), int(n_M)
    return pol


def strategy_posterior_greedy(eps_M_grid: np.ndarray, n_M: int,
                               n_lookahead_eps: int = 11):
    """Choose (eps_M, n_M=fixed) that maximises expected A_H under the
    current posterior over theta_H. eps_M_grid is the set of M's
    candidate budgets."""
    def pol(history, posterior, cfg):
        best_score = -np.inf
        best_eps_M = float(eps_M_grid[0])
        for eps_M in eps_M_grid:
            # Expected A_H over theta in posterior
            ex_A_H = 0.0
            total_p = 0.0
            for theta, p_theta in zip(cfg.types, posterior):
                if p_theta < 1e-9:
                    continue
                # Predict H's BR for this theta
                eps_H_br, res = H_best_response(
                    B_H=theta.B, C_H=theta.C, eps_M=float(eps_M),
                    n_H=theta.n, n_M=int(n_M),
                    eps_grid=cfg.eps_grid,
                    expected_batch=cfg.expected_batch,
                    steps=cfg.steps, delta=cfg.delta,
                )
                if not res.get("feasible", False):
                    continue
                ex_A_H += p_theta * res["A_H"]
                total_p += p_theta
            if total_p > 0:
                ex_A_H /= total_p
            if ex_A_H > best_score:
                best_score = ex_A_H
                best_eps_M = float(eps_M)
        return best_eps_M, int(n_M)
    return pol


# -----------------------------------------------------------------------------
# Posterior update: M observes (eps_H_obs, u_H_obs); compute the likelihood
# of each candidate theta_H matching that observation.
# -----------------------------------------------------------------------------

def posterior_update(posterior, eps_M, n_M, eps_H_obs, u_H_obs, cfg):
    new_post = np.zeros_like(posterior)
    for k, theta in enumerate(cfg.types):
        if posterior[k] < 1e-9:
            continue
        eps_H_pred, res = H_best_response(
            B_H=theta.B, C_H=theta.C, eps_M=float(eps_M),
            n_H=theta.n, n_M=int(n_M),
            eps_grid=cfg.eps_grid,
            expected_batch=cfg.expected_batch,
            steps=cfg.steps, delta=cfg.delta,
        )
        if not res.get("feasible", False):
            continue
        u_H_pred = theta.B * value_fn(res["sigma"]) - theta.C * res["A_H"]
        # Likelihood: Gaussian on the predicted (eps_H, u_H) vs observed.
        # Use log-space eps comparison since eps is log-spaced.
        d_eps = np.log(max(eps_H_pred, 1e-3)) - np.log(max(eps_H_obs, 1e-3))
        d_u = u_H_pred - u_H_obs
        sig = cfg.posterior_noise
        log_lik = -0.5 * ((d_eps/sig)**2 + (d_u/sig)**2)
        new_post[k] = posterior[k] * np.exp(log_lik)
    # Normalise
    Z = new_post.sum()
    if Z > 0:
        new_post /= Z
    else:
        new_post = posterior.copy()
    return new_post


# -----------------------------------------------------------------------------
# Simulator.
# -----------------------------------------------------------------------------

def simulate(theta_H_true: ThetaH, m_policy: Callable, cfg: GameConfig,
              T: int = 20, initial_posterior=None, verbose: bool = True
              ) -> dict:
    K = len(cfg.types)
    if initial_posterior is None:
        posterior = np.ones(K) / K
    else:
        posterior = np.array(initial_posterior, dtype=float)
        posterior /= posterior.sum()
    history = []
    for t in range(T):
        # 1. M picks (eps_M, n_M)
        eps_M_t, n_M_t = m_policy(history, posterior, cfg)
        # 2. H best-responds
        eps_H_t, res_H = H_best_response(
            B_H=theta_H_true.B, C_H=theta_H_true.C, eps_M=eps_M_t,
            n_H=theta_H_true.n, n_M=n_M_t,
            eps_grid=cfg.eps_grid, expected_batch=cfg.expected_batch,
            steps=cfg.steps, delta=cfg.delta,
        )
        if not res_H.get("feasible", False):
            if verbose:
                print(f"  t={t}: infeasible profile; skipping update")
            history.append(dict(t=t, eps_M=eps_M_t, n_M=n_M_t,
                                  eps_H=eps_H_t, feasible=False,
                                  A_H=0.0, A_M=0.0, u_H=0.0,
                                  sigma=float("nan"),
                                  posterior_before=posterior.copy()))
            continue
        # 3. Reveal
        u_H_t = theta_H_true.B * value_fn(res_H["sigma"]) - theta_H_true.C * res_H["A_H"]
        # 4. M's posterior update
        post_before = posterior.copy()
        posterior = posterior_update(
            posterior, eps_M_t, n_M_t, eps_H_t, u_H_t, cfg,
        )
        # 5. Record
        history.append(dict(
            t=t, eps_M=eps_M_t, n_M=n_M_t,
            eps_H=eps_H_t, sigma=res_H["sigma"],
            A_H=res_H["A_H"], A_M=res_H["A_M"], u_H=u_H_t,
            feasible=True,
            posterior_before=post_before, posterior_after=posterior.copy(),
        ))
        if verbose:
            print(f"  t={t}: eps_M={eps_M_t:.2f}, eps_H={eps_H_t:.2f}, "
                  f"A_H={res_H['A_H']:.3f}, u_H={u_H_t:.3f}, "
                  f"posterior={posterior.round(3).tolist()}")
    return dict(history=history, posterior_final=posterior,
                theta_H_true=theta_H_true, cfg=cfg)
