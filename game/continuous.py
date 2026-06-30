"""
Continuous-action Privacy Externality Game.

For visualisation and equilibrium analysis we replace the coarse discrete
action grid by a fine log-spaced grid (default 25 points in [eps_min,
eps_max]). The mechanism solver is invoked at each grid point; results are
cached. The "continuous" approximation is in the *limit* of grid refinement.

We provide:
  * sweep_2player_grid: evaluate the full K x K payoff array.
  * find_pure_NE_2player: brute-force pure-NE finder on the dense grid.
  * best_response_curve: BR_i(eps_-i) returned as an array.
  * fixed_point_BR: simultaneous-BR fixed-point iteration on the dense grid.

The dense grid is "continuous enough" that NE positions converge as the grid
is refined; we report sensitivity to grid resolution in the appendix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

logging.getLogger("absl").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from .mechanism import solve_mechanism_cached
from .advantage import mia_advantage_subsampled_gaussian
from .utility import PlayerProfile, utility as compute_utility, value_fn


@dataclass
class ContinuousGame:
    players: Tuple[PlayerProfile, ...]
    group_sizes: Tuple[int, ...]
    expected_batch_size: int
    steps: int
    delta: float = 1e-12
    eps_min: float = 0.5
    eps_max: float = 64.0
    K: int = 25  # number of log-spaced points

    def grid(self) -> np.ndarray:
        return np.logspace(np.log10(self.eps_min), np.log10(self.eps_max), self.K)

    def evaluate(self, eps_vec: Sequence[float]) -> dict:
        """Return mechanism solution + advantages + utilities for eps_vec."""
        res = solve_mechanism_cached(
            eps_target=list(eps_vec),
            group_sizes=list(self.group_sizes),
            expected_batch_size=self.expected_batch_size,
            steps=self.steps, delta=self.delta,
        )
        advs = np.array([
            mia_advantage_subsampled_gaussian(
                sigma=float(res.sigma), sample_rate=float(p),
                steps=self.steps,
            )
            for p in res.sample_rates
        ])
        utils = np.array([
            compute_utility(p, sigma=float(res.sigma), A_self=float(advs[i]))
            for i, p in enumerate(self.players)
        ])
        return dict(
            eps_vec=tuple(float(e) for e in eps_vec),
            sigma=float(res.sigma),
            rates=res.sample_rates,
            advantages=advs,
            utilities=utils,
            feasible=bool(res.feasible),
        )


def sweep_2player_grid(game: ContinuousGame) -> dict:
    """K x K x 2 utility array, K x K x 2 advantage array, K x K sigma."""
    actions = game.grid()
    K = len(actions)
    U = np.full((K, K, 2), np.nan)
    A = np.full((K, K, 2), np.nan)
    S = np.full((K, K), np.nan)
    feas = np.zeros((K, K), dtype=bool)
    for i, e1 in enumerate(actions):
        for j, e2 in enumerate(actions):
            ev = game.evaluate((float(e1), float(e2)))
            if ev["feasible"]:
                U[i, j] = ev["utilities"]
                A[i, j] = ev["advantages"]
                S[i, j] = ev["sigma"]
                feas[i, j] = True
    return dict(actions=actions, U=U, A=A, sigma=S, feasible=feas)


def find_pure_NE_2player(sweep: dict) -> list:
    """Brute-force pure-NE finder. NE = (i, j) such that player 1's i is
    argmax over feasible rows of column j AND player 2's j is argmax over
    feasible columns of row i."""
    U = sweep["U"]
    feas = sweep["feasible"]
    K = U.shape[0]
    nes = []
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            # P1: argmax over rows i' with feas[i', j] of U[i', j, 0]
            col = U[:, j, 0].copy()
            col[~feas[:, j]] = -np.inf
            i_br = int(np.argmax(col))
            # P2: argmax over cols j' with feas[i, j'] of U[i, j', 1]
            row = U[i, :, 1].copy()
            row[~feas[i, :]] = -np.inf
            j_br = int(np.argmax(row))
            if i_br == i and j_br == j:
                nes.append((i, j))
    return nes


def best_response_curve(sweep: dict, player: int) -> np.ndarray:
    """Return an array BR[j] giving the argmax-index for `player`'s BR as a
    function of the OTHER player's index j. (For 2-player.)"""
    U = sweep["U"]
    feas = sweep["feasible"]
    K = U.shape[0]
    if player == 0:
        out = np.full(K, -1, dtype=int)
        for j in range(K):
            col = U[:, j, 0].copy()
            col[~feas[:, j]] = -np.inf
            if np.all(np.isneginf(col)):
                continue
            out[j] = int(np.argmax(col))
        return out
    else:
        out = np.full(K, -1, dtype=int)
        for i in range(K):
            row = U[i, :, 1].copy()
            row[~feas[i, :]] = -np.inf
            if np.all(np.isneginf(row)):
                continue
            out[i] = int(np.argmax(row))
        return out
