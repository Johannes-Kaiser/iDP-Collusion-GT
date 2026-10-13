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
    min_sample_rate: float = 1e-5  # participation floor (0 disables it)

    def grid(self) -> np.ndarray:
        return np.logspace(np.log10(self.eps_min), np.log10(self.eps_max), self.K)

    def evaluate(self, eps_vec: Sequence[float]) -> dict:
        """Return mechanism solution + advantages + utilities for eps_vec."""
        res = solve_mechanism_cached(
            eps_target=list(eps_vec),
            group_sizes=list(self.group_sizes),
            expected_batch_size=self.expected_batch_size,
            steps=self.steps, delta=self.delta,
            min_sample_rate=self.min_sample_rate,
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


def sweep_2player_grid(game: ContinuousGame, n_workers: int = 1) -> dict:
    """K x K x 2 utility array, K x K x 2 advantage array, K x K sigma.

    Exploits two symmetries:

      * Mechanism symmetry: (eps_i, eps_j) and (eps_j, eps_i) share the
        same mechanism solution (same sigma, sample rates permuted).
        This is inherent to the mechanism regardless of the players'
        weights: the coordinator only sees a multiset of budgets.

      * Player-swap: for the (j, i) cell we reuse the (i, j) mechanism
        state and just swap the player-to-group assignment. Utility is
        V(sigma) - w_k * A_of_the_group_that_player_k_ends_up_playing.

    So we only ever evaluate the upper triangle (j >= i). This is a 2x
    saving on the expensive PLD-accounting work.

    When `n_workers > 1`, the upper triangle is farmed out to a process
    pool row-wise. Each worker keeps its own mechanism/advantage caches;
    each unique multiset is solved at most once per worker's stripe.
    """
    actions = game.grid()
    K = len(actions)
    U = np.full((K, K, 2), np.nan)
    A = np.full((K, K, 2), np.nan)
    S = np.full((K, K), np.nan)
    feas = np.zeros((K, K), dtype=bool)

    if n_workers <= 1:
        for i in range(K):
            e1 = float(actions[i])
            for j in range(i, K):
                e2 = float(actions[j])
                ev = game.evaluate((e1, e2))
                if ev["feasible"]:
                    U[i, j] = ev["utilities"]
                    A[i, j] = ev["advantages"]
                    S[i, j] = ev["sigma"]
                    feas[i, j] = True
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        stripes = _row_stripes(K, n_workers)
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for stripe in stripes:
                fut = pool.submit(_sweep_upper_stripe, game, stripe, K)
                futures[fut] = stripe
            for fut in as_completed(futures):
                stripe_U, stripe_A, stripe_S, stripe_feas, stripe = fut.result()
                i0, i1 = stripe[0], stripe[-1] + 1
                U[i0:i1] = stripe_U
                A[i0:i1] = stripe_A
                S[i0:i1] = stripe_S
                feas[i0:i1] = stripe_feas

    _mirror_lower_from_upper(U, A, S, feas, K, game)
    return dict(actions=actions, U=U, A=A, sigma=S, feasible=feas)


def _mirror_lower_from_upper(U, A, S, feas, K, game):
    """Fill lower triangle from upper. For (i, j) with i > j: the mechanism
    at multiset {i, j} was solved at (j, i) upper cell, where player 1 held
    eps_j and player 2 held eps_i. So sample rate at eps_j is stored as
    A[j, i, 0] (player 1's advantage there) and rate at eps_i as A[j, i, 1].
    For the (i, j) cell player 1 plays eps_i and player 2 plays eps_j, so:
        A[i, j, 0] = A[j, i, 1]   (player 1 at eps_i)
        A[i, j, 1] = A[j, i, 0]   (player 2 at eps_j)
    U follows from (sigma, A) and each player's weight.
    """
    from .utility import value_fn
    w1 = float(game.players[0].w)
    w2 = float(game.players[1].w)
    for i in range(K):
        for j in range(i):
            if not feas[j, i]:
                continue
            feas[i, j] = True
            S[i, j] = S[j, i]
            A[i, j, 0] = A[j, i, 1]
            A[i, j, 1] = A[j, i, 0]
            v = value_fn(float(S[i, j]))
            U[i, j, 0] = v - w1 * A[i, j, 0]
            U[i, j, 1] = v - w2 * A[i, j, 1]


def _row_stripes(K: int, n_workers: int):
    per = max(1, K // n_workers)
    stripes = []
    for start in range(0, K, per):
        stop = min(K, start + per)
        stripes.append(list(range(start, stop)))
    return stripes


def _sweep_upper_stripe(game: ContinuousGame, rows, K: int):
    actions = game.grid()
    n = len(rows)
    U = np.full((n, K, 2), np.nan)
    A = np.full((n, K, 2), np.nan)
    S = np.full((n, K), np.nan)
    feas = np.zeros((n, K), dtype=bool)
    for ii, i in enumerate(rows):
        e1 = float(actions[i])
        for j in range(i, K):
            e2 = float(actions[j])
            ev = game.evaluate((e1, e2))
            if ev["feasible"]:
                U[ii, j] = ev["utilities"]
                A[ii, j] = ev["advantages"]
                S[ii, j] = ev["sigma"]
                feas[ii, j] = True
    return U, A, S, feas, rows


# --------------------------------------------------------------------------
# Weight-independent precompute + weight-dependent utility derivation.
#
# The mechanism solve (sigma, per-group sampling rates, MIA advantages)
# depends only on the eps multiset, the group sizes, the expected batch
# size, the step count and delta. It does NOT depend on the players'
# privacy weights w_i. So we can precompute the mechanism grid once and
# derive the utility grid for many different (w_1, w_2) configurations at
# essentially zero extra cost.
# --------------------------------------------------------------------------

def precompute_mechanism_grid(
    K: int,
    group_sizes,
    expected_batch_size: int,
    steps: int,
    delta: float = 1e-12,
    eps_min: float = 0.5,
    eps_max: float = 64.0,
    n_workers: int = 1,
) -> dict:
    """Compute (sigma, per-group MIA advantage, feasibility) on the upper
    triangle of the K x K eps grid, exploiting the multiset symmetry of the
    mechanism. Returns a dict:
        actions:  (K,) log-spaced eps grid
        sigma:    (K, K) upper triangle populated
        A:        (K, K, 2) A[i, j, 0] = MIA advantage of the group with
                  eps_i at profile (eps_i, eps_j); A[i, j, 1] same for
                  eps_j. Lower triangle is mirrored.
        feasible: (K, K) boolean mirrored across the diagonal.
    """
    # Any two identical-weight players will do -- the returned advantages
    # and sigma are weight-independent.
    game = ContinuousGame(
        players=(PlayerProfile("dummy1", w=1.0),
                 PlayerProfile("dummy2", w=1.0)),
        group_sizes=tuple(group_sizes),
        expected_batch_size=int(expected_batch_size),
        steps=int(steps), delta=float(delta),
        eps_min=eps_min, eps_max=eps_max, K=K,
    )
    swept = sweep_2player_grid(game, n_workers=n_workers)
    return dict(
        actions=swept["actions"],
        sigma=swept["sigma"],
        A=swept["A"],
        feasible=swept["feasible"],
    )


def sweep_from_grid(mech_grid: dict, w1: float, w2: float) -> dict:
    """Derive a (U, A, sigma, feasible) sweep dict for the given weight pair
    from a precomputed mechanism grid. This is cheap: O(K^2) elementwise ops.
    """
    from .utility import value_fn
    actions = mech_grid["actions"]
    K = len(actions)
    sigma = mech_grid["sigma"]
    A = mech_grid["A"]
    feas = mech_grid["feasible"]
    U = np.full((K, K, 2), np.nan)
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            v = value_fn(float(sigma[i, j]))
            U[i, j, 0] = v - w1 * float(A[i, j, 0])
            U[i, j, 1] = v - w2 * float(A[i, j, 1])
    return dict(actions=actions, U=U, A=A, sigma=sigma, feasible=feas)


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
