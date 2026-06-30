"""
Higher-level solvers on top of the Game evaluator.

  * fictitious_play(...): converges to a mixed Nash equilibrium when one exists.
  * find_pareto_front(...): Pareto-optimal action profiles.
  * coalition_attack_sweep(...): for k = 1..n-1, find Stackelberg coalition's
    best joint move and the resulting target advantage.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

from .game import Game, ProfileEvaluation


def fictitious_play(game: Game, n_rounds: int = 200,
                     start: Sequence[float] | None = None,
                     seed: int = 0) -> Tuple[np.ndarray, List[Tuple[float, ...]]]:
    """Run fictitious play. Returns:
        empirical_mixed: (n, |A|) array of empirical action frequencies for each player,
        action_history: list of pure action profiles played each round.
    """
    rng = np.random.default_rng(seed)
    actions = list(game.cfg.actions)
    n = game.n
    K = len(actions)
    counts = np.zeros((n, K), dtype=int)
    if start is None:
        start = [actions[K // 2]] * n
    last = list(start)
    history: List[Tuple[float, ...]] = [tuple(last)]
    for i_a, a in enumerate(last):
        counts[i_a, actions.index(a)] += 1
    for t in range(n_rounds):
        # each player best-responds to empirical mixed strategy of others
        new = list(last)
        for i in range(n):
            # expected utility of action a_i given empirical play of others
            best_u = -np.inf
            best_a = last[i]
            for ai in actions:
                expected_u = 0.0
                weight_sum = 0.0
                # sample over empirical mixture of others: enumerate other-action joint freq
                # Use direct expectation: for each combination of other actions, prob = prod freq.
                # For small games this is exact.
                from itertools import product
                others_idx = [j for j in range(n) if j != i]
                joints = list(product(*[range(K) for _ in others_idx]))
                for joint in joints:
                    eps_vec = list(last)
                    eps_vec[i] = ai
                    p = 1.0
                    for slot, kj in zip(others_idx, joint):
                        p *= counts[slot, kj] / max(counts[slot].sum(), 1)
                        eps_vec[slot] = actions[kj]
                    if p == 0:
                        continue
                    ev = game.evaluate(eps_vec)
                    if not ev.feasible:
                        continue
                    expected_u += p * ev.utilities[i]
                    weight_sum += p
                if weight_sum > 0:
                    expected_u /= weight_sum
                if expected_u > best_u:
                    best_u = expected_u
                    best_a = ai
            new[i] = best_a
            counts[i, actions.index(best_a)] += 1
        last = new
        history.append(tuple(last))
    freqs = counts / counts.sum(axis=1, keepdims=True)
    return freqs, history


def payoff_grid_2player(game: Game) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a 2-player game, return the (K, K, 2) utility array, advantages, and
    sigma grid."""
    assert game.n == 2
    actions = list(game.cfg.actions)
    K = len(actions)
    U = np.zeros((K, K, 2))
    A = np.zeros((K, K, 2))
    S = np.zeros((K, K))
    for i, e1 in enumerate(actions):
        for j, e2 in enumerate(actions):
            ev = game.evaluate((e1, e2))
            if ev.feasible:
                U[i, j] = ev.utilities
                A[i, j] = ev.advantages
                S[i, j] = ev.sigma
            else:
                U[i, j] = np.nan
                A[i, j] = np.nan
                S[i, j] = np.nan
    return U, A, S


def coalition_attack_sweep(game: Game, target: int,
                            coalition_sizes: Sequence[int] | None = None,
                            non_coalition_action: float | None = None,
                            ) -> List[dict]:
    """For each k in coalition_sizes, compute the coalition's worst-case
    (target-vulnerability-maximising) joint move, with the rest of the players
    fixed at `non_coalition_action` (if given) or best-responding.

    Returns a list of dicts with k, coalition_indices, action_vec, sigma,
    A_target, A_coalition_mean.
    """
    if coalition_sizes is None:
        coalition_sizes = list(range(1, game.n))
    rest_pool = [i for i in range(game.n) if i != target]
    out = []
    for k in coalition_sizes:
        coalition = rest_pool[:k]
        vec, ev = game.stackelberg_coalition(
            coalition=coalition, target=target, lambda_target=10.0,
            non_coalition_action=non_coalition_action,
        )
        if vec is None:
            out.append({"k": k, "coalition": coalition,
                        "action_vec": None, "sigma": None,
                        "A_target": None, "feasible": False})
            continue
        out.append({
            "k": k,
            "coalition": coalition,
            "action_vec": vec,
            "sigma": ev.sigma,
            "A_target": float(ev.advantages[target]),
            "A_coalition_mean": float(np.mean([ev.advantages[i] for i in coalition])),
            "feasible": ev.feasible,
        })
    return out
