"""
End-to-end game evaluator. A `Game` holds:
  * `n` players with parameter profiles,
  * an action set `actions` (a sorted list of eps values),
  * a global mechanism config (group sizes, batch, iterations, delta).

Methods:
  * `evaluate_profile(eps_vec)` -> (sigma, p_vec, A_vec, utility_vec)
  * `payoff_grid_2player()` -> n-d array of utilities for plotting
  * `best_response(player_idx, eps_others)` -> argmax-utility action
  * `find_nash_equilibria()` -> list of pure NE
  * `best_response_dynamics(start, max_iter)` -> trajectory + final state
  * `stackelberg_coalition(coalition, target)` -> coalition's best joint action
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

from .mechanism import solve_mechanism_cached, MechanismResult
from .advantage import mia_advantage_subsampled_gaussian
from .utility import PlayerProfile, utility as compute_utility, value_fn


@dataclass
class GameConfig:
    group_sizes: Tuple[int, ...]
    expected_batch_size: int
    steps: int
    delta: float = 1e-12
    actions: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    players: Tuple[PlayerProfile, ...] = field(default_factory=tuple)


@dataclass
class ProfileEvaluation:
    eps_vec: Tuple[float, ...]
    sigma: float
    sample_rates: np.ndarray
    advantages: np.ndarray
    utilities: np.ndarray
    feasible: bool


class Game:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self._eval_cache: dict = {}
        assert len(cfg.players) == len(cfg.group_sizes), (
            f"#players {len(cfg.players)} != #groups {len(cfg.group_sizes)}"
        )

    @property
    def n(self) -> int:
        return len(self.cfg.players)

    # --------------------------------------------------------------
    # Single-profile evaluation
    # --------------------------------------------------------------
    def evaluate(self, eps_vec: Sequence[float]) -> ProfileEvaluation:
        key = tuple(float(e) for e in eps_vec)
        if key in self._eval_cache:
            return self._eval_cache[key]
        res: MechanismResult = solve_mechanism_cached(
            eps_target=list(key), group_sizes=list(self.cfg.group_sizes),
            expected_batch_size=self.cfg.expected_batch_size,
            steps=self.cfg.steps, delta=self.cfg.delta,
        )
        advs = np.array([
            mia_advantage_subsampled_gaussian(
                sigma=float(res.sigma), sample_rate=float(p),
                steps=self.cfg.steps,
            )
            for p in res.sample_rates
        ])
        utils = np.array([
            compute_utility(p, sigma=float(res.sigma), A_self=float(advs[i]))
            for i, p in enumerate(self.cfg.players)
        ])
        out = ProfileEvaluation(
            eps_vec=key, sigma=float(res.sigma),
            sample_rates=res.sample_rates, advantages=advs,
            utilities=utils, feasible=res.feasible,
        )
        self._eval_cache[key] = out
        return out

    # --------------------------------------------------------------
    # Best-response of one player
    # --------------------------------------------------------------
    def best_response(self, i: int, eps_others: Sequence[float]) -> Tuple[float, ProfileEvaluation]:
        best_u = -np.inf
        best_action = None
        best_eval = None
        for a in self.cfg.actions:
            full = list(eps_others)
            full.insert(i, float(a))
            ev = self.evaluate(full)
            if not ev.feasible:
                continue
            if ev.utilities[i] > best_u:
                best_u = ev.utilities[i]
                best_action = float(a)
                best_eval = ev
        return best_action, best_eval

    # --------------------------------------------------------------
    # Best-response dynamics  -> pure NE (if it converges)
    # --------------------------------------------------------------
    def best_response_dynamics(self, start: Sequence[float] | None = None,
                                max_iter: int = 30) -> List[Tuple[float, ...]]:
        if start is None:
            mid = self.cfg.actions[len(self.cfg.actions) // 2]
            current = [mid] * self.n
        else:
            current = list(start)
        history = [tuple(current)]
        for _ in range(max_iter):
            changed = False
            for i in range(self.n):
                eps_others = [current[j] for j in range(self.n) if j != i]
                ba, _ = self.best_response(i, eps_others)
                if ba is not None and ba != current[i]:
                    current[i] = ba
                    changed = True
            history.append(tuple(current))
            if not changed:
                break
        return history

    # --------------------------------------------------------------
    # Brute-force find all pure NE (small games only)
    # --------------------------------------------------------------
    def find_pure_nash_equilibria(self) -> List[Tuple[float, ...]]:
        from itertools import product
        actions = self.cfg.actions
        equilibria: List[Tuple[float, ...]] = []
        for profile in product(actions, repeat=self.n):
            ev = self.evaluate(profile)
            if not ev.feasible:
                continue
            is_ne = True
            for i in range(self.n):
                others = [profile[j] for j in range(self.n) if j != i]
                ba, _ = self.best_response(i, others)
                if ba is None:
                    is_ne = False
                    break
                # Allow tie: any utility-maximising action counts as best.
                ev_alt = self.evaluate(tuple(others[:i] + [ba] + others[i:]))
                if ev_alt.utilities[i] > ev.utilities[i] + 1e-9:
                    is_ne = False
                    break
            if is_ne:
                equilibria.append(tuple(profile))
        return equilibria

    # --------------------------------------------------------------
    # Coalition Stackelberg: a coalition C jointly commits to an action vector
    # over its members to maximise lambda * A_target  +  sum_{i in C} u_i;
    # the rest of the players best-respond.
    # --------------------------------------------------------------
    def stackelberg_coalition(self, coalition: Sequence[int], target: int,
                               lambda_target: float = 1.0,
                               non_coalition_action: float | None = None,
                               ) -> Tuple[Tuple[float, ...], ProfileEvaluation]:
        """For each coalition action profile, lock it in and best-respond the
        non-coalition players (or take a fixed default action). Return the
        coalition action that maximises target's MIA advantage (+ lambda *
        coalition's joint own-utility)."""
        from itertools import product
        actions = self.cfg.actions
        coalition = list(coalition)
        non_C = [i for i in range(self.n) if i not in coalition and i != target]
        best_obj = -np.inf
        best_vec: Tuple[float, ...] | None = None
        best_eval: ProfileEvaluation | None = None
        for coalition_act in product(actions, repeat=len(coalition)):
            eps_vec = [None] * self.n
            for ci, ai in zip(coalition, coalition_act):
                eps_vec[ci] = float(ai)
            if non_coalition_action is not None:
                for j in non_C:
                    eps_vec[j] = float(non_coalition_action)
                # target best-responds in this fixed setting
                eps_others = [eps_vec[j] for j in range(self.n) if j != target]
                ba_t, _ = self.best_response(target, eps_others)
                if ba_t is None:
                    continue
                eps_vec[target] = ba_t
            else:
                # full best-response by non-coalition players (greedy multi-round)
                for j in [target] + non_C:
                    eps_vec[j] = actions[len(actions) // 2]  # init
                for _ in range(10):
                    changed = False
                    for j in [target] + non_C:
                        eps_others = [eps_vec[k] for k in range(self.n) if k != j]
                        ba, _ = self.best_response(j, eps_others)
                        if ba is not None and ba != eps_vec[j]:
                            eps_vec[j] = ba
                            changed = True
                    if not changed:
                        break
            ev = self.evaluate(tuple(eps_vec))
            if not ev.feasible:
                continue
            A_target = float(ev.advantages[target])
            joint_own = float(sum(ev.utilities[ci] for ci in coalition))
            obj = lambda_target * A_target + joint_own
            if obj > best_obj:
                best_obj = obj
                best_vec = tuple(eps_vec)
                best_eval = ev
        return best_vec, best_eval


# -----------------------------------------------------------------------------
# Convenience constructors
# -----------------------------------------------------------------------------

def make_n_player_game(
    players: Sequence[PlayerProfile],
    group_sizes: Sequence[int],
    expected_batch_size: int,
    steps: int,
    actions: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    delta: float = 1e-12,
) -> Game:
    cfg = GameConfig(
        group_sizes=tuple(group_sizes),
        expected_batch_size=int(expected_batch_size),
        steps=int(steps), delta=float(delta),
        actions=tuple(float(a) for a in actions),
        players=tuple(players),
    )
    return Game(cfg)
