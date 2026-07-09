"""
Private level-2 utility oracle for Experiment E2 (adaptive agents).

This mirrors the physics that E1's ``game_experiments/flpop_common.py``
provides (population sampler + cached level-2 utility oracle). The physics
constants below are a SINGLE CONFIG SOURCE with an override hook: if
``flpop_common`` exists and exports the corresponding names, its values win,
so E1 and E2 always run the same physics. Switching E2 entirely over to the
shared oracle remains a one-line change in ``run_flpop_e2.py``.

Physics (identical to ``game.game.Game.evaluate``):
    * solve (sigma, p_1..p_n) from (eps_target, |G_i|, E_b, I, delta) via
      ``game.mechanism.solve_mechanism_cached`` (multiset-canonical cache).
    * A_i = ``game.advantage.mia_advantage_subsampled_gaussian(sigma, p_i, I)``
      (PLD-exact, lru-cached by (sigma, p, steps)).
    * V(sigma) = 1 / (1 + sigma**2); u_i = V - w_i * A_i.

Population sampler (plan A5, provisional representative = seed 0):
    n = 20; |G_i| ~ Uniform{2000..12000} integers; w_i ~ LogNormal(0, 0.7);
    numpy default_rng(pop_id).

The E_b rule is a function of the population size M so that batch scaling
(e.g. E_b = 0.0064*M) is representable; I = EPOCHS * floor(M / E_b).
"""
from __future__ import annotations

import logging
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logging.getLogger("absl").setLevel(logging.ERROR)

from game.mechanism import solve_mechanism_cached
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn

# --- Physics config (single source; flpop_common overrides if present) ----
N_PLAYERS = 20
ACTIONS: Tuple[float, ...] = (10.0, 14.0, 19.0, 26.0, 37.0, 50.0)
DELTA = 1e-12
EPOCHS = 10
GROUP_MIN, GROUP_MAX = 2000, 12000       # Uniform{2000..12000} inclusive
W_LOGNORM_SIGMA = 0.7


def expected_batch(M: int) -> int:
    """E_b rule. Default: fixed 256. Overridden by flpop_common if present."""
    return 256


def _apply_shared_config() -> str:
    """Pull (ACTIONS, EPOCHS, DELTA, E_b rule, sampler constants) from
    ``game_experiments.flpop_common`` when available. Accepts either a
    callable ``expected_batch(M)`` / ``EB_RULE`` or a scalar
    ``EXPECTED_BATCH`` / ``E_B`` / ``EB``. Returns the config source tag."""
    global ACTIONS, EPOCHS, DELTA, expected_batch
    global N_PLAYERS, GROUP_MIN, GROUP_MAX, W_LOGNORM_SIGMA
    try:
        from game_experiments import flpop_common as fc  # type: ignore
    except Exception:
        return "local-defaults"
    # name mapping: (local global, candidate names in flpop_common)
    for target, names in [("ACTIONS", ("ACTION_GRID", "ACTIONS")),
                          ("EPOCHS", ("EPOCHS",)),
                          ("DELTA", ("DELTA",)),
                          ("N_PLAYERS", ("N_PLAYERS",)),
                          ("GROUP_MIN", ("GROUP_MIN",)),
                          ("GROUP_MAX", ("GROUP_MAX",)),
                          ("W_LOGNORM_SIGMA", ("W_SIGMA", "W_LOGNORM_SIGMA"))]:
        for name in names:
            if hasattr(fc, name):
                globals()[target] = getattr(fc, name)
                break
    ACTIONS = tuple(float(a) for a in ACTIONS)
    eb = None
    for name in ("expected_batch_for", "expected_batch", "EB_RULE"):
        if callable(getattr(fc, name, None)):
            eb = getattr(fc, name)
            break
    if eb is None:
        for name in ("EXPECTED_BATCH", "E_B", "EB"):
            if hasattr(fc, name):
                val = getattr(fc, name)
                eb = val if callable(val) else (lambda M, _v=int(val): _v)
                break
    if eb is not None:
        expected_batch = eb
    return "flpop_common"


CONFIG_SOURCE = _apply_shared_config()


def config_tag(M: int) -> str:
    """Short tag encoding the physics config, for cache-file names. Distinct
    (E_b, epochs, grid) settings must never share a cache file."""
    eb = int(expected_batch(M))
    grid = "-".join(f"{a:g}" for a in ACTIONS)
    return f"Eb{eb}_ep{EPOCHS}_grid{grid}"


@dataclass
class Population:
    pop_id: int
    group_sizes: np.ndarray   # int, shape (n,)
    weights: np.ndarray       # float, shape (n,)
    expected_batch: int       # E_b = expected_batch(M)
    steps: int                # I = EPOCHS * floor(M / E_b)

    @property
    def n(self) -> int:
        return len(self.group_sizes)

    @property
    def M(self) -> int:
        return int(self.group_sizes.sum())


def sample_population(pop_id: int = 0, n: int = None,
                      w_mu: float = 0.0) -> Population:
    """Population sampler (plan A5). ``w_mu`` is the log-mean of the weight
    distribution, w_i ~ LogNormal(w_mu, 0.7); the E1-refrozen critical-scale
    setting is w_mu = ln(0.03) = -3.51. Group sizes and the rng stream do not
    depend on w_mu, so populations share sizes across weight regimes. If
    ``flpop_common.sample_population`` exists it takes precedence (same
    signature) -- checked at call time to survive late arrival of the module."""
    try:
        from game_experiments import flpop_common as fc  # type: ignore
        if callable(getattr(fc, "sample_population", None)):
            fpop = (fc.sample_population(pop_id, n=n, w_mu=w_mu)
                    if n is not None else
                    fc.sample_population(pop_id, w_mu=w_mu))
            return Population(
                pop_id=pop_id,
                group_sizes=np.asarray(fpop.group_sizes, dtype=int),
                weights=np.asarray(fpop.weights, dtype=float),
                expected_batch=int(fpop.e_b),
                steps=int(fpop.steps))
    except Exception:
        pass
    if n is None:
        n = N_PLAYERS
    rng = np.random.default_rng(pop_id)
    group_sizes = rng.integers(GROUP_MIN, GROUP_MAX + 1, size=n)
    weights = rng.lognormal(mean=float(w_mu), sigma=W_LOGNORM_SIGMA, size=n)
    M = int(group_sizes.sum())
    eb = int(expected_batch(M))
    steps = int(EPOCHS * math.floor(M / eb))
    return Population(pop_id=pop_id, group_sizes=group_sizes.astype(int),
                      weights=weights.astype(float), expected_batch=eb,
                      steps=steps)


def _evaluate_profile(key: Tuple[float, ...], group_sizes: Sequence[int],
                      eb: int, steps: int,
                      discretization: float) -> dict:
    """Weight-free level-2 evaluation of one eps profile (sigma, V, p, A).
    Utilities are attached by the Oracle at retrieval time, so cached entries
    are shared across weight regimes (w_mu sweeps reuse every solve).
    Module-level so it is picklable for ProcessPoolExecutor workers."""
    res = solve_mechanism_cached(
        eps_target=list(key), group_sizes=list(group_sizes),
        expected_batch_size=int(eb), steps=int(steps), delta=DELTA,
    )
    sigma = float(res.sigma)
    A = np.array([
        mia_advantage_subsampled_gaussian(
            sigma=sigma, sample_rate=float(p), steps=int(steps),
            discretization=float(discretization),
        )
        for p in res.sample_rates
    ])
    V = float(value_fn(sigma))
    return dict(sigma=sigma, V=V,
                p=np.asarray(res.sample_rates, dtype=float),
                A=A, feasible=bool(res.feasible))


# Worker globals set once per process via _init_worker.
_W: dict = {}


def _init_worker(group_sizes, eb, steps, discretization):
    _W["group_sizes"] = list(group_sizes)
    _W["eb"] = int(eb)
    _W["steps"] = int(steps)
    _W["disc"] = float(discretization)
    logging.getLogger("absl").setLevel(logging.ERROR)


def _worker_eval(key):
    return key, _evaluate_profile(key, _W["group_sizes"], _W["eb"],
                                  _W["steps"], _W["disc"])


class Oracle:
    """Level-2 utility oracle over a fixed population with a disk-persistent
    cache keyed by the (exact-grid) eps profile tuple."""

    def __init__(self, population: Population, cache_path: Path | None = None,
                 discretization: float = 1e-2):
        """discretization: PLD value-discretization interval. 1e-2 is the fast
        dynamics setting (max |dA| ~ 1e-2 vs 1e-3, plan failure-modes); pass
        1e-3 to re-verify final equilibria at paper precision. Distinct
        discretizations use distinct cache files."""
        self.pop = population
        self.group_sizes = list(population.group_sizes)
        self.weights = np.asarray(population.weights, dtype=float)
        self.eb = int(population.expected_batch)
        self.steps = int(population.steps)
        self.discretization = float(discretization)
        self.cache_path = cache_path
        self._cache: Dict[Tuple[float, ...], dict] = {}
        if cache_path is not None and cache_path.exists():
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)
            for v in self._cache.values():   # legacy entries stored 'u'
                v.pop("u", None)
        self.n_eval = 0
        self.n_hit = 0

    @staticmethod
    def _key(eps_vec: Sequence[float]) -> Tuple[float, ...]:
        return tuple(round(float(e), 6) for e in eps_vec)

    def _with_u(self, entry: dict) -> dict:
        out = dict(entry)
        out["u"] = entry["V"] - self.weights * entry["A"]
        return out

    def evaluate(self, eps_vec: Sequence[float]) -> dict:
        """Return {sigma, V, p (n,), A (n,), u (n,), feasible} for a profile."""
        key = self._key(eps_vec)
        cached = self._cache.get(key)
        if cached is not None:
            self.n_hit += 1
            return self._with_u(cached)
        out = _evaluate_profile(key, self.group_sizes, self.eb, self.steps,
                                self.discretization)
        self._cache[key] = out
        self.n_eval += 1
        return self._with_u(out)

    def evaluate_many(self, profiles: Sequence[Sequence[float]],
                      pool: "ProcessPoolExecutor | None" = None
                      ) -> List[dict]:
        """Evaluate many profiles, using the shared cache for hits and (if a
        pool is given) farming the unique misses to worker processes. Returns
        results aligned to ``profiles``."""
        keys = [self._key(p) for p in profiles]
        miss = {}
        for k in keys:
            if k not in self._cache and k not in miss:
                miss[k] = None
        miss_keys = list(miss)
        if miss_keys:
            if pool is None:
                for k in miss_keys:
                    self._cache[k] = _evaluate_profile(
                        k, self.group_sizes, self.eb, self.steps,
                        self.discretization)
                    self.n_eval += 1
            else:
                for k, out in pool.map(_worker_eval, miss_keys, chunksize=1):
                    self._cache[k] = out
                    self.n_eval += 1
        self.n_hit += len(keys) - len(miss_keys)
        return [self._with_u(self._cache[k]) for k in keys]

    def make_pool(self, n_workers: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(self.group_sizes, self.eb, self.steps,
                      self.discretization),
        )

    def save(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._cache, f)
            tmp.replace(self.cache_path)


def make_oracle(pop_id: int = 0, cache_path: Path | None = None,
                discretization: float = 1e-2, n: int = None,
                w_mu: float = 0.0) -> Oracle:
    return Oracle(sample_population(pop_id, n=n, w_mu=w_mu),
                  cache_path=cache_path, discretization=discretization)
