"""
Shared infrastructure for the realistic FL-consortium population experiments
(E1: equilibrium prediction, E2: adaptive discovery).

This module owns:
  * the *population sampler* -- how a heterogeneous 20-participant FL consortium
    is drawn (group sizes, privacy weights) from a seed;
  * the *level-2 utility oracle* -- a persistently-cached wrapper around
    ``game.game.Game`` that maps an epsilon profile to the PLD-exact,
    mechanism-specific level-2 MIA advantage and player utilities
    ``u_i = V(sigma) - w_i * A_i`` with ``V(sigma) = 1 / (1 + sigma^2)``;
  * small shared helpers (Gini, action grid, frozen defaults).

Every quantity produced here is *level-2* (analytic PLD advantage of the
subsampled Gaussian mechanism), never a trained-model measurement. Keep the
public API stable: the E2 sibling script imports ``sample_population``,
``Oracle``, ``ACTION_GRID``, ``gini`` and the frozen ``DELTA / EXPECTED_BATCH /
EPOCHS`` constants.

Frozen defaults (see plan A4/A5; do not change without re-running the scan):
  n = 20 participants
  |G_i| ~ Uniform{2000, ..., 12000}
  w_i ~ LogNormal(mu=0, sigma=0.7)
  action grid eps in {10, 14, 19, 26, 37, 50}  (log-spaced on [10, 50])
  delta = 1e-12   (non-negotiable: spreads the level-2 advantage over eps)
  E_b = 256, epochs = 10, steps I = epochs * floor(M / E_b), M = sum |G_i|
"""
from __future__ import annotations

import logging
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("absl").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.mechanism import solve_mechanism_cached  # noqa: E402
from game.advantage import mia_advantage_subsampled_gaussian  # noqa: E402
from game.utility import value_fn  # noqa: E402

# --------------------------------------------------------------------------
# Frozen defaults (A4/A5).
# --------------------------------------------------------------------------
N_PLAYERS = 20
GROUP_MIN, GROUP_MAX = 2000, 12000          # Uniform{2000,...,12000}
W_MU, W_SIGMA = 0.0, 0.7                     # LogNormal(0, 0.7)
ACTION_GRID: Tuple[float, ...] = (10.0, 14.0, 19.0, 26.0, 37.0, 50.0)
DELTA = 1e-12

# Batch-size rule. A fixed E_b = 256 at M ~ 147k (E_b/M ~ 0.17%) compresses the
# sigma range to [0.36, 0.60] and the uniform-ray V spread to ~0.14, which at
# w ~ 1 makes eps = 10 strictly dominant for everyone (the game degenerates to
# the public-goods corner; see 28_flpop_e1.md, M2). E_b must scale with M to
# keep the V-contrast alive; BATCH_FRACTION below is frozen by the A4b rescan.
BATCH_FRACTION: float | None = None   # None -> fixed EXPECTED_BATCH
EXPECTED_BATCH = 256                  # used when BATCH_FRACTION is None
EPOCHS = 10


def expected_batch_for(M: int) -> int:
    """Frozen per-population expected batch size: round(BATCH_FRACTION * M),
    or the fixed EXPECTED_BATCH when BATCH_FRACTION is None."""
    if BATCH_FRACTION is None:
        return int(EXPECTED_BATCH)
    return int(round(BATCH_FRACTION * M))

# PLD discretisation. The advantage PLD dominates oracle cost: at 1e-3 a single
# subsampled-Gaussian advantage takes ~6.5s at steps~5750, and a heterogeneous
# profile needs up to n distinct calls (~130s). At 1e-2 it is ~200x faster and
# the advantage shifts by <=~1.2e-2 (T17/R2). We run best-response dynamics and
# scans at DISC_FAST and verify final equilibria at DISC_EXACT.
DISC_FAST = 1e-2
DISC_EXACT = 1e-3


def _sig6(x: float) -> float:
    """Round to 6 significant digits so advantage-cache lookups hit."""
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(round(x, 6 - int(math.floor(math.log10(abs(x)))) - 1))

CACHE_DIR = ROOT / "game_experiments" / "_cache_flpop"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Population sampler (A5).
# --------------------------------------------------------------------------
@dataclass
class Population:
    """A sampled FL consortium: fixed group sizes and privacy weights, plus
    the mechanism scale (expected batch, epochs) resolved at sampling time."""
    pid: int
    group_sizes: np.ndarray   # (n,) int
    weights: np.ndarray       # (n,) float, w_i > 0
    e_b: int = 0              # expected batch size (resolved in sample_population)
    epochs: int = EPOCHS

    def __post_init__(self):
        if self.e_b <= 0:
            self.e_b = expected_batch_for(self.M)

    @property
    def n(self) -> int:
        return len(self.group_sizes)

    @property
    def M(self) -> int:
        return int(self.group_sizes.sum())

    @property
    def steps(self) -> int:
        """I = epochs * floor(M / E_b)."""
        return int(self.epochs * (self.M // self.e_b))

    @property
    def mech_tag(self) -> str:
        """Cache-file tag identifying the mechanism scale."""
        return f"EB{self.e_b}_ep{self.epochs}"


# Adult-sized appendix populations (D5): smaller silos.
GROUP_MIN_ADULT, GROUP_MAX_ADULT = 1000, 5000


def sample_population(pid: int, n: int = N_PLAYERS,
                     group_range: Tuple[int, int] = (GROUP_MIN, GROUP_MAX),
                     w_mu: float = W_MU) -> Population:
    """Population ``pid`` uses seed ``pid`` (reproducible across scripts).

    |G_i| ~ Uniform{group_range[0], ..., group_range[1]},
    w_i ~ LogNormal(w_mu, W_SIGMA). The default range is the main-experiment
    consortium; pass ``(1000, 5000)`` for the Adult-sized appendix (D5).
    ``w_mu`` shifts the weight distribution on the log scale (median weight =
    exp(w_mu)); group sizes are unaffected because they are drawn before the
    weights from the same seeded generator.
    """
    rng = np.random.default_rng(pid)
    lo, hi = group_range
    group_sizes = rng.integers(lo, hi + 1, size=n)
    weights = rng.lognormal(mean=w_mu, sigma=W_SIGMA, size=n)
    return Population(pid=pid, group_sizes=group_sizes, weights=weights.astype(float))


# --------------------------------------------------------------------------
# Level-2 utility oracle (A3) with a persistent per-population cache.
# --------------------------------------------------------------------------
@dataclass
class OracleResult:
    eps_vec: Tuple[float, ...]
    sigma: float
    sample_rates: np.ndarray   # p_i
    advantages: np.ndarray     # A_i (level-2)
    utilities: np.ndarray      # u_i = V(sigma) - w_i A_i
    feasible: bool

    @property
    def V(self) -> float:
        return value_fn(self.sigma)


def _round_eps(e: float) -> float:
    # Actions are drawn from a fixed grid; round to kill float noise so cache
    # keys are exact.
    return round(float(e), 6)


class Oracle:
    """Level-2 oracle for one fixed population.

    ``evaluate(eps_vec)`` returns an :class:`OracleResult`. Solutions are cached
    two ways: (i) in-memory + module-global mechanism/advantage caches (via
    ``game.game.Game``), and (ii) persistently to ``cache_path`` keyed by the
    *multiset* of ``(round(eps_i), |G_i|)`` pairs -- so permutations of a profile
    and repeated best-response evaluations share work, and a requeued Slurm task
    resumes from disk. The cache stores ``sigma`` and, per distinct
    ``(eps, size)`` pair, ``(p, A)``; utilities are recomputed from ``w_i`` on
    read (weights are not in the key).

    IMPORTANT (concurrency): each population gets its own cache file. Do NOT
    point two concurrent array tasks at the same ``cache_path``.
    """

    def __init__(self, population: Population,
                 actions: Sequence[float] = ACTION_GRID,
                 cache_path: Path | None = None,
                 persist: bool = True,
                 discretization: float = DISC_FAST):
        self.pop = population
        self.actions = tuple(float(a) for a in actions)
        self.persist = persist
        self.discretization = float(discretization)
        self.steps = population.steps
        self.group_sizes = [int(g) for g in population.group_sizes]
        tag = "e" if abs(self.discretization - DISC_EXACT) < 1e-9 else "f"
        # Non-default mechanism scales get their own cache namespace so results
        # at different (E_b, epochs) can never mix.
        mech = ("" if (population.e_b == EXPECTED_BATCH
                       and population.epochs == EPOCHS)
                else f"_{population.mech_tag}")
        self.cache_path = (
            cache_path if cache_path is not None
            else CACHE_DIR / f"oracle_pop{population.pid:03d}_{tag}{mech}.pkl"
        )
        self._disk: Dict = {}
        if self.persist and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._disk = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                self._disk = {}
        self._dirty = 0

    # -- persistent multiset cache -------------------------------------
    def _key(self, eps_vec: Sequence[float]) -> Tuple:
        pairs = sorted(
            (_round_eps(e), int(g))
            for e, g in zip(eps_vec, self.pop.group_sizes)
        )
        return tuple(pairs)

    def _from_disk(self, eps_vec: Sequence[float]) -> OracleResult | None:
        rec = self._disk.get(self._key(eps_vec))
        if rec is None:
            return None
        sigma = rec["sigma"]
        pa = rec["pa"]  # dict (eps_rounded, size) -> (p, A)
        n = self.pop.n
        p = np.empty(n)
        A = np.empty(n)
        for i in range(n):
            k = (_round_eps(eps_vec[i]), int(self.pop.group_sizes[i]))
            p[i], A[i] = pa[k]
        u = value_fn(sigma) - self.pop.weights * A
        return OracleResult(
            eps_vec=tuple(float(e) for e in eps_vec), sigma=float(sigma),
            sample_rates=p, advantages=A, utilities=u, feasible=rec["feasible"],
        )

    def _to_disk(self, res: OracleResult) -> None:
        pa: Dict = {}
        for i in range(self.pop.n):
            k = (_round_eps(res.eps_vec[i]), int(self.pop.group_sizes[i]))
            pa[k] = (float(res.sample_rates[i]), float(res.advantages[i]))
        self._disk[self._key(res.eps_vec)] = dict(
            sigma=float(res.sigma), pa=pa, feasible=bool(res.feasible),
        )
        self._dirty += 1
        if self.persist and self._dirty >= 32:
            self.flush()

    def flush(self) -> None:
        if not self.persist:
            return
        tmp = self.cache_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._disk, f)
        tmp.replace(self.cache_path)
        self._dirty = 0

    # -- main entry point ----------------------------------------------
    def evaluate(self, eps_vec: Sequence[float]) -> OracleResult:
        hit = self._from_disk(eps_vec)
        if hit is not None:
            return hit
        mech = solve_mechanism_cached(
            eps_target=[float(e) for e in eps_vec],
            group_sizes=self.group_sizes,
            expected_batch_size=self.pop.e_b,
            steps=self.steps, delta=DELTA,
        )
        sigma = _sig6(float(mech.sigma))
        advs = np.array([
            mia_advantage_subsampled_gaussian(
                sigma=sigma, sample_rate=_sig6(float(p)),
                steps=self.steps, discretization=self.discretization,
            )
            for p in mech.sample_rates
        ])
        u = value_fn(sigma) - self.pop.weights * advs
        res = OracleResult(
            eps_vec=tuple(float(e) for e in eps_vec), sigma=float(sigma),
            sample_rates=np.asarray(mech.sample_rates, dtype=float),
            advantages=advs, utilities=np.asarray(u, dtype=float),
            feasible=bool(mech.feasible),
        )
        self._to_disk(res)
        return res

    def best_response(self, i: int, eps_vec: Sequence[float]) -> Tuple[float, float]:
        """Best grid action for player ``i`` against the rest of ``eps_vec``
        (ties break toward the current value ``eps_vec[i]``). Infeasible
        candidates are skipped. Returns (best_eps, best_utility)."""
        cur = float(eps_vec[i])
        best_a, best_u = None, -np.inf
        for a in self.actions:
            trial = list(eps_vec)
            trial[i] = float(a)
            res = self.evaluate(trial)
            if not res.feasible:
                continue
            ui = float(res.utilities[i])
            # tie-break toward current value
            if ui > best_u + 1e-12 or (abs(ui - best_u) <= 1e-12 and a == cur):
                best_u, best_a = ui, float(a)
        if best_a is None:
            best_a = cur  # no feasible action; hold
        return best_a, best_u


# --------------------------------------------------------------------------
# Stateless profile solver (top-level so it is picklable for process pools).
# Returns everything needed to reconstruct utilities for any weight vector.
# --------------------------------------------------------------------------
def solve_profile_raw(eps_vec: Sequence[float], group_sizes: Sequence[int],
                      steps: int, discretization: float = DISC_FAST,
                      expected_batch: int | None = None) -> Dict:
    """Solve one epsilon profile: returns dict(sigma, rates, advs, feasible).
    Label-invariant caching is handled inside ``solve_mechanism_cached`` and the
    advantage lru-cache; ``sigma`` and ``p`` are rounded to 6 sig-digits so
    cache lookups hit."""
    e_b = EXPECTED_BATCH if expected_batch is None else int(expected_batch)
    mech = solve_mechanism_cached(
        eps_target=[float(e) for e in eps_vec],
        group_sizes=[int(g) for g in group_sizes],
        expected_batch_size=e_b, steps=int(steps), delta=DELTA,
    )
    sigma = _sig6(float(mech.sigma))
    advs = np.array([
        mia_advantage_subsampled_gaussian(
            sigma=sigma, sample_rate=_sig6(float(p)),
            steps=int(steps), discretization=float(discretization),
        )
        for p in mech.sample_rates
    ])
    return dict(sigma=float(sigma),
                rates=np.asarray(mech.sample_rates, dtype=float),
                advs=advs, feasible=bool(mech.feasible))


def _worker_solve(task):
    """Picklable pool worker:
    task = (eps_tuple, gsizes_tuple, steps, disc[, expected_batch]).
    Returns (eps_tuple, result_dict)."""
    eps_t, gsizes, steps, disc = task[:4]
    e_b = task[4] if len(task) > 4 else None
    return eps_t, solve_profile_raw(eps_t, gsizes, steps, disc, expected_batch=e_b)


# --------------------------------------------------------------------------
# Shared helpers.
# --------------------------------------------------------------------------
def gini(x: Sequence[float]) -> float:
    """Standard Gini coefficient on non-negative values. 0 for an empty or
    all-zero vector."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    s = a.sum()
    if n == 0 or s <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s)) - (n + 1) / n)


def lorenz_curve(x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (population fraction, cumulative-share) points for a Lorenz curve,
    including the (0,0) origin."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0 or a.sum() <= 0:
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    cum = np.concatenate([[0.0], np.cumsum(a) / a.sum()])
    frac = np.linspace(0.0, 1.0, n + 1)
    return frac, cum


def grid_midpoint(actions: Sequence[float] = ACTION_GRID) -> float:
    """Third-of-six grid value used to initialise best-response dynamics."""
    a = list(actions)
    return float(a[len(a) // 2 - 1]) if len(a) == 6 else float(a[len(a) // 2])


if __name__ == "__main__":
    import time
    pop = sample_population(0)
    print(f"pop0: n={pop.n} M={pop.M} steps={pop.steps}")
    print(f"  sizes={pop.group_sizes.tolist()}")
    print(f"  weights={np.round(pop.weights, 3).tolist()}")
    orc = Oracle(pop, persist=False)
    for e in ACTION_GRID:
        t0 = time.time()
        res = orc.evaluate([e] * pop.n)
        print(f"  uniform eps={e:>4}: sigma={res.sigma:.3f} feasible={res.feasible} "
              f"medA={np.median(res.advantages):.4f} V={res.V:.4f} "
              f"({time.time()-t0:.2f}s)")
