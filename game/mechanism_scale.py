"""
Sensitivity-based iDP ("Scale") mechanism solver.

Per Boenisch et al. (NeurIPS 2023) the Scale realisation of iDP contracts a
per-record noise-to-sensitivity ratio. Concretely, the coordinator picks a
shared Gaussian noise multiplier

    sigma_scale := ( N^{-1} sum_g |G_g| / sigma_g )^{-1}     (weighted harmonic mean)

together with per-group clipping norms

    c_g := c_baseline * (sigma_scale / sigma_g)

so that group g's effective per-record noise-to-sensitivity ratio
sigma_scale / c_g equals sigma_g / c_baseline (the ratio that would deliver
the target (eps_g, delta)-DP guarantee under Poisson sampling at rate
E_b / M over I steps). Under this definition the coupling constraint

    sum_g |G_g| * c_g = N * c_baseline

is an analytic identity: it reduces to sigma_scale * sum_g |G_g|/sigma_g = N,
which is the definition of sigma_scale as the weighted harmonic mean of the
per-group sigmas.

The MIA advantage of group i under Scale is the advantage of the sampling
Gaussian at effective noise sigma_i = c_baseline * sigma_scale / c_i and
sampling rate E_b / M, computed via PLD accounting exactly as for the
sampling mechanism in ``game/mechanism.py``. By construction this collapses
to mia_advantage_subsampled_gaussian(sigma_g, p, I): each group's realised
per-record privacy is exactly what its own eps_g contracts, independent of
the other groups' eps. The residual PEG structure runs through the shared
sigma_scale in V(sigma_scale) only.

This is the second instantiation used to stress-test the paper's
abstract mechanism condition (T18 / H2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .mechanism import epsilon_rdp


# ---------------------------------------------------------------------------
# Per-group sigma solver: find sigma_g so eps_rdp(sigma_g, p_fixed, I, delta) == eps_g
# eps_rdp is monotone decreasing in sigma at fixed p, so bisection converges.
# ---------------------------------------------------------------------------

def _sigma_for_eps(eps_target: float, sample_rate: float, steps: int,
                   delta: float, sigma_range: Tuple[float, float] = (0.1, 500.0),
                   tol: float = 1e-3) -> float:
    if eps_target <= 0.0:
        return sigma_range[1]
    lo, hi = sigma_range
    e_lo = epsilon_rdp(lo, sample_rate, steps, delta)
    e_hi = epsilon_rdp(hi, sample_rate, steps, delta)
    # If even the highest sigma leaves eps above target, sigma is at ceiling
    if e_hi > eps_target:
        return hi
    # If even the smallest sigma gives eps below target, sigma is at floor
    if e_lo < eps_target:
        return lo
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        e_mid = epsilon_rdp(mid, sample_rate, steps, delta)
        if e_mid > eps_target:
            lo = mid
        else:
            hi = mid
        if hi / max(lo, 1e-12) < 1.0 + tol:
            break
    return math.sqrt(lo * hi)


@dataclass
class MechanismScaleResult:
    sigma_scale: float
    sigma_per_group: np.ndarray
    clip_per_group: np.ndarray
    eps_per_group: np.ndarray
    steps: int
    delta: float
    eps_target: np.ndarray
    group_sizes: np.ndarray
    expected_batch_size: int
    sample_rate: float
    c_baseline: float
    feasible: bool


def solve_mechanism_scale(
    eps_target: Sequence[float],
    group_sizes: Sequence[int],
    expected_batch_size: int,
    steps: int,
    delta: float = 1e-12,
    c_baseline: float = 1.0,
    sigma_range: Tuple[float, float] = (0.1, 500.0),
    min_c: float = 1e-5,
    max_c_ratio: float = 1e3,
) -> MechanismScaleResult:
    """Solve the Scale mechanism at fixed sampling rate p = E_b / M.

    Returns per-group Gaussian noise sigma_g meeting each (eps_g, delta)-DP
    guarantee at that rate, the shared harmonic-mean sigma_scale, and the
    per-group clipping norms c_g. Feasibility flags cells at which any c_g
    falls below ``min_c`` or the c_g / c_baseline ratio exceeds ``max_c_ratio``
    (numerical stability of the PLD accountant at extreme effective sigmas).
    """
    eps_target = np.asarray(eps_target, dtype=float)
    group_sizes = np.asarray(group_sizes, dtype=float)
    N = float(group_sizes.sum())
    sample_rate = float(expected_batch_size) / N

    sigma_g = np.array([
        _sigma_for_eps(float(e), sample_rate, steps, delta, sigma_range=sigma_range)
        for e in eps_target
    ])

    # Weighted harmonic mean of per-group sigmas across records
    inv_sigma = group_sizes / sigma_g
    sigma_scale = float(N / inv_sigma.sum())

    # Per-group clipping norms so that sigma_scale / c_g == sigma_g / c_baseline
    c_g = c_baseline * (sigma_scale / sigma_g)

    # Sanity: coupling constraint (analytic identity)
    coupling_lhs = float((group_sizes * c_g).sum())
    coupling_rhs = float(N * c_baseline)
    coupling_ok = abs(coupling_lhs - coupling_rhs) < 1e-6 * coupling_rhs

    # Realised eps per group under (sigma_g, sample_rate, steps)
    eps_per_group = np.array([
        epsilon_rdp(float(s), sample_rate, steps, delta) for s in sigma_g
    ])

    # Feasibility: no degenerate clipping norms, no extreme ratios
    feasible = (
        coupling_ok
        and bool(np.all(c_g >= min_c))
        and bool(np.max(c_g / c_baseline) <= max_c_ratio)
        and bool(np.max(c_baseline / np.maximum(c_g, 1e-30)) <= max_c_ratio)
    )

    return MechanismScaleResult(
        sigma_scale=sigma_scale,
        sigma_per_group=sigma_g,
        clip_per_group=c_g,
        eps_per_group=eps_per_group,
        steps=int(steps),
        delta=float(delta),
        eps_target=eps_target,
        group_sizes=group_sizes.astype(int),
        expected_batch_size=int(expected_batch_size),
        sample_rate=sample_rate,
        c_baseline=float(c_baseline),
        feasible=bool(feasible),
    )


# ---------------------------------------------------------------------------
# Canonicalised cache: equivalent (multi-set) profiles share the same solution.
# ---------------------------------------------------------------------------

_MECH_CACHE_SCALE: dict = {}


def solve_mechanism_scale_cached(eps_target, group_sizes, expected_batch_size,
                                  steps, delta=1e-12, c_baseline=1.0
                                  ) -> MechanismScaleResult:
    n = len(eps_target)
    eps_t = [float(e) for e in eps_target]
    gs = [int(g) for g in group_sizes]
    pairs = list(zip(eps_t, gs))
    sorted_pairs_idx = sorted(range(n), key=lambda i: pairs[i])
    sorted_pairs = [pairs[i] for i in sorted_pairs_idx]
    key = (tuple(sorted_pairs), int(expected_batch_size), int(steps),
           float(delta), float(c_baseline))
    if key not in _MECH_CACHE_SCALE:
        sorted_eps = [p[0] for p in sorted_pairs]
        sorted_gs = [p[1] for p in sorted_pairs]
        _MECH_CACHE_SCALE[key] = solve_mechanism_scale(
            eps_target=sorted_eps, group_sizes=sorted_gs,
            expected_batch_size=expected_batch_size, steps=steps,
            delta=delta, c_baseline=c_baseline,
        )
    cached = _MECH_CACHE_SCALE[key]
    inv = [0] * n
    for sort_pos, orig_idx in enumerate(sorted_pairs_idx):
        inv[orig_idx] = sort_pos
    return MechanismScaleResult(
        sigma_scale=cached.sigma_scale,
        sigma_per_group=np.array([cached.sigma_per_group[inv[i]] for i in range(n)]),
        clip_per_group=np.array([cached.clip_per_group[inv[i]] for i in range(n)]),
        eps_per_group=np.array([cached.eps_per_group[inv[i]] for i in range(n)]),
        steps=cached.steps, delta=cached.delta,
        eps_target=np.array(eps_t),
        group_sizes=np.array(gs),
        expected_batch_size=int(expected_batch_size),
        sample_rate=cached.sample_rate,
        c_baseline=cached.c_baseline,
        feasible=cached.feasible,
    )


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve_mechanism_scale(
        eps_target=[8.0, 32.0],
        group_sizes=[25_000, 25_000],
        expected_batch_size=128,
        steps=int(5 * 50_000 / 128),
        delta=1e-12,
    )
    print(f"sigma_scale = {res.sigma_scale:.4f}")
    print(f"sigma_g     = {res.sigma_per_group}")
    print(f"c_g         = {res.clip_per_group}")
    print(f"eps_g (realised) = {res.eps_per_group} (target {res.eps_target})")
    print(f"coupling: sum |G_g| c_g = {(res.group_sizes*res.clip_per_group).sum():.4f}"
          f"  target = {res.group_sizes.sum()*res.c_baseline:.4f}")
    print(f"feasible = {res.feasible},  took {time.time()-t0:.2f}s")
