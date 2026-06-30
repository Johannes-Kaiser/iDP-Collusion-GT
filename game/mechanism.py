"""
Sampling-based iDP mechanism solver.

Per Boenisch et al. (NeurIPS 2023, Definition 4), given per-group RDP budgets
``rho_g`` and a fixed expected batch size, the mechanism picks a single noise
multiplier ``sigma`` and per-group sampling rates ``p_g`` satisfying:

    (C1)  rho_g  =  I * 2 * p_g^2 * alpha / sigma^2     (RDP at order alpha)
    (C2)  sum_g  (|G_g| / M) * p_g  =  E_b / M

We convert each target epsilon (at common delta) to its tight RDP budget via the
standard RDP -> (eps, delta) bound and then enforce (C1) and (C2) via bisection
on sigma. The final mechanism's MIA advantage is then computed by PLD accounting
of an I-fold subsampled Gaussian on the resulting (p_g, sigma) parameters.

This mirrors the actual mechanism implementation in opacus_new but without the
PyTorch dependency, so it can be re-used in the game-theoretic harness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
from dp_accounting import rdp, dp_event


# ---------------------------------------------------------------------------
# RDP accountant for one (p, sigma, I) point, accurate via dp_accounting.
# ---------------------------------------------------------------------------
_DEFAULT_ORDERS = tuple(
    list(np.arange(1.25, 8.0, 0.25)) + list(range(8, 65)) + [80, 100, 150, 250, 500]
)


@lru_cache(maxsize=400_000)
def epsilon_rdp(sigma: float, sample_rate: float, steps: int, delta: float) -> float:
    """eps(sigma, p, steps, delta) using RDP composition (tight subsampled-
    Gaussian RDP)."""
    if sample_rate <= 0.0:
        return 0.0
    acc = rdp.RdpAccountant(orders=list(_DEFAULT_ORDERS))
    event = dp_event.PoissonSampledDpEvent(
        sample_rate, dp_event.GaussianDpEvent(noise_multiplier=sigma)
    )
    acc.compose(event, steps)
    return float(acc.get_epsilon(delta))


# ---------------------------------------------------------------------------
# Per-group sample-rate solver: pick the p in [0, 1] making eps == eps_target.
# eps_rdp is monotone increasing in p, so binary search converges.
# ---------------------------------------------------------------------------

def _sample_rate_for_eps(sigma: float, eps_target: float, steps: int, delta: float,
                          tol: float = 1e-3) -> float:
    if eps_target <= 0.0:
        return 0.0
    if epsilon_rdp(sigma, 1.0, steps, delta) < eps_target:
        return 1.0
    if epsilon_rdp(sigma, 1e-9, steps, delta) > eps_target:
        return 0.0
    lo_p, hi_p = 1e-9, 1.0
    for _ in range(40):
        mid = math.sqrt(lo_p * hi_p) if lo_p > 0 else 0.5 * hi_p
        e = epsilon_rdp(sigma, mid, steps, delta)
        if e > eps_target:
            hi_p = mid
        else:
            lo_p = mid
        if hi_p / max(lo_p, 1e-12) < 1.0 + tol:
            break
    return math.sqrt(lo_p * hi_p)


@dataclass
class MechanismResult:
    sigma: float
    sample_rates: np.ndarray
    eps_per_group: np.ndarray
    steps: int
    delta: float
    eps_target: np.ndarray
    group_sizes: np.ndarray
    expected_batch_size: int
    feasible: bool


# ---------------------------------------------------------------------------
# Solve (sigma, p_g) from (eps_target_g, |G_g|, E_b, I, delta).
# ---------------------------------------------------------------------------

def solve_mechanism(
    eps_target: Sequence[float],
    group_sizes: Sequence[int],
    expected_batch_size: int,
    steps: int,
    delta: float = 1e-12,
    sigma_range: Tuple[float, float] = (0.3, 200.0),
) -> MechanismResult:
    eps_target = np.asarray(eps_target, dtype=float)
    group_sizes = np.asarray(group_sizes, dtype=float)
    target_B = float(expected_batch_size)

    def realised_batch(sigma: float) -> Tuple[float, np.ndarray]:
        rates = np.array([
            _sample_rate_for_eps(sigma, float(e), steps, delta) for e in eps_target
        ])
        B = float((group_sizes * rates).sum())
        return B, rates

    lo, hi = sigma_range
    B_lo, rates_lo = realised_batch(lo)
    B_hi, rates_hi = realised_batch(hi)
    while B_hi < target_B and hi < 1e5:
        hi *= 2
        B_hi, rates_hi = realised_batch(hi)
    while B_lo > target_B and lo > 1e-3:
        lo *= 0.5
        B_lo, rates_lo = realised_batch(lo)
    if B_hi < target_B - 1e-3:
        return MechanismResult(
            sigma=hi, sample_rates=rates_hi,
            eps_per_group=np.array([epsilon_rdp(hi, float(r), steps, delta) for r in rates_hi]),
            steps=steps, delta=delta, eps_target=eps_target,
            group_sizes=group_sizes.astype(int), expected_batch_size=int(target_B),
            feasible=False,
        )
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        B_mid, rates_mid = realised_batch(mid)
        if B_mid > target_B:
            hi, B_hi, rates_hi = mid, B_mid, rates_mid
        else:
            lo, B_lo, rates_lo = mid, B_mid, rates_mid
        if abs(B_mid - target_B) < 5e-3 * target_B:
            break
    sigma_star = math.sqrt(lo * hi)
    _, rates_star = realised_batch(sigma_star)
    eps_per_g = np.array([epsilon_rdp(sigma_star, float(r), steps, delta) for r in rates_star])
    return MechanismResult(
        sigma=sigma_star, sample_rates=rates_star, eps_per_group=eps_per_g,
        steps=steps, delta=delta, eps_target=eps_target,
        group_sizes=group_sizes.astype(int), expected_batch_size=int(target_B),
        feasible=True,
    )


# ---------------------------------------------------------------------------
# Canonicalised cache: equivalent (multi-set) profiles share the same solution.
# ---------------------------------------------------------------------------

_MECH_CACHE: dict = {}

def solve_mechanism_cached(eps_target, group_sizes, expected_batch_size, steps,
                            delta=1e-12) -> MechanismResult:
    key = (
        tuple(sorted(zip(map(float, eps_target), map(int, group_sizes)))),
        int(expected_batch_size), int(steps), float(delta),
    )
    if key not in _MECH_CACHE:
        _MECH_CACHE[key] = solve_mechanism(
            eps_target=eps_target, group_sizes=group_sizes,
            expected_batch_size=expected_batch_size, steps=steps, delta=delta,
        )
    return _MECH_CACHE[key]


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve_mechanism(
        eps_target=[8.0, 32.0],
        group_sizes=[25_000, 25_000],
        expected_batch_size=128,
        steps=int(5 * 50_000 / 128),
        delta=1e-12,
    )
    print(f"sigma={res.sigma:.4f}")
    print(f"rates={res.sample_rates}")
    print(f"eps={res.eps_per_group} (target {res.eps_target})")
    print(f"batch={int((res.sample_rates * res.group_sizes).sum())} (target {res.expected_batch_size})")
    print(f"feasible={res.feasible}, took {time.time()-t0:.2f}s")
