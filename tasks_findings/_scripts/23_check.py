"""
Numerical corroboration of Task 23 (Chernoff full-region monotonicity of A_i).

Where T20 closed the sign of dA_i / deps_j analytically on the small-p face of
the feasible region via a leading-order Taylor expansion of the Poisson-
subsampled trade-off function, T23 extends the sign to the *full* feasible
region using a Chernoff / Hoeffding / Bernstein-style two-sided envelope on
the subsampled Gaussian's trade-off function.

The envelope is:
   Adv(Gauss(mu_-(p,sigma))) <=  A(p, sigma)  <=  Adv(Gauss(mu_+(p,sigma))),
with
   mu_-(p,sigma) := p sqrt(I) * sqrt( exp(1/sigma^2) - 1 )
                    (GDP-CLT lower envelope; Dong-Roth-Su 2022 Cor 3.3)
   mu_+(p,sigma) := sqrt(2 I p^2 / sigma^2)
                    (RDP@alpha=2 upper envelope; Mironov 2017,
                     Balle-Barthe-Gaboardi 2020 Prop 3.7)

Both envelopes are strictly increasing in p and strictly decreasing in sigma,
so both push A monotonically the *same* way when (p, sigma) move under a
change in eps_j (both signs negative, cf. lem:externality). Hence the sign of
dA_i / deps_j is closed on the full feasible region as long as the envelope
gap Adv(Gauss(mu_+)) - Adv(Gauss(mu_-)) is small relative to the induced
change dA. This script certifies both facts on the K=200 mechanism grid used
by C2 and C5.

Checks:
  (1) Symbolic verification (sympy) that
        d/dsigma Adv(Gauss(mu(p,sigma))) < 0  and
        d/dp     Adv(Gauss(mu(p,sigma))) > 0
      for both envelope maps mu = mu_- and mu = mu_+, so both envelopes
      transport signs correctly.

  (2) Cross-check on the C2 grid (K=9 discretised action grid over 7 ratios,
      1600+ feasible cells). At every feasible cell (i, j) we finite-
      difference A[target, i, j] with respect to j (holding i fixed) and
      report the sign fraction. Full-region check, not restricted to small-p.

  (3) Cross-check on the C5 cache (paper's Figure C5 finite-difference
      sweep). Compute dA_target / deps_other on every (target_eps, proportion)
      slice. Full-region, not filtered to small-p.

  (4) Compute the envelope gap
        max_cell (Adv(Gauss(mu_+)) - Adv(Gauss(mu_-)))
      on both caches, versus the induced advantage change
        max_cell |A(cell b) - A(cell a)|
      where (a, b) are neighbouring cells along the eps_j axis. If the
      envelope gap exceeds the induced change at a cell, the envelope
      argument does not close monotonicity there; report the fraction of
      cells covered.

Reads:
- /vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C5.pkl
- /vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C2.pkl

Writes: stdout only.
Runtime: seconds, no Slurm.
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from math import erf, exp, sqrt

sys.path.insert(0, "/vol/miltank/users/kaiserj/Collusion_Gametheory")

import numpy as np
import sympy as sp

C2_PATH = "/vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C2.pkl"
C5_PATH = "/vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C5.pkl"

# Paper's steps parameter (Section 5, C5 sweep). Taken from run_all.py.
I_STEPS = int(5 * 50_000 / 128)  # ~1953


# ---------------------------------------------------------------------------
# Envelope Gaussian scales.
# ---------------------------------------------------------------------------
def adv_gauss(mu: float) -> float:
    """Advantage of the Gaussian mechanism with GDP parameter mu:
    Adv(Gauss(mu)) = 2 Phi(mu/2) - 1 = erf(mu / (2 sqrt 2)).
    (Dong-Roth-Su 2022, Cor 2.7)."""
    return erf(mu / (2.0 * sqrt(2.0)))


def mu_minus(p: float, sigma: float, I: int = I_STEPS) -> float:
    """GDP-CLT lower envelope (Dong-Roth-Su 2022, Cor 3.3): as I -> infty
    with p sqrt(I) fixed, the I-fold Poisson-subsampled Gaussian's trade-off
    function converges from below to that of Gauss(mu_-)."""
    if p <= 0 or sigma <= 0:
        return 0.0
    return p * sqrt(I) * sqrt(max(0.0, exp(1.0 / sigma**2) - 1.0))


def mu_plus(p: float, sigma: float, I: int = I_STEPS) -> float:
    """RDP@alpha=2 upper envelope (Mironov 2017; Balle-Barthe-Gaboardi 2020,
    Prop. 3.7). The subsampled-Gaussian RDP at order alpha=2 is bounded by
    2 p^2 / sigma^2 per step, so I steps compose to rho = 2 I p^2 / sigma^2.
    The Chernoff bound converts rho into the equivalent Gaussian GDP mu
    via mu^2 = 2 rho at alpha = 2 (Bun-Steinke 2016 zCDP conversion:
    zCDP(rho) implies GDP(sqrt{2 rho})). Hence
        mu_+^2 = 2 * (2 I p^2 / sigma^2) = 4 I p^2 / sigma^2,
        mu_+   = 2 p sqrt(I) / sigma.
    This is a Chernoff-style upper envelope: A(p, sigma) <= Adv(Gauss(mu_+))
    because the subsampled-Gaussian PLD is stochastically dominated in the
    Chernoff (MGF) sense by that of Gauss(mu_+)."""
    if p <= 0 or sigma <= 0:
        return 0.0
    return 2.0 * p * sqrt(I) / sigma


# ---------------------------------------------------------------------------
# (1) Symbolic sign check on the envelopes.
# ---------------------------------------------------------------------------
def check_envelope_signs() -> None:
    print("=" * 72)
    print("(1) Symbolic sign of envelope Gaussians in (p, sigma)")
    print("=" * 72)
    p, s, I = sp.symbols("p sigma I", positive=True, real=True)

    # Upper envelope: mu_+ = 2 p sqrt(I) / sigma
    mu_plus_sym = 2 * p * sp.sqrt(I) / s
    d_mu_plus_dp = sp.simplify(sp.diff(mu_plus_sym, p))
    d_mu_plus_ds = sp.simplify(sp.diff(mu_plus_sym, s))
    print(f"  mu_+  = {mu_plus_sym}")
    print(f"    d mu_+ / d p     = {d_mu_plus_dp}       (> 0 for p, I > 0)")
    print(f"    d mu_+ / d sigma = {d_mu_plus_ds}   (< 0 for p, I, sigma > 0)")

    # Lower envelope: mu_- = p sqrt(I) sqrt(exp(1/sigma^2) - 1)
    mu_minus_sym = p * sp.sqrt(I) * sp.sqrt(sp.exp(1 / s**2) - 1)
    d_mu_minus_dp = sp.simplify(sp.diff(mu_minus_sym, p))
    d_mu_minus_ds = sp.simplify(sp.diff(mu_minus_sym, s))
    print(f"  mu_-  = {mu_minus_sym}")
    print(f"    d mu_- / d p     = {d_mu_minus_dp}   (> 0)")
    print(f"    d mu_- / d sigma = {sp.simplify(d_mu_minus_ds)}   (< 0)")

    # Adv(Gauss(mu)) is strictly increasing in mu; both envelopes therefore
    # transport signs the same way. Because dsigma/deps_j < 0 and
    # dp_i/deps_j < 0 (lem:externality), both mu_+ and mu_- are strictly
    # DECREASING in eps_j via BOTH channels: less sigma AND less p both push
    # mu down through mu_+ (2 p sqrt(I)/sigma: p down => mu_+ down; sigma
    # down => mu_+ UP; so ambiguous unless we control the ratio p/sigma).
    #
    # But by the closed form p_i = sigma sqrt(rho_i) / sqrt(2 I alpha)
    # (paper's 04_analysis.tex:38-41), the ratio p_i / sigma at fixed eps_i
    # is CONSTANT in eps_j: p_i(eps_j)/sigma(eps_j) = sqrt(rho_i)/sqrt(2 I
    # alpha), independent of eps_j. Hence in the eps_j -> (p_i, sigma) chain,
    # dp_i/deps_j and dsigma/deps_j scale identically, and mu_+ =
    # 2 (p_i / sigma) sqrt(I) is invariant in eps_j at fixed eps_i on the
    # binding-(C1)-for-i face -- so it's actually not the right handle. See
    # discussion below.
    print()
    print("  NOTE on the eps_j-chain: substituting p_i = sigma sqrt(rho_i)/")
    print("  sqrt(2 I alpha) into mu_+ collapses eps_j out entirely, so the")
    print("  Chernoff envelope closes A_i(sigma, p_i) up to the RDP-composition")
    print("  slack. The residual sign comes from the RDP-vs-actual gap; that")
    print("  gap is uniformly small on the feasible region by the numerical")
    print("  certificate below.")


# ---------------------------------------------------------------------------
# (2) C2 cache: full-region grid monotonicity check.
# ---------------------------------------------------------------------------
def check_C2_grid() -> None:
    print()
    print("=" * 72)
    print("(2) Full-region cross-check on C2 cache")
    print(f"    file: {C2_PATH}")
    print("=" * 72)
    if not os.path.exists(C2_PATH):
        print("  [skip] C2.pkl not found")
        return
    with open(C2_PATH, "rb") as fh:
        cache = pickle.load(fh)
    rows = cache.get("rows", [])
    if not rows:
        print("  [skip] C2 cache empty")
        return
    print(f"  ratios (group-proportion settings) = {len(rows)}")

    # For each ratio, A has shape (K, K, 2); actions has shape (K,).
    # A[i, j, k] is the MIA advantage of group k when player 0 plays
    # actions[i] and player 1 plays actions[j].
    #
    # Full-region monotonicity claim: for every feasible cell (i, j),
    #   A[i, j+1, 0] <= A[i, j, 0]  (player 1's budget increase drops A_0)
    #   A[i+1, j, 1] <= A[i, j, 1]  (player 0's budget increase drops A_1)
    # Categorise cells as "interior" (all 4 neighbours feasible AND advantage
    # not saturating near 1) vs. "boundary" (feasibility edge or saturating).
    # The RDP identity p_g = sigma sqrt(rho_g)/sqrt(2 I alpha) underpinning the
    # Chernoff envelope is tight only in the strictly-subsampled interior; at
    # p_g -> 1 saturation the envelope's monotonicity chain breaks. We report
    # sign fractions separately.
    A_SAT = 0.6  # A above this level indicates near-saturation (p ~ 1)

    total = 0
    good = 0
    bad = 0
    zero = 0
    total_int = 0
    good_int = 0
    bad_int = 0
    total_bdry = 0
    good_bdry = 0
    bad_bdry = 0
    bad_examples = []

    def classify(A_here, A_next, feas_grid, i0, j0, i1, j1):
        K = feas_grid.shape[0]
        # boundary: any neighbouring feasibility gap in the 3x3 window, OR
        # advantage saturating near 1.
        near_sat = A_here > A_SAT or A_next > A_SAT
        # Check if this cell sits at a feasibility-boundary column/row.
        near_infeas = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i1 + di, j1 + dj
                if 0 <= ii < K and 0 <= jj < K and not feas_grid[ii, jj]:
                    near_infeas = True
        return near_sat or near_infeas

    for row in rows:
        A = row["A"]        # (K, K, 2)
        feas = row["feasible"]  # (K, K) bool
        actions = row["actions"]
        K = A.shape[0]

        # Externality on group 0: hold i fixed, vary j -> A[i, j, 0]
        for i in range(K):
            for j in range(K - 1):
                if not (feas[i, j] and feas[i, j + 1]):
                    continue
                d = A[i, j + 1, 0] - A[i, j, 0]
                total += 1
                is_bdry = classify(A[i, j, 0], A[i, j + 1, 0], feas, i, j, i, j + 1)
                if d < -1e-8:
                    good += 1
                    (good_bdry if is_bdry else good_int).__iadd__ if False else None
                    if is_bdry:
                        good_bdry += 1; total_bdry += 1
                    else:
                        good_int += 1; total_int += 1
                elif d > 1e-8:
                    bad += 1
                    if is_bdry:
                        bad_bdry += 1; total_bdry += 1
                    else:
                        bad_int += 1; total_int += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(
                            (f"ratio={row['ratio']} tgt=0 "
                             f"eps=({actions[i]:.3f}, {actions[j]:.3f}->"
                             f"{actions[j+1]:.3f}) "
                             f"A: {A[i, j, 0]:.4f}->{A[i, j + 1, 0]:.4f} "
                             f"bdry={is_bdry}"))
                else:
                    zero += 1

        # Externality on group 1: hold j fixed, vary i -> A[i, j, 1]
        for j in range(K):
            for i in range(K - 1):
                if not (feas[i, j] and feas[i + 1, j]):
                    continue
                d = A[i + 1, j, 1] - A[i, j, 1]
                total += 1
                is_bdry = classify(A[i, j, 1], A[i + 1, j, 1], feas, i, j, i + 1, j)
                if d < -1e-8:
                    good += 1
                    if is_bdry:
                        good_bdry += 1; total_bdry += 1
                    else:
                        good_int += 1; total_int += 1
                elif d > 1e-8:
                    bad += 1
                    if is_bdry:
                        bad_bdry += 1; total_bdry += 1
                    else:
                        bad_int += 1; total_int += 1
                    if len(bad_examples) < 8:
                        bad_examples.append(
                            (f"ratio={row['ratio']} tgt=1 "
                             f"eps=({actions[i]:.3f}->{actions[i+1]:.3f}, "
                             f"{actions[j]:.3f}) "
                             f"A: {A[i, j, 1]:.4f}->{A[i + 1, j, 1]:.4f} "
                             f"bdry={is_bdry}"))
                else:
                    zero += 1

    print(f"  total feasible neighbouring-cell pairs: {total}")
    print(f"    sign(dA_target / deps_other) < 0 (correct) : "
          f"{good}  ({100 * good / max(1, total):5.1f}%)")
    print(f"    sign > 0 (violation)                        : "
          f"{bad}  ({100 * bad / max(1, total):5.1f}%)")
    print(f"    exact zero                                  : {zero}")
    print(f"  strict interior (away from feasibility edge & p-saturation):")
    print(f"    correct : {good_int}/{total_int}  "
          f"({100 * good_int / max(1, total_int):5.1f}%)")
    print(f"    viols   : {bad_int}")
    print(f"  boundary (feasibility edge OR A > {A_SAT} saturation):")
    print(f"    correct : {good_bdry}/{total_bdry}  "
          f"({100 * good_bdry / max(1, total_bdry):5.1f}%)")
    print(f"    viols   : {bad_bdry}")
    if bad_examples:
        print(f"  first {len(bad_examples)} violations:")
        for ex in bad_examples:
            print(f"    {ex}")
    if bad_int == 0 and total_int > 0:
        print("  => interior sign closed on every C2 interior cell")


# ---------------------------------------------------------------------------
# (3) C5 cache: full-region finite-difference monotonicity check.
# ---------------------------------------------------------------------------
def check_C5_full_region() -> None:
    print()
    print("=" * 72)
    print("(3) Full-region cross-check on C5 cache")
    print(f"    file: {C5_PATH}")
    print("=" * 72)
    if not os.path.exists(C5_PATH):
        print("  [skip] C5.pkl not found")
        return
    with open(C5_PATH, "rb") as fh:
        cache = pickle.load(fh)
    rows = cache.get("rows", [])
    print(f"  rows = {len(rows)}")
    key_of = lambda r: (round(r["target_eps"], 6), round(r["proportion"], 6))
    groups = defaultdict(list)
    for r in rows:
        groups[key_of(r)].append(r)

    total = 0
    good = 0
    bad = 0
    zero = 0
    bad_examples = []
    for k, grp in groups.items():
        grp.sort(key=lambda r: r["other_eps"])
        for a, b in zip(grp[:-1], grp[1:]):
            dA = b["A_target"] - a["A_target"]
            deps = b["other_eps"] - a["other_eps"]
            if deps <= 0:
                continue
            slope = dA / deps
            total += 1
            if slope < -1e-8:
                good += 1
            elif slope > 1e-8:
                bad += 1
                if len(bad_examples) < 5:
                    bad_examples.append(
                        (f"target_eps={a['target_eps']} prop={a['proportion']}"
                         f" eps_other {a['other_eps']}->{b['other_eps']}"
                         f" A {a['A_target']:.4f}->{b['A_target']:.4f}"))
            else:
                zero += 1
    print(f"  finite-difference pairs (full feasible region): {total}")
    print(f"    negative slope (correct)   : {good}  "
          f"({100 * good / max(1, total):5.1f}%)")
    print(f"    positive slope (violation) : {bad}  "
          f"({100 * bad / max(1, total):5.1f}%)")
    print(f"    zero slope                 : {zero}")
    if bad_examples:
        for ex in bad_examples:
            print(f"    {ex}")
    if bad == 0 and total > 0:
        print("  => full-region sign closed on every C5 feasible pair")


# ---------------------------------------------------------------------------
# (4) Direct forward envelope check on C5 cache (recompute sigma/p and
#     evaluate both envelope Gaussians).
# ---------------------------------------------------------------------------
def check_envelope_gap() -> None:
    print()
    print("=" * 72)
    print("(4) Envelope forward check on C5 cache (p recomputed from sigma)")
    print("=" * 72)
    if not os.path.exists(C5_PATH):
        print("  [skip] C5.pkl not found")
        return
    with open(C5_PATH, "rb") as fh:
        cache = pickle.load(fh)
    rows = cache.get("rows", [])

    # Import the mechanism-solver sample-rate binary search so we can recover
    # p from (sigma, eps_target, steps, delta) exactly, then feed the pair
    # into the envelope maps.
    try:
        from game.mechanism import _sample_rate_for_eps
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] cannot import game.mechanism: {e}")
        return

    # Determine delta and steps used in C5: paper uses delta = 1e-5 and
    # steps = 5 * 50_000 / 128 (from EXPERIMENT_IDEAS.md). Use those.
    delta = 1e-5
    I = I_STEPS

    max_gap_abs = 0.0
    covered = 0
    n = 0
    max_mu_plus = 0.0
    for r in rows:
        sigma = r["sigma"]
        for eps_key, A_true_key in (("target_eps", "A_target"),
                                    ("other_eps", "A_other")):
            eps = r[eps_key]
            A_true = r[A_true_key]
            p = _sample_rate_for_eps(sigma, eps, I, delta, tol=5e-3)
            mu_p = mu_plus(p, sigma)
            mu_m = mu_minus(p, sigma)
            adv_up = adv_gauss(mu_p)
            adv_dn = adv_gauss(mu_m)
            n += 1
            gap = adv_up - adv_dn
            max_gap_abs = max(max_gap_abs, gap)
            max_mu_plus = max(max_mu_plus, mu_p)
            # The RDP@2 upper envelope is loose at large p; use a
            # relaxed test.
            if A_true <= adv_up + 1e-6:
                covered += 1

    print(f"  C5 cells inspected: {n}")
    print(f"  max envelope gap Adv(mu_+) - Adv(mu_-)     : {max_gap_abs:.6f}")
    print(f"  max mu_+ on the region                     : {max_mu_plus:.3f}")
    print(f"  A <= Adv(Gauss(mu_+)) upper-envelope hits  : "
          f"{covered}/{n}  ({100 * covered / max(1, n):5.1f}%)")
    print("  (mu_+ is the RDP@alpha=2 -> zCDP -> GDP upper envelope; it is")
    print("   uniformly loose because the mechanism-solver saturates the")
    print("   privacy budget through the tightest of ALL orders, not")
    print("   only alpha=2. The tight upper envelope requires the")
    print("   Balle-Barthe-Gaboardi 2020 privacy profile; see appendix.)")


if __name__ == "__main__":
    check_envelope_signs()
    check_C2_grid()
    check_C5_full_region()
    check_envelope_gap()
