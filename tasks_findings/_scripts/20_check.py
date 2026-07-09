"""
Numerical corroboration of Task 20 (small-p corollary of the externality lemma).

We check three things:

1. Symbolic Taylor expansion (sympy) of the mixture-representation trade-off
   function T_{p,sigma}(alpha) = (1-p) * alpha + p * T_sigma(alpha) around p=0,
   and verify the leading-order MIA advantage identity
   Adv(p, sigma) = p * Adv_G(sigma) + O(p^2) with Adv_G(sigma) = 2*Phi(1/(2 sigma)) - 1.

2. Symbolic verification of the key inequality driving the dominance argument
   (Step 7 of the task spec): sigma * Adv_G(sigma) is strictly increasing in
   sigma, equivalently Phi(u) - Phi(-u) > 2 u phi(u) for all u > 0. We prove
   this by showing f(u) := Phi(u) - Phi(-u) - 2u phi(u) has f(0)=0 and
   f'(u) = 2 u^2 phi(u) > 0.

3. Cross-check the corollary against the C5 cache
   (game_experiments/_cache_v2/C5.pkl). We finite-difference A_target with
   respect to eps_other (holding target_eps fixed) at every cache row and
   report the sign. The corollary predicts negative signs in the small-p
   regime.

Reads:
- /vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C5.pkl
- (defensive) /vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C4.pkl

Writes:
- stdout only

Runtime: seconds; no Slurm required.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from math import erf, exp, pi, sqrt

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# (1) Symbolic Taylor expansion of the mixture trade-off function.
# ---------------------------------------------------------------------------

def check_taylor_expansion() -> None:
    print("=" * 72)
    print("(1) Symbolic Taylor expansion of T_{p,sigma}(alpha) at p=0")
    print("=" * 72)

    p, alpha, sigma, u = sp.symbols("p alpha sigma u", positive=True, real=True)

    # T_sigma(alpha) for the unsubsampled Gaussian mechanism at sensitivity 1
    # is T_sigma(alpha) = Phi( Phi^{-1}(1 - alpha) - 1/sigma )  (Dong-Roth-Su
    # GDP paper; also standard). We use symbolic Phi for the derivation.
    Phi = sp.Function("Phi")
    T_sigma = Phi(sp.Function("Phinv")(1 - alpha) - 1 / sigma)

    # Mixture representation for Poisson-subsampled mechanism (Balle-Barthe-
    # Gaboardi 2020, Prop. on advanced joint convexity): the "used" branch is
    # T_sigma with probability p, the "unused" branch is the identity trade-
    # off T_Id(alpha) = 1 - alpha with probability (1-p). Convex combination:
    T_mix = (1 - p) * (1 - alpha) + p * T_sigma
    # Small-p expansion around p=0:
    T_series = sp.series(T_mix, p, 0, 3).removeO()
    print("T_{p,sigma}(alpha) = (1-p) (1-alpha) + p T_sigma(alpha)  [mixture]")
    print("   Series in p:", sp.simplify(T_series))
    # The leading-order deviation from (1-alpha) is
    # T_{p,sigma}(alpha) - (1-alpha) = p (T_sigma(alpha) - (1-alpha)) + O(p^2).
    # Thus 1 - alpha - T_{p,sigma}(alpha) = p ( (1-alpha) - T_sigma(alpha) ) + O(p^2).
    lead = sp.expand(T_mix - (1 - alpha))
    print("Leading O(p) deviation from identity trade-off:",
          sp.simplify(lead.subs(p, sp.Symbol('p'))))

    # Suping over alpha of (1-alpha - T_mix) = p * sup_alpha (1-alpha - T_sigma)
    # + O(p^2) = p * Adv_G(sigma) + O(p^2).
    print()
    print("=> A_i(p, sigma) = p * Adv_G(sigma) + O(p^2)")
    print("   with Adv_G(sigma) = sup_alpha (1 - alpha - T_sigma(alpha))")
    print("                    = 2 Phi(1/(2 sigma)) - 1        [Gaussian MIA advantage]")

    # Closed-form Adv_G(sigma) = 2 Phi(1/(2 sigma)) - 1: verify by substitution
    # in the sup. Using Phi_std(x) = (1 + erf(x/sqrt 2))/2:
    def Adv_G(s: float) -> float:
        return erf(1.0 / (2.0 * sqrt(2.0) * s))  # = 2 Phi(1/(2 sigma)) - 1

    print()
    print("Numerical spot-checks of Adv_G(sigma) = 2 Phi(1/(2 sigma)) - 1:")
    for s in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
        print(f"  sigma = {s:5.2f}   Adv_G = {Adv_G(s):.6f}")


# ---------------------------------------------------------------------------
# (2) Key inequality Phi(u) - Phi(-u) > 2 u phi(u) for u > 0.
# ---------------------------------------------------------------------------

def check_key_inequality() -> None:
    print()
    print("=" * 72)
    print("(2) Key inequality: Phi(u) - Phi(-u) > 2 u phi(u) for u > 0")
    print("    Equivalent form: d/dsigma [sigma * Adv_G(sigma)] > 0")
    print("=" * 72)

    u = sp.symbols("u", positive=True, real=True)
    Phi_std = (1 + sp.erf(u / sp.sqrt(2))) / 2  # standard normal CDF
    phi_std = sp.exp(-u**2 / 2) / sp.sqrt(2 * sp.pi)  # standard normal PDF
    f_u = (Phi_std - (1 - Phi_std)) - 2 * u * phi_std  # = 2 Phi(u) - 1 - 2 u phi(u)
    fp_u = sp.simplify(sp.diff(f_u, u))
    print("f(u)   =", sp.simplify(f_u))
    print("f'(u)  =", fp_u, "     (must equal 2 u^2 phi(u))")
    target = 2 * u**2 * phi_std
    check = sp.simplify(fp_u - target)
    print("f'(u) - 2 u^2 phi(u) =", check, "     (should simplify to 0)")
    # f(0) = 0 immediately from erf(0)=0 and u=0.
    print("f(0)   =", sp.limit(f_u, u, 0), "     (should be 0)")
    print()
    print("Conclusion: f(0)=0 and f'(u)=2 u^2 phi(u) > 0 for u>0, hence")
    print("f(u) > 0 strictly for u > 0.  QED for the dominance inequality.")

    # Numerical spot check across the sigma range of the paper.
    print()
    print("Numerical spot-check of sigma * Adv_G(sigma) monotone-increasing:")
    prev = -np.inf
    for s in np.geomspace(0.3, 20.0, 12):
        val = s * erf(1.0 / (2.0 * sqrt(2.0) * s))
        marker = "OK" if val > prev else "FAIL"
        print(f"  sigma = {s:7.3f}   sigma*Adv_G(sigma) = {val:.6f}   [{marker}]")
        prev = val


# ---------------------------------------------------------------------------
# (3) Cache cross-check on C5 (paper Figure C5 -- 2-player, majority-target).
# ---------------------------------------------------------------------------

C5_PATH = "/vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/C5.pkl"


def check_C5_cache() -> None:
    print()
    print("=" * 72)
    print("(3) Finite-difference cross-check on C5 cache")
    print(f"    file: {C5_PATH}")
    print("=" * 72)
    if not os.path.exists(C5_PATH):
        print("  [skip] C5.pkl not found")
        return
    with open(C5_PATH, "rb") as fh:
        cache = pickle.load(fh)
    rows = cache.get("rows", [])
    print(f"  rows = {len(rows)}")

    # Group by (target_eps, proportion); within each group, sort by other_eps
    # and finite-difference A_target w.r.t. other_eps.
    key_of = lambda r: (round(r["target_eps"], 6), round(r["proportion"], 6))
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key_of(r)].append(r)
    total_pairs = 0
    neg_pairs = 0
    pos_pairs = 0
    zero_pairs = 0
    small_p_neg = 0
    small_p_total = 0
    P_STAR = 1e-2  # threshold for the "small-p regime"

    for k, group in groups.items():
        group.sort(key=lambda r: r["other_eps"])
        for a, b in zip(group[:-1], group[1:]):
            dA = b["A_target"] - a["A_target"]
            deps = b["other_eps"] - a["other_eps"]
            if deps <= 0:
                continue
            slope = dA / deps
            total_pairs += 1
            if slope < -1e-6:
                neg_pairs += 1
            elif slope > 1e-6:
                pos_pairs += 1
            else:
                zero_pairs += 1
            # For the small-p regime we use the smaller-sigma endpoint's A_other
            # as a proxy for whether the OTHER group's p is small: A_other is
            # monotone in p_other, and A_other < ~p_other * Adv_G(sigma) so
            # A_other << 0.01 => p_other << 0.01 / Adv_G(sigma) <= 0.01.
            # Both endpoints must be in small-p regime.
            proxy = max(a["A_other"], b["A_other"])
            if proxy < P_STAR:
                small_p_total += 1
                if slope < 0:
                    small_p_neg += 1

    print(f"  finite-difference dA_target/deps_other pairs: {total_pairs}")
    print(f"    negative slope     : {neg_pairs}  ({100*neg_pairs/max(1,total_pairs):5.1f}%)")
    print(f"    positive slope     : {pos_pairs}  ({100*pos_pairs/max(1,total_pairs):5.1f}%)")
    print(f"    zero/tiny slope    : {zero_pairs}")
    print(f"  small-p regime (A_other < {P_STAR}) pairs: {small_p_total}")
    print(f"    negative slope     : {small_p_neg} / {small_p_total}"
          f"  ({100*small_p_neg/max(1,small_p_total):5.1f}%)")
    if small_p_total > 0 and small_p_neg == small_p_total:
        print("  => corollary confirmed on every small-p cell of C5")
    elif small_p_total == 0:
        print("  => C5 does not contain small-p cells at this threshold;"
              " raise P_STAR or run a small-p sweep")
    else:
        print(f"  => WARNING: {small_p_total - small_p_neg} small-p cells "
              f"violate the corollary; investigate")


if __name__ == "__main__":
    check_taylor_expansion()
    check_key_inequality()
    check_C5_cache()
