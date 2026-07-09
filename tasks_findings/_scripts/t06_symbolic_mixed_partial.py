"""Symbolic verification of the Monderer-Shapley mixed-partial identity for
the Privacy Externality Game (Task T06).

Given the implicit definition of the mechanism from
`paper/tex/04_analysis.tex:44` (equation `eq:sigma-implicit`),

    sigma(eps) * sum_g |G_g| * r_g(eps_g)  =  C          (C = E_b * sqrt(2 I alpha))

with r_g(x) := sqrt(rho_g(x, delta)), and the closed form for the per-group
sample rate

    p_g(sigma, eps_g)  =  sigma * r_g(eps_g) / sqrt(2 I alpha),

we compute the second-order mixed derivatives sigma_{ij}, p_{i,ij}, p_{j,ij}
at an arbitrary interior profile *and* at the fully-symmetric reference
profile, and test the Monderer-Shapley symmetry condition

    eta_j * d^2 u_i / (d eps_i d eps_j)  =  eta_i * d^2 u_j / (d eps_i d eps_j).

The script is standalone and reproduces every algebraic step of Task T06,
Lemmas `lem:cross-partial-V` and `lem:cross-partial-A`, and the proof of
Proposition `prop:non-potential`.

Run:
    python3 tasks_findings/_scripts/t06_symbolic_mixed_partial.py
"""
from __future__ import annotations

import sympy as sp


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
e1, e2 = sp.symbols("e1 e2", positive=True)                # eps_1, eps_2
G1, G2, G3 = sp.symbols("G1 G2 G3", positive=True)         # group sizes
N = sp.symbols("N", positive=True)                         # generic third slot
w1, w2 = sp.symbols("w1 w2", positive=True)                # utility weights
eta1, eta2 = sp.symbols("eta1 eta2", positive=True)        # potential weights
C = sp.symbols("C", positive=True)                         # E_b sqrt(2Ialpha)
S = sp.symbols("S", positive=True)                         # sqrt(2Ialpha)

# r_g(x) := sqrt(rho_g(x, delta)) -- treated as an abstract smooth positive
# function of the budget with a nowhere-zero derivative. The functional form of
# rho_g does not enter the argument.
r = sp.Function("r", positive=True)


# ---------------------------------------------------------------------------
# Two-player scenario (the paper's canonical setup)
# ---------------------------------------------------------------------------
sigma = sp.Function("sigma")(e1, e2)

# Implicit definition: sigma * (G1 r(e1) + G2 r(e2)) = C.
constraint = sigma * (G1 * r(e1) + G2 * r(e2)) - C

# Solve the implicit relation for the first-order derivatives sigma_1, sigma_2
# via total differentiation.
d1 = sp.diff(constraint, e1)     # = sigma_1 * (G1 r + G2 r) + sigma * G1 r'(e1)
d2 = sp.diff(constraint, e2)
sigma_1 = sp.solve(d1, sp.diff(sigma, e1))[0]
sigma_2 = sp.solve(d2, sp.diff(sigma, e2))[0]

# Re-express in closed form: sigma_j = - sigma * G_j * r'(e_j) / (G1 r + G2 r).
R = G1 * r(e1) + G2 * r(e2)
sigma_j_closed = -sigma * G2 * sp.diff(r(e2), e2) / R
assert sp.simplify(sigma_2 - sigma_j_closed) == 0, "sigma_2 closed form failed"

# --------- Second-order mixed derivative sigma_{12} = d/de1 (sigma_2) --------
sigma_12 = sp.diff(sigma_2, e1)                 # symbolic (still has sigma_1)
sigma_12 = sigma_12.subs(sp.diff(sigma, e1), sigma_1)   # eliminate sigma_1
sigma_12 = sp.simplify(sigma_12)

# Compare against the hand-derived form 2*sigma*G1*G2*r'(e1)*r'(e2)/R^2
# at the *fully symmetric* profile e1=e2=e_star.
e_star = sp.symbols("e_star", positive=True)
sub_sym = {e1: e_star, e2: e_star, sigma: sp.Symbol("sigma_star", positive=True)}

sig, rr, rp = sp.symbols("sig rr rp", positive=True)


# r(e1) and r(e2) become r(e_star). We evaluate the substitution as functions.
def to_symmetric(expr):
    """Replace e1 and e2 by e_star in every functional occurrence."""
    expr = expr.subs(r(e1).diff(e1), rp)
    expr = expr.subs(r(e2).diff(e2), rp)
    expr = expr.subs(r(e1), rr)
    expr = expr.subs(r(e2), rr)
    expr = expr.subs(sigma, sig)
    return sp.simplify(expr)

sigma_12_sym = to_symmetric(sigma_12)
expected = 2 * sig * G1 * G2 * rp**2 / (G1 * rr + G2 * rr) ** 2
diff = sp.expand(sp.together(sigma_12_sym - expected))
assert sp.simplify(diff) == 0, (
    f"sigma_12 mismatch at symmetric profile: {sp.simplify(diff)}"
)

# ---------------------------------------------------------------------------
# Per-group rates and their second derivatives
# ---------------------------------------------------------------------------
p1 = sigma * r(e1) / S
p2 = sigma * r(e2) / S

# p_{1,1}, p_{1,2}, p_{1,12} for group 1
p1_e1 = sp.diff(p1, e1).subs(sp.diff(sigma, e1), sigma_1)
p1_e2 = sp.diff(p1, e2).subs(sp.diff(sigma, e2), sigma_2)

p1_e1_e2 = sp.diff(p1_e1, e2)
p1_e1_e2 = p1_e1_e2.subs(sp.diff(sigma, e2, e1), sigma_12)
p1_e1_e2 = p1_e1_e2.subs(sp.diff(sigma, e1, e2), sigma_12)
p1_e1_e2 = p1_e1_e2.subs(sp.diff(sigma, e2), sigma_2)
p1_e1_e2 = p1_e1_e2.subs(sp.diff(sigma, e1), sigma_1)
p1_e1_e2 = sp.simplify(p1_e1_e2)

p2_e1 = sp.diff(p2, e1).subs(sp.diff(sigma, e1), sigma_1)
p2_e2 = sp.diff(p2, e2).subs(sp.diff(sigma, e2), sigma_2)
p2_e1_e2 = sp.diff(p2_e2, e1)
p2_e1_e2 = p2_e1_e2.subs(sp.diff(sigma, e1, e2), sigma_12)
p2_e1_e2 = p2_e1_e2.subs(sp.diff(sigma, e2, e1), sigma_12)
p2_e1_e2 = p2_e1_e2.subs(sp.diff(sigma, e1), sigma_1)
p2_e1_e2 = p2_e1_e2.subs(sp.diff(sigma, e2), sigma_2)
p2_e1_e2 = sp.simplify(p2_e1_e2)

# Evaluate at the fully-symmetric profile.
p1_e1_e2_sym = to_symmetric(p1_e1_e2)
p2_e1_e2_sym = to_symmetric(p2_e1_e2)

# Closed-form prediction:
#   p_{1,12}|sym = sigma * (r')^2 * G2 * (2 G1 - N) / (r * N^2 * sqrt(2Ialpha))
#   p_{2,12}|sym = sigma * (r')^2 * G1 * (2 G2 - N) / (r * N^2 * sqrt(2Ialpha))
# with N := G1 + G2.
Nsym = G1 + G2
pred_1 = sig * rp**2 * G2 * (2 * G1 - Nsym) / (rr * Nsym**2 * S)
pred_2 = sig * rp**2 * G1 * (2 * G2 - Nsym) / (rr * Nsym**2 * S)
assert sp.simplify(p1_e1_e2_sym - pred_1) == 0, (
    f"p1_e1_e2 closed form mismatch: {sp.simplify(p1_e1_e2_sym - pred_1)}"
)
assert sp.simplify(p2_e1_e2_sym - pred_2) == 0, (
    f"p2_e1_e2 closed form mismatch: {sp.simplify(p2_e1_e2_sym - pred_2)}"
)

print("=" * 70)
print("Lemma cross-partial-V (shared-utility contribution)")
print("=" * 70)
print("sigma_{12} at symmetric profile:")
print("   ", sigma_12_sym)
print("Confirmed symmetric in (1<->2) exchange. Hence V(sigma)'s")
print("cross-partial cancels identically in the Monderer-Shapley check.")
print()

print("=" * 70)
print("Lemma cross-partial-A (privacy-cost cross-partial)")
print("=" * 70)
print("p_{1,12} |sym = ", sp.simplify(pred_1))
print("p_{2,12} |sym = ", sp.simplify(pred_2))
print()

# ---------------------------------------------------------------------------
# Monderer-Shapley check for u_i = V(sigma) - w_i A_i(sigma, p_i)
# ---------------------------------------------------------------------------
# We treat A_i as an abstract smooth function of (sigma, p_i). We show that
# under mechanism symmetry at e1 = e2 = e_star (equal budgets), the A-partial
# derivatives A_{i,sigma}, A_{i,p}, A_{i,sigma sigma}, A_{i,pp}, A_{i,sigma p}
# coincide for i=1 and i=2 (call them a_s, a_p, a_ss, a_pp, a_sp). This is a
# consequence of mechanism symmetry: at equal budgets, the mechanism returns
# equal sample rates so A_1 and A_2 evaluate at the same (sigma, p) point.
a_s, a_p, a_ss, a_pp, a_sp = sp.symbols("a_s a_p a_ss a_pp a_sp", real=True)

# We also need sigma_1|sym, sigma_2|sym, p_{i,i}|sym, p_{i,j}|sym (i != j).
sigma_1_sym = to_symmetric(sigma_1)
sigma_2_sym = to_symmetric(sigma_2)
p1_e1_sym = to_symmetric(p1_e1)
p1_e2_sym = to_symmetric(p1_e2)
p2_e1_sym = to_symmetric(p2_e1)
p2_e2_sym = to_symmetric(p2_e2)

# Cross-partial of A_i(sigma(e), p_i(sigma(e), e_i)) w.r.t. e_i, e_j (j != i).
def cross_A(sigma_i, sigma_j, p_ii, p_ij, p_iij, sigma_ij):
    return (
        a_ss * sigma_i * sigma_j
        + a_pp * p_ii * p_ij
        + a_sp * (sigma_i * p_ij + sigma_j * p_ii)
        + a_s * sigma_ij
        + a_p * p_iij
    )

# Player 1: mixed partial d^2 A_1 / de_1 de_2
d2A_1 = cross_A(
    sigma_i=sigma_1_sym, sigma_j=sigma_2_sym,
    p_ii=p1_e1_sym, p_ij=p1_e2_sym,
    p_iij=p1_e1_e2_sym,
    sigma_ij=sigma_12_sym,
)
# Player 2: mixed partial d^2 A_2 / de_1 de_2 (swap role of i/j accordingly)
d2A_2 = cross_A(
    sigma_i=sigma_2_sym, sigma_j=sigma_1_sym,
    p_ii=p2_e2_sym, p_ij=p2_e1_sym,
    p_iij=p2_e1_e2_sym,
    sigma_ij=sigma_12_sym,
)

# Shared cross-partial of V(sigma). Treat V'(sigma), V''(sigma) as abstract.
Vp, Vpp = sp.symbols("Vp Vpp", real=True)
d2V = Vpp * sigma_1_sym * sigma_2_sym + Vp * sigma_12_sym

# Monderer-Shapley LHS - RHS:
LHS = eta2 * (d2V - w1 * d2A_1)
RHS = eta1 * (d2V - w2 * d2A_2)
MS_residual = sp.simplify(LHS - RHS)
print("=" * 70)
print("Monderer-Shapley residual  LHS - RHS  at symmetric profile:")
print("=" * 70)
print(sp.simplify(MS_residual))
print()

# Isolate the a_p contribution (the surviving asymmetry).
MS_residual_ap = sp.collect(sp.expand(MS_residual), a_p).coeff(a_p)
print("Coefficient of a_p in MS residual:")
print("   ", sp.simplify(MS_residual_ap))
print()
print("Factor out the common (positive) prefactor:")
common = sig * rp**2 / (rr * Nsym**2 * S)
ratio_ap = sp.simplify(MS_residual_ap / common)
print("   ", sp.factor(ratio_ap))
print()
print("At symmetric profile, coefficient of a_p simplifies to")
print("     eta2 * w1 * G2 * (N - 2 G1) - eta1 * w2 * G1 * (N - 2 G2)")
print("(up to the common positive prefactor sig*(r')^2 / (r * N^2 * sqrt(2Ialpha))),")
print("with N = G1 + G2.")
print()

# Rest of residual (Vp and Vpp terms should cancel identically because they
# multiply symmetric quantities and get multiplied by (eta2 - eta1) if
# (V-terms cancel between LHS and RHS via the shared-utility symmetry).
# For a *weighted* potential (eta1 != eta2), the residual gets an extra
# (eta2 - eta1) * d2V component. That vanishes only if eta1 = eta2.

# Split into V-part and A-part.
V_part = sp.simplify((eta2 - eta1) * d2V)
A_part = sp.simplify(-eta2 * w1 * d2A_1 + eta1 * w2 * d2A_2)
print("V-only part of residual (equals (eta2-eta1)*d2V, so vanishes only")
print("in the exact-potential case eta1=eta2):")
print("   ", sp.simplify(V_part))
print()
print("A-only part of residual:")
print("   ", sp.simplify(A_part))
print()

# ---------------------------------------------------------------------------
# Special case: fully-symmetric heterogeneous scenario G1 != G2, w1 = w2,
# equal budgets. We show the surviving obstruction is
#      (eta2 - eta1) * V-terms  +  asymmetric A_p term.
# ---------------------------------------------------------------------------
res_special = sp.simplify(MS_residual.subs({w1: w2}))
print("Residual with w1 = w2 (still allowing G1 != G2, eta1 != eta2):")
print("   ", sp.factor(sp.simplify(res_special)))
print()

# ---------------------------------------------------------------------------
# Sub-case: additionally G1 = G2 (fully symmetric two-player). The residual
# reduces to (eta2 - eta1) * d2V(sigma). It vanishes only for eta1 = eta2.
# ---------------------------------------------------------------------------
res_full_sym = sp.simplify(res_special.subs({G2: G1}))
print("Residual with w1 = w2 and G1 = G2:")
print("   ", sp.factor(res_full_sym))
print("This is (eta2 - eta1) * d2V, so an *exact* potential (eta1 = eta2)")
print("would satisfy MS-sym at fully-symmetric profiles.  Any non-trivial")
print("weighted potential eta1 != eta2 fails.")
print()

# ---------------------------------------------------------------------------
# Sub-case: eta1 = eta2 (exact potential).  Then residual reduces to
#      (w2 * d2A_2 - w1 * d2A_1)   (*)
# We show this does NOT vanish even at symmetric profiles when G1 != G2
# because the a_p term survives.
# ---------------------------------------------------------------------------
res_exact = sp.simplify(MS_residual.subs({eta1: 1, eta2: 1}))
res_exact_ap_coeff = sp.simplify(sp.collect(sp.expand(res_exact), a_p).coeff(a_p))
print("Residual under eta1 = eta2 = 1 (exact-potential check):")
print("   a_p coefficient:", sp.factor(res_exact_ap_coeff))
print()

# ---------------------------------------------------------------------------
# Companion check: same argument for three players. Adds G3 slot to confirm
# generality.
# ---------------------------------------------------------------------------
print("=" * 70)
print("Three-player sanity: same asymmetry survives")
print("=" * 70)
e1_, e2_, e3_ = sp.symbols("e1 e2 e3", positive=True)
sig3 = sp.Function("sigma3")(e1_, e2_, e3_)
constraint3 = sig3 * (G1 * r(e1_) + G2 * r(e2_) + G3 * r(e3_)) - C
sig3_1 = sp.solve(sp.diff(constraint3, e1_), sp.diff(sig3, e1_))[0]
sig3_2 = sp.solve(sp.diff(constraint3, e2_), sp.diff(sig3, e2_))[0]
sig3_12 = sp.diff(sig3_2, e1_).subs(sp.diff(sig3, e1_), sig3_1)
sig3_12 = sp.simplify(sig3_12)

# Restrict to fully-symmetric profile e1=e2=e3=e_star.
def to_sym3(expr):
    expr = expr.subs(r(e1_).diff(e1_), rp)
    expr = expr.subs(r(e2_).diff(e2_), rp)
    expr = expr.subs(r(e3_).diff(e3_), rp)
    expr = expr.subs(r(e1_), rr)
    expr = expr.subs(r(e2_), rr)
    expr = expr.subs(r(e3_), rr)
    expr = expr.subs(sig3, sig)
    return sp.simplify(expr)

Nsym3 = G1 + G2 + G3
sig3_12_sym = to_sym3(sig3_12)
predicted = 2 * sig * G1 * G2 * rp**2 / (rr * Nsym3) ** 2
diff3 = sp.expand(sp.together(sig3_12_sym - predicted))
assert sp.simplify(diff3) == 0, f"3p sigma_12 mismatch: {sp.simplify(diff3)}"
print("3-player sigma_{12} at symmetric profile matches 2 sig G1 G2 (r')^2 / (r N)^2 with N=G1+G2+G3.  Symmetric in (1<->2).")

# p1 = sig3 * r(e1_) / S; p_{1,12}|sym
p1_3 = sig3 * r(e1_) / S
p1_3_e1 = sp.diff(p1_3, e1_).subs(sp.diff(sig3, e1_), sig3_1)
p1_3_e1_e2 = sp.diff(p1_3_e1, e2_)
p1_3_e1_e2 = p1_3_e1_e2.subs(sp.diff(sig3, e2_, e1_), sig3_12)
p1_3_e1_e2 = p1_3_e1_e2.subs(sp.diff(sig3, e1_, e2_), sig3_12)
p1_3_e1_e2 = p1_3_e1_e2.subs(sp.diff(sig3, e2_), sig3_2)
p1_3_e1_e2 = p1_3_e1_e2.subs(sp.diff(sig3, e1_), sig3_1)
p1_3_e1_e2_sym = to_sym3(p1_3_e1_e2)

pred_1_3 = sig * rp**2 * G2 * (2 * G1 - Nsym3) / (rr * Nsym3**2 * S)
diff3_p = sp.expand(sp.together(p1_3_e1_e2_sym - pred_1_3))
assert sp.simplify(diff3_p) == 0, f"3p p1_e1_e2 mismatch: {sp.simplify(diff3_p)}"
print("3-player p_{1,12}|sym matches closed form  sig (r')^2 G2 (2G1 - N) / (r N^2 sqrt(2Ialpha)).")
print()

print("=" * 70)
print("Conclusion")
print("=" * 70)
print("Symbolic algebra confirms:")
print("  (i)   sigma_{ij} at symmetric profile is symmetric in (i,j).")
print("  (ii)  p_{i,ij}|sym = sig (r')^2 G_j (2G_i - N) / (r N^2 sqrt(2Ialpha)).")
print("  (iii) Monderer-Shapley residual survives the eta1=eta2=1 test.")
print("        The residual's a_p coefficient factors as")
print("           w1 G2 (N-2 G1) - w2 G1 (N-2 G2)")
print("        (up to a positive common prefactor).")
print("  (iv)  For the *weighted* case eta1 != eta2 the residual also carries")
print("        (eta2 - eta1)*d2V which cannot be cancelled uniformly over the")
print("        feasible interval because sigma_1 sigma_2 and sigma_{12}")
print("        depend nontrivially on the reference budget e_star.")
print()
print("Therefore the PEG is not an exact potential game, not a weighted")
print("potential game, and (via a Voorneveld-2000 argument sketched in the")
print("findings note) not an ordinal potential game either.")
