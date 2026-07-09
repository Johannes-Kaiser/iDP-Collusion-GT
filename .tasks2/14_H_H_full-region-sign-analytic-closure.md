# Full-region externality sign: analytic closure is OPEN (theorem downgraded)

**Priority:** H (soundness)  **Effort:** H (theory)  **Category:** theory

## What happened (2026-07-03 proof audit)
The old paper's Theorem "full-region monotonicity of A_i" (ported into main2)
rested on a Chernoff-envelope chaining argument (T23,
tasks_findings/23_chernoff_full_region_monotonicity.md) that does NOT close:
- Along an eps_j-move at fixed eps_i, the binding (Budget) constraint keeps
  p_i/sigma constant, so the upper-envelope scale mu_+ = 2p√I/sigma is
  INVARIANT — the two chained monotonicities cancel exactly.
- The CLT lower envelope mu_- = p√I·sqrt(e^{1/sigma^2}-1) moves AGAINST the
  claimed direction along the ray (sigma·sqrt(e^{1/sigma^2}-1) rises as sigma
  falls).
- The BBG profile monotonicities (A up in p, down in sigma) pull in opposite
  directions along the ray. T23's §3.3 "the residual gap inherits the sign"
  is an assertion, not a proof.

## Current paper state (main2, honest)
thm:externality-full is now a Proposition with two parts: (i) leading-order
(RDP-2) invariance — the externality is a strictly higher-order phenomenon,
which is itself a nice structural fact (it explains why the effect hides as
slack below the contract); (ii) the residual sign: analytic in the small-p
regime (cor:externality-small-p, rigorous), numerically certified on 99.1%
of the K=200 interior (app:certificate). app:proof-full-region documents the
reduction, the obstruction, and the certificate. Downstream statements
(slack monotonicity, coalition corollary, non-strategyproofness witness)
carry the "analytic at small p, certified beyond" status.

## What would close it
Candidate routes for a genuine full-region proof:
1. Work with the exact subsampled-Gaussian PLD along the constraint ray
   p = c·sigma and show d/d(sigma) A(c·sigma, sigma) > 0 directly (the
   small-p bracket is the first term of this derivative; need control of
   the higher-order mixture terms, e.g. via log-concavity of the PLD).
2. Second-order RDP expansion: the solver holds RDP at one order fixed;
   quantify the signed movement of the profile at neighbouring orders
   (alpha-derivative of the subsampled-Gaussian RDP along the ray).
3. f-DP calculus: express T_{p,sigma} via the tensor power of the
   single-step trade-off and differentiate the sup-gap along the ray using
   the exact single-step mixture form.

## Acceptance
Either a proof (then restore Theorem status and delete the certificate
qualifier from dependent statements) or a documented counterexample search
at the fine-discretisation solver (if the 0.9% residual cells at h=32
contain genuine sign violations in the saturated region, the proposition's
interior restriction needs sharpening instead).
