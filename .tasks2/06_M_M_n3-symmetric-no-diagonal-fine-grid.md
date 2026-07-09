# n>=3 symmetric game: resolve the diagonal-FP tension on a fine grid

**Priority:** M  **Effort:** M (CPU Slurm)  **Category:** experiment/theory

## Issue
T19 (`tasks_findings/_scripts/19_h3_summary.txt`): under SYMMETRIC weights,
fictitious play on the K=7 grid settles on a single symmetric profile for
every n in 3..8 (Gini 0), while the 2-player theorem says no interior
symmetric NE exists. main2 (Sec. V, n-player subsection) explains this as a
discretisation effect (the diagonal deviation gain is second-order and falls
below a K=7 grid step) and flags the n>=3 analogue of the no-symmetric-NE
proposition as conjectural. That explanation is plausible but unverified.

## What
Two attacks, either closes it:
1. Numerical: rerun the symmetric-w FP at n=3 on a K=50+ grid (mechanism
   solves are the bottleneck; the n=3 tensor is K^3 — restrict to a
   neighbourhood of the diagonal). Check whether the diagonal profile
   survives as an FP attractor / discrete NE at fine resolution.
2. Analytic: extend app:proof-no-symmetric (aggregator structure
   sigma = C / sum_i s_i, p_i = C' s_i / sum s) to n players — the FOC
   collapse and the second-order cancellation generalise; only the
   mixed-partial bookkeeping changes.

## Acceptance
Either the proposition is stated for general n (with proof in
app:proofs), or the numerical check at K>=50 confirms/refutes the
discretisation explanation and the prose is updated.
