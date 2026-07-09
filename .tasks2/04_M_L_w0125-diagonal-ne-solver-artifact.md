# Verify the lone diagonal NE at w=0.125 is a solver artifact

**Priority:** M  **Effort:** L  **Category:** verification / bug

## The suspicious data point
In the K=200 C7 sweep (`tasks_findings/_scripts/_out/10/poev_C7.csv`), the
w=0.125 slice contains exactly one diagonal pure NE, at grid index
(163,163) = (26.606, 26.606). This is the only diagonal NE among 241 across
the whole sweep. Direct inspection of the mechanism cache
(`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`) shows sigma along row
163 is locally NON-MONOTONE in eps_1 (0.5151 -> 0.5054 -> 0.4739 -> 0.4915
-> 0.4807 at eps_1 = 21.89 / 24.73 / 26.61 / 28.62 / 32.34), violating
Lemma 1's strict monotonicity — a solver-tolerance glitch, plausibly making
the cell spuriously attractive. main2 (Sec. V "shared shift in privacy
preference" + app:certificate) currently reports it as a solver-tolerance
artifact.

## What
Re-solve the mechanism on rows/columns 150-180 with tighter bisection
tolerances (e.g. 1e-5 relative instead of 1e-3 in
`game/mechanism.py`), recompute u_i in that neighbourhood at w=0.125, and
check whether (163,163) survives as an NE.
- If it disappears: the paper's wording stands; note it here and close.
- If it SURVIVES at tight tolerance: this contradicts
  Prop. no-symmetric-NE unless the cell violates the proposition's
  interiority/responsiveness hypotheses — investigate which hypothesis
  fails (likely eq. sign-condition or p-saturation) and adjust the
  proposition's scope statement in `paper/tex2/04_analysis.tex`.

## Acceptance
A definitive classification of the (26.6, 26.6) cell, reflected in main2.
