# Roughgarden smoothness PoA: decide whether to include

**Priority:** L  **Effort:** M  **Category:** theory

## Status
T22 derived a (lambda, mu)-smoothness bound with closed-form constants only
in the low-privacy-weight regime; it is ill-defined at high w. The
worst-NE/SO aggregate-advantage ratio is ~12.5x across sweeps (vs 4.8x at
w=1 from poev_C1.csv: worst NE sum A = 0.318 vs SO 0.066). main2 does NOT
include the smoothness analysis (only the PoEV numbers).

## What
Either finish the smoothness write-up with its honest regime restriction and
add it as an appendix (helps against "PoEV is just a table" criticism), or
record a decision to drop it. If added: reconcile the 12.5x cross-sweep vs
4.8x at-w=1 ratios explicitly (they condition on different weight regimes).
