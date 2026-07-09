# Existence theorem: full-region gap (26% of cells fail increasing differences)

**Priority:** M  **Effort:** H (theory)  **Category:** theory

## Issue
T21: the supermodular existence route closes only on the small-p sublattice;
~26% of interior feasible cells on the full K=200 grid fail the numerically
tested increasing-differences condition. main2 states this honestly
(app:proofs, "the sublattice qualifier is not removable with this proof
strategy") and relies on grid enumeration elsewhere. A reviewer may push on
existence for the full region.

## What
Options, in rough order of value/cost:
1. Characterise WHERE increasing differences fail (suspected: low-sigma /
   high-p saturation region) and prove existence there by a different route
   (e.g., Reny's better-reply security for discontinuous games, or
   quasi-concavity along the feasibility boundary).
2. Weaken the claim to "existence on the DP-SGD operating region" and add a
   figure/stat delineating the region where the lattice argument applies.
3. Leave as-is (current paper wording is defensible).

## Acceptance
A decision recorded here; if (1) or (2), update Thm. existence +
app:proofs.

## Update (2026-07-03, symmetry investigation)
Numerics on the K=200 cache (tasks_findings/_scripts/26_symmetry_breaking.py,
summary in 26_symmetry_summary.txt) show the budgets are predominantly
STRATEGIC SUBSTITUTES near the diagonal (cross-difference negative on
66.7/79.8/91.1% of diagonal cells at strides h=4/8/16). Consequence: the
all-coordinates-reversed lattice argument cannot work for n>2 (substitutes
cannot be complements for all pairs simultaneously). main2 now states the
existence theorem for n=2 via the one-coordinate reversal; the n>2 case is
open and should be attacked via the aggregative structure (sigma depends on
opponents only through the sum) in the Novshek/Kukushkin Cournot tradition
(novshek1985cournot is already in refs.bib).

Also open: the symmetric mixed equilibrium was computed exactly only at
K=22 (support enumeration; {2.52: 0.756, 3.17: 0.244}, value 0.304).
Verify at K=200 (needs a scalable symmetric-NE solver) and for n>2.
