# Errors found in the OLD paper during the rewrite (fixed in main2, kept for audit)

**Priority:** M (audit trail)  **Effort:** —  **Category:** record

All of these were discovered while rewriting `main.tex` -> `main2.tex` on
2026-07-03. They are FIXED in main2; listed here so nobody re-imports them
from `main_old.pdf` / `paper/tex/`.

1. **NE-count inconsistency.** Old Sec. C6 claimed the NE set is "eight
   isolated points, stable under refinement (K=44)" while the old PoEV
   section and V-ablation appendix used 48 NEs at K=200, and
   `fig_C6_locus_K200` was captioned as K=44. Truth (poev_C1.csv +
   C6_K200.pkl): the count GROWS with refinement (8 @ K=22 -> 48 @ K=200);
   what is stable is off-diagonality, mirror pairing, and the ratio band.
   main2 states the refinement behaviour honestly.
2. **PoEV maximum misattributed.** Old text: "maximum +0.04 at (1.43, 9.33)".
   Truth (poev_C1.csv): max +0.0400 at (1.61, 10.53) and mirror; (1.43,
   9.33) has PoEV -0.032.
3. **Old table tab:NE (K=22, 8 NEs) contradicted the K=200 prose.**
   Replaced by a single K=200 representative table in main2.
4. **Social optimum inconsistency.** Old tab:NE quoted SO = (5.04, 25.40)
   with A = (0.007, 0.535) (K=22), old PoEV table SO = (10.28, 12.80)
   (K=200). main2 uses K=200 only, and notes SO is itself one of the 48
   NEs (price of stability 0) — a fact the old paper missed.
5. **Old C7 prose was K=22** ("at w=4 a single diagonal NE at (0.5,0.5)");
   at K=200 all 18 w=4 NEs are OFF-diagonal. main2 uses the K=200 sweep,
   with the one genuine anomaly (w=0.125 diagonal cell, see task 04).
6. **T16 regret figure defect** (found by the T16 agent, already fixed):
   the original instance had the uniform-prior arm coinciding with the
   oracle arm, making the "uniform" baseline zero-regret. Redesigned with
   truth w=0.6; fig_h8 + 16_regret_summary.txt are the fixed versions.
7. **fig_C9 synthetic** (see tasks 02, 10).
8. **h=1 finite-difference certificate** (72.4% sign agreement) was the
   old paper's implicit basis for "verified numerically"; the honest
   stride-dependent numbers (99.1% at h=32) are now in app:certificate.
