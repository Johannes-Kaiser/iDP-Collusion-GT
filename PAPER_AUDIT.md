# Paper audit: which claims in `paper/` still hold (2026-10-12)

This audit checks every quantitative or structural claim in `paper/main.tex`
and `paper/tex/*.tex` against three sources:

- the post-solver-fix caches (the fix is dated 2026-07-13);
- the exact-accounting campaign in `nummeric_eval.md`;
- fresh cache-only recomputations made for this audit:
  - `tasks_findings/_scripts/82_paper_audit.py` → `82_paper_audit.txt`
  - `tasks_findings/_scripts/t06_improvement_cycle.py`, re-run on the post-fix C1 cache

No mechanism was re-solved. Topic tags (T#) refer to `FINDINGS_MAP.md`.

**Verdicts**

| verdict | meaning |
|---|---|
| **HOLDS** | the claim matches the current data |
| **STALE** | the claim stands, but its numbers have changed |
| **SCOPED** | true only on the object named (e.g. the K=200 grid), not as worded |
| **FALSIFIED** | contradicted by the current data |
| **UNVERIFIED** | rests on a pre-fix run whose cache is missing or was never re-run |

## Resolution of the central contradiction: is there a symmetric equilibrium?

The paper contradicts itself on its central question:

- **Main text** (abstract, §IV): exactly one symmetric pure equilibrium,
  at ε* = 2.96.
- **Fig. 3 and Fig. 14 captions:** every pure NE lies off the diagonal.

The earlier version of this audit also leaned both ways. It used diagonal
grid NE to falsify the captions, while calling the 2.96 grid NE an
artefact.

**Settled 2026-10-12** by a dense exact-accounting slice:
`game_experiments/t390_local_slice.py`, Slurm 292170 →
`tasks_findings/_scripts/83_t390_local_slice.pkl`. The run evaluates 14
exact own-budget points around the diagonal at ε₁ ∈ {2.548, 2.92}
(T = 390, λ = 1, no accountant).

1. **The exact game has a symmetric NE.**
   - The own-budget slice is a concave **peak** at both columns
     (curvature ≈ −0.024 per (log ε)²).
   - BR(2.548) ≈ 2.62 and BR(2.92) ≈ 2.38, so the symmetric fixed point
     sits at **ε* ≈ 2.59** (bracketed in (2.55, 2.92)), with σ ≈ 1.16.
   - The best-response slope there is ≈ −0.65, so best-response play is
     **locally stable**. The limit model's |BR'| = 1.045 does not carry
     over to the exact game.
   - Far deviations (frontier corners) are lower by ≥ 1.2e-3 (52c
     columns).
   - A spline through the old 21-point columns had suggested a trough.
     That was an interpolation artefact across the frontier kink.
2. **The deployed game shows the same point shifted and textured.**
   - The 89-order RDP accountant moves it to 2.965 on the K = 200 grid,
     with a margin of 1.0e-4.
   - At 16× resolution the ~1e-4 sawtooth leaves a near-tie plateau
     (regret 1.7e-5) with no exact pure NE in the window.
   - The point exists as an equilibrium of the game; its pure-NE status
     on any grid is a resolution artefact.
3. **The diagonal grid points of the V-shape ablation are the same
   object:** one symmetric point for each nonlinear V, at a V-dependent budget,
   and none for linear V = −σ. Only V1 has been checked under exact
   accounting.

**Verdict:** the main text is right in substance (one symmetric
equilibrium, set by V saturation, at T = 390). Its location (2.96) and
its "25-point refinement" are deployed-grid artefacts. The captions of
Figs. 3 and 14 and the "zero diagonal equilibria" in the sensitivity
appendix are wrong.

Also unresolved by sampling: the onset and temptation-hump numbers in
nummeric_eval §20.1 (λ = 2 onset "by T ≈ 128", λ = 1 "at 390") come
from 31-point columns of the same coarseness. Treat them as UNVERIFIED
until re-run densely.

## Summary

| verdict | count | where |
|---|---|---|
| FALSIFIED | 10 | V-shape ablation counts and caption (Fig. 14), K=22 caption, group-size sorting, welfare-best symmetric NE, "exposed player compensated", loose-player near-indifference, ordinal-potential witness, n-player numbers, sensitivity appendix, non-DSIC witness numbers |
| SCOPED | 4 | symmetric NE at 2.96, λ = 2 "no pure NE", strict certificate, "crossing" mechanism |
| STALE | 3 | attack numbers (C4, 18–53×), Fig. 8 caption count |
| UNVERIFIED | 5 | BR-PEG numbers, λ-Pareto, H1 measured instance, Roughgarden/PoA, composition-onset numbers |
| HOLDS | the rest | externality sign, slack range, 43-NE structure, ratio band, Hawk–Dove numbers, self-sorting λ sweep, PoEV sweep, sensitivity-based control, H4 aware-target values, non-potential (exact) |

---

## FALSIFIED

### F1 · V-shape ablation (Fig. 14, `tab:H5-v-shape`, `app:v-shape`; T15)

**Paper:** six V shapes give 48/60/48/44/48/54 NE, and "100% of the pure
NE sit off the diagonal".

**Post-fix grid:**

| V | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|
| pure NE | 43 | 27 | 33 | 45 | 37 | 30 |
| diagonal NE | 1 (at 2.965) | 1 (at 1.359) | 1 (at 4.171) | 1 (at 1.911) | 1 (at 3.268) | 0 |

**What survives:**
- The asymmetric NE of every V lie on the same (V-free) participation
  frontier; V only changes their extent.
- One symmetric point for each nonlinear V, at a V-dependent location. This
  matches the paper's own main text (the symmetric equilibrium's
  location moves with V).
- The caption's "100% off-diagonal" contradicts the paper's own §IV; see
  the Resolution section.

**Cause:** the paper figure and table are pre-fix (2026-07-02). The
pre-fix NE clouds were scattered by solver errors.

**Fix:** re-run `15_v_shape.py` on the current cache. The post-fix
figure is drawn as `slides/assets/fig_v_shape_postfix.png`.

### F2 · Fig. 3 caption (`04_analysis.tex:395`): "Every pure NE lies off the diagonal"

**Post-fix K=22 cache:** 7 pure NE, including the diagonal (2.52, 2.52)
with margin 9e-4. The figure itself shows the star on the diagonal. The
caption contradicts §IV's own "present already at K = 22".

### F3 · Group-size sorting (`04_analysis.tex:490–513`, Fig. 5; T7)

**Paper:** "the smaller group plays the smaller equilibrium budget … at
every ratio", with ε₁* ∈ [1.5, 3] vs ε₂* ≈ 10 and A₁ < 0.01 vs
A₂ ∈ [0.24, 0.32].

**Post-fix `C2.pkl` (9-point grid):**
- For |G₁|/|G₂| < 1 the NE come in **both orientations** (mirror pairs).
  At ratio 0.1, for example: (1.68, 5.66) and (5.66, 1.68). Group size
  does not select who hides.
- For ratio ≥ 2 **only symmetric** NE remain.
- The majority's A peaks at 0.24, not 0.24–0.32.

The preference-sorting result (λ ratios) is unaffected; see HOLDS.

### F4 · "The symmetric equilibrium is the welfare-best one" (`01_intro.tex:81`, `08_discussion.tex:15`)

At λ = 1 the welfare-best NE is (1.612, 10.794) with W = 0.859. The
symmetric grid NE has W = 0.642.

The symmetric NE *is* the one with the **least aggregate advantage**
(ΣA = 0.207, PoEV −0.33). That reading holds.

### F5 · "Strictly higher realised risk compensated by the shared gain in model utility" (`main.tex:110`, `04_analysis.tex:532`)

At every frontier NE the exposed player's utility is 0.23–0.27, against
0.43–0.59 for the protected player. The Hawk–Dove restriction gives
0.253 vs 0.505.

The welfare analysis (`59_alt_plays`) finds that no equilibrium
compensates the carrier. The exposed player is strictly worse off.

### F6 · Loose player "near-indifferent along the wall, payoff varies by less than 10⁻⁴" (`04_analysis.tex:343–346`)

The exact FOC at (1.12, 7.13) shows the loose player strictly blocked by
the feasibility wall: derivative +9.7e-3 into the wall (`exact_foc.py`,
nummeric_eval §5).

Along the NE segment the loose payoff spans 0.225–0.267. The equilibria
are corner equilibria with two one-sided conditions, not indifference
points.

### F7 · Ordinal-potential witness (`09_appendix.tex:505–517`)

The quoted K=22 4-cycle (1.587, 5.040) → … does **not exist** on the
post-fix C1 cache: `t06_improvement_cycle.py` finds no length-4
improvement cycle.

**The claim survives with a new witness.** The post-fix K=200 grid has
a 6-step strict improvement cycle at λ = 1:
(2.824, 3.432) → (2.824, 2.499) → (2.561, 2.499) → (2.561, 3.268) →
(2.267, 3.268) → (2.267, 3.432). Gains are 2.4e-4 to 1.5e-3.
- The smallest gain is only ~2× the accountant sawtooth.
- The λ = 1 orbit survives the 16× zoom (nummeric_eval §20.5) but was
  not tested under dense orders.

The exact-potential part holds independently: off-diagonal cross-partials
of A₁ and A₂ differ by a median 135% relative.

### F8 · n-player numbers (`05_experiments.tex:22–45`, Fig. 7; T10)

**Paper:**
- self-sorting exact for n ≤ 8, **83%** at n = 10;
- A_max/A_min from **480** (n = 3) to **75** (n = 10);
- symmetric weights at n = 10 give "asymmetric profiles with advantage
  ratios near 5".

**Post-fix `H3_nplayer.pkl`:**
- Kendall τ < 0 in **100%** of profiles at every n, n = 10 included.
  The `H3_nplayer_table.tex` value of 0.00 for n ≥ 6 is a known
  clustering bug.
- A ranges: n = 3 0.004–0.174 (≈ 44×), n = 4–5 ≈ 330×, n = 6–10
  ≈ 120–175×.
- Symmetric weights at **every** n, including n = 10, converge to the
  grid floor (1, …, 1). That is a corner of the K = 7 action set, not
  an interior symmetric profile.
- The current Fig. 7 shows NE count and Gini, not the sorting curve its
  caption describes.

### F9 · Sensitivity appendix (`09_appendix.tex:826–844`)

**Paper:** "8 → 48 NE", "zero diagonal equilibria", R3 subgrid 12 / 12 / 6
NE, certificate 95.5% / 92.7%.

**Post-fix:**
- 7 → 43 NE, with one diagonal NE at each resolution.
- R3: 16 / 16 / 8 NE (`17_r1_r5_summary.txt`).
- Certificate: 0 sign violations at K = 200.

### F10 · Non-DSIC witness numbers (`09_appendix.tex:797–800`, `07_mitigation.tex`)

**Paper:** best report ≈ 2.92 against ε̂₂ = 4.0 and ≈ 1.21 against 0.5
(07_mitigation: "against a loose opponent ≈ 2.9, against a tight one
≈ 1.2").

**Post-fix K=200:** 1.82 against 3.97, and 3.43 against 0.5.

The proposition still holds, because the best report depends on the
opponent. The direction has **reversed** and is now consistent with
strategic substitutes: tighten against a loose opponent. The prose
example must be rewritten.

---

## SCOPED (true only on the object named)

| claim | status |
|---|---|
| "exactly one symmetric pure equilibrium at ε* = 2.96, confirmed on a 25-point local refinement" (`04_analysis.tex:236`, abstract, table_NE) | Existence and uniqueness: HOLDS (exact game, see Resolution). Location: exact ε* ≈ 2.59 (σ ≈ 1.16, locally BR-stable). 2.96 is the deployed-grid image; the "25-point refinement" is a grid statement (the 16× zoom leaves only a 1.7e-5 near-tie). |
| λ = 2 "no pure equilibrium, only mixed play survives" (`04_analysis.tex:358`, `app:wsweep`) | True for the 89-order accountant at K = 200. With 761 Rényi orders the cycle collapses onto 4 pure NE at ε ≈ 1.025–1.029. |
| sign certificate "25,612 of 25,612" strict, both directions (`04_analysis.tex:142`, `app:certificate`) | Own direction: 25,612/25,612 ✓. Cross direction: 25,291 strictly negative + **321 exact ties** at the floor, 0 positive. Write "no violation". |
| "play settles … where the loci cross" (`04_analysis.tex:288–298`, "crossing band") | The equilibria are where the two players' corner branches **coincide** on the participation wall (a continuum), not transversal crossings. |

## STALE (claim stands, numbers moved)

| claim | paper | cache |
|---|---|---|
| C4 passive target (`06_attacks.tex:28`) | 0.442, 0.483, 0.531, 0.607, 0.728 (+65%) | 0.443, 0.479, 0.534, 0.608, 0.729 (+64%) |
| aware-target collapse (abstract, intro, `06_attacks`) | 18–53× | 17.9–51.5× |
| Fig. 8 caption (`05_experiments.tex`) | "48 pure NE, all off-diagonal" | 43, one diagonal |

## UNVERIFIED (pre-fix, not re-run)

| claim | note |
|---|---|
| BR-PEG: oracle ε_M* ≈ 3.2, A_H ≈ 0.454, regret R(60) = 1.31 / 8.3 / 8.4 / 13.2 | `bayesian_demo.pkl` is missing from `_cache_v2`; the regret run is from 2026-07-03 |
| H1 measured instance | Shadow models trained on pre-fix (σ, p). The audit shows σ within 2.6% and rates up to 18.5% off at (4, 32). The floor conclusion is unlikely to change, but it was never retrained. |
| Roughgarden smoothness / PoA (if quoted) | pre-fix (`22_check.py`, 2026-07-02) |
| λ-Pareto coalition sweep | cache dated on the fix day; not re-checked |

## HOLDS (verified on current caches)

- **Externality:** sign holds (0 violations). Slack at ε_i = 4 swings
  0.79–0.96 against a ceiling of 0.963. The share curves cross at
  ε_O ≈ ε_T (Fig. 15).
- **λ = 1 NE structure:** 43 pure NE (42 frontier mirror pairs + 1
  diagonal grid NE). Ratio band 6.38–6.70, median 6.54. The tight player
  is within one cell of the floor; A reaches 0.32 vs < 0.001.
- **Diagonal stats:** 200 feasible diagonal cells, median best-deviation
  gain 0.122. Of 200 diagonal starts, 197 leave the diagonal and 134
  end at a pure NE. Cross-differences are negative in 81 / 90 / 95% of
  cells.
- **Hawk–Dove at (1.118, 7.131):** 0.505 / 0.253; mixed P(tight) = 0.445,
  value 0.349.
- **Self-sorting under λ ratios:** 170/174, with all 4 exceptions at 0.5.
- **Common-λ sweep (Fig. 16):** 44 / 62 / 34 / 43 / 0 / 1 NE.
  - PoEV: 100% inflationary at λ ≤ 0.25 (max +1.65), 94% at 0.5, 14% at
    1, −0.82 at λ = 4.
  - Uniform baseline 5.73, ΣA = 0.310. The PoEV range at λ = 1 is
    [−0.33, +0.05], median −0.18.
- **Sensitivity-based control:** unique NE (5.73, 5.73); single diagonal
  NE at λ ∈ {0.25, 0.5, 1}. At λ = 0.125 it is an asymmetric pair; at 2
  and 4 it is diagonal.
- **H4 aware target:** 0.014 / 0.010 / 0.018 / 0.041. Leader budgets
  44.1 / 6.8 / 64 / 0.5; target 9.9 / 6.8 / 14.4 / 0.73.
- **No exact or weighted potential:** cross-partial mismatch. The
  ordinal part needs the new witness (F7).
- **Exact theory, consistent with the paper:** T = 1 and T = 2 have no
  interior symmetric NE at any weight. The asymmetric equilibria form a
  continuum segment on the analytic frontier.

## Action list for the paper (in order of severity)

1. Make the symmetric-equilibrium story consistent across the paper:
   - One symmetric NE, exact location ε* ≈ 2.59 at T = 390, λ = 1, set by
     V saturation; it appears on the deployed grid at 2.96.
   - Fix the Fig. 3 / Fig. 14 captions and the sensitivity appendix.
   - Regenerate Fig. 14 and `tab:H5-v-shape` on the post-fix grid.
2. Remove or rewrite the group-size sorting paragraph and Fig. 5 caption
   (F3).
3. Replace "welfare-best" with "least aggregate advantage" (F4), and drop
   "compensated" (F5).
4. Replace the near-indifference description with the corner (two
   one-sided conditions) description (F6, SCOPED crossing).
5. State the exact-accounting location (≈ 2.59, stable) next to the grid
   value; drop the "25-point refinement" sentence.
6. Swap the ordinal-cycle witness for the K=200 6-cycle (F7).
7. Fix the n-player paragraph and Fig. 7 caption (F8); fix the `run_nplayer`
   table generator.
8. Fix the sensitivity appendix (F9), the non-DSIC example (F10), and the
   stale attack numbers.
9. Mark BR-PEG and H1 as pending re-runs, or re-run them.
