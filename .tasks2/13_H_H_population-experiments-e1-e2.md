# E1 + E2: Realistic FL-consortium population experiments

**Priority:** H  **Effort:** H (CPU Slurm for dynamics; GPU Slurm for training)
**Category:** experiment + paper restructure
**Depends on:** existing `game/mechanism.py`, `game/advantage.py`, `game/utility.py`
**Produces:** three new main-body figures, a new experiments narrative, demotion
of the current measured-LiRA section to the appendix.

## 0. Purpose and target narrative

Two experiments that together move the paper from "equilibria exist" to "the
game is descriptive and behaviourally plausible":

- **E1 (equilibrium prediction).** In realistic, heterogeneous FL populations,
  the equilibrium predicts how participants self-select budgets: preferences
  sort budgets (Panel A), budgets set realized level-2 advantage (Panel B),
  and equilibrium play *creates privacy inequality* relative to uniform
  budgets (Panel C, Lorenz/Gini — the memorable figure).
- **E2 (adaptive discovery).** Myopic agents that know only their own utility
  — no equilibrium solver — converge to the same outcomes. Answers "real
  organizations don't solve Nash equilibria."

Final paper arc: theory (externality) → population simulation (self-sorting,
inequality) → adaptive dynamics (reachable by simple agents). The current
measured-LiRA section moves to the appendix (step 6).

**Level discipline (hierarchy eq:hierarchy in main2):** all game dynamics run
on the *level-2* advantage (PLD-exact, mechanism-specific). Real training is
used only to (a) measure accuracy at selected profiles (validates V) and (b)
optionally, in the appendix, measure LiRA. Never mix levels in one plot
without labelling them.

---

## Phase A — shared infrastructure (CPU, mostly login-safe; anything > a few
minutes goes to Slurm per CLAUDE.md Rule 3)

**A1. Purchase100 data.**
1. Obtain the Purchase-100 dataset as used in the MIA literature (Shokri et
   al. 2017 preprocessed version: 197,324 records, 600 binary features, 100
   classes; commonly distributed as `purchase100.npz` or via the
   privacy-evaluation repositories). Store under the repo's existing dataset
   root (find it by inspecting `experiment_scripts/mia/exp_yaml/*.yaml` and
   the loader in `experiment_scripts/mia/`; mirror how `adult` is wired).
2. Add a `purchase100` entry to the dataset loader used by
   `experiment_scripts/mia/04_mia_pergroup.py` (same normalisation pattern as
   existing tabular datasets; train/test split 80/20, fixed seed 42).
3. Sanity check (login node, minutes): train the reference MLP (step A2)
   non-privately for 5 epochs; expected test accuracy well above 1% chance
   (literature: 70–90% non-private). Record the number in the findings file.
4. Adult stays available as the appendix sanity dataset (already wired).

**A2. Reference model (fixed for every experiment).**
MLP: input → 256 → ReLU → 256 → ReLU → output(100). No dropout, no batch
norm (DP-SGD-compatible). This exact architecture is used everywhere; write
it once in a shared module and import it from both training scripts.

**A3. n-group level-2 utility oracle.**
1. Wrap `game.mechanism.solve_mechanism` + `game.advantage.
   mia_advantage_subsampled_gaussian` into a single function: input a
   population spec (list of group sizes, list of budgets, E_b, steps, delta)
   → output (sigma, p_1..p_n, A_1..A_n, u_1..u_n) with u_i = V(sigma) −
   w_i·A_i and V(sigma)=1/(1+sigma²).
2. Cache aggressively, two layers: (i) memoise mechanism solutions by the
   *multiset* of (|G_i|, eps_i) pairs (the solver is label-invariant); (ii)
   the advantage is already lru-cached by (sigma, p, steps) — keep sigma and
   p rounded to 6 significant digits before lookup so cache hits actually
   occur.
3. Persist the cache to `game_experiments/_cache_flpop/oracle_cache.pkl`
   between Slurm tasks (one cache file per population is fine; do NOT share
   one writable file across concurrent array tasks).

**A4. Pre-flight feasibility scan (decides E_b and epochs; do this before
anything else).**
1. Proposed defaults: delta = 1e-12 (NON-NEGOTIABLE: delta must be set very
   low; a very small delta spreads the level-2 advantage over a much wider
   eps range, which is what gives the game strategic content — at loose
   delta the advantage saturates early and the grid collapses into
   near-indifference), expected batch E_b = 256, epochs = 10,
   steps I = epochs × floor(M/E_b) where M = Σ|G_i| (population-specific).
2. Sample 10 populations per the E1 protocol (B1). For each, evaluate the
   oracle on: the all-uniform profiles (all six grid values) and 50 random
   profiles. Record: feasibility rate, sigma range, A_i range, wall time per
   solve.
3. Acceptance: ≥90% of random profiles feasible; sigma within [0.4, 6];
   median solve time ≤ 2 s. If infeasibility concentrates at eps=1 for small
   groups, either raise E_b to 512 or accept and document that the effective
   action set shrinks at the low end (feasibility is part of the game per
   Assumption "Participation" — do not silently drop it).
4. Freeze (E_b, epochs, delta) after this step and use the same values in
   every later step, including training. Write them into the findings file.

**A5. Population sampler.**
- n = 20 participants.
- |G_i| ~ Uniform{2000, ..., 12000} (integers).
- w_i ~ LogNormal(mu=0, sigma=0.7) (median 1, ~95% within [0.25, 4] — matches
  the paper's weight sweeps).
- Action set for everyone: six log-spaced budgets on [10, 50],
  eps ∈ {10, 14, 19, 26, 37, 50}. Rationale: at delta = 1e-12 the
  interesting (spread-out, non-saturated) region of the level-2 advantage
  sits at eps ≈ 10–50; a low-eps grid like {1..32} would pin most players
  against the participation floor with near-zero advantage differences. The
  A4 scan must CONFIRM the spread on this grid: across the six uniform
  profiles, the population-median A_i should span at least [~0.05, ~0.5];
  if it does not, shift the grid (same log spacing) until it does and record
  the final grid in the findings file.
- Seeded: population k uses seed k (0..99); store each population spec as a
  row in `game_experiments/_cache_flpop/populations.csv` (id, sizes, weights).

---

## Phase B — Experiment 1: best-response dynamics on 100 populations
(CPU Slurm array, one task per population; template
`slurm_scripts/run_locus_K200.sh`, request ~4 cores, 8 GB, 2 h per task)

**B1. Decision protocol (sequential best response).**
1. Initialise every player at the grid midpoint (eps = 19 on the default
   grid; use the third-of-six value if the grid shifted in A4). Also record
   results for a second initialisation at the grid top (eps = 50), to probe
   equilibrium selection; same seeds.
2. One round = players 1..20 in fixed index order; player i replaces her
   budget with argmax over the 6 grid values of u_i(eps_i, eps_{-i} current),
   evaluating the oracle 6 times; ties break toward the *current* value.
3. Convergence: a full round with zero changes. Cap at 200 rounds. Detect
   cycles by hashing the profile after each round; if a hash repeats without
   convergence, record the full cycle, mark the population "cycling", and use
   the cycle-average of each measurement for that population (report the
   number of cycling populations explicitly — do not hide them).

**B2. Baseline profiles (computed for every population).**
- *Uniform:* all players at the same eps, for each of the six grid values;
  the "uniform baseline" in figures is the utilitarian-best uniform value
  (argmax of Σu_i among the six).
- *Planner optimum (approximate):* greedy coordinate ascent on Σu_i —
  repeatedly sweep players in random order, setting each to the Σu-maximising
  budget, until no change; run from 5 starts (each uniform grid value except
  1, plus one random profile) and keep the best. Label it "planner
  (approximate)" everywhere; it is a lower bound on the true optimum.

**B3. Per-population measurements (write one results row per population to
`_cache_flpop/E1_results.parquet` or CSV).**
Per participant: chosen eps*, realized level-2 advantage A_i, utility u_i,
privacy weight w_i, group size |G_i|, contribution weight |G_i|/M, sampling
rate p_i. Global: sigma, V(sigma), average eps, variance of eps, Gini
coefficient of the A_i vector (standard definition on non-negative values),
same Gini for the uniform baseline and the planner profile, rounds to
convergence, converged/cycling flag.

**B4. Statistics over the 100 populations (the numbers the paper quotes).**
1. Sorting: Spearman (and Kendall tau) correlation between w_i and eps_i*
   within each population; report the distribution (median, IQR) across
   populations. Expected: strongly negative.
2. Advantage ordering: Spearman correlation between eps_i* and A_i within
   population. Expected: strongly positive.
3. Inequality: paired comparison Gini(equilibrium) vs Gini(uniform-best) per
   population; report median difference and the fraction of populations where
   equilibrium Gini is higher; Wilcoxon signed-rank test p-value.
4. Robustness: repeat 1–3 for the grid-top initialisation; report agreement.

**B5. Accuracy measurement (GPU Slurm; the only training in E1).**
1. Select 10 populations (ids 0..9). For each, train the reference MLP with
   sampling-based iDP-SGD (per-group (sigma, p_i) from the oracle) at three
   profiles: equilibrium, uniform-best, planner. 3 seeds each → 90 runs.
2. Slurm: `--gres=gpu:1`, ~8 cores, 32 GB, ≤ 2 h per run (Purchase100 MLP is
   small); array job over (population, profile, seed).
3. Record test accuracy; report mean ± sd per profile class. Purpose:
   validate that the V ordering of the three profiles matches measured
   accuracy ordering (directional check, not calibration). If it does not,
   report the disagreement honestly and add a limitation sentence — do not
   tune V post hoc.

**B6. Main figure (`paper/figures/fig_flpop_equilibrium.{pdf,svg}`,
full-width, three aligned panels, style from `game/style.py`).**
- Panel A: scatter of w_i (log x-axis) vs equilibrium eps_i* (log y-axis),
  all 2000 participant-points pooled, alpha-blended; overlay the per-decile
  median line. Annotation: median Spearman rho.
- Panel B: scatter of eps_i* vs realized A_i (log-log), same pooling; this
  panel shows players who chose stronger privacy also *received* lower
  realized risk (own-budget monotonicity survives the population setting).
- Panel C: Lorenz curves of realized advantage — equilibrium (solid) vs
  uniform-best (dashed), averaged across populations, with a Gini inset
  (paired dots or a small box plot: Gini_uniform vs Gini_equilibrium).
  Caption message: equilibrium play spontaneously concentrates realized
  privacy risk.
- Every quantity in the figure is level-2; say so in the caption.

---

## Phase C — Experiment 2: adaptive agents (CPU Slurm array over 50 seeds)

**C1. Setting.** One fixed representative population (pick the E1 population
whose (sorting rho, Gini gap) is closest to the across-population medians;
record its id). Same oracle, same action grid. 50 seeds of the learning
dynamics; T = 500 iterations.

**C2. Learning rules (agents know only their own utility).**
- *Primary — myopic local search:* at each iteration, one uniformly random
  agent is activated; she evaluates her utility at her current eps and at one
  adjacent grid value (direction uniform at random; reflect at the grid
  ends); she moves iff utility strictly improves by > 1e-6. No agent ever
  sees another's budget, weight, or advantage.
- *Variant — noisy observation:* same rule with additive N(0, 0.05) noise on
  each utility observation (accept on noisy comparison). Reported in the
  appendix; shows robustness.
- *Variant — bandit:* per-agent EXP3 over the 6 arms, reward = own utility
  rescaled to [0,1], activated agent plays one arm-draw per activation.
  Appendix.

**C3. Trajectories compared (four curves per metric).**
1. Uniform-best (constant line, from B2).
2. Planner approximate (constant line, from B2).
3. PEG equilibrium (constant line: the B1 equilibrium of this population).
4. Adaptive learning (mean ± 1 SE band across 50 seeds, vs iteration).

**C4. Measurements per iteration (log every 5 iterations to keep files
small):** average eps, variance of eps, model-quality term V(sigma), average
utility, average realized advantage, Gini of realized advantage, and
instantaneous regret: mean over agents of [max over own grid of u_i(·,
current others) − u_i(current)] — this needs 6 oracle calls per agent, so
compute it only at the logging iterations.

**C5. End-state analysis.** At T = 500: fraction of seeds whose final profile
is a pure NE (verify: no player can improve by any grid move); distance of
final Gini to the B1 equilibrium Gini; report both.

**C6. Figure 1 (`fig_flpop_learning.{pdf,svg}`, full-width, four panels):**
iteration on x; panels: (a) accuracy proxy V(sigma), (b) average eps, (c)
Gini of realized advantage, (d) mean instantaneous regret (log y). Horizontal
reference lines for uniform-best / planner / PEG equilibrium in each panel.
Expected story to check for (and state honestly if absent): V stabilises
within tens of iterations while Gini keeps rising toward the equilibrium
level — utility converges fast, inequality emerges slowly.

**C7. Figure 2 (`fig_flpop_trajectory.{pdf,svg}`, single column):** pick the
two agents in the C1 population with the most similar (w_i, |G_i|) pairs
(nearly-interchangeable players). Plot their joint budget trajectory in the
(eps_a, eps_b) plane (log-log) for one representative seed plus 5 faint
alternative seeds: start near the diagonal, settle into opposite roles.
Overlay the diagonal and the equilibrium point(s). This is the
population-scale companion of the existing fig_symmetry_breaking panel (b).

---

## Phase D — evaluation, sanity checks, and honesty items

1. Oracle self-consistency: for one population, brute-force the best response
   of one player against a fixed profile by direct PLD evaluation and confirm
   it matches the oracle's argmax (guards against cache-rounding bugs).
2. Grid-boundary audit: report how many equilibrium budgets sit at the grid
   ends, eps = 10 or 50 (boundary-heavy equilibria weaken the sorting claim; if > 30% of
   players are at a boundary, extend the grid one step on that side and
   rerun E1 — the grid is cheap).
3. Feasibility audit: number of BR evaluations rejected as infeasible;
   confirm the participation assumption is doing the exclusion, not solver
   failures (inspect a sample of rejects).
4. Cycling populations (B1): if > 10% cycle, add the cycle statistics to the
   appendix and soften "converges" to "converges or cycles tightly" in the
   prose. Consistent with the paper: the PEG is not a potential game, so
   BR dynamics need not converge — cite Prop. non-potential when discussing.
5. Adult appendix sanity run: repeat E1 analytics-only (no training) on 20
   Adult-sized populations (|G_i| ~ Uniform{1000, 5000}); confirm the same
   qualitative Panel A/B/C pattern; one appendix figure.

## Phase E — paper integration (main2.tex / tex2/)

1. New subsection(s) in Section "Numerical Evaluation": replace the current
   single-subsection measured-instance story with (i) "Does the equilibrium
   predict behaviour in realistic populations?" (E1, fig_flpop_equilibrium)
   and (ii) "Can simple agents discover it?" (E2, fig_flpop_learning +
   fig_flpop_trajectory). Question-form titles, no experiment codenames, all
   quantities labelled level-2 per the hierarchy.
2. Demote the current measured-LiRA subsection (sec:H1 content and fig_H1) to
   a new appendix "A measured lower level of the hierarchy": keep
   fig_advantage_hierarchy in Section II (it is conceptual and uses the
   measured instance only as an example — keep), move the 5×5 measured-game
   analysis and its figure to the appendix, keep the honest floor discussion
   and the finite-sample-floor sentence. Update all \cref references
   (sec:H1 appears in 01_intro, 02_background, 05_experiments,
   08_discussion) and the hierarchy paragraph's "we measure it directly in
   ..." pointer.
3. Update the intro's contribution bullet 2 (or 3) with one clause on the
   population result once numbers exist (e.g. "in simulated 20-participant
   consortia, equilibrium play raises the Gini coefficient of realized
   advantage from X to Y while preferences sort budgets at rho = Z").
4. Discussion: add E2 to the "what survives" story (adaptive reachability),
   and remove/replace the "measured instance is the missing experiment"
   limitation with the new, smaller limitation (accuracy validated
   directionally at 10 populations; LiRA still floor-bound on tabular data).
5. Rebuild per Rule 4 (pdflatex + bibtex, warning-clean), delete any figure
   files that become orphaned (Rule 0/2).

## Phase F — resource budget (Rule 3)

- E1 dynamics: 100 CPU array tasks × ≤ 2 h × 4 cores. E2: 50 tasks × ≤ 1 h.
  Planner + baselines piggyback on the E1 tasks.
- Training (B5): 90 GPU runs ≤ 2 h each, `--gres=gpu:1` — batch as one array.
- Optional appendix LiRA on Purchase100 (only if time permits; separate
  decision): 3 profiles × 128 shadows on population 0 — ~50–80 GPU-h; this
  would also retire the "above-floor measured instance" task
  (.tasks2/01) if Purchase100 LiRA turns out non-flat.

## Acceptance criteria

1. `populations.csv`, `E1_results.*` complete for 100 populations, ≥ 90%
   converged (or cycling documented per D4).
2. Panel A median Spearman rho(w, eps*) ≤ −0.6; Panel B rho(eps*, A) ≥ +0.6
   (if weaker, report the actual value — the direction is the claim, the
   threshold is only an expectation).
3. Gini(equilibrium) > Gini(uniform-best) in a significant majority of
   populations (Wilcoxon p < 0.01), with the Lorenz figure rendering the gap
   visibly.
4. E2: ≥ 80% of seeds end at (or within one grid step per player of) a pure
   NE; regret curve decreasing; Gini curve approaches the equilibrium line.
5. Accuracy ordering at the 3 profile classes consistent with V ordering in
   ≥ 8 of 10 populations, or an honest limitation paragraph.
6. main2.pdf rebuilds warning-clean with the new figures and the demoted
   LiRA appendix; no stale references.

## Failure modes and responses

- *BR dynamics mostly cycle:* switch the headline protocol to asynchronous
  random-order updates (often converges when fixed order cycles); if still
  cycling, report cycle-average outcomes and frame E2 (which needs no
  convergence claim) as the primary behavioural result.
- *No inequality gap (Gini equal):* check whether equilibria are
  boundary-pinned (D2); if genuinely absent, this falsifies the population
  prediction — report it, do not massage; the theory sections stand on their
  own claims.
- *Purchase100 loader/data unavailable:* fall back to Texas-100 (same
  literature role) or, worst case, run E1/E2 fully analytical (they do not
  strictly need a dataset — only B5/A1 do) and keep Adult for the accuracy
  check.
- *Oracle too slow at n = 20:* profile first (the PLD advantage is the cost);
  coarsen PLD discretisation to 1e-2 for the dynamics (T17/R2 measured the
  effect: max |dA| ≈ 1.2e-2) and verify the final equilibria at 1e-3.
