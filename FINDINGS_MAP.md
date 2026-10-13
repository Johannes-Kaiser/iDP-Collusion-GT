# Findings map — the Privacy Externality Game repository

This file maps every part of the repository to what came out of it. It has
four parts:

1. **Topic index**: one row per research question, with its trust level and
   the slide that presents it.
2. **Topics**: per question, the findings, the evidence, and the files.
3. **Directory index**: every file or folder, tagged with its topic.
4. **Contradictions register**: where the paper text disagrees with the
   code, caches, or `nummeric_eval.md`.

Sources of truth, in order: code (`game/`, `game_experiments/`), the
regenerated caches, and the campaign log `nummeric_eval.md`. The paper
(`paper/`) is provisional (CLAUDE.md Rule -1).

## Trust levels

| level | meaning |
|---|---|
| **A — proved / exact** | analytic proof, or exact accounting (closed form, certified quadrature, exact ladder) with quantified error well below the effect size |
| **B — solid numerics** | post-solver-fix (≥ 2026-07-13) cache, deployed RDP accountant, K=200 grid; the structure is robust, but individual grid points and counts are sampling artefacts |
| **C — provisional** | pre-fix cache, coarse grid (K ≤ 22), sample-resolution only, single configuration, or a pending re-run |
| **D — superseded / refuted** | kept only as an audit trail; do not cite |

The solver fix is the hard boundary between levels. On 2026-07-13 an early
break in the σ-bisection was fixed (`game/mechanism.py::solve_mechanism`).
Before the fix, σ was off by up to ~10% on some cells. Every cache built
before that date is level C at best.

---

## 1. Topic index

| # | topic | headline finding | trust | slide |
|---|---|---|---|---|
| T0 | Model & mechanism infrastructure | game = sampling-based iDP solve + PLD advantage + V(σ) | A/B | 1 |
| T1 | The externality | tightening any budget raises everyone else's A_PLD | A (small p) / B (full grid) | 2 |
| T2 | Equilibrium structure (deployed game) | equilibria are asymmetric corner equilibria on the participation frontier | B | 3 |
| T3 | Exact-accounting theory | budgets buy participation before model quality → corner play; symmetric only when a constraint pins both | A | 6 |
| T4 | Symmetric moderation & composition | exactly one symmetric NE, only at composition scale (exact ε* ≈ 2.59 at T = 390, locally stable; deployed grid shows it at 2.96) | A | 4 |
| T5 | Deployed-accountant artefacts | 89-order RDP sawtooth (~1e-4) creates and destroys near-tie NE and cycles | B | 4 (trust box) |
| T6 | Common weight sweep | λ shifts the NE cloud but restores no symmetry | B (λ=2 conditional) | 7 |
| T7 | Heterogeneity & self-sorting | the less privacy-sensitive player carries the batch (group size does **not** sort post-fix) | A (cond. S) / B | 8 |
| T8 | Dynamics, commitment, selection | first mover selects the outcome; leader commits tight | B | 10 (notes) |
| T9 | Welfare | planner optimum asymmetric; selfish play ~9% below it, never compensates the carrier | B | 10 |
| T10 | n players & FL populations | sorting survives n=3–10 (post-fix τ recompute); at n=20 everyone free-rides at the floor | C | 11 |
| T11 | Mechanism specificity | sensitivity-based iDP has a single symmetric NE | B | 9 |
| T12 | Attacks | coalition attack = κ→∞ coalition play; collapses 18–51× vs aware target | B | 12 |
| T13 | Measured adversary | LiRA at floor → externality invisible → NE at accuracy corner | C | 13 |
| T14 | Mitigation & mechanism design | Δ-iDP caps the movable risk; budget reports not DSIC; only convex pricing stabilises moderation | A / C | 14 |
| T15 | Robustness of modelling choices | V-shape (post-fix: frontier invariant, diagonal point moves), δ, PLD discretisation, floor; linear cost owns corner play | B | 5 |
| T16 | Existence & potential structure | NE exists on small-p substitutes box; PEG is not a potential game | A (scoped) | — |
| T17 | Gaussian-DP limit model | potential game, saddle, T_max — **demoted** to T→∞ regime | D for paper | — |
| T18 | Predecessor paper (Kaiser et al. SaTML 2026) | budget-manipulation attack, 62% of targets | external | 1 |
| T19 | Paper, plans, trackers, infrastructure | — | — | — |

---

## 2. Topics

### T0 — Model & mechanism infrastructure

**Game:** u_i = V(σ) − λ_i·A_PLD,i, with V(σ) = 1/(1+σ²) and action
ε_i ∈ [0.5, 64]. The mechanism (Boenisch et al.) sets one shared σ and a
per-group Poisson rate p_i: every (ε_i, δ) budget binds, and the batch is
fixed, Σ|G_i|p_i = B. A profile is feasible iff every p_i ≥ p̄ = 1e-5.

**Canonical instance:** |G_1| = |G_2| = 5000, B = 128, 5 epochs → T = 390
steps, δ = 1e-12, K = 200 log grid, λ = 1.

| role | files |
|---|---|
| mechanism solver (RDP accountant, σ bisection, fixed 2026-07-13) | `game/mechanism.py` |
| sensitivity-based ("Scale") solver | `game/mechanism_scale.py` |
| level-2 advantage A_PLD via dp_accounting PLD | `game/advantage.py` |
| utility, V(σ), aware vs naive player | `game/utility.py` |
| game evaluator, NE finder, BR dynamics, coalition Stackelberg | `game/game.py` |
| grid sweeps, pure-NE enumeration, BR curves | `game/continuous.py` |
| fictitious play, coalition sweep | `game/solvers.py` |
| repeated Bayesian game | `game/bayesian.py` |
| plotting style (all paper figures) | `game/style.py` |
| smoke test | `game/_smoke.py` |
| canonical K=200 mechanism grid (σ, A, feasibility, analytic frontier) | `game_experiments/run_locus.py` → `_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` (`slurm_scripts/run_locus_K200.sh`) |
| analytic participation-frontier mask (exact) | `game_experiments/patch_grid_analytic_feas.py` (`run_patch_grid_feasA.sh`); pre-patch backup `…pkl.pre-analytic-feas.bak` |
| solver tolerance checks | `game_experiments/_verify_tol.py` (`run_tolcheck.sh`) |
| exact analytic model of u_2(ε_2 \| ε_1) (reference derivation) | `tasks_findings/33_u2_analytic_model.md` |

**Trust:** A for the model definition. B for the K=200 grid (post-fix,
batch rtol 1e-4, utility noise ~1e-6). The 89-order RDP accountant is
part of the deployed game's definition; see T5.

---

### T1 — The externality: others' budgets set your realised risk

**Findings**
- **Lemma (A):** ∂σ/∂ε_j < 0 and ∂p_i/∂ε_j < 0. When j tightens, σ rises
  and i is sampled more.
- **Small-p corollary (A):** A_PLD,i ≈ I·p_i·Adv_G(σ). Along the budget
  line p_i ∝ σ this gives ∂A_PLD,i/∂ε_j < 0 (sympy-verified).
- **Full region (B):** the post-fix K=200 grid has **0 sign violations**
  at unit stride. Of 25,612 interior cross differences, 25,291 are
  negative and 321 are exact ties at the floor. All 25,612 own-direction
  differences are positive (recomputed 2026-10-12 from the cache).
  - The analytic full-region closure via a Chernoff envelope (T23) does
    **not** close. The theorem was downgraded to a proposition
    (`.tasks2/14`).
  - The old 72–99% certificate (`12_dade_certificate.txt`, 2026-07-03)
    is pre-fix and superseded.
- **Same contract, different risk (B):** at fixed ε_i = 4,
  A_PLD,i ∈ [0.0009, 0.174] depending only on the opponent. At
  ε_i = 16 the range is [0.065, 0.41]. Slack is monotone in 162/162 and
  137/137 grid pairs (`11_slack_summary.txt`, 2026-07-14).
- **Share effect (B, Fig. 15 = `fig_C5_surface`):** target fixed at
  ε_T = 4: A runs from 0.20 (others at 0.5) to ≈0 (others at 41). At
  ε_T = 16 it runs from 0.55 to 0.03.
  - Curves for different shares cross at ε_O ≈ ε_T. Below the crossing,
    a larger foreign share amplifies the externality; above it, it
    suppresses it.
- **Externality is higher-order (A):** at leading (RDP-2) order a move
  of ε_j keeps i's privacy content fixed. The effect lives in the
  profile shape, so it shows up as slack, not as a contract violation.

**Files**
- Paper findings docs:
  - `tasks_findings/20_small_p_corollary.md` (+ `_scripts/20_check.py`)
  - `23_chernoff_full_region_monotonicity.md` (+ `23_check.py`) — level D: the argument does not close
  - `04_slack_propositions.md` (+ `04_slack_numerics.py`)
  - `11_slack_numerical_panel.md` (+ `11_slack_panel.py` → `fig_slack_panel`, `_cache_v2/slack_panel.pkl`)
  - `12_da_de_numerical_certificate.md` (+ `12_dade_certificate.py/.txt`) — pre-fix
- Share sweep: `game_experiments/run_continuous.py` (experiment C5) →
  `_cache_v2/C5.pkl` → `paper/figures/fig_C5_surface`
- Open items:
  - `.tasks2/05` (slack violations; resolved by the fix — post-fix 0 violations)
  - `.tasks2/14` (full-region analytic closure)
- Predecessor evidence: `notebook_scripts/03_3D_plot_budg_portion_adv.ipynb`,
  `figures/3d_plot_adv4.svg`

---

### T2 — Equilibrium structure of the deployed (RDP, K=200) game

**Findings**
- **43 pure NE on the K=200 grid (B):** 42 off-diagonal mirror pairs plus
  one diagonal grid NE at (2.965, 2.965).
- **The off-diagonal NE lie on the analytic participation frontier (B):**
  - Tight player: rate at p̄, budget 0.81–1.61, A ≈ 7e-4.
  - Loose player: 6.38–6.70× looser (5.19–10.79), A = 0.21–0.32.
  - Utilities: tight 0.43–0.59, loose 0.23–0.27. The protected seat is
    the better one.
- **They are corner equilibria, not points (A, exact FOC):**
  - At (1.12, 7.13) the loose player's derivative points into the wall
    (+9.7e-3); the tight player's derivative points away from it.
  - Two one-sided conditions hold on intervals, so the NE set is a
    **continuum segment**.
  - Grid counts are sampling density: 7 at K=22, 43 at K=200.
- **Diagonal trough (A in small p):** if σ²V'' + 2σV' ≥ 0 (σ ≥ √3), no
  interior diagonal profile is a pure NE.
  - Budgets are strategic substitutes: 81–95% of near-diagonal cross
    differences are negative.
  - BR dynamics leave the diagonal at the first reply in 197/200 runs.
- **Hawk–Dove (B):** restricted to a mirror pair, the game is
  anti-coordination. The tight role pays 0.50 and the exposed role 0.25;
  the mixed NE plays tight with probability 0.45.
  - 2×2 at (1.91, 3.35): (T,T) 0.248, (L,L) 0.340, (T,L) 0.341/0.256.
- **Regret landscape (B):** zero regret on the 43 NE, two low-regret
  frontier channels, and one flat diagonal plateau. Median regret 0.166;
  0.66% of feasible cells below 1e-3.
- **The diagonal grid NE at 2.965 is the deployed image of the exact
  symmetric NE (ε* ≈ 2.59, T4).** As a pure grid NE it is a 1e-4 near-tie
  that disappears at 16× resolution.

**Files**
- Pure NE, locus:
  - `game_experiments/run_locus.py` (C6) → `_cache_v2/C6_K200.pkl` → `fig_C6_locus_K200` (unused in paper)
  - `paper/tex/fig_C6_tikz.tex` built from `_scripts/25_fig_C6_tikz_data.py` and `32_fig_C6_landscape_data.py` → `paper/figures/data/c6*_*.dat`
- K=22 overview: `run_continuous.py` (C1) → `_cache_v2/C1.pkl` → `fig_C1_payoff`
- FOC landscape: `_scripts/31_foc_landscape.py` → `paper/figures/data/foc_landscape/{foc_landscape,br_smooth,u1/u2_conditional_heatmap}.png`, `ridge1.dat`, `valley1.dat`
- Regret / distance maps: `foc_landscape/{regret_heatmap,br_distance_heatmap}.png`
- Exact FOC (pointwise and full grid):
  - `game_experiments/exact_foc.py`
  - `exact_foc_grid.py` (`run_exact_foc_grid.sh`) → `_scripts/34_exact_foc_grid.pkl`, `t390_exact_foc.png`
- Grid-free equilibrium search:
  - `game_experiments/exact_equilibria.py` (`run_exact_eq*.sh`) → `29_exact_equilibria.pkl`
  - `exact_eq_refine.py`
  - `_verify_diag_ne.py` (`run_diagne_check.sh`)
- Diagonal curvature:
  - `game_experiments/diag_curvature_scan.py` (`run_diag_scan.sh`) → `_scripts/28_diag_curvature.pkl`
  - `_verify_sign_condition.py` (`run_signcond_check.sh`)
- Symmetry-breaking anatomy: `_scripts/26_symmetry_breaking.py` → `26_symmetry_summary.txt`, `fig_symmetry_breaking`
- Proposition draft: `tasks_findings/03_no_symmetric_NE_proposition.md` (+ `03_check.py`)
- Representative NE table: `game_experiments/post_analysis.py`, `post_v2.py` → `paper/tex/table_NE.tex`
- Figure polish: `_scripts/polish_C6_C7_C8.py`, `tasks_findings/polish_C6_C7_C8.md`
- BR-trace diagnostic figures:
  - `game_experiments/run_br_trace.py` (`run_br_trace*.sh`) → `paper/figures/fig_br_trace*.{pdf,png,svg}` and `paper/figures/data/br_trace*`
  - These are K=22/K=60 variants (λ, batch, no floor), built 2026-07-13 (borderline fix date): level C.
- Campaign log: `nummeric_eval.md` §§1–6, 10

---

### T3 — Exact-accounting theory: why the game is corner-driven

**Findings (all A, machine-precision or certified)**
- **T=1 closed form:** Adv = p·(2Φ(1/2σ) − 1) is exactly linear in the
  own rate and capped by it.
  - Flat marginal cost against declining marginal value gives corner
    best responses.
  - At λ = 1 privacy cannot compete (Adv ≤ 0.0256).
- **Two-phase slice anatomy** (every slice computed: T ∈ {1,2},
  |G| ∈ {500, 5000, 50000}, λ ∈ [0.001, 100]):
  - Rate phase: participation and exposure rise ~e^ε, so there is
    always a trough.
  - Noise phase: the rate is capped, so extra budget buys lower σ.
  - Best responses sit only at the floor, at the frontier, or inside the
    noise phase. Moderation is **noise-demand**, never partial
    participation.
- **λ bifurcation at T=1 (five regimes, no pure-NE gap):**
  1. symmetric ceiling (λ ≤ 0.03)
  2. isolated box-edge pair (47.14, 64) (0.05–0.09)
  3. frontier segments (0.1–26)
  4. floor pair (0.5, 9.77) (28–36)
  5. symmetric floor (≥ 40)
  - Symmetric play is never an interior NE at T = 1 or T = 2.
- **Composition (T=2 certified quadrature; T up to 390 PLD):**
  - Adv_T is mildly **concave** in p, which refutes the convexity
    hypothesis.
  - Adv/p grows 0.42 → 12.1 (σ = 0.9, T = 1 → 390).
  - The interior-BR λ window slides from 27–36 (T=1) to 8–25 (T=2).
- **Group size at fixed B acts like an effective privacy weight:**
  - |G| = 500: segment near the origin.
  - |G| = 5000: mid segment.
  - |G| = 50000: ceiling NE (64, 64).
- **Budget factorisation lemma:** δ_1 = p·h(r; σ) with
  r = (e^ε − 1 + p)/p, verified to 1.4e-11.
- **Interior tangency |V'| = λ·p_2·|c'(σ)| holds for any V,** validated
  in 3 worlds (1.1%, 0.3%, 6.9%).

**Files**
- T=1:
  - `game_experiments/single_step_game.py`
  - `single_step_grid.py` (`run_single_step_grid.sh`) → `35_single_step_grid.pkl`
  - `single_step_br.py` (`run_single_step_br.sh`, `run_t1_gsweep.sh`) → `36_*`, `37_*`, `40_*`, `41_*`
  - `single_step_lam_sweep.py` (`run_t1_lam_*.sh`) → `38_t1_lam_sweep.pkl`, `39_t1_lam_low.pkl` (component cache, reusable for any λ), `47_t1_lam_G500.pkl`
- T=2:
  - `two_step_game.py`
  - `two_step_br.py` (`run_t2_br.sh`) → `42_two_step_br.pkl`, `48_t2_br_lam126.pkl`
  - `_scripts/45_t2_br_figure.py` → `t2_br_ne.png`
- Slices and interior BR: `_scripts/46_slice_decompositions.py` (`46_slice_data/`) → `t1_slice_decomposition.png`, `interior_slice_decompositions.png`; `49_interior_br_curves.py` → `interior_br_curves.png`
- Group-size sweep: `_scripts/43_gsweep_endpoints.py`, `43_gsweep_figure.py` → `t1_gsweep_lam07.png`
- Curvature vs T: `_scripts/44_adv_curvature.py` → `44_adv_curvature_summary.txt`
- Maths checks: `_scripts/69_math_checks.py`, `70_math_fixes.py` → `.txt`
- Precision: `_scripts/63_mpmath_coverage.py` → `.txt` (T=2 vs 40-digit mpmath ≤ 1.1e-7)
- T=1 figures: `foc_landscape/t1_*.png`. Warning: `t1_lam_full_bifurcation.png` still shows a stale "no pure NE" band (level D annotation).
- Campaign log: `nummeric_eval.md` §§7–9, 12–15, 19, 25

---

### T4 — Symmetric moderation exists only at composition scale

**Findings**
- **T = 390, λ = 1, exact ladder (A, settled 2026-10-12):** the own-budget
  slice at the diagonal is a concave **peak** (dense exact slice, Slurm
  292170 → `83_t390_local_slice.pkl`).
  - BR(2.548) ≈ 2.62 and BR(2.92) ≈ 2.38, so the symmetric NE sits at
    **ε* ≈ 2.59** (σ ≈ 1.16).
  - BR slope ≈ −0.65: locally stable.
  - On the deployed 89-order grid it appears at 2.965 as a 1e-4 near-tie.
  - The earlier "2.548 with margins 1.2e-3 / 4.2e-3" sampled no deviation
    within ±40% of the diagonal.
- **Temptation hump (C: 31-point columns, same coarseness):** the best symmetric deviation gap rises from
  T = 1 (0.011 at λ = 1) to a peak at T ≈ 3–8 (0.024–0.030), then falls.
  - λ = 2 reaches the onset by T ≈ 128; λ = 1 just reaches it at
    T = 390; λ = 0.5 is still before the onset.
  - Rising edge: "hide" deviations save more exposure as composition
    compounds.
- **T ≤ 2:** symmetric stationary points exist, but a loosening deviation
  gains 0.05–0.10 for λ ∈ [2, 16] (A).
- **Linear pricing at any level cannot stabilise moderation.** Convex
  pricing (u = V − λ_q·A²) yields a symmetric interior NE already at
  T = 1 for λ_q ≳ 750 (C: sample resolution, not polished).
- **Stability of the T = 390 NE:** the exact game gives BR slope ≈ −0.65
  (stable). The limit model's |BR'| = 1.045 is wrong for the exact game.

**Files**
- Exact ladder:
  - `game_experiments/multi_step_game.py`
  - `run_t_grid.py` (`run_t_grid.sh`, `run_t_grid_hi.sh`, `run_t390*.sh`) → `50_t_grid.pkl`, `51_t_grid_hi.pkl`, `52_t390_ladder.pkl`, `52b_t390_fixedpoint.pkl`, `52c_t390_saddle.pkl`
- Verdicts: `_scripts/81_verdicts_0717.py`
- Certified brackets: `game_experiments/double_ladder.py`
- Tabulated exact grid (open):
  - `game_experiments/tab_ladder_grid.py` (`run_tab_ladder_*.sh`) → `_cache_v2/exact_grid_K200_T390_g5000_mesh{A,B}.pkl`, `tab_nodes_*.pkl`, `tab_compare_T390_K200.txt`
  - 14% classification disagreement between the two meshes: not usable for NE claims yet.
- Figures: `foc_landscape/onset_curve.png` (annotation stale: says "pre-onset"), `slides/assets/fig_t390_moderation.png`
- Campaign log: `nummeric_eval.md` §§14, 16–18, 20.1

---

### T5 — Deployed-accountant artefacts (two games under one name)

**Findings (B)**
- The 89-order RDP accountant adds a ~1e-4 payoff sawtooth at
  order switches.
  - Switching to 761 orders shifts payoffs by up to 7.9e-4 (median
    1.3e-4).
  - Teeth shrink 3× and multiply (52 → 119).
- **λ = 2 "no pure NE" / undercutting cycle is accountant-conditional.**
  - With 89 orders: 3-cycle at K = 200, 2-cycle at 16× zoom, plus a
    phantom NE pair with margin ~5e-6.
  - With 761 orders: the cycles collapse onto 4 pure NE clustered at
    ε ∈ [1.025, 1.029], two of them symmetric.
- **At λ = 1 the 16× zoom removes every pure NE in the diagonal window**,
  including the K = 200 grid NE at 2.965. Orbiting persists (a 4-state
  loop).
- Order-grid slack: budgets shrink 0.2–0.4% near ε ≈ 1–3 and up to 9.1%
  at high budgets.
- The dense-order recompute of the full K = 200 grid (job 180899) never
  finished: no `_dense.pkl` exists.

**Files**
- `game_experiments/run_cycle_zoom.py` (`run_cycle_zoom*.sh`, `run_cycle1_zoom.sh`) → `55_cycle_zoom.pkl`, `56_cycle_zoom_dense.pkl`, `57_cycle1_zoom.pkl`
- `game_experiments/run_mech_dense.py` (`run_mech_dense.sh`, `out/peg_mech_dense-180899.*`)
- `_scripts/64_zoom_br_fig.py`, `65_zoom_compare_fig.py` → `zoom_br_lam2*.png`
- `_scripts/54_cycle_trajectory.py` → `cycle_trajectory.png`
- Campaign log: `nummeric_eval.md` §§19 (row 4b), 20.2–20.5

---

### T6 — Common privacy weight shifts the cloud, not the symmetry (Fig. 16)

**Findings (B, K = 200, `_cache_v2/C7_K200.pkl` of 2026-07-14)**

| λ | pure NE | diagonal NE |
|---|---|---|
| 0.125 | 44 | none (the old (26.6, 26.6) solver artefact is gone post-fix → `.tasks2/04` resolved) |
| 0.25 | 62 | none |
| 0.5 | 34 | none |
| 1 | 43 | (2.965, 2.965) |
| 2 | 0 | — (accountant-conditional, T5) |
| 4 | 1 | (0.5, 0.5) corner |

- The cloud moves toward lower budgets as λ grows.
- PoEV: at λ ≤ 0.25 every NE inflates aggregate advantage (up to +165%);
  at λ = 1 only 14% do; the λ = 4 corner deflates it by 82%. These come
  from the paper / `10_poev_table.py` (pre-fix run, re-run status
  unclear): C.

**Files**
- `game_experiments/run_locus.py` (C7, `run_locus_C7.sh`) → `fig_C7_lambda_ratio`
- `tasks_findings/10_numerical_poev_table.md`, `_scripts/10_poev_table.py`, `_out/10/`

---

### T7 — Heterogeneity sorts the roles

**Findings**
- **Self-sorting proposition (A, under exchange condition S):** λ_1 > λ_2
  ⇒ ε_1* ≤ ε_2* at every pure NE.
- **λ-ratio sweep (B, `C8_K200.pkl`):** 170/174 heterogeneous NE give the
  lower budget to the more sensitive player. All 4 exceptions are at
  λ_1/λ_2 = 0.5.
- **Group size (D for sorting, 9-point grid, `C2.pkl`):** the post-fix
  cache does **not** show minority sorting.
  - For |G₁|/|G₂| < 1 the NE come in both orientations (mirror pairs).
  - For ratio ≥ 2 only symmetric NE remain.
  - This falsifies the paper's claim (PAPER_AUDIT F3).
- **Carrier rule (A at fixed σ, linear cost, T=1; direction B at
  T = 390):** the batch goes to argmin λ_i/|G_i|.

**Files**
- `game_experiments/run_locus.py` (C8) → `fig_C8_asymmetric_lambda`
- `run_continuous.py` (C2) → `fig_C2_asymmetric`
- `tasks_findings/07_2player_self_sorting.md` (+ `07_check.py`)
- `.tasks2/12` (group-size grid refinement)

---

### T8 — Dynamics, commitment, equilibrium selection

**Findings (B, alternating BR on the K = 200 cache)**
- **Basins depend only on the first mover's budget:** commitment selects
  the outcome.
  - λ = 1: 45 attractors. The wall pair (1.61, 10.79) captures 22.9% of
    starts, diagonal cycles 33.5%, the symmetric grid NE 1.6%.
  - λ = 2: one globally attracting cycle under 89 orders, which is
    accountant-conditional (T5).
- **Stackelberg at λ = 1:** the leader commits *tight* (1.61) and gets
  0.591 versus the follower's 0.267.
- Protocol robustness: simultaneous and damped dynamics were rerun
  (`62_protocols.py`). The first-mover result is specific to the
  alternating protocol.
- Rejected concepts: repeated-game collusion (budgets are one-shot
  contracts), and spiteful play (it destroys V).

**Files**
- `_scripts/57_basins.py`, `58_basin_outcome.py` → `basins.png`
- `_scripts/59_alt_plays.py` → `59_alt_plays_summary.txt`
- `_scripts/62_protocols.py`
- Campaign log: `nummeric_eval.md` §§11, 21, 22

---

### T9 — Welfare

**Findings**
- **Team optimum at λ = 1 is asymmetric (B):** (3.35, 24.13), W = 0.941.
  - Best equal-budget point: 0.885 (a ~6% fairness premium).
  - Best NE: 0.859.
  - The reason is concavity of the composed advantage in the rate.
- **Heterogeneity r = λ_2/λ_1 ∈ {1, 2, 4, 8} (B):**
  - Planner: 0.941 → 0.935.
  - Best NE: 0.859 → 0.853, i.e. about 9% below the planner at every r.
  - Enforced equal budgets: 0.885 → **−0.058**.
  - No equilibrium compensates the carrier.
- **Cycling burns welfare:** at λ = 2 (89-order accountant) welfare is
  0.180 of an achievable 0.533 (conditional, T5).
- **PoEV at λ = 1** vs uniform ε = 5.73 (ΣA = 0.310): range
  [−0.33, +0.05], median −0.18 (B for the post-fix C6 NE; table in
  `paper/tex/table_NE.tex`).
- **Roughgarden smoothness (C, pre-fix):** closes for w ≲ 1 and fails for
  w ≥ 2, because welfare turns negative.

**Files**
- `_scripts/59_alt_plays.py`
- `tasks_findings/22_roughgarden_smoothness_poa.md` (+ `22_check.py`)
- `10_numerical_poev_table.md`
- `.tasks2/08`
- `slides/assets/fig_welfare.png`

---

### T10 — More players: sorting survives, populations free-ride

**Findings**
- **n = 3–5, K = 7, exhaustive NE (C, coarse):** with heterogeneous
  λ ∈ geomspace(0.25, 4), budget order is the exact reverse of λ order
  in 100% of NE.
  - A ranges 0.001–0.33; Gini 0.40–0.48.
  - With symmetric λ there is one symmetric NE; the grid is too coarse to
    resolve the band (`.tasks2/06`).
- **n ≥ 6 (fictitious play):** Kendall τ recomputed from the raw FP
  histories (`82_paper_audit.py`) gives **100% sorting at every n ≤ 10**.
  - The table's 0.00 is a clustering bug in `run_nplayer.py`.
  - The paper's 83% at n = 10 is pre-fix.
  - With symmetric weights, every n converges to the grid floor
    (1, …, 1).
- **E1, n = 20 FL consortia** (100 sampled populations, |G| ~ U{2000…12000},
  w ~ LogNormal(0, 0.7), 6-action grid {10…50}; C: pre-fix, 2026-07-03):
  - **100/100 converge to the uniform floor profile (10, …, 10).**
  - Free-riding dominates, because one player is ~1/20 of the batch.
  - The critical weight is w* ≈ dV/dA ≈ 0.02–0.04.
  - Equilibrium welfare is on average **31% below** an approximate
    sorted planner (planner Gini 0.58).
  - The critical-scale run (w ~ LogNormal(ln 0.03, 0.7)) was never
    analysed.
- **E2 (adaptive myopic agents):** one population cached, no analysis.

**Files**
- `game_experiments/run_nplayer.py` (`run_fp_nplayer*.sh`) → `_cache_v2/H3_nplayer.pkl`, `H3_nplayer_table.tex`
- `_scripts/19_h3_plot.py` → `fig_h3_fp_nplayer`, `19_h3_summary.txt`
- `tasks_findings/19_h3_fictitious_play_nplayer.md`
- `game_experiments/flpop_common.py`, `flpop_a4_scan.py`, `flpop_checks.py`, `run_flpop_e1.py`, `analyse_E1.py`, `replot_E1.py`
- `run_flpop_e2.py`, `_flpop_e2_oracle.py`, `_flpop_e2_figs.py` (`run_flpop_*.sh`)
- Caches: `_cache_flpop/` (100 `E1_pop*.json`, parcache, `populations.csv`), `_cache_flpop_e2/`
- `tasks_findings/28_flpop_e1.md`, `.tasks2/13`
- `slides/assets/fig_e1_population.png`
- Logs: `out/flpop_*`

---

### T11 — The asymmetry is a property of the sampling-based mechanism

**Findings (B, K = 200, `_cache_scale/` of 2026-07-14)**
- Sensitivity-based iDP (personal clipping norms, shared σ): A_i depends
  on ε_i alone.
- Result: **exactly one pure NE, symmetric at (5.73, 5.73)**, and a
  single diagonal NE for λ ∈ {0.25, 0.5, 1}.
- The sampling-based game on the same grid has 43 NE, 42 of them
  asymmetric.

**Files**
- `game/mechanism_scale.py`
- `game_experiments/run_scale.py` (`run_h2_scale.sh`) → `_cache_scale/{mech_grid_K200…, C1_K200, C7_K200, C8_K200}.pkl`
- `_scripts/h2_scale_hero.py` → `fig_h2_scale`
- `tasks_findings/18_h2_scale_mechanism.md`
- `slides/assets/fig_scale_ne.png`

---

### T12 — Attacks are coordinated rational play

**Findings**
- **Corollary (A):** as κ → ∞ the coalition plays one shared budget at
  the participation-preserving minimum, and A_t is non-decreasing in k.
- **Passive target (B, `C4.pkl`):** n = 5, others at ε = 32, 14-point
  grid. A_t = 0.443 → 0.479 / 0.534 / 0.608 / **0.729** for k = 1–4
  (+64%). The coalition plays ε_C = 6.8, then 3.2 at k = 4.
- **Aware target (B, `H4.pkl`):** the target best-responds on a 14-point
  grid. A_t = 0.014 / 0.010 / 0.018 / 0.041, i.e. **18–51× lower** than
  passive.
  - At k = 4 the attacker's argmax is a poisoning corner (ε_C = 0.5,
    σ = 2.40, target utility 0.108).
- **κ-Pareto (C):** the frontier collapses (co-monotone objectives);
  K = 14.
- **Repeated Bayesian attacker (C, pre-fix):**
  - The oracle attack is interior: ε_M* ≈ 3.2, A_H ≈ 0.454.
  - Regret R(60) = 1.31 ± 0.26 with a correct prior, versus 8.3
    (uniform), 8.4 (misspecified) and 13.2 (EXP3).

**Files**
- `game_experiments/run_continuous.py` (C4) → `fig_C4_coalition`
- `run_h4.py` (`run_h4_aware_target.sh`) → `fig_h4_aware_target` (also `_scripts/13_h4_plot.py`)
- `run_lambda_pareto.py` (`run_lambda_pareto.sh`) → `_cache_v2/lambda_pareto.pkl`, `fig_lambda_pareto`
- `game/bayesian.py`, `run_bayesian.py` → `fig_bayesian`
- `_scripts/09_check.py`, `16_regret_plot.py` → `fig_h8_brpeg_regret`, `16_regret_summary.txt`
- Findings docs: `tasks_findings/09_*`, `13_*`, `14_*`, `16_*`
- `.tasks2/09`

---

### T13 — A measured adversary sits at the floor

**Findings (C)**
- **Setup:** Adult, MLP, 5000 records per silo, 5×5 grid
  ε ∈ {2, 4, 8, 16, 32}, 128 DP-SGD shadow models per profile, LiRA.
- **Result:** A_LiRA ∈ [0.107, 0.115] (AUC ≤ 0.506), while
  A_PLD ∈ [0.004, 0.870].
  - The externality term is flat, so the measured NE sit at the accuracy
    corner ((16,16), (32,32)). The analytical NE on the same grid is
    (2, 2).
- **σ audit against the fixed solver (`audit_h1_sigma.py`):**
  - σ deviation ≤ 2.6%.
  - Rate deviation up to 18.5% at (4, 32) / (32, 4).
  - The floor result is unlikely to move, but it has not been retrained.
- **n = 3 federated companion (T25):** never run. The figure
  `fig_C9_federated_measurement_ceiling` is **synthetic** (D: do not
  cite).
- **Advantage hierarchy A_{ε,δ} ≥ A_PLD ≳ A_LiRA (A):** shown on six
  mechanisms through one contract point.

**Files**
- `experiment_scripts/mia/04_mia_pergroup.py`
- `_scripts/24_lira_analysis.py`, `24_paper_figure.py`, `24_mock_figure.py` (mock)
- `slurm_scripts/run_lira_n2_shadow.sh`
- `_cache_H1/`
- `game_experiments/audit_h1_sigma.py` (`run_audit_h1_sigma.sh`, `out/peg_audit_h1-179003.out`)
- T25: `25_toy_federated_analysis.py`, `25_paper_figure.py`, `25_mock_figure.py`, `_cache_T25/`
- Hierarchy: `_scripts/27_advantage_hierarchy.py`, `28_hierarchy_tikz_data.py`, `28b_*` → `paper/tex/fig_hierarchy_*.tex`, `paper/figures/data/hier*.dat`
- Findings docs: `tasks_findings/24_*`, `25_*`
- `.tasks2/01` (above-floor instance: the main missing experiment), `.tasks2/02`

---

### T14 — Mitigation and mechanism design

**Findings**
- **Δ-neutralisation (A, elementary):** if slack_i ≤ Δ, then
  A_{ε_i,δ} − Δ ≤ A_PLD,i ≤ A_{ε_i,δ}. No coalition can move i's risk by
  more than Δ at fixed ε_i.
  - Whether the restricted game is unique or symmetric, and what it
    costs, is open.
- **Non-strategyproof (A + C witness):** the best report depends on the
  opponents (witness 2.92 vs 1.21, pre-fix). Large-market and
  truthful-elicitation rescues do not apply.
- **Convex vs linear pricing (T4):** only the *shape* of the exposure
  price stabilises moderation.
- **Awareness defeats the coalition attack (T12).**

**Files**
- `tasks_findings/04_slack_propositions.md`
- `08_non_strategyproof_myerson_sketch.md` (+ `08_beneficial_misreport.py`)
- `paper/tex/07_mitigation.tex`
- Predecessor Δ bound: `notebook_scripts/04_plot_Delta_bound.ipynb`, `figures/Delta_bound_corrected.svg`

---

### T15 — Robustness of modelling choices

**Findings**
- **V-shape (B, recomputed post-fix, paper Fig. 14 is stale):** six V
  shapes give 27–45 NE.
  - The frontier strings coincide across all V (V1→V distance
    0.006–0.30 decades).
  - Five of six shapes have exactly one diagonal grid NE whose location
    moves with V (1.36–4.17); V = −σ has none.
  - The paper's "100% off-diagonal" is falsified (PAPER_AUDIT F1).
  - Figure: `slides/assets/fig_v_shape_postfix.png`.
- **R1–R5 (B partial):**
  - Participation floor 1e-3 / 1e-5 / 1e-7 keeps all NE off-diagonal.
  - PLD discretisation: ΔA ≤ 4.5e-3.
  - δ, batch/epochs and RDP orders were never run (`.tasks2/03`).
- **Holistic invariance matrix (191 two-player worlds, T ∈ {1, 2}):**
  - Corner play holds in 107/110 linear-cost worlds; cost shape owns it.
  - The trough exists where privacy binds inside the box.
  - The specialisation premium needs heterogeneity, small groups, or
    composition.
  - The "stable moderation" signal was downgraded (screening artefact).
- **Error budget:**
  - T=1: 1e-12.
  - T=2: ≤ 1.1e-7.
  - Ladder budgets: ~1e-4 relative.
  - Certified advantages: ±4.3e-4 at T = 390.

**Files**
- `_scripts/05_V_invariance_worked_example.py`, `15_v_shape.py` → `fig_h5_v_shape`, `15_v_shape_summary.txt`
- `tasks_findings/05_*`, `15_*`
- `_scripts/17_r1_r5.py` (`run_robustness.sh`) → `17_r1_r5_summary.*`, `_cache_v2/R3_rates_K50_stride4.pkl`, `fig_r3_participation`
- `tasks_findings/17_*`
- `game_experiments/robust_game.py`, `run_robustness.py` (`run_holistic.sh`) → `60_robustness.pkl`
- `_scripts/66_invariance.py`, `67_invariance_figs.py` → `invariance_heatmap.png`, `invariance_dials.png`
- `_scripts/68_world_notes.md`
- Campaign log: `nummeric_eval.md` §§19, 23, 24

---

### T16 — Existence and potential structure

**Findings**
- **Existence (A, scoped):** a pure NE exists on any small-p box where
  budgets are strategic substitutes (Topkis / Milgrom–Roberts).
  - About 26% of interior cells fail increasing differences, so there is
    no full-region proof (`.tasks2/07`).
- **No potential (A for exact/weighted; B for ordinal):** the
  mixed-partial violation holds (median 135% mismatch on K = 200).
  - The paper's K = 22 ordinal 4-cycle no longer exists post-fix.
  - The replacement witness is a 6-step improvement cycle on K = 200 at
    λ = 1 (gains 2.4e-4 – 1.5e-3), see PAPER_AUDIT F7.

**Files**
- `tasks_findings/21_supermodular_ne_existence.md` (+ `21_check.py`)
- `06_non_potential_game_proposition.md`
- `_scripts/t06_symbolic_mixed_partial.py`, `t06_numerical_cross_check.py`, `t06_improvement_cycle.py`

---

### T17 — Gaussian-DP limit model (demoted 2026-07-16)

The μ-GDP CLT limit is an exact potential game with a symmetric saddle,
BR-instability |BR'| = 1.045, "no cycling", and a T_max.

It is valid only where per-step losses are small (σ ≳ 1). It is 5–1800×
off on the frontier, and 0% of deployed NE fall inside its near-NE band.
**Do not cite as statements about the deployed game.**

**Files**
- `game_experiments/gdp_limit_game.py` → `53_gdp_gap.pkl`
- `_scripts/stab.py`, `stab2.py`, `stab3.py`, `72_potential_landscapes.py`, `74_balanced_path.py`, `75_ridge_engineering.py`, `76_fixed_landscapes.py`, `77_ridge_arc.py`, `78_hybrid_band_game.py`, `79_analytic_walls.py`, `80_hybrid_near_ne.py`
- `foc_landscape/{potential_landscapes,balanced_path,ridge_engineering,fixed_landscapes,ridge_arc,hybrid_band_game}.png`
- Campaign log: `nummeric_eval.md` §§18, 22.4–22.5, 26–27

---

### T18 — Predecessor paper (Kaiser et al., SaTML 2026, `2601.12922v1.pdf`)

**Findings:** in sampling-based iDP, risk depends on others' budgets.
- A budget-manipulation attack inflates the MIA susceptibility of 62% of
  targeted records while every contract holds.
- (ε, δ, Δ)-iDP was proposed as the fix.

**Files**
- `Readme.md`
- `experiment_scripts/mia/` (`01_budget_control_adv.py`, `02_eval_mia_across_runs_clean.py`, `03_eval_mia_post.py`, `04_mia.py`, `utils/`, `exp_yaml*/`, notebooks)
- `notebook_scripts/00–06_*.ipynb`
- `figures/*.svg`
- `slurm_scripts/run_budget_adv*.sh`, `run_mia*.sh`
- `data/` (datasets)
- `opacus_new/` (vendored Opacus with per-sample rates)
- `differential-privacy/` (vendored Google DP library / dp_accounting)
- `1712.00270v3.pdf` (reference)

---

### T19 — Paper, plans, trackers, infrastructure

- **Paper:** `paper/main.tex`, `paper/tex/*.tex`, `paper/main.pdf`.
  - `paper/figures/` holds all figures (the `fig_E*` set is from the
    first, pre-continuous `run_all.py` → `_cache/E1–E6.pkl`, level D).
  - `paper/history/` holds the old paper (`main_old.*`, `tex_old/`,
    checkpoint PDF).
- **Campaign log:** `nummeric_eval.md` (sections, artefact index, open
  items).
- **Plans and reviews (2026-06-30 / 07-01, historical):**
  `COLLUSION_GAME_PLAN.md`, `claude_nodes.md`, `NOVELTY_REVIEW.md`,
  `NOVELTY_ACTION_PLAN.md`, `NOVELTY_GAME_PLAN.md`,
  `EXPERIMENT_IDEAS.md`, `EXTENSION_DIRECTIONS.md`,
  `HERO_FIGURE_BRIEF.md`.
- **Paper-rewrite drafts:** `tasks_findings/01_foreground_c1_narrative.md`,
  `02_corollary_demotion_bundle.md`.
- **Open task tracker:** `.tasks2/01–15`. `.tasks2/11` is the audit of
  errors in the old paper; `.tasks2/15` covers the A_PLD axis-label
  regeneration.
- **Voice guide for paper prose:** `sources/voice/` (`voice.md`,
  `MANIFEST.md`, 20 reference PDFs).
- **Compute:**
  - `slurm_scripts/*.sh` (mapped per topic above)
  - `out/` (Slurm logs; latest: 180594, 180626/7 landed, 180899 unfinished)
  - `.venv/`, `requirements.txt`
- **Slides:** `slides/build_deck.py`, `slides/make_figs.py`,
  `slides/assets/`, `slides/peg_findings.pptx`.

---

## 3. Directory index (file → topic)

### `game/`
| file | topic |
|---|---|
| `mechanism.py` | T0 |
| `mechanism_scale.py` | T11 |
| `advantage.py` | T0 |
| `utility.py` | T0 |
| `game.py` | T0, T12 |
| `continuous.py` | T0, T2 |
| `solvers.py` | T10, T12 |
| `bayesian.py` | T12 |
| `style.py` | T19 |
| `_smoke.py` | T0 |

### `game_experiments/`
| file | topic |
|---|---|
| `run_continuous.py` (C1, C2, C4, C5) | T2, T7, T12, T1 |
| `run_locus.py` (C6, C7, C8) | T2, T6, T7 |
| `patch_grid_analytic_feas.py` | T0 |
| `run_h4.py` | T12 |
| `run_lambda_pareto.py` | T12 |
| `run_bayesian.py` | T12 |
| `run_scale.py` | T11 |
| `run_nplayer.py` | T10 |
| `run_br_trace.py` | T2, T8 |
| `run_all.py`, `post_analysis.py`, `post_v2.py` | T19 (first-generation pipeline), T9 |
| `exact_equilibria.py`, `exact_eq_refine.py`, `_verify_diag_ne.py` | T2, T4 |
| `exact_foc.py`, `exact_foc_grid.py` | T2 |
| `diag_curvature_scan.py`, `_verify_sign_condition.py` | T2 |
| `_verify_tol.py` | T0 |
| `single_step_game.py`, `single_step_grid.py`, `single_step_br.py`, `single_step_lam_sweep.py` | T3 |
| `two_step_game.py`, `two_step_br.py` | T3 |
| `multi_step_game.py`, `run_t_grid.py`, `double_ladder.py`, `tab_ladder_grid.py` | T4 |
| `gdp_limit_game.py` | T17 |
| `run_cycle_zoom.py`, `run_mech_dense.py` | T5 |
| `robust_game.py`, `run_robustness.py` | T15 |
| `audit_h1_sigma.py` | T13 |
| `flpop_common.py`, `flpop_a4_scan.py`, `flpop_checks.py`, `run_flpop_e1.py`, `analyse_E1.py`, `replot_E1.py` | T10 |
| `run_flpop_e2.py`, `_flpop_e2_oracle.py`, `_flpop_e2_figs.py` | T10 |
| `_cache_v2/` | C1/C2/C4/C5 (T2, T7, T12, T1), C6/C7/C8 (T2, T6, T7), H3 (T10), H4 (T12), slack (T1), R3 (T15), λ-Pareto (T12), exact grid (T4) |
| `_cache_scale/` | T11 |
| `_cache_H1/` | T13 |
| `_cache_T25/` | T13 (D) |
| `_cache_flpop/`, `_cache_flpop_e2/` | T10 |
| `_cache/E1–E6.pkl` | T19 (D) |

### `tasks_findings/` (findings documents)
| file | topic | trust |
|---|---|---|
| 01, 02 | T19 | — |
| 03 | T2 | A/C |
| 04 | T1, T14 | A |
| 05 | T15 | A |
| 06 | T16 | A |
| 07 | T7 | A |
| 08 | T14 | A/C |
| 09 | T12 | C |
| 10 | T6, T9 | C |
| 11 | T1 | B |
| 12 | T1 | D (pre-fix) |
| 13 | T12 | B (re-run) |
| 14 | T12 | C |
| 15 | T15 | C |
| 16 | T12 | C |
| 17 | T15 | B |
| 18 | T11 | B |
| 19 | T10 | C |
| 20 | T1 | A |
| 21 | T16 | A |
| 22 | T9 | C |
| 23 | T1 | D |
| 24 | T13 | C |
| 25 | T13 | D |
| 28_flpop_e1 | T10 | C |
| 33_u2_analytic_model | T0, T3 | A |
| polish_C6_C7_C8 | T2 | — |

### `tasks_findings/_scripts/` (by number)
| files | topic |
|---|---|
| 03, 26 | T2 |
| 04, 11, 12, 20, 23 | T1 |
| 05, 15, 17, 63, 66–70 | T15 (63, 69, 70 also T3) |
| 06 (`t06_*`), 21 | T16 |
| 07 | T7 |
| 08 | T14 |
| 09, 13, 16 | T12 |
| 10, 22 | T9 |
| 19 | T10 |
| 24, 25, 27, 28 | T13 |
| 25_fig_C6, 31, 32, 28_diag_curvature, 29, 34 | T2 |
| 35–49 | T3 |
| 50–52c, 81 | T4 |
| 53, 72–80, stab* | T17 |
| 54, 55–57, 64, 65 | T5 |
| 57–59, 62 | T8, T9 |
| 60, 66, 67, 68 | T15 |
| h2_scale_hero | T11 |
| polish_C6_C7_C8 | T2 |

### Other folders
| path | topic |
|---|---|
| `paper/` | T19 |
| `paper/figures/data/foc_landscape/` | T2–T5, T8, T15, T17 |
| `slurm_scripts/` | see per-topic file lists |
| `out/` | T19 |
| `.tasks2/` | 01, 02 → T13; 03 → T15; 04, 05 → T1/T6; 06 → T10; 07 → T16; 08 → T9; 09 → T12; 10, 15 → T19; 11 → T19; 12 → T7; 13 → T10; 14 → T1 |
| `experiment_scripts/`, `notebook_scripts/`, `figures/`, `data/`, `opacus_new/`, `differential-privacy/`, `Readme.md` | T18 |
| `sources/voice/` | T19 |

---

## 4. Contradictions register

The full claim-by-claim audit of the paper is in [`PAPER_AUDIT.md`](PAPER_AUDIT.md)
(10 falsified, 4 scoped, 3 stale, 4 unverified; script
`tasks_findings/_scripts/82_paper_audit.py`).
