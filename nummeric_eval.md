# Numerical evaluation campaign: exact equilibrium structure of the PEG

Working notes, 2026-07-14/15. This documents the investigation that started
from "why does Figure 4 look jittery / why are NEs off the BR curves" and
ended with an exact, closed-form characterization of the game's equilibrium
geometry. Each section: what we ran, how and why, where the data lives, what
it showed, and what it means for `paper/main.pdf`.

Baseline throughout: the post-solver-fix K=200 mechanism cache
(`game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`,
rebuilt 2026-07-13 after the `solve_mechanism` bisection bug), instance
|G1|=|G2|=5000, B=128, T=390, delta=1e-12, lambda=1, actions
eps in [0.5, 64] (K=200 log grid), utility u_i = V(sigma) - lambda*A_PLD,i,
V(sigma)=1/(1+sigma^2).

---

## 1. Continuous (grid-free) equilibrium search on the real instance

**Setup.** `game_experiments/exact_equilibria.py` (Slurm 179691, fast
tolerances; full-precision twin 179667 cancelled as superseded).
(3a) treats the symmetric equilibrium as a fixed point of the local
best-response map phi(eps) = argmax_e u1(e, eps) (golden-section inner
argmax in a +-15% window, bisection on psi(eps) = log phi(eps) - log eps);
(3b) profiles the loose player's FOC along the analytic participation
frontier. Output: `tasks_findings/_scripts/29_exact_equilibria.pkl`.

**Why.** The K=200 grid reports a symmetric pure NE at eps*=2.9646 and 42
mirror-pair NEs; we wanted the continuous-game objects free of grid
artifacts.

**Results.**
- psi is non-monotone: +0.031 at 2.5, dips to **+0.0036 at 2.958** (at the
  run's tolerance floor), rises to +0.050 at 3.218, crosses zero at
  **3.2555**. The 3.2555 crossing is a branch-jump artifact, refuted by the
  built-in global check (true best response to 3.2555 is 2.285, +0.0023
  utility). The dip at ~2.96 matches the grid NE but the fast run could not
  distinguish "exact fixed point" from "flat near-fixed-point".
- (3b) was invalid in this run: it evaluated *on* the feasibility knife
  edge and the -2 infeasibility sentinel poisoned the finite differences
  (12/17 points). Lesson: evaluate strictly inside the feasible region, or
  use one-sided conditions at the boundary (Section 5 does this properly).

**Interpretation for the paper.** The symmetric equilibrium at 2.96 is
real *on the grid* and consistent with a continuous FOC crossing (Section
6 pins it at 2.9635), but the diagonal is so flat there that
"equilibrium within numerical resolution" is the defensible continuous
claim. The prose in IV.B already scopes the claim to the K=200 game;
keep it scoped.

## 2. FOC landscape by numeric differentiation of the cache

**Setup.** `tasks_findings/_scripts/31_foc_landscape.py` (login-node
numpy; no new solves). Central differences of the cached payoff along the
own axis; zero crossings classified falling (ridge = local max) / rising
(valley); mirror check D2 == D1^T exact on 25,412 shared cells. Data and
plots in `paper/figures/data/foc_landscape/` (`ridge1.dat`,
`valley1.dat`, `foc_landscape.png`).

**Why.** User direction: no dense solver sweeps — read the landscape the
cache already contains and find the expected zero set.

**Results.**
- The raw diagonal FOC oscillates with amplitude ~+-0.03 (a sawtooth) on
  top of a smooth trend crossing zero once near eps~2.9-3.0. 15 alternating
  crossings between 1.96 and 3.62 are sawtooth teeth, not equilibria.
- The flat blob: |dU1/dlog eps1| has median 0.017 inside [1.5,5]^2 vs
  0.083 outside — the central region is near-critical everywhere, which is
  why grid argmax jitters there.
- Global BR per column (Savitzky-Golay smoothed, value-maximal branch;
  `32_fig_C6_landscape_data.py`) has exactly **three branches**: upper
  feasibility wall (opponent tight), interior arc (eps2 in
  [1.65, 4.71]), lower wall = analytic participation frontier (opponent
  loose). The branch jumps are genuine value-crossing discontinuities.

**Paper impact (already applied).** `paper/tex/fig_C6_tikz.tex` rebuilt
from `c6l_*.dat` (frontier fill, three-branch BR curves, NE segment on
the frontier, symmetric star); caption and the "BR loci" paragraph in
`paper/tex/04_analysis.tex` rewritten (zero set of eq:foc read off the
cache; two of three branches closed-form; 42 grid NEs read as samples of
a one-parameter frontier family).

## 3. Equilibrium heatmaps: distance vs regret

**Setup/why.** Two whole-grid views of "how far is each profile from
rest": (a) summed log-budget distance to the two best responses; (b)
summed deviation regret [max_e u_i - u_i]. Rendered from the cache only
(`br_distance_heatmap.png`, `regret_heatmap.png` in
`paper/figures/data/foc_landscape/`; regret page published as artifact
https://claude.ai/code/artifact/b4ee7b8b-f52d-4656-87de-980292c09bcc).

**Results.** The distance map stripes (a cell inherits its row/column
argmax, so one near-tie repaints a whole line — distance-to-argmax is
ill-conditioned under flat payoffs). The regret map is smooth: 43 zeros
(the grid NEs), two low-regret channels along the frontier walls, one
central plateau around the diagonal window, median regret 0.166, only
0.66%/0.19% of feasible cells below 1e-3/1e-4. The 1e-4-amplitude
checkered speckle is the accountant sawtooth.

**Interpretation.** Equilibrium is a thin set: one flat diagonal plateau
+ two frontier channels. Regret (payoff units), not argmax distance
(budget units), is the honest scalar diagnostic under near-indifference.

## 4. Exact analytic model of u2(eps2 | eps1)

**Setup.** `tasks_findings/33_u2_analytic_model.md`: the
approximation-free model. Per-step pair P = (1-p)N(0,s^2) + pN(1,s^2) vs
Q = N(0,s^2); T-fold privacy-loss sum; exact hockey-stick profile
delta(eps); budget function E(sigma,p) as its partial inverse; mechanism
map = two binding budgets + batch (unique by strict monotonicity);
u2 = V(sigma*) - lambda*Adv(sigma*, p2*) with Adv = TV of the composed
pair = delta(0); implicit-function-theorem comparative statics; the one
transcendental object is the T-fold convolution.

**Why.** User asked for the perfect analytical description; this is the
reference the computations below implement or specialize.

## 5. Exact FOC at points; the corner nature of the frontier equilibria

**Setup.** `game_experiments/exact_foc.py`: solve the mechanism once,
assemble du2/deps2 via implicit derivatives of the budget function +
PLD-advantage partials; validate against direct finite differences of u2.

**Results.**
- Interior validation at (4.0, 2.0): implicit chain -5.282e-3 vs direct
  -5.251e-3 (0.6%).
- At the paper's mirror pair (1.12, 7.13): the loose player's derivative
  is **+9.7e-3 pointing into the feasibility wall** (one-sided direct
  check +9.9e-3) — they would loosen further but the next step squeezes
  the opponent's rate below the participation floor. The tight player's
  one-sided derivative off the wall is negative (direct -2.1e-2; the
  smooth-piece implicit value at the exact corner is much smaller,
  -3.8e-5 — strong curvature/kink where the advantage turns on at the
  floor).
- Conclusion: the frontier equilibria are **corner equilibria on both
  sides** — two one-sided sign conditions on the shared constraint
  p_tight = p_bar, no stationarity anywhere. Since inequalities hold on
  intervals, the equilibrium set is generically a **continuum segment**
  of the frontier; the 42 grid NEs are grid samples of it.

**Interpretation for the paper.** This sharpens IV.B: the loose player is
not "near-indifferent along a flat top" — at the equilibrium they are
strictly blocked by the constraint. The star string is a one-parameter
family of strict corner equilibria. Where the paper says the tight player
sits at the feasible infimum "the same feasible infimum the coalition
best response selects", the loose player's coordinate should be described
by the segment, not by point estimates. (Figure/caption partially
updated; full prose propagation pending.)

## 6. Exact FOC field on the full grid (T=390)

**Setup.** `game_experiments/exact_foc_grid.py` (Slurm 179779, ~3h/16
cores). Per feasible cell: p2 recovered by inverting the budget function
at the cached sigma (batch gives p1); budget partials by central
differences on the smooth Renyi branch; advantage partials from a
bicubic spline of a fresh 90x45 log-log Adv(sigma,p) table (PLD
discretization 1e-4). Output `tasks_findings/_scripts/34_exact_foc_grid.pkl`,
plot `t390_exact_foc.png`.

**Error inventory (explicitly):** RDP order-grid accounting is the
game's definition (kinks at order switches); PLD discretization 1e-4;
cached sigma tolerance ~1e-4 rel; the spline is the one true
interpolation (dominant error term); implicit chain valid in the
interior only.

**Results.** Field validated against Section 5 pointwise values. 236
interior ridge zeros trace the same arc as the T=1 game; wall-push
statistics: 48 columns push into the upper wall, 115 into the lower.
The diagonal shows 23 alternating crossings in [1.69, 3.62] — the
sawtooth is **intrinsic to the implemented game** (finite Renyi order
grid), not numerics; the structural falling crossing at **2.9635**
matches the grid NE 2.9646 to three digits.

**Interpretation.** Two consequences. (i) The paper's instance genuinely
has kinked payoffs at ~1e-4 scale; any claim finer than that must either
go through the T=1 companion or change the accountant (continuous-alpha
or PLD budgets = a model change). (ii) The symmetric equilibrium is the
one structural diagonal crossing; everything else on the diagonal is
teeth.

## 7. The T=1 closed-form companion game

**Setup.** `game_experiments/single_step_game.py`. At one step everything
is elementary: threshold x_eps = 1/2 + sigma^2 log((e^eps-1+p)/p);
delta(eps; sigma, p) = (1-p)Phi(-x/s) + p Phi((1-x)/s) - e^eps Phi(-x/s)
(log-safe variants for extreme regimes); **Adv = p (2 Phi(1/(2 sigma)) - 1)**;
all partials closed form (hockey-stick boundary terms vanish). Only
numerics: Brent root-finds of monotone elementary functions
(xtol 1e-13..1e-16). Same |G|, B, delta, action box [0.5, 64].
Validation: solved budgets reproduce eps to ~1e-15; the closed-form FOC
matches direct differentiation to 8 significant digits.

**Why.** User requirement: no 1e-5-level errors anywhere — remove the
single transcendental object (composition) instead of approximating it.

**Grid.** `single_step_grid.py` (Slurm 179788):
`tasks_findings/_scripts/35_single_step_grid.pkl` (sigma, p2, u2, exact
D2 per feasible cell; 21,110/40,000 feasible).

**Key structural fact.** At lambda=1 the T=1 game is degenerate: the
advantage is *linear in the own rate* and capped by it
(Adv <= p <= B/|G| = 0.0256), so privacy can never compete with utility
and 195/200 columns push into the upper wall. The three-branch structure
that mirrors the paper's game exists only for lambda ~ 24-36 (grid scan
of U(lambda) = V - lambda*Adv from the saved fields; lambda=28 chosen).

## 8. Exact best responses and equilibria at T=1 (lambda=28)

**Setup.** `single_step_br.py` (Slurm 179809): 800 columns; per column
the feasible interval from the closed-form frontier, 160 samples of the
exact FOC, Brent-polished zeros (1e-12), global BR by exact value
comparison vs both corners. Output
`tasks_findings/_scripts/36_single_step_br.pkl`; box-clipped BR cached in
`37_t1_br_clipped.pkl`. Artifact (all T=1 figures):
https://claude.ai/code/artifact/5b4e015c-717c-4fcc-8dde-3a82a32f2d6b

**Results.**
- BR2 branch structure: interior for eps1 < 1.2585 (loose optimum
  ~9.7-10.9), lower corner beyond; the loose corner never wins at this
  lambda at 800-column resolution (the 10 corner-hi columns in the coarse
  K=200 scan were argmax artifacts).
- **Exact pure NE: (0.5, 9.768436) and mirror.** Loose player at the
  interior FOC optimum (exact root); tight player pinned at the *action
  floor* with strictly negative derivative (-3.5e-4). The participation
  frontier at that opponent budget lies below 0.5, so the box binds
  instead of the frontier.
- **No symmetric equilibrium**: the BR jumps across the diagonal at
  eps ~ 1.26 without touching it (zero true diagonal fixed points after
  excluding branch-jump interpolation artifacts).
- Fine detail: for eps1 in [9.98, 10.16] and [13.04, 13.12] the two
  corners of the slice tie in value and the global argmax flips twice —
  resolvable only because corner utilities are exact.

## 9. The lambda sweep (privacy weight 0.001 -> 80)

**Setup.** `single_step_lam_sweep.py` (Slurm 179967 for lambda >= 1,
179974 for lambda < 1; outputs `38_t1_lam_sweep.pkl`,
`39_t1_lam_low.pkl`, the latter with component samples cached for
arbitrary future lambda). Exploits linearity of the FOC in lambda:
D(eps2; lambda) = Vterm - lambda*Advterm, so one component sample per
column serves every weight; each root re-polished by exact Brent.
Figures: `t1_lam_full_bifurcation.png`, `t1_br_all_lams.png` (flipbook,
25 panels).

**Results — five regimes.** (Regime 2 corrected 2026-07-15; the sweep
originally reported it as a no-pure-NE window, an analyzer artifact —
the box-edge pair below was missed because the equilibrium candidate
list never included action-ceiling-clipped best responses.)
1. lambda <= 0.03: unique symmetric equilibrium at the box ceiling
   (64, 64) — privacy negligible, both max out.
2. lambda ~ 0.05-0.09: **isolated box-edge pair** (47.14, 64) and its
   mirror. 47.1399 = f^{-1}(64): the column whose participation
   frontier passes exactly through the action ceiling. The loose
   player's frontier corner is clipped at the box (BR = 64, D > 0 up to
   the edge); the tight player's corner_lo lands exactly back on
   f^{-1}(64) (one-sided maximum, D = -6.5e-8 into the corner; utility
   margin over the ceiling corner 2.4e-5). Verified mutual by direct
   exact evaluation at lambda in {0.05, 0.08, 0.1, 0.15}. **There is no
   pure-NE gap in the T=1 bifurcation**: ceiling -> edge pair ->
   segments. (The T=390 lambda=2 nonexistence, Sec. 11, is a different
   regime — there the tight best responses are deep interior and the
   cycle argument stands.)
3. lambda 0.1-26: asymmetric **wall-equilibrium segments** (see Section
   10), sliding inward as lambda grows (the edge pair of regime 2 is
   the segment's birth: at lambda ~ 0.1 the tie budget enters the box
   and the point grows into an arc); coexisting with the floor pair
   from lambda ~ 8.
4. lambda 28-36: unique floor pair; the loose player's interior optimum
   detaches from the feasibility ceiling at lambda ~ 27 and descends
   (9.77 at 28 -> 6.79 at 36).
5. lambda >= 40: unique symmetric corner (0.5, 0.5).

**Headline.** Symmetric play is **never an interior equilibrium at any
weight** — it appears only constraint-pinned (ceiling when privacy is
nearly free, floor when it dominates). Between the extremes the game
rests off-diagonal or not at all.

## 10. The continuum: equilibrium segments, not equilibrium points

**Finding.** Checking mutual best-response consistency *between* the
detected equilibria at lambda=1: g(eps1) = log b(b(eps1))/eps1 is
identically zero (to 1e-5, the trace resolution) along the whole wall
stretch eps1 in (~16.9, ~29.4). The "pairs" previously reported are the
**endpoints of two mirror-image segments** of pure equilibria.

**Mechanism.** The two BR graphs do not cross there — they **coincide**:
the upper wall (p1 = p_bar) is simultaneously player 2's corner branch
(loosen until the opponent's rate hits the floor) and player 1's corner
branch (tighten until the own rate hits the floor). Corner equilibria
need two one-sided inequalities, which hold on intervals; the equilibrium
set is one-dimensional wherever both players' value comparison selects
the shared wall. The segment's upper endpoint is the value-tie budget
bar-eps(lambda) where the pushing player's branch values cross; the lower
endpoint is its image through the wall map lo(bar-eps).

**Interpretation for the paper (the big one).** The T=390 frontier "star
string" (42 grid NEs, ratio band 6.38-6.70) is this exact object sampled
by the grid: a **one-parameter family of strict corner equilibria on the
analytic participation frontier**, endpoints determined by value ties.
This replaces both the "42 equilibria" count and the "near-indifference"
framing: neither indifference nor multiplicity-by-accident — a continuum
by constraint geometry. Sections IV.B / appendix should present: (i) the
frontier as the equilibrium locus, (ii) the two one-sided conditions,
(iii) endpoints as value-tie roots, (iv) grid counts (7 at K=22, 43 at
K=200) as sampling density of the same segment. Figure 4's green
segment is already this; the prose still partially speaks in points.

## 11. Dynamics: fighting for the protected role

**Setup/results.** Alternating best-response dynamics on the cached
T=390 payoffs at lambda=2 (the no-pure-NE weight): from *every* start,
play enters the same **3-cycle** around eps ~ 1.01-1.07, roles swapping
every step — an Edgeworth undercutting cycle (tighten under the
opponent; when starving on utility, someone loosens; chase restarts).
Role assignment in the pure mirror equilibria is not determined by the
primitives (symmetric game): history/first mover picks roles in the
dynamics; the tight role free-rides and is strictly preferred (Hawk-Dove;
mixed P(T)=0.445 in the paper's lottery framing); any heterogeneity
(lambda_i, group size) breaks the tie deterministically (C2/C8 sorting,
170/174).

**Interpretation.** The paper's Hawk-Dove/lottery discussion is
well-supported and can cite a concrete cycling regime: at weights where
no pure NE exists, "bouncing" is the equilibrium-free dynamic reality,
not a pathology.

**Trajectory figure** (`tasks_findings/_scripts/54_cycle_trajectory.py`
-> `paper/figures/data/foc_landscape/cycle_trajectory.png`; artifact
Figure 9): staircase best-response trajectories from eight starts, all
funneling into the attractor within a handful of replies. Alternating
play visits six states built from the three budget values
{1.014, 1.039, 1.065}; the mixed NE randomizes over only two of them
({1.014, 1.065}, cross-hairs in the zoom panel), visualizing the Sec.
11 point that the cycle's time-average (0.0896-0.0899) beats the mixed
value (0.0888) — correlated circling beats independent randomization.
The two BR curves nearly touch without crossing at this weight; play
slides down the channel between them and circles.

## 12. Why interior best responses are rare — composition is the source

**Formula-level diagnosis (T=1).** Adv = p (2 Phi(1/(2 sigma)) - 1) is
exactly linear in the own rate (dAdv/dp independent of p) and capped by
participation (Adv <= p). Flat marginal cost vs declining marginal
utility gain gives corner solutions except in the knife-edge lambda
window. The cross-player externality (eps2 moves sigma, p1, hence A1
and BR1) is fully present at T=1; what is missing is own-cost convexity.

**Composition changes the cost geometry — but not by convexity in p.**
Measuring it exactly (T=2 by certified quadrature, T in {5, 20, 100,
390} by pessimistic connect-the-dots PLD; script
`tasks_findings/_scripts/44_adv_curvature.py`, output
`44_adv_curvature_summary.txt`): at fixed sigma, Adv_T(p) is mildly
**concave** in p for every T >= 2 — d2Adv/dp2 < 0 throughout the
operating range, increasingly so with T — never convex. What
composition does add is (i) **scale**: the per-rate advantage Adv/p
grows superlinearly with T (sigma=0.9: 0.42 at T=1 -> 12.1 at T=390),
so the privacy term stops being negligible relative to V at moderate
lambda; and (ii) **sigma-p coupling**: Adv_T is no longer of the
product form p * c(sigma), so the sigma- and p-channels mix in the FOC
chain instead of separating. The interior FOC roots at T=390 must come
from these two effects; the T=2 exact best-response trace (Sec. 14)
is the direct test of how early they appear. **The interior trade-off
of the PEG remains a composition phenomenon** — at one step the game
is constraint-driven — but the earlier reading "composition makes the
advantage convex in the own rate" is refuted by the exact computation.

**Interpretation for the paper.** This is a crisp, citable observation
for the discussion: sampling-based iDP creates an externality at any T,
but the *interior* privacy-utility bargaining that Prop. 9 and the
symmetric-window analysis describe is bought by composition. Candidate
exact extensions if wanted: convex privacy disutility g(A) at T=1
(keeps everything elementary; makes interior BRs generic), or small-T
exact convolution (T=2-5, machine-precision quadrature, no accountant).

## 13. Group size at fixed batch: sliding the equilibrium continuum

**Setup.** T=1 exact game, lambda = 0.7 for both players, batch B = 128
held fixed, group size |G1| = |G2| = |G| varied one decade in each
direction around the paper instance: |G| in {500, 5000, 50000}. Fixing
B while changing |G| moves exactly one primitive: the per-record rate
ceiling p1 + p2 = B/|G| (0.256 / 0.0256 / 0.00256), which at T=1 also
caps the advantage (Adv <= p). 400 columns x 140 FOC samples per
setting, exact chain throughout (job 180103 ->
`tasks_findings/_scripts/40_t1_br_G500.pkl`, `41_t1_br_G50000.pkl`;
baseline from the lambda-sweep component cache `39_t1_lam_low.pkl`).
Segment endpoints polished as value-tie roots
(`43_gsweep_endpoints.py`); figure `43_gsweep_figure.py` ->
`paper/figures/data/foc_landscape/t1_gsweep_lam07.png`.

**Why.** The paper's heterogeneity experiments (C2/C8 sorting) vary
group size as an asymmetry device; this sweep isolates its *symmetric*
role: |G| sets how much attack surface one record's participation can
accumulate, i.e. how hard the privacy term can bind at all.

**Results.** All three settings are corner games (no interior BR
anywhere); what moves is which corners are mutual:

- |G| = 500 (p-ceiling 0.256): frontier NE segment
  eps1 in [0.50, 11.94], frontier arc (0.50, 12.87) -> (11.94, 27.85),
  clipped at the action floor on the tight side. The tight/loose role
  split starts at eps1* = 11.94.
- |G| = 5000 (baseline, 0.0256): segment eps1 in [21.97, 35.82], arc
  (21.97, 35.82) -> (35.82, 51.40); eps1* = 35.82. Both endpoints
  interior to the action box.
- |G| = 50000 (0.00256): every column is corner_hi and the
  participation frontier exits the box (f(64) = 77.4 > 64): the
  box-clipped best response at eps1 = 64 is 64 itself, and
  **(64, 64) is the unique pure NE** — the privacy term never binds
  inside the action box; the game collapses to the symmetric utility
  race (the lambda -> 0 ceiling regime of Sec. 9).

**Interpretation.** Group size at fixed batch acts as an *effective
privacy weight*: the kind-switch budget eps1* falls monotonically as
B/|G| rises, and the equilibrium continuum slides toward the origin
(heavily sampled populations: someone gets pinned at the floor early)
until, in the other direction, it exits the box entirely and only the
ceiling equilibrium survives. But it is not a pure relabeling of
lambda: the participation frontier itself moves with |G| (at
|G| = 50000 the frontier clears the ceiling, which is why the outcome
is the ceiling NE rather than the no-pure-NE window that the naive
rescaling lambda_eff ~ 0.07 would predict at fixed geometry). For the
paper: subsampled iDP's externality game is governed by the *pair*
(lambda, B/|G|), and "more participants at the same batch" is a
privacy-relaxation lever exactly as strong as lowering lambda —
modulo the frontier geometry it also shifts.

## 14. Exact composition: the T=2 game (and the road to small T > 1)

**Setup/method.** The T-fold profile of the Poisson-subsampled
Gaussian obeys the exact recursion delta_T(eps) =
E_{X~P}[delta_{T-1}(eps - g(X))], with P the dominating mixture
(1-p) N(0, sigma^2) + p N(1, sigma^2) and g(x) = log((1-p) +
p e^{(2x-1)/(2 sigma^2)}) the per-step privacy loss; delta_1 is the
closed form of Sec. 7 extended to all real eps (delta_1 = 1 - e^eps
below eps = log(1-p)). At T=2 this is a **single 1-D integral of
elementary functions** — no privacy-loss discretization grid — and
all partials in (eps, p, sigma) differentiate under the integral into
the closed-form delta_1 partials. Implementation
`game_experiments/two_step_game.py`: per mixture component, composite
Gauss-Legendre in the standardized variable with kink-refined panels
(the delta_1 branch switch is C^1, so no boundary terms in the
derivatives); tails truncated at 13 sigma (mass < 1.3e-38).
Certification: against 40-digit mpmath quadrature, relative error
<= 3e-9 at the budget scale (delta ~ 1e-12) and ~1e-14 at advantage
scale (delta(0)); inside the dp_accounting pessimistic/optimistic
bracket; partials match finite differences to 8 digits; mechanism-map
budgets reproduce to 1e-15. The same recursion nests to T = 3-5
(cost ^ (T-1), fine for spot checks); beyond that the honest tool is
two-sided certified PLD brackets (pessimistic + optimistic
connect-the-dots), running the whole analysis on both ends.

**First exact finding (advantage curvature).** See Sec. 12: Adv_2 =
delta_2(0) is concave-in-p, refuting the convexity reading; the
composed advantage's sigma-p coupling is the candidate source of
interior structure.

**Best-response trace (job 180116, 160 columns x 100 samples ->
`tasks_findings/_scripts/42_two_step_br.pkl`).**
`game_experiments/two_step_br.py` samples the FOC components per column
(the FOC is again linear in lambda, so one pass serves all weights);
zeros polished exactly for lambda in {0.05, 0.7, 2, 28}, and the cached
components scanned over lambda in [0.001, 100] (51 points).

**Results — same taxonomy as T=1, regimes shifted.**
- No interior best response at any of the four polished weights; the
  component scan finds an **interior-BR window at lambda ~ 8-25**
  (peaking near lambda ~ 13 with ~39 of 160 columns interior), vs
  lambda ~ 27-36 at T=1. Composition pulls the interior structure
  toward realistic weights already at two steps.
- Exact equilibria: lambda = 0.05 -> ceiling NE (64, 64) (the frontier
  exits the box: f(64) = 119.5); lambda = 0.7 -> frontier segment,
  eps1 in [16.66, 29.84], arc (16.66, 40.65) -> (29.84, 64.00),
  box-clipped at the ceiling (tie budget eps1* = 40.6496 with
  f(eps1*) = 83.07 > 64); lambda = 2 -> unclipped segment, arc
  (3.296, 17.345) -> (17.345, 41.849); lambda = 28 -> symmetric floor
  (0.5, 0.5) (already past the interior window that T=1 reaches only
  at 28-36).
- The ceiling regime survives to lambda ~ 0.4 at T=2 (vs ~ 0.03 at
  T=1) and hands off directly to the segment regime; no pure-NE gap
  (consistent with the corrected T=1 picture, Sec. 9).
- Budget-geometry inflation: the same (sigma, p) point costs more
  composed budget, so eps-space landmarks move outward (frontier at
  the top column: 82.6 -> 119.5); comparisons across T should be made
  in (sigma, p) space or by regime, not by raw eps values.

**Symmetric stationary points exist at T=2 — but never survive the
loosening deviation.** Interpolating the cached FOC components onto the
diagonal: D(e, e; lambda) has a falling zero e*(lambda) for lambda in
~[2, 16] (e.g. e* = 4.73 at lambda = 5, e* = 1.29 at lambda = 12.6) —
the symmetric point is FOC-stationary along the own-budget slice, the
necessary condition for a symmetric interior NE. But at every lambda
the column's global best response is a large loosening (e.g. from the
stationary 4.73 to 19.9), and the deviation gain is big:
u(BR) - u(e*, e*) in [0.050, 0.101] across the whole window — three
orders of magnitude above the tie margins that decide corners
elsewhere. At T=2 the symmetric candidate always loses to "loosen and
grab the batch." T=1 (campaign, Sec. 9) and now T=2 are therefore both
settled: **no fully interior pure NE exists at any weight**; the T=390
symmetric equilibrium at eps* = 2.96 is genuinely a large-T object,
and the deviation gap (~0.05-0.10 at T=2, negative by T=390) must flip
sign at some onset T*(lambda) in between — locatable with certified
two-sided PLD brackets at intermediate T, since the gap dwarfs any
bracket width.

**Interpretation.** T=2 confirms the T=1 mechanism map: corner games
with a continuum of frontier equilibria at moderate weights, interior
best responses only inside a finite lambda window, symmetric play only
when constraint-pinned. What composition does, quantitatively, is
slide the interior window down (~3x per added step at this scale) and
inflate the budget geometry. The T=390 interior structure at moderate
lambda is the accumulated version of exactly this shift.

## 15. The two-phase slice: the general participation trade-off

**Setup.** Own-budget slices u2(eps2 | eps1) decomposed into their two
components (value gain V(sigma) and privacy cost lam * Adv) together
with the own rate p2, across the settings of Secs. 7-14. Corner case:
`paper/figures/data/foc_landscape/t1_slice_decomposition.png` (T=1,
lam=0.7). Interior cases: `interior_slice_decompositions.png`
(T=1 lam=28; |G|=500 lam=3 and lam=4; T=2 lam=12.6), generated by
`tasks_findings/_scripts/46_slice_decompositions.py` (data cached in
`46_slice_data/`).

**Finding (holds in every slice computed this campaign — T in {1, 2},
|G| in {500, 5000, 50000}, lambda in [0.001, 100]).** The slice has a
universal two-phase anatomy, separated by the batch-rate takeoff:

1. **Rate phase** (own rate below the batch cap). Participation — and
   with it the advantage — grows exponentially in the declared budget
   (the budget is a log-scale quantity), while the shared noise is
   anchored by the opponent: the own pull on sigma is proportional to
   the own batch share. Cost leads value, so the slice always has a
   trough here. "Modest participation at modest exposure" is dominated
   in every setting computed.
2. **Noise phase** (own rate saturated at the batch cap). Extra budget
   converts into lower sigma at fixed rates: value arrives, and the
   only cost channel left is the smooth sigma-channel of the advantage
   (per-round distinguishability rising as sigma falls). Past the
   participation frontier the declared budget goes slack and the
   outcome freezes.

**Consequences.**
- Best responses never sit mid-takeoff: they are the floor corner
  (before the takeoff), the frontier/ceiling (end of the noise phase),
  or an interior point *inside the noise phase*.
- An interior best response exists exactly when the weighted
  sigma-channel cost slope overtakes the value slope before the
  frontier. The three knobs — the privacy weight lambda, the rate
  ceiling B/|G|, and the composition count T — all act through this
  one product (weight x capped-rate advantage sensitivity to sigma),
  which is why each of them alone can detach the peak from the
  frontier and push it inward (exact peaks at 0.985, 0.925, 0.85 and
  0.68 of the frontier budget in the four panels; Brent-polished
  values 9.768436 / 11.904468 / 11.594644 / 8.786123).
- **Privacy moderation in the PEG is noise-demand moderation, not
  participation moderation.** At every interior optimum computed, the
  player is a near-full participant (rate at or near the batch cap)
  who caps how clean a model they demand; the interior FOC is
  effectively a tangency in sigma alone. No setting produced an
  optimum at partial participation.
- The interior regime is a **thin transition band** between corner
  regimes: the peak descends from the frontier and is overtaken by the
  floor corner within a factor ~1.3-2 in the weight (T=1, |G|=5000:
  detachment ~27, floor at ~38; |G|=500: 3 -> 5; T=2: ~8 -> 25). This
  is the structural reason the equilibrium landscape is
  corner-dominated almost everywhere.

**Best-response curves and equilibria in the interior settings**
(`49_interior_br_curves.py` -> `interior_br_curves.png`; G=500 sweep
for lambda in {3, 4}: job 180141 -> `47_t1_lam_G500.pkl`; T=2
lambda=12.6 curve from the component cache ->
`48_t2_br_lam126.pkl`). The interior branch of BR2 is a nearly flat
plateau against weak opponents (the peak location barely depends on
eps1), and equilibria remain constraint-involved even where it exists:
at T=1 the pure NE are floor pairs whose loose coordinate is the
interior optimum — (0.5, 9.768436) at lambda=28, (0.5, 11.904468) at
G=500 lambda=3, (0.5, 8.786123) at G=500 lambda=4, each plus mirror —
and at T=2 lambda=12.6 the equilibria form a frontier segment
(eps1 in ~[0.53, 0.85], images ~[11.2, 12.3]) while the interior
branch carries **no equilibrium at all**: interior best responses and
interior equilibria separate. No fully interior pure NE (both
coordinates at FOC roots, no constraint involved) has appeared in any
setting computed.

**Interpretation for the paper.** This is the campaign's cleanest
portable statement about incentives under sampling-based iDP: the
budget declaration controls two qualitatively different resources in
sequence — first *how often you are seen*, then *how clearly anyone is
seen* — and rational play only ever balances at the margin of the
second. It compresses Secs. 7-14 (corner taxonomy, frontier continua,
interior windows, and the knob equivalences) into one mechanism.

## 16. Stabilizing moderation: the shape of the exposure price, not
its level

**Question.** Can an interior equilibrium — both players voluntarily
moderate, no constraint involved — be manufactured by making exposure
more expensive? Two readings, opposite answers.

**Linear scaling fails at any level.** The FOC and utility are linear
in lambda, so raising the weight re-scales the game instead of taming
it: the symmetric FOC-stationary candidate e*(lambda) slides to
smaller budgets, and the profitable deviation — loosen hard, take the
batch, own the noise — survives at every weight. Measured at T=2
across the whole window lambda in [2, 16]: deviation gain
u(BR) - u(e*, e*) in [0.050, 0.101], never close to zero (Sec. 14).
Mechanism: a linear price charges the deviator's last unit of exposure
the same as the moderate player's first unit; the deviator buys a
bounded exposure increment and collects an accuracy prize that dwarfs
it.

**Convex pricing succeeds — already at T=1.** With
u = V - lam_q * Adv^2 (everything else unchanged), the cached
component sweep (`39_t1_lam_low.pkl`; components are cost-shape
agnostic) shows a **symmetric fully interior pure NE** at the paper
instance, one-shot: for lam_q >~ 750 the diagonal FOC-stationary point
e* in ~[0.8, 4.1] is the *global* best response to itself (column
argmax coincides with the diagonal point to sample resolution ~1e-6;
at lam_q = 562 the batch-grab still wins by 4.9e-3, locating the
regime edge between 562 and 750). The first interior equilibria of the
entire campaign that involve no constraint at all. (Caveat: sample-
grid resolution, not Brent-polished; a small polishing job would bring
these to campaign standard.)

**General statements.**
1. **The level of the exposure price cannot stabilize moderation;
   only its shape can.** Under linear pricing the game is
   scale-invariant in the weight: every regime exists at every price,
   just relocated. Stability of mutual moderation requires the
   deviator's marginal exposure price to exceed the moderate player's
   — i.e. convexity of the disutility in the realized advantage.
2. **Composition is organic convexity.** Repetition makes accumulated
   exposure grow superlinearly with per-round presence (Adv/p rises
   ~30x from T=1 to T=390, Sec. 12), so a heavy participant faces a
   disproportionately worse bill — the same asymmetry the quadratic
   cost imposes by hand. The T=390 symmetric interior NE (eps* = 2.96)
   and the T=1 quadratic-cost NE are the same stabilization mechanism
   reached two ways; the onset T*(lambda) at which the deviation gap
   (+0.05..0.10 at T=2) flips sign is the natural bridge quantity
   (Sec. 17).
3. **Mitigation-design corollary for the paper.** Pricing the privacy
   externality (contractual penalties, exposure taxes, reputational
   costs) stabilizes voluntary moderation only if the price is convex
   in exposure. Any linear tax, at any rate, leaves the
   grab-the-batch instability in place.

## 17. The T-grid: exact composition from 1 to 48 steps (running)

**Goal.** Track the composition knob analytically across a grid of
step counts — regime landmarks, advantage scale, and above all the
symmetric-deviation gap gap(T, lambda) whose zero crossing T*(lambda)
is the onset of symmetric interior equilibria (the bridge between the
T<=2 impossibility of Secs. 14/16 and the T=390 eps* = 2.96
equilibrium).

**Method — no accountant anywhere in the loop.** The exact ladder
recursion delta_T(e) = E_X[delta_{T-1}(e - g(X))]
(`game_experiments/multi_step_game.py`): delta_1 is the closed form
(exact hockey-stick branch delta_t(e) = 1 - e^e below t*log(1-p) at
every level); each composition step is one vectorized Gauss-Legendre
quadrature of the tabulated level below, read through a cubic spline
of log delta on a graded grid (dense at the kink end, sized by the
exact per-step loss moments plus its node-tail maximum). The delta_1
kink is never interpolated — level 2 integrates the closed form
directly; all deeper levels are smooth. Certification: T=1 anchor at
machine precision (budgets 1e-13 relative), T=2 anchor vs the
kink-aware certified quadrature <= 5e-5 relative on budgets (<= 1e-6
on advantages; the residual is deep-tail panel error, quantified),
grid-doubling self-tests <= 3e-7, dp_accounting pessimistic bound
respected at T in {3, 8}. Cost ~0.3 s per composition step per
(sigma, p), independent of the pair.

**Design.** For each T in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48} and
each diagonal budget e1 (12 log-spaced in the action box): the column
u2(. | e1) sampled in *physical* space by the own rate p2 (31 points,
p1 = PSUM - p2 fixed by the batch), sigma root-found from the
opponent's binding budget (one 1-D root per point, warm-started), e2
and Adv_2 read off the same ladder. No nested inversion anywhere; the
diagonal point reproduces e2 = e1 to ~1e-12 as a per-column
self-check. The rows post-process into u2 for any lambda (components
stored, cost-shape agnostic — the same cache also serves convex
pricing g(Adv), Sec. 16). Job 180154 ->
`tasks_findings/_scripts/50_t_grid.pkl` (T <= 48); companion job
180155 -> `51_t_grid_hi.pkl` extends to T in {64, 96, 128} at grid
density PEG_NE = 4500 (T=64 doubling check: budget stable to 9 digits
at production density). Runner `game_experiments/run_t_grid.py`.
Results to be appended.

**Extension to T = 390 (job 180205 -> `52_t390_ladder.pkl`; attempt
180187 failed on a spline-overshoot bug at the sigma-bracket edge,
fixed by capping the tabulated log delta at 0; attempt 180196 was
resubmitted with per-column checkpointing and a 30 h / 6-worker shape
after an ETA audit showed 16 h columns against a 12 h limit).** Same
ladder machinery at grid density PEG_NE = 6000, with the diagonal
candidates focused by the GDP limit (Sec. 18) on
e1 in {1.5, 2.2, 2.92, 3.8, 5.2, 8.0} at 20 column points each — the
region that decides the gap sign at the paper's step count.

**Certification split (why there is no fully-certified budget at large
T).** A rigorous two-sided enclosure engine by measure doubling was
built (`game_experiments/double_ladder.py`): pessimistic/optimistic
directed rounding of the exact per-step loss CDF, exact FFT
self-convolutions in O(log T), directed coarsening and tail handling.
It certifies **advantages** (delta(0) ~ 1e-2..1e-4): enclosures
verified against the exact T=1/T=2/ladder references in every test,
width 4.3e-4 at T = 390. It cannot certify **budgets** (delta = 1e-12
deep tail): FFT roundoff has an absolute noise floor ~1e-16 x bulk
mass per entry, which exceeds the true tail masses (~1e-24) and
inflates both bracket sides by ~1e-4 relative in eps — verified
against the mpmath-adjudicated T=2 budget (3.20080722, matched by
`two_step_game` to 1e-10, missed by both bracket sides by +7e-5).
The same mechanism explains dp_accounting's pessimistic bound dipping
below exact values at tiny delta. Budgets therefore rest on the
cancellation-free smooth ladder (all-positive quadrature sums,
anchored, doubling-tested), with dp_accounting's pessimistic eps as a
one-sided rigorous bound; advantages carry true two-sided
certificates.

## 18. The Gaussian-DP limit: closed forms for very large T

**Model.** By the f-DP central limit theorem, the T-fold composition
converges to mu-GDP with mu = p sqrt(T (e^{1/sigma^2} - 1)); mu-GDP
has the exact closed-form profile delta(eps; mu) = Phi(-eps/mu + mu/2)
- e^eps Phi(-eps/mu - mu/2) and advantage 2 Phi(mu/2) - 1. Every game
object is elementary again — the whole gap machinery runs in
microseconds per point (`game_experiments/gdp_limit_game.py` ->
`53_gdp_gap.pkl`; T in {390, 1560, 6240, 25000, 1e5, 4e5}, lambda in
{0.5, 1, 2, 5}, 12 diagonal columns x 31 points). CLT error is
O(1/sqrt(T)) (Berry-Esseen type), so this is the analytic *limit*
model: exact as T -> infinity, to be validated against the exact
ladder at the overlap (T = 48-390).

**Results.**
- At lambda = 1, T = 390 the symmetric candidate sits at
  **e* = 2.919** — beside the paper instance's grid equilibrium 2.96
  (the residual is the RDP accountant's conservatism plus CLT error).
- The deviation gap at (390, lambda=1) is **+0.0037: still positive
  but small** — in the limit model the batch-grab is marginally
  profitable at the paper's step count, and the gap closes to zero
  between **T = 390 and T = 1560**. The paper's T = 390 sits
  essentially *at* the onset of stable symmetric moderation, which is
  exactly why its symmetric equilibrium lives on a ~1e-4-flat plateau.
- Onsets by weight: lambda = 0.5 stabilizes between 1560 and 6240;
  lambda = 2 by 1560 (floor-dominated by then); lambda = 5 is already
  floor-pinned at 390. By T ~ 6240 every weight tested has settled
  into stable symmetric play (interior or floor).
- Caveats: gap resolution is sample-level (31-point columns), and the
  +-0.004 signal at T = 390 is within CLT error at that T — which is
  precisely what the exact-ladder job 180187 (Sec. 17) adjudicates.

## 19. Error budget: numerical accuracy w.r.t. exact theory

All figures below are *measured* in-session (validation blocks in the
module `__main__`s), not estimated.

| layer | scope | accuracy vs theory | evidence |
|---|---|---|---|
| 0: T=1 closed form | `single_step_game.py`, all T=1 NE/segments/bifurcation | ~1e-12..1e-14 (float64 + Brent xtol 1e-14) | budgets reproduce to 1e-15; FOC vs FD to the FD limit |
| 1: T=2 quadrature | `two_step_game.py`, T=2 ties/segments/gaps | delta rel <= 1.1e-7 at budget scale, <= 3.7e-10 at Adv scale (worst over 40 randomized configs; typical far better) | 40-digit mpmath referee, randomized coverage `63_mpmath_coverage.py`; inside dp_accounting brackets; partials vs FD 8 digits |
| 2: T-grid ladder | `multi_step_game.py`, jobs 180154/180155/180187 | budgets ~1e-4 relative at mid/large T (validated one-sidedly), advantages ~1e-6 | anchors at T=1 (1e-13) and T=2 (<=4.6e-5); doubling 1e-7..1e-10; diagonal self-check 6e-13; **sits just below the rigorous CTD-pessimistic upper bound at T in {8, 48, 390} with margins 6e-5..2.6e-4** (margin includes the bound's own conservatism) |
| 2b: bracket engine | `double_ladder.py` | advantages: rigorous two-sided, width 4.3e-4 at T=390 (delta(0) is bulk-scale, FFT-safe); budgets: NOT usable — FFT noise floor (~1e-16 x bulk >> 1e-24 tail masses) inflates both sides, from +7e-5 at T=2 (mpmath referee) to **+38% at T=390** | the T=48 ladder-vs-bracket budget disagreement (2.5e-3) was entirely the bracket's fault |
| 2c: mid/large-T budget referee | deep-tail budgets, T ~ 8..390 | no *two-sided* referee exists (mpmath cannot nest; FFT accountants share the tail problem), but the CTD-pessimistic eps is a rigorous one-sided bound and the ladder respects it everywhere with ~1e-4 margins | consequence: the T=390 gap-sign determination runs at ~40:1 signal-to-error (u uncertainty ~1e-4 from budgets + 4.3e-4 certified advantage width, vs gap ~4e-3) |
| 2d: GDP limit | `gdp_limit_game.py` | closed form given the model; CLT model error O(1/sqrt(T)): a few % on Adv at T=390 -> +-~0.005 on the gap, shrinking with T | to be validated against the ladder at the T=48..390 overlap |
| 3: sampling | grid argmaxes, consistency scans, unpolished NEs | one grid cell / stated tolerance (e.g. 2e-2 mutuality; quadratic-cost NE ~1e-6 in u) | e.g. sample 8.43 vs polished 8.79 |
| 4: model, not error | the paper's T=390 instance | RDP accounting (conservative bound) + PLD Adv at 1e-3 discretization; ~1e-4 sawtooth is part of the game definition | Sec. 1/5; why eps*=2.96 stays grid-scoped |
| 4b: order-grid slack | `_DEFAULT_ORDERS` (89 Renyi orders) vs 8x-denser grid (`PEG_RDP_DENSE=1`, 761 orders, min order 1.03 vs 1.25) | budgets shrink 0.2-0.4% in the eps~1-3 region (30x the sawtooth) and up to **9.1%** at high budgets (sigma=0.28: 137.53 -> 124.96; optimal order < 1.25 at large eps) | measured 2026-07-15; jobs 180210 (default) vs 180211 (dense) rerun the cycle window under both — separating grid artifacts from accountant artifacts from order-grid slack |

**Reading the budget against the claims.** The finest load-bearing
margin in the campaign is the (47.14, 64) edge pair (utility tie
2.4e-5, one-sided derivative -6.5e-8) — resolved at layer 0 with 4-6
orders to spare. T=2 structural claims (segments, gaps) carry ~8
digits. The T-grid deliverable, gap(T, lambda) at signal scale
0.05-0.10, tolerates the ladder's worst case with a >=500x margin.
Policy: any number that decides a claim is Brent-polished to
layer-0/1 precision; sampling-resolution numbers are flagged as such
(currently only the quadratic-cost NE of Sec. 16 awaits polishing).

## 20. Pending analyses: pre-registered instructions

Skeletons for the analyses of every dataset currently in flight, so
the steps and decision criteria are fixed *before* the results are
seen. Execute per block when the job lands; each block names its
inputs, steps, decision rule, and outputs.

### 20.1 Unified gap(T, lambda) curve — the onset of moderation

**Inputs.** `50_t_grid.pkl` (job 180154, T in {1..48}),
`51_t_grid_hi.pkl` (180155, T in {64, 96, 128}), `52_t390_ladder.pkl`
(180205, T = 390, GDP-guided diagonal candidates), `53_gdp_gap.pkl`
(GDP limit, done). Row format: per (T, e1) a column of
(p2, sigma, e2, A2) plus the diagonal row (p2 = PSUM/2).

**Steps.**
1. Per (T, e1, lambda): u2 = 1/(1+sigma^2) - lambda*A2 per row;
   restrict to rows with e2 in [0.5, 64] (box) plus the diagonal row;
   gap(T, e1; lambda) = max(u2) - u2(diagonal). Sanity: diag_check
   column must be < 1e-8 everywhere; drop and report any ok=False rows.
2. gap_min(T; lambda) = min over e1, and e*(T; lambda) = argmin.
   lambda grid {0.5, 1, 2, 5}. NOTE: gap >= 0 by construction; "onset"
   = first T where gap_min hits 0 within tolerance AND e* is interior
   (e* > 0.5; a floor argmin means the floor regime, not moderation).
3. Cross-validate at overlaps: 48 vs 64 (different jobs, same engine),
   128 vs 390 vs GDP curve. Plot GDP gap(T) as the asymptote; report
   ladder-vs-GDP discrepancy at T = 390 (expected ~ CLT error, a few
   1e-3).
4. Onset extraction: for each lambda, interpolate log-T where
   gap_min crosses tolerance; if no crossing by T = 390, extrapolate
   with the GDP curve and report "onset in (390, T_GDP]".
5. Decision rule at T = 390, lambda = 1 (the accountant-conservatism
   hypothesis): u-uncertainty ~ 5e-4 (budgets 1e-4 + certified adv
   width 4.3e-4). If gap(390, 1) > 1.5e-3: the exact game is
   pre-onset — the paper instance's symmetric NE is materially
   stabilized by accountant conservatism (write this into Secs. 17/18
   and flag for the paper's discussion). If |gap| < 1.5e-3:
   borderline — report as "at onset within numerical resolution".
   If the diagonal is the global max (gap = 0 at interior e*): onset
   is at or before 390 — the hypothesis is refuted.
6. Advantage certification pass at T = 390: for each analysis-relevant
   (sigma, p2) (the diagonal and the argmax rows), run
   `double_ladder.compose(sigma, p2, 390, pess/opt)` -> certified
   Adv interval -> certified u interval; confirm the step-5 verdict
   holds on both interval ends.

**EXECUTED (2026-07-16; jobs 180427/180155/180205 -> 50/51/52 pkl;
174 columns, 0 diagonal self-check failures; figure `onset_curve.png`
= artifact Figure 15).** Verdicts:
- *(Superseded in its sampling detail by Sec. 28: the 21-point columns
  had no deviation within +-40% of the diagonal; the dense exact slice
  confirms existence and puts eps* at ~2.59, BR-stable.)*
- **T = 390, lambda = 1: VERDICT RESOLVED (2026-07-17, jobs
  180626/180627 -> 52b/52c pkl): the exact game HAS a symmetric NE,
  and it sits at the limit game's stationary point eps* = 2.548, not
  at the old candidate 2.92.** Exact-ladder deviation sweeps (21-point
  p2 grid, diagonal self-checks <= 1.4e-10) at e1 in {2.30, 2.43,
  2.548, 2.60}: at every candidate the diagonal is the argmax — gap =
  0 on the sampled deviations. At 2.548 the diagonal beats the nearest
  sampled deviations by 1.2e-3 (toward smaller e2) and 4.2e-3 (toward
  larger), margins ~3x and ~10x the certified advantage-bracket width
  (4.3e-4). The earlier "gap = 0.0010 at 2.920 certified positive"
  stands as stated — 2.92 is NOT an equilibrium — but it was a
  statement about the wrong candidate: the least-tempting symmetric
  point was off the coarse diagonal grid. Revised onset claim: at T =
  390, lambda = 1 the exact game's symmetric moderation NE EXISTS at
  eps* = 2.548 (grid caveat: deviations sampled at 21 p2 points; a
  between-point deviation is not excluded but the profile is smooth
  and unimodal around the diagonal). Consistency: sigma on the
  diagonal is 1.166, inside the CLT-valid regime, which is why the
  limit prediction lands on the exact answer here. Per Sec. 26 the
  point is a potential SADDLE with |BR'| = 1.045: it exists as an NE
  but is best-response unstable — existence and stability now
  separately settled, in opposite directions.
- **The temptation hump (24.7-6) is CONFIRMED** at every weight:
  gap rises from T = 1 (e.g. 0.011 at lambda = 1) to a peak around
  T ~ 3-8 (0.024-0.030), then declines to the onset. Onsets: lambda=2
  by T ~ 128; lambda = 1 just beyond 390; lambda = 0.5 pre-onset at
  390 (GDP: crossing in (1560, 6240]) — ordering as predicted.
- **e* = 2.92 agrees across three independent methods** (exact ladder
  2.92, GDP limit 2.92, RDP grid 2.96 — the 0.04 shift is the
  accountant's conservatism).
- Caveats: gap_min jitter across T is e*-selection noise (12 coarse
  diagonal candidates); the lambda = 5 column at T = 390 is
  unusable (the 52 run's custom diagonal set {1.5..8.0} omits the
  floor region where that weight's least-tempting point lives).
- GDP-vs-ladder at the overlap (register P7): gap difference at
  (390, lambda=1) is 0.0027 — inside the a-few-1e-3 CLT expectation;
  e* identical. The limit model is quantitatively usable for T >~ 390
  with ~3e-3 gap error.

**Outputs.** Figure (gap_min vs T, log-x, per lambda, GDP asymptote
overlaid, onset markers) -> artifact Figure 10; results paragraphs
into Sec. 17 (T <= 390) and Sec. 18 (validation of the GDP limit at
the overlap); Sec. 19 row 2 updated with realized T=390 margins.

### 20.2 Cycle window at 16x grid resolution (default orders)

**Inputs.** `55_cycle_zoom.pkl` (job 180210; K = 220 on
[0.90, 1.20]^2, T = 390, default 89-order accountant — same game as
the K=200 cache).

**Steps.**
1. Build lambda = 2 utilities via `sweep_from_grid(mech, 2, 2)`.
2. **Boundary audit first**: per row/column, the window argmax must be
   interior (index not in {0, K-1}); report the fraction of
   boundary-hitting replies. If any trajectory-relevant reply hits the
   boundary, the window dynamics are invalid there — enlarge the
   window and rerun before interpreting.
3. Alternating BR dynamics from >= 8 starts inside the window (reuse
   `54_cycle_trajectory.py` logic); detect the attractor: collect the
   tail states, find the minimal repeating period (test periods
   1..12 on the index sequence).
4. Compare to K = 200: cycle period, the set of visited budget values,
   loop amplitude (max - min of visited eps). Overlay both loops in
   one panel.
5. Pure-NE check on the window (find_pure_NE_2player): the K = 200
   game has none at lambda = 2 — confirm none appears at 16x
   resolution (a fine-grid pure NE would mean the cycle was a grid
   artifact).

**Decision rule.** (a) Same period and amplitude (values shift by
< one K=200 cell): cycle is grid-robust. (b) Longer-period or
finer-amplitude orbit: the 3-cycle was the grid's quantization of a
smaller/richer orbit — report the resolved structure. (c) A pure NE
appears: the cycling claim in Sec. 11 was a grid artifact — rewrite
Sec. 11 and the artifact Figure 9 caption accordingly.

**EXECUTED (job 180210; 2026-07-15).** Boundary audit clean (0/220
edge hits either direction — window analysis valid). Outcomes (b) and
(c) *both* fired, with a twist that changes their meaning:
- A pure NE pair appears at 16x resolution: (1.0090, 1.0116) + mirror
  — two budgets 0.26% apart that K=200 quantized into one cell. BUT
  its strictness margins are 2.6e-6 / 5.4e-6, roughly **20x below the
  accountant sawtooth (~1e-4)**: the equilibrium is held in place by
  payoff differences smaller than the calculator's own rounding
  texture. Symmetrically, K=200's "no pure NE" was below *its*
  resolution. Neither statement is a property of the game; both are
  properties of the discretization sitting on a ~1e-3-flat,
  1e-4-textured plateau.
- The orbit persists and reshapes: a 2-state loop over four budget
  values {1.006, 1.018, 1.049, 1.058} captures 56/60 random starts
  (the K=200 3-cycle over three values was its coarse quantization);
  the pure NE's basin is 4/60.
- Robust content for Sec. 11 (pending the dense-order twin 180211):
  at lambda = 2 the diagonal neighborhood is a near-indifference
  plateau on which play *orbits* under every non-damped protocol
  (Sec. 24.2) at every resolution tried; isolated pure equilibria on
  the plateau exist or vanish depending on discretization, with
  margins so small (<= 5e-6) that they are economically meaningless —
  any perturbation of the accountant erases or moves them. The
  cycling-as-phenomenon claim survives; the "no pure NE" claim must be
  restated as "no economically meaningful pure NE" with margins
  quoted. Final wording gated on 180211 (does the dense-alpha grid
  move/destroy the pair, as the margin analysis predicts).

**REQUIRED DISCUSSION — two best-response objects with different
dynamics.** The campaign has established that there are two distinct
games hiding under one name, and the paper must discuss both
explicitly rather than blending them:
1. the **theoretical game**: budgets accounted exactly (closed forms
   at T=1, certified quadrature at T=2, the ladder beyond). Its
   best-response curves are smooth; its equilibria are the frontier
   segments, floor pairs, and (at composition scale) the symmetric
   moderation point.
2. the **deployed game**: budgets accounted through a Renyi-divergence
   accountant with a finite order grid — which is what real systems
   actually run. Its best-response curves are sawtooth staircases
   (Figure 11): the argmax rides an accounting tooth and snaps at
   order switches. Fine-scale strategic objects — phantom near-tie
   equilibria, the exact orbit values, snap points — belong to the
   accountant, not the game.
The two have **different dynamics**: smooth BR curves support the
corner/segment/moderation story; sawtooth BR curves additionally
generate tooth-hopping orbits and discretization-dependent rest
points. A participant optimizing against a deployed accountant lives
in game (2); the theory describes game (1); the difference is
measurable (order-grid slack 0.2-9%, Sec. 19 row 4b) and shrinks as
the order grid densifies — the 180211 run was meant to test exactly how fast — it TIMED OUT at
10 h (dense orders are ~4x slower; the K = 220 sweep needs ~24 h or
a smaller window). Resubmission pending — the sawtooth-shrink test
remains open.
Neither object is "the wrong one": the paper should present (1) as
the mechanism's economics and (2) as its implementation reality, with
the sawtooth as the quantified gap between them.

### 20.3 Cycle window under the dense order grid

**Inputs.** `56_cycle_zoom_dense.pkl` (job 180211; identical window,
761 orders, min order 1.03 — budgets tighter by ~0.2-0.4% here).

**Steps.** Repeat 20.2 steps 1-5 on the dense grid; then the
three-way comparison (K200 / zoom-default / zoom-dense): sawtooth
amplitude on the BR curves (quantify tooth height before vs after),
cycle period/amplitude/location shift between order grids.

**Decision rule.** (a) Cycle persists with same structure under dense
orders: cycling is model-intrinsic (Sec. 11's claim upgraded from
assertion to evidence). (b) Cycle deforms materially or collapses to
a pure NE: cycling at lambda = 2 is partly an accountant artifact —
add the caveat to Sec. 11, connect to the order-grid-slack row (Sec.
19, 4b) and to the accountant-conservatism hypothesis (Sec. 18).
Either way: quote the measured sawtooth reduction.

**EXECUTED (2026-07-17; job 180594 -> `56_cycle_zoom_dense.pkl`;
comparison in-session against 55). Decision rule outcome: (b), and
decisively.** Numbers (identical 220-point window [0.90, 1.20],
lambda = 2, only the order grid changes, 89 -> 761):
- **Payoffs shift by up to 7.9e-4 (median 1.3e-4)** — orders of
  magnitude above the BR decision margins (median 2.1e-6), so best
  responses move on 98% of columns (median move 0.014).
- **Sawtooth refines exactly as the discretization hypothesis
  predicts**: max tooth amplitude shrinks 3x (0.064 -> 0.023) while
  the tooth count grows (52 -> 119 jumps > 3 grid steps) — the
  staircase gets finer, it does not disappear. Amplitude scales with
  order-grid spacing.
- **The 2-cycles collapse onto pure NE.** Under 89 orders the
  window's alternating-BR attractors are two 2-cycles, and its only
  pure NE is the phantom pair (1.0090, 1.0116)/(1.0116, 1.0090)
  (margins below the tooth height, and NOT an attractor from the
  sampled starts); under 761 orders alternating BR converges from
  every sampled start, and the window holds 4 pure NE clustered at
  eps in [1.025, 1.029] — two of them exactly symmetric ((1.0263,
  1.0263) and (1.0290, 1.0290)). Cycling at lambda = 2 in this window
  is therefore ACCOUNTANT-CONDITIONAL and Sec. 11 must carry the
  caveat.
- The old phantom NE cell (1.0090, 1.0116) is again not an NE under
  dense orders (regrets 1.5e-5 / 4.6e-6) — near-tie NE locations stay
  conditional; what is robust is the CLUSTER, which the dense grid
  tightens ~6x (89-order orbits spanned [1.006, 1.058]).
- Margins of the 4 dense NEs are 4e-7 to 8e-6 — still near-tie
  against the residual (finer) discretization, so quote locations as
  a cluster, not points.
- **Follow-up submitted:** the main K=200 cache is lambda-independent,
  so one dense-order recompute re-derives every lambda claim with the
  finer accountant — job 180899 (`run_mech_dense.py` ->
  `mech_grid_..._e5_dense.pkl`; the 89-order cache and all its
  consumers untouched). Pre-registered analysis: rerun the wsweep NE
  census (no-pure-NE-at-lambda-2, corner NE at lambda = 4) and the
  Sec. 27 wall-string census on the dense cache; expect wall strings
  to persist (wall-pinned, margin-insensitive) and interior near-tie
  structure to shift within its clusters.

### 20.4 Quadratic-cost equilibria to campaign standard

**Inputs.** `39_t1_lam_low.pkl` components; `single_step_game.py`.

**Steps.** For lam_q in {750, 1000, 1778, 3162}: Brent-polish the
diagonal stationary point of u = V - lam_q*Adv^2 (FOC =
Vterm - 2 lam_q Adv Advterm on the diagonal via fresh exact solves,
not cache interpolation); verify global optimality along the exact
column (trace with the quadratic objective); confirm the
symmetric-interior-NE window edge between lam_q = 562 and 750 by
bisection on the deviation gap.

**Output.** Replace the "sample-resolution, not Brent-polished" caveat
in Sec. 16 with exact values; one row into the Sec. 19 table (layer 3
item resolved).

### 20.5 lambda = 1 cycle window (job 180216 -> `57_cycle1_zoom.pkl`)

Same protocol as 20.2 on the window [2.0, 3.7]^2 (the two mirror
3-cycles found by the basin census, Sec. 21). Null hypothesis:
*plateau artifact* (argmax jitter on near-ties).

**EXECUTED (2026-07-15).** Boundary audit: 3/220 edge hits per
direction (dominant attractor interior — analysis valid). Verdicts:
- **The orbiting is real (robust to 16x refinement); the specific
  orbit is not.** A 4-state loop over budget values ~{2.09..3.14}
  captures 54/60 random starts (plus a minor 7-state orbit, 6/60);
  the K=200 "3-cycles" were its coarse quantization. The basin
  census's orange class survives as a phenomenon.
- **No pure NE anywhere in the window at 16x resolution — including
  the paper's symmetric equilibrium.** The K=200 grid NE at
  (2.9646, 2.9646) has no counterpart on the fine grid: within the
  window, every candidate cell admits an in-window improvement. Same
  epistemic status as the lambda = 2 case (Sec. 20.2): on the
  1e-4-textured plateau, existence of isolated grid equilibria is a
  property of the discretization. **The paper's symmetric-equilibrium
  claim must not rest on the grid object**; the load-bearing version
  of the moderation story is the exact-ladder gap analysis (Secs.
  17-18), with the grid point (2.96) reported as the plateau's
  location, not as an equilibrium point.
- Caveat: default 89-order accountant (sawtooth present); a dense-
  order twin for THIS window was not run — queue one if the lambda=2
  dense twin (180211) shows order-grid sensitivity of orbit
  structure.

### 20.6 Bookkeeping after each analysis

Update: the data/artifact index (rows for 55/56 + figures), the open
items list, task #6, memory (`exact-equilibrium-campaign`), and the
artifact page (figure numbering continues from 9; keep one figure per
finding, captions self-contained). Any claim that changes a number
quoted earlier in this file: edit the earlier section in place (this
file follows Rule 0 locally — no "previously we thought" prose).

## 21. Basins of attraction: who ends up where, from where

**Setup.** Alternating best-response dynamics on the K=200 cache from
*every* feasible start (player 2 replies first), attractor detected by
state repetition, lambda in {1, 2}. Figure (2-D class-colored basins
with legend + sample trajectories): `tasks_findings/_scripts/57_basins.py`
-> `paper/figures/data/foc_landscape/basins.png` (artifact Figure 10).
`58_basin_outcome.py` computes the 1-D outcome data (resting budgets
of both players vs the first mover's commitment) that backs the
commitment-keeping findings below; its chart form was tried and
rejected as less readable than the 2-D map.

**Results.**
- **The responder's start is irrelevant — basins are functions of the
  first mover's budget alone** (vertical stripes). After one reply the
  initial eps2 is forgotten: whoever shows up first with a committed
  budget selects the long-run outcome. Alternating play is implicitly
  Stackelberg; role assignment is first-mover assignment.
- **lambda = 2: one attractor, 100% basin** — the undercutting 3-cycle
  is globally attracting (upgrades Sec. 11's "from every start tried"
  to "from every feasible grid start"). Convergence within <= 3
  rounds.
- **Commitment-keeping regions.** In the "other segment" ranges the
  first mover's resting budget equals the commitment exactly (dots on
  the identity line): the responder jumps straight to the frontier and
  the pair absorbs at (x, f(x)) — the first mover *chooses their own
  segment point*. Whether the kept commitment is the tight seat
  (responder loose above) or the loose seat (responder tight below)
  flips with the commitment level; the blue ranges instead bounce play
  to the dominant wall pair, with the role again set by the
  commitment's magnitude (commit ~1.6 -> hold the protected seat;
  commit ~10.8 -> carry the batch).
- **lambda = 1: 45 attractors, and the symmetric equilibrium is
  dynamically marginal.** The two mirror **3-cycles around the
  diagonal window** (states ~{2.27-3.43}) capture 33.5% of starts;
  the dominant wall pairs (1.612, 10.794) + mirror capture 22.9%; the
  remaining ~40 attractors are individual frontier-segment points with
  slivers of basin (history parks play at whichever segment point it
  first hits — the continuum's selection is pure path dependence).
  The symmetric NE (2.965, 2.965) attracts **1.6%** of starts.
  Convergence everywhere within <= 6 rounds.

**Caveats — updated with test outcomes.** The lambda = 1 diagonal
cycles: TESTED (job 180216, Sec. 20.5) — the orbiting survives 16x
refinement (a 4-state loop, 90% basin in the window), so the orange
class is a real phenomenon, though its specific states were
grid-quantized; and the fine grid removes ALL pure rest points in the
window, including the (2.9646, 2.9646) grid NE. The first-mover
result is alternating-protocol-specific (Sec. 24.2).

**Interpretation for the paper.** Two sentences the discussion can
carry: (i) equilibrium *selection* in the PEG is pure path dependence
— the continuum of frontier equilibria has no focal point, the first
committed budget picks the split; (ii) at the paper's own weight the
symmetric equilibrium, though it exists, is nearly unreachable by
myopic play — most of the diagonal's neighborhood orbits it instead —
consistent with T = 390 sitting at the moderation onset (Sec. 18).

## 22. Alternative plays: commitment, collaboration, fairness, and how
T rescales them

All computations cache-only or closed-form
(`tasks_findings/_scripts/59_alt_plays.py` ->
`59_alt_plays_summary.txt`; K=200 cache via `sweep_from_grid` for
arbitrary weights, `C2.pkl` for group-size asymmetry,
`gdp_limit_game.py` for high T). Resolution caveats: K=200 grid, C2 is
a 9-point action grid (direction-reliable only), GDP-limit numbers
carry CLT error (a few % at T=390, shrinking in T). The summary file
samples the GDP columns slightly coarser than the runs quoted below,
so its limit-model entries differ at the few-percent level (register
row D3); cache-based entries (a)-(c) match exactly.

### 22.1 Stackelberg commitment (T=390 cache)

Budget declarations *are* commitments, and the basin analysis (Sec.
21) showed alternating play is de facto leader-follower; Stackelberg
is that logic optimized. lambda=1: the leader commits **tight**
(eps=1.61), conscripting the opponent into carrying the batch: leader
0.591 / follower 0.267, vs 0.322 each at the symmetric NE — the
leader's best commitment is *less* participation, not more. lambda=2
(no pure NE): commitment **cures the cycling** — leader 0.0928 /
follower 0.0962, both above the cycle time-average (~0.0899) and the
mixed value (0.0888); a second-mover advantage appears (the follower,
replying to a frozen target, does better).

### 22.2 The collaborative game (u = V - lam(A1 + A2)), identical
players

The team optimum is **asymmetric**: (3.35, 24.13), welfare 0.941 —
one carrier, one protected — vs 0.885 at the best equal-budget
(diagonal) point: a **~6% fairness premium**. Root cause is Sec. 12's
theorem: composed exposure is *concave in the rate*, so an equal
batch split maximizes total exposure and concentration minimizes it.
Selfish play is close to efficient (best NE welfare 0.859, ~9% below
the planner) *and* picks asymmetry for the planner's reason; the
catastrophic regime is not lopsidedness but **cycling** (lambda=2:
0.180 of 0.533 achievable — two thirds of welfare burned).
Implementations that make cooperation self-enforcing: budget-matching
(mechanism enforces eps1 = eps2; the game collapses to the diagonal
where interests coincide — unanimity at 11.33, no transfers, operator-
enforceable; costs the 6%), joint liability (charge each player total
group leakage: the team optimum becomes a Nash equilibrium, but the
carrier needs a side payment), and the named classics (Lindahl
pricing; Kantian equilibrium = the diagonal optimum with a norm
instead of a mechanism). Unaided, fair shares are hopeless: unilateral
deviation from (11.33, 11.33) gains +0.16.

### 22.3 Heterogeneous players: who should carry

Assignment rule (closed form at T=1, direction confirmed at T=390 and
in C2): **the batch goes to the player minimizing lambda_i / |G_i|** —
privacy weight per unit of data capacity. Low-weight players and big
groups carry; big groups enjoy *economies of scale in privacy* (same
batch spread over more records = lower per-record rate = concavity
bonus). Two corollaries with numbers (lam2/lam1 = r, T=390):

| r | planner W | equal-budget W | best-NE W |
|---|---|---|---|
| 1 | 0.941 | 0.885 | 0.859 |
| 2 | 0.940 | 0.674 | 0.858 |
| 4 | 0.938 | 0.360 | 0.856 |
| 8 | 0.935 | **-0.058** | 0.853 |

- **Equal treatment is not fair treatment.** Enforced equal budgets go
  from a 6% premium (equals) to worse-than-shutdown at r=8. If
  matching is wanted, match per-record burdens (equal budgets between
  equal-size groups already give batch shares proportional to |G|),
  never raw budgets across heterogeneous players.
- **The market sorts but does not compensate.** Best-NE welfare is
  ~0.86 at *every* r: decentralized play assigns the carrier role
  correctly (the C2/C8 sorting seen from the welfare side) and loses
  only ~9% — but no equilibrium ever pays the carrier. Efficient
  assignment is free; fairness needs a transfer.

### 22.4 High T: conscription dies, fairness premium explodes

GDP limit, lambda=1: through T~1500 the tight commitment still
conscripts (leader 0.54/0.36; the follower's payoff collapsing 0.30 ->
0.09 toward refusal); by T~6000 the follower prefers the floor against
*any* commitment — first-mover advantage is worth nothing. Meanwhile
the fairness premium grows 1.5% (390) -> 9.8% (1560) -> 88% (6240) ->
172% (25000; the fair arrangement has negative welfare while
specialization still clears +0.08): saturation makes symmetric
meaningful participation ruinous for both, while concentration caps
the damage at one player. Combination: at high T the welfare-positive
arrangement is asymmetric, but no one can be conscripted into it and
no one volunteers — **transfers stop being a fairness nicety and
become the only instrument that implements the optimum**.

### 22.5 T-invariance at fixed contracts: composition starves value,
it does not inflate exposure

In the GDP limit the map eps -> mu is T-free, the rates split PSUM in
proportion to the mu's, and T is absorbed entirely by sigma: **given
the contracts, every player's exposure is T-invariant; composition
only deflates V(sigma(T))**. Consequences: (i) the lambda=1 floor
collapse at T=25000 is value starvation, not exposure inflation — the
common intuition "more rounds -> more leakage pressure -> act more
privately" is backwards at fixed contracts; (ii) no lambda-rescaling
reproduces a lower-T game exactly (the sqrt(T) rule overshoots into
the ceiling regime because saturation caps the maximum penalty at
lambda while the value scale shrank); (iii) empirically the
"equivalent" weight drifts like a fractional power of T —
lambda_equiv(25000) ~ 0.35 (roughly T^(1/4) over 390 -> 25000)
reproduces the baseline's regime mix, working Stackelberg conscription,
and ~6% fairness premium.

### 22.6 Concepts examined and rejected

**Repeated-game collusion:** inapplicable — the declaration is a
one-shot contract; the T training rounds are mechanism internals, not
stage games (no per-round action, payoff, or observability).
Only cross-*run* repetition (successive trainings between the same
parties, each run one full PEG) would support folk-theorem logic;
that is a model extension, not a reading of these results.
**Relative/spiteful play** (maximize u1 - u2): degenerate — the shared
V cancels identically at every T, leaving a pure exposure race to the
floor; all value destroyed, nothing to analyze.

### 22.7 The intuition ledger: working hypotheses, each with its
evidence scope

These held in every configuration *computed so far* — specific
(lambda, |G|, T) values, one utility form V = 1/(1+sigma^2), one
mechanism (Poisson-subsampled Gaussian), a fixed action box, and (for
dynamics) one update protocol. They are hypotheses to stress-test,
not theorems; items 6-7 additionally depend on the GDP-limit model.

1. Budgets are commitments, and commitment is the game's real power:
   the first committed budget selects the outcome (Sec. 21), and the
   optimal commitment is tightness — conscription, not aggression.
2. Composed exposure is concave in participation; therefore efficiency
   always prefers concentrating the batch, and equal shares are the
   most exposure-expensive allocation that exists.
3. Fairness is cheap only between equals (6%); under heterogeneity,
   equal treatment is the worst policy on the table, and the fair
   policy is efficient assignment (min lambda/|G| carries) plus a
   transfer.
4. Selfish play gets the assignment right and the compensation never:
   ~10% from optimal at every heterogeneity level tried, with the
   carrier systematically unpaid.
5. The expensive failure is cycling, not inequality (2/3 of welfare
   at lambda=2 vs ~10% anywhere else); commitment is a sufficient
   cure for cycling.
6. At fixed contracts, more composition starves value rather than
   inflating exposure (exact in the GDP limit); the interesting game
   lives where lambda balances the value scale, and that balance
   drifts sub-sqrt(T).
7. At high T, every unilateral instrument dies (commitment,
   volunteering) while specialization becomes mandatory — transfers
   are the last instrument standing.
8. Preferences over *relative* outcomes destroy the game entirely,
   because the value is shared: any status-seeking component in the
   utility is an existential threat to the system's value.

## 23. Register of assumptions, approximations, and discretizations

Everything below is load-bearing somewhere. Sec. 19 covers numerical
error; this section covers what is *assumed or discretized by
construction*. Tags: [A] modeling assumption, [P] approximation,
[D] discretization. Each entry names the claims it can bite.

**Model definition [A].**
- A1 Two players, equal group sizes except where varied (C2, Sec. 13).
  All continuum/segment geometry is 2-player; n-player untested here.
- A2 Utility u = V - lambda*A: *linear* exposure disutility, single
  weight; V(sigma) = 1/(1+sigma^2) is a qualitative proxy (convex-ERM
  bound). The two-phase slice anatomy (Sec. 15), the corner taxonomy,
  and the trough all depend on V's saturation shape — no other V has
  been tried (cheap T=1 sensitivity run pending). Convex disutility
  (Sec. 16) is the one utility variant computed.
- A3 V is a pure public good (identical for both); A_i is the
  worst-case per-record MIA advantage of the own group, attacker sees
  all rounds. Sec. 22's relative-play degeneracy is a direct corollary
  of A3, not a finding beyond it.
- A4 Mechanism: Poisson-subsampled Gaussian, *shared* sigma, batch in
  expectation (E[batch] = 128), both budgets binding by construction,
  participation floor p_bar = 1e-5. The participation frontier — and
  hence the entire equilibrium continuum — is an artifact of shared
  sigma + fixed batch + floor; per-player noise would dissolve it.
- A5 Action box eps in [0.5, 64], delta = 1e-12 fixed. Edge-pair NEs,
  ceiling NEs, and high-T box compression (Sec. 22.5) are properties
  of the box as much as of the game; box sensitivity untested.
- A6 One-shot simultaneous declarations for equilibrium analysis;
  *alternating* best response for all dynamics (Secs. 11, 21). The
  first-mover determinism of Sec. 21 is a property of that protocol;
  no other protocol has been run yet.
- A7 Complete information: both players know both utilities/weights.

**Accounting of the paper instance (T = 390) [P].**
- P1 RDP composition is an upper bound on eps, not the exact profile;
  exact-vs-RDP budget gap at T = 390 unquantified until job 180205
  lands (the GDP-limit comparison suggests a few % near the symmetric
  point, and Sec. 18's onset reading depends on it).
- P2 Discrete Renyi order grid: the ~1e-4 payoff sawtooth *and* the
  0.2-9.1% slack (Sec. 19 row 4b). Every T=390 equilibrium object
  (2.96, the 42-NE string, the lambda=2 cycle, basins, Stackelberg
  argmaxes) is conditional on the 89-order grid; dense-order zoom
  (job 180211) is the pilot test.
- P3 PLD advantage at value-discretization 1e-3 (pessimistic CTD);
  its error vs the exact TV is unquantified — checkable with the
  bracket engine, which certifies advantages (Sec. 17).

**Exact-campaign machinery [P].**
- P4 T=2 quadrature: mpmath-certified at ~10 hand-picked plus 40
  randomized (sigma, p) configurations at both game-relevant scales
  (`63_mpmath_coverage.py`): worst 3.7e-10 relative at the advantage
  scale, 1.1e-7 at the budget scale — resolved from 'extrapolated'
  to 'randomly sampled coverage'.
- P5 Ladder: spline of tabulated levels, fixed panels, tail
  truncation, graded-grid heuristics; validated at anchors and by
  doubling at *selected* (sigma, p, T), one-sided budget bound at spot
  points — coverage is spotty by nature, not uniform.
- P6 Bracket engine: advantage certificates modulo non-directed FFT
  roundoff ~1e-15; budget side unusable (Sec. 17).
- P7 GDP limit: CLT model, error O(1/sqrt(T)) *not yet measured
  against the ladder*; the mu formula is itself the limit object. All
  of Secs. 18, 22.4, 22.5 (onset location, conscription death,
  premium explosion, T-invariance of exposure, lambda_equiv ~ 0.35)
  are limit-model statements pending the overlap validation.

**Discretizations in analyses [D].**
- D1 K=200 action grid (2.47%/cell): all cache-based argmaxes (BRs,
  NEs, basins, cycle states, Stackelberg commitments, welfare optima)
  are quantized to cells; near-ties on the 1e-4 diagonal plateau are
  below the payoff resolution that separates cells.
- D2 Exact-game columns: 100-160 log samples; interior BRs are
  Brent-polished where load-bearing, sample-grid elsewhere (the
  8.43 -> 8.79 episode; quadratic-cost NE still at sample level).
- D3 Sweeps: 51-point lambda scans, 12 x 31 T-grid columns, 30-41
  point GDP columns, 9-point C2 grid — regime boundaries, gap minima,
  and welfare optima are only as sharp as these grids.
- D4 Detection tolerances: 2e-2 mutual-consistency, 5e-2 dedupe,
  window-restricted argmaxes in zooms (boundary audits mandatory).
- D5 Dynamics are grid dynamics: cycles, periods, and basins are
  statements about the K=200 finite game; the zooms test grid
  refinement, not continuity.

**How to use this register.** Before promoting any claim to the
paper: name its row(s) here; if a row is untested (A2/A5/A6 box,
utility, protocol; P7 overlap), either run the sensitivity check or
scope the claim in the text. The three-way artifact table planned in
Sec. 20 (grid / order grid / plateau) covers P2 + D1 + D5 for the
cycle claims; nothing yet covers A2, A5, A6 — the cheapest
highest-value sensitivity runs still open.

## 24. Holistic robustness program: plan and execution log

Goal: resolve every open row of the register (Sec. 23) — jointly, not
one anecdote at a time. Two instruments: (A) a parametrized exact
engine sweeping *all* assumption axes at once, and (B) targeted checks
for the rows the sweep cannot reach. Verdicts land in the invariance
matrix (24.6); paper claims then cite matrix rows.

### 24.1 The harness (instrument A) — job 180346

`game_experiments/robust_game.py`: the exact game with every register
assumption as a constructor knob — T in {1, 2} (closed form /
certified quadrature; no accountant), group sizes (n >= 2 players),
participation floor, contract delta, action box, value shape V in
{1/(1+s^2), 1/(1+s), e^-s}, cost shape in {linear, quadratic}, weights
(incl. heterogeneous). `run_robustness.py` runs a fixed battery per
configuration: b1 trough presence (cost-before-value), b2 corner
dominance (BR kind census), b3 equilibrium-taxonomy flags, b4
symmetric deviation gap, b5 specialization premium, b6 Stackelberg
value/role. Design: baseline + one-factor-at-a-time on all nine axes
(main effects) + 160 jointly randomized configurations (interactions)
+ two 3-player spot checks + a 10-configuration T=2 replication.
Output `60_robustness.pkl`; analysis: per finding, the fraction of
configurations where it holds, split by axis — the **invariance
matrix** (finding x axis -> holds / flips at known boundary / scoped).
Baseline smoke reproduced known values (kind census 6:1 hi:lo at
lam=0.7; specialization premium exactly 0 at T=1, confirming the
premium is a composition effect).

### 24.2 Protocol robustness (instrument B, register A6) — DONE

`62_protocols.py`, K=200 cache, 250 random starts per cell, four
update styles. Results:

| protocol | lam=1 | lam=2 |
|---|---|---|
| alternating (baseline) | 32% cycles, 68% fixed pts (wall pairs dominant) | 100% the 3-cycle |
| simultaneous | 97% cycles — but dominated by 2/6-cycles, the classic simultaneous-BR *seat-swapping* pathology | 100% cycles (6- and 3-) |
| random order | 38% cycles, wall pairs dominant — matches alternating | 100% cycles (mixed lengths) |
| damped (half-step in log) | 60% 2-cycles + off-NE rest points | 0% cycles — all park near (1.04, 1.04) |

Verdicts: the lambda=2 cycling is **robust to who moves when**
(simultaneous, random order) — not an artifact of turn-taking. The
damped rows are **inconclusive on this grid**: half-steps smaller than
the 2.47% cell quantize to zero, so damped play stalls at rest points
that are not equilibria (lambda=2 has no pure NE — those (1.04, 1.04)
parkings are grid stalls, not convergence). Damped dynamics needs
continuous actions or the 16x zoom grids -> follow-up 24.4(d). The
first-mover determinism of Sec. 21 is confirmed to be
alternating-specific (random order spreads outcomes over the same
attractor family without single-player determinism).

### 24.3 Analyses pre-registered on running jobs (Sec. 20)

Unchanged and binding: 20.1 gap/onset curve (jobs 180154/155/205) with
certified-advantage pass; 20.2/20.3 cycle zooms incl. dense Renyi
orders (180210/211); 20.5 lambda=1 plateau cycles (180216); GDP-vs-
ladder overlap validation (register P7) as part of 20.1 step 3.

### 24.4 Targeted follow-ups, each gated on a landing

(a) **Dense-order K=200 recompute** (register P2): gate = zoom
    comparison 180210 vs 180211 shows equilibrium objects moving by
    more than one K=200 cell. If triggered: full mech-grid job with
    PEG_RDP_DENSE=1 + downstream cache regeneration (task #2 merge).
(b) **PLD-advantage discretization** (register P3): 180205 landed;
    still to run — certify Adv at its solution points with
    `double_ladder.compose` and compare against
    `mia_advantage_subsampled_gaussian` at discretization 1e-3;
    quantifies the paper's advantage-side error.
(c) **GDP overlap validation** (register P7): ladder vs
    `gdp_limit_game` budgets/advantages/gaps at T in {48, 64, 96,
    128, 390}; publishes the empirical CLT error curve; decides which
    Sec. 18/22.4/22.5 claims are quotable.
(d) **Damped dynamics without grid stalls** (from 24.2): damped BR on
    the zoom grids (0.125% cells) and/or the exact T=1 game; settles
    whether gentle play escapes the cycle.
(e) **Quadratic-cost polish** (Sec. 20.4, register D2): Brent-polish
    the symmetric interior NE window; small local job.
(f) **mpmath coverage** (register P4): extend the T=2 adjudication
    from ~10 hand-picked to ~40 randomized (e, sigma, p) points
    including budget-scale cases; small local job.
(g) **Stackelberg/welfare on exact games** (register D1): harness b5/
    b6 already replaces the grid argmaxes at T in {1, 2}; the T=390
    cache numbers stay quoted as grid-resolution.
(h) **n >= 3 players** (register A1): harness includes 3-player
    diagonal spot checks; a full n-player treatment (frontier
    geometry, role census) is scoped OUT of this campaign — recorded
    as future work, not silently dropped.

### 24.5 Paper-facing items (after 24.1-24.4 verdicts)

Invariance-matrix figure/table for the appendix; evidence-scoped
rewrite of every campaign claim that enters the paper; the
claim x tooling-threat x verdict table (Sec. 20's three-way artifact
comparison plus 24.2's protocol column); figure set per the step-back
review (slice decomposition two-panel, T=1 bifurcation, onset curve,
one dynamics figure).

### 24.6 Invariance matrix — FILLED (job 180346; `66_invariance.py`
-> `66_invariance.txt`; 193/193 configs succeeded; figures
`67_invariance_figs.py` -> `invariance_heatmap.png` +
`invariance_dials.png`, artifact Figures 13-14)

| finding | held | flips along | scope statement for the paper |
|---|---|---|---|
| corner-dominated best responses | 122/191; **107/110 under linear cost** (the 3 misses: high-lam interior windows) | cost shape (quad: 15/81) | "With linear exposure pricing, best responses are corner-driven across every value shape, weight, group size, floor, box, and contract delta we tested; convex pricing makes interior replies generic." |
| trough (worst-compromise dip) | 107/191; 47/49 at mid weights | weight (tiny lam: 13/31), group size (G50000: 8/36), narrow box (8/35) | "The dip exists exactly where privacy binds inside the action space — it vanishes when exposure is nearly free (tiny weight or very large groups) or when the box cuts off the rate takeoff." |
| specialization premium > 1e-4 | 34/191 | heterogeneity (17/37 het vs 17/154 hom), group size (G500: 13/44; G50000: 0/36), weight (0/31 tiny) | "At T <= 2 the fairness premium is usually ~0; it turns on with heterogeneity, small groups, and larger weights. The 6% premium at T=390 (Sec. 22.2) is composition-scale concavity, NOT a small-T universal." |
| fair-shares strictly better | 12/191 | (mostly quadratic cost) | convex pricing can make equal shares the *efficient* split — consistent with Sec. 16's mechanism |
| diagonal optimum interior | 26/191 | group size (G500: 16/44; G50000: 0/36) | interior symmetric optima at T <= 2 are a small-group (high rate-ceiling) phenomenon |
| symmetric moderation stable (gap < 1e-3 at interior diag) | 21/191, concentrated at G500 (14/44), never at G50000 | group size | **DOWNGRADED**: the stationarity criterion (Sec. 25.7) found no diagonal stationary point in the first re-screened world — the battery's value-level test over-detects on flat/corner coincidences. All 21 worlds need criterion re-screening before this row is quotable. |
| n = 3 spot checks | both: diagonal optimum at the box ceiling at lam = 0.7 | — | matches the 2-player baseline regime; full n-player treatment remains future work |

### 24.7 World-level notes — SIGNALS ONLY, each needs targeted
follow-up

**Status caveat, read first.** Everything in this subsection is a
*signal* from the screening battery, not an established finding: each
bullet rests on one or a handful of configurations, evaluated at
screening resolution (7 columns x 48 samples, no Brent polish, value-
level checks only — register rows D2/D3 apply in full). The battery is
designed to point at where to look, not to settle anything. Before any
of these enters the paper or the ledger, it needs its own targeted,
polished computation; suggested follow-up is noted per item.

- **The three linear-cost worlds with interior best replies are all
  "high effective weight" worlds**, across *different* value shapes:
  the known w=28 baseline window; an exp-value, unequal-groups world
  at w=38; a small-group (G=500) world at w=2.5 in a narrow box. The
  interior window is not special to V = 1/(1+s^2) — it tracks
  effective exposure weight wherever it appears.
  *Follow-up: polish the three windows (Brent on the exact FOC) and
  map the window boundary as a function of effective weight across V
  shapes — one small sweep.*
- **Progressive pricing fails to create interior play exactly where
  exposure is unreachable**: 5 of the 6 sampled quad-cost corner
  holdouts are G=50000 worlds (exposure diluted ~1e-3; quadratic cost
  ~w*1e-6 — the price never bites inside the box, so the race to the
  ceiling survives).
  *Follow-up: confirm with a wq scan at G=50000 that interior play
  returns once wq ~ 1e5 (the scaling prediction).*
- **Every fair-shares-win world is a progressive-pricing world**, and
  they concentrate where exposure is intense (G=500: premiums down to
  -12.7%). Convex pain + heavy exposure makes equal splits *efficient*
  — the exact mirror of Sec. 16's stabilization mechanism.
  *Follow-up: polish the largest cases (-8.6%, -12.7%) to campaign
  precision; verify the welfare argmax is not a sampling artifact of
  the 7x48 grid.*
- **Stable symmetric moderation at T=1 does NOT require progressive
  pricing**: four of the small-group stable worlds have plain linear
  cost (w ~ 6-11, G=500). Intense per-record exposure alone can
  stabilize the middle ground one-shot.
  *Follow-up EXECUTED (Sec. 25.7): the stationarity criterion finds
  NO diagonal stationary point in the tested world (V=1/(1+s),
  w=8.28, G=500) — this signal was a screening-resolution artifact
  for at least that world. The remaining stable-moderation worlds
  need the same re-screening before any of them is believed.*
- **"Equal-player" specialization is mostly size-asymmetry in
  disguise**: the big premiums among equal-weight worlds sit at
  G = 2000/8000 (up to 29%); with fully identical players at T <= 2
  the premium stays at fractions of a percent.
  *Follow-up: separate the two heterogeneity axes (size vs weight) in
  a small factorial before quoting either as the driver.*
- **The T=2 twins reveal a temptation hump**: at low weights the
  deviation gap *rises* from T=1 to T=2 (e.g. 0.008 -> 0.016 at the
  baseline) — one extra composition step deepens the exposure savings of the
  *hide* deviation (Check 3, Sec. 25.6 — the earlier batch-grab
  reading here was wrong). Meanwhile at w=28 the gap collapses 0.038 ->
  0.000 (a moderation onset already at T=2 for high weight), and the
  quad w=1000 twin flips to fair-wins (-2.75%) with gap 0.0013.
  Composition first *destabilizes* moderation at low weights, then
  stabilizes it — the gap(T) curve should be hump-shaped.
  *Follow-up: directly testable within hours — the running T-grid
  (jobs 180154/155/205) covers T = 1..390; when assembling the onset
  curve, also report argmax_T gap (the hump), not only the zero
  crossing (Sec. 20.1 step 2 extended). Mechanism identified by Check
  3 (Sec. 25.6): the rise is growing exposure savings of the HIDE
  deviation, not a growing grab prize — the earlier wording here was
  corrected accordingly.*

**Ledger corrections (Sec. 22.7 items re-scoped by the matrix).**
Item 2 (specialization always efficient) is now *conditional*: it
requires composition-scale concavity, heterogeneity, or small groups;
at T <= 2 with identical moderate players the premium is ~0 and can
even reverse under convex pricing. Item 1 (corner play) is confirmed
but owned by the *linear pricing* assumption specifically — the one
axis that flips it — not by the mechanism geometry. The trough claim
gains a precise domain (privacy binding inside the box).

## 25. Findings vs. the math: audit, sharpened statements, and gaps

Audit of every empirical signal against the derivations we hold
(`33_u2_analytic_model.md`, Secs. 12/15/16/22). Three categories:
already predicted, sharpened here (with numeric validation in
`69_math_checks.py` -> `69_math_checks.txt`), and genuinely missing.

### 25.1 Lemma (budget factorization) — FIXED: the exact statement
the refuted exponentiality lemma was groping for

With r = (e^eps - 1 + p)/p and x = 1/2 + sigma^2 log r, the exact
T=1 profile factorizes:

    delta_1(eps; sigma, p) = p * h(r; sigma),
    h(r; sigma) = Phi((1-x)/sigma) - r * Phi(-x/sigma).

Proof is two lines of algebra on the closed form (1 - p - e^eps =
-p r, and x depends on (eps, p) only through r). Verified numerically
to 1.4e-11 over 200 random (sigma, p, eps)
(`70_math_fixes.py` -> `70_math_fixes.txt`). Consequences:
- the budget curve delta_1 = dbar at fixed sigma is the exact
  one-parameter family p = dbar/h(r), e^eps - 1 = dbar (r-1)/h(r);
- exact proportionality p ~ (e^eps - 1) holds iff r is constant along
  the curve; the measured drift (r: 959 -> 2501 over eps in [0.5, 6]
  at sigma = 0.9, a factor 2.6) IS the correction the refuted lemma
  missed, now exactly characterized rather than approximated;
- the deep-rate-phase scaling d log p/d eps ~ e^eps/(e^eps - 1) used
  by Sec. 15's two-phase story survives as the slowly-drifting-r
  limit.

### 25.2 Lemma (Jensen split) — WAS IMPLICIT, NOW EXPLICIT

At fixed sigma and fixed total rate p1 + p2: if per-player exposure
disutility is *convex* in the own rate, the equal split minimizes
total disutility (Jensen); if *concave*, the extreme split does; if
linear, indifferent. This single lemma predicts, with no computation:
premium ~ 0 at T=1 with linear cost (advantage linear in p),
fair-shares-wins under quadratic cost (battery block C: all 12 such
worlds are quad — as predicted), and specialization under composition
(A_T concave in p, Sec. 12). Cross-sigma effects (the split moves the
solved noise) are NOT covered and can override — this is the lemma's
scope boundary, and the reason welfare claims still need computation.

### 25.3 Lemma (carrier rule) — WAS HEURISTIC, NOW DERIVED AT FIXED
NOISE

At fixed sigma, linear cost: welfare cost of one batch slot supplied
by group i is lam_i c(sigma)/G_i (a slot costs dp_i = 1/G_i of rate,
each record priced lam_i c). Minimizing total cost over the batch
allocation is a linear program whose solution loads the batch onto
argmin lam_i/G_i — the assignment rule of Sec. 22.3, now exact under
its stated conditions (fixed sigma, linear disutility, T=1
product-form advantage).

### 25.4 Proposition (interior tangency, general V) — GENERALIZED

The noise-phase tangency |V'(sigma)| = lam * p2 * |c'(sigma)| (derived
for the baseline V in Sec. 15's post-cap analysis) holds for ANY
differentiable V — the interior window exists wherever the ratio
|V'|/|c'| crosses lam * p2 inside the box's sigma range. This is the
math behind signal 24.7-1 (interior windows across V shapes tracking
effective weight). For quadratic cost the same identity holds with
the local weight lam_eff = 2 lam_q A (chain rule), which also gives
the threshold prediction for signal 24.7-2 (quad cost inert at
G=50000 until lam_q ~ 1/(2A) x window-weight, i.e. order 1e4-1e5).
Validation: **Check 2 PASSED** — at the interior peaks of all three
signal-A worlds the identity holds to 1.1%, 0.3%, 6.9% (the last at
the coarsest peak localization): |V'| = 0.647 vs 0.640 (inv_sq,
w=28), 0.623 vs 0.621 (exp, w=38.1, unequal groups), 0.260 vs 0.243
(inv, w=2.54, G=500, narrow box). The interior-window mathematics is
correct and V-shape-general.

### 25.5 GDP-limit T-invariance — ALREADY DERIVED, NOW LOAD-BEARING

Sec. 22.5's derivation (eps -> mu map T-free; rates split PSUM
proportionally to mu; T absorbed by sigma) is the piece of math that
*predicted* value starvation and the failure of naive lambda
rescaling; the battery and the GDP scans matched it. No sharpening
needed; flagged here as the limit-model anchor the exact ladder must
validate at the overlap (register P7).

### 25.6 The temptation hump — MATH WAS MISSING; decomposition
supplied, theorem still open

Nothing in our derivations predicted the T=1 -> T=2 *rise* of the
deviation gap (signal 24.7-6). Supplied here: the exact decomposition
gap(T) = DV(T) - DC(T) — the deviation's value change minus its
exposure-cost change. **Check 3 computed it and corrected the
hypothesized mechanism**: at low weight (lam = 0.7, baseline), the
binding deviation from the symmetric candidate (the box ceiling) is
not the batch-grab but the **hide** — tightening to the frontier
preimage of the ceiling (e2 = 47.14 at T=1, 29.84 at T=2; both are
the geometry-pinned f^{-1}(64) values known from Secs. 9/14). The
decomposition reads gap = (small value sacrifice) + (exposure
savings): T=1: 0.0083 = -0.0006 + 0.0089; T=2: 0.0156 = -0.0020 +
0.0176. **The rising edge of the hump is the exposure-savings term
growing with composition** — hiding pays more as exposure compounds —
not a growing grab prize. The eventual decline at large T (GDP tail:
gap -> 0 from above) must then come from the symmetric candidate
itself moving to budgets where there is little exposure left to save
(the moderation onset). A proper theorem (existence/location of the
hump in T at fixed lambda) is OPEN; the T-grid data can locate it
empirically first.

### 25.7 Proposition (one-shot symmetric stationarity) — FIXED:
criterion derived, implemented, and validated as a rejector

At a symmetric point (e, e), rates are p = PSUM/2 and the implicit
chain collapses by symmetry (dsigma/deps2 = 1/(2 Fs), dp2/deps2 =
1/(2 Fp), with (Fs, Fp) the closed-form budget partials). The
own-budget slice derivative is

    g(e) = V'(sigma)/(2 Fs)
           - lam [ c'(sigma) p/(2 Fs) + c(sigma)/(2 Fp) ].

Symmetric interior NE requires: g(e*) = 0 with a falling crossing,
AND u(e*, e*) >= every corner/tangency candidate of the column (the
global check). Both validations behaved correctly and informatively
(`70_math_fixes.py`):
- G=500, V=1/(1+s), lam=8.28 (the battery's "stable moderation"
  world): **no g-root exists at all** — the screening signal was a
  resolution artifact (Sec. 24.7-4 updated). The criterion caught a
  false positive on first use.
- Baseline G=5000, lam=0.7: a falling root EXISTS at e* = 20.45 — a
  previously unnoticed symmetric stationary point at T=1 — but it is
  globally dominated (u_sym = 0.9018 vs column max 0.9336, the hide
  deviation): no NE, consistent with the campaign, now with the
  sharper reason ("stationary but dominated", not "no stationarity").

### 25.8 Definition (n-player feasible region and its faces) — the
missing geometry, written down

n players, groups G_i, batch B, floor pbar. For a budget profile
(eps_1..eps_n), the mechanism solves sigma from
sum_i G_i phi(sigma, eps_i) = B (phi = the rate consuming eps_i at
noise sigma, increasing in both arguments); the profile is feasible
iff a solution exists with all rates >= pbar. For each nonempty
subset S of players, the face F_S = {profiles with p_i = pbar exactly
for i in S}: on F_S all pinned players share the single budget value
E(sigma, pbar), the free players' budgets satisfy one batch equation
given sigma, so **F_S has codimension |S|** (2-player: F_{1} is the
1-dimensional frontier curve, recovering the involution f/f^{-1}).
Corner equilibria with pinned set S require |S| one-sided conditions
plus stationarity/corner conditions for the free players — the
2-player "two one-sided inequalities" structure (Sec. 5) generalizes
face by face. Equilibrium analysis on these faces (which S can be
mutual) remains open, but the geometry is no longer missing.

### 25.9 Math still missing (honest list, updated)

- ~~Rate-phase formula~~ FIXED: the factorization lemma (25.1) is the
  exact statement; no approximation needed.
- ~~Symmetric stability criterion~~ FIXED: derived and implemented
  (25.7); and in the limit model the characterization is now a
  THEOREM (Sec. 26): symmetric interior NE can exist but are never
  potential maximizers (saddle theorem) and never BR-stable (no-go:
  |BR'| = A/(A-c) > 1). What remains is the finite-T/exact-game
  version of these statements.
- ~~n-player frontier geometry~~ WRITTEN (25.8, definition + face
  dimensions); the equilibrium analysis on the faces is the remaining
  open part.
- **Hump theorem** (25.6): monotonicity structure of gap(T) — still
  open; mechanism identified (hide-deviation savings growth), T-grid
  will locate the peak empirically.
- **Uniform finite-T error bound eta(T)** (from the Sec. 26 referee):
  a Berry-Esseen-type f-DP bound with explicit constants, uniform
  over the box, to make the limit-to-exact transfer unconditional.
- **Deployed-game (sawtooth) dynamics**: no formal model of
  tooth-hopping orbits (Sec. 20.2); possibly out of scope — currently
  evidence-only.

## 26. The stability formulation: the limit game is an exact potential
game

**The theorem-shaped answer to "is there a stable equilibrium at high
T".** Change variables from budgets to privacy-loss scales: mu_i =
mu(eps_i), the T-free contract map of Sec. 22.5. In these coordinates
the GDP-limit game is

    u_i(mu_1, mu_2) = F_T(mu_1 + mu_2) - lam * g(mu_i),
    F_T(s) = L/(1+L), L = log(1 + (s/PSUM)^2 / T),   g(m) = 2 Phi(m/2) - 1

— shared value through the *sum* only, private cost through the *own*
term only. Therefore Phi(mu_1, mu_2) = F_T(mu_1+mu_2) - lam g(mu_1) -
lam g(mu_2) is an **exact potential**: dPhi/dmu_i = du_i/dmu_i. (This
is a direct corollary of the T-invariance result — the advantage
depends only on the own contract — and is destroyed at finite T
exactly to the extent that the advantage regains sigma-dependence.)
Consequences, all classical for potential games on compact strategy
sets:
1. **Existence**: a pure NE exists for every T and lambda (any
   maximizer of Phi on the box).
2. **No cycling**: best-response paths strictly increase Phi, so the
   limit game cannot cycle; BR dynamics converge to the NE set.
3. **Stability semantics — sharpened by the second-order theory**:
   "stable equilibrium" = local maximizer of Phi. THEOREM (saddle):
   a symmetric interior stationary point is NEVER a local maximizer —
   along the anti-diagonal fiber (mu1+mu2 fixed), F is constant and
   strict concavity of g makes the symmetric point a strict MINIMUM
   of Phi along its whole fiber; whenever the own-coordinate SOC
   holds it is exactly a saddle. THEOREM (no-go for symmetric BR
   stability): at any interior symmetric BR fixed point with strict
   SOC, writing F'' = -A, lam g'' = -c (A > c > 0), the BR slope is
   BR' = -A/(A-c), so |BR'| > 1 ALWAYS — symmetric interior equilibria
   are linearly unstable under exact best-response dynamics, with
   escape rate (A/(A-c))^2 per round. Only asymmetric or corner
   profiles can be potential maximizers / BR-stable. (This resolves
   the "characterize where the symmetric criterion passes" item of
   Sec. 25.9 in the limit model: interior symmetric NE can EXIST as
   strict Nash equilibria — Nash needs only own-slice maximality,
   which the anti-diagonal direction cannot break — but they are
   never dynamically stable.)
4. **Dominance threshold (strict, explicit)**: since F_T' <=
   4 mu_max/(PSUM^2 T) and g' >= phi(mu_max/2) on the box, for
   T > T0(lam) = 4 mu_max / (lam PSUM^2 phi(mu_max/2)) the floor is
   strictly dominant for both players: the symmetric floor corner is
   the unique, strict, globally attracting equilibrium. (Paper
   instance numbers, CORRECTED by referee: mu_max = mu(64) = 6.34
   [not ~5.6 — that value corresponds to eps ~ 54, an arithmetic slip]
   -> T0 ~ 1.5e7 at lam = 1. Deliberately conservative; the empirical
   floor regime starts around T ~ 5e3.)
5. **Transfer to the exact game (CORRECTED — conditional, margin-
   normalized)**: near-potential theory (Candogan-Ozdaglar-Parrilo)
   is a finite-game theory in payoff units: an eta-perturbation turns
   exact NE into eta-approximate NE and confines better-reply
   trajectories to eta-scaled approximate-equilibrium sets. It does
   NOT directly bound strategy displacement or orbit geometry. The
   defensible statement: beyond T0 the limit game has a uniform
   own-derivative margin m(T) = lam phi(mu_max/2) - 4 mu_max/(PSUM^2
   T) > 0; IF the exact game is within eta of the limit game in sup
   norm on the box, THEN every exact-game NE satisfies
   mu_i <= mu_min + 2 eta/m(T) and BR orbits are confined to that
   slab — displacement bounded by eta/margin, not by eta, and
   conditional on a *proven uniform* eta(T) (our ~3e-3 is measured at
   sampled points at one T; a uniform Berry-Esseen-type f-DP bound is
   the missing ingredient, added to Sec. 25.9).

**The derivation, step by step (each assumption flagged).**
1. *Limit model.* Per-step subsampled Gaussian, shared sigma, rates
   p_i, batch p_1 + p_2 = PSUM. By the f-DP CLT, player i's composed
   view is mu_i-GDP with mu_i = p_i sqrt(T c(sigma)),
   c(sigma) = e^{1/sigma^2} - 1; the mu-GDP curve is delta(eps; mu) =
   Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2).
   [A1: limit model; finite-T deviation measured ~3e-3 at T=390.
   Referee note: Bu et al.'s theorem is stated for p -> 0 with
   p sqrt(T) -> const; the T -> infinity sweep at a fixed eps-box has
   p fixed and sigma_T -> infinity, which is covered instead by the
   Dong-Roth-Su triangular-array f-DP CLT (third moments vanish like
   mu^2 p/sigma); same limit parameter to leading order — cite the
   triangular-array version for the sweep.]
2. *Contract pins the loss scale, T-free.* delta(eps_i; mu_i) =
   delta_bar defines the strictly increasing bijection
   mu_i = mu(eps_i), independent of T (the Sec. 22.5 invariance).
3. *Inversion: rates drop out.* p_i sqrt(T c(sigma)) = mu_i for both
   i, plus p_1 + p_2 = PSUM, give p_i = PSUM mu_i/(mu_1 + mu_2) and
   c(sigma) = s^2/(PSUM^2 T) with s = mu_1 + mu_2, i.e.
   sigma(s) = [log(1 + s^2/(PSUM^2 T))]^{-1/2}: **the shared noise
   depends on the contracts only through their sum**.
   [A2: both budgets binding, rates above floor — valid on the cone
   mu_i/s >= pbar/PSUM; floor faces need the corner analysis.]
4. *Exposure is own-contract only.* A_i = TV of the composed pair =
   delta(0; mu_i) = 2 Phi(mu_i/2) - 1 =: g(mu_i).
5. *Separation.* u_i = V(sigma(s)) - lam_i g(mu_i) = F_T(s) -
   lam_i g(mu_i), with F_T(s) = L/(1+L), L = log(1 + s^2/(PSUM^2 T))
   for V = 1/(1+sigma^2).
6. *Exact potential (Monderer-Shapley, finite deviations).* With
   Phi = F_T(mu_1+mu_2) - lam_1 g(mu_1) - lam_2 g(mu_2):
   u_i(mu_i', mu_-i) - u_i(mu_i, mu_-i) = Phi(mu_i', mu_-i) -
   Phi(mu_i, mu_-i) identically. Per-player strictly monotone
   reparametrization mu(eps) transfers the property to eps
   coordinates.
7. *Threshold algebra.* F_T'(s) = L'/(1+L)^2 <= L' <= 2s/(PSUM^2 T)
   <= 4 mu_max/(PSUM^2 T) on s <= 2 mu_max; g'(m) = phi(m/2) >=
   phi(mu_max/2) on the box. T > T0 = 4 mu_max/(lam PSUM^2
   phi(mu_max/2)) makes du_i/dmu_i < 0 everywhere: floor strictly
   dominant.

**Generality falling out for free (with referee qualifiers).** Steps
5-6 use neither the V shape nor the cost shape nor n = 2: any u_i =
W(sigma(s)) - C_i(mu_i) with any n is an exact potential game with
Phi = W(sigma(s)) - sum_i C_i(mu_i). This (i) explains why the
invariance sweep found the V shape nearly irrelevant, (ii) extends
existence/no-cycling to all *continuous* cost shapes (continuity of
W, C_i is what the compactness consequences use), (iii) gives
n-player existence in the limit model ON THE CONE INTERIOR — the cone
mu_i/s >= pbar/PSUM tightens with n (worst case mu_min/((n-1)mu_max +
mu_min)); at the paper's parameters it stays nonbinding up to
n ~ 32. Floor faces still need the corner analysis. Also: the
stochastic-stability remark (consequence 3) is a finite-game theorem
(log-linear dynamics) and needs a discretization caveat in the
continuous game.

**Where exactness ends.** Everything above is the limit model. At
finite T the composed profile is not exactly GDP and the advantage
regains sigma-dependence; the finite-deviation identity in step 6
then carries an error of exactly that size — which is why the
deployed game can cycle and the limit game provably cannot. The
measured deviation (~3e-3 at T=390, at sampled points) enters the
transfer statement of consequence 5 only as the conditional eta; a
uniform bound eta(T) with explicit constants is open (Sec. 25.9).

**Refereed (2026-07-16).** An adversarial referee pass verified the
mu(eps) bijection (with the clean identity ddelta/dmu =
phi(mu/2 - eps/mu) > 0), the inversion, the advantage formula, the
exact-potential identity incl. reparametrization invariance, and the
threshold algebra — and found two errors, fixed above: the plugged
threshold numbers (mu(64) = 6.34, T0 ~ 1.5e7, not 5.6/4e6) and the
near-potential transfer overclaim (now stated conditionally and
margin-normalized). Full report in the session log.

**Numeric confirmation + second-referee corrections (2026-07-16).**
At T = 390, lam = 1 (constructive referee, scripts `stab*.py`):
- The true symmetric stationary point is **eps = 2.5476** (the earlier
  2.43 was grid-conditional; at 2.43 the symmetric gradient is
  +0.029, not 0). It IS a strict Nash equilibrium of the limit game
  (own-slice global max, margins 0.049 over the floor and 0.549 over
  the ceiling) — i.e. **in the limit model the moderation NE already
  exists at T = 390** — but it is a SADDLE of Phi (eigenvalues
  e+ = -1.685, e- = +0.037) and BR-unstable with |BR'| = 1.045
  (9.1%/round escape; a perturbation of 1e-4 needs ~80 exact rounds
  to surface, which is why coarse/diagonal-preserving dynamics appear
  to converge to it).
- The potential's global maximizer is asymmetric: (0.5, 4.864),
  Phi = 0.2437 vs 0.2403 at the symmetric point — a corner-interior
  KKT point that is BR-STABLE because the strict corner absorbs
  (composite BR derivative 0 despite |dBR2/dmu1| = 1.087).
- "No cycling" is confirmed (every trajectory converges to some NE);
  the earlier claim that all starts converge to the SYMMETRIC point
  was a start-set artifact — generic starts converge to the
  asymmetric pair. Corrected here.
- Tight dominance threshold: T0_tight = 1.396e7 at lam = 1 (root of
  F'(2 mu_max) = lam phi(mu_max/2)); the conservative closed form
  1.47e7 is within 5% — keep the closed form.
- Hessian formulas for the record: F'' = [L''(1+L) - 2L'^2]/(1+L)^3
  with L' = 2s/(aT+s^2), L'' = 2(aT-s^2)/(aT+s^2)^2, a = PSUM^2;
  g''(m) = -(m/4) phi(m/2).

**Addendum: the balanced path and the lambda-free expiration date
(2026-07-16, `74_balanced_path.py` -> `balanced_path.png`, artifact
Figure 17).** The potential already contains the full utility (F_T IS
V(sigma(s))). Rescaling lambda with T to pin the symmetric stationary
point at a fixed eps* — the exact equivalent-weight path lam*(T) =
F_T'(2 mu*)/phi(mu*/2) — yields:
- Self-consistency: lam*(390) = 1.000 at eps* = 2.548 — the paper's
  weight is exactly the balanced weight at the paper's T.
- **Lambda-free existence criterion**: substituting lam = F'/phi into
  the own-coordinate curvature gives H11 = F''(2m) + (m/4) F'(2m) =:
  psi_T(m) — the weight drops out. An interior symmetric NE can exist
  at scale m only while psi_T(m) < 0. The window closes for every m
  in the box at **T_max ~ 1.57e4** (paper box/PSUM; grows with the
  box top and B/|G|): beyond it, NO weight scaling supports interior
  symmetric moderation. At the pinned point mu(2.548) the criterion
  already fails between T = 1500 and 6000.
- Knife-edge sharpening: along the path |BR'| = A/(A - (m/4)F') rises
  1.045 (T=390) -> 1.11 (1500) -> ~3.5 near extinction.
- lam*(T) ~ 1/T at large T; the empirical regime-mix equivalence
  (lam_equiv(25000) ~ 0.35, Sec. 22.5) is a different, flatter notion
  (point-pinning gives 0.21).
Scope: limit model; same referee-grade scrutiny as the rest of Sec.
26 applies (the lambda-elimination is two lines from the verified
stationarity and Hessian formulas, but T_max inherits A1).

**Addendum 2: ridge engineering (2026-07-16,
`75_ridge_engineering.py` -> `ridge_engineering.png`, artifact Figure
18).** The anti-diagonal fiber through the saddle has closed-form
profile Phi(t) - Phi(0) = -lam[C(g(m+t)) + C(g(m-t)) - 2C(g(m))]
(shared value constant on the fiber). Two interventions, exactly:
- **Floor truncation**: the fiber ends at t_max = m* - mu(eps_min);
  raising the action floor 0.5 -> 2.4 shrinks the slide prize from
  3.2e-3 to ~0. Damage control only — the no-go theorem is
  box-independent, so interior BR-stability is not restored.
- **Progressive pricing flips the pass into a peak** at the
  closed-form threshold gamma*(m) = 1 + g(m) m/(4 phi(m/2)) (which is
  simultaneously the |BR'| < 1 condition). **The threshold is
  T-FREE**: it depends only on the moderation level m in loss-scale
  units — T and lambda jointly select WHICH level is stationary, the
  required dose is then read off gamma*(m) regardless of how (T,
  lambda) produced it. General price list: gamma* = 1.006 at eps* =
  1, 1.036 at 2.55, 1.13 at 5, 1.53 at 10 — modest moderation is
  cheap to stabilize at ANY composition depth, generous moderation is
  expensive at any depth. (The (T = 390, lambda = 1) instance selects
  m = 0.375 i.e. eps* = 2.548 -> 3.6%; that pairing is one sample of
  the general curve, per Rule -1 not an anchor.) Verified:
  anti-diagonal curvature +0.014 at gamma = 1.05. Sec. 16's
  convex-pricing mechanism now has its minimal dose as a function of
  the moderation level.
  Scope: limit game; local flip condition (own-slice SOC unaffected
  at these gamma, margin -1.68). Landscape confirmation
  (`76_fixed_landscapes.py` -> `fixed_landscapes.png`, Figure 19): at
  gamma = 1.05 the GLOBAL potential maximizer coincides with the
  moderation point (Phi-gap 0.00000) — moderation becomes the game's
  global attractor; floor-raising alone (eps_min = 2, linear price)
  squeezes the asymmetric peak's advantage to 2.4e-4 without flipping
  stability.

**Consequential caveat discovered by (a): the T = 390 pre-onset
verdict (Sec. 20.1) is candidate-grid-conditional.** The limit game's
BR fixed point sits at eps ~ 2.43, BETWEEN the diagonal candidates of
both the GDP scan and the exact-ladder run (which used {1.5, 2.2,
2.92, 3.8, ...}). The certified positive gap at 2.92 stands, but the
minimum over symmetric candidates may vanish near 2.43. Targeted
exact-ladder columns at {2.30, 2.43, 2.60} submitted as job 180596 ->
`52b_t390_fixedpoint.pkl`; the second referee then located the TRUE
symmetric stationary point at eps = 2.5476 (2.43 was
grid-conditional), so an additional column at 2.548 runs as job
180613 -> `52c_t390_saddle.pkl`. **RESOLVED (2026-07-17, jobs
180626/180627): the exact game agrees with the limit game on
existence — the diagonal is the deviation argmax at all four
candidates, with margin 1.2e-3 (>= 3x the certified bracket width) at
2.548. The moderation NE of the exact game at T = 390 sits at the
limit-predicted eps* = 2.548; it is BR-unstable per the saddle/no-go
analysis. Full verdict at Sec. 20.1.** This is the potential theory's
strongest exact-game confirmation so far: an interior prediction
made in closed form (stationarity of Phi), refereed, then verified by
the exact ladder at the paper's step count — on the diagonal, where
sigma = 1.17 keeps the CLT honest (the seam only bites at the
walls).

## 27. The arc from the analytic solution to the numeric Renyi-DP
game — two theories glued at a measurable seam

**Goal.** Present the potential/ridge theory (Sec. 26) as the analytic
solution and connect it to the numeric RDP-accounted instance, where
the ridge should appear as a band of Nash / near-Nash equilibria.

**Model correspondence (project terms, Rule -1 checked).** Both sides
use the project's mechanism exactly: groups |G_i|, Poisson-subsampled
Gaussian with shared sigma, expected batch sum G_i p_i = B, contracts
(eps_i, delta = 1e-12), u_i = V(sigma) - lam_i A_i with A_i the
record-level MIA advantage of the T-fold composition. The two sides
differ ONLY in accounting (exact/limit profile vs 89-order RDP) and
resolution.

**The naive quantitative arc test FAILED — instructively**
(`77_ridge_arc.py` -> `ridge_arc.png`, artifact Figure 20). Overlaying
the limit game's near-NE band (regret <= 2e-3) on the deployed K=200
regret map: the band covers 4.3% of feasible cells and predicts
better-than-typical deployed regret (median 0.024 vs 0.113 off-band)
— but **0% of the deployed game's 43 exact NE fall inside it**. The
deployed equilibria (the frontier string) live at sigma ~ 0.25 with
one player at the participation floor, and there the limit model
breaks down by orders of magnitude, verified with the exact ladder:
floor player (Tp = 0.004 expected participations): exact budget 21.7
vs GDP-limit 4.1 (5x off); carrier at the same sigma: exact 156 vs
GDP 289,200 (1800x off — mu-GDP explodes when the per-step loss
e^{1/sigma^2}-1 is huge). The CLT limit is valid only where per-step
losses are small (sigma >~ 1; measured 3e-3 payoff error at the
moderation point) — NOT on the frontier.

**The correct arc (both halves already built).**
- **Interior / moderation region (sigma >~ 1, both players
  meaningfully participating):** the potential theory governs — the
  near-flat ridge, the saddle/no-go theorems, the near-NE plateau.
  The deployed game shows it as the central low-regret plateau and
  the orbits (Secs. 3, 11, 20.2/20.5).
- **Frontier region (extreme rate splits, floor-pinned players,
  small sigma):** the EXACT corner theory governs — the one-sided
  conditions and value-tie endpoints of Secs. 5/10/14, which produce
  the equilibrium *segments* (continua) the deployed game samples as
  its 42-NE string. No Gaussian limit applies there, and none is
  needed: the exact T=1/T=2 closed forms and the ladder already cover
  it.
- The deployed instance is the union of both, textured by the
  accountant sawtooth (Sec. 20.2): its near-NE band = analytic ridge
  plateau (interior) + constraint channels (frontier), which is
  precisely the geometry the Sec. 3 regret map found empirically.

**Status of the user-proposed story.** "A ridge of near-optimal
arrangements where free play under linear pricing always slides to
one player carrying" holds — with the ridge's interior described by
the potential theory and its frontier end by the corner theory, and
the slide/instability theorem (no-go) proven in the former, observed
in both. A fully quantitative single-map arc would need an
exact-ladder regret map over the box (expensive: ~10^4 ladder solves)
or domain-restricted comparisons; open item.

**Addendum: the hybrid band game (`78_hybrid_band_game.py` ->
`hybrid_band_game.png`, artifact Figure 21).** Transferring the exact
feasibility band into the analytic setting and playing constrained BR
(analytic payoffs, exact walls): (i) each player's analytic BR branch
rides the wall exactly through the deployed NE strings — the
wall-riding geometry transfers; (ii) MUTUAL wall equilibria do not
appear in the hybrid (4/15 NE on walls, only at branch endpoints):
the tight player's payoff at floor rates is where the limit pricing
is 5x off, so their corner branch fails to coincide — mutual wall
segments require exact payoffs (corner theory), confirming the seam;
(iii) the hybrid produces an 11-NE string along the interior ridge
fiber (grid ties on the near-flat ridge through the saddle) — the
analytic twin of the deployed diagonal plateau strings. "NE where BR
bands overlap" is thus realized in two forms, one per regime:
near-tie strings on the interior ridge (potential theory) and
coincident wall branches on the frontier (corner theory).

**Addendum 2: the two arcs run in opposite directions, and the
analytic model has no walls inside the box (`79_analytic_walls.py`).**
Two clarifications prompted by reading Figure 20 side by side.
(1) *Orientation.* The analytic ridge is a level set of the total
per-step signal, mu1 + mu2 ~ const — a falling trade-off arc in the
(eps1, eps2) plane. The deployed NE strings ride the participation
walls, which are rising curves (a bigger opponent budget relaxes the
wall on the tight player). The two arcs cross; they never coincide.
The earlier suptitle overclaimed the correspondence and is replaced
("Two different arcs: ..."); the quantitative version was already on
record (0% of deployed NEs inside the analytic regret<=2e-3 band).
(2) *The walls themselves do not exist in the analytic solution.* In
the limit model the batch constraint gives p_i = PSUM * mu_i / s, so
the participation frontier p_i = pbar is the ray mu_i / s =
pbar/PSUM = 3.9e-4. Inside the box eps in [0.5, 64]^2 the smallest
value of mu_i/s is mu(0.5)/(mu(0.5)+mu(64)) = 0.078/6.42 = 0.0121 —
a factor 31 above the wall; pinning an eps=0.5 player to the floor
would need an opponent at mu ~ 199, vs. mu(64) = 6.34 at the box
edge. So the analytic game is feasible on 100% of the box, while the
exact game loses 35.5% of it to the walls. The wall NE are therefore
creatures of the exact accounting alone: the limit model prices the
floor player's spend ~5x too cheap (the Sec. 27 seam), which pushes
its frontier ~30x outside the box. The analytic game's own analogue
of "one player pinned, one loose" is the corner-dominance pair (box
floor eps=0.5 + interior opponent) — the *box* edge substitutes for
the missing participation wall. Scope: this instance's PSUM, pbar,
box, and delta; the 31x margin is instance-specific, but the
mechanism (CLT underpricing at floor rates => frontier displaced
outward) is the seam and travels with it.

**Addendum 3: the constrained-BR construction, refereed with
tolerance (`80_hybrid_near_ne.py`).** Objection (user): analytic BRs
restricted to the exact feasible band, NE = overlap — wall NE should
appear. Verdict: the construction is valid and its failure is exactly
one-sided. At every one of the 43 deployed wall NEs, the TIGHT
(pinned) player's hybrid regret is identically 0: the analytic BR
rides the wall to precisely the deployed point, on both wings. The
LOOSE player's hybrid regret is what breaks the overlap: 3e-3 at the
string's inner end, growing monotonically to 1.5e-1 at the outer end
(payoff span 0.79) — analytically the loose player wants to retreat
inward, because the limit model overprices the carrier's exposure at
small sigma (the other face of the Sec. 27 seam; carrier exact-vs-GDP
budget 156 vs 2.9e5). Tolerance sweep: tau = 5e-3 captures 2/43
deployed NEs, tau = 2e-2 captures 11/43, always from the
branch-crossing endpoints inward; recovering the full string needs
payoff-scale tolerance, i.e. the gap is mispricing, not grid noise.
Refined statement replacing Addendum 2's blanket negative: HALF of
each wall equilibrium is analytic — the pinned player's wall-riding
is reproduced exactly by the limit payoffs — and the strict-overlap
NE set (the 4 endpoint stars of Figure 21) marks where the loose
player's analytic interior BR crosses the wall. What requires exact
accounting is only the loose player's incentive to HOLD a large
budget against a pinned opponent. Scope: T = 390, lam = 1, K = 200
band; one-sidedness should persist wherever the seam does (the tight
side involves no small-sigma pricing), but the regret magnitudes are
instance-specific.

**Decision (2026-07-16): the limit model is demoted to the high-T
convergence regime and stays out of the paper's main line.** The
disqualifying assumption is the loss of sigma-dependence in the
advantage — the very thing that manufactures the exact-potential
structure. At finite T that dependence is real: the exact game is
provably not a potential game (Prop. in 04_analysis) and the deployed
game cycles, so "no cycling", the saddle/no-go theorem, T_max, and
gamma*(m) are statements about the T -> infinity convergence regime,
not about the deployed instance at T = 390 (where the interior
agreement is only measured, ~3e-3 at one point, with no uniform
eta(T) bound). Consequences: none of Sec. 26 / its addenda enters the
paper body; at most a back-matter/appendix remark scoped explicitly
as the high-T limit. The paper's interior-moderation and mitigation
claims are carried by exact-game evidence alone (T = 1/T = 2 closed
forms, the exact ladder, deployed measurements; convex pricing is
already confirmed in the exact T = 1 game, Sec. 16). Secs. 26-27
remain in this log as analysis record.

---

## 28. Dense exact slice at the diagonal (T = 390, lambda = 1) — 2026-10-12

**Why.** The paper (one symmetric NE at 2.96) and its Fig. 3/14 captions
(no diagonal NE) contradict each other; the 21-point ladder columns of
Sec. 20.1 sample no deviation within +-40% of the diagonal, and a spline
through them suggested a trough.

**Setup.** `game_experiments/t390_local_slice.py` (Slurm 292170,
`slurm_scripts/run_t390_local_slice.sh`, 28 workers, 1h37): exact ladder
(PEG_NE = 6000), e1 in {2.548, 2.92}, 13 log-spaced p2 in [0.0112,
0.0169] plus the diagonal (e2 2.09-3.98 resp. 2.43-4.47). Output
`tasks_findings/_scripts/83_t390_local_slice.pkl`.

**Results.**
- Both slices are concave peaks (quadratic fit curvature -0.024 per
  (log e2)^2); the spline trough was an artefact of interpolating across
  the frontier kink.
- BR(2.548) = 2.62 (gain over the diagonal 7e-6), BR(2.92) = 2.38 (gain
  1.0e-3). Linear interpolation of BR(e) - e gives the symmetric fixed
  point **eps* ~ 2.59** (bracket (2.55, 2.92)), sigma ~ 1.16 < sqrt(3).
- BR slope ~ -0.65 (strategic substitutes, |BR'| < 1): locally stable in
  the exact game — the limit model's |BR'| = 1.045 does not transfer.

**Interpretation.** The exact game has exactly one symmetric equilibrium
at T = 390, lambda = 1, created by V saturation (consistent with the
trough proposition, which requires sigma >= sqrt(3)). The deployed
89-order grid shows it at 2.965 with a 1e-4 margin and dissolves it into
a 1.7e-5 near-tie plateau at 16x — a resolution property, not
non-existence. The V-shape ablation's diagonal grid points (one per
nonlinear V, none for V = -sigma) are the same object. Onset numbers of
Sec. 20.1 rest on 31-point columns and need the same dense treatment.

## Synthesis: what main.pdf should say (and mostly now says)

1. **Symmetry breaking is constraint-vs-saturation, exactly.** The T=1
   game proves the pattern in closed form: symmetric equilibria exist
   only where a constraint pins both players; asymmetric equilibria live
   on the shared participation constraint; the diagonal is otherwise a
   trough crossed by a BR jump. The T=390 saturation-window symmetric
   equilibrium (eps*=2.96) is the one addition composition makes
   (V-saturation flipping the diagonal curvature, Prop. 9's complement)
   — but the 16x zoom (Sec. 20.5) shows the grid NE dissolves at finer
   resolution: present it as the *location of the moderation plateau*
   (with the continuous FOC crossing 2.9635 and the exact-ladder gap
   analysis of Secs. 17-18 as the substantive claim), not as an
   isolated equilibrium point of the deployed game.
2. **The asymmetric equilibria are a continuum on the analytic
   frontier**, corner-pinned on both sides; grid counts are sampling
   density, and the ratio band is the segment seen through the frontier
   geometry. Figure 4 (rebuilt) shows exactly this.
3. **The lambda sweep is a bifurcation without a pure-NE gap** at T=1:
   ceiling -> isolated box-edge pair (f^{-1}(64), 64) -> frontier
   segments -> floor pair -> symmetric floor. Pure-NE nonexistence
   (mixed play as an undercutting cycle) is a genuinely
   composition-scale phenomenon: it occurs at T=390, lambda=2, where
   the tight best responses are interior, but not in the T=1 or T=2
   corner games. The paper's wsweep observations (no pure NE at
   lambda=2, corner NE at lambda=4) sample the T=390 curve.
4. **Accountant discreteness is part of the game definition** at T=390:
   payoff kinks of ~1e-4 at Renyi order switches produce the observed
   near-indifference strings; the T=1 companion (or a PLD-budget
   variant) is the clean-room version.
5. **Composition is the source of interior structure** — worth one
   discussion paragraph; it explains *why* the analysis must be about
   iterative training and not one-shot release.
6. **The theoretical (smooth) and the accountant-driven (sawtooth)
   best responses are different objects with different dynamics** and
   the paper must treat them as such: exact accounting gives the
   economics (segments, corners, moderation onset); deployed
   Renyi-order accounting adds a staircase texture that creates its
   own fine-scale dynamics (tooth-hopping orbits, near-tie phantom
   equilibria with parts-per-million margins). State every equilibrium
   claim with its object named and, where it is the deployed game,
   with margins against the sawtooth amplitude (Secs. 20.2, 19 row
   4b).
7. **The limit/potential model does not speak in the paper.** Its
   structural results hold only in the high-T convergence regime
   (sigma-independent advantage); per the 2026-07-16 decision at the
   end of Sec. 27 they are excluded from the main line — at most an
   appendix remark scoped as the T -> infinity limit. Everything the
   paper claims rests on exact accounting and deployed measurement.

## Data / artifact index

| item | where |
|---|---|
| clean K=200 cache | `game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` |
| continuous eq. search | `tasks_findings/_scripts/29_exact_equilibria.pkl` (jobs 179667/179691) |
| FOC landscape + BR data | `tasks_findings/_scripts/31_foc_landscape.py`, `paper/figures/data/foc_landscape/`, `32_fig_C6_landscape_data.py`, `paper/figures/data/c6l_*.dat` |
| regret/distance heatmaps | `paper/figures/data/foc_landscape/{regret,br_distance}_heatmap.png`; artifact b4ee7b8b |
| exact model note | `tasks_findings/33_u2_analytic_model.md` |
| exact FOC pointwise | `game_experiments/exact_foc.py` |
| exact FOC field T=390 | `tasks_findings/_scripts/34_exact_foc_grid.pkl` (job 179779), `t390_exact_foc.png` |
| T=1 model + grid | `game_experiments/single_step_game.py`, `single_step_grid.py`, `35_single_step_grid.pkl` (job 179788) |
| T=1 exact BR (lam=28) | `single_step_br.py`, `36_single_step_br.pkl` (job 179809), `37_t1_br_clipped.pkl` |
| lambda sweep | `single_step_lam_sweep.py`, `38_t1_lam_sweep.pkl` (179967), `39_t1_lam_low.pkl` (179974, with component cache) |
| T=1 figures | `paper/figures/data/foc_landscape/t1_*.png`; artifact 5b4e015c (BR/NE, bifurcation, flipbook) |
| group-size sweep (T=1, lam=0.7) | `40_t1_br_G500.pkl`, `41_t1_br_G50000.pkl` (job 180103), `43_gsweep_endpoints.py`, `43_gsweep_figure.py`, `t1_gsweep_lam07.png` |
| advantage curvature vs T | `44_adv_curvature.py`, `44_adv_curvature_summary.txt` |
| T=2 exact game + BR | `game_experiments/two_step_game.py`, `two_step_br.py`, `42_two_step_br.pkl` (job 180116) |
| slice decompositions | `46_slice_decompositions.py`, `46_slice_data/`, `t1_slice_decomposition.png`, `interior_slice_decompositions.png` |
| interior-BR curves + NE | `47_t1_lam_G500.pkl` (job 180141), `48_t2_br_lam126.pkl`, `49_interior_br_curves.py`, `interior_br_curves.png` |
| exact T-grid ladder | `game_experiments/multi_step_game.py` (validation in `__main__`), `run_t_grid.py`, `50_t_grid.pkl` (job 180154) |
| cycle dynamics figure | `54_cycle_trajectory.py`, `cycle_trajectory.png` (lambda=2 BR trajectories + 3-cycle attractor) |
| basins + outcome data | `57_basins.py` (figure), `58_basin_outcome.py` (1-D outcome data), `basins.png` |
| alternative plays | `59_alt_plays.py`, `59_alt_plays_summary.txt` (Stackelberg, collaborative, heterogeneity, GDP high-T) |
| holistic robustness | `robust_game.py`, `run_robustness.py`, `60_robustness.pkl` (job 180346), `66_invariance.py` + `.txt` (matrix), `62_protocols.py` (protocol census), `63_mpmath_coverage.py` + `.txt` |
| zoom microscope figures | `64_zoom_br_fig.py`, `65_zoom_compare_fig.py`, `zoom_br_lam2*.png` (artifact Figures 11-12) |
| two-theory arc + hybrid | `77_ridge_arc.py` (`ridge_arc.png`), `78_hybrid_band_game.py` (`hybrid_band_game.png`), `79_analytic_walls.py` (analytic wall location), `80_hybrid_near_ne.py` (one-sided seam regrets) |
| 07-17 verdict landings | `52b_t390_fixedpoint.pkl` (job 180626), `52c_t390_saddle.pkl` (180627), `56_cycle_zoom_dense.pkl` (180594); analysis `81_verdicts_0717.py`; dense K=200 cache = job 180899 (`run_mech_dense.py`) |
| paper edits so far | `paper/tex/fig_C6_tikz.tex` (rebuilt), `paper/tex/04_analysis.tex` (caption + BR-loci paragraph) |

## Open items

- **Exact-ladder K=200 grid via (sigma, p) tabulation, twin offset
  meshes (job 180682, `game_experiments/tab_ladder_grid.py` ->
  `_cache_v2/exact_grid_K200_T390_g5000_mesh{A,B}.pkl`,
  `tab_compare_T390_K200.txt`).** Closes the Sec. 27 open item (the
  exact regret map over the box) without ~10^4 per-cell ladder solves:
  E_T and Adv_T tabulated on 44x40 / 45x41 log-spaced (sigma, p)
  meshes (mesh B = geometric midpoints of A, shared endpoints), bicubic
  interpolation of the logs, mechanism inversion on the interpolants;
  output mirrors the mech-grid schema so sweep_from_grid /
  find_pure_NE_2player run unchanged. Error instruments: (i) per-cell
  disagreement of the two offset-mesh grids in game coordinates
  (sigma, A, U, feasibility classification), (ii) ~126 off-node
  anchors from the (sigma, p2, e2, A2) rows of `52_t390_ladder.pkl`.
  Smoke test at T=2 (10x9 mesh) against the certified quadrature:
  eps rel err 2e-4..5e-3 (worst at the small-sigma corner), adv
  ~1e-5..2e-4; production mesh is ~4.4x denser per axis. Decision
  rule (in the compare report): interpolation is trusted only where
  the offset-mesh utility disagreement sits well below the payoff
  scale it must resolve; classification-disagreement cells and
  failures get per-cell ladder solves before any equilibrium claim.
- Propagate the continuum framing through IV.B prose and the appendix
  (symbreak section); decide whether the T=1 companion becomes an
  appendix exhibit.
- Exact segment endpoints bar-eps(lambda) as root-finds (cheap; component
  cache in `39_t1_lam_low.pkl` makes any new lambda ~free).
- T=2 exact BR results (job 180116) into Sec. 14; decide whether T=3-5
  nested quadrature or certified two-sided PLD brackets are needed to
  locate the onset of interior best responses in T.
- The regret heatmap as a possible replacement/companion for Figure 4
  (pgfplots port pending user decision).
