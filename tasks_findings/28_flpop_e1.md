# E1 — realistic FL-consortium equilibrium prediction (findings)

Level discipline: every number here is **level-2** (PLD-exact analytic MIA
advantage of the subsampled-Gaussian mechanism). No trained-model measurement
enters E1's dynamics.

## Frozen parameters (A4/A5)

| parameter | value |
|-----------|-------|
| players n | 20 |
| group sizes \|G_i\| | Uniform{2000,...,12000} (int) |
| privacy weights w_i | LogNormal(mu=0, sigma=0.7) |
| delta | 1e-12 (non-negotiable) |
| expected batch E_b | 256 |
| epochs | 10 |
| steps I | 10 * floor(M/256), M = sum \|G_i\| (~5000-6000 at n=20) |
| V(sigma) | 1/(1+sigma^2) |
| u_i | V(sigma) - w_i * A_i |
| PLD discretisation (dynamics) | 1e-2 (DISC_FAST) |
| PLD discretisation (verification) | 1e-3 (DISC_EXACT) |
| action grid | **{10, 14, 19, 26, 37, 50}** (log-spaced on [10,50]; frozen by A4 scan) |

## Engineering notes

- The advantage PLD dominates oracle cost. At disc=1e-3, one subsampled-Gaussian
  advantage is ~6.5 s at I~5750; a heterogeneous 20-player profile needs up to
  n distinct advantage calls (~130 s). At disc=1e-2 the same call is ~0.03 s
  (~200x) and the advantage shifts by <=~1.2e-2 (consistent with T17/R2). We run
  all dynamics/scans at disc=1e-2 and verify final equilibria at 1e-3.
- The mechanism solve (RDP accountant bisection) is ~4.4 s for a uniform profile
  and ~13 s for a heterogeneous one (per-group sample-rate bisection dominates).
  Solves are cached by the multiset of (round(eps_i,6), |G_i|) pairs, so
  permutations and repeated BR visits share work.
- `flpop_common.py` (population sampler + level-2 `Oracle` + picklable
  `solve_profile_raw`) is the stable API the E2 sibling imports.
- `run_flpop_e1.py` uses a parallel `ParCache` (process pool over the 6
  candidate solves per BR/planner step).

## A4 pre-flight scan (job 174341; 10 populations, 6 uniform + 40 random
profiles each, disc=1e-2)

| grid | actions | feasibility (random) | sigma range | median-A across uniform profiles | med solve |
|------|---------|---------------------|-------------|----------------------------------|-----------|
| **default (FROZEN)** | {10,14,19,26,37,50} | **0.993** | [0.358, 0.601] | [0.225, 0.281, 0.354, 0.437, 0.565, 0.674] | 8.8 s |
| low | {5,7,10,15,21,30} | 0.665 | [0.412, 1.015] | [0.152 ... 0.485] | 8.5 s |
| mid | {6,9,13,19,28,40} | 0.642 | [0.383, 0.713] | [0.166 ... 0.591] | 7.3 s |

Decision: **default grid frozen.** It is the only candidate meeting the >=90%
feasibility bar; the lower grids lose ~1/3 of random profiles because the
participation floor (min sample rate 1e-5) binds at small eps for small groups
— exactly the failure mode the plan warned a low-eps grid would produce.

Honest deviations from the A4 acceptance targets:
- *Spread*: median A spans [0.225, 0.674], not [~0.05, ~0.5]. Pushing the low
  end toward 0.05 requires lower eps, which the feasibility columns above rule
  out. Achieved spread (0.45 absolute, 3x relative) still gives the grid clear
  strategic content. Documented rather than grid-shifted.
- *Sigma*: [0.36, 0.60], below the indicative [0.4, 6] window at large eps;
  V(sigma) still spreads meaningfully (0.735 -> 0.885 across the grid).
- *Solve time*: median 8.8 s/profile vs the <=2 s target. The RDP-bisection
  mechanism solve dominates (the PLD advantage is already at disc=1e-2,
  ~0.03 s). Absorbed by 6-way parallel candidate evaluation per Slurm task.

## D1 oracle self-consistency + D3 feasibility audit (job 174346)

- **D1 PASS** (9/9 probes): for populations 0-2, 3 random (profile, player)
  probes each, the cached/sig6-rounded oracle's best-response argmax matches a
  direct, uncached, unrounded PLD evaluation; max |utility difference| across
  all probed actions 2.3e-6 (grid: default, disc=1e-2).
- **D3**: 1/180 random profiles infeasible (0%, 2%, 0% for pops 0-2). The one
  reject has min sample-rate p = 4.68e-6 < 1e-5 at (eps=10, |G|=7118),
  sigma=0.368 --- i.e. the participation floor does the exclusion, not a
  solver failure. Consistent with the A4 feasibility rate (0.993).

## M2 calibration: BR dynamics on populations 0-2 (job 174343)

**Headline (negative, structural): at the plan's weight scale the game is
degenerate.** In all 3 calibration populations, sequential best response
converged in 2 rounds from BOTH initialisations (grid midpoint 19 and grid top
50) to the *uniform floor profile* (10, 10, ..., 10). Consequently
rho(w, eps*) = rho(eps*, A) = 0 and Gini(equilibrium) = Gini(uniform) = 0
identically (at a uniform profile every group gets the same sampling rate, so
all A_i are equal). 100% of players sit at the lower grid boundary (D2
boundary audit fails by construction).

Diagnosis (quantified on pop 0 from the cached solves): a unilateral move from
the all-10 profile changes the mover's advantage by dA ~ +0.14..+0.71 but
moves the shared value only by dV ~ +0.003..+0.03 (one player is ~1/20 of the
batch). The critical privacy weight below which any upward deviation pays is
w* = dV/dA ~ 0.02-0.04 (0.11 for the largest group's first step). The plan's
w_i ~ LogNormal(0, 0.7) has essentially no mass below 0.05
(P(w<0.05) ~ 1e-5), so eps = 10 is a strictly dominant strategy for every
player ever sampled: free-riding on the accuracy externality dominates at
n = 20. (The plan's weight scale was calibrated on n = 2 games, where one
player's budget moves sigma ~10x more.)

The planner optimum on the same populations is interior and *sorted*: e.g.
pop 0 planner = [50,50,50,19,37,14,...,14] (low-w players assigned high eps),
Gini_planner = 0.61, sum-utility 14.29 vs 10.42 at the equilibrium -- the
externality creates a large price of anarchy, and the *planner*, not the
equilibrium, exhibits budget sorting at this weight scale.

Wall time: 28-37 min per population (1589-1842 profile solves, 6 workers).

## Coordination decision (option c: both stories)

The orchestrator pre-authorised running BOTH: (1) the plan-scale
(w ~ LogNormal(0, 0.7)) 100-population array as the *free-riding headline*
(universal maximal privacy demand at n=20; measured critical weight
w* ~ dV/dA ~ 1/n ~ 0.02-0.04; ~27% welfare loss vs the sorted planner), and
(2) if the critical-scale weight probe restores interior sorted equilibria,
a second 100-population array at w ~ LogNormal(ln 0.03, 0.7) as the
population spec for the Panel A/B/C figure. Justification for (2): weights
expressed in units where the privacy-accuracy trade-off is active at n=20.
Quotable intuition to verify later: the free-riding threshold scales like
1/n (one player is 1/n of the batch, so dV/deps_i ~ (1/n) dV/d(uniform eps)).

E_b stays fixed at 256 (an E_b-rescaling remedy was considered and set aside
in favour of the weight-scale remedy; the mechanism plumbing now supports
per-population E_b via ``flpop_common.expected_batch_for`` if this is ever
revisited -- caches are namespaced by (E_b, epochs) so scales cannot mix).

## E1 results (100 populations, plan parameters) -- job 174385

(pending; the earlier array 174350 was cancelled mid-flight during
coordination and resubmitted unchanged as 174385)

## Weight-scale probe (w_i ~ LogNormal(ln 0.03 = -3.51, 0.7), pops 0-2) --
job 174351

Probe centred at the measured critical threshold w* ~ 0.03; shares the
weight-free solve cache with the main run.

(pending)
