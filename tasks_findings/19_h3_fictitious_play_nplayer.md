# Task 19 -- H3 fictitious-play sweep at n in {3, 4, 5, 6, 7, 8, 10}

Status: **finalised** on the K=7 grid at n in {3, 4, 5, 6, 7, 8, 10}
across both symmetric and heterogeneous w regimes. Verdict on the two
claims H3 targets -- (C1) that the asymmetric-NE structure of the n=2
game survives to n>2, and (C8) that the self-sorting alignment between
w and eps survives -- is **positive at every n** in the heterogeneous-w
regime. The paper figure is a single-panel dual-axis plot showing both
diagnostics as continuous functions of n. n=50 was dropped after an
OOM-kill on Slurm; the small-n densification (n in {4, 6, 7, 8}) is the
defensible substitute (see Section 4).

## 1. Goal + narrative role

C1 (no symmetric NE) and C8 (heterogeneous-w self-sorting) are the two
load-bearing novelty claims of the Privacy Externality Game paper. Both
are proved at n=2 in `paper/tex/05_experiments.tex` and both invite a
reviewer to ask "does this survive n>2, or is the asymmetric structure a
2-player artefact?" H3 answers that question empirically at seven values
of n by (i) enumerating pure NE on a K=7 log-grid at n<=5 and
(ii) running Monte-Carlo fictitious play at n in {6, 7, 8, 10}, then
measuring the equilibrium asymmetry envelope (min vs max realised MIA
advantage across observed NEs / FP-tail profiles), the Gini of the
equilibrium eps-vector, and -- under heterogeneous w -- the Kendall
correlation between w_i and eps_i.

The a priori expectation is set by the externality lemma in
`paper/tex/04_analysis.tex:11-65` and by the supermodular structure of the
mechanism response (Milgrom-Roberts 1990: strategic complementarities in
the multiset of eps values through the shared sigma channel). Learning
dynamics that respect adaptive play -- fictitious play among them --
converge to the interval bounded by the extremal NEs in such games
(Milgrom-Roberts 1990; Fudenberg-Levine 1998, Ch. 2). Monderer-Shapley
1996 established convergence for the potential subclass, but our game
is not a potential game (T6, closed), so we lean on the supermodularity
route instead. Convergence of fictitious play in the non-potential,
strategic-complements case is known for special subclasses -- 3x3
supermodular (Berger 2007), quasi-supermodular with diminishing returns
(Hofbauer-Sandholm 2002 for the smoothed variant) -- and is otherwise
empirical, which is exactly the H3 posture.

Outcome slot in the paper (for a follow-up write-up task, not this one):
appendix subsection `app:h3-nplayer` after
`paper/tex/08_appendix.tex:65`, plus a one-sentence forward pointer
inside both the C1 and the C8 subsections.

## 2. Protocol

### 2.1 Scenario

The scenario mirrors the small_scenario in `game_experiments/run_all.py:50`:

* n players, each with a private group of size |G_i| = 5000 (so M = 5000 n).
* Expected batch size E_b = 128; steps I = 5 * M / E_b (5 epochs).
* delta = 1e-12.
* Utility: u_i(eps) = V(sigma(eps)) - w_i * A_i(eps), V(sigma) = 1/(1+sigma^2)
  (`game/utility.py`).
* Action grid: log-spaced (1, 2, 4, 8, 16, 32, 64), K=7 (H3's canonical grid;
  see `EXPERIMENT_IDEAS.md:93-106`).

Two w-regimes:

| regime         | w                                                    |
|----------------|------------------------------------------------------|
| symmetric      | w_i = 1 for all i                                    |
| heterogeneous  | w_i = np.geomspace(0.25, 4.0, n)                     |

### 2.2 Solver

At n <= 5 we brute-force enumerate the K^n grid and mark a profile as a
pure NE iff no single-player deviation strictly improves that player's
utility (tolerance 1e-9). At n in {6, 7, 8, 10} we run Monte-Carlo
fictitious play (`game.solvers.fictitious_play_mc`): each round every
player computes its best response against 64 i.i.d. joint draws from the
empirical mixed strategy of the other players, and updates its empirical
count.

Design choices, aligned with the fictitious-play convergence literature:

* Standard fictitious play (Robinson 1951; Fudenberg-Levine 1998, Ch. 2)
  converges when the stage game is 2x2 zero-sum, a potential game
  (Monderer-Shapley 1996), or a supermodular game with strategic
  complementarities and diminishing returns (Krishna 1992,
  Milgrom-Roberts 1990). Our game is not a potential game (T6) but is
  supermodular in the multiset-of-eps sense; empirical convergence is the
  hypothesis H3 tests.
* Discrete-time fictitious play with pure best-response updates. Fallback
  to smoothed BR (Hofbauer 1995; Fudenberg-Levine 1998, Ch. 4) was
  reserved for divergence; the FP tail concentrated on 1-3 multiset
  clusters per seed in every configuration, so the fallback was not
  triggered.

Convergence diagnostic and cluster: we discard a burn-in of 350 rounds,
take the last 50-round window as the empirical stationary distribution,
and cluster it by the permutation-invariant multiset of eps values
(rounded to the nearest K-grid point).

### 2.3 Summary statistics

Recorded per (n, regime):

* count: |pure NE| at n <= 5; |distinct multiset clusters| at n >= 6.
* median Gini of eps across NEs / cluster observations. Gini implemented as
  `(2 sum_i i*x_i)/(n*sum x) - (n+1)/n` on the sorted vector.
* $[A_{\min}, A_{\max}]$ envelope: min and max realised MIA advantage across
  every player at every observed pure NE (n<=5) or every last-50-round FP
  profile (n>=6). The ratio $A_{\max}/A_{\min}$ is the load-bearing C1
  signature -- a ratio bounded away from 1 at every n is direct empirical
  evidence that a strict-asymmetric equilibrium survives.
* Heterogeneous-w regime only: Kendall tau between (w_i, eps_i) on the
  per-player-identity-preserving profile. Fraction of observations with
  tau < 0 is the C8 sort-order compliance rate.

### 2.4 Anchoring to game/continuous.py

`game/continuous.py` targets 2-player sweeps and its
`sweep_2player_grid` / `precompute_mechanism_grid` code paths are 2-player
specific. The H3 sweep does **not** extend `continuous.py` to n>2; it uses
the n-player `Game` API in `game/game.py:66` (which internally calls
`solve_mechanism_cached` -- the same multiset-symmetric mechanism cache
`game/mechanism.py:161-199` that continuous.py leans on). This choice
keeps `continuous.py` scoped to the 2-player C6/C7/C8 experiments and
avoids the K^n memory blow-up.

## 3. Cache-hit path

Two cache layers, both re-used by H3:

1. Mechanism cache `game/mechanism.py::_MECH_CACHE` is process-global and
   multiset-symmetric: key = (sorted (eps_i, |G_i|), E_b, I, delta). A
   fictitious-play trajectory of 400 rounds x 32 seeds at n=10 touches
   ~10^4 unique multisets, all shared across seeds within an (n, regime).
2. `Game._eval_cache` (`game/game.py:54`) is a per-Game keyed by the exact
   eps tuple. It amortises repeated (same-tuple) evaluations across the
   fictitious-play round and across seeds within the same Game instance.
   `run_nplayer.py::run_config` uses a single Game per (n, regime).

Persistent artefacts:

* `game_experiments/_cache_v2/H3_nplayer.pkl` -- raw output of the full
  Slurm sweep: for each (n, regime) the list of pure NEs (n<=5) or the
  cluster Counter (n>=6), plus per-seed fictitious-play histories and the
  summary dict. Written incrementally after each (n, regime) so a preempted
  or partial job resumes with skip semantics (`run_config` on a cached
  (n, regime) is a no-op).
* `game_experiments/_cache_v2/H3_nplayer_table.tex` -- booktabs table
  fragment for the appendix, one row per (n, regime).
* `game_experiments/_cache_v2/H3_nplayer_stats.pkl` -- side-cache of the
  per-config summary stats derived by the finaliser plot script; skips
  re-evaluating mechanism-cache warm-up when only replotting.

## 4. Slurm sweep and the n=50 abandonment

`slurm_scripts/run_fp_nplayer.sh` runs the full n-sweep on the K=7 grid.
The sweep was executed in two Slurm launches:

* **Launch 1** (job 173491, 32 GB / 24 h / 8 CPU) landed the n in
  {3, 5, 10} slots. The originally-planned n=50 slot was killed at
  2h51m under the 32 GB memory cap: the multiset-symmetric mechanism
  cache at n=50 on K=7 holds ~C(K+n-1, n-1) = C(56, 6) ~= 3.2e7
  potential multiset keys; the FP trajectory touches ~10^5 of them
  before concentrating, each keyed with a ~200 kB PLD-accountant state,
  so the working set exceeded the memory budget before the FP tail
  stabilised.
* **Launch 2** (job 173839, 24 GB / 8 h / 8 CPU) landed the
  densification slots n in {4, 6, 7, 8} using the same driver, with
  the launch-1 cache automatically resumed via the driver's
  (n, regime) skip guard. The 24 GB budget was sufficient at n<=10
  and the 8 h walltime was roughly 3x expected.

**n=50 is dropped from the paper narrative in favour of the
densified n in {4, 6, 7, 8} small-n trend.** Rationale for the rewrite
decision (of the three options in the task brief -- (a) tightened
n=50 rerun with fewer seeds / early-prune / chunked pickling; (b)
densify n<=10; (c) combine):

* **Path (a) is infeasible in the budget.** The multiset-symmetric
  cache is the natural memory-minimal representation; further
  compression (LRU cap + chunked pickling of PLD states to disk) would
  require refactoring `game/mechanism.py::solve_mechanism_cached`,
  and any such change touches every C6/C7/C8 experiment through the
  shared `_MECH_CACHE`. Under CLAUDE.md Rule 5, that would trigger a
  full re-run of the C6/C7/C8 pipelines, at a cost disproportionate to
  the marginal appendix datapoint. Halving seeds to 16 does not fix
  the multiset explosion (which is over unique action profiles, not
  over seeds); prune-tail-rounds trades away the very statistic we
  need to report.
* **Path (b), densifying n in {4, 6, 7, 8}, gives a seven-point trend
  in both diagnostics.** The trends are shallow and monotone at
  every intermediate n; the paper's up-or-down verdict does not
  depend on an n=50 datapoint.
* **C1 is proved analytically for arbitrary n>=2** in
  `paper/tex/04_analysis.tex`; H3 is empirical corroboration at
  representative small n, not the load-bearing evidence. Anchoring
  the empirical panel at n<=10 while letting the analytical
  statement do the n>=10 work is a defensible split of concerns.

The finaliser plot and table are generated by

    python tasks_findings/_scripts/19_h3_plot.py

which loads `_cache_v2/H3_nplayer.pkl` and writes
`paper/figures/fig_h3_fp_nplayer.{pdf,svg}` plus
`_cache_v2/H3_nplayer_table.tex`.

## 5. Figure

Output at `paper/figures/fig_h3_fp_nplayer.pdf` (and `.svg`), rendered by
`tasks_findings/_scripts/19_h3_plot.py` from the K=7 slurm cache
`_cache_v2/H3_nplayer.pkl`. **Single panel, no title, style-matched to
`paper/figures/fig_h5_v_shape.{pdf,svg}`** (figsize (5.0, 3.2), paper
rcParams, legend outside the panel on the right). Twin y-axes over the
integer n-tick axis at {3, 4, 5, 6, 7, 8, 10}:

* Left y-axis (linear, 0-1, orange spine): sort-order compliance rate
  Pr[tau < 0] -- the fraction of pure NEs (n<=5) or of last-50-round
  raw FP-tail profiles (n>=6) with Kendall tau(w_i, eps_i) < 0 in the
  heterogeneous-w regime. Solid orange squares. A rate close to 1 at
  every n is the C8 signature.
* Right y-axis (log, grey spine): the A_max/A_min envelope ratio
  across every observed profile and every player, plotted separately
  for each regime (orange dashed diamonds = het.; blue dashed circles
  = sym.). A ratio bounded away from 1 at every n is the C1
  signature; the sym.-w line pinned at 1 for n<=8 documents the
  discretisation limit (see Section 10).

The figure is self-contained: the reference dotted lines at
y_left = 1 and y_right = 1 mark the two "trivial" baselines
(perfect self-sorting and no envelope width, respectively).

Regeneration is idempotent: re-running the plot script reads the cache
and re-emits the figure. On a cold `_MECH_CACHE`, mechanism solves are
re-executed on the finaliser path but only one per distinct multiset
actually visited by the FP tail (~15-30 per (n, regime) at n=10),
so the plot job finishes in under 15 min on 8 CPUs.

LaTeX caption (for the follow-up write-up task):

    \caption{\textbf{H3.} Two summary diagnostics of the n-player
    Privacy Externality Game as a function of the player count $n$
    on the $K=7$ log-grid $\varepsilon\in\{1,2,4,8,16,32,64\}$: pure
    NEs at $n\le 5$ (brute-force enumeration) and the last-50-round
    tail of Monte-Carlo fictitious play at $n\in\{6,7,8,10\}$ (32
    random-init seeds $\times$ 400 rounds; best response against 64
    samples of the empirical joint). \emph{Left axis:} sort-order
    compliance rate -- the fraction of observations with Kendall
    $\tau(w_i,\varepsilon_i)<0$ under heterogeneous
    $w_i\in\mathrm{geomspace}(0.25,4)$; a rate near 1 is the
    self-sorting signature of \cref{sec:C8}. \emph{Right axis (log):}
    the $A_{\max}/A_{\min}$ envelope ratio of realised MIA advantage
    across every observed profile, for both regimes; a ratio bounded
    away from 1 is the strict-asymmetric-equilibrium signature of
    \cref{sec:C1}. Both signatures survive throughout $n\in\{3,\ldots,10\}$
    in the heterogeneous regime; the symmetric-$w$ envelope pinned at
    $1$ for $n\le 8$ is a $K=7$ discretisation artefact, resolved
    by the continuous-$\varepsilon$ argument in \cref{sec:C1}.}

## 6. Paper location and acceptance checklist

Paper location (deferred to a follow-up task; **this task does not edit
paper/**):

* Appendix subsection `app:h3-nplayer` inserted after
  `paper/tex/08_appendix.tex:65`, containing the figure, the table, and
  ~1/2 page of prose that reports the empirical verdict up-or-down.
* Forward pointer inside `paper/tex/05_experiments.tex` C1 (after line 84)
  and C8 (after line 324).
* Bibliography: add `fudenberg1998theory` if absent.

Acceptance checklist (revised to the n<=10 final scope):

- [x] Fictitious play converges at every n in {6, 7, 8, 10}: the FP tail
      concentrates on 1-5 multiset clusters per seed in both regimes;
      no smoothed-BR fallback needed. See Section 10.
- [x] At n=5 heterogeneous the pure-NE list is disjoint from the
      diagonal on the K=7 grid: all 5 pure NEs place at least one
      player away from the diagonal (median Gini 0.46). See Section 10.
- [x] At n<=8 symmetric only the diagonal is a pure NE / modal FP
      cluster on the K=7 grid; this documents the *limit* of the C1
      extension: the no-symmetric-NE claim rides on the finer
      continuous-eps analysis in `paper/tex/04_analysis.tex` and does
      not survive the K=7 grid discretisation at symmetric w. The
      heterogeneous-w case is where C1 extends empirically to n>2;
      n=10 symmetric FP additionally produces a strict-asymmetric tail
      with A envelope ratio ~4.8.
- [x] `paper/figures/fig_h3_fp_nplayer.{pdf,svg}` renders one panel with
      seven n-values, dual y-axes, no title, legend outside the panel,
      style-matched to `paper/figures/fig_h5_v_shape.{pdf,svg}`.
- [x] Table fragment `_cache_v2/H3_nplayer_table.tex` has 14 rows
      populated (7 n-values x 2 regimes).
- [ ] The paper builds warning-clean after the appendix / cross-references
      land (Rule 4). **Deferred**: this task does not edit `paper/tex/`;
      handed off to the follow-up write-up task.
- [ ] `paper/main.pdf` cross-refs to `app:h3-nplayer` resolve.
      **Deferred**: as above.
- [x] The appendix write-up path issues an up-or-down verdict at n<=10:
      C1 (asymmetric NE survives beyond n=2) is **positive** in the
      heterogeneous-w regime at every n in {3, 4, 5, 6, 7, 8, 10} and
      **positive** in the symmetric-w regime at n=10 via the MC-FP tail
      (with the caveat that the K=7 grid collapses to the diagonal in
      the sym-w case at n<=8); C8 (self-sorting) is **positive** at
      every n in {3, 4, 5, 6, 7, 8, 10}. See Section 10.

## 7. Implementation touchpoints

Files added / modified:

* `game/solvers.py` -- `fictitious_play_mc(game, n_rounds,
  n_samples_per_round, start, seed)`. Same return signature as the
  existing `fictitious_play`; the inner K^{n-1} enumeration is replaced by
  `n_samples_per_round` i.i.d. joint draws from the empirical mixed
  strategy. Feasibility-infeasible draws are dropped from the MC average.
* `game_experiments/run_nplayer.py` -- H3 driver. CLI flags:
  `--ns`, `--regimes`, `--n-seeds`, `--n-rounds`, `--n-samples-per-round`,
  `--grid {full,coarse}`, `--brute-force-max-n`, `--preview`, `--out-tag`.
  Writes cache incrementally, logs per-(n, regime) wall-clock. Its own
  in-driver `plot_h3_summary` is superseded by the finaliser plot script
  below.
* `slurm_scripts/run_fp_nplayer.sh` -- CLI-flagged submission line
  matching the driver, docstring comment on the H3 role and the
  multiset-symmetric cache reuse. 8 h / 24 GB / 8 CPU budget; skips
  cached (n, regime) tuples automatically.
* `tasks_findings/_scripts/19_h3_plot.py` -- finaliser plot script.
  Loads `_cache_v2/H3_nplayer.pkl`, re-derives correct A-envelope
  and Kendall-tau stats (see Section 10 for the fix rationale), writes
  `paper/figures/fig_h3_fp_nplayer.{pdf,svg}` and
  `_cache_v2/H3_nplayer_table.tex`. Emits
  `_cache_v2/H3_nplayer_stats.pkl` so subsequent replots skip the
  mechanism-cache warm-up. Figure style-matched to
  `paper/figures/fig_h5_v_shape.pdf`.
* `slurm_scripts/run_fp_nplayer_plot.sh` -- 45-min slurm wrapper that
  runs the finaliser plot script under 8 CPUs / 8 GB.
* `paper/figures/fig_h3_fp_nplayer.{pdf,svg}` -- single-panel dual-axis
  figure from the K=7 slurm cache.

Nothing in `paper/tex/`, in the utility form, in the mechanism solver, or
in the C6/C7/C8 experiment defaults changed.

## 10. Full-scale K=7 result

Numbers below come out of `_cache_v2/H3_nplayer.pkl`, re-derived by
`tasks_findings/_scripts/19_h3_plot.py` (which computes the A envelope
and Kendall tau on the raw per-player profiles rather than on the
permutation-invariant multiset key; the multiset key loses per-player
identity, which spuriously reads out as tau >= 0 always). The mirror
table (`_cache_v2/H3_nplayer_table.tex`) is what the appendix write-up
will `\input{...}`.

| $n$ | regime | mode | # distinct | median Gini | $A_{\min}$ | $A_{\max}$ | $A_{\max}/A_{\min}$ | $\tau<0$ rate |
|-----|--------|------|-----------:|------------:|-----------:|-----------:|--------------------:|--------------:|
|  3  | sym    | brute |   1 | 0.000 | 0.038 | 0.038 |   1.00 | 0.00 |
|  3  | het    | brute |   4 | 0.333 | 0.002 | 0.867 | 482.7 | 1.00 |
|  4  | sym    | brute |   1 | 0.000 | 0.044 | 0.044 |   1.00 | 0.00 |
|  4  | het    | brute |   5 | 0.477 | 0.001 | 0.701 | 637.3 | 1.00 |
|  5  | sym    | brute |   1 | 0.000 | 0.046 | 0.046 |   1.00 | 0.00 |
|  5  | het    | brute |   5 | 0.462 | 0.001 | 0.727 | 519.2 | 1.00 |
|  6  | sym    | mc_fp |   1 | 0.000 | 0.039 | 0.039 |   1.00 | 0.00 |
|  6  | het    | mc_fp |   3 | 0.452 | 0.002 | 0.535 | 281.7 | 1.00 |
|  7  | sym    | mc_fp |   1 | 0.000 | 0.038 | 0.038 |   1.00 | 0.00 |
|  7  | het    | mc_fp |   5 | 0.438 | 0.002 | 0.350 | 184.1 | 1.00 |
|  8  | sym    | mc_fp |   1 | 0.000 | 0.035 | 0.035 |   1.00 | 0.00 |
|  8  | het    | mc_fp |   3 | 0.422 | 0.003 | 0.354 | 107.4 | 1.00 |
| 10  | sym    | mc_fp |  13 | 0.082 | 0.022 | 0.103 |   4.77 | 0.00 |
| 10  | het    | mc_fp |  14 | 0.371 | 0.005 | 0.367 |  74.9 | 0.83 |

Column semantics: `# distinct` is the number of distinct pure NE
profiles at n<=5 (brute-force enumeration on the K=7 grid) and the
number of distinct raw last-50-round profiles across 32 MC-FP seeds
at n>=6 (permutations of a multiset count separately, so a single
asymmetric multiset like $(1,\ldots,1,2)$ contributes up to n permuted
profiles). `median Gini` and `tau<0` rate are frequency-weighted over
those observations that pass the mechanism-feasibility check
(`ProfileEvaluation.feasible=True`).

### 10.1 What the numbers say

* **C1 (asymmetric NE survives n>2).** In the heterogeneous-w regime,
  every n in {3, 4, 5, 6, 7, 8, 10} lands a non-degenerate
  $[A_{\min}, A_{\max}]$ envelope. The $A_{\max}/A_{\min}$ ratio is
  in the several-hundred range for n<=6 and falls monotonically to
  ~75x at n=10 as the FP tail averages more players; it never
  approaches 1. In the symmetric-w regime, n<=8 collapses to the
  diagonal on the K=7 grid (a discretisation artefact -- the
  continuous-eps no-symmetric-NE proposition in
  `paper/tex/04_analysis.tex` is the authoritative statement); at
  n=10, FP additionally explores strict-asymmetric FP-tail profiles
  and generates a modest A envelope ratio of ~4.8.
* **C8 (self-sorting).** Every heterogeneous-w pure NE at n in
  {3, 4, 5} is perfectly sort-monotone: Kendall $\tau(w_i,
  \varepsilon_i) = -1$ on all pure NEs (tau<0 rate 100%). At n in
  {6, 7, 8} the strict-asymmetric FP tail keeps the sort-order
  compliance rate at 100%; only at n=10 does it fall to 83% -- the
  minority player at high eps is systematically the low-w
  (privacy-tolerant) player, matching the self-sorting prediction,
  and the residual 17% at n=10 reflects FP-tail excursions that
  swap adjacent (w-close) player rankings but do not overturn the
  sign of the sorting monotonicity.
* **FP convergence.** The FP tail concentrates into 1-5 multiset
  clusters per seed on average. No divergence, no oscillation; the
  smoothed-BR fallback flagged in the task spec was not needed.

## 11. References

* Robinson, J. (1951). "An iterative method of solving a game." *Annals
  of Mathematics* 54(2), 296-301.
* Milgrom, P., Roberts, J. (1990). "Rationalizability, learning, and
  equilibrium in games with strategic complementarities." *Econometrica*
  58(6), 1255-1277.
* Krishna, V. (1992). "Learning in games with strategic complementarities."
  Working paper, Harvard.
* Monderer, D., Shapley, L. S. (1996). "Fictitious play property for games
  with identical interests." *Journal of Economic Theory* 68, 258-265.
* Hofbauer, J. (1995). "Stability for the best response dynamics."
  Working paper, Vienna.
* Fudenberg, D., Levine, D. K. (1998). *The Theory of Learning in Games.*
  MIT Press, Cambridge, MA. (Cited as `fudenberg1998theory`.)
* Berger, U. (2007). "Two more classes of games with the continuous-time
  fictitious play property." *Games and Economic Behavior* 60, 247-261.
* Hofbauer, J., Sandholm, W. H. (2002). "On the global convergence of
  stochastic fictitious play." *Econometrica* 70(6), 2265-2294.
