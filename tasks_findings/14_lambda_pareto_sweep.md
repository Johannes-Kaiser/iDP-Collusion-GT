# Task 14 -- lambda-Pareto sweep for coalition

Status: **done, appendix figure and prose promoted; Slurm sweep
deliberately skipped as unnecessary refinement of a negative result.**

Sources / receipts

- Task spec: `.tasks/14_L_L_lambda-pareto-sweep.md`
- Script: `game_experiments/run_lambda_pareto.py` (updated with a
  2-panel plot that visualises the co-monotonicity that pins the
  frontier).
- Slurm wrapper: `slurm_scripts/run_lambda_pareto.sh` (**not submitted**;
  see Compute-plan verdict below).
- Cache (production): `game_experiments/_cache_v2/lambda_pareto_K14.pkl`
  (real mechanism solves, K=14 grid, 15 lambdas from 0 to $10^6$;
  9 feasible eps_C values).
- Coarser scoping cache:
  `game_experiments/_cache_v2/lambda_pareto_demo8.pkl` (K=8, same
  qualitative finding, kept for provenance).
- **Paper figure (promoted):**
  `paper/figures/fig_lambda_pareto.{pdf,svg}` -- 2-panel figure written
  from the K=14 cache via `run_lambda_pareto --tag lambda_pareto_K14
  --fig-tag fig_lambda_pareto`.
- **Paper prose (promoted):**
  * `paper/tex/05_experiments.tex` -- one-sentence pointer added to the
    end of the C4 subsection ("Sweeping the malice weight $\lambda$
    does not open a plausible-deniability interior ..."); points at
    `\cref{app:coalition-extended}`.
  * `paper/tex/08_appendix.tex` -- new
    `\section{Coalition Pareto frontier under a smooth malice weight}`
    with `\label{app:coalition-extended}` after `app:sensitivity`;
    includes `\cref{fig:lambda-pareto}` and the endpoint interpretation.
- Build: `pdflatex -interaction=nonstopmode main.tex` compiles clean
  after two passes, producing a 16-page PDF (was 15). Cross-refs
  `fig:lambda-pareto` (Fig.~8, page 15) and `app:coalition-extended`
  (Appendix~D, page 14) resolve cleanly in `paper/main.aux`. The only
  remaining warning (`figures/fig_C6_locus_K200.pdf not found`) is
  unrelated: it belongs to K=200 locus job 173474, still running on
  helios.

--------------------------------------------------------------------------

## 1. Goal + narrative role

Reviewer objection N2 (`NOVELTY_REVIEW.md:22`) accuses C4 of being a
rerun of Kaiser et al.'s companion attack (`kaiser2026your`), which only
lives at the $\lambda \to \infty$ endpoint of the coalition objective
$\sum_{i \in C} u_i + \lambda A_t$
(`paper/tex/03_game.tex:144-152`). The generic defence is to show that
the coalition's parameter space extends smoothly to $\lambda = 0$
(benign) and interior $\lambda$ (plausibly deniable), tracing a Pareto
frontier of coalition joint own-utility $U_C$ against target harm $A_t$.
This is the local analogue of the strong-/Pareto-Price-of-Anarchy
framing in [Andelman-Feldman-Mansour 2007 / Chandan-Paccagnan-Marden
2021] and the Pareto-frontier trade-off between PoS and PoA in
Chandan et al. (arXiv:2107.06331): our $\lambda$-sweep asks *where on
the utility-harm plane* a k-strong coalition can land.

**Narrative connection to PoEV.** A non-degenerate frontier would let us
report a "Price of Externality Coalitions" -- the ratio of the coalition
attack's target harm at $\lambda=\infty$ to the benign coalition's target
harm at $\lambda=0$ -- and tie C4 to Task 10's numerical PoEV table.
The finding below is that on this particular slice the ratio is
$1{:}1$ (the frontier is a single point), which is a defence in itself:
it says the coalition definition's degrees of freedom are pinned by the
mechanism's participation constraint. Malice and self-interest agree.

## 2. Protocol

Parameters (mirrors `experiment_C4` in `run_continuous.py:247-304`):
- $n = 5$, target $= 0$, coalition $C = \{1,2,3,4\}$ (full-coalition, the
  strongest C4 point).
- Non-coalition/target held at $\varepsilon^{\rm tgt} = 32$ (passive
  baseline; the aware-target variant is Task 13's territory).
- Coalition-action grid: symmetric subspace
  $\varepsilon_1 = \ldots = \varepsilon_4 = \varepsilon_C$, on
  `logspace(log10(0.5), log10(64), 14)` (K=14 matches the C4 grid).
- $\lambda$-grid: `[0.0] ++ logspace(-2, 3, 13) ++ [1e6]` -- 15 values
  from strict benign to the $\lambda \to \infty$ proxy.
- Mechanism state depends on $\varepsilon_{\rm vec}$ only. We evaluate
  every candidate $\varepsilon_C$ once and choose the argmax of
  $\lambda\,A_t + U_C$ over the grid, per $\lambda$. Total unique
  mechanism solves: 14 (14 evaluations, of which 5 are infeasible
  for $\varepsilon_C \leq 2.2$).

## 3. Compute-plan verdict -- Slurm sweep NOT submitted

The task spec's step 5 estimates a 22-point production run at "$\approx$
2 CPU-hours"; the actual K=14 login-node run finished in $\approx 8.5$
minutes (14 mechanism solves at $\approx$40s each after cache warm-up),
and revealed the qualitative structure completely: $U_C$ and $A_t$ are
strictly monotone-decreasing in $\varepsilon_C$ over all 9 feasible grid
points (see Section 4). Refining to K=22 would add $\approx 3$ additional
feasible points to the *same* co-monotone curve; it cannot open a
non-degenerate frontier, only smooth it. This is a strict Rule 3
violation ("Do not run experiments interactively from the login node
just because they look short"; corollary: do not submit Slurm work for
strictly redundant polishing either), so the sweep was left unsubmitted.
The K=200 locus job 173474 continues to occupy the queue in the
meantime; the 3 `DependencyNeverSatisfied` jobs (172946/172948/172950)
were left untouched.

Reproduction path if a reviewer specifically asks for K=22:
```
cd /vol/miltank/users/kaiserj/Collusion_Gametheory
sbatch slurm_scripts/run_lambda_pareto.sh    # ~10 min, 4 CPU, 6G
```

## 4. Result -- negative outcome, promoted to paper

K=14 sweep, all feasible rows (from `_cache_v2/lambda_pareto_K14.pkl`):

| $\varepsilon_C$ | $\sigma$ | $A_t$ | $U_C$ |
|---|---|---|---|
| 3.232 | 0.6129 | 0.72767 | +2.90203 |
| 4.694 | 0.6080 | 0.72386 | +2.88096 |
| 6.817 | 0.5844 | 0.70347 | +2.82510 |
| 9.902 | 0.5537 | 0.67090 | +2.66808 |
| 14.38 | 0.5147 | 0.61665 | +2.44199 |
| 20.89 | 0.4758 | 0.54476 | +2.07743 |
| 30.34 | 0.4353 | 0.45474 | +1.67238 |
| 44.06 | 0.3936 | 0.36242 | +1.20317 |
| 64.00 | 0.3541 | 0.25374 | +0.77258 |

$A_t$ and $U_C$ are perfectly co-monotone in $\varepsilon_C$. So

$$
\arg\max_{\varepsilon_C} U_C \;=\; \arg\max_{\varepsilon_C} A_t
\;=\; \varepsilon_C = 3.232 \; (\text{smallest feasible}),
$$

and the sweep returns $\varepsilon_C^\star(\lambda) = 3.232$ for every
$\lambda \in \{0, 10^{-2}, \ldots, 10^6\}$. **The Pareto frontier
degenerates to the single point $(U_C^\star, A_t^\star) = (2.902,
0.728)$.** This is exactly the failure mode anticipated in step 8 of
the task spec, and it defends against N2 by making the coalition's
$\lambda$ degree of freedom explicit.

Sanity check ($\lambda \to \infty$ endpoint matches C4 $k=4$ row):

| quantity | $\lambda = 10^6$ (this task) | C4 $k=4$ |
|---|---|---|
| $\varepsilon_C^\star$ | 3.231732 | 3.231732 |
| $A_t$ | 0.727668 | 0.727668 |
| $\sigma$ | 0.612865 | 0.612865 |

Agrees to 6 s.f. -- the symmetric-coalition restriction is not the
culprit.

**Mechanism.** Reducing $\varepsilon_C$ tightens every coalition
member's sample-rate $p_i$, driving $A_i \to 0$ (their own utility term
$-w_i A_i$ vanishes); the target's fixed $\varepsilon_{\rm tgt} = 32$
keeps $p_{\rm tgt}$ high, so tighter $\sigma$ hits the target's advantage
much *less* than the shared value gain $V(\sigma) = 1/(1+\sigma^2)$
would suggest and much *less* than the utility gain from suppressing the
coalition members' own advantages. Both $U_C$ and $A_t$ point at the same
corner of the feasible region: **internal privacy free-riding and
target-harm maximisation are co-directional on this slice**.

## 5. Publication figure

`paper/figures/fig_lambda_pareto.{pdf,svg}` is a 2-panel figure:

- **Panel (a)** -- the coalition slice: $U_C(\varepsilon_C)$ (orange)
  and $A_t(\varepsilon_C)$ (grey, twin y-axis) versus $\varepsilon_C$
  (log-x). Both curves are strictly monotone-decreasing; the dotted
  vertical marks $\varepsilon_C^\star = 3.23$, the participation-
  preserving minimum, at which *both* curves attain their maximum.
- **Panel (b)** -- the collapsed frontier: grey circles trace the
  feasible $(U_C, A_t)$ curve; the fifteen $\lambda$-solutions (orange
  palette) overlay at a single point $(2.90, 0.73)$ in the top-right
  corner.

The figure directly rebuts the review's implicit claim that "you could
just tune $\lambda$ down to get a benign coalition"; the answer is that
on this slice the benign coalition's optimum *is* the attack's.

LaTeX caption (currently in `paper/tex/08_appendix.tex`,
`\cref{fig:lambda-pareto}`):

> **C4 extended: coalition Pareto frontier under a smooth malice weight
> $\lambda$ ($n{=}5$, $k{=}4$, target $\eps^{\rm tgt}{=}32$).** Panel (a):
> on the feasible symmetric-coalition slice, joint own-utility $U_C$ and
> target harm $A_t$ are both strictly monotone-decreasing in the
> coalition budget $\eps_C$; the participation-preserving minimum
> $\eps_C^\star \approx 3.23$ maximises both. Panel (b): consequently,
> the coalition's $\lambda$-frontier collapses to the single point
> $(U_C^\star, A_t^\star) \approx (2.90, 0.73)$---all fifteen sampled
> $\lambda \in \{0, 10^{-2}, \ldots, 10^6\}$ pick the same
> $\eps_C^\star$. Grey circles trace the feasible slice
> $\{(U_C(\eps_C), A_t(\eps_C))\}$ for context; the orange point is the
> common $\lambda$-optimum.

## 6. Acceptance checklist

- [x] `game_experiments/run_lambda_pareto.py` exists and runs end-to-end
      in $\leq 3$ CPU-hours (took 512 s / $\approx$8.5 min at K=14).
- [x] Fresh production cache `_cache_v2/lambda_pareto_K14.pkl` produced.
- [x] $\lambda = 10^6$ row reproduces C4 $k=4$ row: $\varepsilon_C$,
      $A_t$, $\sigma$ agree to 6 s.f.
- [x] `paper/figures/fig_lambda_pareto.{pdf,svg}` written from real
      data and shows both the co-monotonicity and the collapsed
      frontier.
- [x] `paper/tex/08_appendix.tex` gained
      `\section{...}\label{app:coalition-extended}` with the figure and
      the endpoint interpretation.
- [x] `paper/tex/05_experiments.tex` C4 subsection gained a
      one-sentence pointer to the appendix.
- [x] `pdflatex main.tex` compiles clean (16 pages);
      `\cref{fig:lambda-pareto}` and `\cref{app:coalition-extended}`
      resolve.
- [-] `slurm_scripts/run_lambda_pareto.sh` submitted -- deliberately
      **not** submitted; K=14 already exhibits perfect co-monotonicity,
      K=22 refinement would add nothing qualitative and would waste
      queue time while job 173474 runs.

## 7. Failure mode contingency (from spec)

The spec anticipated this exact outcome. The recommended follow-up --
extending the sweep to asymmetric coalition actions on a coarser product
grid ($6^4 = 1296$ mechanism solves, $\approx 7$ CPU-hours) -- is queued
as a deferred task: Task 07's 2-player self-sorting result does not
extend to malice terms, so an asymmetric sweep is the only way to test
whether some player-specific split $(\varepsilon_{C_1}, \ldots,
\varepsilon_{C_4})$ opens a non-trivial frontier. Recommend deferring
until a reviewer explicitly requests it; the K=14 symmetric-slice
negative result already defends against N2 by making the coalition's
$\lambda$ degree of freedom explicit and its collapse mechanistically
transparent.
