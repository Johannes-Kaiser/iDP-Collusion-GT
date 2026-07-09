# BR-PEG follow-ups: strategising target, regret rates, lambda-Pareto

**Priority:** L  **Effort:** M-H  **Category:** theory/experiment

## What main2 leaves open (stated in Sec. Discussion)
1. Two-sided repeated game: a target who manages what her responses reveal
   (Aumann-Maschler). Even a small simulation (H plays a revelation-managing
   policy vs the posterior-greedy M) would sharpen the "type uncertainty is
   not a durable shield" message or qualify it.
2. Finite-horizon regret RATES in the style of Balcan et al. — the current
   argument gives lock-in after finite time (constant regret) for exact
   identification; with observation noise (the fig_h8 instance) an explicit
   T-dependence bound is derivable from the KL separation of the type
   likelihoods.
3. lambda-Pareto coalition frontier (T14): script exists
   (`game_experiments/run_lambda_pareto.py`), production Slurm sweep never
   submitted. Would quantify the benign-to-malicious transition of
   Def. coalition (eq. coalition-util) as lambda sweeps 0 -> infinity.

## Acceptance
Any of the three lands as an appendix subsection with numbers, or a
decision to defer is recorded here.
