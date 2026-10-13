# Task 15: Regenerate figures with $A_{\mathrm{PLD},i}$ axis labels

**DONE (2026-07-09).** All eight figures regenerated plot-only from the
existing caches via Slurm job 177257 (CPU, 1m19s): the six $A$-label
figures plus `fig_C7_lambda_ratio` / `fig_C8_asymmetric_lambda` (the C7/C8
paper figures are produced by `tasks_findings/_scripts/polish_C6_C7_C8.py`,
not `run_locus.py`; both scripts were updated). Stale
`fig_C7_w_ratio.*` / `fig_C8_asymmetric_w.*` deleted, the two
`\includegraphics` references updated, `main.pdf` rebuilt clean (24 pages).

The paper's per-player advantage notation changed from $A_i$ to
$A_{\mathrm{PLD},i}$ (macro `\Apld{i}` in `paper/main2.tex`). The prose,
equations, and `tex/table_NE.tex` are updated, but six included figures
still carry the old $A_i$ / $A_1$ / $A_2$ axis labels:

- figures/fig_C1_payoff.pdf        (game_experiments/run_continuous.py)
- figures/fig_C2_asymmetric.pdf
- figures/fig_H1_n2_measured_NE.pdf
- figures/fig_h2_scale.pdf         (game_experiments/run_scale.py)
- figures/fig_h4_aware_target.pdf  (game_experiments/run_h4.py)
- figures/fig_slack_panel.pdf

Fix: change the matplotlib label strings (`$A_i$` → `$A_{\mathrm{PLD},i}$`
etc.) in the generating scripts and replot from the existing
`_cache_v2` pickles — no recomputation needed, so this is a plot-only
rerun (fast; per repo Rule 3 still submit via Slurm CPU if it exceeds a
couple of minutes). Then rebuild `paper/main2.pdf` and delete stale
`.svg` siblings.

## Extension: weight notation w -> lambda (2026-07-09)

The paper now writes the privacy weight as $\lambda_i$ (was $w_i$); the
coalition malice weight became $\kappa$. Two more figures carry stale
$w$ labels and w-based filenames (Rule 0: rename files whose meaning
changed):

- figures/fig_C7_w_ratio.pdf   -> regenerate as fig_C7_lambda_ratio.pdf
  (game_experiments/run_locus.py or post_v2.py; axis $w_2/w_1$ -> $\lambda_2/\lambda_1$)
- figures/fig_C8_asymmetric_w.pdf -> fig_C8_asymmetric_lambda.pdf

Update the two \includegraphics references (tex/05_experiments.tex:169,
tex/09_appendix.tex:573) and delete the stale files after regeneration.
