# Slack monotonicity: isolated numerical violations need tighter solver

**Priority:** M  **Effort:** L-M  **Category:** verification

## Issue
T11 (`tasks_findings/_scripts/11_slack_summary.txt`) reports slack
monotonicity in eps_j on only 131/162 feasible pairs at anchor eps_i=4, and
a worst raw DECREASE of -0.280 at eps_i=16 (attributed to an isolated
mechanism-solver artifact). Prop. slack-monotonicity in main2 claims strict
monotonicity analytically; the paper does not currently quote the 131/162
number. A referee reproducing the grid could find the violations and read
them as counterexamples.

## What
Re-run the slack panel with tighter solver tolerance (same knob as task 04)
and quantify how many violations remain; they should shrink toward zero if
they are tolerance artifacts. Then either (a) add one honest sentence to
app:certificate quantifying residual violations, or (b) if violations
persist at tight tolerance, investigate whether they cluster at the
saturation boundary where the lemma's hypotheses fail, and scope the prose.

## Acceptance
app:certificate (or the slack figure caption) carries an explicit
violation count with an explanation that survives tight tolerance.
