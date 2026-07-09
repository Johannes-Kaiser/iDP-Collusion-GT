# Outstanding robustness sweeps: delta, batch/epochs, RDP orders (R1/R4/R5)

**Priority:** M  **Effort:** M (CPU Slurm)  **Category:** experiment

## Why
main2's sensitivity appendix states: "Sweeps over delta, batch size and
epoch count, and the accountant's Renyi orders are deferred to follow-up
runs." T17 completed only R2 (PLD discretisation, max |dA| = 1.19e-2) and
R3 (participation threshold: |NE| = 6 / 12 / 12 at p_min = 1e-3 / 1e-5 /
1e-7). R1 (delta sweep), R4 (batch/epoch sweep), R5 (RDP-order sweep) need
Slurm reruns of the K=200 mechanism grid per setting.

## What
Extend `tasks_findings/_scripts/17_r1_r5.py`; one `run_locus.py`-style K=200
grid per setting (48 cores, ~1-3 h each per T18's timing). Report |NE|,
off-diagonal fraction, externality-sign fraction per setting; add one
sentence + numbers to app:sensitivity in `paper/tex2/09_appendix.tex`.

## Acceptance
Numbers in the appendix replace the "deferred to follow-up runs" sentence.
