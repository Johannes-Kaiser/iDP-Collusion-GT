# Run the toy federated n=3 shadow-model experiment for real

**Priority:** H  **Effort:** M-H (GPU, Slurm)  **Category:** experiment

## Why
`paper/figures/fig_C9_federated_measurement_ceiling.{pdf,svg}` was rendered
from SYNTHETIC per-silo advantages (T25 Slurm run was deliberately declined;
see tasks_findings/25_*, .tasks/25_*). It is therefore EXCLUDED from main2
— do not re-add it as if measured. The n=3 federated instance would be the
natural companion to the n=2 measured instance.

## What
Submit the T25 pipeline (infra shared with T24, which is complete): 15
feasible profiles on the n=3 grid, shadow models per profile, LiRA scoring
per silo. Then either produce a real `fig_C9`-style figure or fold the
result into the measured-instance section of main2. Delete the synthetic
figure files once superseded (CLAUDE.md Rule 0/2).

## Acceptance
- `game_experiments/_cache_T25/` populated from a real Slurm run;
- figure regenerated from measured data only, or an explicit decision
  recorded here to drop the experiment.
