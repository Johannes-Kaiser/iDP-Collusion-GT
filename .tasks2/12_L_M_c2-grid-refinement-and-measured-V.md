# Refine the group-size sweep grid; optional measured-V calibration

**Priority:** L  **Effort:** M (CPU Slurm)  **Category:** experiment

## C2 grid
The group-size-ratio sweep behind `fig_C2_asymmetric` runs at K=9
(`game_experiments/run_continuous.py`, KGRID_FAST) because no cache is
reusable across ratios. Every other headline sweep in main2 is K=200. The
qualitative claim (minority plays lower budget, absorbs lower advantage) is
robust, but the quoted equilibrium values (eps_1* in [1.5,3], eps_2* ~ 10,
A_2* in [0.24, 0.32]) are coarse-grid values. Rerun at K>=50 per ratio on
Slurm and refresh figure + quoted numbers in Sec. V ("Who absorbs the
externality?").

## Measured-V option
EXPERIMENT_IDEAS.md proposes calibrating w from measured ranges
(w = range(V_measured)/range(A_measured)) and/or using measured
shadow-accuracy as V. With the H1 cache in place
(`game_experiments/_cache_H1/`), re-derive the analytical game with
V taken from measured test accuracy — a cheap bridge between the analytical
and measured instances that would strengthen Sec. V-H.
