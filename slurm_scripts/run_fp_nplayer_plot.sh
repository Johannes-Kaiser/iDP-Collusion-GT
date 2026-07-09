#!/bin/bash

#SBATCH --job-name=peg_h3_plot
#SBATCH --output=./out/peg_h3_plot-%A.out
#SBATCH --error=./out/peg_h3_plot-%A.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source ./.venv/bin/activate

# T19 finaliser: regenerate paper/figures/fig_h3_fp_nplayer.{pdf,svg} and
# the tex table from the K=7 slurm cache game_experiments/_cache_v2/
# H3_nplayer.pkl. Re-evaluates every pure-NE / tail-round profile via
# solve_mechanism_cached so we can report the correct A envelope and
# Kendall tau (the driver's plot_h3_summary had computed both off the
# permutation-invariant multiset, which loses per-player identity).
python tasks_findings/_scripts/19_h3_plot.py
