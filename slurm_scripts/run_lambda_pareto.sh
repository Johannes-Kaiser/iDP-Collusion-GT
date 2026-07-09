#!/bin/bash

#SBATCH --job-name=peg_lambda_pareto
#SBATCH --output=./out/peg_lambda_pareto-%A.out
#SBATCH --error=./out/peg_lambda_pareto-%A.err
#SBATCH --time=00:30:00              # expected ~8 min at K=22; safety cushion
#SBATCH --cpus-per-task=4            # one worker; mechanism solve is single-threaded
#SBATCH --mem=6G                     # PLD/LRU caches for 5-player n-tuples

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=1

source ./.venv/bin/activate

# Task 14 -- lambda Pareto sweep for coalition (n=5, k=4, target eps=32).
# Loop is |coal_grid| mechanism solves; each ~20s on one CPU. Total ~7 min.
python -m game_experiments.run_lambda_pareto --K 22
