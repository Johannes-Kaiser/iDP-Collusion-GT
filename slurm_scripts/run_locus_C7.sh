#!/bin/bash

#SBATCH --job-name=peg_C7
#SBATCH --output=./out/peg_C7-%A.out
#SBATCH --error=./out/peg_C7-%A.err
#SBATCH --time=00:45:00              # PLD-accounting is single-threaded; ~15 min per ratio, 6 ratios
#SBATCH --cpus-per-task=2             # single-threaded workload; one extra core for headroom
#SBATCH --mem=4G                      # sweep grids are small; NE cache < 100 MB

export PYTHONUNBUFFERED=true

source ./.venv/bin/activate

python -m game_experiments.run_locus --only C7 --K7 22
