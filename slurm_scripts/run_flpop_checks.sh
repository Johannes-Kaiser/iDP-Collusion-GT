#!/bin/bash

#SBATCH --job-name=flpop_chk
#SBATCH --output=./out/flpop_chk-%A.out
#SBATCH --error=./out/flpop_chk-%A.err
#SBATCH --partition=universe
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

export PYTHONUNBUFFERED=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ./.venv/bin/activate

GRID="${GRID:-default}"
# D1 oracle self-consistency + D3 feasibility audit on 3 populations.
python -m game_experiments.flpop_checks --check all --pops 0,1,2 --grid "${GRID}" --disc 0.01
