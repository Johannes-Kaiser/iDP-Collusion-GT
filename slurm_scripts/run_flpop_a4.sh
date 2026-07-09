#!/bin/bash

#SBATCH --job-name=flpop_a4
#SBATCH --output=./out/flpop_a4-%A.out
#SBATCH --error=./out/flpop_a4-%A.err
#SBATCH --partition=universe
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=16
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ./.venv/bin/activate

# A4 pre-flight feasibility/spread scan: compare candidate action grids on the
# first 10 populations (uniform + random profiles) at DISC_FAST. Decides the
# frozen action grid for E1/E2.
python -m game_experiments.flpop_a4_scan --pops 10 --n-random 40 \
    --grids default,low,mid
