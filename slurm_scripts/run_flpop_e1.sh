#!/bin/bash

#SBATCH --job-name=flpop_e1
#SBATCH --output=./out/flpop_e1-%A_%a.out
#SBATCH --error=./out/flpop_e1-%A_%a.err
#SBATCH --partition=universe
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --array=0-99%25

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ./.venv/bin/activate

# E1 best-response dynamics: one array task per population (seed = array id).
# GRID is passed in via --export=GRID=<name> (default: default).
GRID="${GRID:-default}"
WMU="${WMU:-0.0}"
python -m game_experiments.run_flpop_e1 --pop "${SLURM_ARRAY_TASK_ID}" \
    --grid "${GRID}" --w-mu "${WMU}" --disc 0.01 --max-rounds 200
