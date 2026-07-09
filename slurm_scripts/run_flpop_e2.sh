#!/bin/bash
#
# E2 -- adaptive agents discover the equilibrium (Phase C of the FL-population
# experiments). CPU only: E2 trains no models. All level-2 (PLD-exact) game
# dynamics on one fixed representative population (pop_id 0).
#
# Modelled on slurm_scripts/run_locus_K200.sh: PEG_N_WORKERS worker processes
# evaluate unique eps profiles; the 50 seeds run lock-step so each iteration's
# candidate profiles form one parallel batch over a shared oracle cache.
#
# Usage:
#   sbatch slurm_scripts/run_flpop_e2.sh references
#   sbatch slurm_scripts/run_flpop_e2.sh learn myopic
#   sbatch slurm_scripts/run_flpop_e2.sh learn noisy --n-seeds 20
#   sbatch slurm_scripts/run_flpop_e2.sh learn exp3  --n-seeds 20

#SBATCH --job-name=flpop_e2
#SBATCH --output=./out/flpop_e2-%A.out
#SBATCH --error=./out/flpop_e2-%A.err
#SBATCH --partition=universe
#SBATCH --time=08:00:00              # cushion; expected 1-3 h on 48 cores
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G                    # ~500 MB/worker for PLD + RDP LRU caches

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=48

source ./.venv/bin/activate

python -m game_experiments.run_flpop_e2 "$@" --workers 48
