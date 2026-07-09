#!/bin/bash
#
# T18 / H2: Scale (sensitivity-based iDP) cross-mechanism check.
#
# Runs the Scale-mode C1 (K=22), C4 (n=5), C7 (K=200) and C8 (K=200) sweeps
# in one process. The Scale mechanism grid at K=200 is computed once
# (parallelised across 48 workers) and reused for the C7/C8 weight sweeps.
#
# Modelled line-for-line on `slurm_scripts/run_locus_K200.sh`; only the
# job name, output/error prefixes and the entry-point module differ.

#SBATCH --job-name=peg_h2_scale
#SBATCH --output=./out/peg_h2_scale-%A.out
#SBATCH --error=./out/peg_h2_scale-%A.err
#SBATCH --time=03:00:00              # safety cushion; expected ~1 h on 48 cores
#SBATCH --cpus-per-task=48           # PEG_N_WORKERS uses these
#SBATCH --mem=24G                    # ~500 MB per worker for PLD/LRU caches at K=200

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=48

source ./.venv/bin/activate

# One process runs Scale-mode C1 (K=22 for the sampling-iDP-matched grid),
# C7 (K=200), C8 (K=200) and the K=200 hero comparison figure in sequence.
# The shared Scale mech-grid at K=200 is computed once (parallelised across
# 48 workers) and reused for C7, C8 and the hero.
python -m game_experiments.run_scale --K1 22 --K7 200 --K8 200 --Khero 200
