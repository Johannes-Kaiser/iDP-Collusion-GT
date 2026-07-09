#!/bin/bash

#SBATCH --job-name=peg_locus_K200
#SBATCH --output=./out/peg_locus_K200-%A.out
#SBATCH --error=./out/peg_locus_K200-%A.err
#SBATCH --time=06:00:00              # safety cushion; expected wall ~1-2h on 48 cores
#SBATCH --cpus-per-task=48           # PEG_N_WORKERS uses these
#SBATCH --mem=24G                    # ~500 MB per worker for PLD/LRU caches at K=200

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=48

source ./.venv/bin/activate

# One process runs C6, C7, C8 in sequence; the shared mech-grid at K=200
# is computed once (parallelised across 48 workers) and reused for all
# weight settings.
python -m game_experiments.run_locus --K6 200 --K7 200 --K8 200
