#!/bin/bash

#SBATCH --job-name=peg_h4_aware
#SBATCH --output=./out/peg_h4_aware-%A.out
#SBATCH --error=./out/peg_h4_aware-%A.err
#SBATCH --time=02:00:00              # safety cushion; expected wall ~15-30 min on 24 cores
#SBATCH --cpus-per-task=24           # PEG_N_WORKERS uses these
#SBATCH --mem=12G                    # ~500 MB per worker for PLD/RDP LRU caches

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=24

source ./.venv/bin/activate

# H4: aware-target coalition Stackelberg (companion to C4). Sweeps
#   k in {1,2,3,4}  x  eps_C (14 log-spaced)  x  eps_t (14 log-spaced)
# = 784 mechanism solves, farmed to 24 workers. Also loads the C4
# passive-target baseline from _cache_v2/C4.pkl for the overlay.
python -m game_experiments.run_h4
