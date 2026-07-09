#!/bin/bash

#SBATCH --job-name=peg_fp_nplayer
#SBATCH --output=./out/peg_fp_nplayer-%A.out
#SBATCH --error=./out/peg_fp_nplayer-%A.err
#SBATCH --time=08:00:00              # densification of n in {3..10}; n=50 dropped (see T19 findings)
#SBATCH --cpus-per-task=8            # run_nplayer serial in Python; 8 cores for BLAS/PLD threading
#SBATCH --mem=24G                    # LRU mech cache stays <8GB per prior H3 traces at n<=10

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source ./.venv/bin/activate

# H3: fictitious-play (Monte-Carlo BR expectation) sweep of the symmetric
# n-player PEG for n in {3, 4, 5, 6, 7, 8, 10}. Symmetric and heterogeneous
# w regimes; 32 random-init seeds per configuration; 400 rounds each; last
# 50 rounds clustered by the permutation-invariant multiset of eps values.
# Pure-NE brute-force at n <= 5 (K=7 grid). Mechanism-solve cache is
# multiset-symmetric (see game/mechanism.py::solve_mechanism_cached) and
# shared across seeds within each (n, regime). n=3, 5, 10 are already
# cached from the prior job (173491) and skipped via run_config's cache
# guard; this job lands the n in {4, 6, 7, 8} slots that densify the
# small-n trend curve. n=50 was dropped after the prior job's OOM; the
# n<=10 sweep already delivers a monotone trend in both diagnostics.
python -m game_experiments.run_nplayer \
    --ns 3,4,5,6,7,8,10 \
    --n-seeds 32 --n-rounds 400 --n-samples-per-round 64 \
    --regimes symmetric,heterogeneous \
    --grid full \
    --brute-force-max-n 5 \
    --out-tag H3_nplayer
