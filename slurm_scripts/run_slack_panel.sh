#!/bin/bash

#SBATCH --job-name=peg_slack_panel
#SBATCH --output=./out/peg_slack_panel-%A.out
#SBATCH --error=./out/peg_slack_panel-%A.err
#SBATCH --time=00:15:00              # cache-hit only; runs in <30s
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

export PYTHONUNBUFFERED=true

source ./.venv/bin/activate

# Loads the K=200 mechanism-grid cache produced by run_locus_K200.sh and
# emits paper/figures/fig_slack_panel.{pdf,svg} plus the reproducibility
# pickle game_experiments/_cache_v2/slack_panel.pkl. No mechanism
# recomputation.
python tasks_findings/_scripts/11_slack_panel.py
