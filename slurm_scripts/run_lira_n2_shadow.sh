#!/bin/bash
# LiRA H1 shadow-model sweep for the n=2 Privacy Externality Game (T24).
#
# For each of the 25 action profiles on the 5x5 grid eps in {2,4,8,16,32}
# (Player 1 x Player 2), train 128 DP-SGD shadow models on Adult
# (drop-in for Credit Card Default; the shared-loader alias for the latter
# is OpenML `credit-g` at 1000 rows, too small for a 5000-per-silo pool)
# with per-silo Poisson sampling and score membership via LiRA.
#
# CLAUDE.md Rule 3: GPU only because DP-SGD trains an NN; smallest that fits;
# CPU/mem/time tight to the tabular workload, not the queue maximum.

#SBATCH --job-name=lira_H1_n2
#SBATCH --output=./out/lira_H1_n2-%A_%a.out
#SBATCH --error=./out/lira_H1_n2-%A_%a.err
#SBATCH --time=0-04:00:00           # ~1 GPU-h realised (128 tabular MLPs); 4h ceiling.
#SBATCH --gres=gpu:1                # Rule 3: single smallest GPU; not a full node.
#SBATCH --cpus-per-task=8           # Opacus per-sample-grad workers + data loading.
#SBATCH --mem=32G                   # Adult tabular is <2 GB; 32 GB covers Opacus per-sample-grad.
#SBATCH --array=0-24                # 25 profiles on the 5x5 grid.

set -euo pipefail

REPO=/vol/miltank/users/kaiserj/Collusion_Gametheory
cd "${REPO}"
export PYTHONUNBUFFERED=true
source ./.venv/bin/activate

PROFILE_ID=${SLURM_ARRAY_TASK_ID}
CACHE=${REPO}/game_experiments/_cache_H1
GRID=${CACHE}/mechanism_grid.pkl

if [[ ! -f "${GRID}" ]]; then
    echo "ERROR: mechanism grid not found at ${GRID}."
    echo "Run first (login-node CPU, <1 min):"
    echo "  python ${REPO}/tasks_findings/_scripts/24_lira_analysis.py --build-grid"
    exit 1
fi

# Extract (sigma, p1, p2) for this profile from the pre-computed cache.
read -r SIGMA P1 P2 <<<"$(./.venv/bin/python - <<PY
import pickle, sys
with open("${GRID}", "rb") as f:
    grid = pickle.load(f)
row = grid[${PROFILE_ID}]
assert row["feasible"], f"Profile ${PROFILE_ID} not feasible: {row}"
print(row["sigma"], row["p1"], row["p2"])
PY
)"

echo "Profile ${PROFILE_ID}: sigma=${SIGMA} p1=${P1} p2=${P2}"

SAVEDIR=${CACHE}/n2_adult/${PROFILE_ID}
mkdir -p "${SAVEDIR}"

./.venv/bin/python ${REPO}/experiment_scripts/mia/04_mia_pergroup.py \
    --exp_yaml ${REPO}/experiment_scripts/mia/exp_yaml/adult_H1_n2.yaml \
    --pg_sigma "${SIGMA}" \
    --pg_sample_rates "${P1},${P2}" \
    --group_sizes 5000,5000 \
    --silo_split modulo2 \
    --seed 42 \
    --profile_idx "${PROFILE_ID}" \
    --savedir_result "${SAVEDIR}"

# Post-training: fit per-silo LiRA Gaussians, dump A_measured.npy and V_measured.npy.
./.venv/bin/python ${REPO}/tasks_findings/_scripts/24_lira_analysis.py \
    --score-profile "${PROFILE_ID}" \
    --lira-variant offline-fixed-variance
