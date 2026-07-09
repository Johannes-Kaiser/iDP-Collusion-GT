#!/bin/bash
# T25 -- toy federated n=3 shadow-model sweep for the PEG.
#
# Adapts the T24 n=2 LiRA harness (slurm_scripts/run_lira_n2_shadow.sh) to
# three silos on Credit Card Default with a coarse action grid
# eps in {2, 8, 32} per silo (3^3 = 27 joint profiles).
#
# Federated setup (see tasks_findings/25_toy_federated_n3_shadow_model.md
# Section 2 for the design rationale):
#   * 3 silos of ~10,000 rows each; IID split by row-index modulo 3.
#     A non-IID variant is available via --silo_split label_dirichlet.
#   * Per-silo DP-SGD (item-level (eps_i, delta)-iDP), joint model updated
#     via synchronous FedSGD server-side aggregation (McMahan et al. 2017;
#     the DP-FedAvg-with-per-silo-item-level-DP setup of McMahan et al.
#     2018, "Learning Differentially Private Recurrent Language Models").
#   * Per-silo (sigma_i, p_i) chosen so that each silo's accountant returns
#     exactly its target eps_i under a shared step count -- consistent with
#     game.mechanism.solve_mechanism.
#
# CLAUDE.md Rule 3: GPU because DP-SGD trains a neural network; single
# smallest GPU; tight time/mem/cpu; NOT launched by an agent.
#
# CLAUDE.md Rule 0: this Slurm script is authored fresh for the n=3 setup;
# it replaces (not extends) the T24 n=2 script. No "v2" / "v3" suffixes.

#SBATCH --job-name=lira_n3_fed
#SBATCH --output=./out/lira_n3_fed-%A_%a.out
#SBATCH --error=./out/lira_n3_fed-%A_%a.err
#SBATCH --time=0-08:00:00           # ~5 GPU-h realised (32 fed shadows @ ~10 min); 8h ceiling.
#SBATCH --gres=gpu:1                # Rule 3: single smallest GPU.
#SBATCH --cpus-per-task=8           # Opacus per-sample grads + 3 silo dataloaders.
#SBATCH --mem=32G                   # Credit-Card Default fits in <2 GB; 32 GB covers Opacus.
#SBATCH --array=0-26                # 27 profiles on the 3-per-silo grid (3^3).

set -euo pipefail

export PYTHONUNBUFFERED=true
source ./.venv/bin/activate

PROFILE_ID=${SLURM_ARRAY_TASK_ID}
REPO=/vol/miltank/users/kaiserj/Collusion_Gametheory
CACHE=${REPO}/game_experiments/_cache_T25
GRID=${CACHE}/mechanism_grid.pkl

if [[ ! -f "${GRID}" ]]; then
    echo "ERROR: mechanism grid not found at ${GRID}."
    echo "Run first (login-node CPU, <1 min):"
    echo "  python ${REPO}/tasks_findings/_scripts/25_toy_federated_analysis.py --build-grid"
    exit 1
fi

# Extract (sigma, p1, p2, p3) for this profile from the pre-computed cache.
read -r SIGMA P1 P2 P3 <<<"$(./.venv/bin/python - <<PY
import pickle
with open("${GRID}", "rb") as f:
    grid = pickle.load(f)
row = grid[${PROFILE_ID}]
assert row["feasible"], f"Profile ${PROFILE_ID} not feasible: {row}"
print(row["sigma"], row["p1"], row["p2"], row["p3"])
PY
)"

echo "Profile ${PROFILE_ID}: sigma=${SIGMA} p=(${P1}, ${P2}, ${P3})"

SAVEDIR=${CACHE}/n3_credit_card/${PROFILE_ID}
mkdir -p "${SAVEDIR}"

# 04_mia_pergroup.py (from T24) already accepts per-group (sigma, p_g); T25
# passes three groups instead of two and sets --silo_split modulo3 for the
# IID row-modulo partition. The FedSGD server aggregation is enabled with
# --federated fedsgd (added to 04_mia_pergroup.py as part of T25 -- see
# tasks_findings/25_toy_federated_n3_shadow_model.md Section 3).
./.venv/bin/python ${REPO}/experiment_scripts/mia/04_mia_pergroup.py \
    --exp_yaml ${REPO}/experiment_scripts/mia/exp_yaml/credit_card_default_n3_toy.yaml \
    --pg_sigma "${SIGMA}" \
    --pg_sample_rates "${P1},${P2},${P3}" \
    --group_sizes 10000,10000,10000 \
    --silo_split modulo3 \
    --federated fedsgd \
    --seed 42 \
    --savedir_result "${SAVEDIR}"

# Post-training: fit per-silo LiRA Gaussians, dump A_measured.npy and V_measured.npy.
./.venv/bin/python ${REPO}/tasks_findings/_scripts/25_toy_federated_analysis.py \
    --score-profile "${PROFILE_ID}" \
    --lira-variant offline-fixed-variance
