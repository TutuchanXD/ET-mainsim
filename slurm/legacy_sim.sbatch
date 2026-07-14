#!/usr/bin/env bash
#SBATCH --job-name=et_legacy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-etbase-clu}"

export ET_DATA_DIR="${ET_DATA_DIR:-/cluster/home/cxgao/ET/Photsim7-data}"
export RESULTS_ROOT="${RESULTS_ROOT:-/cluster/home/cxgao/sshfs-share/et-mainsim}"
ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-/cluster/home/cxgao/ET/ET-mainsim}"
export PYTHONPATH="${ET_MAINSIM_ROOT}/src:${PYTHONPATH:-}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID}}"
cd "${ET_MAINSIM_ROOT}"

args=(
  python -m et_mainsim run legacy-sim
  --preset "${PRESET:-full-effects-production}"
  --run-id "${RUN_ID:-legacy-full-effects-${SLURM_JOB_ID}}"
  --device cuda
  --gpus "${GPU_IDS:-0}"
  --ray-actor-count "${RAY_ACTOR_COUNT:-1}"
  --ray-num-cpus "${RAY_NUM_CPUS:-1}"
  --ray-num-gpus "${RAY_NUM_GPUS:-1}"
)

if [[ -n "${FRAMES:-}" ]]; then
  args+=(--frames "${FRAMES}")
fi
if [[ -n "${RUN_COUNT:-}" ]]; then
  args+=(--run-count "${RUN_COUNT}")
fi
if [[ -n "${STARS_PER_RUN:-}" ]]; then
  args+=(--stars-per-run "${STARS_PER_RUN}")
fi

"${args[@]}"
