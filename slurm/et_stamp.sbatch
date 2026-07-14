#!/usr/bin/env bash
#SBATCH --job-name=et_stamp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-etbase-clu}"

export ET_DATA_DIR="${ET_DATA_DIR:-/cluster/home/cxgao/ET/Photsim7-data}"
export ET_FOCALPLANE_ROOT="${ET_FOCALPLANE_ROOT:-/cluster/home/cxgao/ET/et_focalplane}"
export RESULTS_ROOT="${RESULTS_ROOT:-/cluster/home/cxgao/sshfs-share/et-mainsim}"
if [[ -z "${INPUT_TABLE:-}" ]]; then
  : "${GAIA_CATALOG_DIR:?Set GAIA_CATALOG_DIR for catalog input mode}"
fi
ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-/cluster/home/cxgao/ET/ET-mainsim}"
export PYTHONPATH="${ET_MAINSIM_ROOT}/src:${PYTHONPATH:-}"
cd "${ET_MAINSIM_ROOT}"

args=(
  python -m et_mainsim run et-stamp
  --preset "${PRESET:-production}"
  --run-id "${RUN_ID:-et-stamp-${SLURM_JOB_ID}}"
  --backend local-subprocess
  --device cuda
  --gpus "${GPU_IDS:-0}"
)

if [[ -n "${INPUT_TABLE:-}" ]]; then
  args+=(--input-table "${INPUT_TABLE}")
fi
if [[ -n "${FRAMES:-}" ]]; then
  args+=(--frames "${FRAMES}")
fi
if [[ -n "${TARGET_LIMIT:-}" ]]; then
  args+=(--target-limit "${TARGET_LIMIT}")
fi

"${args[@]}"
