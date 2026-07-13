#!/usr/bin/env bash
#SBATCH --job-name=et_full_frame
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-etbase-clu}"

export ET_DATA_DIR="${ET_DATA_DIR:-/cluster/home/cxgao/ET/Photsim7-data}"
export ET_FOCALPLANE_ROOT="${ET_FOCALPLANE_ROOT:-/cluster/home/cxgao/ET/et_focalplane}"
export RESULTS_ROOT="${RESULTS_ROOT:-/cluster/home/cxgao/sshfs-share/et-mainsim}"
: "${GAIA_CATALOG_DIR:?Set GAIA_CATALOG_DIR to the cluster Gaia catalog directory}"

ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-/cluster/home/cxgao/ET/ET-mainsim}"
PRESET="${PRESET:-production}"
RUN_ID="${RUN_ID:-et-full-frame-${SLURM_JOB_ID}}"
GPU_IDS="${GPU_IDS:-0,1}"
WORKERS_PER_DEVICE="${WORKERS_PER_DEVICE:-1}"

cd "${ET_MAINSIM_ROOT}"
python -c "import et_mainsim, photsim7; print('et_mainsim', et_mainsim.__version__, 'photsim7', photsim7.__file__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

args=(
  python -m et_mainsim run et-full-frame
  --preset "${PRESET}"
  --run-id "${RUN_ID}"
  --backend local-subprocess
  --device cuda
  --gpus "${GPU_IDS}"
  --workers-per-device "${WORKERS_PER_DEVICE}"
)

if [[ -n "${FRAMES:-}" ]]; then
  args+=(--frames "${FRAMES}")
fi
if [[ -n "${FRAME_INDICES:-}" ]]; then
  args+=(--frame-indices "${FRAME_INDICES}")
fi
if [[ -n "${TARGET_EPOCH_JYEAR:-}" ]]; then
  args+=(--target-epoch-jyear "${TARGET_EPOCH_JYEAR}")
fi

"${args[@]}"
