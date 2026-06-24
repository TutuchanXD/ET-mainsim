#!/usr/bin/env bash
#SBATCH --job-name=main_rd_g17_360cont
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_ROOT=/cluster/home/cxgao/ET
export ET_MAINSIM_ROOT=/cluster/home/cxgao/ET/ET-mainsim
export PHOTSIM7_ROOT="${PHOTSIM7_ROOT:-/cluster/home/cxgao/ET/Photosim7}"
if [[ ! -d "${PHOTSIM7_ROOT}/photsim7" && -d /cluster/home/cxgao/ET/Photsim7/photsim7 ]]; then
  export PHOTSIM7_ROOT=/cluster/home/cxgao/ET/Photsim7
fi
export PHOTSIM7_DATA_DIR=/cluster/home/cxgao/ET/Photsim7-data
export ET_DATA_DIR="${PHOTSIM7_DATA_DIR}"
export ET_FOCALPLANE_ROOT=/cluster/home/cxgao/ET/et_focalplane
export RESULTS_ROOT=/cluster/home/cxgao/sshfs-share/GRB/full_sim
export PYTHONPATH="${PHOTSIM7_ROOT}:${ET_FOCALPLANE_ROOT}/src:${ET_MAINSIM_ROOT}/main_rd_g18_parallel:${PYTHONPATH:-}"

RUN_LABEL=main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s
RUN_DIR="${RESULTS_ROOT}/${RUN_LABEL}"
STAR_CACHE="${STAR_CACHE:-${RUN_DIR}/cache/stars_${RUN_LABEL}.npz}"
FRAME_INDICES="$(seq -s, 120 359)"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing RUN_DIR=${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${STAR_CACHE}" ]]; then
  echo "Missing STAR_CACHE=${STAR_CACHE}" >&2
  echo "Use the existing 120-frame star cache or regenerate it before continuing." >&2
  exit 2
fi
if [[ -f "${RUN_DIR}/frames/frame_000120.npy" ]]; then
  echo "Refusing to start because frame_000120.npy already exists in ${RUN_DIR}." >&2
  echo "Inspect the directory and rerun manually if this is an intentional resume." >&2
  exit 3
fi

BACKUP_DIR="${RUN_DIR}/metadata_before_extend360_${SLURM_JOB_ID:-manual}_$(date +%Y%m%dT%H%M%S)"
mkdir -p "${BACKUP_DIR}"
cp -a "${RUN_DIR}"/run_config.json "${BACKUP_DIR}/" 2>/dev/null || true
cp -a "${RUN_DIR}"/effects_timeseries.npz "${BACKUP_DIR}/" 2>/dev/null || true
cp -a "${RUN_DIR}"/worker_*.json "${BACKUP_DIR}/" 2>/dev/null || true
if [[ -d "${RUN_DIR}/logs" ]]; then
  cp -a "${RUN_DIR}/logs" "${BACKUP_DIR}/logs"
fi
echo "metadata_backup=${BACKUP_DIR}"

python "${ET_MAINSIM_ROOT}/main_rd_grb/simulate_main_rd_full_10s_g17_extend360.py" \
  --frames 360 \
  --frame-indices "${FRAME_INDICES}" \
  --gpus 0,1,2 \
  --workers-per-gpu 4 \
  --output-root "${RESULTS_ROOT}" \
  --star-cache "${STAR_CACHE}" \
  --psd-motion-path "${PHOTSIM7_DATA_DIR}/pds/ET_psd3-2.pkl" \
  --preview-count 0

date
