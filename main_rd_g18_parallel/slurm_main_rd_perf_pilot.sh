#!/usr/bin/env bash
#SBATCH --job-name=mainrd_perf_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=72
#SBATCH --mem=640G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_ROOT=/cluster/home/cxgao/ET
export PHOTSIM7_ROOT=/cluster/home/cxgao/ET/Photsim7-mainrd-perf
export PHOTSIM7_DATA_DIR=/cluster/home/cxgao/ET/Photsim7-data
export ET_DATA_DIR="${PHOTSIM7_DATA_DIR}"
export ET_FOCALPLANE_ROOT=/cluster/home/cxgao/ET/et_focalplane
export PYTHONPATH="/cluster/home/cxgao/ET/ET-mainsim-mainrd-perf:/cluster/home/cxgao/ET/ET-mainsim-mainrd-perf/main_rd_g18_parallel:/cluster/home/cxgao/ET/Photsim7-mainrd-perf:/cluster/home/cxgao/ET/et_focalplane/src:${PYTHONPATH:-}"

SCRIPT=/cluster/home/cxgao/ET/ET-mainsim-mainrd-perf/main_rd_g18_parallel/simulate_main_rd_8900x9120_g18.py
EVALUATOR=/cluster/home/cxgao/ET/ET-mainsim-mainrd-perf/main_rd_g18_parallel/evaluate_main_rd_benchmark.py
CACHE_ROOT=/cluster/home/cxgao/sshfs-share/main_rd_perf_8900x9120
OUTPUT_ROOT=/cluster/home/cxgao/sshfs-share/main_rd_perf_8900x9120_pilot_20260701_172817
PSD_PATH=/cluster/home/cxgao/ET/Photsim7-data/pds/ET_psd3-2.pkl
FRAMES=48

mkdir -p /cluster/home/cxgao/sshfs-share/slurm_logs "${OUTPUT_ROOT}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

for MAG in 16 16.5 17 17.5 18; do
  TAG=${MAG//./p}
  STAR_CACHE="${CACHE_ROOT}/main_rd_8900x9120_g_lt_${TAG}/cache/stars_main_rd_8900x9120_g_lt_${TAG}.npz"
  echo "== main_rd pilot G<${MAG} cache=${STAR_CACHE} =="
  test -f "${STAR_CACHE}"
  python "${SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --star-cache "${STAR_CACHE}" \
    --frames "${FRAMES}" \
    --mag-limit "${MAG}" \
    --gpus 0,1,2 \
    --workers-per-gpu 8 \
    --preview-count 0 \
    --psd-motion-path "${PSD_PATH}"
  python "${EVALUATOR}" "${OUTPUT_ROOT}" --output-prefix "${OUTPUT_ROOT}/benchmark_after_g_${TAG}"
done

python "${EVALUATOR}" "${OUTPUT_ROOT}" --output-prefix "${OUTPUT_ROOT}/benchmark_final"

date
