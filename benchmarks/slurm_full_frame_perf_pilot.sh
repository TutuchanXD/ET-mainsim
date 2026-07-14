#!/usr/bin/env bash
#SBATCH --job-name=et_ff_perf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-etbase-clu}"

ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-/cluster/home/cxgao/ET/ET-mainsim}"
export ET_MAINSIM_ROOT
export PYTHONPATH="${ET_MAINSIM_ROOT}/src:${ET_MAINSIM_ROOT}:${PYTHONPATH:-}"
export ET_DATA_DIR="${ET_DATA_DIR:-/cluster/home/cxgao/ET/Photsim7-data}"
export ET_FOCALPLANE_ROOT="${ET_FOCALPLANE_ROOT:-/cluster/home/cxgao/ET/et_focalplane}"
: "${GAIA_CATALOG_DIR:?Set GAIA_CATALOG_DIR}"

CACHE_ROOT="${CACHE_ROOT:-/cluster/home/cxgao/sshfs-share/slurm_validation/et-mainsim-final-canonical-caches}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/home/cxgao/sshfs-share/benchmarks/et_ff_perf_${SLURM_JOB_ID}}"
FRAMES="${FRAMES:-1}"
mkdir -p "${OUTPUT_ROOT}"
cd "${ET_MAINSIM_ROOT}"

for MAG in 16 16.5 17 17.5 18; do
  TAG=${MAG//./p}
  CACHE="${CACHE_ROOT}/g_lt_${TAG}/stars.npz"
  test -f "${CACHE}"
  for GPU in 0 1 2; do
    CUDA_VISIBLE_DEVICES="${GPU}" python -m benchmarks.run_full_frame_thermal_load \
      --output-root "${OUTPUT_ROOT}" \
      --run-id "g${TAG}-gpu${GPU}" \
      --catalog-cache "${CACHE}" \
      --frames "${FRAMES}" \
      --mag-limit "${MAG}" \
      --jitter-models 100 \
      --max-stars 100000 \
      --seed "$((20260714 + GPU))" \
      --gpu 0 &
  done
  wait
  python benchmarks/evaluate_main_rd_benchmark.py \
    "${OUTPUT_ROOT}" \
    --output-prefix "${OUTPUT_ROOT}/benchmark_g${TAG}"
done
