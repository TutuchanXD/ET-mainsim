#!/usr/bin/env bash
#SBATCH --job-name=et-source-var-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

for name in \
  ET_MAINSIM_WORKTREE \
  PHOTSIM7_WORKTREE \
  PYTHON_BIN \
  SN_CSV \
  OUTPUT_ROOT \
  ET_DATA_DIR \
  ET_FOCALPLANE_ROOT
do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: ${name}" >&2
    exit 64
  fi
done

assigned_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
if [[ -z "${assigned_gpu}" ]]; then
  echo "Slurm did not assign a visible GPU" >&2
  exit 69
fi

# The shared H100 node can contain non-Slurm CUDA processes. Wait until the
# GPU selected by Slurm has no existing compute context instead of colliding
# with another user's run.
poll_seconds="${GPU_IDLE_POLL_SECONDS:-30}"
timeout_seconds="${GPU_IDLE_TIMEOUT_SECONDS:-21600}"
wait_started="${SECONDS}"
while nvidia-smi \
  --id="${assigned_gpu}" \
  --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk 'NF { found=1 } END { exit !found }'
do
  if (( SECONDS - wait_started >= timeout_seconds )); then
    echo "Assigned GPU ${assigned_gpu} stayed busy for ${timeout_seconds}s" >&2
    exit 75
  fi
  echo "Assigned GPU ${assigned_gpu} is busy; waiting ${poll_seconds}s"
  sleep "${poll_seconds}"
done

export ET_DATA_DIR
export ET_FOCALPLANE_ROOT
export MPLBACKEND=Agg
export PYTHONPATH="${ET_MAINSIM_WORKTREE}/src:${PHOTSIM7_WORKTREE}${PYTHONPATH:+:${PYTHONPATH}}"

run_id="${RUN_ID:-source-variability-sn-validation}"
seed="${RUN_SEED:-20260715}"
job_output_root="${OUTPUT_ROOT%/}/${SLURM_JOB_ID}"
mkdir -p "${job_output_root}"

cd "${ET_MAINSIM_WORKTREE}"
"${PYTHON_BIN}" scripts/validate_source_variability_stamp.py \
  --sn-csv "${SN_CSV}" \
  --output-root "${job_output_root}" \
  --data-root "${ET_DATA_DIR}" \
  --focalplane-registry "${ET_FOCALPLANE_ROOT}" \
  --frame-count 22 \
  --coadd-size 2 \
  --run-id "${run_id}" \
  --seed "${seed}" \
  --backend local-subprocess \
  --device cuda \
  --gpus "${assigned_gpu}" \
  --workers-per-device 1

echo "Validation products: ${job_output_root}/validation_products"
