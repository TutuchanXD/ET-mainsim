#!/usr/bin/env bash
# Fail-closed campaign completion gate for a formal Galaxy stamp delivery.
#
# Submit this job with afterok on the complete render array, then make the
# standard-analysis array depend on this job rather than directly on rendering:
#
#   ET_STAMP_MANIFEST=/cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
#   ET_STAMP_CODE_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/ET-mainsim-analysis-v3 \
#   sbatch --dependency=afterok:<render-array-job-id> \
#     scripts/galaxy_campaign_delivery_qc_slurm.sh

#SBATCH --job-name=et_galaxy_campaign_qc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

: "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"
: "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"

if [[ ! -f "${ET_STAMP_MANIFEST}" ]]; then
  echo "ET_STAMP_MANIFEST is not a regular file: ${ET_STAMP_MANIFEST}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${PYTHONPATH:-}"

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
OUTPUT_JSON="${RUN_ROOT}/quality_control/injected_campaign_delivery_qc.json"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "output_json=${OUTPUT_JSON}"
date

exec python "${ET_STAMP_CODE_ROOT}/scripts/audit_galaxy_campaign_delivery.py" \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --case injected \
  --output-json "${OUTPUT_JSON}" \
  --require-complete
