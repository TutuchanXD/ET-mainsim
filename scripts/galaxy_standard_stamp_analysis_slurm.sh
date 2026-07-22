#!/usr/bin/env bash
# Publish one formal Galaxy injected/60 s standard analysis after its 90 daily
# delivery shards have completed.  The maintained Python CLI owns all
# readiness, identity, SHA-256, schema, and atomic-publication checks.
#
# Submit only after a source-specific afterok dependency has been constructed:
#   ET_STAMP_MANIFEST=/cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
#   ET_STAMP_CODE_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/ET-mainsim-v3 \
#   ET_STAMP_PHOTSIM_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/Photsim7 \
#   ET_STAMP_ET_COORD_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/et_focalplane \
#   ET_STAMP_DATA_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/Photsim7-data \
#   ET_STAMP_ANALYSIS_SOURCE_ID=<Gaia-source-id> \
#   sbatch --dependency=afterok:<all-90-target-day-job-ids> \
#     scripts/galaxy_standard_stamp_analysis_slurm.sh
#
# This launcher intentionally has no replacement path: published analysis
# directories are immutable and an incomplete source must be diagnosed rather
# than silently replaced.

#SBATCH --job-name=et_galaxy_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

: "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"
: "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"
: "${ET_STAMP_PHOTSIM_ROOT:?Set ET_STAMP_PHOTSIM_ROOT to the deployed Photsim7 checkout}"
: "${ET_STAMP_ET_COORD_ROOT:?Set ET_STAMP_ET_COORD_ROOT to the deployed et_focalplane checkout}"
: "${ET_STAMP_ANALYSIS_SOURCE_ID:?Set ET_STAMP_ANALYSIS_SOURCE_ID to one source ID}"

ET_STAMP_DATA_ROOT="${ET_STAMP_DATA_ROOT:-/cluster/home/cxgao/ET/Photsim7-data}"
ET_STAMP_ANALYSIS_BATCH_FRAMES="${ET_STAMP_ANALYSIS_BATCH_FRAMES:-256}"

if [[ ! -f "${ET_STAMP_MANIFEST}" ]]; then
  echo "ET_STAMP_MANIFEST is not a regular file: ${ET_STAMP_MANIFEST}" >&2
  exit 2
fi
if [[ ! "${ET_STAMP_ANALYSIS_SOURCE_ID}" =~ ^[0-9]+$ ]]; then
  echo "ET_STAMP_ANALYSIS_SOURCE_ID must be a non-negative decimal integer" >&2
  exit 2
fi

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
OUTPUT_DIR="${RUN_ROOT}/analysis/source_${ET_STAMP_ANALYSIS_SOURCE_ID}/injected/coadd_60s"

# shellcheck disable=SC1091
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_DATA_DIR="${ET_STAMP_DATA_ROOT}"
export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${ET_STAMP_PHOTSIM_ROOT}:${ET_STAMP_ET_COORD_ROOT}/src:${PYTHONPATH:-}"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "source_id=${ET_STAMP_ANALYSIS_SOURCE_ID} case=injected cadence_seconds=60"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "output_dir=${OUTPUT_DIR}"
echo "ET_DATA_DIR=${ET_DATA_DIR}"
date

exec python -m et_mainsim.standard_stamp_analysis \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${ET_STAMP_ANALYSIS_SOURCE_ID}" \
  --case injected \
  --cadence-seconds 60 \
  --output-dir "${OUTPUT_DIR}" \
  --batch-frames "${ET_STAMP_ANALYSIS_BATCH_FRAMES}"
