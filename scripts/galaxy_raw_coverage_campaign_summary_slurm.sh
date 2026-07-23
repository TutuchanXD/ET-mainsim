#!/usr/bin/env bash
# Publish one all-source raw-10-s coverage-aware Galaxy summary after every
# source task in the coverage array has succeeded.  Submit explicitly with:
#
#   ET_STAMP_COVERAGE_POLICY_JSON=<RUN_ROOT>/analysis/raw_10s_coverage_v2_policy.json \
#   sbatch --dependency=afterok:<coverage-array-job-id> \
#     scripts/galaxy_raw_coverage_campaign_summary_slurm.sh
#
# The Python entry point independently re-audits the complete ten-source
# contract.  The Slurm dependency is scheduling convenience, not provenance.

#SBATCH --job-name=et_galaxy_raw_cov_summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

: "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"
: "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"
: "${ET_STAMP_COVERAGE_POLICY_JSON:?Set ET_STAMP_COVERAGE_POLICY_JSON to the immutable frozen policy JSON}"

if [[ ! -f "${ET_STAMP_MANIFEST}" ]]; then
  echo "ET_STAMP_MANIFEST is not a regular file: ${ET_STAMP_MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${ET_STAMP_COVERAGE_POLICY_JSON}" ]]; then
  echo "ET_STAMP_COVERAGE_POLICY_JSON is not a regular file: ${ET_STAMP_COVERAGE_POLICY_JSON}" >&2
  exit 2
fi

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
ET_STAMP_CAMPAIGN_QC_JSON="${ET_STAMP_CAMPAIGN_QC_JSON:-${RUN_ROOT}/quality_control/injected_campaign_delivery_qc.json}"
OUTPUT_DIR="${RUN_ROOT}/analysis/campaign/injected/raw_10s_coverage_v2_summary"

if [[ ! -f "${ET_STAMP_CAMPAIGN_QC_JSON}" ]]; then
  echo "ET_STAMP_CAMPAIGN_QC_JSON is not a regular file: ${ET_STAMP_CAMPAIGN_QC_JSON}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${PYTHONPATH:-}"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) case=injected"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "campaign_qc=${ET_STAMP_CAMPAIGN_QC_JSON}"
echo "coverage_policy=${ET_STAMP_COVERAGE_POLICY_JSON}"
echo "output_dir=${OUTPUT_DIR}"
date

exec python -m et_mainsim.raw_coverage_campaign_summary \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --campaign-qc "${ET_STAMP_CAMPAIGN_QC_JSON}" \
  --coverage-policy "${ET_STAMP_COVERAGE_POLICY_JSON}" \
  --output-dir "${OUTPUT_DIR}"
