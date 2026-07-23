#!/usr/bin/env bash
# Read every final Galaxy delivery HDF5 header after campaign QC has passed.
#
# Submit after campaign QC and make downstream source analysis depend on this
# audit as well, for example:
#   ET_STAMP_MANIFEST=/cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
#   ET_STAMP_CODE_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/ET-mainsim-analysis-v7 \
#   sbatch --dependency=afterok:<campaign-qc-job-id> \
#     scripts/galaxy_delivery_provenance_audit_slurm.sh

#SBATCH --job-name=et_galaxy_provenance_audit
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

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
ET_STAMP_CAMPAIGN_QC_JSON="${ET_STAMP_CAMPAIGN_QC_JSON:-${RUN_ROOT}/quality_control/injected_campaign_delivery_qc.json}"
OUTPUT_JSON="${RUN_ROOT}/quality_control/injected_campaign_provenance_psf_audit.json"

if [[ ! -f "${ET_STAMP_CAMPAIGN_QC_JSON}" ]]; then
  echo "ET_STAMP_CAMPAIGN_QC_JSON is not a regular file: ${ET_STAMP_CAMPAIGN_QC_JSON}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${PYTHONPATH:-}"

python - "${ET_STAMP_MANIFEST}" "${ET_STAMP_CAMPAIGN_QC_JSON}" <<'PY'
import hashlib
import json
import os
import sys

manifest_path, qc_path = sys.argv[1:]
with open(qc_path, encoding="utf-8") as stream:
    qc = json.load(stream)
if qc.get("schema_id") != "et_mainsim.galaxy_campaign_delivery_qc.v1":
    raise SystemExit("campaign QC receipt has an unsupported schema")
if qc.get("ready") is not True:
    raise SystemExit("campaign QC receipt is not ready; do not audit an incomplete delivery")
if qc.get("case") != "injected":
    raise SystemExit("campaign QC receipt is not for injected Galaxy delivery")
identity = qc.get("manifest_identity")
if not isinstance(identity, dict):
    raise SystemExit("campaign QC receipt lacks manifest identity")
with open(manifest_path, "rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if identity.get("sha256") != digest:
    raise SystemExit("campaign QC receipt binds a different production manifest")
if identity.get("size_bytes") != os.path.getsize(manifest_path):
    raise SystemExit("campaign QC receipt has a different production-manifest size")
PY

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) case=injected"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "campaign_qc=${ET_STAMP_CAMPAIGN_QC_JSON}"
echo "output_json=${OUTPUT_JSON}"
date

exec python -m et_mainsim.galaxy_delivery_provenance_audit \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --output-json "${OUTPUT_JSON}" \
  --require-complete
