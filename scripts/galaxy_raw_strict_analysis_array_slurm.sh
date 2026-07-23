#!/usr/bin/env bash
# Publish the raw-10-s strict reference curve for every source in one completed
# formal Galaxy campaign.  This is the only producer of raw_10s_strict; the
# coverage-aware v2 reducer consumes its immutable CSV/manifest afterwards.
#
# Submit only after the complete campaign-QC job, for example:
#   ET_STAMP_MANIFEST=/cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
#   ET_STAMP_CODE_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/ET-mainsim-analysis-v4 \
#   ET_STAMP_PHOTSIM_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/Photsim7 \
#   ET_STAMP_ET_COORD_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/et_focalplane \
#   ET_STAMP_DATA_ROOT=/cluster/home/cxgao/ET/stamp-science-<date>/Photsim7-data \
#   sbatch --dependency=afterok:<campaign-qc-job-id> \
#     scripts/galaxy_raw_strict_analysis_array_slurm.sh
#
# The internal QC receipt check remains mandatory even when Slurm dependencies
# are accidentally omitted.  Published directories are immutable.

#SBATCH --job-name=et_galaxy_raw_strict
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%1
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%A_%a.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%A_%a.err

set -euo pipefail

: "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"
: "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"
: "${ET_STAMP_PHOTSIM_ROOT:?Set ET_STAMP_PHOTSIM_ROOT to the deployed Photsim7 checkout}"
: "${ET_STAMP_ET_COORD_ROOT:?Set ET_STAMP_ET_COORD_ROOT to the deployed et_focalplane checkout}"

ET_STAMP_DATA_ROOT="${ET_STAMP_DATA_ROOT:-/cluster/home/cxgao/ET/Photsim7-data}"
ET_STAMP_ANALYSIS_BATCH_FRAMES="${ET_STAMP_ANALYSIS_BATCH_FRAMES:-256}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as an array job}"

if [[ ! -f "${ET_STAMP_MANIFEST}" ]]; then
  echo "ET_STAMP_MANIFEST is not a regular file: ${ET_STAMP_MANIFEST}" >&2
  exit 2
fi

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
ET_STAMP_CAMPAIGN_QC_JSON="${ET_STAMP_CAMPAIGN_QC_JSON:-${RUN_ROOT}/quality_control/injected_campaign_delivery_qc.json}"

# shellcheck disable=SC1091
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_DATA_DIR="${ET_STAMP_DATA_ROOT}"
export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${ET_STAMP_PHOTSIM_ROOT}:${ET_STAMP_ET_COORD_ROOT}/src:${PYTHONPATH:-}"

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
    raise SystemExit("campaign QC receipt is not ready; do not analyse an incomplete delivery")
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

read -r SOURCE_ID TARGET_COUNT <<<"$(
  python - "${ET_STAMP_MANIFEST}" "${ARRAY_INDEX}" <<'PY'
import json
import sys

manifest_path, array_index_text = sys.argv[1:]
array_index = int(array_index_text)
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
if manifest.get("schema_id") != "et_mainsim.galaxy_stamp_production.v1":
    raise SystemExit("manifest is not a formal Galaxy production")
targets = manifest.get("targets")
if not isinstance(targets, list):
    raise SystemExit("manifest targets must be a list")
if len(targets) != 10:
    raise SystemExit(
        f"formal Galaxy raw analysis requires exactly 10 targets, got {len(targets)}"
    )
if array_index < 0 or array_index >= len(targets):
    raise SystemExit(f"array index {array_index} is outside [0, {len(targets)})")
source_id = int(targets[array_index]["source_id_int64"])
print(source_id, len(targets))
PY
)"

OUTPUT_DIR="${RUN_ROOT}/analysis/source_${SOURCE_ID}/injected/raw_10s_strict"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "array_index=${ARRAY_INDEX} source_id=${SOURCE_ID} target_count=${TARGET_COUNT} case=injected cadence_seconds=10"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "campaign_qc=${ET_STAMP_CAMPAIGN_QC_JSON}"
echo "output_dir=${OUTPUT_DIR}"
echo "ET_DATA_DIR=${ET_DATA_DIR}"
date

exec python -m et_mainsim.standard_stamp_analysis \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${SOURCE_ID}" \
  --case injected \
  --cadence-seconds 10 \
  --output-dir "${OUTPUT_DIR}" \
  --batch-frames "${ET_STAMP_ANALYSIS_BATCH_FRAMES}"
