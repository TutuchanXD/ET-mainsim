#!/usr/bin/env bash
# Render one formal Galaxy target x time shard on node-local scratch, then copy
# its complete raw/coadd HDF5 set to shared storage and publish it atomically.
#
# This launcher is deliberately separate from the direct shared-filesystem
# launcher.  It must be used only by a scheduler plan that owns each
# (source_id, shard_id) exactly once; never mix it with a direct array for the
# same formal run.

#SBATCH --job-name=et_galaxy_stamp_stage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%A_%a.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%A_%a.err

set -euo pipefail

: "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"
: "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"
: "${ET_STAMP_PHOTSIM_ROOT:?Set ET_STAMP_PHOTSIM_ROOT to the deployed Photsim7 checkout}"
: "${ET_STAMP_ET_COORD_ROOT:?Set ET_STAMP_ET_COORD_ROOT to the deployed et_focalplane checkout}"

ET_STAMP_DATA_ROOT="${ET_STAMP_DATA_ROOT:-/cluster/home/cxgao/ET/Photsim7-data}"
ET_STAMP_FOCALPLANE_REGISTRY="${ET_STAMP_FOCALPLANE_REGISTRY:-${ET_STAMP_ET_COORD_ROOT}/data}"
ET_STAMP_CASE="${ET_STAMP_CASE:-injected}"
ET_STAMP_BATCH_SIZE="${ET_STAMP_BATCH_SIZE:-64}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as an array job}"

if [[ "${ET_STAMP_CASE}" != "injected" && "${ET_STAMP_CASE}" != "static" ]]; then
  echo "ET_STAMP_CASE must be injected or static, got ${ET_STAMP_CASE}" >&2
  exit 2
fi

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_DATA_DIR="${ET_STAMP_DATA_ROOT}"
export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${ET_STAMP_PHOTSIM_ROOT}:${ET_STAMP_ET_COORD_ROOT}/src:${PYTHONPATH:-}"

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
SCRATCH_BASE="${ET_STAMP_LOCAL_SCRATCH_ROOT:-${TMPDIR:-/tmp}/et_stamp_scratch}"
MIN_SCRATCH_KB="${ET_STAMP_MIN_SCRATCH_KB:-31457280}"
mkdir -p "${SCRATCH_BASE}"
AVAILABLE_SCRATCH_KB="$(df -Pk "${SCRATCH_BASE}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_SCRATCH_KB}" =~ ^[0-9]+$ ]] || [[ "${AVAILABLE_SCRATCH_KB}" -lt "${MIN_SCRATCH_KB}" ]]; then
  echo "node-local scratch below required capacity: available_kb=${AVAILABLE_SCRATCH_KB:-unknown} required_kb=${MIN_SCRATCH_KB}" >&2
  exit 2
fi

JOB_SCRATCH="$(mktemp -d "${SCRATCH_BASE%/}/galaxy-stamp-${SLURM_JOB_ID:-manual}-${ARRAY_INDEX}-XXXXXX")"
cleanup() {
  if [[ -n "${JOB_SCRATCH:-}" && -d "${JOB_SCRATCH}" ]]; then
    rm -rf -- "${JOB_SCRATCH}"
  fi
}
trap cleanup EXIT
LOCAL_CASE_ROOT="${JOB_SCRATCH}/cases/${ET_STAMP_CASE}"

read -r SOURCE_ID SHARD_ID TARGET_COUNT SHARD_COUNT <<<"$(
  python - "${ET_STAMP_MANIFEST}" "${ARRAY_INDEX}" <<'PY'
import json
import sys

manifest_path, array_index_text = sys.argv[1:]
array_index = int(array_index_text)
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
targets = manifest.get("targets")
delivery = manifest.get("delivery")
if not isinstance(delivery, dict):
    raise SystemExit("manifest lacks delivery object")
if delivery.get("execution_mode") != "staged_local_scratch_v1":
    raise SystemExit(
        "staged launcher requires delivery.execution_mode='staged_local_scratch_v1'"
    )
time_plan_relative = delivery.get("time_plan_relative_path")
if not isinstance(targets, list) or not isinstance(time_plan_relative, str):
    raise SystemExit("manifest lacks formal target/time-plan fields")
from pathlib import Path
time_plan_path = Path(manifest_path).resolve().parent / time_plan_relative
with time_plan_path.open(encoding="utf-8") as stream:
    time_plan = json.load(stream)
shard_entries = time_plan.get("shards")
if not isinstance(shard_entries, list) or not shard_entries or not targets:
    raise SystemExit("manifest has no formal targets or time shards")
total = len(targets) * len(shard_entries)
if array_index < 0 or array_index >= total:
    raise SystemExit(f"array index {array_index} is outside [0, {total})")
target_index, shard_index = divmod(array_index, len(shard_entries))
source_id = int(targets[target_index]["source_id_int64"])
shard_id = int(shard_entries[shard_index]["shard_id"])
print(source_id, shard_id, len(targets), len(shard_entries))
PY
)"

printf -v SHARD_NAME "shard_%05d" "${SHARD_ID}"
FORMAL_DELIVERY_ROOT="${RUN_ROOT}/cases/${ET_STAMP_CASE}/stamps/target_${SOURCE_ID}/delivery"
FORMAL_SHARD_ROOT="${FORMAL_DELIVERY_ROOT}/${SHARD_NAME}"
residue_paths=()
for artifact in \
  "${FORMAL_SHARD_ROOT}" \
  "${FORMAL_DELIVERY_ROOT}/.${SHARD_NAME}.lock" \
  "${FORMAL_DELIVERY_ROOT}/.${SHARD_NAME}.staged-publish.lock"; do
  if [[ -e "${artifact}" || -L "${artifact}" ]]; then
    residue_paths+=("${artifact}")
  fi
done
shopt -s nullglob
for artifact in \
  "${FORMAL_DELIVERY_ROOT}/.${SHARD_NAME}".*.partial \
  "${FORMAL_DELIVERY_ROOT}/.${SHARD_NAME}".*.incoming; do
  residue_paths+=("${artifact}")
done
shopt -u nullglob
if (( ${#residue_paths[@]} > 0 )); then
  echo "formal shard residue prevents staged production:" >&2
  printf '  %s\n' "${residue_paths[@]}" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "array_index=${ARRAY_INDEX} source_id=${SOURCE_ID} shard_id=${SHARD_ID} target_count=${TARGET_COUNT} shard_count=${SHARD_COUNT} case=${ET_STAMP_CASE}"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "formal_case_root=${RUN_ROOT}/cases/${ET_STAMP_CASE}"
echo "local_case_root=${LOCAL_CASE_ROOT} available_scratch_kb=${AVAILABLE_SCRATCH_KB}"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

python "${ET_STAMP_CODE_ROOT}/scripts/run_galaxy_independent_stamp_production.py" run-target \
  --manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${SOURCE_ID}" \
  --case "${ET_STAMP_CASE}" \
  --shard-id "${SHARD_ID}" \
  --data-root "${ET_STAMP_DATA_ROOT}" \
  --focalplane-registry "${ET_STAMP_FOCALPLANE_REGISTRY}" \
  --device cuda \
  --batch-size "${ET_STAMP_BATCH_SIZE}" \
  --output-root "${LOCAL_CASE_ROOT}"

python -m et_mainsim.staged_stamp_delivery publish \
  --staged-case-root "${LOCAL_CASE_ROOT}" \
  --formal-case-root "${RUN_ROOT}/cases/${ET_STAMP_CASE}" \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${SOURCE_ID}" \
  --shard-id "${SHARD_ID}" \
  --case "${ET_STAMP_CASE}"

date
