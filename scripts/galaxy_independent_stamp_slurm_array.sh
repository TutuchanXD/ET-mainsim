#!/usr/bin/env bash
# Run exactly one target x globally aligned time shard of a prepared Galaxy run.
#
# Submit a one-day benchmark first, for example:
#   ET_STAMP_MANIFEST=/cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
#   sbatch --array=0-0 scripts/galaxy_independent_stamp_slurm_array.sh
#
# After the benchmark has passed its delivery/photometry checks, submit the
# formal injected array with at most the available GPU count in parallel:
#   sbatch --array=0-899%3 scripts/galaxy_independent_stamp_slurm_array.sh
#
# Array ordering is stable and recorded in stdout: target order in the frozen
# manifest, then ascending time-shard ID.  It deliberately never invokes a
# whole-target/no-shard run.

#SBATCH --job-name=et_galaxy_stamp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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

read -r SOURCE_ID SHARD_ID TARGET_COUNT SHARD_COUNT <<<"$(
  python - "${ET_STAMP_MANIFEST}" "${ARRAY_INDEX}" <<'PY'
import json
import sys

manifest_path, array_index_text = sys.argv[1:]
array_index = int(array_index_text)
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
targets = manifest.get("targets")
shards = manifest.get("delivery", {}).get("time_plan_identity")
time_plan_relative = manifest.get("delivery", {}).get("time_plan_relative_path")
if not isinstance(targets, list) or not isinstance(time_plan_relative, str):
    raise SystemExit("manifest lacks formal v2 target/time-plan fields")
from pathlib import Path
time_plan_path = Path(manifest_path).resolve().parent / time_plan_relative
with time_plan_path.open(encoding="utf-8") as stream:
    time_plan = json.load(stream)
shard_entries = time_plan.get("shards")
if not isinstance(shard_entries, list) or not shard_entries:
    raise SystemExit("time plan has no shards")
if not targets:
    raise SystemExit("manifest has no targets")
total = len(targets) * len(shard_entries)
if array_index < 0 or array_index >= total:
    raise SystemExit(f"array index {array_index} is outside [0, {total})")
target_index, shard_index = divmod(array_index, len(shard_entries))
source_id = int(targets[target_index]["source_id_int64"])
shard_id = int(shard_entries[shard_index]["shard_id"])
print(source_id, shard_id, len(targets), len(shard_entries))
PY
)"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "array_index=${ARRAY_INDEX} source_id=${SOURCE_ID} shard_id=${SHARD_ID} target_count=${TARGET_COUNT} shard_count=${SHARD_COUNT} case=${ET_STAMP_CASE}"
echo "manifest=${ET_STAMP_MANIFEST}"
echo "ET_DATA_DIR=${ET_DATA_DIR}"
echo "focalplane_registry=${ET_STAMP_FOCALPLANE_REGISTRY}"
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
  --batch-size "${ET_STAMP_BATCH_SIZE}"

date
