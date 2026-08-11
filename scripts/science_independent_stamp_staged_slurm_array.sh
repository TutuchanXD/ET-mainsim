#!/usr/bin/env bash
# Render one formal Aster/varlc/wdlc target x shard on node-local scratch,
# then publish the complete raw/coadd bundle atomically to the formal run.
#
# Submit the initial Cartesian campaign with --array=0-(targets*shards-1).
# For dynamic tail work, set ET_STAMP_TASK_LIST and its immutable
# ET_STAMP_TASK_LIST_SHA256, then use --array=0-(remaining-tasks-1).

#SBATCH --job-name=et_science_stamp_stage
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
ET_STAMP_DEVICE="${ET_STAMP_DEVICE:-cuda}"
ET_STAMP_BATCH_SIZE="${ET_STAMP_BATCH_SIZE:-64}"
ET_STAMP_CONDA_SH="${ET_STAMP_CONDA_SH:-/cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh}"
ET_STAMP_CONDA_ENV="${ET_STAMP_CONDA_ENV:-etbase-clu}"
ET_STAMP_PYTHON_BIN="${ET_STAMP_PYTHON_BIN:-python}"
ET_STAMP_TASK_LIST="${ET_STAMP_TASK_LIST:-}"
ET_STAMP_TASK_LIST_SHA256="${ET_STAMP_TASK_LIST_SHA256:-}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as an array job}"

if [[ "${ET_STAMP_CASE}" != "injected" && "${ET_STAMP_CASE}" != "static" ]]; then
  echo "ET_STAMP_CASE must be injected or static, got ${ET_STAMP_CASE}" >&2
  exit 2
fi
if [[ "${ET_STAMP_DEVICE}" != "cuda" && "${ET_STAMP_DEVICE}" != "cpu" ]]; then
  echo "ET_STAMP_DEVICE must be cuda or cpu, got ${ET_STAMP_DEVICE}" >&2
  exit 2
fi
if [[ -n "${ET_STAMP_TASK_LIST}" && -z "${ET_STAMP_TASK_LIST_SHA256}" ]]; then
  echo "ET_STAMP_TASK_LIST_SHA256 is required with ET_STAMP_TASK_LIST" >&2
  exit 2
fi
if [[ -z "${ET_STAMP_TASK_LIST}" && -n "${ET_STAMP_TASK_LIST_SHA256}" ]]; then
  echo "ET_STAMP_TASK_LIST is required with ET_STAMP_TASK_LIST_SHA256" >&2
  exit 2
fi
if [[ ! -r "${ET_STAMP_CONDA_SH}" ]]; then
  echo "ET_STAMP_CONDA_SH is not readable: ${ET_STAMP_CONDA_SH}" >&2
  exit 2
fi

# The cluster Conda deployment is configurable.
# shellcheck disable=SC1090
source "${ET_STAMP_CONDA_SH}"
conda activate "${ET_STAMP_CONDA_ENV}"

export ET_DATA_DIR="${ET_STAMP_DATA_ROOT}"
export PYTHONPATH="${ET_STAMP_CODE_ROOT}/src:${ET_STAMP_PHOTSIM_ROOT}:${ET_STAMP_ET_COORD_ROOT}/src:${PYTHONPATH:-}"

RUN_ROOT="$(cd "$(dirname "${ET_STAMP_MANIFEST}")" && pwd -P)"
SCRATCH_BASE="${ET_STAMP_LOCAL_SCRATCH_ROOT:-${TMPDIR:-/tmp}/et_science_stamp_scratch}"
MIN_SCRATCH_KB="${ET_STAMP_MIN_SCRATCH_KB:-31457280}"
mkdir -p "${SCRATCH_BASE}"
AVAILABLE_SCRATCH_KB="$(df -Pk "${SCRATCH_BASE}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_SCRATCH_KB}" =~ ^[0-9]+$ ]] || [[ "${AVAILABLE_SCRATCH_KB}" -lt "${MIN_SCRATCH_KB}" ]]; then
  echo "node-local scratch below required capacity: available_kb=${AVAILABLE_SCRATCH_KB:-unknown} required_kb=${MIN_SCRATCH_KB}" >&2
  exit 2
fi

# Read each JSON input from one byte snapshot, reject duplicate JSON keys, and
# bind an optional remaining-task list to the exact production-manifest bytes.
if ! TASK_SELECTION="$(
  "${ET_STAMP_PYTHON_BIN}" - \
    "${ET_STAMP_MANIFEST}" \
    "${ARRAY_INDEX}" \
    "${ET_STAMP_TASK_LIST}" \
    "${ET_STAMP_TASK_LIST_SHA256}" \
    "${ET_STAMP_CASE}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys


def fail(message):
    raise SystemExit(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_object(path, *, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be an existing non-symlink file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not strict JSON: {error}")
    if not isinstance(payload, dict):
        fail(f"{label} must contain a JSON object")
    return raw, payload


def strict_nonnegative_int(value, *, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer")
    return value


def content_identity(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def same_identity(actual, expected):
    return (
        type(expected) is dict
        and set(expected) == {"sha256", "size_bytes"}
        and type(expected["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is not None
        and type(expected["size_bytes"]) is int
        and expected["sha256"] == actual["sha256"]
        and expected["size_bytes"] == actual["size_bytes"]
    )


(
    manifest_text,
    array_index_text,
    task_list_text,
    expected_task_sha,
    requested_case,
) = sys.argv[1:]
if requested_case not in {"injected", "static"}:
    fail("ET_STAMP_CASE must be injected or static")
if re.fullmatch(r"0|[1-9][0-9]*", array_index_text) is None:
    fail("SLURM_ARRAY_TASK_ID must be a non-negative decimal integer")
array_index = int(array_index_text)
manifest_input_path = Path(manifest_text).expanduser()
if manifest_input_path.is_symlink():
    fail("production manifest must not be a symbolic link")
manifest_path = manifest_input_path.resolve()
manifest_raw, manifest = read_json_object(manifest_path, label="production manifest")
manifest_identity = content_identity(manifest_raw)
if (
    manifest.get("schema_id") != "et_mainsim.science_stamp_production.v1"
    or type(manifest.get("schema_version")) is not int
    or manifest.get("schema_version") != 1
):
    fail("launcher requires et_mainsim.science_stamp_production.v1")
if manifest.get("production_track") not in {"aster", "varlc", "wdlc"}:
    fail("production manifest has an unsupported production_track")

delivery = manifest.get("delivery")
if not isinstance(delivery, dict):
    fail("production manifest lacks delivery object")
execution_mode = delivery.get("execution_mode")
if execution_mode != "staged_local_scratch_v1":
    fail(
        "staged launcher requires delivery.execution_mode="
        "'staged_local_scratch_v1'; direct_shared_filesystem cannot be mixed "
        "with staged publication"
    )
relative_time_plan = delivery.get("time_plan_relative_path")
if not isinstance(relative_time_plan, str) or not relative_time_plan:
    fail("production manifest lacks delivery.time_plan_relative_path")
time_plan_input_path = manifest_path.parent / relative_time_plan
if time_plan_input_path.is_symlink():
    fail("time plan must not be a symbolic link")
time_plan_path = time_plan_input_path.resolve()
try:
    time_plan_path.relative_to(manifest_path.parent)
except ValueError:
    fail("time plan path escapes the production run root")
time_plan_raw, time_plan = read_json_object(time_plan_path, label="time plan")
if not same_identity(content_identity(time_plan_raw), delivery.get("time_plan_identity")):
    fail("time plan content identity differs from production manifest")

targets = manifest.get("targets")
shard_entries = time_plan.get("shards")
if not isinstance(targets, list) or not targets:
    fail("production manifest has no formal targets")
if not isinstance(shard_entries, list) or not shard_entries:
    fail("time plan has no formal shards")
source_ids = []
for index, target in enumerate(targets):
    if not isinstance(target, dict):
        fail(f"target {index} must be an object")
    source_ids.append(
        strict_nonnegative_int(
            target.get("source_id_int64"),
            label=f"target {index} source_id_int64",
        )
    )
if len(source_ids) != len(set(source_ids)):
    fail("production manifest contains duplicate source identities")
shard_ids = []
for index, shard in enumerate(shard_entries):
    if not isinstance(shard, dict):
        fail(f"shard {index} must be an object")
    shard_ids.append(
        strict_nonnegative_int(
            shard.get("shard_id"),
            label=f"shard {index} shard_id",
        )
    )
if len(shard_ids) != len(set(shard_ids)):
    fail("time plan contains duplicate shard identities")

task_list_digest = "none"
if task_list_text:
    task_input_path = Path(task_list_text).expanduser()
    if task_input_path.is_symlink():
        fail("task list must not be a symbolic link")
    task_path = task_input_path.resolve()
    task_raw, task_payload = read_json_object(task_path, label="task list")
    task_list_digest = hashlib.sha256(task_raw).hexdigest()
    normalised_expected_sha = expected_task_sha.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-fA-F]{64}", normalised_expected_sha) is None:
        fail("ET_STAMP_TASK_LIST_SHA256 must be one hexadecimal SHA-256")
    if task_list_digest != normalised_expected_sha.lower():
        fail("task list SHA-256 differs from ET_STAMP_TASK_LIST_SHA256")
    if set(task_payload) != {
        "schema_id",
        "schema_version",
        "case",
        "production_manifest_identity",
        "tasks",
    }:
        fail("task list must contain only the five frozen schema fields")
    if (
        task_payload.get("schema_id")
        != "et_mainsim.science_stamp_task_list.v1"
        or type(task_payload.get("schema_version")) is not int
        or task_payload.get("schema_version") != 1
    ):
        fail("unsupported science stamp task-list schema")
    if task_payload.get("case") != requested_case:
        fail("task list case differs from ET_STAMP_CASE")
    if not same_identity(
        manifest_identity,
        task_payload.get("production_manifest_identity"),
    ):
        fail("task list production manifest identity differs")
    entries = task_payload.get("tasks")
    if not isinstance(entries, list) or not entries:
        fail("task list tasks must be a non-empty array")
    source_id_set = set(source_ids)
    shard_id_set = set(shard_ids)
    tasks = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"source_id", "shard_id"}:
            fail(f"task {index} must contain only source_id and shard_id")
        task = (
            strict_nonnegative_int(entry["source_id"], label=f"task {index} source_id"),
            strict_nonnegative_int(entry["shard_id"], label=f"task {index} shard_id"),
        )
        if task in seen:
            fail(f"task list contains duplicate task {task}")
        if task[0] not in source_id_set:
            fail(f"task {index} has unknown source_id {task[0]}")
        if task[1] not in shard_id_set:
            fail(f"task {index} has unknown shard_id {task[1]}")
        seen.add(task)
        tasks.append(task)
    task_mode = "remaining-list"
else:
    if requested_case == "static":
        fail("static production requires an explicit case-bound task list")
    tasks = [(source_id, shard_id) for source_id in source_ids for shard_id in shard_ids]
    task_mode = "manifest-cartesian"

if array_index >= len(tasks):
    fail(f"array index {array_index} is outside [0, {len(tasks)})")
source_id, shard_id = tasks[array_index]
print(
    source_id,
    shard_id,
    len(source_ids),
    len(shard_ids),
    len(tasks),
    task_mode,
    task_list_digest,
    manifest_identity["sha256"],
    manifest_identity["size_bytes"],
)
PY
)"; then
  echo "science stamp task selection failed before rendering" >&2
  exit 2
fi
read -r \
  SOURCE_ID \
  SHARD_ID \
  TARGET_COUNT \
  SHARD_COUNT \
  TASK_COUNT \
  TASK_MODE \
  TASK_LIST_DIGEST \
  SELECTED_MANIFEST_SHA256 \
  SELECTED_MANIFEST_SIZE_BYTES <<<"${TASK_SELECTION}"

# The worker and publisher reopen the manifest by path.  Recheck the exact
# selection snapshot at both state transitions so a bound task list cannot be
# redirected to replacement campaign bytes after task resolution.
verify_selected_manifest_identity() {
  "${ET_STAMP_PYTHON_BIN}" - \
    "${ET_STAMP_MANIFEST}" \
    "${SELECTED_MANIFEST_SHA256}" \
    "${SELECTED_MANIFEST_SIZE_BYTES}" <<'PY'
import hashlib
from pathlib import Path
import re
import sys


manifest_text, expected_sha256, expected_size_text = sys.argv[1:]
if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
    raise SystemExit("selected production manifest SHA-256 is invalid")
if re.fullmatch(r"0|[1-9][0-9]*", expected_size_text) is None:
    raise SystemExit("selected production manifest size is invalid")
manifest_input_path = Path(manifest_text).expanduser()
if manifest_input_path.is_symlink() or not manifest_input_path.is_file():
    raise SystemExit("production manifest changed after task selection")
raw = manifest_input_path.read_bytes()
if (
    len(raw) != int(expected_size_text)
    or hashlib.sha256(raw).hexdigest() != expected_sha256
):
    raise SystemExit("production manifest changed after task selection")
PY
}

if ! verify_selected_manifest_identity; then
  echo "science stamp production manifest verification failed before rendering" >&2
  exit 2
fi

printf -v SHARD_NAME "shard_%05d" "${SHARD_ID}"
FORMAL_CASE_ROOT="${RUN_ROOT}/cases/${ET_STAMP_CASE}"
FORMAL_DELIVERY_ROOT="${FORMAL_CASE_ROOT}/stamps/target_${SOURCE_ID}/delivery"
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

JOB_SCRATCH="$(mktemp -d "${SCRATCH_BASE%/}/science-stamp-${SLURM_JOB_ID:-manual}-${ARRAY_INDEX}-XXXXXX")"
cleanup() {
  if [[ -n "${JOB_SCRATCH:-}" && -d "${JOB_SCRATCH}" ]]; then
    rm -rf -- "${JOB_SCRATCH}"
  fi
}
trap cleanup EXIT
LOCAL_CASE_ROOT="${JOB_SCRATCH}/cases/${ET_STAMP_CASE}"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "array_index=${ARRAY_INDEX} source_id=${SOURCE_ID} shard_id=${SHARD_ID} target_count=${TARGET_COUNT} shard_count=${SHARD_COUNT} task_count=${TASK_COUNT} task_mode=${TASK_MODE} case=${ET_STAMP_CASE}"
echo "manifest=${ET_STAMP_MANIFEST} task_list=${ET_STAMP_TASK_LIST:-none} task_list_sha256=${TASK_LIST_DIGEST}"
echo "formal_case_root=${FORMAL_CASE_ROOT}"
echo "local_case_root=${LOCAL_CASE_ROOT} available_scratch_kb=${AVAILABLE_SCRATCH_KB}"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

"${ET_STAMP_PYTHON_BIN}" \
  "${ET_STAMP_CODE_ROOT}/scripts/run_science_independent_stamp_production.py" \
  run-target \
  --manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${SOURCE_ID}" \
  --case "${ET_STAMP_CASE}" \
  --shard-id "${SHARD_ID}" \
  --data-root "${ET_STAMP_DATA_ROOT}" \
  --focalplane-registry "${ET_STAMP_FOCALPLANE_REGISTRY}" \
  --device "${ET_STAMP_DEVICE}" \
  --batch-size "${ET_STAMP_BATCH_SIZE}" \
  --output-root "${LOCAL_CASE_ROOT}"

# Do not let a renderer that observed replacement campaign bytes reach formal
# publication, even when the replacement remains internally self-consistent.
if ! verify_selected_manifest_identity; then
  echo "science stamp production manifest verification failed before publication" >&2
  exit 2
fi

# The module CLI validates the completed bundle and calls
# publish_staged_independent_stamp_shard for one atomic formal publication.
"${ET_STAMP_PYTHON_BIN}" -m et_mainsim.staged_stamp_delivery publish \
  --staged-case-root "${LOCAL_CASE_ROOT}" \
  --formal-case-root "${FORMAL_CASE_ROOT}" \
  --production-manifest "${ET_STAMP_MANIFEST}" \
  --source-id "${SOURCE_ID}" \
  --shard-id "${SHARD_ID}" \
  --case "${ET_STAMP_CASE}"

date
