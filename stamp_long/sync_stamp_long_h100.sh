#!/usr/bin/env bash
# Synchronize local code/data to the H100 cluster before submitting stamp_long jobs.
#
# Usage:
#   ./stamp_long/sync_stamp_long_h100.sh
#
# Optional overrides:
#   REMOTE=119.78.226.37
#   REMOTE_ET_ROOT=/cluster/home/cxgao/ET

set -euo pipefail

REMOTE="${REMOTE:-119.78.226.37}"
REMOTE_ET_ROOT="${REMOTE_ET_ROOT:-/cluster/home/cxgao/ET}"
LOCAL_ET_ROOT="${LOCAL_ET_ROOT:-/home/cxgao/ET}"

RSYNC_COMMON=(
  rsync
  -az
  --info=progress2
  --exclude __pycache__
  --exclude .pytest_cache
  --exclude .mypy_cache
  --exclude .ruff_cache
)

ssh "${REMOTE}" "mkdir -p '${REMOTE_ET_ROOT}' '${REMOTE_ET_ROOT}/ET-mainsim' '${REMOTE_ET_ROOT}/Photsim7' '${REMOTE_ET_ROOT}/Photsim7-data'"

"${RSYNC_COMMON[@]}" "${LOCAL_ET_ROOT}/ET-mainsim/" "${REMOTE}:${REMOTE_ET_ROOT}/ET-mainsim/"
"${RSYNC_COMMON[@]}" "${LOCAL_ET_ROOT}/Photsim7/" "${REMOTE}:${REMOTE_ET_ROOT}/Photsim7/"
"${RSYNC_COMMON[@]}" "${LOCAL_ET_ROOT}/Photsim7-data/" "${REMOTE}:${REMOTE_ET_ROOT}/Photsim7-data/"

ssh "${REMOTE}" "cd '${REMOTE_ET_ROOT}/ET-mainsim' && git rev-parse HEAD || true; cd '${REMOTE_ET_ROOT}/Photsim7' && git rev-parse HEAD || true"
