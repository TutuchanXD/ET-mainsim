#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n etbase \
  python "${SCRIPT_DIR}/smoke_one_frame.py" --device cuda --mag-limit 15 --save-npz &

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n etbase \
  python "${SCRIPT_DIR}/smoke_one_frame.py" --device cuda --mag-limit 16 --save-npz &

wait

