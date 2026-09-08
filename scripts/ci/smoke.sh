#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${CI_PYTHON:-python}" -m ci.run_local smoke "$@"
