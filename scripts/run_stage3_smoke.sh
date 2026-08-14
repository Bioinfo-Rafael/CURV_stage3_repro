#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Create the environment using the README setup commands." >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -B "${ROOT_DIR}/scripts/run_stage3_smoke.py" "$@"

