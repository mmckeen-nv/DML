#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export DML_STORAGE_DIR="${DML_STORAGE_DIR:-${ROOT_DIR}/data/playground}"

streamlit run "${ROOT_DIR}/examples/playground/playground.py"
