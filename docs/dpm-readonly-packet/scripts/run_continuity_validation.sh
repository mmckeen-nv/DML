#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON_BIN" -m pytest -q \
  tests/unit/test_ingress_preflight_packet_guard.py \
  tests/unit/test_dpm_plugin_validation_scaffold.py \
  tests/unit/test_dpm_config_contract.py \
  tests/test_continuity_checkpoint_contract.py \
  tests/unit/test_project_continuity_source_contract.py \
  tests/unit/test_relationship_continuity_source_contract.py \
  tests/unit/test_continuity_recall_no_leak.py \
  tests/unit/test_runtime_coherence_regressions.py \
  tests/unit/test_preference_graph_schema.py \
  tests/unit/test_replay_overlay_schema.py \
  tests/smoke/test_layout.py
