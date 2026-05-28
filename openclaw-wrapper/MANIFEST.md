# OpenClaw Wrapper Mirror Manifest

This directory is a mirror of the active OpenClaw-facing Daystrom DML wrapper bundle from the workspace.

## Source of mirror
- `/Users/markmckeen/.openclaw/workspace/skills/daystrom-dml/`

## Mirrored files

### Scripts
- `scripts/dml_memory.py`
- `scripts/dml_frontier_prepare.py`
- `scripts/tuning_utils.py`

### Config
- `config/dml_gpu_only.yaml`
- `config/dml_gpu_dashboard.yaml`

### Tests
- `tests/test_dml_memory.py`
- `tests/test_tuning_utils.py`
- `tests/test_benchmark_metrics.py`

### Docs / operator interface
- `SKILL.md`
- `DEPLOY_PROD.md`

## Important note
This is currently a **preserved mirror bundle**, not the live operational entrypoint.

Live operational entrypoint remains:
- `/Users/markmckeen/.openclaw/workspace/skills/daystrom-dml/scripts/dml_memory.py`

## Why mirrored separately
The earlier staged wrapper under `dml/skills/scripts/` did not match the active command surface.
This mirror exists so the full active wrapper lineage is preserved in the durable home before any future cutover work.
