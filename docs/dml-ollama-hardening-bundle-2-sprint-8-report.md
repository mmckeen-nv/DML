# DML Ollama Hardening Bundle 2 — Sprint 8 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added a derived `--snapshot-json` view that turns the existing durable migration artifact into a compact progress-only JSON surface
  - keeps the output focused on live migration visibility:
    - migration status
    - phase and phase detail
    - checked / total / remaining counts
    - current item and last completed item
    - migration counters and timing
    - embedded one-line status string for wrappers/loggers
  - does not change migration behavior or persistence logic; it only reformats the already-written artifact
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the new derived snapshot shape from the durable artifact

## Why this mattered

The one-line helper from sprint 7 is good for terminals, but dashboards, wrappers, and tiny status exporters still had to scrape text if they wanted structured progress.

This sprint keeps the change narrow and visibility-only:

- one command now yields machine-readable progress without reading raw migration JSON directly
- the snapshot is derived from the same durable artifact already written by migration
- no migration redesign
- no new migration state
- no extra control plane

## Operator commands

Compact one-line status:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --one-line
```

Derived progress-only JSON snapshot:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --snapshot-json
```

## Example snapshot output

```json
{
  "migration_status": "running",
  "phase": "reembed",
  "phase_detail": "re-embedding incompatible stored vectors",
  "progress": {
    "pct": 7.79,
    "checked": 18,
    "total": 231,
    "remaining": 213
  },
  "current_item": {
    "index": 18,
    "preview": "..."
  },
  "last_completed": {
    "index": 17,
    "preview": "..."
  },
  "status_line": "migration_status=running | phase=reembed | progress=7.79% | checked=18/231 | remaining=213 | current=18 | last_completed=17 | report=/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json"
}
```

## Validation run for this bounded sprint

- `pytest -q /home/nvidia/.openclaw/workspace/dml/dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no migration redesign
- no unrelated DML refactors
- no new persistence artifact; the snapshot is derived from the existing durable report
