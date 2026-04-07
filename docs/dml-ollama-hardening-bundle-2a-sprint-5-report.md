# DML Ollama Hardening Bundle 2a — Sprint 5 Report

Date: 2026-04-07
Scope: progress visibility for live-store migration state only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added `--write-snapshot-json` so the existing derived progress snapshot can be materialized to a stable file
  - default output path:
    - `/home/nvidia/.openclaw/workspace/out/dml-ollama-live-store-migration-snapshot.json`
  - keeps the helper read-only and artifact-driven
  - does not change migration behavior, persistence logic, or control flow
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended the migration visibility regression to assert the derived snapshot can be written to disk and matches the in-memory snapshot shape

## Why this mattered

The helper could already print migration state to stdout, but that still made wrappers and status surfaces depend on shell capture.

This sprint keeps the change narrow and visibility-only:

- operators can materialize the current live-store migration state as a stable file
- the file is derived from the durable migration artifact already written by the migration path
- no migration redesign
- no new migration state
- no widened control plane

## Operator command

Write the derived progress snapshot to disk:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --write-snapshot-json
```

Explicit output path:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py \
  --write-snapshot-json /home/nvidia/.openclaw/workspace/out/dml-ollama-live-store-migration-snapshot.json
```

## Example output file

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
  "migration_counts": {
    "mismatched": 3,
    "reembedded": 2,
    "failed": 0,
    "target_dim": 1536
  },
  "timing": {
    "started_at": "2026-04-07T18:00:00Z",
    "updated_at": "2026-04-07T18:00:12Z",
    "elapsed_ms": 12000.0
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
- no new persistence artifact beyond a derived visibility file generated from the existing durable report
