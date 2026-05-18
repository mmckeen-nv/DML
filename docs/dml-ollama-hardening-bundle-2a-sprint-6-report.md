# DML Ollama Hardening Bundle 2a — Sprint 6 Report

Date: 2026-04-07
Scope: progress visibility for live-store migration state only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added derived freshness visibility on top of the existing migration artifact
  - status line now includes:
    - `freshness=<fresh|recent|stale|unknown>`
    - `updated_age_s=<seconds since report updated>`
  - markdown and progress-snapshot outputs now surface the same freshness fields
  - change remains read-only and artifact-driven; no migration control-path changes
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended the migration visibility regression to assert freshness is exposed in terminal, markdown, and JSON snapshot views

## Why this mattered

The existing helper showed migration progress, but operators still had to infer whether the report was current.

This sprint keeps scope narrow and visibility-only:

- makes it obvious whether the live-store migration report is still fresh
- exposes report age without requiring raw JSON inspection
- avoids redesigning migration state, persistence, or retry behavior

## Operator commands

Plain terminal status:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py
```

One-line status with freshness:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --one-line
```

Write the markdown progress surface:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --write-markdown
```

Write the JSON progress snapshot:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --write-snapshot-json
```

## Example visibility line

```text
migration_status=ok | phase=done | progress=100.00% | checked=4000/4000 | remaining=0 | current=4000 | last_completed=4000 | freshness=fresh | updated_age_s=4.2 | report=/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
```

## Validation run for this bounded sprint

- `pytest -q dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no migration redesign
- no new durable migration state
- no unrelated DML refactors
