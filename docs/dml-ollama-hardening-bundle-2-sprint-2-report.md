# DML Ollama Hardening Bundle 2 — Sprint 2 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/daystrom_dml/dml_adapter.py`
  - upgraded the embedding-compatibility report from an end-of-run artifact into an in-flight progress artifact
  - writes migration progress while the live-store rewrite is still running
  - added bounded operator fields:
    - `phase`
    - `total_items`
    - `last_checked_index`
    - `progress_pct`
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the new progress fields are persisted into the durable migration report

## Why this mattered

The previous visibility pass showed when migration started and ended, but a large live-store rewrite could still look like a startup tar pit while it was running.

Now the same durable report tells the operator:
- whether the path is in `probe`, `scan`, `reembed`, or `done`
- how many items are in scope
- how many items have been checked so far
- what percentage of the pass has completed

## Progress artifact

Expected artifact path inside the active store:

```text
/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
```

Expected shape after this sprint:

```json
{
  "status": "running|ok|migrated|partial|probe-failed|no-items|zero-dimension-probe",
  "phase": "init|probe|scan|reembed|done",
  "total_items": 0,
  "checked": 0,
  "last_checked_index": 0,
  "progress_pct": 0.0,
  "mismatched": 0,
  "reembedded": 0,
  "failed": 0,
  "target_dim": 0,
  "elapsed_ms": 0.0,
  "report_path": ".../embedding_compatibility_report.json"
}
```

## Host migration smoke command

Use this exact host command to exercise the live-store migration surface:

```bash
timeout 90s bash /home/nvidia/.openclaw/workspace/skills/daystrom-dml/scripts/dml_ollama_smoke.sh live-store
```

After it returns, inspect the durable progress artifact directly:

```bash
cat /home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
```

## Validation run for this bounded sprint

- `pytest -q /home/nvidia/.openclaw/workspace/dml/dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no architecture changes
- no unrelated DML refactors
- no widening beyond live-store migration progress visibility for the Ollama path
