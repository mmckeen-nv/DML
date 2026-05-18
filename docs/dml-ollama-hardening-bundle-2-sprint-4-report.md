# DML Ollama Hardening Bundle 2 — Sprint 4 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/daystrom_dml/dml_adapter.py`
  - upgraded the embedding-compatibility progress artifact from coarse liveness into an operator-readable in-flight status surface
  - now persists bounded anti-tar-pit fields:
    - `phase_detail`
    - `remaining_items`
    - `last_completed_item_index`
    - `last_completed_item_preview`
    - `phase_started_at`
  - writes explicit per-item progress detail for both scan and re-embed legs so startup no longer looks like a blank "running" state
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the new anti-opacity fields are written into the durable migration report

## Why this mattered

Sprint 3 showed that migration was alive, but a large live-store rewrite could still feel like a startup tar pit because the durable artifact did not say what work had just completed or what phase the process was actually spending time in.

Now the report answers the operator questions that matter during slow startup:
- **what is the migration doing right now?**
- **what was the last item it definitely completed?**
- **how many items are still left before startup can move on?**
- **when did the current phase begin?**

That keeps the Ollama live-store migration inspectable instead of looking like an opaque warm-up stall.

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
  "phase_detail": "re-embedding item 18/231 due to dimension mismatch (384 -> 1536)",
  "total_items": 231,
  "checked": 18,
  "remaining_items": 213,
  "last_checked_index": 18,
  "last_completed_item_index": 17,
  "last_completed_item_preview": "operator asked for the ollama-only path",
  "progress_pct": 7.79,
  "current_item_index": 18,
  "current_item_preview": "long persisted memory text ...",
  "started_at": "2026-04-07T17:11:00+00:00",
  "updated_at": "2026-04-07T17:11:09+00:00",
  "phase_started_at": "2026-04-07T17:11:08+00:00",
  "mismatched": 3,
  "reembedded": 2,
  "failed": 0,
  "target_dim": 1536,
  "elapsed_ms": 9012.44,
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
