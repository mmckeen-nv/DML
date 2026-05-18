# DML Ollama Hardening Bundle 2 — Sprint 3 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/daystrom_dml/dml_adapter.py`
  - upgraded the embedding-compatibility progress artifact so the slow re-embed leg is operator-visible while it is happening
  - now persists bounded liveness fields:
    - `current_item_index`
    - `current_item_preview`
    - `started_at`
    - `updated_at`
  - flushes the report immediately when a mismatched item enters the `reembed` phase, instead of only after that item finishes
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the new liveness fields are written into the durable migration report

## Why this mattered

Sprint 2 made the report show coarse percentage progress, but a large live-store startup could still feel sticky during one expensive re-embed.

Now the durable artifact answers the two operator questions that matter during the tar-pit window:
- **what item is it on right now?**
- **has the report advanced recently, or is startup actually wedged?**

That keeps the Ollama migration path bounded and inspectable instead of looking like a black box.

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
  "current_item_index": 0,
  "current_item_preview": null,
  "started_at": "2026-04-07T17:06:00+00:00",
  "updated_at": "2026-04-07T17:06:01+00:00",
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
