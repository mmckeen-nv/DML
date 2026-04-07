# DML Ollama Hardening Bundle 2 — Sprint 6 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added markdown status-card output on top of the existing durable migration artifact
  - supports `--write-markdown` to materialize a live-store migration snapshot into a visible file
  - default snapshot target is:
    - `/home/nvidia/.openclaw/workspace/out/dml-ollama-live-store-migration-status.md`
  - keeps the helper read-only and artifact-driven; no new migration state or redesign
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the markdown status card renders expected progress fields and can be written to disk

## Why this mattered

We already had:
- the durable JSON migration artifact
- a plain-text helper for terminal inspection

What was still missing was a drop-in progress surface that could be saved and surfaced elsewhere without re-reading raw JSON.

This sprint keeps scope narrow:
- one command can now emit a markdown status card from the live artifact
- the card is suitable for `out/`-style operator surfaces and progress snapshots
- no migration control-plane changes
- no widening beyond visibility into current live-store migration state

## Operator commands

Plain terminal status:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py
```

Write the default markdown snapshot:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --write-markdown
```

Write to an explicit path:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py \
  --report /home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json \
  --write-markdown /home/nvidia/.openclaw/workspace/out/dml-ollama-live-store-migration-status.md
```

## Example markdown snapshot

```md
# DML Ollama Live-Store Migration Status

- report_path: `/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json`
- status: `running`
- phase: `reembed`
- detail: re-embedding item 18/231 due to dimension mismatch (384 -> 1536)
- progress: `7.79% (18/231, remaining=213)`
- migration_counts: `mismatched=3 reembedded=2 failed=0 target_dim=1536`
- current_item: `index=18` preview=`long persisted memory text ...`
- last_completed: `index=17` preview=`operator asked for the ollama-only path`
- timing: `started_at=2026-04-07T17:11:00+00:00 updated_at=2026-04-07T17:11:09+00:00 elapsed_ms=9012.44`
```

## Validation run for this bounded sprint

- `pytest -q /home/nvidia/.openclaw/workspace/dml/dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no migration redesign
- no unrelated DML refactors
- no new persistence format beyond the existing durable migration artifact
