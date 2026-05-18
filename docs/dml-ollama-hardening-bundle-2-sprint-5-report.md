# DML Ollama Hardening Bundle 2 — Sprint 5 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added a tiny operator helper that reads the durable migration artifact and renders a plain-text status view
  - keeps visibility focused on the live-store migration report that already exists instead of adding a new control plane
  - surfaces the fields that matter during a slow startup pass:
    - `status`
    - `phase`
    - `phase_detail`
    - `progress_pct`
    - `checked/total/remaining`
    - current item preview
    - last completed item preview
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended the existing migration-report regression to assert the helper renders the expected status lines from the durable artifact

## Why this mattered

The migration report was already durable and detailed, but reading raw JSON during a live startup pass is awkward.

This sprint adds one narrow progress surface with no redesign:
- operators can run one command and immediately see whether migration is moving
- the helper points at the same durable report file used by the runtime
- no new migration state, no duplicate report format, no widened control logic

## Operator command

Use this exact command against the active live-store report:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py
```

Optional explicit-path form:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py \
  --report /home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
```

Optional raw JSON passthrough:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --json
```

## Example output

```text
report_path: /home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
status: running
phase: reembed
detail: re-embedding item 18/231 due to dimension mismatch (384 -> 1536)
progress: 7.79% (18/231, remaining=213)
migration_counts: mismatched=3 reembedded=2 failed=0 target_dim=1536
current_item: index=18 preview=long persisted memory text ...
last_completed: index=17 preview=operator asked for the ollama-only path
timing: started_at=2026-04-07T17:11:00+00:00 updated_at=2026-04-07T17:11:09+00:00 elapsed_ms=9012.44
```

## Validation run for this bounded sprint

- `pytest -q /home/nvidia/.openclaw/workspace/dml/dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no migration redesign
- no unrelated DML refactors
- no new state store beyond the existing durable migration artifact
