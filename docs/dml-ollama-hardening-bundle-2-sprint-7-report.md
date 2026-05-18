# DML Ollama Hardening Bundle 2 — Sprint 7 Report

Date: 2026-04-07
Scope: live-store migration hardening for the new Ollama embedding path only

## What changed

- `dml_core/scripts/embedding_compatibility_status.py`
  - added a shell-friendly `--one-line` status view derived from the existing durable migration artifact
  - surfaces the key live-store migration fields in a single compact line:
    - migration status
    - phase
    - progress percentage
    - checked/total counts
    - remaining items
    - current item index
    - last completed item index
    - report path
  - also embeds the same one-line snapshot into the markdown status card so saved progress surfaces carry the compact summary too
- `dml_core/daystrom_dml/tests/test_dml.py`
  - extended regression coverage to assert the one-line helper and markdown status card render the expected migration fields

## Why this mattered

Sprint 6 made the migration visible in markdown, but operators still had to read multiple lines or inspect the card body to answer the fastest question:

- is the live-store migration moving right now, and roughly where is it?

This sprint keeps the change narrow and visibility-only:

- one command now yields a compact status line suitable for terminals, logs, wrappers, and lightweight dashboards
- the markdown card carries the same compact summary without introducing a second report format
- no migration redesign
- no new control plane
- no new persistence artifact beyond the existing durable report

## Operator commands

Default multi-line terminal status:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py
```

Compact one-line status:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --one-line
```

Write the default markdown snapshot with embedded one-line summary:

```bash
python /home/nvidia/.openclaw/workspace/dml/dml_core/scripts/embedding_compatibility_status.py --write-markdown
```

## Example one-line output

```text
migration_status=running | phase=reembed | progress=7.79% | checked=18/231 | remaining=213 | current=18 | last_completed=17 | report=/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json
```

## Validation run for this bounded sprint

- `pytest -q /home/nvidia/.openclaw/workspace/dml/dml_core/daystrom_dml/tests/test_dml.py -k embedding_compatibility_migration_writes_report`
  - PASS

## Boundaries kept

- no migration redesign
- no unrelated DML refactors
- no new persistence format beyond the existing durable migration artifact
