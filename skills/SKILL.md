---
name: daystrom-dml
description: Use Daystrom DML as a local memory and frontier prompt-preparation substrate for OpenClaw-style long-horizon agents.
---

# Daystrom DML

Use DML silently as durable memory for long-horizon agents. Retrieve before context-sensitive work, ingest durable facts after work, and write handoffs before compaction or pause.

## Core commands

- Resume continuity:
  - `python3 skills/daystrom-dml/scripts/dml_memory.py resume --session-id "$DML_SESSION_ID" --no-require-gpu`
- Retrieve scoped memory:
  - `python3 skills/daystrom-dml/scripts/dml_memory.py retrieve --query "<current task>" --session-id "$DML_SESSION_ID" --top-k 6 --ground-truth-policy low-confidence --no-require-gpu`
- Ingest durable memory:
  - `python3 skills/daystrom-dml/scripts/dml_memory.py ingest --text "<fact>" --kind action --session-id "$DML_SESSION_ID" --meta '{"source":"openclaw"}' --no-require-gpu`
- Write a compaction handoff:
  - `python3 skills/daystrom-dml/scripts/dml_memory.py handoff --thread "<thread>" --state "<state>" --task "<task>" --next-action "<next>" --session-id "$DML_SESSION_ID" --no-require-gpu`

## Operating policy

- Do not say "according to DML" in normal user-facing answers. Use memory as background context unless asked.
- Keep `tenant_id=openclaw` for a single local user. Use unique `--session-id` values for parallel sessions.
- Store decisions, blockers, active files, commands, test results, user preferences, and next actions.
- Avoid storing secrets, raw noisy logs, transient speculation, or full generated outputs unless the output itself is the durable artifact.
- Use `handoff` whenever compaction, shutdown, or a long pause is likely.

## Frontier inference pipeline

Start the provider:

- `dml-provider --storage-dir "$DML_STORE" --host 127.0.0.1 --port 8765`

Prepare a compact frontier prompt from scoped memory:

- `python3 skills/daystrom-dml/scripts/dml_frontier_prepare.py --prompt-file task.md --session-id "$DML_SESSION_ID" --top-k 8 --frontier-max-tokens 1200`

Modes:

- `--frontier-prompt-only`: print only the frontier prompt.
- `--telemetry-only`: print compact retrieval/token telemetry.

The skill prepares prompts only. The harness or endpoint client owns model invocation and secrets. Never commit API keys or write them into memory.

## Validation

Run before beta-facing changes:

- `python3 skills/daystrom-dml/scripts/recall_eval.py --output-dir /tmp/dml-recall-eval`
- `python3 skills/daystrom-dml/scripts/stress_harness.py --writes 6 --workers 3 --tenants 1 --sessions 4`
- `python3 skills/daystrom-dml/scripts/resume_quality_smoke.py --sessions 3`

## Install

Use the repo installer to sync the OpenClaw skill bundle:

- `scripts/install_daystrom_dml.sh --profile openclaw`
