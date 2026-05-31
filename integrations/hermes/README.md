# Hermes / Citizen Snips integration

This directory contains the Hermes memory provider plugin used to connect
Citizen Snips to Daystrom DML memory and DPM/personality overlays.

The plugin is intentionally memory/personality-only. It does **not** route model
inference through the DML frontier pipeline.

## Provider shape

`daystrom_dml` now uses a gated memory-provider shape:

- **Normal conversational turns:** inject only the bounded Daystrom Personality
  Matrix / DPM overlay, so identity and style remain stable without spending the
  context window on a rolling transcript.
- **Compaction or context-loss recovery:** inject scoped `Active Continuity`
  when the turn explicitly asks to rehydrate, resume, restore, or recover prior
  context/state.
- **Explicit memory recall:** retrieve bounded semantic memory when the user asks
  to remember, look up, or reconstruct prior decisions.
- **Long-horizon continuation:** retrieve scoped continuity for multi-turn,
  multi-session setup/migration/project work where earlier state materially
  affects the next action.

The intended contract is:

```text
normal turn -> bounded personality overlay only
rehydration / explicit recall / long-horizon continuation -> personality + scoped continuity/retrieval
```

DML should be a selective rehydration and long-horizon recall substrate, not a
shadow transcript or live rolling log.

## Hygiene rules

The provider rejects or strips common context-pollution sources before writeback
and before rendering recalled memory:

- `<memory-context>...</memory-context>` blocks
- injected `=== Daystrom ... ===` blocks
- DPM/personality overlay scaffolding rendered as semantic memory
- gateway/system wrapper notes and interrupted-turn boilerplate
- tool logs, Codex/process notifications, test output, and truncation markers
- raw role-prefixed transcript residue
- credential-like sensitive fields, which are redacted rather than persisted

`maintenance_scan.py` is dry-run by default and reports obvious polluted records.
Use `--apply` only after reviewing the report; it quarantines via metadata rather
than deleting records.

## Version

Hermes plugin version: `0.2.0`.

This version includes the Citizen Snips DML gating and hygiene hardening for
bounded context use.

## Operational notes

- Install under a Hermes profile as `plugins/daystrom_dml/`.
- Configure Hermes with `memory.provider: daystrom_dml` and point
  `memory.daystrom_dml.integration_dir` / `storage_dir` at the desired DML
  runtime bundle and store.
- Restart the Hermes CLI/gateway after updating the plugin code so the provider
  module is reloaded.

## Focused validation

From the DML repository root:

```bash
PYTHONPATH="$PWD/dml_core:$PWD" python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python -m py_compile integrations/hermes/plugins/daystrom_dml/__init__.py \
  integrations/hermes/plugins/daystrom_dml/maintenance_scan.py \
  integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
```

When touching the Hermes provider interface itself, also run the relevant Hermes
agent memory-provider tests from the Hermes checkout.
