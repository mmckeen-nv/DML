# Hermes / Citizen Snips integration

This directory contains the Hermes memory-provider plugin that connects Hermes/Citizen Snips to Daystrom DML memory, DPM/personality overlays, and DCN retrieval-policy gates.

The plugin is intentionally a memory/personality/cognition integration. It does **not** route model inference through the DML frontier pipeline unless a separate harness explicitly uses DIP/frontier preparation.

## Provider shape

`daystrom_dml` is the always-available memory provider shape for Hermes profiles that need durable continuity:

- **Normal turns:** with `retrieval_policy: always`, DML retrieval is part of core operation. The user should not need to say “use DML.”
- **Explicit heuristic mode:** `retrieval_policy: heuristic` preserves the old opt-in retrieval behavior for constrained contexts.
- **Disabled retrieval:** `retrieval_policy: never` / `off` / `disabled` prevents DML retrieval while keeping provider behavior explicit.
- **DPM overlay:** bounded personality/preference context remains subordinate to current-turn instructions.
- **Active continuity:** resume/handoff state is injected when retrieval is enabled and available.
- **DCN:** active-read / active-learn modes can observe and gate retrieval decisions while keeping governed promotion boundaries.

The intended default contract is:

```text
normal turn -> DPM overlay + scoped DML retrieval when retrieval_policy=always
heuristic mode -> DPM overlay + DML retrieval only for explicit recall/rehydration/long-horizon continuation
never/off -> DPM overlay only, no DML retrieval
```

DML should be a compact semantic continuity substrate, not a shadow transcript or live rolling log.

## Hygiene rules

The provider rejects or strips common context-pollution sources before writeback and before rendering recalled memory:

- `<memory-context>...</memory-context>` blocks
- injected `=== Daystrom ... ===` blocks
- DPM/personality overlay scaffolding rendered as semantic memory
- gateway/system wrapper notes and interrupted-turn boilerplate
- tool logs, Codex/process notifications, test output, and truncation markers
- raw role-prefixed transcript residue
- credential-like sensitive fields, which are redacted rather than persisted

`maintenance_scan.py` is dry-run by default and reports obvious polluted records. Use `--apply` only after reviewing the report; it quarantines via metadata rather than deleting records.

## Version

Hermes plugin version: `0.3.0`.

This version includes Citizen Snips DML default retrieval, DPM overlay rendering, DCN active-read gates, cognition-gated iteration extension decisions, and memory hygiene hardening for bounded context use.

## Operational notes

- Install under a Hermes profile as `plugins/daystrom_dml/`.
- Configure Hermes with `memory.provider: daystrom_dml` and point `memory.daystrom_dml.integration_dir` / `storage_dir` at the desired DML runtime bundle and store.
- Prefer `memory.daystrom_dml.retrieval_policy: always` for normal DML-enabled agents.
- For Hermes versions with adaptive turn-budget support, keep cognition-gated extension enabled by default so incomplete useful work gets bounded extra tool iterations while completed/noisy loops stop at the normal cap.
- Restart the Hermes CLI/gateway after updating plugin code or turn-budget config so the provider module and agent loop settings are reloaded.

Example:

```yaml
agent:
  max_turns_auto_extend: true
  max_turns_extension_policy: cognition
  max_turns_extension: 30
  max_turns_hard_cap: 300

memory:
  provider: daystrom_dml
  daystrom_dml:
    integration_dir: /path/to/integrations/daystrom-dml
    storage_dir: /path/to/dml-store
    retrieval_policy: always
    enable_memory: true
    enable_personality: true
    sync_turns: true
    timeout_seconds: 8
```

## Focused validation

From the DML repository root:

```bash
PYTHONPATH="$PWD/dml_core:$PWD" python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
PYTHONPATH="$PWD/dml_core:$PWD" python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
python -m py_compile integrations/hermes/plugins/daystrom_dml/__init__.py \
  integrations/hermes/plugins/daystrom_dml/maintenance_scan.py \
  integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py \
  integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
```

When touching the Hermes provider interface itself, also run the relevant Hermes agent memory-provider tests from the Hermes checkout.
