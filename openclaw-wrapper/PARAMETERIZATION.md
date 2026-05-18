# Mirrored Wrapper Parameterization Notes

## Scope
This file documents the bounded implementation-root-only parameterization applied to the mirrored durable-home wrapper copy.

File patched:
- `scripts/dml_memory.py`

## What changed
The mirrored wrapper no longer hardcodes the DML core import root exclusively to:
- `/Users/markmckeen/.openclaw/workspace/dml/dml_core`

Instead, it now resolves DML core in this order:

1. `DAYSTROM_DML_CORE` environment override, if set
2. self-relative durable-home candidate:
   - `../.. / dml / dml_core`
3. workspace fallback:
   - `${OPENCLAW_WORKSPACE:-/Users/markmckeen/.openclaw/workspace}/dml/dml_core`

## Additional mirrored-wrapper ingest control
The mirrored wrapper now also exposes an explicit ingest-mode flag for deterministic proof/simple-note testing:
- `--relaxed-ingest`

When enabled for `ingest`, it intentionally disables:
- chunking
- noise filtering

This is meant for bounded proof/simple-note flows and does not change default ingest behavior.

## What did not change
The following remain intentionally pinned for this pass:
- runtime storage defaults
- operator artifact boundaries
- workspace GPU venv assumptions
- live OpenClaw operational entrypoint

## Safety intent
This pass is only meant to decouple the mirrored wrapper's implementation-root assumption enough for future isolated testing.
It is not an operational cutover.

## Live entrypoint remains
- `/Users/markmckeen/.openclaw/workspace/skills/daystrom-dml/scripts/dml_memory.py`
