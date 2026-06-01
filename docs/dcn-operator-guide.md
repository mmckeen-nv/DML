# Daystrom Cognition Network Operator Guide

This guide covers the current safe DCN operator surfaces. DCN remains a deterministic cognition-control layer: it can inspect intent, plan retrieval/context policy, produce cognitive packets, capture feedback, and run offline eval probes. It must not own DML storage, DPM preference/personality state, or DIP/frontier inference.

## Component boundaries

- **DML**: durable memory storage, retrieval, compression, hygiene, and audit.
- **DPM**: explicit personality/preference overlay and curated user corrections.
- **DCN**: deterministic cognition control, policy/audit/feedback/eval, and readiness gates.
- **DIP**: inference preparation boundary; optional frontier/base execution remains separate.

## Modes

The Hermes plugin exposes `dcn.mode` with these values:

- `disabled`: legacy DML/DPM provider behavior; no DCN telemetry.
- `observe_only`: DCN records hashed structural recommendations while legacy behavior remains authoritative.
- `active_read`: DCN can gate bounded DPM overlay and DML retrieval, with fallback to legacy behavior on DCN failure.
- `active_learn`: reserved for later phases; do not promote without rollbackable policy overlays and eval evidence.

## Operator CLI surfaces

When the provider is running, inspect plans and packets without changing DML/DPM/DIP state:

```bash
dml dcn observe --text "continue the DML work" --session-id abc
dml dcn packet --text "continue the DML work" --session-id abc
```

Record outcome feedback and inspect recent audit/feedback entries:

```bash
dml dcn feedback --decision-id <decision-id> --outcome verified --signals '{"tests_passed":true}'
dml dcn audit-tail --limit 20
```

Inspect and move explicit procedural-learning overlays:

```bash
dml dcn policy show
dml dcn policy export --output dcn-policy.json --snapshot-only
dml dcn policy import --input dcn-policy.json
```

Policy import/export is bounded to the DCN procedural overlay. The deterministic v0 policy remains the immutable baseline, and validation rejects wrong schemas/base refs. Overlay fields remain allowlist-only routing/gating fields such as memory mode preference, query templates, verification requirement, tool recommendation, context budget adjustment, and writeback strictness. Do not use policy import/export for identity, values, user preferences, autonomy permissions, safety boundaries, secret-handling rules, raw prompts, raw memory context, or DPM state.

## Provider eval smoke readiness probe

When the provider is running, use the read-only readiness probe:

```bash
dml dcn eval-smoke
# Alias:
dml dcn readiness
```

The CLI calls:

```http
GET /api/dcn/eval/smoke
```

Expected successful shape:

```json
{
  "status": "ok",
  "component": "daystrom-cognition-network",
  "mode": "offline_fixture_smoke",
  "report": {
    "passed": true,
    "suite_id": "provider-dcn-eval-smoke",
    "summary": {
      "case_count": 3,
      "passed_count": 3,
      "failed_count": 0,
      "avg_precision_at_k": 1.0,
      "avg_recall_at_k": 1.0,
      "max_pollution_score": 0.0,
      "blocked_polluting_items": 1
    }
  }
}
```

The CLI exits `0` only when the provider returns `status: ok` and `report.passed: true`; otherwise it prints the response and exits `1`. Transport/HTTP failures exit through the existing provider CLI error path.

## Safety invariants

The eval smoke route is safe as a readiness/promotion probe because it is:

- `GET` only; no request body or caller-provided fixtures.
- Offline fixture-only; no live DML adapter/store calls.
- No DPM state mutation or preference/personality writes.
- No DIP/frontier/network calls.
- Metrics/hashes only; no raw fixture text, prompt scaffolds, raw transcripts, tool logs, or secret-like strings.

## Promotion gate

Do not promote DCN to a stronger runtime mode unless all of these pass in the target lane:

```bash
uv run --with pytest python -m pytest dml_core/daystrom_dml/tests/test_provider_dcn_api.py dml_core/daystrom_dml/tests/test_cognition_evaluation.py dml_core/daystrom_dml/tests/test_provider_cli.py -q
uv run python integrations/hermes/plugins/daystrom_dml/smoke_dcn_eval.py
dml dcn eval-smoke
```

For broader readiness, also run:

```bash
uv run --with pytest python -m pytest dml_core/daystrom_dml/tests -q
python3 integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
uv run python integrations/hermes/plugins/daystrom_dml/smoke_dcn_eval.py
python3 integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
```

Treat any raw prompt/context/tool/secret leakage in eval output as a stopline, even if retrieval metrics pass.
