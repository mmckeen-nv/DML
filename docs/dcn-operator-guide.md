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
dml dcn policy checkpoint --label before-active-learn
dml dcn policy checkpoints
dml dcn policy rollback --checkpoint-id <checkpoint-id>
dml dcn seed-trial --input sanitized-feedback.json --output dcn-seed-trial-artifact.json
dml dcn seed-propose --input sanitized-feedback.json --output dcn-seed-proposal.json
dml dcn seed-loop --input sanitized-feedback.json --output dcn-seed-loop-artifact.json
dml dcn promote --mode active_learn --checkpoint-id <checkpoint-id> --hygiene-evidence '{"passed":true,"artifact_hash":"..."}'
dml dcn promotions --limit 20
```

Policy import/export is bounded to the DCN procedural overlay. The deterministic v0 policy remains the immutable baseline, and validation rejects wrong schemas/base refs, unknown profile fields, invalid enum values, and runaway context-budget drift. Overlay fields remain allowlist-only routing/gating fields such as memory mode preference, query templates, verification requirement, tool recommendation, context budget adjustment, and writeback strictness. Do not use policy import/export for identity, values, user preferences, autonomy permissions, safety boundaries, secret-handling rules, raw prompts, raw memory context, or DPM state. Create a checkpoint before importing or promoting stronger modes, and use rollback to return to a known checkpoint or baseline if eval/readiness fails.

Active-learn promotion is fail-closed. `dml dcn promote --mode active_learn` requires an existing checkpoint ID, a passing built-in provider eval smoke report, and explicit hygiene evidence such as the artifact hash from `smoke_hygiene.py`. Promotion records only sanitized audit metadata: previous/target mode, checkpoint ID, rollback command, policy digest, eval summary/hash, hygiene evidence, operator, and reason digest. Raw transcripts, raw prompts, tool logs, secrets, and raw memory context must never appear in promotion evidence. The Hermes plugin only honors `active_learn` when that sanitized promotion evidence is configured under `memory.daystrom_dml.dcn.promotion` or supplied as `DAYSTROM_DCN_PROMOTION_EVIDENCE`; otherwise it falls back closed to `active_read` while recording the requested mode.

`dml dcn seed-trial` is the non-promoting learning-loop bridge. It consumes sanitized feedback/proposal JSON, validates candidate procedural overlay updates through the same allowlisted learning policy used by DCN, and emits a portable artifact with accepted updates, rejected updates, unsupported policy-pressure reports, and a candidate policy snapshot. It does not import the policy, promote runtime mode, call live provider APIs, or persist raw prompts/transcripts/tool logs/secrets. Use unsupported policy-pressure entries when the current schema is too small, e.g. a proposed `preferred_tool_sequence` field for future human-reviewed schema expansion.

`dml dcn seed-propose` activates the local seed model as an offline candidate proposer. By default it calls Ollama `llama3:8b`, asks for JSON-only procedural candidates over the current allowlist, rejects model items outside that allowlist, and preserves unsupported schema pressure separately. `dml dcn seed-loop` runs `seed-propose` and `seed-trial` in one command so the model-generated proposal is immediately validated into a non-promoting trial artifact. These commands make the new learning path active while preserving the promotion boundary: no policy import or `active_learn` mode change happens until the artifact passes eval, hygiene, checkpoint, and operator promotion gates.

## Provider eval smoke readiness probe

When the provider is running, use the read-only readiness probe:

```bash
dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only
# Alias:
dml dcn readiness --output dcn-eval-artifact.json --artifact-only
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
      "case_count": 9,
      "passed_count": 9,
      "failed_count": 0,
      "avg_precision_at_k": 1.0,
      "avg_recall_at_k": 1.0,
      "max_pollution_score": 0.0,
      "blocked_polluting_items": 3
    }
  },
  "artifact": {
    "schema_version": "dcn-eval-artifact-v1",
    "passed": true,
    "summary": {
      "case_count": 9,
      "passed_count": 9,
      "failed_count": 0,
      "avg_precision_at_k": 1.0,
      "avg_recall_at_k": 1.0,
      "max_pollution_score": 0.0,
      "blocked_polluting_items": 3
    },
    "coverage": {
      "case_ids": ["clean_resume_retrieval", "..."],
      "task_types": ["admin", "answer", "code_change", "debugging", "planning", "recall"],
      "retrieval_modes": ["hybrid", "none", "resume", "semantic"],
      "writeback_modes": ["durable_signal_only", "none", "preference_candidate"],
      "frontier_modes": ["direct", "dml_context"],
      "risk_levels": ["low", "medium"],
      "reason_codes": ["code_task", "...", "verification_needed"],
      "tool_recommendation_cases": 5,
      "verification_required_cases": 4,
      "confirmation_required_cases": 1
    },
    "readiness": {
      "ready": true,
      "gate_count": 15,
      "failed_gates": [],
      "gates": [
        {"name": "suite_passed", "passed": true, "severity": "blocker", "observed": true, "required": true},
        {"name": "minimum_case_count", "passed": true, "severity": "blocker", "observed": 9, "required": ">=9"},
        {"name": "zero_pollution", "passed": true, "severity": "blocker", "observed": 0.0, "required": 0.0},
        {"name": "pollution_filter_exercised", "passed": true, "severity": "blocker", "observed": 3, "required": ">=3"},
        {"name": "frontier_mode_coverage", "passed": true, "severity": "blocker", "observed": ["direct", "dml_context"], "required": ["direct", "dml_context"]},
        {"name": "confirmation_required_exercised", "passed": true, "severity": "blocker", "observed": 1, "required": ">=1"}
      ]
    },
    "artifact_hash": "<stable sha256 prefix>",
    "redaction_policy": {
      "prompts_included": false,
      "fixture_text_included": false,
      "transcripts_included": false,
      "tool_logs_included": false,
      "secrets_included": false
    }
  }
}
```

The CLI exits `0` only when the provider returns `status: ok`, `report.passed: true`, and `artifact.readiness.ready: true`; otherwise it prints the response and exits `1`. Transport/HTTP failures exit through the existing provider CLI error path. Use `--output ... --artifact-only` when you need a portable promotion/readiness artifact; it writes only deterministic coverage, metrics, outcome metadata, gate verdicts, and hashes.

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
uv run python integrations/hermes/plugins/daystrom_dml/smoke_dcn_eval.py --output dcn-eval-artifact.json
dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only
```

For broader readiness, also run:

```bash
uv run --with pytest python -m pytest dml_core/daystrom_dml/tests -q
python3 integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
uv run python integrations/hermes/plugins/daystrom_dml/smoke_dcn_eval.py
python3 integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
```

Treat any raw prompt/context/tool/secret leakage in eval output as a stopline, even if retrieval metrics pass.
