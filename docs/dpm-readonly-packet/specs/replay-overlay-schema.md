# DPM Replay Overlay Schema

Status: canonical draft
Layer: DPM replay overlay artifact

## Purpose
Define the canonical bounded replay overlay artifact that DPM may emit in read-capable modes.

The replay overlay is the compact runtime-facing summary produced from higher-level DPM records. It is not a transcript, not a persistence format, and not runtime wiring. Its purpose is to carry only the minimum deterministic continuity guidance needed for a resumed turn while remaining auditable.

## Non-goals
This schema does not define:
- runtime injection or prompt wiring
- durable write behavior
- storage/index layout for plugin records
- scoring/training algorithms for preference inference
- transcript replay or full historical summaries

## Design constraints
A compliant replay overlay must be:
- bounded in size
- explicitly scoped
- subordinate to explicit current-turn user instructions
- auditable back to included/excluded source summaries
- deterministic enough to serialize as fixtures and validate in tests

## Canonical object
A replay overlay document is a JSON object with this top-level shape:

```json
{
  "schema_version": "dpm.replay-overlay.v1",
  "overlay_id": "overlay:thread:self-architecture-portrait:2026-04-14T10:05:00Z",
  "mode": "active-read",
  "generated_at": "2026-04-14T10:05:00Z",
  "scope": {
    "primary": "thread",
    "thread_id": "thread:self-architecture-portrait",
    "project_id": "project:dpm",
    "relationship_id": "relationship:mark-nv"
  },
  "retrieval_order_applied": [
    "explicit_current_turn",
    "thread",
    "project",
    "relationship",
    "preference_graph"
  ],
  "overlay": {
    "persona_summary": "Be direct, concise, and respect privacy boundaries.",
    "style_directives": [
      "Prefer concise answers over padded phrasing.",
      "Keep thread-specific formal tone when compatible with the current request."
    ],
    "do_not_do": [
      "Do not let relationship-level humor override a brief/formal current-turn request."
    ],
    "open_questions": [],
    "max_chars": 280,
    "rendered_text": "Direct, concise, privacy-safe. In this thread, keep a slightly formal tone when it does not conflict with the user's current instruction."
  },
  "effective_constraints": {
    "explicit_instruction_precedence": "always_override",
    "narrowest_scope_wins": true,
    "cross_scope_fallback_requires_compatibility": true,
    "writes_allowed": false
  },
  "sources": [],
  "audit": {},
  "override_state": {}
}
```

## Top-level fields
- `schema_version`: fixed schema identifier. For this packet use `dpm.replay-overlay.v1`.
- `overlay_id`: stable identifier for this overlay emission.
- `mode`: one of `observe-only`, `active-read`, `active-write`.
- `generated_at`: ISO-8601 UTC timestamp.
- `scope`: explicit scope identity for the overlay.
- `retrieval_order_applied`: ordered list of retrieval layers considered during assembly.
- `overlay`: bounded runtime-facing continuity guidance.
- `effective_constraints`: hard constraints that shaped the final overlay.
- `sources`: ordered source summaries actually consulted.
- `audit`: bounded explanation of inclusion, exclusion, and conflicts.
- `override_state`: machine-readable summary of how explicit instructions constrained the result.

## Scope object
The `scope` object identifies the overlay target and compatible fallbacks.

Fields:
- `primary`: one of `thread`, `project`, `relationship`.
- `thread_id`: nullable string.
- `project_id`: nullable string.
- `relationship_id`: nullable string.

Rules:
- `primary` determines the narrowest scope the overlay is assembled for.
- if `primary == "thread"`, `thread_id` must be present.
- if `primary == "project"`, `project_id` must be present.
- if `primary == "relationship"`, `relationship_id` must be present.
- broader scope ids may be present as compatible context, but must not erase the primary scope.

## Retrieval order
`retrieval_order_applied` is an ordered list drawn from:
- `explicit_current_turn`
- `thread`
- `project`
- `relationship`
- `preference_graph`

Rules:
- the list must preserve descending precedence.
- if a layer is absent, all lower-precedence layers that remain must still preserve order.
- `explicit_current_turn`, when present, must be first.

## Overlay payload
The `overlay` object carries only bounded runtime guidance.

Fields:
- `persona_summary`: short summary sentence or paragraph.
- `style_directives`: ordered list of concise positive instructions.
- `do_not_do`: ordered list of concise negative constraints.
- `open_questions`: ordered list of unresolved questions or ambiguities.
- `max_chars`: positive integer hard cap for `rendered_text`.
- `rendered_text`: final bounded plain-text overlay intended for runtime consumption.

Rules:
- `rendered_text` length must be less than or equal to `max_chars`.
- `style_directives`, `do_not_do`, and `open_questions` must remain summary-sized.
- overlay content must not cite hidden provenance that is absent from `sources`/`audit`.
- overlay content must not override explicit current-turn instructions.

## Effective constraints
The `effective_constraints` object makes safety-visible decisions explicit.

Fields:
- `explicit_instruction_precedence`: must equal `always_override`.
- `narrowest_scope_wins`: boolean.
- `cross_scope_fallback_requires_compatibility`: boolean.
- `writes_allowed`: boolean.

Rules:
- `explicit_instruction_precedence` must be `always_override`.
- `narrowest_scope_wins` must be `true`.
- `cross_scope_fallback_requires_compatibility` must be `true`.
- `writes_allowed` may be `true` only in `active-write`; for this packet fixtures keep it `false`.

## Source summary schema
Each entry in `sources` is a bounded explanation of one consulted source.

```json
{
  "source_id": "thread:self-architecture-portrait",
  "scope": "thread",
  "kind": "thread_summary",
  "included": true,
  "priority": 2,
  "confidence": 0.78,
  "updated_at": "2026-04-14T09:55:00Z",
  "summary": "Thread-specific preference for slightly more formal phrasing."
}
```

### Source fields
- `source_id`: stable source identifier.
- `scope`: one of `thread`, `project`, `relationship`, `global`.
- `kind`: short source family label such as `thread_summary`, `project_summary`, `relationship_summary`, `preference_graph`, `current_turn_instruction`.
- `included`: boolean.
- `priority`: integer precedence position; lower is stronger.
- `confidence`: normalized value in `[0.0, 1.0]`.
- `updated_at`: ISO-8601 UTC timestamp.
- `summary`: bounded one-line summary of what the source contributed.

Rules:
- `sources` must be ordered by effective precedence, not arbitrary insertion order.
- every included source that materially shapes `rendered_text` must appear in `sources`.
- excluded candidates belong in `audit.excluded_sources`, not `sources`.

## Audit schema
The `audit` object remains summary-sized and machine-readable.

```json
{
  "included_source_ids": [
    "turn:current",
    "thread:self-architecture-portrait",
    "relationship:mark-nv"
  ],
  "excluded_sources": [
    {
      "source_id": "relationship:humor-default",
      "reason": "overridden_by_explicit_instruction"
    }
  ],
  "conflicts_detected": [
    "relationship humor preference conflicts with current-turn brief/formal instruction"
  ],
  "notes": [
    "Thread scope was preserved; relationship guidance was narrowed rather than erased."
  ]
}
```

Required fields:
- `included_source_ids`: ordered list of source ids that materially shaped the overlay.
- `excluded_sources`: list of objects with `source_id` and `reason`.
- `conflicts_detected`: list of conflict summary strings.
- `notes`: list of bounded explanatory notes.

## Override state schema
The `override_state` object records how explicit current-turn instructions changed the effective overlay.

Fields:
- `has_explicit_instruction`: boolean.
- `instruction_source_id`: nullable string.
- `override_applied`: boolean.
- `suppressed_source_ids`: ordered list of source ids or node ids whose influence was reduced or nullified.
- `effective_for_turn`: ordered list of concise active turn constraints.

Rules:
- when `override_applied` is `true`, `has_explicit_instruction` must also be `true`.
- suppressed items must be explainable via `audit.conflicts_detected` and/or `audit.excluded_sources`.

## Mode semantics
- `observe-only`: overlay may be assembled for evaluation, but must not be treated as authoritative runtime guidance.
- `active-read`: overlay may be emitted for runtime consumption; writes remain disallowed.
- `active-write`: overlay may be emitted and accompanied by write-safe downstream behavior, but this packet does not implement writes.

## Validation rules
A replay overlay document is valid for this packet when:
- `schema_version` equals `dpm.replay-overlay.v1`
- `mode` is one of `observe-only`, `active-read`, `active-write`
- all required top-level fields are present
- `scope.primary` is valid and its corresponding id is present
- `retrieval_order_applied` preserves precedence order
- every `confidence` lies within `[0.0, 1.0]`
- every timestamp is ISO-8601 UTC text
- `overlay.max_chars` is a positive integer
- `len(overlay.rendered_text) <= overlay.max_chars`
- `effective_constraints.explicit_instruction_precedence == "always_override"`
- `effective_constraints.narrowest_scope_wins is true`
- `effective_constraints.cross_scope_fallback_requires_compatibility is true`
- `override_state.override_applied` cannot be true unless `override_state.has_explicit_instruction` is true
- every `audit.included_source_ids` entry refers to a `sources[].source_id`

## Fixture intent
Packet 4 fixtures should demonstrate:
- relationship-only overlay guidance
- thread-scoped overlay refinement over broader relationship defaults
- explicit current-turn override suppression of lower-precedence guidance
- bounded rendered overlay text
- auditable included/excluded source behavior
