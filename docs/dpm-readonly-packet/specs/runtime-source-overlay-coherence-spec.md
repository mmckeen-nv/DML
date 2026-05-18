# Runtime Source/Overlay Coherence Spec

Status: canonical draft
Scope: next-sprint Packet 1 runtime lane coherence only
Layer: contract alignment across project source objects, replay overlay artifacts, runtime seam payloads, and audit surfaces

## Purpose
Define one canonical shape-mapping so the DPM runtime lane uses the same conceptual objects and field meanings across:
- source records consulted during continuity reads
- replay overlay artifacts emitted for runtime-facing consumption
- the runtime seam object returned by the current continuity recall seam
- audit fields that explain inclusion, exclusion, precedence, and override behavior

This spec is intentionally about coherence, not implementation wiring. It exists so contract docs, fixtures, and the current runtime seam all describe the same shapes with compatible names and semantics.

## Non-goals
This spec does not define:
- prompt/runtime injection wiring
- durable or active-write mutation behavior
- storage/index backend layout
- preference inference algorithms
- durable/main promotion

## Canonical layers
For this packet, the coherent runtime read path consists of four layers:
1. canonical source object families
2. canonical replay overlay artifact
3. canonical runtime seam payload
4. canonical audit alignment rules

The same precedence and scope semantics must hold across all four layers.

## Precedence and scope model
Canonical precedence order for read-capable modes:
1. `explicit_current_turn`
2. `thread`
3. `project`
4. `relationship`
5. `preference_graph`

Rules:
- narrower compatible scope wins over broader scope
- lower-precedence layers may refine but never contradict higher-precedence layers
- explicit current-turn instructions always override plugin-derived guidance
- project scope may refine a thread-scoped read only when project identity is compatible
- broader relationship/preference layers must not silently substitute for incompatible thread/project identity

## Canonical object families

### 1. Source record family
A source record is the canonical consulted-input object family.

For Packet 1, the concrete source contracts are:
- thread continuity source: existing runtime/checkpoint-backed thread input
- project continuity source: `dpm.project-source.v1`
- relationship/preference inputs: referenced conceptually by overlay schema and plugin contract
- explicit current-turn instruction source: conceptual highest-precedence source when present

Every source consulted by overlay assembly must normalize to this runtime-facing source summary shape:

```json
{
  "source_id": "project:dpm",
  "scope": "project",
  "kind": "project_summary",
  "included": true,
  "priority": 2,
  "confidence": 0.8,
  "updated_at": "2026-04-14T18:00:00Z",
  "summary": "Project continuity compatible with DPM runtime seam."
}
```

Required normalized fields:
- `source_id`: stable identifier for the consulted source
- `scope`: one of `thread`, `project`, `relationship`, `global`
- `kind`: source family label such as `thread_summary`, `project_summary`, `relationship_summary`, `preference_graph`, `current_turn_instruction`
- `included`: whether the source materially shaped the resulting overlay
- `priority`: effective precedence position in the consulted result; lower is stronger
- `confidence`: normalized score in `[0.0, 1.0]`
- `updated_at`: ISO-8601 UTC timestamp
- `summary`: bounded summary of the source contribution

Normalization rules:
- thread/project source artifacts may have richer source-local fields, but runtime read outputs must expose the normalized summary shape above in `sources`
- the normalized summary shape is the only source shape the replay overlay and runtime seam may expose publicly for Packet 1
- excluded candidates do not belong in `sources`; they belong in audit

### 2. Project continuity source to normalized source summary mapping
The canonical mapping from `dpm.project-source.v1` to a runtime-consulted source summary is:

| Project source field | Normalized source field | Notes |
| --- | --- | --- |
| `source_id` | `source_id` | preserved exactly |
| `scope` | `scope` | must remain `project` |
| implicit source family | `kind` | set to `project_summary` for Packet 1 |
| inclusion decision at read time | `included` | derived, not stored in source artifact |
| `priority` | `priority` | preserved if already in merged precedence order |
| `confidence` | `confidence` | preserved exactly |
| `updated_at` | `updated_at` | preserved exactly |
| `summary` / bounded contribution | `summary` | must remain bounded summary text |

Additional project-source fields such as `label`, `directives`, `constraints`, `boundedness`, and `audit_hint` are source-artifact fields. They may inform overlay assembly and audit reasoning, but they do not change the normalized `sources[]` summary contract exposed to runtime.

### 3. Replay overlay artifact family
The replay overlay remains the canonical runtime-facing artifact schema: `dpm.replay-overlay.v1`.

Its top-level fields are:
- `schema_version`
- `overlay_id`
- `mode`
- `generated_at`
- `scope`
- `retrieval_order_applied`
- `overlay`
- `effective_constraints`
- `sources`
- `audit`
- `override_state`

Coherence rules:
- `sources` must use the normalized source summary shape defined above
- `audit.included_source_ids` must correspond exactly to the subset of normalized `sources[].source_id` values with material influence
- `retrieval_order_applied` must use the canonical precedence labels from this spec
- `scope` must identify the active target scope without erasing broader compatible context ids
- `override_state` must express whether explicit current-turn instructions constrained lower-precedence guidance

### 4. Runtime seam object family
For Packet 1, the current runtime seam object is the payload returned by `scripts/continuity_recall.py::build_payload`.

Canonical seam shape:

```json
{
  "retrieval_status": "hit",
  "thread": {},
  "checkpoint_excerpt": "...",
  "self_state": {},
  "open_loops": [],
  "handoff": {},
  "dpm_overlay": {
    "schema_version": "dpm.replay-overlay.v1"
  }
}
```

Runtime seam rules:
- `dpm_overlay` is the seam field that carries the canonical replay overlay artifact when DPM active-read output exists
- when no overlay is emitted, `dpm_overlay` must be `null` rather than a partial or alternate shape
- the seam must not invent a second overlay schema parallel to `dpm.replay-overlay.v1`
- seam-level metadata such as `retrieval_status`, `thread`, and `checkpoint_excerpt` may remain seam-specific, but they must not redefine overlay/source/audit semantics already defined by the overlay contract

## Canonical field alignment

### Scope alignment
The following scope vocabulary is canonical across source, overlay, and seam contexts:
- `thread`
- `project`
- `relationship`
- `global`

Alignment rules:
- source summary `scope` and overlay `scope.primary` must use the same vocabulary family
- a thread-primary overlay may still include `project_id` and `relationship_id` as broader compatible context
- project-derived guidance included under thread scope must still appear in `sources[]` with `scope: "project"`

### Precedence alignment
The following precedence labels are canonical across contracts:
- `explicit_current_turn`
- `thread`
- `project`
- `relationship`
- `preference_graph`

Alignment rules:
- `retrieval_order_applied` must preserve this descending order whenever listed items are present
- `sources[].priority` must be numerically consistent with the same effective order
- audit inclusion/exclusion reasoning must explain precedence suppression using these same layer names or directly corresponding source ids

### Audit field alignment
The audit contract is coherent only when these meanings stay stable:

- `audit.included_source_ids`
  - ordered source ids that materially shaped the resulting overlay
  - every id must correspond to some `sources[].source_id`

- `audit.excluded_sources[]`
  - bounded records for candidate sources not used in the final overlay
  - each entry must contain exactly `source_id` and `reason`

- `audit.conflicts_detected[]`
  - bounded conflict summaries describing precedence or compatibility conflicts

- `audit.notes[]`
  - bounded explanatory notes about why the assembled result is valid or narrowed

Canonical exclusion reasons currently supported across Packet 1 source/overlay coherence:
- `overridden_by_explicit_instruction`
- `conflicts_with_thread_scope`
- `incompatible_project_identity`
- `boundedness_violation`
- `narrower_scope_sufficient`

### Override alignment
`override_state` is the canonical machine-readable explanation of current-turn suppression.

Required meanings:
- `has_explicit_instruction`: whether an explicit current-turn source was present
- `instruction_source_id`: source id for that explicit instruction, or `null`
- `override_applied`: whether explicit current-turn instructions materially constrained lower-precedence guidance
- `suppressed_source_ids`: ordered ids of sources whose influence was reduced or nullified
- `effective_for_turn`: ordered concise active constraints or effective source ids for this turn

Coherence rules:
- if `override_applied` is `true`, `has_explicit_instruction` must also be `true`
- any suppressed source named in `override_state.suppressed_source_ids` must be explainable via `audit.excluded_sources` and/or `audit.conflicts_detected`
- the runtime seam must not expose a second override explanation format outside `dpm_overlay.override_state` for DPM-specific override behavior

## Cross-artifact invariants
The coherence spec is satisfied when all of the following are true:
- replay overlays use `dpm.replay-overlay.v1`
- project continuity source artifacts use `dpm.project-source.v1`
- overlay `sources[]` expose normalized source summaries rather than raw project/thread artifact shapes
- the runtime seam carries the overlay only through `dpm_overlay`
- `audit.included_source_ids` align with `sources[].source_id`
- precedence order is consistent between `retrieval_order_applied`, `sources[].priority`, and audit explanations
- thread scope remains primary during thread-scoped reads even when compatible project guidance is included
- explicit current-turn instructions remain the highest-precedence override mechanism

## Runtime seam mapping for the current implementation
For the current Packet 1 seam implementation in `scripts/continuity_recall.py`:
- `build_payload(...)["dpm_overlay"]` is the canonical replay overlay carrier
- `build_dpm_overlay(...)` must emit `sources[]` using normalized source summaries
- `collect_runtime_read_sources(...)` may use richer internal source objects, but emitted overlay `sources[]` must remain normalized
- seam-specific fields outside `dpm_overlay` may support recall diagnostics, but they are not part of the overlay contract

## Acceptance examples

### Example A: thread-only active-read overlay
- seam returns `dpm_overlay`
- overlay `retrieval_order_applied == ["thread"]`
- overlay `sources` contains only the thread normalized source summary
- `audit.included_source_ids == [thread source id]`

### Example B: thread plus compatible project overlay
- seam returns `dpm_overlay`
- overlay `retrieval_order_applied == ["thread", "project"]`
- overlay `sources[0].scope == "thread"`
- overlay `sources[1].scope == "project"`
- audit shows both included source ids in the same order
- thread remains `scope.primary == "thread"`

### Example C: explicit instruction suppresses broader guidance
- overlay `retrieval_order_applied` begins with `explicit_current_turn`
- `override_state.override_applied == true`
- suppressed lower-precedence source ids appear in `override_state.suppressed_source_ids`
- audit explains suppression via conflict and/or exclusion records

## Packet 1 acceptance notes
Packet 1 coherence is satisfied when the project lane contains:
- this coherence spec artifact
- existing source and replay overlay specs that remain compatible with it
- regression/smoke tests demonstrating current contracts still pass

This packet does not require:
- runtime wiring changes
- durable/main edits
- active-write implementation
