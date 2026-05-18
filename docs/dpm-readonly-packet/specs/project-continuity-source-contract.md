# Project Continuity Source Contract

Status: canonical draft
Scope: Packet A project lane contract only
Layer: post-foundation scoped source contract for project continuity inputs

## Purpose
Define the canonical project-scoped continuity source contract that sits between thread-local continuity and broader relationship/global layers.

This contract answers:
- what a project continuity source record must contain
- how project scope participates under a thread-scoped read
- how boundedness is enforced for source summaries and runtime-facing overlays
- how audit must show whether project scope was included, refined, skipped, or excluded

## Non-goals
This packet does not define:
- runtime wiring or prompt injection
- durable/main edits
- relationship/global source expansion beyond precedence references
- active-write behavior or mutation flows
- storage/index backend implementation details

## Layer boundary
The continuity metadata foundation remains the source of truth for thread identity and retrieval-safe scope boundaries.

This contract defines only the project-scoped source artifact and its read-time interpretation. It must not redefine:
- thread identity
- checkpoint ownership
- registry ownership
- explicit-user-instruction precedence

## Canonical source scopes for this packet
For Packet A, continuity reads may consider these conceptual source scopes in precedence order:
1. explicit current-turn instruction
2. thread continuity source
3. project continuity source
4. broader relationship/global layers by reference only

Interpretation rules:
- thread scope remains primary when a thread-scoped read is in progress
- project scope may refine or fill gaps when thread scope is silent or compatible
- project source must never erase, broaden, or contradict thread-specific constraints
- broader layers must not be used as substitutes when thread or project identity is incompatible

## Canonical project continuity source object
A project continuity source document is a JSON object with this shape:

```json
{
  "schema_version": "dpm.project-source.v1",
  "source_id": "project:dpm",
  "project_id": "project:dpm",
  "scope": "project",
  "label": "DPM project continuity",
  "summary": "Project context favors concise implementation-focused communication and contract-first changes.",
  "directives": [
    "Prefer contract-first changes before runtime wiring.",
    "Keep implementation notes concise and test-backed."
  ],
  "constraints": [
    "Do not let project defaults override thread-specific instructions.",
    "Do not exceed bounded runtime overlay limits."
  ],
  "priority": 3,
  "confidence": 0.7,
  "updated_at": "2026-04-14T10:00:00Z",
  "boundedness": {
    "summary_max_chars": 160,
    "directive_max_items": 4,
    "constraint_max_items": 4
  },
  "audit_hint": "Project scope may refine thread continuity when compatible, but thread scope stays primary."
}
```

## Required fields
- `schema_version`: fixed string `dpm.project-source.v1`
- `source_id`: stable source identifier; for project scope it must start with `project:`
- `project_id`: stable project identity string
- `scope`: fixed string `project`
- `label`: short human-readable label
- `summary`: bounded one-line summary of project-scoped continuity guidance
- `directives`: ordered list of bounded positive instructions
- `constraints`: ordered list of bounded negative or limiting rules
- `priority`: integer precedence position used in merged source ordering
- `confidence`: normalized value in `[0.0, 1.0]`
- `updated_at`: ISO-8601 UTC timestamp
- `boundedness`: per-record hard caps
- `audit_hint`: bounded explanation of intended merge behavior

## Required field rules
- `summary` must be non-empty and length must be less than or equal to `boundedness.summary_max_chars`
- `directives` length must be less than or equal to `boundedness.directive_max_items`
- `constraints` length must be less than or equal to `boundedness.constraint_max_items`
- every directive and constraint string must be concise summary text, not transcript content
- `priority` for project scope must be weaker than thread scope and stronger than broader relationship/global layers
- `confidence` must be within `[0.0, 1.0]`
- `audit_hint` must remain bounded summary text

## Thread-scoped precedence behavior
When the active read target is a thread, precedence is:
1. explicit current-turn instruction
2. thread continuity source
3. project continuity source
4. broader fallback layers only when compatible and allowed by higher-level contracts

### Thread-scoped interpretation rules
- if thread source exists, it remains the narrowest controlling scope
- project source may contribute only guidance that is compatible with thread scope
- project source may fill omissions left by thread scope, but not replace thread identity or thread-specific constraints
- if thread source explicitly says "formal for this thread" and project source says "casual by default", project guidance must be excluded or narrowed in audit
- if thread source is absent, project source may become the primary consulted continuity source for that read, but the read must still remain within compatible project identity

## Boundedness rules
Boundedness is mandatory for both source artifacts and any derived source selection.

### Source artifact boundedness
Each project source record must declare hard caps in `boundedness`.
Minimum packet rules:
- `summary_max_chars` must be a positive integer and no greater than `160`
- `directive_max_items` must be a positive integer and no greater than `4`
- `constraint_max_items` must be a positive integer and no greater than `4`

### Source selection boundedness
When project scope is included in a merged result:
- at most one project source record may materially shape the result for this packet
- only summary-sized `summary`, `directives`, and `constraints` content may flow into audit or overlay assembly
- included project guidance must remain subordinate to the final overlay or runtime cap defined elsewhere
- project scope must not inject transcript-like history, unbounded notes, or raw conversation excerpts

## Inclusion and exclusion audit rules
Any read path that consults project scope must make project participation visible in audit.

Minimum audit outcomes for project scope:
- included: project source shaped the result compatibly beneath thread scope
- narrowed: project source was partially used because thread scope constrained it
- excluded: project source was rejected due to thread conflict, explicit instruction override, incompatibility, or boundedness violation
- skipped: project source was not consulted because a narrower scope fully satisfied the read or project identity was unavailable

Minimum exclusion reasons for this packet:
- `overridden_by_explicit_instruction`
- `conflicts_with_thread_scope`
- `incompatible_project_identity`
- `boundedness_violation`
- `narrower_scope_sufficient`

## Acceptance examples

### Example A: project source included under thread scope
- thread source says: keep slightly formal tone for this thread
- project source says: prefer concise implementation-focused communication
- result: project source may contribute concise implementation focus because it is compatible with thread formality

### Example B: project source narrowed under thread scope
- thread source says: keep answers highly formal and avoid humor
- project source says: default to casual shorthand in this project
- result: project source may still contribute contract-first workflow guidance, but casual shorthand must be excluded or narrowed in audit

### Example C: project source excluded for boundedness
- project source summary exceeds declared cap or includes transcript-like content
- result: project source must be excluded with `boundedness_violation`

## Fixture intent for Packet A
Packet A fixtures should demonstrate:
- a valid project source record
- thread-scoped inclusion of compatible project guidance
- thread-scoped exclusion/narrowing when project guidance conflicts with thread scope
- boundedness rejection for oversized project source content

## Packet A acceptance notes
Packet A is satisfied when the project lane contains:
- this canonical project continuity source contract
- example fixtures for valid and rejected project source behavior
- contract tests covering precedence under thread scope and boundedness rules

Packet A does not require:
- runtime assembly code
- durable/main promotion
- active-write support
- relationship/global source expansion artifacts
