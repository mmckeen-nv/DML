# Relationship Continuity Source Contract

Status: canonical draft
Scope: Packet 4 relationship lane contract only
Layer: post-foundation scoped source contract for relationship continuity inputs

## Purpose
Define the canonical relationship-scoped continuity source contract that sits beneath thread and project continuity while remaining the next safe bounded read source.

This contract answers:
- what a relationship continuity source record must contain
- how relationship scope participates under thread-scoped and project-scoped reads
- how boundedness is enforced for stable relationship summaries and runtime-facing overlays
- how exclusions and audit must show whether relationship scope was included, narrowed, skipped, or excluded

## Non-goals
This packet does not define:
- runtime wiring or prompt injection
- durable/main edits
- active-write behavior or mutation flows
- relationship memory consumption logic in runtime assembly
- storage/index backend implementation details
- preference-graph expansion beyond compatibility references

## Layer boundary
The continuity metadata foundation remains the source of truth for identity, retrieval-safe scope boundaries, and explicit instruction precedence.

This contract defines only the relationship-scoped source artifact and its read-time interpretation. It must not redefine:
- thread identity
- project identity
- checkpoint ownership
- registry ownership
- explicit-user-instruction precedence

## Canonical source scopes for this packet
For Packet 4, continuity reads may consider these conceptual source scopes in precedence order:
1. explicit current-turn instruction
2. thread continuity source
3. project continuity source
4. relationship continuity source
5. broader global/preference layers by reference only

Interpretation rules:
- thread scope remains primary when a thread-scoped read is in progress
- project scope remains stronger than relationship scope when both are compatible and present
- relationship scope may contribute only stable cross-thread guidance when narrower scopes are silent or compatible
- relationship source must never erase, broaden, or contradict thread-specific constraints
- relationship source must never override compatible project-scoped guidance that is already narrower than relationship scope
- broader layers must not be used as substitutes when relationship identity is incompatible or unavailable

## Canonical relationship continuity source object
A relationship continuity source document is a JSON object with this shape:

```json
{
  "schema_version": "dpm.relationship-source.v1",
  "source_id": "relationship:mark-nv",
  "relationship_id": "relationship:mark-nv",
  "scope": "relationship",
  "label": "Mark_NV relationship continuity",
  "summary": "Stable cross-thread preference for direct, concise, privacy-safe assistance.",
  "directives": [
    "Prefer concise answers over padded phrasing.",
    "Keep continuity guidance summary-sized and audit-friendly."
  ],
  "constraints": [
    "Do not override explicit current-turn instructions.",
    "Do not override narrower thread or project constraints.",
    "Do not inject transcript-like relationship history into runtime overlays."
  ],
  "priority": 4,
  "confidence": 0.74,
  "updated_at": "2026-04-14T10:15:00Z",
  "boundedness": {
    "summary_max_chars": 160,
    "directive_max_items": 4,
    "constraint_max_items": 4,
    "audit_note_max_items": 3
  },
  "exclusion_rules": [
    "Exclude relationship guidance when explicit current-turn instructions conflict.",
    "Exclude relationship guidance when thread or project scope provides narrower incompatible constraints.",
    "Exclude relationship guidance when the source violates declared boundedness."
  ],
  "audit_hint": "Relationship scope may provide stable cross-thread defaults only when compatible with narrower scopes."
}
```

## Required fields
- `schema_version`: fixed string `dpm.relationship-source.v1`
- `source_id`: stable source identifier; for relationship scope it must start with `relationship:`
- `relationship_id`: stable relationship identity string
- `scope`: fixed string `relationship`
- `label`: short human-readable label
- `summary`: bounded one-line summary of stable cross-thread continuity guidance
- `directives`: ordered list of bounded positive instructions
- `constraints`: ordered list of bounded negative or limiting rules
- `priority`: integer precedence position used in merged source ordering
- `confidence`: normalized value in `[0.0, 1.0]`
- `updated_at`: ISO-8601 UTC timestamp
- `boundedness`: per-record hard caps
- `exclusion_rules`: ordered list of bounded rejection rules for safe reads
- `audit_hint`: bounded explanation of intended merge behavior

## Required field rules
- `summary` must be non-empty and length must be less than or equal to `boundedness.summary_max_chars`
- `directives` length must be less than or equal to `boundedness.directive_max_items`
- `constraints` length must be less than or equal to `boundedness.constraint_max_items`
- `exclusion_rules` must be non-empty and summary-sized
- every directive, constraint, and exclusion rule string must be concise summary text, not transcript content
- `priority` for relationship scope must be weaker than project scope and stronger than broader global/preference fallback layers
- `confidence` must be within `[0.0, 1.0]`
- `audit_hint` must remain bounded summary text

## Narrower-scope precedence behavior
When the active read target is a thread, precedence is:
1. explicit current-turn instruction
2. thread continuity source
3. project continuity source
4. relationship continuity source
5. broader fallback layers only when compatible and allowed by higher-level contracts

When the active read target is a project and no thread source is active, precedence is:
1. explicit current-turn instruction
2. project continuity source
3. relationship continuity source
4. broader fallback layers only when compatible and allowed by higher-level contracts

### Relationship-scoped interpretation rules
- if thread source exists, it remains the narrowest controlling scope
- if project source exists, it remains stronger than relationship scope for the active project
- relationship source may contribute only guidance that is compatible with narrower scopes
- relationship source may fill omissions left by thread or project scope, but not replace thread identity, project identity, or narrower constraints
- if thread or project scope says "formal for this thread/project" and relationship scope says "casual by default", relationship guidance must be excluded or narrowed in audit
- if thread and project sources are absent, relationship source may become the primary consulted continuity source for that read, but the read must still remain within compatible relationship identity

## Boundedness rules
Boundedness is mandatory for both source artifacts and any derived safe-read selection.

### Source artifact boundedness
Each relationship source record must declare hard caps in `boundedness`.
Minimum packet rules:
- `summary_max_chars` must be a positive integer and no greater than `160`
- `directive_max_items` must be a positive integer and no greater than `4`
- `constraint_max_items` must be a positive integer and no greater than `4`
- `audit_note_max_items` must be a positive integer and no greater than `3`

### Source selection boundedness
When relationship scope is included in a merged result:
- at most one relationship source record may materially shape the result for this packet
- only summary-sized `summary`, `directives`, `constraints`, and exclusion outcomes may flow into audit or overlay assembly
- included relationship guidance must remain subordinate to the final overlay or runtime cap defined elsewhere
- relationship scope must not inject transcript-like history, unbounded notes, raw conversation excerpts, or hidden chain-of-thought style reasoning

## Exclusion rules
Any safe-read path that consults relationship scope must make relationship participation visible in audit.

Minimum audit outcomes for relationship scope:
- included: relationship source shaped the result compatibly beneath narrower scopes
- narrowed: relationship source was partially used because thread/project scope constrained it
- excluded: relationship source was rejected due to explicit instruction override, narrower-scope conflict, incompatible relationship identity, or boundedness violation
- skipped: relationship source was not consulted because a narrower scope fully satisfied the read or relationship identity was unavailable

Minimum exclusion reasons for this packet:
- `overridden_by_explicit_instruction`
- `conflicts_with_thread_scope`
- `conflicts_with_project_scope`
- `incompatible_relationship_identity`
- `boundedness_violation`
- `narrower_scope_sufficient`

## Audit expectations
Any fixture or future implementation claiming to consult relationship scope must make these expectations visible in audit-friendly output:
- the consulted `relationship_id`
- whether relationship guidance was included, narrowed, excluded, or skipped
- the exclusion reason when relationship guidance was not included
- at most `boundedness.audit_note_max_items` summary notes for relationship-specific decisions
- enough explanation to justify suppressed relationship guidance without exposing transcript-like detail

## Acceptance examples

### Example A: relationship source included under project scope
- project source says: prefer contract-first implementation notes in this project
- relationship source says: prefer direct, concise assistance across threads
- result: relationship source may contribute concise/direct defaults because it is compatible with the narrower project scope

### Example B: relationship source narrowed under thread scope
- thread source says: keep this thread highly formal and avoid humor
- relationship source says: default to friendly conversational warmth across threads
- result: relationship source may still contribute directness or privacy-safe guidance, but warmth/humor defaults must be excluded or narrowed in audit

### Example C: relationship source excluded for boundedness
- relationship source summary exceeds declared cap or includes transcript-like content
- result: relationship source must be excluded with `boundedness_violation`

## Fixture intent for Packet 4
Packet 4 fixtures should demonstrate:
- a valid relationship source record
- narrower-scope-compatible inclusion of relationship guidance
- exclusion or narrowing when relationship guidance conflicts with thread/project scope
- boundedness rejection for oversized relationship source content
- audit-visible exclusion reasons for safe-read behavior

## Packet 4 acceptance notes
Packet 4 is satisfied when the relationship lane contains:
- this canonical relationship continuity source contract
- example fixtures for valid, compatible, conflicting, and rejected relationship source behavior
- contract tests covering precedence beneath narrower scopes, boundedness rules, exclusion rules, and audit expectations

Packet 4 does not require:
- runtime assembly code
- durable/main promotion
- active-write support
- direct consumption of relationship memory in runtime overlays
