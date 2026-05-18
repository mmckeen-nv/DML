# DPM Preference Graph Schema

Status: canonical draft
Layer: DPM weighted preference/value graph

## Purpose
Define a bounded, auditable schema for expressing stable user-facing preferences and values as weighted graph records.

This schema is storage-oriented. It does not define trainer logic, scoring algorithms, or runtime wiring.

## Design goals
- represent preferences as explicit graph nodes and edges
- carry confidence, recency, provenance, and scope in machine-readable form
- support thread, project, relationship, and global scopes without silently crossing boundaries
- stay compatible with the DPM contract requirement that explicit current-turn instructions outrank inferred history
- remain small enough to audit and serialize as fixtures

## Canonical object
A preference graph document is a JSON object with this top-level shape:

```json
{
  "schema_version": "dpm.preference-graph.v1",
  "graph_id": "rel:mark-nv",
  "subject_id": "user:mark-nv",
  "generated_at": "2026-04-14T00:00:00Z",
  "default_policy": {
    "explicit_instruction_precedence": "always_override",
    "conflict_mode": "preserve_and_audit",
    "decay_policy": "recency_weighted"
  },
  "nodes": [],
  "edges": [],
  "audit": {}
}
```

## Top-level fields
- `schema_version`: fixed schema identifier. For this packet use `dpm.preference-graph.v1`.
- `graph_id`: stable identifier for this graph document.
- `subject_id`: entity whose preferences/values are represented.
- `generated_at`: ISO-8601 UTC timestamp for graph emission/update.
- `default_policy`: graph-wide defaults for conflict and precedence handling.
- `nodes`: list of preference/value nodes.
- `edges`: list of weighted relations between nodes.
- `audit`: bounded provenance summary for the graph snapshot.

## Node schema
Each node represents a preference dimension, value, instruction-derived constraint, or supporting evidence aggregate.

```json
{
  "id": "pref.directness",
  "kind": "preference_dimension",
  "label": "Directness",
  "scope": "relationship",
  "state": "active",
  "weight": 0.86,
  "confidence": 0.91,
  "polarity": "prefer_high",
  "value_type": "scalar",
  "value": {
    "target": 0.9,
    "allowed_range": [0.7, 1.0]
  },
  "evidence": {
    "support_count": 8,
    "contradiction_count": 1,
    "last_supported_at": "2026-04-13T18:42:00Z",
    "last_contradicted_at": "2026-03-29T09:15:00Z"
  },
  "provenance": [
    {
      "type": "thread_summary",
      "source_id": "thread:self-architecture-portrait",
      "observed_at": "2026-04-13T18:42:00Z",
      "note": "User repeatedly asked for direct answers without padding."
    }
  ],
  "constraints": {
    "overridden_by_explicit_instruction": true,
    "ttl_days": 90,
    "requires_review_if_confidence_below": 0.55
  },
  "updated_at": "2026-04-13T18:42:00Z"
}
```

### Node fields
- `id`: unique node identifier within the graph.
- `kind`: one of:
  - `preference_dimension`
  - `value_commitment`
  - `interaction_style`
  - `safety_boundary`
  - `instruction_override`
  - `evidence_anchor`
- `label`: human-readable display label.
- `scope`: one of `thread`, `project`, `relationship`, `global`.
- `state`: one of `active`, `decayed`, `conflicted`, `suppressed`.
- `weight`: normalized importance score in `[0.0, 1.0]`.
- `confidence`: normalized certainty score in `[0.0, 1.0]`.
- `polarity`: one of `prefer_high`, `prefer_low`, `binary_yes`, `binary_no`, `neutral`, `mixed`.
- `value_type`: one of `scalar`, `enum`, `boolean`, `set`, `text_hint`.
- `value`: typed payload for the node.
- `evidence`: bounded aggregate counters/timestamps.
- `provenance`: bounded list of source summaries supporting the node.
- `constraints`: policy flags for expiration, override, or review behavior.
- `updated_at`: ISO-8601 UTC timestamp.

## Edge schema
Each edge expresses a directional relationship between two nodes.

```json
{
  "id": "edge.directness.supports.concision",
  "from": "pref.directness",
  "to": "pref.concision",
  "relation": "supports",
  "weight": 0.74,
  "confidence": 0.79,
  "scope": "relationship",
  "evidence": {
    "support_count": 5,
    "last_supported_at": "2026-04-11T17:20:00Z"
  },
  "provenance": [
    {
      "type": "project_summary",
      "source_id": "project:dpm",
      "observed_at": "2026-04-11T17:20:00Z",
      "note": "Direct responses and concise responses were jointly reinforced."
    }
  ],
  "updated_at": "2026-04-11T17:20:00Z"
}
```

### Edge fields
- `id`: unique edge identifier.
- `from`: source node id.
- `to`: destination node id.
- `relation`: one of:
  - `supports`
  - `constrains`
  - `conflicts_with`
  - `specializes`
  - `inherits_from`
  - `suppressed_by`
- `weight`: normalized relation strength in `[0.0, 1.0]`.
- `confidence`: normalized certainty in `[0.0, 1.0]`.
- `scope`: one of `thread`, `project`, `relationship`, `global`.
- `evidence`: bounded counters/timestamps.
- `provenance`: bounded supporting source summaries.
- `updated_at`: ISO-8601 UTC timestamp.

## Scope rules
- `thread` nodes/edges apply only to the exact thread identity that produced them.
- `project` nodes/edges may refine relationship/global behavior for work inside a named project.
- `relationship` nodes/edges capture stable cross-thread interaction tendencies when compatible with narrower scopes.
- `global` nodes/edges are the weakest historical layer and must not erase narrower constraints.

## Conflict rules
When records disagree:
- explicit current-turn instruction wins
- narrower scope wins over broader scope
- within the same scope, prefer higher confidence when recency is comparable
- within the same scope, prefer newer evidence when confidence is comparable
- unresolved conflicts must be preserved as `state: conflicted` and/or `relation: conflicts_with`

## Bounded audit schema
The `audit` object should remain summary-sized:

```json
{
  "source_count": 6,
  "included_sources": [
    "thread:self-architecture-portrait",
    "project:dpm",
    "relationship:mark-nv"
  ],
  "excluded_sources": [
    {
      "source_id": "global:generic-style",
      "reason": "narrower_scope_conflict"
    }
  ],
  "conflicts_detected": [
    "pref.humor conflicts with instruction.override.formal_mode"
  ],
  "notes": [
    "Explicit current-turn instructions are not stored as durable wins unless write mode is enabled."
  ]
}
```

## Validation rules
A document is valid for this packet when:
- `schema_version` equals `dpm.preference-graph.v1`
- every node id is unique
- every edge id is unique
- every edge endpoint refers to an existing node id
- every `weight` and `confidence` lies within `[0.0, 1.0]`
- every scope is one of `thread`, `project`, `relationship`, `global`
- every node/edge timestamp is ISO-8601 UTC text
- audit remains bounded and summary-oriented

## Fixture intent
Packet 3 fixtures should demonstrate:
- stable relationship preferences
- project-scoped refinements
- thread-scoped overrides
- explicit-instruction suppression paths
- at least one conflict edge
