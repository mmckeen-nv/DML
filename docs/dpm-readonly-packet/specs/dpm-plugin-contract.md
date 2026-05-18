# DPM Plugin Contract

Status: canonical draft
Layer: post-foundation additive plugin layer on top of continuity metadata

## Purpose
DPM is an optional plugin layer that consumes the continuity metadata foundation and produces a bounded, auditable continuity overlay for runtime use.

The plugin exists to:
- improve long-horizon interaction continuity without changing the underlying metadata substrate
- express relationship continuity, stable interaction preferences, and style-shaping signals as explicit plugin outputs
- provide a deterministic retrieval and merge contract for thread, project, and relationship-scoped plugin data
- ensure any continuity shaping remains reviewable, disableable, and subordinate to explicit user intent

## Non-goals
DPM is not responsible for:
- replacing or rewriting the continuity metadata foundation
- changing durable/main behavior by itself
- inferring hidden intent beyond the evidence captured in plugin records
- replay injection/runtime wiring implementation
- autonomous self-modification or self-authorized policy changes
- overriding explicit user instructions with inferred preferences, historical summaries, or stale relationship state

## Layer boundary
The continuity metadata foundation remains the source of truth for thread/session identity and retrieval-safe scoping.

DPM is a post-foundation layer that may read foundation metadata and emit plugin records/overlays, but it must not redefine:
- thread identity
- checkpoint path contracts
- registry ownership
- core retrieval safety boundaries

## Canonical runtime object
The plugin contract is satisfied when DPM can produce a bounded plugin result with these conceptual parts:
- mode: current lifecycle mode
- sources: ordered records actually used for the result
- overlay: compact continuity guidance produced from those records
- audit: machine-readable explanation of why each source was included or excluded
- override_state: whether explicit user instruction constrained or nullified plugin guidance

This document defines contract behavior only, not storage or runtime implementation.

## Lifecycle modes
DPM supports four lifecycle modes.

### 1. disabled
Behavior:
- no plugin retrieval
- no plugin overlay output
- no plugin writes
- runtime behaves as if DPM does not participate

### 2. observe-only
Behavior:
- plugin may inspect candidate records/signals for evaluation
- no plugin overlay may influence runtime behavior
- no durable plugin writes unless explicitly defined by a separate write-safe contract
- audit may record what would have been used

Use case:
- validation, dry-run comparison, safety review

### 3. active-read
Behavior:
- plugin may retrieve and assemble plugin records
- plugin may emit a bounded overlay for runtime consumption
- plugin does not create or mutate durable plugin state
- audit must identify all consulted records and exclusions

Use case:
- safe read-only rollout

### 4. active-write
Behavior:
- plugin may retrieve records, emit overlays, and write/update plugin state
- all writes must remain scoped, attributable, and auditable
- writes must not bypass the explicit-user-instruction override rule

Use case:
- full plugin operation with accountability

## Retrieval precedence
When DPM is allowed to read, retrieval precedence is:
1. explicit current-turn user instructions
2. thread-local plugin continuity
3. project-scoped plugin continuity
4. relationship memory
5. weighted preference graph

### Interpretation rules
- higher-precedence sources constrain lower-precedence sources
- lower-precedence sources may refine but never contradict higher-precedence sources
- if thread-local data exists, project/global layers must not erase or broaden thread-specific constraints
- if a scoped query misses at a narrower level, fallback may proceed only to the next allowed scope under existing metadata safety rules
- fallback must never treat unrelated global/plugin data as a substitute for an explicitly incompatible thread identity

## Merge and conflict rules
When multiple plugin sources are present:
- preserve the narrowest compatible scope
- prefer more recent evidence within the same scope when confidence is comparable
- prefer higher-confidence evidence when recency is comparable
- mark unresolved conflicts in audit output rather than silently collapsing them
- if conflict would materially change runtime behavior, explicit user instruction wins and conflicting plugin guidance must be suppressed

## Audit contract
Every non-disabled plugin read/write path must be auditable.

Minimum audit fields:
- mode
- retrieval_order_applied
- included_sources
- excluded_sources
- exclusion_reason per excluded source
- effective_constraints
- conflicts_detected
- override_state
- write_intent or write_result when in active-write mode

Audit requirements:
- identify which source records shaped the final overlay
- identify which candidate records were ignored and why
- make scope boundaries visible (thread/project/relationship/global preference layer)
- make override application visible when user instructions narrowed or nullified plugin output
- remain bounded and summary-oriented; no transcript dump is required to satisfy auditability

## Explicit-user-instruction override rule
Explicit current user instructions always outrank DPM-derived guidance.

Mandatory behavior:
- if the user gives a direct instruction that conflicts with plugin guidance, the conflicting plugin guidance must be discarded or constrained
- the runtime must never justify ignoring a direct user request by citing relationship memory, preference weights, or prior continuity overlays
- when a user instruction is newer, narrower, or more specific than plugin state, the user instruction becomes the effective rule for the current turn
- if appropriate, active-write mode may later record that the user expressed a new preference, but only after the current-turn instruction has already been honored

Examples of override-triggering instructions:
- "be brief"
- "do not use prior preference memory for this"
- "ignore previous tone assumptions"
- "for this project, use formal language"

## Safety constraints
DPM must remain:
- optional
- disableable at runtime
- additive rather than foundational
- bounded in output size
- scoped according to foundation retrieval safety
- transparent through audit output

DPM must never:
- override direct user instructions
- fabricate confidence or provenance
- cross incompatible scope boundaries silently
- rely on opaque hidden state to shape behavior without audit trace

## Acceptance notes for Packet 1
Packet 1 is satisfied by a canonical contract artifact that defines:
- purpose
- non-goals
- lifecycle modes
- retrieval precedence
- audit contract
- explicit-user-instruction override rule

Packet 1 does not require:
- runtime implementation
- preference inference logic
- replay injection code
- durable/main edits

See also: `specs/config/dpm-lifecycle-config-contract.md` and `specs/config/dpm-config.schema.json` for the Packet 2 lifecycle/config contract artifacts.
