# DPM Runtime Roadmap

Status: active implementation plan
Updated: 2026-05-18

## Current State

DPM is now implemented as a runtime-capable personality matrix:

- active-read overlays can be loaded from replay overlay JSON
- preference graphs can render bounded runtime overlays
- `DMLAdapter.personality_overlay(...)` exposes the active overlay
- `retrieve_context(...)` can include a `=== Personality Matrix ===` block
- active-write can record explicit preference signals into a DPM preference graph
- active-read remains read-only, and disabled / observe-only remain inert

The next work is about making DPM usable, inspectable, live-integrated, and maintainable.

## Non-Negotiable Invariants

- explicit current-turn user instructions always outrank DPM guidance
- DPM must remain disableable
- active-read must not write
- active-write must keep every write scoped, auditable, and bounded
- thread/project/relationship scope must not silently cross incompatible boundaries
- overlays must stay compact enough to improve continuity without causing context bloat
- user-facing controls must allow inspection and deletion of learned preferences

## Phase 1: Live Runtime Integration

Goal: make the running agent path actually use DPM.

Tasks:
- wire OpenClaw/live wrapper config to pass `dpm.enable`, `dpm.mode`, and graph paths into `DMLAdapter`
- ensure the live wrapper uses `retrieve_context(...)` or `build_preamble(...)` paths that include DPM
- add an end-to-end restart smoke: write preference, restart adapter/wrapper, retrieve overlay
- add environment examples for active-read and active-write

Acceptance:
- a live run can record `I prefer concise status updates`
- the preference graph persists across restart
- the next retrieval includes a bounded personality matrix block
- disabling DPM removes the block without changing normal memory retrieval

## Phase 2: DPM API And CLI

Goal: make DPM governable from outside Python internals.

Tasks:
- add `GET /dpm/overlay`
- add `GET /dpm/graph`
- add `POST /dpm/preference`
- add `DELETE /dpm/preference/{node_id}` or a suppress endpoint
- add a small CLI surface for the same operations
- return audit metadata for reads and writes

Acceptance:
- users can inspect the current overlay
- users can record an explicit preference through the API
- users can list, suppress, or remove a preference node
- active-read write attempts are rejected or return inert/no-op results

## Phase 3: Scope Intelligence

Goal: make preference learning land at the right scope.

Tasks:
- accept explicit `thread_id`, `project_id`, and `relationship_id` in API/wrapper calls
- map DML tenant/client/session metadata into DPM scopes
- add scope-specific graph node handling
- preserve narrowest-scope-wins behavior in overlay rendering
- add tests for relationship default overridden by project preference, and project preference overridden by thread preference

Acceptance:
- project-local style preferences do not leak into unrelated projects
- thread-local preferences do not overwrite relationship defaults
- broader preferences refine only when compatible with narrower scope

## Phase 4: Preference Extraction V2

Goal: improve signal quality without turning DPM into uncontrolled inference.

Tasks:
- classify explicit preference statements into known dimensions: concision, directness, warmth, initiative, caution, detail level, formality
- detect negative preferences and suppressions more reliably
- distinguish durable preference from one-turn instruction
- add confidence based on explicitness, repetition, contradiction, and source
- keep uncertain signals in review state instead of active state

Acceptance:
- obvious preferences are recorded with sensible labels and dimensions
- ambiguous statements do not become active preferences automatically
- contradictions create reviewable conflicts instead of silent overwrites

## Phase 5: Conflict Resolution And Overlay Budgeting

Goal: make overlays smarter under pressure.

Tasks:
- reserve a fixed DPM token budget separate from semantic memory budget
- add conflict summaries to overlay audit
- suppress lower-precedence guidance when current-turn instruction conflicts
- sort overlay directives by scope, confidence, recency, and safety relevance
- expose fallback/reason fields in retrieval reports

Acceptance:
- DPM never causes oversized context responses
- retrieval reports explain why an overlay was included, suppressed, or conflicted
- current-turn instructions visibly suppress stale personality guidance

## Phase 6: Maintenance And Review

Goal: keep the personality matrix healthy over time.

Tasks:
- add graph decay for stale low-confidence preferences
- add pruning/consolidation for duplicate nodes
- add a review queue for conflicted or low-confidence nodes
- add import/export and backup helpers
- add telemetry: overlay used, graph writes, conflict count, write latency, overlay token cost

Acceptance:
- stale preferences decay instead of living forever
- conflicts are reviewable
- the graph can be backed up and restored
- metrics make DPM behavior observable in production

## Immediate Next Sprint

Recommended order:

1. Wire the live wrapper and config path.
2. Add `/dpm/overlay`, `/dpm/graph`, and `/dpm/preference`.
3. Add restart persistence smoke.
4. Add delete/suppress preference node.
5. Add fixed overlay token budget.

This sequence gets DPM out of the lab and into a controllable runtime loop before adding smarter inference.
