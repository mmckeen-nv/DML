# Continuity Metadata Foundation

Status: draft
Scope: foundation layer before DPM plugin work

## Goal
Improve continuity by making thread/session metadata survive ingest, compaction, and retrieval.

## This is NOT the plugin
This foundation layer is additive metadata and retrieval plumbing.
DPM comes later as the optional plugin layer on top.

## Required stable fields
- provider
- channel
- chat_id
- topic_id
- thread_label
- thread_key
- session_scope

## Required behavior
- preserve stable thread metadata on ingest
- preserve stable thread metadata through compaction/abstraction
- prevent merges across incompatible thread identity
- allow retrieval filtered by stable thread metadata
- do not fall back to unrelated global memory when an explicitly thread-scoped query misses
- align checkpoint / registry / recall on the same stable identifiers

## Initial implementation order
1. retrieval filtering + no-leak fallback rules
2. compaction metadata preservation
3. CLI metadata support
4. checkpoint/recall metadata alignment
5. regression tests

## Promotion rule
Do work in project lane first.
Promote only narrow validated additive changes back to durable/main.
