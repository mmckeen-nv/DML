# DPM Plugin Blueprint

Status: draft
Layer: post-foundation plugin layer on top of continuity metadata

## Purpose
DPM is an optional plugin layer that uses the continuity metadata foundation to improve long-horizon interaction continuity.

It should model:
- relationship continuity
- weighted preferences / value graph
- stable interaction style shaping
- thread / project / global precedence for recalled persona context

## Non-goals
- rewriting core DML
- replacing the continuity metadata foundation
- pretending to be mystical or opaque
- uncontrolled self-modification

## Plugin capabilities
1. Relationship memory
   - remembered interaction style
   - trust/tone expectations
   - recurring preferences in how help should feel

2. Weighted preference graph
   - dimensions like directness, concision, initiative, seriousness, privacy caution
   - reinforcement over time
   - confidence and recency weighting

3. Replay overlay
   - inject a compact personality/relationship context block on resume
   - thread-local first, project second, global relationship memory third

4. Plugin lifecycle
   - disabled
   - observe-only
   - active-read
   - active-write

## Required constraints
- optional and disableable
- auditable outputs
- bounded summaries rather than transcript dumps
- preserve thread/project/global scope boundaries
- never override explicit user instructions with stale personality inference

## Retrieval order
1. thread-local continuity
2. project continuity
3. relationship memory
4. weighted preference graph

## Initial implementation areas
- plugin contract/spec
- preference graph schema
- replay overlay schema
- lifecycle/config flags
- validation bundle
