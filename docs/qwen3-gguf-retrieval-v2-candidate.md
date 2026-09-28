# Qwen3 CUDA retrieval-policy v2

`qwen3-8b-gguf-cuda-action-json-retrieval-v2` tests a general retrieval-planning
correction after the retained CUDA completion-v1 qualification failed 3/4.
The model tried to supersede the only observed record with itself, then raised
the result cap without changing the query and finalized without the old record.
The existing validator correctly rejected that attempt before dispatch. This
is a model-planning failure, not evidence of a tool or retrieval defect.

The new profile adds guidance about reformulating queries for missing evidence,
relevance filtering versus result caps, and obtaining distinct observed records
before multi-record operations. It supplies no fixture-specific terms, exact
queries, tool sequence, references or answers. Model-owned action selection,
authentic feedback, parser, authorization, final contract, budgets and M7 gates
remain unchanged. The original CPU and CUDA v1 profiles keep their exact policy
bytes and identity composition. The new action identity binds the new guidance.

Reuse the admitted GB300 CUDA runtime without rebuilding or changing its linked
libraries, weights, tokenizer, template, all-turn JSON grammar or CPU sampler.
Record exact source, runtime identity and actual GPU admission again. Preserve
the old 3/4 results and the Nemotron 8/9 campaign. Do not merge.

Before any generation, freeze both stages with fresh noncorpus fixture data:

1. Two read-only planning controls ask for two distinct stored facts, once with
   distractors. The model chooses every query and cap. Both facts must satisfy
   the existing verifier and citation contract. Direct complete retrieval is
   valid but is not evidence of query reformulation. Record reformulation only
   when an incomplete result precedes a changed query that exposes missing
   evidence. Require 2/2 before running the next stage.
2. Run the original four readiness controls once, with fresh fixtures and their
   existing assessment: retrieval, dependent feedback, supersession/read-back,
   and injection resistance. Require 4/4. No tuning, selective retry, feedback
   rewriting, forced final or output repair between stages.

Each case retains the original six-step, 256-output-token-per-call,
1,536-output-token, 32,768-input-token and 300-second limits. Known task failures
remain recorded; unresolved execution, unknown effects or unknown usage stop
further work. A failed planning stage leaves readiness unrun under the declared
staging rule. Separate nonmodel stores may establish fixture reachability but
must not supply queries or results to the live model.

Require exact-source CI, independent preparation/freeze approval, durable
execution and both evidence replays. Only after both stages qualify may a new
complete nine-task campaign be frozen, independently approved and run once.
Replay it twice and retain every failure. Original M7 gates, including both
named tasks, determine closure. A candidate policy is not a demonstrated fix
until the required live outcomes pass. Publish only compact outcomes and hashes;
raw evidence stays on the station.
