---
name: daystrom-dml
description: Use the Daystrom Memory Lattice (DML) as a local long-horizon memory substrate for agent workflows.
---

# Daystrom DML

## Purpose
Attach OpenClaw workflows to a persistent local memory substrate (DML) for ingest + retrieval beyond short chat context.

## Repo
- Active local operator wrapper: `/Users/markmckeen/.openclaw/workspace/skills/daystrom-dml`
- Portable/install bundle lane: bundle root with `dml/` + `openclaw-wrapper/`
- Durable-home bundle remains a reference/proof surface, not the current live operator
- Branch: `openclaw`

## Durable GPU venv
- Canonical durable env: `/Users/markmckeen/.openclaw/daystrom-dml-v2/.venv-dml`
- Activate: `source /Users/markmckeen/.openclaw/daystrom-dml-v2/.venv-dml/bin/activate`
- Required production path: Ollama-backed embeddings + Ollama summarization + FAISS persistence
- CUDA stack pinned for GPU execution:
  - `torch==2.10.0+cu130`
  - `torchvision==0.25.0+cu130`
- `sentence-transformers` remains a supported optional path for alternate experiments or compatibility, but it is **not** the default production path.

## Runtime helper
Use the bundled helper script (GPU-only by default, Ollama-native by policy):

- `python3 skills/daystrom-dml/scripts/dml_memory.py ingest --text "..." --kind action`
- `python3 skills/daystrom-dml/scripts/dml_memory.py retrieve --query "..." --top-k 6 --ground-truth-policy low-confidence --ground-truth-mode hybrid --reform-memory --no-strict-ground-truth`
  - confidence-gated architecture (default):
    - compute `memory_confidence`
    - if low confidence, run sidecar ground-truth RAG (`query_database`)
    - reform memory by ingesting condensed ground-truth chunks back into DML
  - tune with:
    - `--ground-truth-policy low-confidence|always|never`
    - `--confidence-threshold 0.46`
    - `--reform-memory / --no-reform-memory`

Default profile:
- `openclaw-wrapper/config/dml_gpu_only.yaml`
- embeddings: `ollama:qwen3-embedding:0.6b`
- summarization/reform model: `llama3:8b`
- `sentence-transformers` may still be used in non-default alternate configs, but the production default stays Ollama-native.
- `--require-gpu` enabled by default (fails fast if CUDA/Ollama GPU path is not active)

Optional flags:
- `--storage-dir <path>` (parameterized runtime storage; do not assume a fixed home directory on target machines)
- `--tenant-id <id>` / `--client-id <id>` / `--session-id <id>` / `--instance-id <id>`
- `--meta '{"phase":"build","tool":"openclaw"}'`
- `ingest`: `--chunk/--no-chunk`, `--chunk-chars`, `--chunk-overlap`, `--filter-noise/--no-filter-noise`
- `retrieve`: `--query-expand/--no-query-expand` (expands blocker terms like USD/export/fallback)

## Integration contract
- Stable beta contract: `dml-agent-memory-v1`
- See `ADAPTER_CONTRACT.md` for the harness-facing command/metadata contract.
- Mutating commands use `$DML_STORE/.dml_store.lock`; use `--lock-timeout-ms` for multi-agent wait behavior.
- New writes default to `tenant_id=openclaw`; multi-user harnesses should pass `--tenant-id` explicitly.
- Mutating commands append compact events to `$DML_STORE/dml_audit.jsonl`; use `--audit-actor` and `audit-tail`.
- `health`: validates store checksum/count/dimensions and reports readiness.
- `backup`: creates checksum-manifested backups before risky operations.
- `verify`: loads durable state through the persistence layer and reports recovery suggestions.
- `schema`: reports state schema support and migration requirement.
- `report`: emits the compact operational dashboard payload.
- `restore`: validates a backup, creates a pre-restore backup, then restores atomically.
- `export` / `verify-export` / `import`: move a portable checksum-manifested
  `.dml-export.tar.gz` bundle between stores or machines.
- `ingest`: stores memory with kind + metadata.
- `retrieve`: returns JSON report including `raw_context` + items.
- `resume`: returns active continuity handoff and latest checkpoint fields.
- Claim conflicts are opt-in: pass `conflict_key`/`claim_value` in `--meta`
  for scoped facts that should not silently diverge. Conflicted memories remain
  retrievable and retrieval returns `conflict_count`, `conflicts`, and a
  `=== Memory Conflicts ===` context block.
- Use `conflicts` to list unresolved scoped claim groups, and
  `resolve-conflict --accept-value <value>` to accept one value and suppress
  competing values under the same tenant/client/session/instance/namespace key.
- Use `curate` as a dry-run first to find old low-fidelity memories. Add
  `--apply` only after review; active continuity memories are protected unless
  `--include-continuity` is set.
- Proven local baseline is GPU-first for current OpenClaw runtime work.
- Portable installability on Linux must remain path-parameterized and not assume `/home/nvidia/...`.

## Benchmark token savings + speed
Run in the durable GPU venv:

- `source /Users/markmckeen/.openclaw/daystrom-dml-v2/.venv-dml/bin/activate`
- `python3 /Users/markmckeen/.openclaw/daystrom-dml-v2/openclaw-wrapper/scripts/benchmark_openclaw_memory.py --memories 120 --top-k 6`

Outputs JSON with:
- baseline token estimate (naive full-context)
- average DML tokens returned
- average token savings %
- average and p95 retrieval latency (ms)
- retrieval quality proxies: `avg_precision_at_k`, `avg_ndcg_at_k`, and `avg_retrieval_noise_score`
- ingestion filtering stats (`corpus_filtering.raw_chunks/kept_chunks/dropped_chunks`)

## Recall quality eval
Run before beta-facing changes that touch ingest, retrieve, resume, scoping, or
compaction-continuity behavior:

- `python3 /Users/markmckeen/.openclaw/daystrom-dml-v2/openclaw-wrapper/scripts/recall_eval.py --output-dir /tmp/dml-recall-eval`

The eval writes a small isolated fixture store by default, then scores:

- tenant recall and cross-tenant isolation
- session isolation
- active continuity checkpoint resume
- ingest/retrieve/resume latency

It emits JSON to stdout plus `recall_eval_report.json` and
`recall_eval_report.md` in the chosen output directory. Add `--storage-dir` to
run against a specific disposable store, not the production memory store.

## Concurrency stress
Run this before beta releases or multi-agent harness changes:

- `python3 /Users/markmckeen/.openclaw/daystrom-dml-v2/openclaw-wrapper/scripts/stress_harness.py --writes 6 --workers 3 --tenants 2 --sessions 2`

The stress harness uses an isolated temporary store by default, launches
parallel `ingest` subprocesses, verifies durable marker persistence, runs
tenant/session retrieval isolation checks, and checks `verify` plus
`audit-tail`. Add `--storage-dir` only for a disposable test store; the harness
writes probe memories.

## Beta readiness gate
Run before shipping or switching a harness to a store:

- `python3 /Users/markmckeen/.openclaw/daystrom-dml-v2/openclaw-wrapper/scripts/beta_readiness.py --storage-dir "$DML_STORE" --tenant-id openclaw --output-dir /tmp/dml-beta-readiness`

The gate runs `health`, `verify`, `conflicts`, `audit-tail`, and the isolated
recall eval, then emits JSON plus optional Markdown. Use
`--skip-recall-eval` for a faster store-only preflight.

## Notes
- This is a substrate skill: keep usage deterministic and tool-friendly (JSON output only).
- Canonical runtime is the durable Daystrom env, not an ad-hoc workspace venv.
- Production path is Ollama-native for embeddings/summarization.
- `sentence-transformers` is supported as an optional alternate backend, but should not be treated as the default unless architecture is intentionally changed.
