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
- `health`: validates store checksum/count/dimensions and reports readiness.
- `backup`: creates checksum-manifested backups before risky operations.
- `verify`: loads durable state through the persistence layer and reports recovery suggestions.
- `restore`: validates a backup, creates a pre-restore backup, then restores atomically.
- `ingest`: stores memory with kind + metadata.
- `retrieve`: returns JSON report including `raw_context` + items.
- `resume`: returns active continuity handoff and latest checkpoint fields.
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

## Notes
- This is a substrate skill: keep usage deterministic and tool-friendly (JSON output only).
- Canonical runtime is the durable Daystrom env, not an ad-hoc workspace venv.
- Production path is Ollama-native for embeddings/summarization.
- `sentence-transformers` is supported as an optional alternate backend, but should not be treated as the default unless architecture is intentionally changed.
