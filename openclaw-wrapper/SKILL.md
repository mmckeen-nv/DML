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

## Agent usage policy
- Use DML silently as memory substrate. Do not tell the user "according to DML" unless they ask how memory was used.
- Call `resume` at the start of a session, after compaction, and before continuing a long-running task.
- Call `retrieve` before a turn that depends on prior decisions, active files, blockers, test results, or user preferences.
- Call `handoff` before compaction, shutdown, model handoff, or any long pause.
- Call `ingest` after durable facts: decisions, accepted plans, changed files, commands run, failures, fixes, tests, external constraints, and next actions.
- Keep `tenant_id=openclaw` for a single local user. Pass a unique `--session-id` for concurrent sessions; omit `--session-id` only for intentional tenant-wide recall.
- Prefer compact, factual memories with stable anchors. Avoid storing raw secrets, full logs, transient speculation, or noisy terminal output.

## Long-horizon continuity loop
Use this loop for compaction survival and multi-hour agent runs:

1. Start or reuse a session id:
   - `python3 skills/daystrom-dml/scripts/dml_memory.py session --label "<project-or-thread>"`
2. Resume:
   - `python3 skills/daystrom-dml/scripts/dml_memory.py resume --session-id "$DML_SESSION_ID" --no-require-gpu`
3. Retrieve before important inference:
   - `python3 skills/daystrom-dml/scripts/dml_memory.py retrieve --query "<current task>" --session-id "$DML_SESSION_ID" --top-k 6 --ground-truth-policy low-confidence --no-reform-memory --no-require-gpu`
4. Store durable state:
   - `python3 skills/daystrom-dml/scripts/dml_memory.py ingest --text "<fact>" --kind action --session-id "$DML_SESSION_ID" --meta '{"source":"openclaw","phase":"execute"}' --no-require-gpu`
5. Write a survival checkpoint:
   - `python3 skills/daystrom-dml/scripts/dml_memory.py handoff --thread "<thread>" --state "<current state>" --task "<task>" --next-action "<next action>" --session-id "$DML_SESSION_ID" --no-require-gpu`

For extreme long-horizon runs, create a handoff whenever the active plan, file set, blocker, or test status changes. That keeps late-session facts durable even when the LLM context compacts.

## Frontier inference pipeline
DML can act as a prompt-preparation layer in front of a frontier model. The skill should prepare a compact prompt from scoped memory, then the harness or user-approved endpoint performs inference.

Start the local provider:

- `dml-provider --storage-dir "$DML_STORE" --host 127.0.0.1 --port 8765`

Prepare a DML-assisted frontier prompt:

- `python3 skills/daystrom-dml/scripts/dml_frontier_prepare.py --prompt-file task.md --session-id "$DML_SESSION_ID" --top-k 8 --frontier-max-tokens 1200`

Useful output modes:

- `--frontier-prompt-only`: print only the prompt to send to the frontier model.
- `--telemetry-only`: print compact token/latency/retrieval telemetry.

Inference pipeline policy:

- Use `/api/frontier/prepare` or `dml_frontier_prepare.py` for preparation only. Do not embed API keys in skill files, command examples, memory, or committed config.
- The prepared prompt should include only retrieved memory context, the current task, and any optional local draft. It should not include demo transcripts unless the current task explicitly asks for a demo.
- Compare estimates honestly: direct input tokens are a baseline estimate supplied by the harness; DML savings are meaningful only when the baseline represents real omitted transcript/context.
- Store frontier results back into DML only when they become durable decisions, code changes, tests, blockers, or next actions.

## Integration contract
- Stable beta contract: `dml-agent-memory-v1`
- See `ADAPTER_CONTRACT.md` for the harness-facing command/metadata contract.
- Mutating commands use `$DML_STORE/.dml_store.lock`; default wait is 30000ms,
  and `--lock-timeout-ms` can tune multi-session wait behavior.
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
- `session`: creates/reuses a stable session id for a local OpenClaw thread.
- `handoff`: writes a structured active continuity checkpoint for compaction or shutdown recovery.
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
- Single-user multi-session contract: keep `tenant_id=openclaw`, pass a unique
  `--session-id` per concurrent OpenClaw session for session-local recall, and
  omit `--session-id` only for intentional tenant-wide recall.
- Continuity checkpoints should carry `updated_at` or `captured_at`; `resume`
  selects the newest active checkpoint after retrieval, not the first retrieval
  hit.
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
Use `--tenants 1 --sessions 4` to smoke one OpenClaw user with multiple
concurrent sessions.

## Resume quality smoke
Run after continuity changes:

- `python3 /Users/markmckeen/.openclaw/daystrom-dml-v2/openclaw-wrapper/scripts/resume_quality_smoke.py --sessions 3`

The smoke writes several session handoffs to an isolated store, then verifies
tenant-wide `resume` selects the newest active checkpoint.

## Provider and worker
- Provider UI/API: `dml serve --storage-dir "$DML_STORE" --host 127.0.0.1 --port 8765`
- Ollama-compatible memory clone: `dml-ollama --storage-dir "$DML_STORE" --host 127.0.0.1 --port 11435`
- CLI client: `dml status`, `dml remember --text "..."`, `dml recall --query "..." --context-only`
- Frontier prompt preparation: `python3 skills/daystrom-dml/scripts/dml_frontier_prepare.py --prompt "..." --telemetry-only`
- Agent profile installer: `scripts/install_daystrom_dml.sh --profile openclaw` or `--profile hermes`
- Background queue worker: `python3 skills/daystrom-dml/scripts/dml_background_worker.py --once`
- Installer: `scripts/install_daystrom_dml.sh`

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
