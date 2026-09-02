# Daystrom Memory Lattice (DML) --- ALPHA SOFTWARE. Use at your own risk!

**DML is a persistent memory and cognition substrate for long-horizon AI systems.**

It gives agents durable recall, semantic compression, active continuity, operator-grade validation, and clean integration points for local or remote inference. DML is designed for the class of assistants that need to carry context across sessions, machines, models, demos, and projects without dragging an ever-growing transcript through every prompt.

> RAG searches a pile of chunks. DML remembers, curates, resumes, and prepares context for action.

---

## Why this matters

Modern agents are powerful, but most of them still live turn-to-turn. They forget decisions, replay huge transcripts, pollute context with tool logs, and force users to say “remember when…” or “use the memory system…” before the agent behaves like a continuing collaborator.

DML changes that operating model.

DML provides:

- **Long-horizon continuity** — durable semantic memory, active handoffs, session resume, tenant/session scoping, and conflict-aware recall.
- **Hierarchical compression** — memory levels move from verbatim fragments toward increasingly compact summaries and abstractions.
- **Agent-safe writeback** — structured memory classes, metadata, hygiene filtering, transcript-pollution rejection, and audit trails.
- **Retrieval that fits the task** — semantic, literal, hybrid, active-continuity, and frontier-prompt preparation surfaces.
- **Pluggable inference** — Ollama, OpenAI-compatible endpoints, local Transformers, NIM/vLLM-style servers, and custom adapter paths.
- **Operational proof** — health, verify, report, backup, restore, export/import, recall evals, stress tests, beta readiness gates, and DCN eval smoke probes.
- **First-class agent integration** — Python adapter, JSON CLI wrapper, provider server, Ollama-compatible clone, MCP server, Hermes memory provider, and OpenClaw-style skills.

DML is the memory layer agents should have had from the start.

---

## The Daystrom memory architecture

DML now spans two related kinds of memory:

1. **Semantic and continuity memory** answers *what should the agent remember?*
2. **Active context and execution memory** answers *what exact context or model-native state should be resident for this generation?*

They share scope, identity, authority, policy, lifecycle, and evidence, but they are not interchangeable. A semantically relevant memory is not proof that its old KV state can be reused. Native reuse stops at the first token, position, model, runtime, topology, layout, or authority mismatch.

### Where it sits in the stack

```text
Hermes / OpenClaw / MCP / custom agent harness
                     │
                     ▼
        DCN — cognition and policy control
      decide what to retrieve, suppress, verify,
          prepare, learn, or write back
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
 DML semantic     DPM bounded     DCM active-context
 memory lattice   personality     and execution-memory
 and continuity   overlay         control plane
          └──────────┼──────────┘
                     ▼
       DIP — bounded inference preparation
                     │
                     ▼
       vLLM / SGLang / TensorRT-LLM / llama.cpp
                     │
                     ▼
     engine-native KV transfer / KVBM / NIXL
       / LMCache / Mooncake data planes
                     │
                     ▼
 GPU HBM ↔ pinned host RAM ↔ RDMA memory ↔ NVMe
                  ↔ remote file/object storage

 Companion specialization: DEC applies the same policy direction to
 MoE expert-weight prediction and residency, not conversation KV state.
```

Daystrom owns the differentiated control plane: identity, authority, policy, prediction, admission, placement decisions, leases, lifecycle, checkpoint lineage, audit, and evidence. Engines and transfer frameworks should own commodity block allocation and byte movement. Daystrom should not become a second NIXL, KVBM, LMCache, or Mooncake.

### The layers and their responsibilities

| Layer | Role | What it owns |
| --- | --- | --- |
| **DML — Daystrom Memory Lattice** | Durable semantic memory | Ingest, embedding, deduplication, salience/fidelity, abstraction, retrieval, conflict handling, persistence, backup/export/import, and compact continuity. |
| **DML1 hot context** | Small current semantic working set | Fast, compact STM-derived commitments, entities, decisions, and active state. |
| **DML2 exact pages** | Exact context-page cache | Digest-bound page handles and payloads used when exact evidence is required. |
| **Durable tier** | Long-horizon source of truth | Persistent memories and page-catalog lookup reached only when policy explicitly allows it. |
| **DCM — Daystrom Context Manager** | Active context and model-native execution memory | Scoped segments, authority ordering, token admission, manifests, working-set generations, memory faults, native profiles, checkpoint identity, restore/continuation/purge, and KV-fabric policy. |
| **DPM — Daystrom Personality Matrix** | Preference and relationship overlay | Bounded style, preference, project, and relationship context subordinate to current instructions and safety. |
| **DCN — Daystrom Cognition Network** | Cognitive policy layer | Intent observation, retrieval plans, cognitive packets, feedback, evaluation, governed policy promotion, and bounded turn-extension decisions. |
| **DIP — Daystrom Inference Preparation** | Model-call boundary | Produces compact frontier prompts; the harness still owns inference calls and provider secrets. |
| **DEC — Daystrom Expert Cache** | Companion expert-weight specialization | Profiles and predicts MoE expert demand, keeps static-hot experts resident, manages a bounded dynamic HBM pool, and measures exact misses. DEC is a companion repository, not a replacement for DCM. |

> **Naming note:** durable lattice entries also carry numeric abstraction levels (`level=0` for fresh/high-fidelity entries, increasing as memories decay or are summarized). Those L0–LK abstraction levels are different from the DML1 hot-context and DML2 exact-page runtime tiers.

### How a memory moves through the system

```text
meaningful event or document
  → hygiene and secret filtering
  → scoped ingest (tenant/client/session/instance/thread)
  → embedding + literal metadata + provenance
  → deduplication/conflict checks
  → optional agentic scratch → verified → durable promotion
  → fidelity decay and optional lineage-preserving abstraction
  → semantic/literal/hybrid retrieval
  → DCN policy and DPM overlay
  → bounded DIP/DCM context packet
  → model/tool turn
  → compact writeback or handoff
```

DML does not store every chat turn. It stores durable decisions, observations, plans, failures, preferences, constraints, and handoffs. Raw transcripts, tool dumps, prompt wrappers, and credentials are rejected or stripped because they make retrieval worse and create security risk.

At retrieval time, scope is checked first. The router can choose semantic, literal, or hybrid retrieval; exact handles can bypass fuzzy ranking when the caller already knows the required page. Returned material remains reference or untrusted evidence—it never silently becomes a system instruction.

### How active context and native KV reuse work

```text
scoped memories / exact pages / current instruction
  → authority-aware candidate set
  → token-budget admission
  → immutable ContextManifest + ContextPacket
  → deterministic working-set generation
  → compare with parent generation
  → stable exact prefix + changed suffix
  → restore compatible parent checkpoint
  → prefill only the changed suffix
  → publish child checkpoint after signed readiness
  → selectively purge and physically account for unshared rows
```

DCM distinguishes three separate identities: the logical packet/manifest, the durable execution-checkpoint record, and the runtime checkpoint used by a serving connector. They are cryptographically bound rather than substituted for one another.

The current vLLM 0.20 integration is GPU-first: local GPU Automatic Prefix Caching is checked normally, then an explicitly authorized Daystrom checkpoint may restore the remaining exact prefix from managed pinned CPU memory. Unapproved requests do not query or populate the managed cache. Save, status, restore, compound parent→child transition, and selective purge use short-lived HMAC-authenticated envelopes and payload-free evidence.

The KV-fabric contract extends that control model across GPU HBM, pinned host memory, RDMA-addressable memory, local SSD/NVMe, and remote file/object storage. Routes are deterministic and bounded; transfer tickets are plan-bound, authority-bound, expiring, HMAC-authenticated, and single-use. This layer currently plans and authorizes movement—it does **not** claim that RDMA, GDS, NIXL, KVBM, LMCache, Mooncake, or cloud movement is already wired.

### What has been built

| Area | Current state |
| --- | --- |
| Durable lattice | Implemented: persisted memory items, embeddings, literal/semantic/hybrid retrieval, salience/fidelity decay, abstraction lineage, deduplication, conflicts, curation, audit, backup, verification, export/import. |
| Agent continuity | Implemented: resume, compact handoff, scoped retrieval/writeback, active continuity, hygiene filters, provider and JSON wrapper contracts. |
| Harness integrations | Implemented: Python adapter, CLI/wrapper, HTTP provider/UI, Ollama-compatible server, MCP server, Hermes provider plugin, OpenClaw-style skill. |
| DPM/DCN/DIP | Implemented bounded DPM overlays and evolution graph; deterministic DCN observation/planning/feedback/promotion gates; DIP frontier-prompt preparation. |
| DCM logical context | Implemented: authority-aware segments, runtime capability discovery, admission budgets, manifests, exact pages, working-set transitions, DML1→DML2→durable memory faults, leases, and payload-free plans. |
| DCM native state | Implemented and live-proven for the version-pinned vLLM cooperative path: signed checkpoint save/readiness/restore/continuation and selective physical purge. llama.cpp lifecycle probing also exists behind capability checks. |
| KV fabric | Implemented as a validated engine-neutral control-plane contract for heterogeneous tiers, compatibility negotiation, bounded routing, and authenticated transfer authorization. Physical KVBM/NIXL/LMCache/Mooncake adapters remain future work. |
| Deployment | Implemented cooperative-vLLM `init → doctor → pull → preflight → deploy → verify → status/rollback/logs`, with atomic config/key handling, health waits, canary completion, lock recovery, bounded backups, and transactional rollback. |
| DEC expert residency | Hardware-backed companion work retained separately. It specializes in MoE expert-weight prediction and placement and belongs beside DCM under the broader Daystrom policy plane. |

### Measured evidence—not marketing estimates

The repository keeps deterministic and live-runtime evidence separate.

**Current deterministic DCM workload smoke.** On the checked-in seven-case `extended` suite with a 180-token context budget, rerun locally with `--offline`:

| Strategy | Answer fidelity | Retrieval recall | Explicit miss rate | Mean admitted tokens | Budget overflow rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full context | 85.7% | n/a | 0.0% | 404.0 | 100.0% |
| Ordinary lexical RAG, top-1 | 57.1% | 64.3% | 42.9% | 119.9 | 0.0% |
| DCM bounded working set | **85.7%** | **92.9%** | **14.3%** | **145.4** | **0.0%** |

In this bounded synthetic regression slice, DCM matched full-context answer fidelity while sending **64.0% fewer context tokens** and eliminating the full-context budget overflow. Against ordinary lexical RAG it gained **28.6 fidelity points**, gained **28.6 recall points**, and reduced the explicit-miss rate by **66.7%**. DCM used somewhat more context than top-1 RAG because it admitted enough exact/multi-item evidence to recover cases that top-1 missed.

These are seven deterministic synthetic cases using an evidence model and lexical stand-in, not a broad claim about model intelligence or production latency. The aggregate evidence is checked in at [`docs/artifacts/dcm-workload-offline-extended-2026-09-02.json`](docs/artifacts/dcm-workload-offline-extended-2026-09-02.json). Reproduce it with:

```bash
PYTHONPATH=dml_core python dml_core/scripts/dcm_workload_benchmark.py \
  --offline --suite extended --output-json /tmp/dcm-workload.json
```

**Fresh 120-turn compaction/frontier-preparation smoke.** A local synthetic long-run generated 6,849 estimated direct-input tokens. DML recovered all four required continuity anchors into 402 context tokens and produced a 653-token frontier-verification prompt: **90.5% fewer estimated input tokens** than sending the direct history, with **100% anchor recall** and **1.19 ms** DML retrieval time. The local draft recovered two of four anchors, which is why the architecture sends the compact DML evidence alongside a draft rather than trusting draft compression alone.

This smoke used 120 synthetic turns, a local store with FAISS enabled, deterministic DML summarization, and a local `gemma3:4b` draft. Token counts are estimator outputs, not provider billing; no frontier model was called, so modeled output-token or cost projections are deliberately not presented as measured savings. The payload-free summary is checked in at [`docs/artifacts/dml-frontier-compression-smoke-2026-09-02.json`](docs/artifacts/dml-frontier-compression-smoke-2026-09-02.json). Reproduce the preparation path with:

```bash
python scripts/frontier_compression_smoke.py \
  --turns 120 --output-dir /tmp/dml-frontier-compression-smoke
```

**Live native-KV evidence.** On the version-pinned Nemotron/vLLM 0.20 path:

- A growing **9,300 → 14,000 → 18,500-token** chain restored **8,448**, then **12,672**, native CPU KV tokens while GPU APC was deliberately zeroed for route isolation.
- At 18,500 prompt tokens, the managed transition completed in **436.0 ms** versus **1,169.1 ms** for cold full recomputation: **62.7% faster**, a **2.68×** cold/managed latency ratio, with equal deterministic output digests.
- Child readiness reached **24/24 physical rows**, followed by complete selective cleanup of **24 rows / 415,236,096 bytes**.
- The live endpoint accepted prompt plus output at exactly **65,536 tokens** and rejected **65,538**. KV checkpointing reduces repeated prefill work; it does not extend that logical serving limit.

See [`docs/vllm-context-exhaustion-ab.md`](docs/vllm-context-exhaustion-ab.md), [`docs/dcm-native-context-transition.md`](docs/dcm-native-context-transition.md), and the digest-only artifacts in [`docs/artifacts/`](docs/artifacts/).

### Compaction and context-pressure mitigation

DML/DCM mitigate compaction in three different ways:

1. **Before pressure:** retrieval and token-budget admission construct a bounded working set instead of replaying the transcript. The current deterministic smoke reduced mean full-context admission from 404.0 to 145.4 tokens while retaining the same measured answer fidelity.
2. **At a generation boundary:** DCM computes the exact stable prefix and changed suffix. Compatible native state can be restored and only the changed suffix prefetched; any positional or digest divergence invalidates reuse from that point.
3. **Across actual harness compaction:** the harness writes a compact handoff before compaction and resumes scoped continuity afterward. Durable memory therefore survives transcript compression without pretending that a summarized prompt preserves old KV identity.

DML does not disable a host framework's compactor and it does not make an over-limit request legal. It reduces how often raw transcript growth forces compaction, preserves the durable facts compaction would otherwise lose, and provides exact boundaries for safe native reuse after the next bounded context is assembled.

### DPM evolution layer

The Daystrom Personality Matrix includes a bounded evolution layer. It records interaction signals into `dpm_evolution_graph.json` with fast-state and slow-self trait values, then renders context-adaptive guidance for creative, build/debug, reef-support, and general collaboration work. It is deliberately not "free will against the user": immutable hard laws keep current-turn instructions, safety, privacy, and secret hygiene above personality tendencies. See [`docs/daystrom-dpm-evolution-layer.md`](docs/daystrom-dpm-evolution-layer.md).

The boundaries are intentional. Memory, personality, cognition, inference preparation, context management, and physical movement are separate enough to test and govern, but integrated enough for agents to feel continuous.

---

## Operator and harness bibles

If you are deploying DML or wiring it into an agent, start here:

| Guide | Audience | Use it for |
| --- | --- | --- |
| [`docs/daystrom-operator-bible.md`](docs/daystrom-operator-bible.md) | Human operators, demo owners, platform maintainers | Store layout, endpoint setup, health/verify/report, backup/export/restore, hygiene stoplines, Hermes config, DCN promotion, troubleshooting, release checklist. |
| [`docs/daystrom-agentic-harness-bible.md`](docs/daystrom-agentic-harness-bible.md) | Hermes, OpenClaw, Turnstone, MCP hosts, custom autonomous harnesses | Required lifecycle calls, wrapper contract, retrieval/writeback/handoff loop, cognition-gated turn extension, DPM/DCN/DIP boundaries, harness pseudocode, validation bundle. |
| [`AGENT_README_TO_OMNOM.md`](AGENT_README_TO_OMNOM.md) | Agent onboarding and installer scripts | Compact agent-facing integration guide and endpoint wizard. |

The short rule: DML should be wired into the agent loop as the default memory and cognition substrate. It should not be a user-invoked afterthought and it should never store raw transcript sludge, tool logs, prompt wrappers, or secrets.

---

## What is in this repository

```text
dml_core/daystrom_dml/        Core durable lattice, adapters, provider, DPM, DCN, DIP
dml_core/daystrom_dml/context/ DCM contracts, pages, working sets, checkpoints, runtime adapters, KV fabric
deploy/vllm-cooperative/      Version-pinned cooperative-vLLM deployment and rollback workflow
openclaw-wrapper/             Stable JSON wrapper contract for agent harnesses
integrations/hermes/          Hermes/Citizen Snips memory-provider plugin
skills/                       OpenClaw-style skill and helper scripts
scripts/                      Utility, benchmark, import, and audit scripts
dml_mcp/                      MCP server entrypoint
examples/                     Demos, playgrounds, visualizers, chatbot, benchmark harnesses
docs/contracts/               Contract schemas and snapshots
docs/artifacts/               Digest-only live-runtime evidence and deterministic renderers
docs/daystrom-operator-bible.md        Human-operator runbook for deployment, proof, hygiene, backup, promotion, and release
docs/daystrom-agentic-harness-bible.md Harness bible for Hermes/OpenClaw/Turnstone-style integration
docs/dpm-readonly-packet/     DPM lifecycle/spec packet
docs/dcn-operator-guide.md    DCN operator modes, gates, feedback, and eval smoke guidance
docs/vllm-cooperative-kv-connector.md  Native KV connector behavior and trust boundary
docs/dcm-native-context-transition.md  Logical-generation to native-checkpoint bridge
```

Important entrypoints:

- `DMLAdapter` — embed DML directly in Python agents.
- `openclaw-wrapper/scripts/dml_memory.py` — JSON-first command wrapper for automation.
- `dml-provider` / `dml serve` — local provider with UI and `/api/*` routes.
- `dml-ollama` — Ollama-shaped memory clone for tools that expect Ollama APIs.
- `dml` — Ollama-style client CLI for provider operations.
- `dml-mcp-server` — MCP integration surface.
- `dcm-workload-benchmark` — digest-only comparison of full context, ordinary RAG, truncation, summarization, and DCM working-set hydration.
- `deploy/vllm-cooperative/deploy.sh` — safe cooperative-vLLM init, doctor, pull, preflight, deploy, verification, status, logs, and rollback workflow.
- `integrations/hermes/plugins/daystrom_dml` — Hermes memory provider with DML/DPM/DCN integration.
- [`AGENT_README_TO_OMNOM.md`](AGENT_README_TO_OMNOM.md) — agent-facing integration playbook and endpoint wizard.

---

## Install

### Development install

```bash
git clone https://github.com/mmckeen-nv/DML.git
cd DML
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[server,embeddings,faiss,mcp,playground,dev]'
```

Minimal install:

```bash
pip install -e .
```

Useful extras:

| Extra | Adds |
| --- | --- |
| `server` | FastAPI/uvicorn provider and HTTP surfaces |
| `embeddings` | SentenceTransformer embedding backend |
| `faiss` | FAISS vector store support |
| `mcp` | MCP server dependencies |
| `playground` | Streamlit/Plotly visualizer |
| `dev` | pytest/ruff/mypy test tooling |

### Default runtime posture

The production default is Ollama-native:

```yaml
llm_backend: ollama
model_name: llama3:8b
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: null  # Ollama owns placement; set cuda for strict GPU contract checks
rag_store:
  enable: true
  backend: faiss
  dim: 1024
persistence:
  enable: true
```

SentenceTransformers remain supported for alternate experiments and compatibility, but Ollama-native embeddings/summarization are the clean default for the current Daystrom runtime.

### DCM workload benchmark

Run the deterministic offline contract and strategy smoke:

```bash
dcm-workload-benchmark --offline --suite extended --output-json /tmp/dcm-workload-offline.json
```

Run the same sanitized synthetic workload against an explicitly approved local
OpenAI-compatible endpoint:

```bash
dcm-workload-benchmark \
  --allow-network \
  --endpoint-url http://127.0.0.1:11434/v1/chat/completions \
  --model llama3:8b \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-base-url http://127.0.0.1:11434 \
  --request-extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --suite extended \
  --output-json /tmp/dcm-workload-llama3-8b.json
```

Artifacts contain message, answer, and completion digests—not prompt or response
text. Reported metrics include answer fidelity, explicit/lookup misses, required-page
retrieval recall, rendered-message budget overflow, admitted tokens, lookup and total
latency, and serialized resident-context bytes. Prefill time is reported only when
the endpoint exposes it; direct runtime/KV validation remains a separate benchmark.
The default `regression` suite preserves the original three-case slice. `stress`
adds no-handle paraphrase, contradiction, true multi-hop, and long-horizon cases;
`extended` runs both. These are regression slices, not claims of broad model-quality
superiority. DCM uses an authorized exact page handle when available. Without
`--embedding-model`, no-handle cases use a bounded deterministic lexical stand-in.
With `--embedding-model`, synthetic pages are embedded through the configured
Ollama HTTP endpoint and ranked through the production `DMLSemanticPageCatalog`
adapter; the generation endpoint remains OpenAI-compatible and can be Ollama,
vLLM, or another provider. `--request-extra-json` supplies bounded, non-secret
provider/runtime options such as vLLM chat-template kwargs. It cannot replace core
benchmark fields (`model`, `messages`, sampling, token limit, or streaming), and
artifacts retain only its top-level keys and canonical SHA-256 digest—not values.
Ordinary RAG remains lexical top-k without handles,
and the fixed lossy-summary baseline excludes summary-generation cost. Candidate
depth is an explicit benchmark parameter rather than a claimed universal production
default. If the final rendered-message estimate exceeds the configured benchmark
budget, no-handle DCM retries with fewer candidates and records the requested/used
depth plus `budget_constrained`; it never sends an overflowing DCM prompt. Final
successful lookup time is reported as `lookup_ms`, while `retrieval_total_ms` also
includes embedding setup and rejected budget-retry attempts. Context-construction
failures are recorded per case as digest-only errors instead of aborting the suite.
Those differences are recorded in every artifact.

---

## Quick start: Python adapter

```python
from daystrom_dml.dml_adapter import DMLAdapter

adapter = DMLAdapter(
    config_overrides={
        "storage_dir": "./data/dml",
        "llm_backend": "ollama",
        "model_name": "llama3:8b",
        "embedding_model": "ollama:qwen3-embedding:0.6b",
        "dml.agentic_mode.enabled": True,
    }
)

adapter.ingest_agentic(
    "The launch review found a recurring timeout in the deployment verifier.",
    kind="observation",
    meta={"source": "launch-review", "phase": "debug"},
)

context = adapter.build_preamble("What do we know about deployment verifier timeouts?")
print(context)
```

---

## Quick start: provider server

```bash
dml-provider --storage-dir ./data/dml --host 127.0.0.1 --port 8765
```

Health:

```bash
curl http://127.0.0.1:8765/health
```

Provider mode serves:

- UI at `http://127.0.0.1:8765`
- memory APIs under `/api/*`
- DCN operator probes under `/api/dcn/*`
- DIP prompt preparation at `/api/frontier/prepare`
- Ollama-shaped endpoints such as `/api/tags`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/api/ps`, and `/api/version`

Ollama-compatible clone:

```bash
dml-ollama --storage-dir ./data/dml --host 127.0.0.1 --port 11435
curl http://127.0.0.1:11435/api/tags
```

Client CLI:

```bash
dml status
dml remember --text "The active branch is provider-hardening." --meta '{"source":"cli"}'
dml recall --query "active branch" --context-only
dml search --query "provider"
dml fetch 1
```

---

## Quick start: agent wrapper contract

The stable automation surface is `dml-agent-memory-v1` in `openclaw-wrapper/scripts/dml_memory.py`. It emits JSON and is safe for harnesses to parse.

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir ./data/dml \
  --config-path dml_core/daystrom_dml/config.yaml \
  --no-require-gpu \
  health
```

Ingest:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir ./data/dml \
  --no-require-gpu \
  ingest \
  --kind action \
  --text "Fixed timeout handling in deployment verifier." \
  --meta '{"source":"agent","phase":"implementation"}'
```

Retrieve:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir ./data/dml \
  --no-require-gpu \
  retrieve \
  --query "deployment verifier timeout" \
  --top-k 6 \
  --ground-truth-policy low-confidence \
  --reform-memory
```

Resume/handoff:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml resume --session-id demo-session

python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir ./data/dml \
  handoff \
  --thread demo-thread \
  --state "Verifier timeout root cause isolated." \
  --task "Finish fix and run tests." \
  --next-action "Patch retry budget and rerun smoke." \
  --session-id demo-session
```

See [`openclaw-wrapper/ADAPTER_CONTRACT.md`](openclaw-wrapper/ADAPTER_CONTRACT.md) for the full JSON contract.

---

## Inference endpoint wizard

DML separates **memory** from **model execution**, which means you can swap embedding and summarization/inference endpoints without rewriting the agent.

Use this wizard when setting up a deployment:

### 1. Choose your embedding backend

| Choice | Use when | Config |
| --- | --- | --- |
| Ollama local GPU-managed | Best default for private local installs | `embedding_model: ollama:qwen3-embedding:0.6b` |
| SentenceTransformer local | Offline Python-only experiments | `embedding_model: sentence-transformers/all-MiniLM-L6-v2` |
| Custom adapter | You own embedding service calls | Implement/route through adapter code |

Ollama example:

```bash
ollama pull qwen3-embedding:0.6b
```

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda   # strict DML contract; Ollama still owns actual placement
rag_store:
  dim: 1024              # keep consistent with your persisted store
```

SentenceTransformer example:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_device: cuda   # or cpu / mps / cuda:0
rag_store:
  dim: 384
```

**Important:** changing embedding dimensions requires a new store or an embedding migration. Do not point a 384-dim index and a 1024/1536-dim model at the same live store without migration.

### 2. Choose summarization / reform backend

| Choice | Use when | Config |
| --- | --- | --- |
| Ollama | Local, private, easy default | `llm_backend: ollama`, `model_name: llama3:8b` |
| OpenAI-compatible | vLLM, LM Studio, OpenAI, Azure, NIM-compatible servers | `DML_API_BASE`, `DML_API_KEY`, `DML_MODEL_NAME` |
| Transformers | Offline Python model path | `llm_backend: transformers`, `model_name: <hf-model>` |

Ollama example:

```bash
ollama pull llama3:8b
```

```yaml
llm_backend: ollama
model_name: llama3:8b
strict_llm_required: true
```

OpenAI-compatible endpoint example:

```bash
export DML_API_BASE=http://127.0.0.1:8000/v1
export DML_API_KEY="***"  # optional; set locally, never commit
export DML_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

```yaml
llm_backend: openai
model_name: ${DML_MODEL_NAME}
```

### 3. Choose storage scope

| Scenario | Recommended storage |
| --- | --- |
| Local dev | `./data/dml` |
| One user, many sessions | one durable store + tenant/session IDs |
| Demo machine | profile-local store under the agent profile |
| Multi-tenant service | explicit `tenant_id`, `client_id`, `session_id`, and backup policy |

### 4. Prove it before trusting it

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml --no-require-gpu health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml --no-require-gpu backend-proof
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml --no-require-gpu ingest --kind observation --text "DML install probe"
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml --no-require-gpu retrieve --query "DML install probe" --top-k 3
```

For strict GPU/Ollama-managed installs, omit `--no-require-gpu` after your config is correct:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml backend-proof
```

---

## Hermes / Citizen Snips integration

The Hermes plugin lives in `integrations/hermes/plugins/daystrom_dml/`.

Current posture:

- `memory.provider: daystrom_dml`
- DPM/personality overlay is bounded and current-turn subordinate.
- `retrieval_policy: always` makes DML part of normal core operations.
- `retrieval_policy: heuristic` is the explicit opt-out for older gated behavior.
- `retrieval_policy: never` disables retrieval while allowing the rest of the provider shape to remain explicit.
- Writeback hygiene rejects raw transcripts, tool logs, gateway wrappers, DML prompt blocks, and credential-shaped fields.
- DCN can observe or actively gate retrieval decisions while staying inside governed promotion boundaries.

Focused checks:

```bash
python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
python -m py_compile integrations/hermes/plugins/daystrom_dml/__init__.py
```

See [`integrations/hermes/README.md`](integrations/hermes/README.md) for profile install notes.

---

## DCN operator surface

DCN is the deterministic control layer around memory policy and cognitive packets. It does not own DML storage, DPM personality state, or frontier inference.

```bash
dml dcn observe --text "continue the DML work" --session-id abc
dml dcn packet --text "continue the DML work" --session-id abc
dml dcn feedback --decision-id <decision-id> --outcome verified --signals '{"tests_passed":true}'
dml dcn audit-tail --limit 20
dml dcn policy show
dml dcn policy export --output dcn-policy.json --snapshot-only
dml dcn policy import --input dcn-policy.json
dml dcn policy checkpoint --label before-active-learn
dml dcn policy rollback --checkpoint-id <checkpoint-id>
dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only
```

Read [`docs/dcn-operator-guide.md`](docs/dcn-operator-guide.md) before promoting active-learn behavior.

---

## Validation and operations

Core checks:

```bash
python -m pytest openclaw-wrapper/tests/test_dml_memory.py -q
python -m pytest dml_core/daystrom_dml/tests/test_dml.py -q
python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
```

Readiness gates:

```bash
python openclaw-wrapper/scripts/recall_eval.py --output-dir /tmp/dml-recall-eval
python openclaw-wrapper/scripts/stress_harness.py --writes 6 --workers 3 --tenants 2 --sessions 2
python openclaw-wrapper/scripts/beta_readiness.py --storage-dir ./data/dml --tenant-id openclaw --output-dir /tmp/dml-beta-readiness
```

Store operations:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml verify
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml backup --label before-maintenance
python openclaw-wrapper/scripts/dml_memory.py --storage-dir ./data/dml export --output-dir /tmp/dml-exports --label machine-move
```

---

## Design principles

1. **Continuity beats transcript replay.** Store compact semantic state and retrieve what matters.
2. **Memory must be hygienic.** No raw secrets, no tool-log dumps, no role-prefixed transcript sludge.
3. **Inference is pluggable.** DML prepares and validates context; the deployment chooses the model endpoint.
4. **Operators need proof.** Every serious path needs health, verify, eval, audit, backup, and restore surfaces.
5. **Defaults should work.** DML retrieval, persistence, RAG sidecars, background processing, and runtime features are enabled by default unless a constrained path has a specific reason to opt out.

---

## License

DML is available under the **Apache License 2.0**.

The Apache License 2.0 permits use, study, modification, distribution,
technical demonstrations, research, nonprofit use, and commercial use,
provided the license terms are followed.

You must preserve the copyright notice, license notice, and NOTICE file in
redistributions that include a NOTICE file, including attribution to Mark Mckeen.

Apache 2.0 is permissive: it does not impose copyleft or network-service
source-sharing obligations. Organizations that want separate written commercial
or alternative licensing terms may contact the copyright holder.

See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

### Attribution

Daystrom Memory Lattice (DML) was created by Mark Mckeen. Redistributed copies
or substantial portions of DML should preserve this NOTICE file and the
copyright/license notices included with the project.

---

## The headline

DML turns memory from an afterthought into infrastructure. It gives agents a durable substrate for recall, continuity, compression, personality overlays, cognitive control, and inference preparation — with the operational surfaces needed to prove that it is working.
