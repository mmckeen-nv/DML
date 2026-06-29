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

## The Daystrom stack

### DPM evolution layer

The Daystrom Personality Matrix now includes a bounded evolution layer. It records interaction signals into a `dpm_evolution_graph.json` with fast-state and slow-self trait values, then renders context-adaptive personality guidance for creative, build/debug, reef-support, and general collaboration work. The layer is deliberately not "free will against the user": immutable hard laws keep current-turn instructions, safety, privacy, and secret hygiene above personality tendencies. See [`docs/daystrom-dpm-evolution-layer.md`](docs/daystrom-dpm-evolution-layer.md).


DML is the memory layer, but the repository also contains the surrounding Daystrom control surfaces:

| Layer | Role | What it does |
| --- | --- | --- |
| **DML — Daystrom Memory Lattice** | Memory substrate | Ingests, embeds, stores, retrieves, summarizes, resumes, verifies, backs up, exports, and curates durable memory. |
| **DPM — Daystrom Personality Matrix** | Preference/personality overlay | Maintains bounded relationship/project/personality context without turning memories into prompt bloat. |
| **DCN — Daystrom Cognition Network** | Cognitive control layer | Observes intent, emits cognitive packets, gates retrieval policy, captures feedback, evaluates readiness, and manages safe policy promotion. |
| **DIP — Daystrom Inference Preparation** | Inference boundary | Prepares compact frontier prompts from scoped memory; the calling harness owns the actual model call and secret handling. |

The boundaries are intentional. Memory, personality, cognition, and inference preparation are separate enough to test and govern, but integrated enough for agents to feel continuous.

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
dml_core/daystrom_dml/        Core lattice, adapter, server, provider, DPM, DCN, DIP, tests
openclaw-wrapper/             Stable JSON wrapper contract for agent harnesses
integrations/hermes/          Hermes/Citizen Snips memory-provider plugin
skills/                       OpenClaw-style skill and helper scripts
scripts/                      Utility, benchmark, import, and audit scripts
dml_mcp/                      MCP server entrypoint
examples/                     Demos, playgrounds, visualizers, chatbot, benchmark harnesses
docs/contracts/               Contract schemas and snapshots
docs/daystrom-operator-bible.md        Human-operator runbook for deployment, proof, hygiene, backup, promotion, and release
docs/daystrom-agentic-harness-bible.md Harness bible for Hermes/OpenClaw/Turnstone-style integration
docs/dpm-readonly-packet/     DPM lifecycle/spec packet
docs/dcn-operator-guide.md    DCN operator modes, gates, feedback, and eval smoke guidance
```

Important entrypoints:

- `DMLAdapter` — embed DML directly in Python agents.
- `openclaw-wrapper/scripts/dml_memory.py` — JSON-first command wrapper for automation.
- `dml-provider` / `dml serve` — local provider with UI and `/api/*` routes.
- `dml-ollama` — Ollama-shaped memory clone for tools that expect Ollama APIs.
- `dml` — Ollama-style client CLI for provider operations.
- `dml-mcp-server` — MCP integration surface.
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
