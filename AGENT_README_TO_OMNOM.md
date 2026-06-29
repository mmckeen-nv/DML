# AGENT_README_TO_OMNOM.md

**Read this before integrating Daystrom DML into an agent.**

This guide is for agents, harnesses, coding assistants, demo operators, and installer scripts that need to wire DML into a real system. DML is not a sidecar trick; treat it as the memory and cognition substrate for the agent.

For the full operator-grade manuals, read these first:

- [`docs/daystrom-operator-bible.md`](docs/daystrom-operator-bible.md) — human operator deployment, validation, hygiene, backup/restore, DCN promotion, and release runbook.
- [`docs/daystrom-agentic-harness-bible.md`](docs/daystrom-agentic-harness-bible.md) — canonical harness implementation guide for Hermes, OpenClaw, Turnstone, MCP hosts, and custom autonomous agents.

---

## Prime directive

Integrate DML as a **default operational layer**, not as a feature the user must remember to invoke.

A DML-enabled agent should:

1. Retrieve relevant DML context before decisions that depend on prior state.
2. Write compact durable facts after decisions, fixes, validations, blockers, and handoffs.
3. Resume from active continuity when a session starts, compacts, restarts, or crosses machines.
4. Keep secrets, raw transcripts, tool logs, and prompt scaffolding out of memory.
5. Provide the user with a smooth **endpoint wizard** when embedding/summarization/inference choices are not already configured.
6. State clearly when DML is degraded or unavailable instead of pretending memory was used.

---

## What DML is in

DML is the memory layer inside the broader Daystrom stack:

- **DML — Daystrom Memory Lattice:** persistent semantic memory, active continuity, summaries, retrieval, backup/restore, export/import, evals, stress tests.
- **DPM — Daystrom Personality Matrix:** compact preference/personality overlays that are bounded and subordinate to current user instructions.
- **DCN — Daystrom Cognition Network:** deterministic cognition-control layer for retrieval policy, cognitive packets, feedback, eval smoke, and governed promotion.
- **DIP — Daystrom Inference Preparation:** memory-scoped prompt preparation for frontier/base model calls. DIP prepares; your harness calls the model.

Repository surfaces:

```text
dml_core/daystrom_dml/        Core Python package: adapter, server, provider, DPM, DCN, DIP
openclaw-wrapper/             Stable JSON command contract for agents
integrations/hermes/          Hermes memory-provider plugin
skills/                       OpenClaw-style skill helpers
dml_mcp/                      MCP entrypoint
examples/                     Demos, visualizer, chatbot, benchmark harnesses
docs/                         Contracts, DCN guide, DPM packet/specs
```

---

## Integration decision tree

### Python agent

Use `DMLAdapter` directly when you own the Python runtime.

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
    "Deployment verifier timed out during smoke validation.",
    kind="observation",
    meta={"source": "agent", "phase": "debug"},
)

context = adapter.build_preamble("What do we know about verifier timeouts?")
```

### Tool-calling agent or external harness

Use the JSON wrapper when you need deterministic stdout, shared-store locks, audit records, and shell/process orchestration.

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path "$DML_CONFIG" \
  --no-require-gpu \
  health
```

### Local provider/API

Run provider mode:

```bash
dml-provider --storage-dir "$DML_STORE" --host 127.0.0.1 --port 8765
```

Then use:

- `GET /health`
- `/api/*` memory routes
- `/api/frontier/prepare` for DIP prompt preparation
- `/api/dcn/*` for DCN observations, packets, feedback, policy, and eval smoke
- Ollama-shaped endpoints such as `/api/chat`, `/api/embed`, `/api/embeddings`, and `/api/tags`

### Hermes integration

Install/copy `integrations/hermes/plugins/daystrom_dml/` into the Hermes profile and configure:

```yaml
memory:
  provider: daystrom_dml
  daystrom_dml:
    integration_dir: /path/to/integrations/daystrom-dml
    storage_dir: /path/to/dml-store
    retrieval_policy: always
    enable_memory: true
    enable_personality: true
    sync_turns: true
    timeout_seconds: 8
```

Expected behavior:

- `retrieval_policy: always` means normal turns can retrieve DML without the user saying “use DML”.
- `retrieval_policy: heuristic` is the opt-out for old explicit-recall-only behavior.
- `retrieval_policy: never` disables DML retrieval.
- The provider must redact secrets and reject transcript/tool-log pollution.

Run:

```bash
python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
```

### MCP integration

Run the MCP server entrypoint and expose narrow DML memory tools to the host agent:

```bash
dml-mcp-server
```

Recommended MCP surface: health, remember, recall, resume, handoff, report, and frontier-prepare.

---

## Required agent behavior

### Before meaningful work

1. Check DML health.
2. Resume active continuity if this is a resumed session or long-running task.
3. Retrieve scoped memory for the current task.
4. If DML fails, report `DML degraded/unavailable` and continue with caveats.

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" resume --session-id "$DML_SESSION_ID"
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" retrieve --query "$TASK" --session-id "$DML_SESSION_ID" --top-k 6
```

### After durable work

Ingest compact semantic memory, not the whole transcript.

Good:

```text
Decision: Use Ollama qwen3-embedding:0.6b for the default embedding path.
Validation: smoke_dcn.py and test_dml_memory.py passed on PR #92.
Next action: deploy updated DML integration to the demo profile.
```

Bad:

```text
User: ...
Assistant: ...
<tool output> huge log dump ...
```

Write memory:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  ingest \
  --kind action \
  --text "Decision: Use Ollama qwen3-embedding:0.6b for default embedding path." \
  --meta '{"source":"agent","phase":"configuration"}'
```

### Before compaction, shutdown, handoff, or model switch

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  handoff \
  --thread "$THREAD_ID" \
  --state "Current state in compact factual form." \
  --task "Task being worked." \
  --next-action "Next safe action." \
  --session-id "$DML_SESSION_ID"
```

---

## Memory hygiene rules

Never store:

- API keys, access tokens, passwords, bearer headers, private keys, cookies, or `.env` contents.
- Raw transcripts.
- Tool logs or terminal dumps.
- Prompt-injected DML blocks.
- Personality overlay scaffolding as if it were a user fact.
- Speculation labeled as fact.
- Huge files or generated reports that are already reproducible.

Always store:

- Durable decisions.
- Accepted plans.
- Changed files and why they changed.
- Tests run and outcomes.
- Blockers and root causes.
- External constraints.
- Next actions.
- Configuration facts that affect future runs.

Use metadata:

```json
{
  "source": "agent-name-or-harness",
  "phase": "install|debug|validate|handoff|demo",
  "repo": "owner/repo",
  "session_id": "stable-session-id"
}
```

---

## Endpoint wizard: required user experience

If endpoints are not configured, do not dump a config file and walk away. Provide the user with a short wizard. The wizard should produce a concrete config, validation commands, and a rollback note.

### Wizard script for agents

Ask these questions in order. Use defaults when the user says “just make it work”.

#### 1. Where should memory live?

Offer:

1. Local project store: `./data/dml`
2. User profile store: `~/.dml/<profile>`
3. Agent profile store: `<agent-profile>/dml-store`
4. Custom path

Default: local project store for dev, profile store for production agents.

#### 2. What should generate embeddings?

Offer:

1. **Ollama local** — default, private, GPU-managed when Ollama uses GPU.
2. **SentenceTransformer local** — Python-only, offline-capable.
3. **Custom/remote embedding service** — advanced path.

Default:

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda
```

Tell the user:

- Ollama owns actual GPU/Metal/CUDA placement for `ollama:*` models.
- `embedding_device: cuda` is the strict DML-side contract for GPU-intended installs.
- If they change embedding dimensions, they need a new store or migration.

#### 3. What should summarize/reform memory?

Offer:

1. **Ollama local** — default.
2. **OpenAI-compatible endpoint** — vLLM, LM Studio, OpenAI, Azure, NIM-compatible services.
3. **Transformers local** — offline model path.

Default:

```yaml
llm_backend: ollama
model_name: llama3:8b
```

#### 4. Do they want DML to prepare frontier prompts?

Offer:

1. Preparation only — DML returns `frontier_prompt`; harness calls model.
2. Provider-integrated generation — DML server routes generation through configured backend.
3. Memory only — no inference route.

Default for agents: preparation only unless the deployment explicitly owns model credentials.

#### 5. What endpoint details are needed?

For OpenAI-compatible endpoints ask:

- Base URL, e.g. `http://127.0.0.1:8000/v1`
- Model name, e.g. `meta-llama/Llama-3.1-8B-Instruct`
- Whether an API key is required
- Context window and token budget
- Whether requests may leave the machine

Do not ask the user to paste secrets into chat if avoidable. Tell them to set environment variables locally.

```bash
export DML_API_BASE=http://127.0.0.1:8000/v1
export DML_API_KEY="$YOUR_KEY"  # optional; set locally, never commit
export DML_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

#### 6. Generate the config

Template:

```yaml
capacity: 4000
dml_top_k: 8
token_budget: 800

embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda
strict_embedding_required: true

llm_backend: ollama
model_name: llama3:8b
strict_llm_required: true

storage_dir: ./data/dml

persistence:
  enable: true
  path: ./dml_state.jsonl
  interval_sec: 300

rag_store:
  enable: true
  path: ./rag_index.faiss
  meta_path: ./rag_meta.json
  backend: faiss
  dim: 1024

dpm:
  enable: true
  mode: active-write
  preference_graph_path: ./dpm_preference_graph.json
  token_budget: 80
```

For SentenceTransformer, adjust:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_device: cuda
rag_store:
  dim: 384
```

For OpenAI-compatible summarization, adjust:

```yaml
llm_backend: openai
model_name: ${DML_MODEL_NAME}
```

and set environment variables rather than committing secrets.

#### 7. Validate

Run these and show the user real output summaries:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" backend-proof
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" ingest --kind observation --text "DML endpoint wizard probe" --no-filter-noise
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" retrieve --query "DML endpoint wizard probe" --top-k 3
```

If strict GPU validation fails but Ollama is known to be GPU-managed, either fix the config so strict Ollama-managed validation passes, or temporarily use `--no-require-gpu` and clearly report the reduced proof level.

#### 8. Persist the handoff

After setup succeeds:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path "$DML_CONFIG" \
  handoff \
  --thread setup \
  --state "DML endpoint wizard completed; backend proof and ingest/retrieve passed." \
  --task "Use DML as default memory substrate." \
  --next-action "Start normal agent workflow with retrieve + writeback enabled."
```

---

## Recommended default profiles

### Local private workstation

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda
llm_backend: ollama
model_name: llama3:8b
storage_dir: ~/.dml/default
persistence:
  enable: true
rag_store:
  enable: true
  backend: faiss
  dim: 1024
```

### GPU demo box

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda
strict_embedding_required: true
llm_backend: ollama
model_name: llama3:8b
strict_llm_required: true
storage_dir: ./stores/demo-runtime-store
persistence:
  enable: true
rag_store:
  enable: true
  backend: faiss
  dim: 1024
```

### Remote OpenAI-compatible inference

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda
llm_backend: openai
model_name: ${DML_MODEL_NAME}
storage_dir: ./data/dml
persistence:
  enable: true
rag_store:
  enable: true
  backend: faiss
  dim: 1024
```

Environment:

```bash
export DML_API_BASE=https://your-endpoint.example/v1
export DML_API_KEY="$YOUR_KEY"  # set locally, never commit
export DML_MODEL_NAME=your-model-name
```

---

## Verification checklist for agents

Before declaring success, prove it:

- [ ] `health` returns `status: ok` or a clearly explained degraded state.
- [ ] `backend-proof` identifies the intended embedding and summarization backends.
- [ ] A probe `ingest` writes at least one chunk or reports a duplicate intentionally.
- [ ] A probe `retrieve` can recall the probe memory.
- [ ] Store path is explicit and durable.
- [ ] No secrets were written to config, logs, memory, PRs, or docs.
- [ ] If Hermes is involved, `smoke_hygiene.py` and `smoke_dcn.py` pass.
- [ ] If DCN active-learn is involved, promotion evidence and rollback checkpoint exist.
- [ ] User receives the exact config path, store path, commands run, and next action.

---

## Common failure modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `GPU-only mode requires torch with CUDA support` | SentenceTransformer/CUDA path selected without torch CUDA | Use Ollama-managed embedding path or install CUDA torch for ST backend. |
| Dimension mismatch | Store/index was created with different embedding dimension | Use a new store or run migration; do not mix dimensions blindly. |
| Retrieval returns irrelevant old state | Missing tenant/session scoping or stale memories | Pass `tenant_id`/`session_id`, curate, or resolve conflicts. |
| Agent keeps forgetting | It is not calling retrieve/resume by default | Move DML calls into core loop, not user-prompt-only paths. |
| Prompt bloat | Raw transcripts/tool logs are being stored or injected | Enforce hygiene filters and compact handoffs. |
| Endpoint works manually but not in agent | Env vars not loaded in agent process | Print a safe config summary; restart process with env/profile loaded. |

---

## The correct user-facing promise

A DML-enabled agent should be able to say:

> “I have a durable memory substrate. I retrieve relevant prior state automatically, write compact durable updates after meaningful work, and can switch embedding/summarization/inference endpoints through a guided setup without exposing secrets.”

That is the bar. Build to it.
