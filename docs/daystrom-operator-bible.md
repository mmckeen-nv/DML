# Daystrom DML Operator Bible

This is the operator runbook for deploying, validating, maintaining, and safely promoting Daystrom DML in a real agent environment.

DML is not a chat-history folder and it is not a generic RAG cache. Treat it as platform infrastructure: memory substrate, continuity spine, hygiene boundary, and cognition support layer for long-horizon agents.

## 1. Operator mental model

The Daystrom stack has four separate responsibilities:

| Layer | Owns | Does not own |
| --- | --- | --- |
| **DML — Daystrom Memory Lattice** | Durable memory, summaries, embeddings, retrieval, active continuity, conflict handling, backup/export/verify | Personality authority, model execution, raw transcript storage |
| **DPM — Daystrom Personality Matrix** | Bounded preference/personality overlays and explicit user corrections | Safety rules, current-turn instruction priority, secret policy |
| **DCN — Daystrom Cognition Network** | Deterministic cognition control: retrieval gates, cognitive packets, feedback, evals, governed promotion, turn-extension decisions | Free-form autonomy, values, identity, raw memories |
| **DIP — Daystrom Inference Preparation** | Prompt/context preparation for a downstream frontier/base model | Secret ownership or unapproved model calls |

The boundaries matter. Operators should be able to validate and roll back each layer independently.

## 2. Production posture

A correctly operated DML deployment has these properties:

- **Memory is on by default.** Agents retrieve relevant memory during normal work; the user should not need to say “use DML.”
- **Writeback is hygienic.** Store compact semantic facts, decisions, validations, blockers, and handoffs. Do not store raw transcripts, tool logs, prompt wrappers, DML overlay blocks, or secrets.
- **Continuity is active.** Long tasks and resumed sessions use `resume`, `retrieve`, and `handoff` rather than replaying giant transcripts.
- **Persistence is contract-aligned.** Wrapper operations write and verify the same portable JSONL state file: `<storage_dir>/dml_state.jsonl`.
- **Cognition is bounded.** DCN can gate retrieval and grant additional Hermes tool iterations only under conservative evidence; it must deny completed/noisy/no-progress loops.
- **Inference is explicit.** DML may prepare context, but the operator decides what endpoint performs generation and where credentials live.

## 3. Install and store layout

Recommended operator variables:

```bash
export DML_STORE=/srv/daystrom/stores/default
export DML_CONFIG=/srv/daystrom/config/dml.yaml
export DML_TENANT=openclaw
```

Local development defaults can use:

```bash
export DML_STORE=./data/dml
export DML_CONFIG=openclaw-wrapper/config/dml_gpu_only.yaml
```

A healthy store may contain:

```text
dml_state.jsonl                 Portable durable DML state; health/verify/export source of truth
.ingest_dedup_sha256.txt        Ingest dedup index
dml_audit.jsonl                 Append-only wrapper audit events
rag_index.faiss                 Optional FAISS sidecar
rag_meta.json                   Optional RAG metadata sidecar
dpm_preference_graph.json       DPM preference graph
dpm_evolution_graph.json        DPM fast/slow evolution graph
embedding_compatibility_report.json
backups/
exports/
```

Stopline: if wrapper `ingest` reports `chunks_ingested > 0` but `dml_state.jsonl` does not exist or does not increase, the wrapper and adapter persistence contract are misaligned. The foreground wrapper must enable:

```yaml
persistence:
  enable: true
  path: dml_state.jsonl
  interval_sec: 0
```

## 4. Endpoint wizard for operators

Ask these questions before live deployment.

### 4.1 Where does memory live?

| Deployment | Recommended store |
| --- | --- |
| Local dev | `./data/dml` |
| Single-user workstation | profile-local durable store |
| Demo machine | agent-profile store under the demo profile |
| Multi-tenant service | explicit tenant/client/session/instance scoped store |

### 4.2 What generates embeddings?

Default private runtime:

```yaml
embedding_model: ollama:qwen3-embedding:0.6b
embedding_device: cuda   # strict DML-side intent; Ollama owns actual placement
rag_store:
  enable: true
  backend: faiss
  dim: 1024
```

Alternative local Python path:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_device: cuda   # or cpu/mps/cuda:0
rag_store:
  dim: 384
```

Do not mix embedding dimensions in one live store without migration.

### 4.3 What summarizes/reforms memory?

Default local runtime:

```yaml
llm_backend: ollama
model_name: llama3:8b
strict_llm_required: true
```

OpenAI-compatible endpoint:

```bash
export DML_API_BASE=http://127.0.0.1:8000/v1
export DML_API_KEY="$YOUR_KEY"       # set locally; never commit
export DML_MODEL_NAME=your-model-name
```

```yaml
llm_backend: openai
model_name: ${DML_MODEL_NAME}
```

Do not commit endpoints that reveal private routing, VPN names, auth headers, provider secrets, or machine-specific access paths.

### 4.4 Is DIP enabled?

Choose one:

1. **Memory only** — DML retrieves context; caller constructs final prompt.
2. **Prompt preparation** — DML returns a compact frontier prompt; caller owns model execution.
3. **Provider-integrated generation** — DML provider routes model calls through configured backend.

For agents and harnesses, use prompt preparation unless the deployment explicitly owns credentials and egress policy.

## 5. Health and proof bundle

Run these before trusting a store:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" --no-require-gpu health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" --no-require-gpu verify
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" --no-require-gpu report
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --config-path "$DML_CONFIG" --no-require-gpu backend-proof
```

Then prove write/read:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path "$DML_CONFIG" \
  --no-require-gpu \
  ingest \
  --kind observation \
  --text "DML operator proof: ingest and retrieve must find this marker." \
  --meta '{"source":"operator-proof","tenant_id":"openclaw","merge_policy":"never"}' \
  --no-filter-noise

python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path "$DML_CONFIG" \
  --no-require-gpu \
  retrieve \
  --tenant-id "$DML_TENANT" \
  --query "operator proof ingest retrieve marker" \
  --top-k 3 \
  --ground-truth-policy never \
  --no-reform-memory
```

Expected proof:

- `health.status == ok`
- `verify.status == ok`
- `state.exists == true`
- `checksum_ok == true`
- `count_ok == true`
- `record_count` increases or the ingest explicitly reports duplicate skip
- retrieve returns the marker or a clearly explained scoped miss

## 6. Runtime operation loop

### 6.1 Start of task or session

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu resume --session-id "$DML_SESSION_ID"
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu retrieve --session-id "$DML_SESSION_ID" --tenant-id "$DML_TENANT" --query "$TASK" --top-k 6
```

### 6.2 During work

Store durable facts only:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  ingest \
  --kind action \
  --text "Decision: Use portable JSONL persistence for wrapper foreground ingests." \
  --meta '{"source":"operator","phase":"implementation","tenant_id":"openclaw"}'
```

### 6.3 Before compaction, model switch, shutdown, or handoff

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  handoff \
  --thread "$THREAD_ID" \
  --state "Compact factual state; no transcript." \
  --task "Current task." \
  --next-action "Next safe action." \
  --tenant-id "$DML_TENANT" \
  --session-id "$DML_SESSION_ID"
```

## 7. Hermes/Citizen Snips operation

Hermes uses DML through the profile plugin at `integrations/hermes/plugins/daystrom_dml`.

Minimum intended config:

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
    dcn:
      mode: active_read

agent:
  max_turns: 90
  max_turns_auto_extend: true
  max_turns_extension_policy: cognition
  max_turns_extension: 30
  max_turns_hard_cap: 300
```

Validation:

```bash
hermes memory status
python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
python -m py_compile integrations/hermes/plugins/daystrom_dml/__init__.py
```

Operational notes:

- `retrieval_policy: always` is the normal mode. `heuristic` is the opt-out.
- DPM overlays are advisory and current-turn subordinate.
- DML writeback should reject raw Discord/gateway wrappers, `<memory-context>`, DPM overlay scaffolding, tool logs, and role-prefixed transcripts.
- Code/config/plugin changes require a fresh Hermes process or gateway restart.

## 8. Cognition-gated turn extension

Hermes turn flexibility is a bounded cognition-gated path, not “infinite turns.”

Expected behavior:

1. Hermes starts with `agent.max_turns`, typically `90`.
2. At budget exhaustion only, Hermes builds `hermes.iteration_extension.v1` run state.
3. Hermes asks the active memory provider for `decide_iteration_extension(run_state)`.
4. DML/DCN grants `+30` only when evidence says the task is incomplete and there was recent tool work.
5. DML/DCN denies complete, noisy, looping, or no-tool-progress states.
6. Hard cap stops the turn at `300`.

Proof markers:

- Hermes contains `agent/iteration_extension.py` and calls `decide_iteration_extension` from `agent/conversation_loop.py`.
- DML Hermes plugin implements `decide_iteration_extension`.
- Smoke tests include a grant and deny path for `hermes.iteration_extension.v1`.

Do not confuse turn flexibility with Discord approval, gateway timeout, or compression. A manual approval prompt can still block even when turn extension is working.

## 9. Hygiene stoplines

Immediately stop and repair if any durable memory contains:

- Real secrets, tokens, cookies, private keys, `.env` contents.
- Raw transcripts or role-prefixed dialogue.
- Full terminal/tool logs.
- Prompt scaffolding such as `<memory-context>`, `Daystrom Personality Matrix Overlay`, or `Daystrom DML Retrieved Memory`.
- Summarizer residue such as “Here is a summary…” or “summary of the content in 256 characters.”
- Stale false continuity that conflicts with verified GitHub/runtime state.

Basic scan:

```bash
python - <<'PY'
import pathlib, re, json, os
store = pathlib.Path(os.environ.get('DML_STORE', './data/dml'))
patterns = [
  'Here is a summary', 'summary of the content', '256 character limit',
  '<memory-context>', 'Daystrom Personality Matrix Overlay',
  'Daystrom DML Retrieved Memory', 'BEGIN PRIVATE KEY', 'api_key', 'password'
]
for name in ['dml_state.jsonl', 'dml_audit.jsonl']:
    p = store / name
    text = p.read_text(errors='ignore') if p.exists() else ''
    print(name, {pat: len(re.findall(re.escape(pat), text, re.I)) for pat in patterns})
PY
```

## 10. Backup, export, restore

Before risky operations:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" backup --label before-maintenance
```

Portable export:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" export --output-dir /tmp/dml-exports --label machine-move
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" verify-export --bundle /tmp/dml-exports/<bundle>.dml-export.tar.gz
```

Import on target:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" import --bundle /tmp/dml-exports/<bundle>.dml-export.tar.gz
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" verify
```

## 11. DCN promotion and rollback

DCN active-learn promotion is fail-closed. Do not promote stronger behavior unless you have:

- Passing provider eval smoke artifact.
- Passing hygiene smoke artifact/hash.
- Explicit checkpoint ID.
- Rollback command.
- Sanitized evidence only.

Commands:

```bash
dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only
dml dcn policy checkpoint --label before-active-learn
dml dcn promote --mode active_learn --checkpoint-id <checkpoint-id> --hygiene-evidence '{"passed":true,"artifact_hash":"..."}'
dml dcn policy rollback --checkpoint-id <checkpoint-id>
```

Never include raw prompts, transcripts, tool logs, secrets, or raw memory context in promotion evidence.

## 12. Troubleshooting map

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `chunks_ingested > 0` but `dml_state.jsonl` missing/unchanged | Wrapper adapter persisted legacy `dml_store.json` instead of portable JSONL | Enable `persistence.path: dml_state.jsonl`; rerun wrapper verify |
| Retrieval works but verify fails | Adapter loaded legacy state; wrapper verifies JSONL contract | Migrate/repersist into `dml_state.jsonl` |
| Agent “forgets” unless asked | DML retrieve is user-prompt gated | Move retrieval into core loop; use `retrieval_policy: always` |
| Prompt bloat | Raw transcripts/tool logs stored or injected | Purge polluted memories; enforce hygiene filters |
| Flat visualizer lattice | Missing `lattice_row/col/layer/neighbors` | Run lattice integrity repair/import path; verify coordinate spread |
| Turn flex does not help approval failures | Discord/gateway approval path is blocked | Fix approval mode or button handling; turn extension only handles iteration budget exhaustion |
| Dimension mismatch | Store created with another embedder dimension | New store or embedding migration |
| DPM feels stale | Old preference nodes outrank newer explicit corrections | Verify DPM graph recency/strength selection; reseed clean preferences |

## 13. Release checklist

Before publishing or saying a deployment is ready:

- [ ] `git status` clean or intended diff only.
- [ ] `health`, `verify`, `report` pass on target store.
- [ ] Ingest/retrieve proof passes with scoped tenant/session.
- [ ] Hygiene scan clean for durable state/audit files.
- [ ] Relevant tests pass: wrapper, core DML, Hermes plugin smokes.
- [ ] Added-line secret scan clean.
- [ ] Docs use placeholders, not real endpoints/secrets.
- [ ] GitHub branch/PR/CI state verified after push.
- [ ] Runtime restart/deploy status stated separately from repo readiness.

## 14. Operator promise

A correct operator can say:

> DML is healthy, scoped, persistent, hygienic, backed up, and integrated into the agent’s normal loop. I can prove what store it uses, what backend it calls, what it remembers, what it refuses to store, how to restore it, and how to roll back stronger cognition modes.

That is the operating bar.
