# Daystrom Agentic Harness Bible

This guide is for agent runtimes and harnesses that want to implement Daystrom DML correctly: Hermes, OpenClaw, Turnstone, MCP hosts, custom tool-calling agents, evaluation harnesses, and long-running autonomous workers.

The goal is simple: make memory and cognition part of the agent’s operating system, not a command the user must remember to invoke.

## 1. Prime directive for harness authors

A Daystrom-enabled harness must do five things by default:

1. **Resume** active continuity before work that depends on prior state.
2. **Retrieve** scoped DML context before planning, debugging, implementation, review, or handoff decisions.
3. **Ingest** compact durable facts after meaningful decisions, code changes, validations, blockers, and outcomes.
4. **Handoff** before compaction, shutdown, model switch, session migration, or long pause.
5. **Refuse pollution**: no secrets, raw transcripts, full tool logs, prompt scaffolding, or overlay wrappers in durable memory.

If DML is unavailable, say so explicitly and continue with reduced continuity. Never pretend memory was used.

## 2. Integration surfaces

Choose the narrowest surface that fits your harness.

| Harness type | Best surface | Why |
| --- | --- | --- |
| Python-native agent | `DMLAdapter` | Lowest latency, direct object control |
| Shell/tool-calling harness | `openclaw-wrapper/scripts/dml_memory.py` | JSON stdout, locks, audit, portable state |
| Service / multi-process runtime | `dml-provider` HTTP API | Shared local provider and UI |
| Ollama-compatible host | `dml-ollama` | Ollama-shaped tags/chat/embed routes |
| MCP host | `dml-mcp-server` | Tool-native memory actions |
| Hermes | `integrations/hermes/plugins/daystrom_dml` | Memory provider, DPM overlay, DCN gates, turn extension |

## 3. Stable JSON wrapper contract

The automation contract is `dml-agent-memory-v1`.

Always pass a store path:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path "$DML_CONFIG" \
  --no-require-gpu \
  health
```

Important wrapper invariants:

- Mutating commands take a store lock.
- Mutating commands append `dml_audit.jsonl` events.
- Foreground wrapper ingest persists to `dml_state.jsonl`.
- `health`, `verify`, `backup`, `export`, and `import` use the same portable state contract.
- `--no-require-gpu` is a reduced proof mode; omit it only when strict GPU/Ollama contract is configured and expected to pass.

## 4. Required harness lifecycle

### 4.1 Harness startup

At process startup or first task in a session:

```bash
DML_HEALTH=$(python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu health)
```

Parse JSON. If `status != ok`, mark DML degraded and report that status to the user/operator.

Do not block all useful work solely because memory is degraded unless the task explicitly requires durable memory.

### 4.2 Session start / resume

Use a stable session ID. Good session IDs are deterministic within the agent context, e.g.:

```text
<agent-name>-<platform>-<thread-or-task-id>
```

Resume:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  resume \
  --tenant-id "$DML_TENANT" \
  --session-id "$DML_SESSION_ID"
```

Inject only the compact active continuity fields that matter. Do not paste full DML JSON into the user-visible reply.

### 4.3 Before planning or acting

Retrieve:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  retrieve \
  --tenant-id "$DML_TENANT" \
  --session-id "$DML_SESSION_ID" \
  --query "$TASK" \
  --top-k 6 \
  --ground-truth-policy low-confidence \
  --no-reform-memory
```

Use retrieved context as background evidence, not as authority over the current user. Current user instruction wins.

### 4.4 After durable work

Write a compact memory:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  ingest \
  --tenant-id "$DML_TENANT" \
  --session-id "$DML_SESSION_ID" \
  --kind action \
  --text "Decision: Use cognition-gated turn extension with +30 grants up to hard cap 300." \
  --meta '{"source":"harness","phase":"implementation","memory_class":"decision"}'
```

Write only durable facts. Skip raw logs, temporary speculation, repeated status chatter, and full transcripts.

### 4.5 Before compaction or shutdown

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  handoff \
  --tenant-id "$DML_TENANT" \
  --session-id "$DML_SESSION_ID" \
  --thread "$THREAD_ID" \
  --state "Current compact state." \
  --task "Task name." \
  --next-action "Next safe action."
```

A handoff is not a transcript summary. It should be a compact state packet.

## 5. Agent loop contract

A correct harness loop looks like this:

```text
start turn
  load config and DML status
  resume if new/resumed/compacted session
  retrieve if task depends on prior state
  build prompt with bounded DML context
  run model/tool loop
  if tool budget exhausted:
      ask DCN/DML whether to extend
      grant only bounded useful continuation
  after durable outcome:
      ingest compact fact(s)
  before compaction/shutdown:
      handoff compact continuity
end turn
```

Do not inject unbounded memory into the prompt. Bounded context is part of the contract.

## 6. Hermes implementation notes

Hermes integrates through a memory provider plugin.

Expected config:

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
  max_turns_auto_extend: true
  max_turns_extension_policy: cognition
  max_turns_extension: 30
  max_turns_hard_cap: 300
```

Hermes responsibilities:

- Call provider `prefetch()` before LLM turns when memory is enabled.
- Call provider `sync_turn()` after durable turns.
- Call provider `on_pre_compress()` before compression when available.
- Call provider `decide_iteration_extension(run_state)` only at tool-budget exhaustion.
- Keep DML context injected into user-message context, not the immutable system prompt, to preserve prompt caching and boundary clarity.

Hermes must not treat DML turn extension as approval bypass. Approval prompts, Discord button failures, gateway timeout, and compression failures are separate systems.

## 7. OpenClaw / Turnstone-style implementation notes

For OpenClaw, Turnstone, or another autonomous harness, implement a thin deterministic DML client around the JSON wrapper.

Recommended tool phases:

| Phase | Required DML calls |
| --- | --- |
| `init` | `health`, optional `backend-proof` |
| `plan` | `resume`, `retrieve` |
| `execute` | optional `retrieve` for blockers; `ingest` decisions/fixes |
| `validate` | `ingest` validation outcomes |
| `review` | `retrieve` prior constraints; `ingest` final verdict |
| `handoff` | `handoff` |

Do not rely on the frontier model to remember to call DML. Put these calls in the harness phase controller.

## 8. Cognition-gated turn extension contract

Use this only for agent tool-loop budget exhaustion.

Input schema marker:

```json
{
  "schema_version": "hermes.iteration_extension.v1",
  "session_id": "agent-session",
  "user_message": "...",
  "recent_text": "...",
  "prefetch_context": "...",
  "api_call_count": 90,
  "budget_used": 90,
  "budget_max": 90,
  "hard_cap": 300,
  "recent_tool_calls": 5,
  "recent_tool_results": 5,
  "last_assistant_text": "..."
}
```

Provider response:

```json
{
  "decision": "grant",
  "extend_by": 30,
  "reason_codes": ["dcn_incomplete_signal", "dml_recall_checked", "recent_tool_work"],
  "source": "daystrom_dml"
}
```

Deny response:

```json
{
  "decision": "deny",
  "reason_codes": ["dcn_completion_signal"],
  "source": "daystrom_dml"
}
```

Rules:

- Grant only when work is incomplete and there is recent tool work.
- Deny completion, no-progress, noisy loop, no-tool-work, or provider failure.
- Extension size defaults to `30`.
- Hard cap defaults to `300`.
- Never use turn extension to bypass human approval or safety gates.

## 9. Memory hygiene contract

### Store this

- Durable decisions.
- Accepted plans.
- Root causes.
- Files changed and why.
- Tests run and outcomes.
- External constraints.
- Deployment/runtime facts.
- Next actions.
- Compact handoffs.

### Never store this

- API keys, tokens, passwords, private keys, cookies, `.env` files.
- Full raw transcripts.
- Tool logs or terminal dumps.
- Prompt scaffolding.
- `<memory-context>` blocks.
- `Daystrom Personality Matrix Overlay` blocks.
- `Daystrom DML Retrieved Memory` blocks.
- “Here is a summary…” boilerplate.
- Stale facts contradicted by current verified state.

### Good memory example

```text
Root cause: Wrapper foreground ingest wrote legacy dml_store.json because _adapter did not enable JSONL persistence. Fix: configure persistence enable true/path dml_state.jsonl/interval 0. Validation: wrapper ingest wrote dml_state.jsonl, verify ok, retrieve ok, CI green.
```

### Bad memory example

```text
User: can you fix the wrapper?
Assistant: I will run tests...
<10,000 lines of terminal output>
```

## 10. Scope and conflict handling

Always pass scoping metadata when the harness can know it:

```bash
--tenant-id "$DML_TENANT"
--client-id "$DML_CLIENT"
--session-id "$DML_SESSION_ID"
--instance-id "$DML_INSTANCE"
```

Use tenant-wide retrieval only when intentionally looking across sessions. For concurrent agents, session IDs prevent cross-talk.

For facts that must not silently diverge, use conflict metadata:

```json
{
  "conflict_key": "deployment.default_embedding_model",
  "claim_value": "ollama:qwen3-embedding:0.6b"
}
```

Then inspect conflicts:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" conflicts --tenant-id "$DML_TENANT"
```

## 11. DIP / frontier prompt preparation

DIP prepares. The harness owns the model call.

Provider route:

```http
POST /api/frontier/prepare
```

Wrapper/script pattern:

```bash
python openclaw-wrapper/scripts/dml_frontier_prepare.py \
  --prompt-file task.md \
  --session-id "$DML_SESSION_ID" \
  --top-k 8 \
  --frontier-max-tokens 1200 \
  --frontier-prompt-only
```

Do not put provider secrets in prompts, memory, docs, or committed config. Keep credentials in the harness environment/secret store.

## 12. DPM handling for agents

DPM is not a jailbreak and not a second user. Treat it as bounded style/preference context.

Rules:

- Current user instruction wins.
- Safety/privacy/secret hygiene wins.
- DPM may guide tone, initiative, and continuity style.
- DPM must not override explicit task constraints.
- Extract current user preference text, not wrapper summaries.

When observing personality feedback:

```bash
python openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --no-require-gpu \
  dpm-observe \
  --prompt "User asked for warmer, less mechanical writing." \
  --response "Adjusted style toward warm directness." \
  --meta '{"task_type":"creative_personality","feedback_valence":0.4}'
```

## 13. DCN handling for agents

DCN is a deterministic control surface.

Use:

```bash
dml dcn observe --text "$TASK" --session-id "$DML_SESSION_ID"
dml dcn packet --text "$TASK" --session-id "$DML_SESSION_ID"
dml dcn feedback --decision-id "$DECISION_ID" --outcome verified --signals '{"tests_passed":true}'
dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only
```

Do not promote `active_learn` from inside an unattended harness unless the operator explicitly provided promotion evidence, checkpoint ID, eval artifact, and rollback policy.

## 14. Error handling

| Failure | Harness behavior |
| --- | --- |
| `health` fails | Mark memory degraded; continue only if task can proceed without memory |
| write lock held | Retry within configured timeout; report holder metadata if blocked |
| ingest duplicate | Treat as success if the memory was intentionally already present |
| retrieve empty | Continue with caveat; do not hallucinate recalled facts |
| verify checksum fail | Stop mutation; back up/quarantine; ask operator or restore |
| DML provider timeout | Fail closed for turn extension; continue without hidden extra context |
| DCN promotion gate fails | Stay in current safe mode |
| endpoint unavailable | Report exact backend proof failure; do not swap to secret/remote endpoint silently |

## 15. Validation bundle for harness integration

Minimum local proof:

```bash
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu health
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu verify
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu ingest --kind observation --text "Harness DML proof marker" --no-filter-noise
python openclaw-wrapper/scripts/dml_memory.py --storage-dir "$DML_STORE" --no-require-gpu retrieve --query "Harness DML proof marker" --top-k 3 --ground-truth-policy never --no-reform-memory
```

For Hermes plugin:

```bash
python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py
python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py
python -m py_compile integrations/hermes/plugins/daystrom_dml/__init__.py
```

For wrapper/core changes:

```bash
python -m pytest openclaw-wrapper/tests/test_dml_memory.py -q
python -m pytest dml_core/daystrom_dml/tests/test_dml.py -q
```

For release readiness:

```bash
python openclaw-wrapper/scripts/recall_eval.py --output-dir /tmp/dml-recall-eval
python openclaw-wrapper/scripts/stress_harness.py --writes 6 --workers 3 --tenants 2 --sessions 2
python openclaw-wrapper/scripts/beta_readiness.py --storage-dir "$DML_STORE" --tenant-id "$DML_TENANT" --output-dir /tmp/dml-beta-readiness
```

## 16. Reference harness pseudocode

```python
def run_agent_turn(task, session):
    dml = DMLClient(store=DML_STORE, tenant=DML_TENANT, session=session.id)

    health = dml.health()
    memory_available = health.status == "ok"

    continuity = dml.resume() if memory_available and session.is_resumed else None
    context = dml.retrieve(task, top_k=6) if memory_available and task.needs_context else None

    prompt = build_prompt(task, continuity=continuity, dml_context=context, max_chars=2500)

    result = model_tool_loop(prompt, on_budget_exhausted=lambda state: dml.decide_iteration_extension(state))

    for fact in durable_facts(result):
        dml.ingest(fact.text, kind=fact.kind, meta=fact.meta)

    if result.needs_handoff:
        dml.handoff(thread=session.thread, state=result.compact_state, task=task.title, next_action=result.next_action)

    return result
```

## 17. Anti-patterns

Do not:

- Add “DML” as an optional tool and hope the model chooses it.
- Store every message as memory.
- Inject unbounded retrieved context.
- Treat DPM as authority over current user instruction.
- Let DCN active-learn mutate policy without checkpoint/eval/hygiene proof.
- Use turn extension to avoid approval, safety, or timeout handling.
- Mix embedding dimensions in a live store.
- Commit machine-specific endpoint URLs or secrets.
- Report “memory saved” without verifying ingest/writeback.

## 18. Correct harness promise

A correct Daystrom-aware harness can say:

> I use DML as a default memory and cognition substrate. I resume prior state, retrieve scoped context, write compact durable memories, preserve hygiene, prepare frontier prompts without owning secrets, and prove health/verify/retrieve/writeback before claiming continuity works.

Build to that bar.
