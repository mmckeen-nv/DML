# Daystrom Cognition Network Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the Daystrom Cognition Network (DCN) as the adaptive cognitive control layer between frontier/base LLMs and Daystrom memory/personality/inference substrates.

**Architecture:** The Daystrom Platform will have four cleanly separated components: DML for memory, DPM for personality/identity/preference overlays, DCN for adaptive attention/routing/policy control, and DIP for frontier/inference prompt preparation and execution. DCN is not a RAG replacement, not a personality graph, and not a frontier model; it is the brainstem-like policy network that predicts what the agent should attend to, retrieve, suppress, verify, delegate, and write back.

**Tech Stack:** Python 3.10+, dataclasses/Pydantic-compatible JSON contracts, existing `dml_core/daystrom_dml` package, FastAPI/provider server, pytest, existing DML adapter/provider CLI, optional local SLM/policy backend later.

**Status:** Draft plan created 2026-05-31.

---

## 0. Naming and Platform Boundaries

The Daystrom Platform should be described as four cooperating systems with explicit contracts:

- **DML — Daystrom Memory Lattice**
  - Owns durable semantic memory, retrieval, compression, continuity checkpoints, conflicts, curation, audit, and memory hygiene.
  - Does **not** own personality values, online policy learning, or final frontier inference.

- **DPM — Daystrom Personality Matrix**
  - Owns stable identity/personality/preference overlays, scoped preference graph, explicit user corrections, inspectable value/personality constraints, and bounded runtime rendering.
  - Does **not** own semantic memory retrieval, tool routing, or hidden self-modifying policy.

- **DCN — Daystrom Cognition Network**
  - Owns cognitive control: intent classification, attention gating, retrieval planning, context budgeting, tool/delegation hints, verification policy, writeback policy, risk flags, outcome feedback, and adaptive policy improvement.
  - It is the “brainstem” layer: it regulates what the frontier model sees and what gets written back.
  - Does **not** own raw memory persistence, personality values, or final user-facing reasoning.

- **DIP — Daystrom Inference Pipeline**
  - Owns prepared-prompt construction and optional frontier/base inference execution.
  - Consumes DML/DPM/DCN outputs.
  - Remains explicitly unfinished/prototype until promoted behind a stable contract.

Canonical flow:

```text
User/event/tool stimulus
  -> DCN.observe/classify
  -> DCN.plan_context
  -> DML.retrieve + DPM.overlay
  -> DCN.assemble/cognitive_packet
  -> DIP.prepare or host agent/frontier LLM
  -> tool/action/final response
  -> verifier/outcome signal
  -> DML.write durable memory + DCN.learn policy feedback + DPM explicit preference update only
```

---

## 1. Non-Negotiable Invariants

1. Current-turn user instruction outranks DPM overlays and DCN learned policy.
2. DCN may learn procedural policy, but must not silently mutate identity, values, autonomy boundaries, or user preferences.
3. DML stores durable episodes/facts; DCN learns from episodes; DPM stores explicit/curated personality and preference structure.
4. Every DCN decision must be inspectable as JSON: inputs, chosen mode, retrieval plan, budget, risk, verification policy, writeback policy, and reason codes.
5. DCN starts deterministic/rules-first. Online learning is added only after observability and rollback exist.
6. DIP must remain optional. The host agent can consume a DCN cognitive packet without routing through the unfinished inference pipeline.
7. Memory writeback is gated: no raw transcripts, tool logs, secrets, prompt scaffolding, or synthetic claims.
8. DCN must support observe-only, active-read, and active-learn modes.
9. All APIs must accept tenant/client/session/instance scope where relevant.
10. Clean component imports: DML can run without DCN; DPM can run without DCN; DCN can be disabled without breaking DML retrieval.

---

## 2. Public API Contracts

### 2.1 DML Stable Contract

Namespace: `/api/dml/*` and Python `DMLAdapter`.

Required operations:

```http
POST /api/dml/remember
POST /api/dml/retrieve
POST /api/dml/handoff
POST /api/dml/resume
GET  /api/dml/health
GET  /api/dml/conflicts
POST /api/dml/resolve-conflict
POST /api/dml/curate
```

Python-facing shape:

```python
adapter.remember(text, kind, tenant_id, client_id=None, session_id=None, instance_id=None, meta=None)
adapter.retrieve_context(query, tenant_id, client_id=None, session_id=None, instance_id=None, top_k=6, budget_tokens=None)
adapter.handoff(thread, state, task, next_action, tenant_id, session_id=None, meta=None)
adapter.resume(tenant_id, session_id=None, top_k=6)
```

DML output must include:

```json
{
  "raw_context": "...",
  "items": [],
  "context_tokens": 0,
  "conflict_count": 0,
  "scope": {"tenant_id": "...", "client_id": null, "session_id": null, "instance_id": null},
  "telemetry": {"latency_ms": 0.0, "retrieved_items": 0},
  "audit": {"reason": "...", "policy": "..."}
}
```

### 2.2 DPM Stable Contract

Namespace: `/api/dpm/*` and Python `DPMRuntime` or adapter facade.

Required operations:

```http
GET  /api/dpm/overlay
GET  /api/dpm/graph
POST /api/dpm/preference
POST /api/dpm/suppress
DELETE /api/dpm/preference/{node_id}
GET  /api/dpm/audit
```

Python-facing shape:

```python
dpm.render_overlay(scope, budget_tokens=300, current_instruction=None) -> DPMOverlay
dpm.record_preference(signal, scope, source, confidence=None) -> PreferenceWriteResult
dpm.suppress(node_id, reason, scope) -> PreferenceWriteResult
dpm.inspect_graph(scope=None) -> PreferenceGraphReport
```

DPM overlay output must include:

```json
{
  "overlay_text": "...",
  "directives": [],
  "suppressed": [],
  "conflicts": [],
  "budget": {"limit_tokens": 300, "used_tokens": 0},
  "mode": "disabled|observe_only|active_read|active_write",
  "audit": {"source_graph": "...", "reason_codes": []}
}
```

### 2.3 DCN Stable Contract

Namespace: `/api/dcn/*` and Python `CognitionController`.

Required operations:

```http
POST /api/dcn/observe
POST /api/dcn/plan-context
POST /api/dcn/cognitive-packet
POST /api/dcn/feedback
GET  /api/dcn/policy
GET  /api/dcn/audit
POST /api/dcn/policy/export
POST /api/dcn/policy/import
```

Core request:

```json
{
  "event": {
    "type": "user_message|tool_result|cron|gateway_resume|compaction_resume",
    "content": "...",
    "metadata": {}
  },
  "scope": {
    "tenant_id": "openclaw",
    "client_id": null,
    "session_id": null,
    "instance_id": null,
    "thread_id": null,
    "project_id": null,
    "relationship_id": null
  },
  "constraints": {
    "max_total_context_tokens": 6000,
    "max_memory_tokens": 1800,
    "max_personality_tokens": 300,
    "allow_tools": true,
    "allow_learning": false
  }
}
```

Core response:

```json
{
  "intent": {
    "task_type": "answer|code_change|recall|planning|debugging|creative|admin|unknown",
    "confidence": 0.0,
    "needs_memory": false,
    "needs_personality": true,
    "needs_tools": false,
    "needs_verification": false
  },
  "risk": {
    "level": "low|medium|high|blocked",
    "reasons": [],
    "requires_confirmation": false
  },
  "retrieval_plan": {
    "mode": "none|resume|semantic|continuity|hybrid",
    "queries": [],
    "top_k": 6,
    "budget_tokens": 1800,
    "ground_truth_policy": "never|low_confidence|always"
  },
  "personality_plan": {
    "mode": "none|bounded_overlay",
    "budget_tokens": 300,
    "suppress_if_conflicts_with_current_turn": true
  },
  "tool_plan": {
    "allowed": true,
    "recommended_tools": [],
    "verification_required": []
  },
  "writeback_plan": {
    "mode": "none|durable_signal_only|handoff|preference_candidate",
    "forbidden_classes": ["raw_transcript", "tool_log", "secret", "prompt_scaffold"]
  },
  "frontier_plan": {
    "mode": "direct|dml_context|verify_local_draft|full_frontier|no_frontier",
    "max_input_tokens": 6000,
    "max_output_tokens": 1200
  },
  "reason_codes": [],
  "policy_version": "dcn-policy-v0"
}
```

Cognitive packet response:

```json
{
  "packet_version": "daystrom-cognitive-packet-v1",
  "scope": {},
  "dcn_plan": {},
  "dml_context": {},
  "dpm_overlay": {},
  "assembled_context": "...",
  "guardrails": [],
  "telemetry": {},
  "audit": {}
}
```

Feedback request:

```json
{
  "decision_id": "...",
  "outcome": "accepted|corrected|failed|verified|user_rejected|tool_failed|test_passed|test_failed",
  "signals": {
    "used_retrieved_memory": true,
    "retrieval_helpful": true,
    "missing_context": false,
    "stale_context": false,
    "verification_caught_issue": false
  },
  "notes": "..."
}
```

### 2.4 DIP Stable/Prototype Contract

Namespace: `/api/dip/*` only after promotion. Until then, keep `/api/frontier/prepare` marked prototype.

Required future operations:

```http
POST /api/dip/prepare
POST /api/dip/infer          # optional; may stay disabled by policy
GET  /api/dip/telemetry
```

DIP consumes the DCN cognitive packet and returns:

```json
{
  "mode": "prepare_only|infer",
  "frontier_prompt": "...",
  "local_draft": "...",
  "max_tokens": 1200,
  "telemetry": {},
  "policy": {"inference_enabled": false, "reason": "prototype_disabled"}
}
```

---

## 3. Repository Layout Target

Create a clean namespace while preserving existing code:

```text
dml_core/daystrom_dml/
  memory/                 # future DML-only internals; wrapper around current memory_store/retrievers
  personality/            # future DPM-only internals; wrapper around personality_matrix
  cognition/              # new DCN package
    __init__.py
    schema.py
    policy.py
    controller.py
    feedback.py
    audit.py
    learning.py
  inference/              # future DIP package; wrapper around frontier_pipeline
    __init__.py
    schema.py
    prepare.py
  api_contracts.py        # shared scope/errors/envelope types
  provider_server.py      # route registration
  dml_adapter.py          # backwards-compatible facade
```

Do not move existing modules in the first sprint. Add facades and compatibility imports first, then migrate internals later.

---

## 4. Phase Plan

### Phase 0: Architecture Freeze and Contract Docs

**Objective:** Lock the names, boundaries, and JSON contracts before writing behavior.

**Files:**
- Create: `docs/daystrom-platform-boundaries.md`
- Create: `docs/api/daystrom-cognitive-packet-v1.md`
- Modify: `README.md`

**Steps:**
1. Add a platform overview naming DML, DPM, DCN, DIP.
2. Document component ownership and explicit non-ownership.
3. Document the cognitive packet schema.
4. Mark existing `frontier_pipeline.py` as DIP prototype, not DCN.
5. Add README section: “Daystrom Platform vs traditional RAG.”

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_dml.py -q`
- Manual doc check: every component has clear owns/does-not-own bullets.

**Commit:** `docs: define daystrom platform boundaries and dcn contract`

---

### Phase 1: Shared API Envelopes and Scope Types

**Objective:** Introduce typed request/response envelopes shared by DML/DPM/DCN/DIP.

**Files:**
- Create: `dml_core/daystrom_dml/api_contracts.py`
- Create: `dml_core/daystrom_dml/tests/test_api_contracts.py`

**Implementation targets:**
- `DaystromScope`
- `TokenBudget`
- `AuditInfo`
- `RiskInfo`
- `ReasonCode`
- `ComponentMode`
- `ContractError`

**TDD steps:**
1. Test `DaystromScope` serializes tenant/client/session/instance/thread/project/relationship fields.
2. Test defaults preserve existing `tenant_id="openclaw"` behavior.
3. Test token budget rejects negative limits.
4. Implement minimal dataclasses with `to_dict()` / `from_dict()`.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_api_contracts.py -q`

**Commit:** `feat: add shared daystrom api contract types`

---

### Phase 2: DCN Schema Package

**Objective:** Add inspectable schemas for DCN inputs, plans, cognitive packets, and feedback without changing runtime behavior.

**Files:**
- Create: `dml_core/daystrom_dml/cognition/__init__.py`
- Create: `dml_core/daystrom_dml/cognition/schema.py`
- Create: `dml_core/daystrom_dml/tests/test_cognition_schema.py`

**Implementation targets:**
- `CognitionEvent`
- `CognitionConstraints`
- `IntentAssessment`
- `RetrievalPlan`
- `PersonalityPlan`
- `ToolPlan`
- `WritebackPlan`
- `FrontierPlan`
- `CognitionPlan`
- `CognitivePacket`
- `CognitionFeedback`

**TDD steps:**
1. Test default user-message request yields valid empty/default plan objects.
2. Test all schema objects round-trip through dict/JSON.
3. Test forbidden writeback classes include raw transcript/tool log/secret/prompt scaffold.
4. Implement schemas.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_schema.py -q`

**Commit:** `feat: add dcn schema contracts`

---

### Phase 3: Deterministic DCN Policy v0

**Objective:** Build a rules-first cognition policy that classifies tasks and produces plans without learning.

**Files:**
- Create: `dml_core/daystrom_dml/cognition/policy.py`
- Create: `dml_core/daystrom_dml/tests/test_cognition_policy.py`

**Policy v0 rules:**
- Explicit memory/recall/continue/resume wording -> `retrieval_plan.mode="hybrid"` or `resume`.
- Code/build/run/verify/debug wording -> `needs_tools=true`, `needs_verification=true`.
- Setup/configuration/troubleshooting -> memory likely useful, tools likely useful.
- Casual greeting/simple answer -> no DML retrieval unless post-compaction/continuation metadata exists.
- Appreciation/personality preference corrections -> `writeback_plan.mode="preference_candidate"`, but DPM active-write still controls actual write.
- Side-effecting actions -> medium/high risk and verification/confirmation flags where appropriate.

**TDD steps:**
1. Test “continue the DML hardening work” produces continuity/hybrid retrieval.
2. Test “hello again” does not retrieve semantic memory.
3. Test “run tests and fix failures” recommends terminal/test verification.
4. Test “remember I prefer concise updates” creates a preference candidate, not a DML durable fact directly.
5. Implement deterministic policy.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_policy.py -q`

**Commit:** `feat: add deterministic dcn policy v0`

---

### Phase 4: DCN Controller Facade

**Objective:** Connect DCN policy to existing DML retrieval and DPM overlay rendering to produce a cognitive packet.

**Files:**
- Create: `dml_core/daystrom_dml/cognition/controller.py`
- Create: `dml_core/daystrom_dml/tests/test_cognition_controller.py`
- Modify: `dml_core/daystrom_dml/dml_adapter.py` only if a convenience facade is needed.

**Implementation targets:**
```python
class CognitionController:
    def __init__(self, adapter=None, dpm=None, policy=None): ...
    def observe(self, event, scope=None, constraints=None) -> CognitionPlan: ...
    def plan_context(self, event, scope=None, constraints=None) -> CognitionPlan: ...
    def cognitive_packet(self, event, scope=None, constraints=None) -> CognitivePacket: ...
    def feedback(self, feedback) -> FeedbackResult: ...
```

**TDD steps:**
1. Use fake adapter returning `raw_context` to test DML retrieval is called only when policy requests it.
2. Use fake DPM returning overlay to test bounded DPM overlay is included separately from memory.
3. Test current-turn conflict suppression passes current instruction into DPM rendering.
4. Test cognitive packet includes separate `dcn_plan`, `dml_context`, `dpm_overlay`, and `assembled_context`.
5. Implement controller.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_controller.py -q`

**Commit:** `feat: assemble dcn cognitive packets`

---

### Phase 5: Provider API Routes

**Objective:** Expose DCN APIs without breaking existing DML provider routes.

**Files:**
- Modify: `dml_core/daystrom_dml/provider_server.py`
- Create: `dml_core/daystrom_dml/tests/test_provider_dcn_api.py`

**Endpoints:**
- `POST /api/dcn/observe`
- `POST /api/dcn/plan-context`
- `POST /api/dcn/cognitive-packet`
- `POST /api/dcn/feedback`
- `GET /api/dcn/policy`
- `GET /api/dcn/audit`

**TDD steps:**
1. Test `/api/dcn/observe` returns intent/retrieval/writeback/frontier plans.
2. Test `/api/dcn/cognitive-packet` returns packet version and assembled context.
3. Test `/api/dcn/feedback` accepts outcome signals and returns no-op/recorded status in v0.
4. Ensure existing `/health` and `/api/*` DML routes still pass.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_provider_dcn_api.py dml_core/daystrom_dml/tests/test_provider_server.py -q`

**Commit:** `feat: expose dcn provider api routes`

---

### Phase 6: DIP Rename/Wrapper Boundary

**Objective:** Make the existing frontier pipeline visibly become DIP prototype while preserving backwards compatibility.

**Files:**
- Create: `dml_core/daystrom_dml/inference/__init__.py`
- Create: `dml_core/daystrom_dml/inference/schema.py`
- Create: `dml_core/daystrom_dml/inference/prepare.py`
- Modify: `dml_core/daystrom_dml/frontier_pipeline.py`
- Create: `dml_core/daystrom_dml/tests/test_inference_prepare.py`

**Implementation targets:**
- `DIPPrepareRequest`
- `DIPPrepareResult`
- `InferencePreparationPipeline`
- Compatibility alias to existing `FrontierCompressionPipeline`.

**TDD steps:**
1. Test DIP prepare accepts a DCN cognitive packet.
2. Test existing `FrontierCompressionPipeline.prepare(...)` still works.
3. Test DIP reports `inference_enabled=false` by default.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_inference_prepare.py dml_core/daystrom_dml/tests/test_provider_server.py -q`

**Commit:** `feat: define dip prepare boundary around frontier pipeline`

---

### Phase 7: Feedback Store and Audit Trail

**Objective:** Persist DCN decisions and outcomes for later learning without changing policy yet.

**Files:**
- Create: `dml_core/daystrom_dml/cognition/audit.py`
- Create: `dml_core/daystrom_dml/cognition/feedback.py`
- Create: `dml_core/daystrom_dml/tests/test_cognition_feedback.py`

**Implementation targets:**
- JSONL audit under storage dir: `dcn_audit.jsonl`.
- Decision id on every cognition plan/packet.
- Feedback append with outcome and signals.
- No online learning in this phase.

**TDD steps:**
1. Test every plan has a stable decision id.
2. Test feedback append and audit-tail read.
3. Test no secrets/raw tool logs are persisted when redaction policy is active.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_feedback.py -q`

**Commit:** `feat: add dcn feedback audit trail`

---

### Phase 8: Observe-Only Hermes Integration

**Objective:** Let Hermes/Citizen Snips call DCN for telemetry and planning without changing live prompt behavior yet.

**Files:**
- Modify: `integrations/hermes/plugins/daystrom_dml/__init__.py`
- Modify: `integrations/hermes/plugins/daystrom_dml/plugin.yaml`
- Create/modify: `integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py`
- Create: `integrations/hermes/plugins/daystrom_dml/smoke_dcn.py`

**Behavior:**
- Config flag: `dcn.mode=disabled|observe_only|active_read|active_learn`.
- In observe-only mode, DCN logs what it would have done, but existing gated DML/DPM behavior remains authoritative.
- No memory prompt injection changes in observe-only.

**TDD/smoke steps:**
1. Add smoke that a casual greeting yields no retrieval recommendation.
2. Add smoke that a continuation request yields retrieval recommendation.
3. Verify hygiene smoke still blocks transcript/tool-log pollution.

**Verification:**
- `python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py`
- `python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py`

**Commit:** `feat: add observe-only dcn hermes integration`

---

### Phase 9: Active-Read DCN Integration

**Objective:** Allow DCN to control DML/DPM retrieval/overlay decisions while still using deterministic policy v0.

**Files:**
- Modify: `integrations/hermes/plugins/daystrom_dml/__init__.py`
- Modify: `integrations/hermes/plugins/daystrom_dml/smoke_dcn.py`

**Behavior:**
- DCN decides whether to retrieve DML context.
- DCN decides whether to include bounded DPM overlay.
- Existing memory hygiene filters still run after DCN assembly.
- Fallback: if DCN fails, existing gated DML/DPM path runs.

**TDD/smoke steps:**
1. Casual short turn: DPM bounded overlay only, no active continuity unless explicitly needed.
2. Long-horizon continuation: DML active continuity + relevant retrieval.
3. Explicit current-turn contradiction suppresses stale DPM guidance.
4. DCN failure falls back cleanly.

**Verification:**
- `python integrations/hermes/plugins/daystrom_dml/smoke_dcn.py`
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_controller.py -q`

**Commit:** `feat: enable active-read dcn prompt gating`

---

### Phase 10: Policy Learning v1 — Safe Procedural Adaptation

**Objective:** Add constrained self-improvement for routing/gating policies only.

**Files:**
- Create: `dml_core/daystrom_dml/cognition/learning.py`
- Create: `dml_core/daystrom_dml/tests/test_cognition_learning.py`
- Modify: `dml_core/daystrom_dml/cognition/policy.py`

**Allowed learned policy fields:**
- retrieval query templates
- memory mode preference by task type
- verification requirement by task type
- tool recommendation by task type
- context budget adjustment by task type
- writeback strictness by source/task class

**Forbidden learned fields:**
- identity
- values
- user preferences
- safety boundaries
- autonomy permissions
- secret-handling rules

**TDD steps:**
1. Positive feedback for helpful retrieval increases retrieval likelihood for similar task class.
2. Stale-context feedback decreases trust for the matching memory mode/query template.
3. Tool-failed feedback increases verification or alternate-tool recommendation.
4. Attempt to learn forbidden identity/value field is rejected and audited.
5. Export/import learned policy works.
6. Rollback to deterministic policy works.

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests/test_cognition_learning.py -q`

**Commit:** `feat: add safe dcn procedural learning v1`

---

### Phase 11: DCN Evaluation Harness

**Objective:** Prove DCN improves context quality and reduces memory pollution before treating it as core runtime.

**Files:**
- Create: `scripts/dcn_eval.py`
- Create: `dml_core/daystrom_dml/tests/test_dcn_eval_fixtures.py`
- Create: `docs/dcn-evaluation.md`

**Metrics:**
- retrieval precision proxy
- unnecessary retrieval rate
- stale context inclusion rate
- useful memory inclusion rate
- prompt token savings
- verification omission rate
- writeback pollution rate
- user correction rate, when available
- policy decision latency

**Fixture scenarios:**
1. Casual greeting.
2. Explicit memory recall.
3. Long project continuation after compaction.
4. Code debugging request.
5. User preference correction.
6. Conflicting stale memory.
7. Dangerous/side-effecting action.
8. Frontier prepare request.

**Verification:**
- `python scripts/dcn_eval.py --fixture-set smoke --output-dir /tmp/dcn-eval`
- JSON and Markdown reports are written.

**Commit:** `feat: add dcn evaluation harness`

---

### Phase 12: Documentation, CLI, and Promotion Gate

**Objective:** Make DCN usable and governable outside Python internals.

**Files:**
- Modify: `dml_core/daystrom_dml/provider_cli.py`
- Create: `docs/dcn-operator-guide.md`
- Create: `docs/daystrom-platform-api.md`
- Modify: `README.md`

**CLI commands:**
```bash
dml dcn observe --text "continue the DML work"
dml dcn packet --text "continue the DML work" --session-id abc
dml dcn feedback --decision-id ... --outcome accepted
dml dcn policy show
dml dcn policy export --output dcn-policy.json
dml dcn policy import --input dcn-policy.json
dml dcn audit-tail --limit 20
```

**Promotion gate:**
DCN can become default active-read only when:
- deterministic policy tests pass
- provider API tests pass
- Hermes smoke hygiene passes
- DCN eval shows lower or equal pollution rate than current gated DML path
- fallback path is tested
- operator docs explain disable/observe/active modes

**Verification:**
- `python -m pytest dml_core/daystrom_dml/tests -q`
- `python integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py`
- `python scripts/dcn_eval.py --fixture-set smoke --output-dir /tmp/dcn-eval`

**Commit:** `docs: add dcn operator guide and promotion gate`

---

## 5. Minimal First Sprint Recommendation

Do these first, in order:

1. Phase 0 — Architecture Freeze and Contract Docs.
2. Phase 1 — Shared API Envelopes and Scope Types.
3. Phase 2 — DCN Schema Package.
4. Phase 3 — Deterministic DCN Policy v0.
5. Phase 4 — DCN Controller Facade with fake DML/DPM tests.

Stop before Hermes active integration until the JSON packet shape and deterministic policy tests are stable.

---

## 6. Acceptance Criteria for “DCN Exists”

DCN is real, not just a concept, when all of these are true:

- `CognitionController.cognitive_packet(...)` returns a versioned packet with separate DCN, DML, DPM, and assembled-context sections.
- DCN can decide no-retrieval for casual turns and retrieval for continuation/recall turns.
- DCN can recommend verification/tool policy for build/debug/run tasks.
- DCN can mark preference candidates without directly mutating DPM.
- DCN decisions have decision IDs and audit records.
- Existing DML retrieval and DPM overlay APIs still work independently.
- DIP/frontier prepare can consume a cognitive packet but remains optional.
- Tests prove DCN can be disabled without changing current DML behavior.

---

## 7. Key Design Decision to Preserve

The Daystrom Platform is no longer “alternative RAG.” It is a layered cognition platform:

```text
DML = memory substrate
DPM = personality/self substrate
DCN = adaptive cognitive control substrate
DIP = inference/frontier preparation substrate
```

DML made memory durable. DPM made identity/preference bounded and inspectable. DCN is the next core function: the system learns how to think with memory instead of merely retrieving it.
