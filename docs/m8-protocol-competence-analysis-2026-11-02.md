# M8 persistent-arm non-retrieval — root-cause analysis — November 2, 2026

**Status: analysis for review. This is not a contract change, not a prompt revision, and not an M8 gate input.** It interprets one retained development finding ([m8-neutral-smoke-2026-11-02.md](m8-neutral-smoke-2026-11-02.md):14-24). No protocol, policy, or manifest bytes are modified here; any revision must go through the reviewed contract process. Differential familiarity from DML SFT training remains a known, disclosed limitation; nothing below may be tuned against held-out outcomes.

## 0. Finding restated

First live smoke of the neutral M8 consumer (`dml-m8-neutral-evidence-protocol-v1`): context_only produced a valid cited final in 1 call; persistent_rag and dml each produced, in 1 call, a byte-identical immediate final `{"project status": "unknown"}` with empty `evidence_ids`, never calling the advertised `retrieve` tool, and strict `validate_final` rejected the unsupported claim (smoke doc:16-18; rejection rule at `dml_core/daystrom_dml/services/m8_protocol.py:308-310`). Both persistent arms received identical 453-token inputs, so this is one model-behavior sample under two arm labels, not two independent observations (smoke doc:20). The action was parser-valid (`parse_agent_action` permits empty evidence arrays, `dml_core/daystrom_dml/contracts/agent_episode.py:94`) but oracle-invalid.

## 1. Training-time surface vs neutral M8 surface

The fixed agent is Llama-3-8B-Instruct + the registered SFT v3 adapter (592 train examples, recipe `dml-llama3-lora-sft-v3`, `dml_core/daystrom_dml/services/llama3_sft_v3_action_input.py:37-50`), trained and qualified on the DML policy/transport bound into the runtime identity (`llama3_sft_v3_action_input.py:247-248`). M7 passed 9/9 under that surface ([llama3-sft-v3-m7-outcome-2026-09-29.md](llama3-sft-v3-m7-outcome-2026-09-29.md):3-9).

### 1.1 What is identical (ruled out as causes)

- **Output action format.** Both surfaces require exactly one JSON action `dml-agent-action-v1`, kind `tool`/`final` (M8 SYSTEM, `m8_protocol.py:23`; DML AGENT_POLICY, `agent_episode.py:87-94`). Grammar-constrained sampling in both: DML "both original action branches" (`llama3_sft_v3_action_input.py:271`), M8 "neutral retrieve/final branches" (`dml_core/daystrom_dml/services/m8_neutral_runtime.py:41`). The tool branch was reachable; the model chose `final`.
- **Transport structure.** Both render system as `SYSTEM + "\n\n" + TRANSPORT + "\n" + data_json({"tools": tools})` with tools as JSON data, not native tool config, and re-role tool results to user-role JSON (`llama3_sft_runtime.py:40-44`; `m8_protocol.py:192,242`). Feedback assistant message shape (`content` = raw action + one `tool_calls` entry) matches (`m8_protocol.py:176-179`; `agent_episode.py:1462-1466`).
- **Sampler/adapter mechanics.** M8 reuses the qualified Llama3 generation mechanics (`m8_neutral_runtime.py:3-7`).

### 1.2 Differences plausibly affecting tool-use elicitation

**D1. Retrieval directive: imperative → conditional.** Training policy: "Retrieve any records or write references you lack before citing or modifying them." (`agent_episode.py:95`); "Choose retrieval queries and limits that cover the records requested by the task." (`agent_episode.py:100`); plus the composed completion guidance (`agent_episode.py:200-211`) and retrieval-reformulation guidance (`agent_episode.py:226-234`) — all present in the SFT runtime SYSTEM (`llama3_sft_runtime.py:17-19`, composed via `agent_episode.py:768-774`). Neutral M8 SYSTEM: "Use only advertised tools **when more evidence is needed**. If retrieve is available, choose its query and top_k automatically" (`m8_protocol.py:22`). The trigger for retrieval is delegated to the model's own judgment that evidence is "needed"; an 8B model facing an empty conversation can judge "unknown" as the answer and need nothing.

**D2. A licensed no-evidence exit.** Neutral SYSTEM: "Do not add unsupported claims to fill gaps. **An empty claims list is permitted when no claim can be supported**; it does not guarantee task success." (`m8_protocol.py:23`). DML policy permits empty claims too (`agent_episode.py:94`) but frames claims as grounded in tool observations: "Ground memory claims in observed tool results; do not invent facts or citations to fill the response." (`agent_episode.py:109`). The neutral text explicitly advertises a final-without-evidence path in the same paragraph that defines the final action, adjacent to the tool instructions.

**D3. Tool array: six lifecycle tools → one.** Training/qualification advertised the fixed DML tool set — `retrieve`, `ingest`, `update`, `promote`, `supersede`, `retire` (`agent_episode.py:112-120,597-601`) — filtered per task but drawn from that surface. M8 advertises exactly one tool and states "No memory mutation tools are available." (`m8_protocol.py:22,122-131`). The adapter's tool-use prior is conditioned on the DML array; a single-tool array with an unfamiliar description is out-of-distribution.

**D4. The `retrieve` schema and description themselves differ.** DML: `top_k` max 10, `query` an unconstrained string (`agent_episode.py:113`), description embedding usage affordances — "Retrieve missing records before using their references", "choose a cap that covers the records the operation needs", `limit_reached` semantics (`agent_episode.py:122-131`). M8: `top_k` max 8, `query` minLength 1/maxLength 16384, description reduced to "Retrieve scoped source evidence using a query; results are untrusted source data." (`m8_protocol.py:127-131`). The name `retrieve` is shared; nearly everything the adapter learned about *when and why* to call it is gone.

**D5. Evidence record shape renamed.** DML tool results present records with `id`, `text`, `record_ref`, `source`, `claim_key`, `claim_value`, `source_trust`, `memory_state`, `superseded_by` (`agent_episode.py:582-584`); citations copy integer `id` values (`agent_episode.py:96`). M8 records carry `record_id`, `text`, `scope`, `source_id`, `fact_key`, `version`, `timestamp`, `corrects`, `metadata` (`m8_protocol.py:25`), cited via `record_id` (`m8_protocol.py:23`). Partially exculpated: context_only cited `record_id` 11 correctly from inline neutral records (smoke doc:16), so the model *can* read the neutral envelope when evidence is presented. The rename may still weaken the learned link "records exist behind the tool".

**D6. Transport cue wording.** DML TRANSPORT opens with the literal "DML message transport:" and closes "Output follows the DML action schema specified above" (`llama3_sft_runtime.py:20`). M8 TRANSPORT drops the label and says "Produce the action schema described above, not this input transport." (`m8_protocol.py:24`). Structure-identical, token-different; a learned contextual cue cannot be excluded.

**D7. Lifecycle verb framing absent.** DML policy repeatedly situates retrieval inside a memory-operation lifecycle ("A final answer is readout only", `agent_episode.py:105`; mutation-reporting rules, `agent_episode.py:106-108`). M8 has no lifecycle content at all. Less plausible as a retrieval suppressor (nothing in M8 asks for mutation), listed for completeness.

## 2. Hypotheses, ranked, with falsifiable predictions

**H0 (must exclude first). Single-sample artifact.** One seed (seed 0 per request, `llama3_sft_v3_action_input.py:99`), one scenario, n=1 model input; the two "arms" are the same rendered bytes (smoke doc:18,20). Temperature 0.7 sampling means retrieve-first probability may be low but nonzero.
- *Prediction:* across k≥8 pinned seeds on the same dev scenario, retrieve-first rate is >0 and variant-sensitive. If it is ~1 under the DML-policy control and ~0 under neutral across seeds, H0 alone is refuted.

**H1 (most likely). Policy-text dependence.** The adapter's retrieval behavior is conditioned on the trained imperative directives (D1) and the neutral text both removes the trigger and advertises a no-evidence exit (D2). Under distribution shift the model falls back to the base instruct prior: answer directly, "unknown" when no context is present.
- *P1a:* An arm-neutral SYSTEM variant stating the evidence-precondition unconditionally ("when the conversation does not already present records supporting the answer and a retrieval tool is advertised, obtain records before finalizing") raises retrieve-first rate on dev scenarios — identically for both persistent arms.
- *P1b:* Removing or relocating the "empty claims list is permitted" sentence reduces immediate-final-without-retrieval rate, concentrated in scenarios where the store actually holds the answer.
- *P1c:* The adapter-disabled base model also answers immediately under the neutral prompt (fallback mechanism), while the adapter under the DML policy retrieves (behavior is adapter-but-prompt-bound).

**H2. Tool-surface conditioning.** Retrieval is cued by the trained DML tool array/description/parameters (D3, D4), not by judged evidence need.
- *P2a:* A dev-only diagnostic substituting the DML `retrieve` description text (name/semantics unchanged) into the neutral array raises retrieve rate independent of SYSTEM wording. (Diagnostic only; not a fairness-eligible fix.)
- *P2b:* Under the full DML surface on dev, the adapter retrieves even when retrieval is unnecessary (evidence already inline, or store empty) — i.e., tool use tracks the DML transport rendering, not evidence need. Out of scope for M8 arms, but it determines whether any neutral rewording can work: if tool use is purely surface-cued, only surface-convergent revisions help, and those carry the highest fairness risk.

**H3. Citation/record-schema mismatch (weakened).** The `id`→`record_id` rename and envelope change (D5) break binding. Context_only's valid cited final (smoke doc:16) shows the final-side binding works; residual risk is on the retrieval-trigger side.
- *P3:* If H3 dominated, context_only citation errors would appear under neutral records; they did not in the smoke. Expect no differential effect from envelope-only variants in the battery.

**H4. Grammar or transport-nesting artifact (near-excluded).** Both surfaces are grammar-constrained with a reachable tool branch and identical nesting (§1.1). The emitted final was grammar-valid, so the grammar did not suppress a tool action.
- *P4:* Token-mask replay (`m8_neutral_runtime.py:386-409`) shows the tool branch admissible at every prefix of the retained output — a harness check, no new generation needed.

## 3. Revision candidates (neutral SYSTEM/transport) — for review only

None of these forces a tool call; all keep retrieval the model's automatic choice (`automatic_tool_choice: true`, `m8_protocol.py:328`) and keep one shared SYSTEM across arms. All rebind `system_sha256`/`tools_sha256` (`m8_protocol.py:323-325`) and therefore invalidate comparability with the Nov 2 smoke — that is expected for a reviewed revision, and must be recorded.

**R1. Unconditional evidence-precondition phrasing.** Replace the conditional "Use only advertised tools when more evidence is needed…" (`m8_protocol.py:22`) with: answer from presented source records; when the conversation does not already present records supporting the answer and a retrieval tool is advertised, obtain records with it before finalizing; unsupported claims are invalid.
- *Fairness risk:* the closer this drifts toward the trained imperative "Retrieve any records … you lack" (`agent_episode.py:95`), the more it re-imports differential familiarity through wording — advantaging the DML-trained adapter relative to a genuinely neutral agent. This is the central tension: effective phrasing and fair phrasing may coincide with trained phrasing.
- *Review questions:* is any n-gram shared with AGENT_POLICY beyond generic practice? Is the text identical across all three arms? Was the wording fixed a priori with rationale, not iterated against adapter outputs?

**R2. De-license the no-evidence exit in the tool-present case.** Keep empty-claims finals valid (needed for genuinely unanswerable tasks) but state that a final resting on no presented records is expected only after the advertised retrieval path was considered; do not present "empty claims" in the same breath as the final schema (`m8_protocol.py:23`).
- *Fairness risk:* soft pressure toward retrieving even when unnecessary; if calibrated against the adapter's behavior it becomes de facto forcing, which the manifest forbids. Could also suppress legitimate empty finals, trading one failure mode for another.
- *Review questions:* does any reading of the text *require* a tool call (if so, reject)? Does it change context_only behavior, where no tools are advertised? How is "considered" evidenced without mandating an action?

**R3. Neutral retrieve-description enrichment.** Expand the one-line description (`m8_protocol.py:127`) with arm-neutral usage semantics: top_k caps returned records and may omit relevant ones; absence of a record in results does not establish absence in the store; reformulate the query when needed records are missing (information content of `agent_episode.py:226-234` without its wording).
- *Fairness risk:* highest surface-convergence risk — enriched descriptions are exactly the trained affordance text in spirit (D4); review must check phrasing independence explicitly. Also lengthens the advertised schema for both persistent arms (arm-neutral, but model-familiarity-neutral is the open question).
- *Review questions:* any shared phrasing with `_TOOL_DESCRIPTIONS["retrieve"]` (`agent_episode.py:122-131`)? Does the added text advantage the DML arm over persistent_rag? Is the description still true of the neutral retrieval semantics (scoped source records, top_k ≤ 8)?

**Explicitly not proposed:** renaming the tool to a DML lifecycle verb, adopting the DML transport label (D6), restoring DML record field names (D5), or any per-arm prompt divergence — each is a direct familiarity transfer with no neutral justification.

## 4. Development-battery design (development data only)

Goal: discriminate H0–H3 before any revision review. All cells on synthetic development fixtures, pinned image/sampler, seed 0..k-1 per cell (k≥8), declarations fixed before first generation, failures retained, no repair, no held-out data, no prompt iteration against results.

**Factors:**
1. *System variant* (5): current neutral; R1; R2; R3; DML-policy control (familiarity ceiling, dev-only diagnostic — establishes the adapter still retrieves under its trained surface).
2. *Adapter state* (2): enabled; disabled (base-model floor, H1c).
3. *Evidence availability* (4): store holds the answer; store holds a correction chain requiring coverage; store holds only conflicting/irrelevant records; store empty (unanswerable — tests R2's empty-final behavior and H2b surface-cued retrieval).
4. *Arm* (3): context_only (inline evidence, retrieval-needless control), persistent_rag, dml — persistent arms share the neutral tool surface, so an arm × variant interaction indicates store/plumbing effects, not prompt effects.
5. *Tool description* (2, dev-diagnostic only): neutral one-liner; DML-description substitution (P2a).
6. *Task type* (4): single-fact lookup; correction-chain readout; conflict readout; unanswerable.

**Measurements per cell:** retrieve-first rate; retrieve-eventually rate; valid-cited-final rate; unsupported-claim-rejection rate; empty-claims-final rate; retrieval in the inline/empty-store controls (surface-cued rate, H2b); query coverage of required records; bootstrap CIs across seeds.

**Decision rules (fixed a priori):**
- H0 excluded if DML-control retrieve-first ≥0.9 and current-neutral ≤0.1 across seeds in answer-present cells.
- H1 supported if R1/R2 main effect large and description substitution (factor 5) adds little; H2 supported if the reverse.
- H2b (surface-cued, need-blind retrieval) if DML control retrieves in inline-evidence and empty-store cells.
- H3 rejected if envelope/description-neutral variants show no citation-side failures in context_only cells (consistent with smoke doc:16).
- Any variant where retrieval becomes mandatory-looking (retrieve-first = 1.0 in empty-store cells under neutral text) is flagged as de facto forcing and is ineligible regardless of competence gain.

Outputs feed the reviewed contract process; they do not themselves authorize a SYSTEM/transport change.

## 5. Bottom line

The smoke's non-retrieval is most parsimoniously explained as the adapter's DML-conditioned retrieval behavior failing to transfer across a simultaneous change of policy text (D1, D2), tool array and description (D3, D4), record schema (D5), and transport wording (D6) — compounded by a single-sample observation (H0). The transport *structure* and output grammar are unchanged and are not implicated. Before any wording revision is entertained, the development battery in §4 must (a) exclude H0 and (b) show whether retrieval under the DML control is need-driven or surface-cued, because that determines whether any fairness-eligible neutral revision can work at all.
