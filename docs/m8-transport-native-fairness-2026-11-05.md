# M8 proposed amendment: transport-native fairness comparison — November 5, 2026

**Status: PROPOSED amendment to the M8 manifest, pending independent review. Nothing here is executed or approved.** Supersedes the neutral-protocol arm binding for the fairness comparison. Motivated by retained evidence: [policy-control diagnostic](m8-policy-control-2026-11-05.md) (36/36 retrieve-first, 24/24 needless) and [protocol-competence analysis](m8-protocol-competence-analysis-2026-11-02.md) (0/72 neutral retrieval; H2 supported, H1 dead).

## The problem the current design cannot solve

The manifest's fairness comparison requires the same policy to operate three arms through a neutral protocol: persistent DML, persistent RAG, context-only. Measured fact: the V3-trained policy's use of the retrieve tool is **conditioned on the DML action/transport surface, not on information need** — under the neutral protocol it never retrieves (0/72); under the DML surface it always retrieves (36/36, including when needless). Any neutral-prompt revision that "fixed" retrieval would measure prompt familiarity (H1 dead per the control run). The comparison as designed therefore cannot run: the treatment (memory backend) is perfectly confounded with the transport the policy was trained on.

## Proposed design

**Run every arm through the DML-native transport.** Same policy, same system prompt, same action schema, same sampler, same step limits, same episode machinery — the full qualified surface, unchanged. The arms differ in exactly one component: **what the `retrieve` tool is backed by**.

| Arm | retrieve backend | What it measures |
|---|---|---|
| `dml_memory` | DML receipt store (production ingestion path; development event memory until the production binding is reviewed) | DML durable memory value |
| `rag_memory` | Independent RAG baseline (`m8_memory_baseline` wrapper, same ingestion events, receipt-free) | non-DML retrieval value |
| `context_inline` | none — events embedded in the prompt as untrusted inline source data; retrieve tool present and functional on an empty store | the evidence-fed context floor (the arm the manifest's −2pt non-inferiority gate was derived against) |
| `no_memory` | retrieve disabled/empty (tool present, returns zero records) | **policy floor**: the policy with a dead retrieve. Not the context floor; it carries no context-only gate |

The policy's retrieve-always behavior stops being a confound and becomes a **constant**: every arm retrieves; the experiment measures what the retrieval returns and what the policy does with it. This isolates precisely the factor M8 exists to evaluate — durable memory — under the surface the policy was actually trained for.

**Finalization weakness is measured, not hidden.** Valid-final rate, unsupported-final rate, step-limit rate, and retrieval-loop statistics are first-class per-arm outcomes. A `no_memory` arm that loops on empty retrieval is recorded as such; it is evidence about the policy, not a harness defect (see the control diagnostic's terminate-on-empty amendment probe for the prompt-level mitigation evaluation — its result, positive or negative, is an input to the campaign freeze, not a silent fix).

## What changes and what does not

- **Unchanged:** the qualified policy surface (weights, adapter, template, sampler, pins); the verifier (truth-based, strict — see [verifier-shape analysis](m8-verifier-shape-analysis-2026-11-02.md)); the calibration machinery and gates; the scenario families; the evidence discipline (retain everything, repair nothing).
- **Changed:** the arm binding in the battery/campaign runners (per-arm retrieve backend instead of per-arm protocol); the manifest's fairness section (this amendment); the neutral protocol's role (retained as protocol-contract machinery and for the neutral-consumer development track, but the fairness comparison no longer depends on neutral-protocol fluency).
- **Explicitly not claimed:** that transport-native arms are "the same experiment" as the original neutral design. They answer the question the milestone exists for (does durable DML memory help this policy?) with the confound removed by construction rather than by assumption.

## Fairness properties (for the reviewer to attack)

1. Identical model/prompt/sampler/surface across arms; the only varying component is the retrieve backend, declared per episode.
2. Backend equality checks: RAG and DML stores receive identical event streams through their respective documented ingestion paths (event-clock parity contract `dml-append-request-v2` applies); `no_memory` receives the same stream and discards it.
3. Cost accounting: retrieval cost is **not** constant across arms and this amendment does not claim otherwise: `rag_memory` pays embedding + vector search, `dml_memory` pays receipt overhead, and `no_memory`/`context_inline` have no store and pay ~nothing per retrieve. What IS held constant across arms: the retrieve-always policy behavior (every arm is bound to the trained surface), the tool surface and step limits, the identical fixture event stream, and the accounting rules (consumer-reported token counts; any unknown per-call count makes the aggregate unknown). The cost gates are therefore *comparable* ratios, not constant-cost comparisons: the DML/RAG cost ratio and p95 gates survive as relative gates between the two store-backed arms; "one query per store" amortization (manifest §Cache/order/cost) has no store for `no_memory`/`context_inline` and does not apply to them.
4. Known asymmetry, declared: `no_memory` is not "context-only" in the old sense (no inline evidence) — it is the policy with a dead retrieve, retained as the **policy floor** only. The context floor is the `context_inline` arm (the diagnostic's inline_evidence cells promoted to a full arm through the same transport, per review condition C2): the same events embedded as untrusted inline source data, with retrieve present and functional on an empty store. The manifest's −2pt context non-inferiority gate is re-derived against `context_inline`, never against `no_memory`.

## Decision requested

Approve/reject: replace the neutral-protocol arm binding with the transport-native binding above for all M8 fairness comparisons. If rejected, the fallback is rescoping the M8 protocol gate (documented in the policy-control record); there is no third option that preserves the current policy and the original design, because the original design's central assumption (surface-neutral retrieval behavior) is falsified by retained evidence.

## Review-condition resolution (2026-11-10, against the 2026-11-09 independent review)

Conditions C2–C7 are implemented in `dml_core/scripts/m8_battery/runner_transport_native.py` and its tests; C1 (re-verify the premise under the qualified qwen3.5-9b-sft-v1 profile) is a campaign action, not runner machinery, and remains open.

- **C2/C3 (fourth arm, gate re-derivation anchor):** `context_inline` is a full arm through the same transport: the fixture events are embedded in the prompt as untrusted inline source data (the diagnostic's `inline_evidence` construction) and the retrieve tool is present and functional on an empty store. Inline episodes carry cell tag `context_inline` and are excluded from the persistent-arm statistics grouping (`summary.persistent_arms` is exactly `["dml_memory", "rag_memory"]`; `summary.arm_roles` records `context_floor`/`policy_floor`). `no_memory` is retained only as the policy floor, as relabeled in the arm table above. The −2pt context non-inferiority gate's reference arm is `context_inline`.
- **C4 (retrieve-return parity):** both persistent arms present retrieve with `top_k` maximum 8 (the manifest fixed-candidate cap), declared per episode in the `episode_started` envelope (`tool_top_k_max`, fixed-protocol validated) and recorded per episode as `retrieve_top_k_cap`; an over-cap retrieve is a proven no-dispatch validation rejection with model feedback (`top_k_cap_exceeded`), never a silent clamp. Presentation order is each backend's own retrieval rank; presentation fields are identical (the bridge's single `_public` shape). Every episode records `return_parity_digest` (sha256 over the canonical presented-record list shape: presentation-order field sets + requested top_k + returned count per retrieve call; backend-assigned id values excluded so the digest is comparable across arms); tests assert structural parity between the dml/rag arms on the same query.
- **C5 (first-action pairing):** `first_retrieve_query` is recorded per episode; `summary.first_retrieve_query_checks` holds the per-scenario/seed identity check across the memory-bearing arms, with any divergence retained as an explicit difference note (harness-defect candidate), never silently dropped.
- **C6 (abstention cause):** every `empty_claims_final` episode records `abstention_cause` in {`no_records_returned`, `records_returned_no_support`, `no_retrieve_call`}, computed from the episode's retrieve calls and presented ids; per-arm counts are in `summary.per_arm[*].abstention_causes`.
- **C7 (cost claim):** the cost-constancy claim is struck; see fairness property 3 above.
