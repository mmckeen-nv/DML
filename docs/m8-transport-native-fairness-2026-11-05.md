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
| `no_memory` | retrieve disabled/empty (tool present, returns zero records) | policy without memory (the context floor) |

The policy's retrieve-always behavior stops being a confound and becomes a **constant**: every arm retrieves; the experiment measures what the retrieval returns and what the policy does with it. This isolates precisely the factor M8 exists to evaluate — durable memory — under the surface the policy was actually trained for.

**Finalization weakness is measured, not hidden.** Valid-final rate, unsupported-final rate, step-limit rate, and retrieval-loop statistics are first-class per-arm outcomes. A `no_memory` arm that loops on empty retrieval is recorded as such; it is evidence about the policy, not a harness defect (see the control diagnostic's terminate-on-empty amendment probe for the prompt-level mitigation evaluation — its result, positive or negative, is an input to the campaign freeze, not a silent fix).

## What changes and what does not

- **Unchanged:** the qualified policy surface (weights, adapter, template, sampler, pins); the verifier (truth-based, strict — see [verifier-shape analysis](m8-verifier-shape-analysis-2026-11-02.md)); the calibration machinery and gates; the scenario families; the evidence discipline (retain everything, repair nothing).
- **Changed:** the arm binding in the battery/campaign runners (per-arm retrieve backend instead of per-arm protocol); the manifest's fairness section (this amendment); the neutral protocol's role (retained as protocol-contract machinery and for the neutral-consumer development track, but the fairness comparison no longer depends on neutral-protocol fluency).
- **Explicitly not claimed:** that transport-native arms are "the same experiment" as the original neutral design. They answer the question the milestone exists for (does durable DML memory help this policy?) with the confound removed by construction rather than by assumption.

## Fairness properties (for the reviewer to attack)

1. Identical model/prompt/sampler/surface across arms; the only varying component is the retrieve backend, declared per episode.
2. Backend equality checks: RAG and DML stores receive identical event streams through their respective documented ingestion paths (event-clock parity contract `dml-append-request-v2` applies); `no_memory` receives the same stream and discards it.
3. Cost accounting: retrieval cost is now incurred by every arm (constant), so the cost gates compare backend costs (DML receipt overhead vs RAG), which is the honest cost question.
4. Known asymmetry, declared: `no_memory` is not "context-only" in the old sense (no inline evidence) — it is the policy with a dead retrieve. If the reviewer wants an inline-evidence arm, it exists as a control in the diagnostic (inline_evidence cells) and can be promoted to an arm with the same transport.

## Decision requested

Approve/reject: replace the neutral-protocol arm binding with the transport-native binding above for all M8 fairness comparisons. If rejected, the fallback is rescoping the M8 protocol gate (documented in the policy-control record); there is no third option that preserves the current policy and the original design, because the original design's central assumption (surface-neutral retrieval behavior) is falsified by retained evidence.
