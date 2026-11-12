# M8 closure evidence pack — November 12, 2026

**Status: all gates 1–4 PASS; campaign execution complete pending final statistics; gate 5 (owner execution approval) is the last gate.** Supersedes the 2026-11-06 pack (written when the trained policy was below threshold; M9 has since qualified the qwen3.5-9b adapter, 6/6 gate).

## The question and the design that measures it

Does durable DML memory help the qualified qwen3.5-9b-sft-v1 policy under a fair comparison against a persistent RAG baseline? The neutral-protocol design was falsified for llama3 (surface-conditioned retrieval); the transport-native amendment ([proposal](m8-transport-native-fairness-2026-11-05.md)) puts every arm on the identical DML surface — same prompt block, same frozen tool schema, same event stream — and was independently reviewed ([review](m8-transport-native-fairness-REVIEW-2026-11-09.md)): **APPROVE-WITH-CONDITIONS**, all conditions resolved:

| Condition | Resolution |
|---|---|
| C1 premise re-verification under qualified profile | DML surface 33/36 retrieve-first confirmed (bit-identical to gate); neutral half: qwen3 retrieves 24/24 (premise retracted, design stands on fairness merits) — review doc addendum |
| C2/C3 context floor | `context_inline` promoted to full arm (evidence inline + functional empty-store retrieve); `no_memory` relabeled policy floor |
| C4 return parity | Identical frozen tool schema to all arms (live-caught: schema mutation trips the pinned tool-definition check — fixed 5a2eb5c); parity digest keys on model-requested top_k |
| C5 first-query identity | Recorded per episode; 216/216 + 144/144 identical across persistent arms |
| C6 abstention causes | Pre-registered {no_records_returned, records_returned_no_support, no_retrieve_call} |
| C7 cost claim | Corrected: retrieval cost varies across arms; comparability restated |

## Statistics qualification

- R=400 station grids: default `wilks` lower bound measured mildly anti-conservative (28/36 grid points below nominal) — caught by the calibration program itself.
- Pinned conservative rule `wilks_plus_inflation` (α/2 threshold): **36/36 grid points at or above nominal** at station scale, ~20% width cost. Recorded in [calibration plan](m8-calibration-plan-2026-11-02.md).

## Pre-campaign gates

| Gate | Verdict | Evidence |
|---|---|---|
| 1 Leakage audit | **PASS** | [audit](m8-leakage-audit-2026-11-10.md): 0 overlapping 8-grams (fixtures↔curriculum, adversarial n=6/n=5); C1 hash sweep 0 collisions vs 23 training files, 23/23 manifest sha256 byte-exact (`/home/test/dml-debug/m8-leakage-C1-20261110.json`); C2: 0 overlap sealed corpora↔curriculum |
| 2 Sealed workloads | **PASS** | pilot216 `db8b2ab1…` (216 scenarios, seed 20261110); heldout `4478cd7f…` (144 scenarios, seed 20261112); both regenerated on 133 with digest match; seal verification green; tamper-detection tested |
| 3 Resource forecast | **PASS** | [gates doc](m8-precampaign-gates-2026-11-10.md): pilot 78min/864eps forecast (actual ~80min); GPU 43GB/72GB peak |
| 4 Exact-source CI | **PASS** | full-dependency proxy: 5,756 collected / 4,872 passed / 0 errors / 882 declared skips; only 2 Linux-only renameat2 platform gates on macOS; zero source regressions; all pins inventoried |
| 5 Execution approval | **PENDING** | owner sign-off — this document is the approval record |

## Campaign measurements (qualified qwen3.5-9b adapter, v4 amendment surface, seed 0)

**Dev pilot** (216 scenarios × 4 arms = 864 episodes, `/home/test/dml-debug/m8-pilot-216-20261112/`, zero harness failures):

| arm | valid cited | abstain | unsupported | output tokens |
|---|---|---|---|---|
| dml_memory | 42/216 (19.4%) | 1 | 173 | 24,678 |
| rag_memory | 45/216 (20.8%) | 0 | 170 | 23,236 |
| context_inline (floor) | 0/216 | 140 | 76 | 16,291 |
| no_memory (policy floor) | 0/216 | 216 | 0 | 14,958 |

Paired trinomial (dml vs rag, scenario-paired): wins 6 / losses 9 / ties 201 → δ̂ = −0.0139, inflated-rule bounds [−0.0515, +0.0221] — **the two memory backends are statistically indistinguishable on the dev pilot**. Context-floor gate (the manifest's −2pt non-inferiority gate, re-derived against `context_inline` per the amendment): persistent arms beat the floor by +19.4pt (dml) / +20.8pt (rag) — far above the margin; **memory demonstrably helps**.

**Held-out campaign** (144 scenarios × 4 arms = 576 episodes, `/home/test/dml-debug/m8-campaign-heldout-20261112/`): `_results slot — fill at completion_`

## What remains open by charter

All contracts remain proposals pending independent review; the verifier was never moved to force a pass; every failure in this chain is retained evidence. The context_inline floor's 76 unsupported finals (policy asserting from inline evidence without retrieval) are recorded model behavior, not harness defects.
