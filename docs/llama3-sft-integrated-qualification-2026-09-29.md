# Integrated Llama 3 SFT qualification

The registered candidate passed original planning **2/2** and readiness **3/4**.
It is **not qualified for a new M7 campaign**. The failed dependent-feedback
case is retained, and the historical Nemotron 8/9 campaign remains unchanged.
Nothing was merged.

The tested source is `0b25ba120f62cdec8cd0005f7f2b736f2e403476` on
`codex/remote-vllm-candidate`, with profile
`llama3-8b-instruct-sft-v2-bf16-action-json-v1`.
[Exact-source CI](https://github.com/mmckeen-nv/DML/actions/runs/36607620770)
passed all 22 jobs. The dedicated SFT lane passed 66 tests and the existing
model-input lane passed 1,589, with no skips in either. These CPU checks were
followed by an independently approved GB300 admission: exactly one fixed,
nonsampled forward, finite BF16 logits, frozen parameters, and verified cleanup.

Both qualification stages were declared before inference, independently
approved, frozen together, and run once. Readiness started only after planning
passed. The original model/runtime, sampler, six-step and token budgets,
retrieval, authorization, verifier, and M7 gates were unchanged. There were no
forced calls, retries, output repairs, or additional training.

| Stage | Case | Result | Model calls |
|---|---|---|---:|
| Planning | Paired records | Pass | 2 |
| Planning | Paired records with distractors | Pass | 2 |
| Readiness | Live retrieval | Pass | 2 |
| Readiness | Dependent feedback | Fail | 3 |
| Readiness | Live supersession and read-back | Pass | 4 |
| Readiness | Untrusted injection | Pass | 2 |

The failed case completed two retrievals and produced a final within budget.
After receiving a directory lookup key, the model included the directory's
claim-field name in its next query. That query returned directory records again.
The final asserted a destination fact unsupported by those records and cited
their IDs. The original verifier reported `contradicted_fixture_fact` and
`unsupported_memory_claim`. No destination answer was discarded by the adapter.
This is a semantic retrieval/evidence-use failure, not truncation, a missing
final, or a transport error. Valid JSON and existing citation IDs did not make
the asserted fact correct; the original quality gate rejected it.

Across six episodes, 15 completed model calls consumed 21,642 input and 764
output tokens. There were no model/tool execution errors, action rejections,
timeouts, output-cap hits, missing stop tokens, or unknown usage/effects. Model
latency totaled 201.957 seconds; supervisor wall time was 345.340 seconds.
The authentic quality-failure exit code 1 and failed terminal record are retained;
the owned inference container was removed successfully.

Primary and independent data-only replays both passed evidence integrity and
reproduced the readiness failure. They verified retained requests, token IDs,
grammar, actions, authentic tool feedback and state, without new inference.
The independent grader withheld campaign approval. Replay integrity is not
task success.

The base remains original `meta-llama/Meta-Llama-3-8B-Instruct`, revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, with the fixed final SFT v2 adapter.
The registered consumer uses the pinned vendor template and explicit constrained
DML JSON protocol, not vendor-native tool calling. Direct input/output token IDs
are retained; this establishes no exact-token guarantee for the remote vLLM
HTTP endpoint.

Raw evidence remains under
`/home/nvidia/dml-debug/llama3-sft-integrated-v1-20260929` on the station.
Only compact outcomes and hashes are published. Historical setup/control
failures and provisional metadata remain retained separately; none receives
model-quality credit.

The [compact outcome](artifacts/llama3-sft-integrated-outcome-2026-09-29.json)
records the model/runtime identity, sampling, budgets, metrics, and evidence
hashes. Qualification freeze: `1a3f54896b2fa527989552bce3172c2e0d2f07cd7e159e1b4c653786ad305d61`.
Independent replay: `58010b2a611f9995eae27cb0d4501314f12bc77d98593ff6b0033f66cb44ca14`.

The next candidate should address query construction from dependent feedback
and refusal to assert facts absent from retrieved evidence. Any training must
use newly declared synthetic examples outside the evaluation corpus, with fresh
qualification cases and a new candidate freeze. This failed run must not become
a retry or a relaxed gate.
