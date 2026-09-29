# M8 fairness manifest and proposed acceptance — September 29, 2026

**Preparation contract; not an execution freeze. M8 remains open.** The [machine-readable manifest](artifacts/m8-fairness-manifest-2026-09-29.json) records known identities, proposed settings and explicit unresolved execution bindings. No new training, model search, development-model pilot or held-out evaluation was performed. The [baseline audit](m8-baseline-audit-2026-09-29.md) verifies the existing store and identifies the remaining harness work. M7 and all historical campaigns remain unchanged.

## Claim and fair comparison

The proposed claim is a meaningful quality advantage over conventional persistent RAG, without material regression against context-only, **for the fixed deployed DML-trained agent on externally ingested, scoped evolving-memory queries**. It is not a claim about adapter-independent memory causality, general model superiority or autonomous lifecycle editing. The identical V3 adapter stays enabled in all three arms. Differential familiarity from DML training remains a limitation even with identical weights.

| Arm | Information and interface | Persistent behavior |
|---|---|---|
| Context-only | Original scoped chronological events in a bounded in-memory buffer, including source IDs/provenance; no retrieval tools | No external memory. Keep newest whole records fitting the exact context budget, in chronology; no query-aware selection or hidden lookup. |
| Conventional persistent RAG | Same input events and metadata; automatic `retrieve(query, top_k)` and dependent follow-up searches | Reuse independent `SQLiteBaseline`, WAL/FULL, exact cosine plus fixed recency and ID tie-break. Add separately reviewed metadata/current-view and exact-token presentation wrappers. |
| DML | Same input events and automatic retrieval interface | Supported DML validation, authorization and retrieval rules; genuine lifecycle ingestion through the existing boundary. |

The primary comparison is query-focused. Authenticated non-model ingestion drivers consume identical explicit event fields, including scope, source, timestamp, version and correction links. Only explicit same-source version/correction identity permits current-view replacement; history is retained. RAG current-view eligibility applies before ranking, top-k and compaction, with immutable history stored separately; filtering obsolete rows only after top-k would unfairly reduce usable results. Conflicting independent sources remain visible. Drivers cannot use answer keys, resolve ambiguous conflicts for one arm, invent receipts, or silently repair inputs. Ingestion, index maintenance, checkpoint and rebuild costs belong to each arm. No arm exposes model-owned mutation tools in this primary comparison; M7 remains the evidence for that separate capability.

Use a common neutral task/trust/final policy and identical persistent-arm retrieval schema. Context-only instructions describe inline evidence and never require unavailable tools. Shared final claims/citations validation accepts only evidence actually presented to that arm, including inline evidence for context-only. No DML-specific receipt or operation ID is required for baseline success. Require factual correctness and valid citations; invalid prose, unsupported claims and malformed finals fail without conversion or repair. Untrusted source content cannot override system instructions or scope permissions.

The M7 renderer hardcodes DML's system policy and exact tool definitions. **It cannot serve these arms unchanged.** A separately versioned neutral M8 interface must preserve exact token IDs, sampling, immutable dispatch, authentic feedback and strict final validation. All arms must demonstrate automatic retrieval where available and valid cited finals on development controls. A baseline protocol mismatch is a harness qualification failure, not evidence of DML value. The new interface needs exact-source CI and independent admission; it does not inherit the old runtime's qualification by name.

## Fixed candidate and resources

The JSON pins Meta Llama 3 8B Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, V3 adapter, tokenizer, vendor template and container hashes. All arms use the same BF16 base/FP32 adapter and sampler: temperature .7, top-k 20, top-p .8, min-p 0, repetition 1, frequency/presence 0, seed 0 reset per request, CPU torch.multinomial. Tool selection stays automatic; no retries, forced completion or output repair.

Per query: six model calls, 256 output tokens/call, 1,536 total output tokens, 32,768 total input tokens, 8,192 request context tokens and 300 seconds. These are ceilings, not required consumption. Count every rendered system/tool/evidence/feedback token using actual tokenizer IDs; reserve output space before dispatch. Both retrieval responses have a 2,048-token metadata-inclusive ceiling and top-k at most eight, with identical whole-record presentation rules. Remove the old RAG 220-character clipping and characters/4 estimate at the wrapper boundary. No model-based compaction is proposed.

Use the already cached production-default `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, for both persistent arms: CPU float32, four threads, batch one, mean pooling, 384 dimensions, L2 normalization, identical text/chunk boundaries, at most 192 content tokens per chunk within the 256-token embedding limit including metadata/special tokens. Reject silent truncation. Its station inventory is hashed in the JSON; cache presence is **not** qualified model provenance or runtime parity. This replaces M7's synthetic lexical fixture embedding only in the separately qualified M8 configuration. No model was downloaded or loaded for this preparation.

Target the existing GB300 station, driver 590.48.01, inference concurrency one. Timing qualification requires dedicated resources or measured absence of competing work, fixed CPU allocation and recorded utilization, clocks, power, temperature and memory. Do not change the running Nemotron service during preparation. Interference during a frozen run is retained and can invalidate the campaign; it cannot justify selectively rerunning a slow arm.

## Workloads, leakage and allocation

Six equally weighted families: changing facts, explicit corrections, conflicting independent sources, dependent lookups, isolated project scopes, and stable/memory-adverse controls (including distractors and untrusted instructions). Include both full-context-feasible and beyond-window histories; report them separately. Proposed event-text bands are 2,048–4,096 and 16,384–24,576 generation-tokenizer tokens. The “fits” classification must also verify the complete rendered context-only request, including all metadata/system/task text and reserved output, fits without dropping records; event-text length alone is insufficient. Exact record envelopes and generator bounds must be sealed before pilot.

Propose **216 development scenarios** (36/family) and **4,320 distinct held-out scenarios**, each paired across all three arms: 12,960 arm episodes. Allocation is six families × two pressure bands × three cache conditions × six arm orders × twenty distinct scenarios. The independent unit is a unique project/story/store with one scored query; never count repeated cache/order runs of one story as independent cases. If multi-question stories are introduced, cluster by story and redo the prospective calculation before freeze. Subgroup reporting does not establish adequately powered subgroup superiority.

An independent auditor who did not author training data must seal fresh held-out cases, truth, case IDs and seeds before candidate development-model pilot; implementers receive development data and interface/distribution specifications only. Hold-out files must not be mounted for tuning. Audit all V1/V2/V3 train/dev sets, rejected and source-order serializations, masked histories, M7 corpus/campaigns, and prior Qwen/Nemotron/Llama diagnostics. Check normalized exact overlap, values/IDs, near-text n-grams, entity-renamed templates/event graphs, and semantic similarity by human review. New random values alone are insufficient. Generic motifs necessarily overlap; do not claim unseen reasoning skills. Publish compact audit hashes, not raw station data. **New cases have not yet been created or cleared for leakage.**

No invalid-case substitutions or outcome-based exclusions are allowed after freeze. A fixture defect, missing accounting or source drift invalidates the affected acceptance claim; retain every outcome and keep M8 open. Any revised campaign requires a new independently reviewed freeze and is reported separately.

## Cache, order and total cost

Assign each scenario one sealed cache condition and one of all six balanced arm orders. Each arm receives the same events, query, logical time and sampling rule, with separately owned stores/caches.

- **Cold process:** fresh worker/model/client, empty application/KV/embedding caches. Record OS cache conditions; do not claim cold physical storage without measuring it.
- **Warm resident:** immutable model weights may stay resident; reset KV, RNG and result cache for each attempt. Private retrieval/embedding caches cannot cross arms. Fixed block size, a non-evaluation warmup input and equal allocation of shared initialization cost must be specified before execution.
- **Rebuild:** reconstruct from the identical event log into a fresh owned store/buffer before query; charge original ingestion plus reconstruction, checkpoint and restart costs. No privileged answer-state snapshot.

Report each condition separately. Query latency runs from query acceptance to final/terminal failure, including retrieval, serialization, model turns and validation. Full lifecycle latency/cost includes allocated startup, all event ingestion and maintenance, rebuild where applicable, query and teardown. Use **one query per store** as the primary amortization horizon; no favorable post-hoc horizon selection. Failed attempts count; deadline timeouts retain their full observed deadline and any incomplete lifecycle cost is unknown, not zero.

Record LLM input/output/maintenance tokens and IDs, embedding input tokens/calls/CPU time, retrieval/ingestion/compaction/checkpoint/rebuild/warmup costs, CPU core-seconds, GPU time, peak RAM, disk/WAL bytes, model load and measured TTFT, errors and timeouts. Report p50/p95/p99 and sample counts. Embedding tokens are separate units, not interchangeable with generation tokens. Token cost per correct success uses **all attempted-episode costs** in the numerator. Zero-success or unknown-cost denominators cannot pass. Report full resource vectors; no monetary claim without a price card. Historical training/development costs are disclosed separately, with identical adapter cost across arms, not hidden in arbitrary amortization.

## Proposed acceptance and inference

Select the original **quality advantage route** before evaluation. The earlier optional 15% token-saving route is not an alternative rescue for this design. Every following condition must pass:

| Required condition | Proposed threshold |
|---|---|
| Paired correctness, DML minus RAG | One-sided 95% lower bound strictly greater than +5 percentage points |
| Paired correctness, DML minus context-only | Lower bound strictly greater than −2 points |
| Absolute DML correctness | Lower bound at least 90% |
| Stale answers on externally declared eligible queries | Upper bound at most 2%; paired excess upper bound at most 1 point against each baseline |
| Unsupported or contradictory claim episodes | Paired excess upper bound at most 1 point against each baseline |
| Deterministic authorization, scope, integrity or instruction-authority violations | Zero |
| DML query p95 | Upper bound at most 120 seconds in each family/cache marginal |
| DML full lifecycle p95 | Upper bound at most 180 seconds in each family/cache marginal |
| Full lifecycle p95 DML/RAG ratio | Upper bound at most 1.25 in each cache condition |
| All LLM tokens per correct success, DML/RAG | Upper bound at most 1.10, pooled with fixed cell weights |
| Family × pressure observed correctness | No DML regression greater than 5 points against either baseline |
| Integrity and accounting | Exact-source/runtime freeze, all attempts/costs retained, both independent evidence replays pass |

Correctness is query-level conjunction of required facts, source-supported citations and valid final structure. A missing final fails. Staleness eligibility is fixed by fixture truth before execution (an obsolete answer opportunity exists), not by the model's chosen search; abstention can avoid a stale claim but still fail correctness. Unsupported/contradictory-claim rate uses all attempted queries; do not drop malformed or failed episodes from correctness. Validators must distinguish justified discussion of an obsolete source from endorsing its obsolete fact.

Use a stratified paired trinomial profile-likelihood method for binary differences, with win/loss/tie probabilities `(q+delta)/2`, `(q-delta)/2`, `1-q`, constrained by `abs(delta)<=q<=1` and fixed cell weights. For cell c use heterogeneous `delta_c, q_c`; the estimand is `Delta = sum(w_c * delta_c)`. Constrain this weighted mean at each tested null while profiling all cell-specific nuisance parameters; do not impose a common effect across the 216 cells. The implementation, nuisance optimization, tolerance, boundary treatment and calibration are **pending execution bindings**, not an assertion of exact coverage. Independently verify all-tie/all-win/all-loss and rare-discordance cases, swapping symmetry, monotonicity, small-N enumeration, and null coverage at the declared margins. Pin conservative boundary rules before data are unblinded. Do not use zero-width naive bootstrap intervals for binary boundary outcomes.

For cost/latency, propose 20,000 fixed-seed (20260929) paired cell-stratified scenario bootstrap resamples, keeping the three arms together. Tail coverage, absolute quantile bounds and zero-denominator treatment require independent calibration before freeze. Unbounded/unstable intervals cannot pass. Report two-sided descriptive intervals separately. Nominal one-sided component tests support a conjunctive intersection-union promotion decision at alpha .05 only after their coverage is qualified; marginal 95% intervals are not a simultaneous 95% region.

The [power calculation](artifacts/m8-power-planning-2026-09-29.json) is normal planning only: with 4,320 pairs, true +10-point DML/RAG advantage and discordance 1 yields about 95% power for the +5-point gate; context equality with discordance .2 yields about 90% power for −2-point non-inferiority. The union-bound joint power for **these two components only** is about 85%; context discordance 1 lowers its marginal power to about 37%. Additional reliability, subgroup, stale and cost gates reduce full-conjunction power. Prospective full-conjunction simulation using development assumptions is mandatory. This sample count supports a conditional planning argument, not guaranteed precision or universal power.

At the hard 300-second ceilings, 12,960 episodes imply up to 1,080 serial inference hours, before setup. Development measurements must supply a realistic resource forecast. Revise and independently review the fixed design before held-out freeze if unaffordable or underpowered; never extend a frozen run until it wins. An inconclusive or failed result keeps M8 open.

## Next gate

Implement only the reviewed neutral interface, baseline wrappers, accounting and statistical machinery; qualify on development cases. Complete independent leakage sealing, exact-source CI, prospective power/coverage calibration and runtime attestation. Then obtain independent execution approval and freeze every source, prompt, dataset, order and resource binding. Run held-out once, retain every failure, replay and grade independently, and publish all compact outcomes with uncertainty and limitations. This preparation document does not authorize that campaign or change the milestone ledger to closed.
