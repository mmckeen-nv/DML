# Exact final-input companion hardening

This seventeenth serial gate closes milestone 3 at the reviewed-source gate of the
[finite release ledger](production-remaining-work-2026-09-18.md). The
[model-input contract](model-input-contract-v1.md) defines the Python-only local
Transformers companion. Independent review accepted **9.6/10**, with no blockers
for this narrow boundary, and the root integration run on initial source
`28de7fe` passed. Its first published CI run, 316, exposed Windows test portability
failures described below. Independent re-review accepted the test-only correction
at **9.6/10**; corrected-source publication and exact-commit CI remain **pending
at this source snapshot**, with outcomes to be recorded in PR #118.
This reviewed-source closure is not a merged release or a claim that
the corrected source has passed remote CI.

The scope is the complete model-input handoff already required by milestone 3.
The memory profile remains a candidate with nine HTTP routes and no generation;
its retrieval counts remain estimates. The companion compiles all final messages,
tool definitions and framing with the pinned real tokenizer, reserves output
capacity, and dispatches the immutable token IDs directly after integrity checks.
It adds no new release milestone and makes no maturity promotion.

## Implementation boundary

The admitted runtime is a built-in GPT-2 model on CPU in float32, a fast tokenizer
loaded from verified local bytes, and the exact full-JSON messages/tools template.
Snapshot verification checks the manifest, model/config/tokenizer/template bytes
and pinned runtime versions, then retains a private verified clone. Remote loading,
custom model code, pickle checkpoints, alternate architectures and additional
template or generation-option channels are outside the boundary.

Immutable request/identity/compiled contracts preserve the complete input and
output reservation. Compilation uses real `apply_chat_template(tokenize=True)`
output and rejects overflow before inference. The consumer authenticates its own
compiled artifacts and checks runtime drift before passing the stored token IDs
to greedy generation. There is no second render or tokenizer pass on execution.
Tool history is represented and counted as data; logical pairing and tool execution
are not established by this boundary.

## Independent evidence and integration acceptance

The independent selection on the initial reviewed source passed **256 tests**, with **0 failures, errors
or skips**, in **5.59 seconds**, with one Starlette deprecation warning:

| Selection | Passing cases |
| --- | ---: |
| Immutable contracts | 109 |
| Local snapshot verification | 75 |
| Consumer runtime | 8 |
| Real model integration | 7 |
| Independent adversarial cases | 57 |

A generated tiny GPT-2 fixture and real tokenizer exercise exact fits, overflow
rejection and the model's actual input IDs. The independent complete-input oracle
manually renders full JSON including Unicode and HTML-like characters, then checks
the actual token IDs. Cases also reject request/artifact mutation, a foreign
consumer, incompatible or drifted runtime, malformed snapshots and hidden
post-count input additions. These real-model checks go beyond test-double handoff
assertions while making no model-quality claim.

The root full suite on `28de7fe`, before the test-only portability correction,
passed **3,598 tests**, with **9 skips** and **3 warnings**,
in **146.54 seconds**. Its JUnit report contains 3,607 total cases, zero failures or
errors, and all **256 new model-input cases with zero skips**. These focused and
full-suite selections overlap and are not additive totals. Maintained and strict
new-surface Ruff, mypy over **66 source files**, Hermes hygiene and the diff check
passed. The [review record](artifacts/model-input-review-2026-09-18.json) preserves
the source hashes, independent acceptance and separately attributed root evidence.
Runtime sources are identical across the correction, but this result is not a
new full-suite run on the corrected test commit. Corrected-source publication
and exact-commit CI remain pending at this source snapshot; their outcomes will
be recorded in PR #118.

The dedicated `model-input-cpu` CI job uses Ubuntu, CPython 3.12 and the exact
`model-input` dependency extra after installing PyTorch 2.8.0 from its official
CPU wheel index. `REQUIRE_MODEL_INPUT_TESTS=1` makes missing or mismatched optional
dependencies a failure. The JUnit acceptance check separately requires a positive
test count, at least one case from each of the five selected modules (including
real runtime and integration cases), and zero failures, errors and skips. Its result is preserved as its
own evidence artifact.

Base and portability jobs retain lightweight dependency installation. Their
optional real-model skips are not passing real-model evidence. Maintained lint
and type checking cover the new contract and service modules. Final acceptance
must identify the reviewed source and keep focused, full-suite and remote CI
results distinct. The prior stage-16 source-hashed review record is historical
and is not rewritten by this gate.

## Post-review Windows test portability correction

The first published CI run, **316**, failed in its **Windows Python 3.13** job.
Windows Python 3.10 was **cancelled by matrix fail-fast**, not recorded as a
failure. The other **eight jobs passed**, including the dedicated CPU lane with
all **256 model-input cases passing and zero skips**. Six pure snapshot cases
used text-mode fixture writes that translated the required LF bytes to CRLF;
the runtime then correctly rejected the altered exact-template fixture. Separately,
pytest's autogenerated ID for an oversized raw-JSON parameter exceeded the Windows
32,767-character environment-value limit when assigned to `PYTEST_CURRENT_TEST`,
producing two setup/teardown errors before that case could run normally.

The narrow correction writes canonical fixture content as explicit UTF-8 bytes
in the pure snapshot tests and shared real-model fixture, and gives the oversized
JSON case a short explicit parameter ID. The complete oversized payload remains
in the test. Runtime code, input limits and admission semantics are unchanged.

The independent grader accepted these test-only changes at **9.6/10**, with zero
blockers. The corrected five-module selection passed **256 tests**, with **0
failures, errors or skips** and one Starlette deprecation warning, in **5.62
seconds** (JUnit duration 5.370 seconds). Its distribution remains 109 contract,
75 snapshot, 8 runtime, 7 integration and 57 adversarial cases. The initial
full-suite evidence remains attributed to `28de7fe`, before this fixture patch;
no corrected-commit full-suite run is claimed. Corrected-source publication and
exact-commit CI remain pending at this source snapshot, with outcomes to be
recorded in PR #118. Milestone accounting and candidate maturity are unchanged.

## Remaining limits and accounting

This gate does not establish language-model quality, task success, training
provenance, correct tool-call sequencing, arbitrary external provider behavior,
durable response replay, crash recovery or production capacity. It also does not
make compiled artifacts durable or exactly-once execution receipts. A tiny model
demonstrates operational correctness of the budget and handoff boundary.

Milestone 3 is closed at the reviewed-source gate after independent acceptance and
the final root integration pass. **10 milestones remain: 8 for the first release
and 2 deferred**.
Milestone 4, crash recovery and filesystem qualification, is next. The repository
remains alpha, the companion is listed only as candidate `exact-model-input-v1`,
and `production_ready=false`.
