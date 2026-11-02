# M8 neutral runtime qualification plan — November 2, 2026

Continues from the [M8 fairness blockers tranche](m8-fairness-blockers-2026-11-02.md). **M8 remains open; this is a plan plus development bindings, not a live qualification.** No model was loaded and no GPU generation occurred.

## Done in this tranche (local, no-model)

`dml_core/daystrom_dml/services/m8_neutral_runtime.py` binds the reviewed neutral protocol (`m8_protocol`, profile `dml-m8-neutral-evidence-protocol-v1`) to the qualified generation mechanics **without relabeling the M7 renderer**:

- `compile_grammar(tokenizer, arm)` compiles the neutral action schema (`m8_protocol.schema_bytes(arm)`) with the pinned xgrammar 0.1.33 settings (vocab 128,256, stops 128001/128009, strict mode, any-whitespace, single thread, no cache). The M7 schema and DML tool definitions are never consulted.
- `project_output(output_ids, tokenizer, arm)` reuses the exact control/vocabulary checks, then parses through the **neutral** parser only (`m8_protocol.parse_action`): unadvertised tools, context-only tool use and malformed actions fail without repair.
- `NeutralM8AdapterRuntime(EnabledAdapterRuntime)` reuses the qualified loader (BF16 base/FP32 LoRA, frozen, CUDA checks) and the identical sampling loop (grammar bitmask, verified vocabulary/controls, temperature .7, top-k 20, top-p .8, CPU `torch.multinomial`, per-request `fork_rng` seed 0). It accepts only an exact `CompiledM8Request` for its own arm and records the same failure-inclusive evidence fields plus `arm` and `protocol_profile`.
- `neutral_identity(arm, base_identity=...)` derives `dml-m8-neutral-action-runtime-v1:` identities from the protocol policy, base snapshot identity and pinned sampler; never an M7 runtime identity.

Controls: `dml_core/tests/test_m8_neutral_runtime.py` (8 tests, tokenizer doubles, fake xgrammar capture) cover parser projection per arm, embedded-control and malformed rejection, finish/stop accounting, identity determinism/arm separation/non-M7, pinned grammar compiler settings with neutral schema bytes, and exact-request/arm admission. Ruff and mypy clean; the full M8 suite remains green.

## Remaining qualification (station-bound, in order)

1. **Asset attestation (running).** Independent inventory on the station: pinned MiniLM inventory hash, Llama base/tokenizer/template hashes, V3 adapter hashes, container digest, driver — against the fairness-manifest values. Report only; no changes.
2. **Neutral consumer class.** `M8NeutralConsumer` mirroring `LocalLlama3SFTV3ActionInputConsumer`'s authenticated-artifact discipline (manifest verification reuse, tokenizer attestation, HMAC-bound compiled artifacts, single-use consumption, replay validation) but compiling through `m8_protocol.compile_request` and executing through `NeutralM8AdapterRuntime`. Written when the attestation above lands; qualified on the station.
3. **Live development smoke.** Minimal all-arm protocol competence on development controls only: each arm must demonstrate automatic retrieval where available and a valid cited final. Interference-free window, concurrency one, retained failures. A baseline protocol mismatch is a harness failure, not a DML win.
4. **Embedding admission.** Pinned MiniLM CPU float32 runtime for both persistent arms; identity declared per store before any development qualification.
5. **Durable execution and cost instrumentation.** Dispatch timeouts, wall ceilings, and the full metering vector from the manifest, wired into `m8_accounting`.

No held-out cases, no pilot, no tuning. Independent review of the blockers contracts and this binding remains a prerequisite for any freeze.
