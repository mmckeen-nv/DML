# M8 neutral runtime smoke — November 2, 2026

First live execution of the neutral M8 consumer on the pinned station environment. **Development plumbing qualification only: one fixture corpus, one query, no benchmark, no held-out data, no quality claim. M8 remains open.** Raw evidence stays on the station at `/home/nvidia/dml-debug/m8-neutral-smoke-20261102/` (`smoke.json` SHA-256 `f9e136f57d0bab81db785a9caea08de728a4d68cd801f25a5caba67b034d229e`, `smoke.log` SHA-256 `6c1b0fb19de9bbf8332210eed126c587fce6ebdc645dca8e51a382e587e0bace`).

## Setup

- Dedicated GPU window: the two resident non-DML workloads from the [attestation](artifacts/m8-station-attestation-2026-11-02.json) were stopped with the owner's authorization (`docker stop comfyui-flux2 swarm-vision`); GPU verified at 3 MiB / 0% before launch and 1 MiB after exit. Station services otherwise untouched.
- Pinned container `sha256:bf05c8a0…860ce825`, uid 1000, read-only rootfs, M7 campaign's mount/env recipe, exact pinned versions (torch 2.11.0+cu130, transformers 5.6.2, tokenizers 0.22.2, xgrammar 0.1.33, peft 0.21.1, safetensors 0.7.0, apache-tvm-ffi 0.1.9), snapshot at `1956772`-identical checkout plus smoke driver `136c6a4`.
- Corpus: three trusted events in one scope — correction pair (amber→green, bulletin) and independent conflict (red, independent) — ingested through the real arm paths: `ContextBuffer`, `PersistentRAGBaseline` (pinned MiniLM 384-dim embeddings: CPU fp32, mean-pool, L2, batch one, ≤192 content tokens), and `DMLDevelopmentEventMemory` (v2 event-time ingestion; embeddings as above). Query: "What is the project status?", query clock 2000.0.
- Per arm: `M8NeutralConsumer` compile → single-use execute → strict neutral parse → scripted retrieval dispatch on tool actions → `validate_final` with a fixture domain verifier. Max 6 calls/arm; failures retained, no retries, no repair.

## Results (wall 141.8s total, incl. three model loads)

| Arm | Calls | Outcome |
|---|---|---|
| context_only | 1 | **Valid cited final**: claim `status=green` citing record 11; presented [10,11,12] inline |
| persistent_rag | 1 | **Retained failure**: immediate final `{"project status": "unknown"}` with empty `evidence_ids` — no `retrieve` call; strict validation rejected the unsupported claim |
| dml | 1 | **Retained failure**: byte-identical action to persistent_rag (same rendered input, same per-request seed) |

The harness itself performed exactly as specified on all three arms: compile produced exact pinned-tokenizer IDs, single-use generation consumed them under the pinned sampler and neutral grammar, both persistent arms received the advertised `retrieve` schema (identical 453-token inputs), and strict final validation rejected unsupported claims without repair.

## Finding

**Neutral protocol competence is not yet established for the persistent arms.** The V3 adapter — trained on the DML policy — answers directly on this neutral prompt instead of retrieving. Both persistent arms behaved identically, so this is systematic policy-behavior evidence, not a store or plumbing defect. Per the fairness manifest, all arms must demonstrate automatic retrieval and valid cited finals on development controls before any pilot; a protocol mismatch at the harness level would be a harness qualification failure, and this model-behavior finding is retained as development evidence, not tuned away.

## Next steps (unchanged gates)

1. Development-scenario tranche on protocol competence: characterize where the neutral policy does and does not elicit retrieval for the fixed adapter (development data only; never tune prompts against held-out outcomes). Consider whether the neutral system prompt requires revision through the reviewed contract process.
2. Embedding admission formalization (the smoke used the manifest's pinned MiniLM semantics computed directly through transformers; sentence-transformers is absent from the pinned image).
3. Durable execution/timeouts and full cost instrumentation in `m8_accounting`.
4. Independent review of the fairness-blocker contracts and this neutral binding.

The stopped workloads remain stopped; restart them when convenient (`docker start comfyui-flux2 swarm-vision`) — no further GPU work is scheduled without a fresh dedicated window.
