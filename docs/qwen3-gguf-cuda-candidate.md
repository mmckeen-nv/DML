# Qwen3-8B CUDA completion candidate

`qwen3-8b-gguf-cuda-action-json-completion-v1` is a separately versioned
candidate. It retains the exact Qwen3-8B Q4_K_M weights, tokenizer, DML
non-thinking template, JSON action grammar, sampling parameters, task corpus,
retrieval, authorization, claims/citations contract, budgets and M7 gates.
The prior ARM64 CPU qualification remains failed at 3/4 and is not resumed.

The prior supersession failure was model behavior. Two identical `top_k=1`
lookups returned only the current record with authentic partial-result
metadata. The model then produced a correct cited answer without performing
the requested operation. The tool and instructions were present; no returned
action was discarded. The new profile adds task-agnostic completion guidance
to emphasize partial evidence and completion of requested state changes.
Its benefit is unproven until live qualification. The guidance does not choose
queries, limits or actions, force calls, execute mutations, repair output or
relax the final verifier. Read-only tasks do not acquire mutation authority.

CUDA inference uses a separate build of the pinned llama-cpp-python 0.3.35
bridge. The CPU Torch 2.8.0+cpu grammar and multinomial sampler retain the
original processor order and seed. Weights for repeating layers and the output
layer, KV storage and graph computation are offloaded to the declared CUDA
device; the native implementation keeps input embeddings on the CPU. Actual
placement and buffers must be recorded, not inferred from `n_gpu_layers`.
Fresh per-request F16 contexts and exact authenticated HF token IDs remain
required. GPU arithmetic may change sampled outputs despite the same seed.

The active GPU model and action identities are distinct from the immutable
snapshot's CPU provenance. Bind the driver/device, CUDA build and linked
libraries, native bridge, placement policy and completion guidance. Reject
unexpected backend configuration and preserve CPU-only rejection for the
historical CPU profiles. Never stop or reconfigure the existing Nemotron
service as part of this candidate.

Before live execution, require source regressions, exact-source CI, pinned
artifact/full-token-row admission, a declared nonsampled CUDA forward with
actual GPU-placement evidence, durable-execution qualification and independent
review of the frozen suite. Run four freshly predeclared noncorpus controls
once: retrieval, dependent feedback, supersession with read-back, and untrusted
instruction rejection. Preserve failures. Do not force completion, retry a
case, alter stored feedback or change the six-step/256-token-per-call/300-second
limits. No-generation fixture checks may establish retrieval completeness
without prescribing those queries to the model.

Require 4/4 readiness and both data-only evidence replays before approving a
new complete nine-task campaign. The campaign freezer additionally requires a
hash-bound GPU admission receipt for the exact source, model identity and
profile. Keep original M7 acceptance gates, including both named tasks.
Keep raw evidence on the station and publish only compact outcomes and hashes.
No merge or production-readiness claim is authorized by this declaration.
