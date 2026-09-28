# Qwen3-8B ARM64 qualification candidate

`qwen3-8b-gguf-arm64-action-json-sampled-v1` is a separate Linux ARM64 CPU
candidate. It retains the existing Qwen GGUF action consumer, including model
selection of tools and generation of cited final answers. It does not route
through the Nemotron endpoint. The original Qwen profile and historical replay
identities remain supported unchanged.

The weights are official `Qwen/Qwen3-8B-GGUF` revision
`7c41481f57cb95916b40956ab2f0b139b296d974`, `Qwen3-8B-Q4_K_M.gguf`, SHA256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
Tokenizer/configuration metadata is pinned to `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`. Snapshot preparation retains the
existing DML Qwen chat template and checks every source file against repository
pins. Admission verifies every tokenizer row against the native model.

The CPU backend uses pinned `llama-cpp-python==0.3.35`, `torch==2.8.0+cpu` and
`xgrammar==0.2.7`, four inference threads, and fresh F16 contexts. Actual ARM
build configuration, linked libraries, installed package hashes, model identity,
admission time and memory must be retained on the station. ARM identity uses a
separate runtime prefix and rejects other host architectures before loading.
Q4_K_M tensors are not equivalent to BF16 weights. Exact source and token
identities do not imply identical floating-point sampling across architectures.

The original all-turn JSON action grammar, DML validation and authorization,
retrieval, tool feedback and claims/citations contract remain in force. Sampling
is unchanged: temperature 0.7, top-k 20, top-p 0.8, min-p 0, seed 0, one
candidate. No Nemotron recovery policy or budget-guidance prompt is imported.
Original limits remain six calls, 256 output tokens per call, 1,536 cumulative
output tokens, 32,768 cumulative input tokens and 300 seconds per task.

`dml_core/scripts/qwen3_gguf_synthetic.py` predeclares four noncorpus controls:
live retrieval, dependent feedback, supersession with read-back, and untrusted
instruction rejection. All four must pass before this candidate is ready for a
new campaign. This is a readiness requirement, not a replacement M7 gate.
Keep failed runs and raw model/token evidence; do not force completion, repair
output or retry selected cases. Model admission within each worker counts
toward its original wall-time limit. Confirmed worker termination precedes
private snapshot cleanup; unresolved termination stops the suite.

Before campaign generation, require exact-source CI, host admission, independent
review, and fresh durable-execution qualification. The separate freezer
`scripts/freeze_qwen3_gguf_campaign.py` verifies a hash-bound readiness
attestation and freezes all nine tasks with the original corpus, budgets and
acceptance gates. The durable launcher permits one start with no automatic
restart or resume. Perform primary and independent data-only evidence replays.
Close M7 only if both named tasks and every original gate pass. No qualification
or campaign outcome is implied by this profile declaration.

Raw evidence, models and runtime inventories stay outside Git on the station.
Publish only compact outcomes and hashes. Preserve every prior Qwen and
Nemotron campaign. Do not merge or claim production readiness.
