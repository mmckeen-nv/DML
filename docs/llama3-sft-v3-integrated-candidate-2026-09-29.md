# Registered Llama 3 SFT v3 candidate

Profile `llama3-8b-instruct-sft-v3-bf16-action-json-v1` registers the
[fixed v3 training checkpoint](llama3-8b-sft-v3-2026-09-29.md) as a separate
enabled-only consumer. Its runtime identity uses
`dml-llama3-sft-action-runtime-v2:`. The existing v2 consumer, manifest format,
checkpoint pins, and historical evidence remain separate.

The new manifest admits only the authenticated final v3 adapter, original base
revision, tokenizer/template, runtime image, dependency versions, and training
provenance. It retains the unchanged generation implementation, source-order
rendering, action grammar, sampling, exact token evidence, authenticated dispatch,
failure accounting, tool validation/authorization, and final-answer verifier.
It cannot substitute a v2 or intermediate checkpoint, disable the adapter,
fall back to CPU inference, repair an output, or automatically retry a failed action.

The new qualification runner uses the original fresh planning and readiness
constructors and scorers. Both stages are declared before generation; planning
must pass 2/2 before readiness 4/4 starts. The campaign freezer requires clean
exact-source CI, runtime/durable admission, independent approval, and both
qualification evidence replays before creating the original complete nine-task
campaign. The original M7 gates and both named-task requirements are unchanged;
all-nine-success is not an additional gate.

Registration and CPU tests alone do not qualify the GPU runtime, the model, or
M7. Raw evidence stays on the station; only compact outcomes and hashes are
published. No merge is authorized.
