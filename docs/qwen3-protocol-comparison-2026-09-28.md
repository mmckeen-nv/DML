# Qwen3 protocol comparison — 2026-09-28

The bounded comparison passed **1/3 cases with the DML JSON protocol and 2/3 with the native Qwen tool protocol**. The predeclared native 3/3 promotion threshold was not met; neither arm is promoted. No new M7 campaign ran; the historical Nemotron campaign remains 8/9 and M7 remains open. Both primary and independent evidence replays passed all 18 responses.

| Development case | DML JSON baseline | Native tool protocol |
| --- | --- | --- |
| Two distinct facts | Omitted second fact; 2 calls | Bare tool object lacks native envelope; rejected; 2 calls |
| Dependent retrieval | Passed; 3 calls | Passed; 3 calls |
| Supersession and readback | Three retrieves then final, no mutation; 4 calls | Passed; 4 calls |

The same predeclared data was used in fresh stores, with baseline then native for each case, once each. The native arm changed the template, response-format instruction, tool-history serialization action grammar and termination admission together. The native arm required terminal EOS; the baseline retained its original acceptance of complete JSON at the token cap. Its supersession success supports investigating protocol compatibility; it does not isolate which change caused the difference, establish general reliability, or resolve the shared two-fact failure.

Both arms used the same pinned Qwen3-8B Q4_K_M weights, CUDA backend, tokenizer rows and CPU Torch sampler. The native arm used the vendor XML tool-call template without the all-turn JSON grammar. It admitted one native call or the unchanged strict final JSON contract, preserved raw output separately from the derived action, and performed no prose repair. Original tool authorization, retrieval, verifier, case scorers and budgets remained in force: six model calls, 256 output tokens per call, 1,536 total output tokens, 32,768 input tokens and 300 seconds per episode.

The repository source remained `37471b68a53cb052876409b5d02b5f08ffb175a8`; the diagnostic implementation was station-only. Preparation passed 41 focused tests and six actual compile checks with zero generation or forward calls. The previously qualified CUDA runtime was reused. GPU monitoring observed all six workers in 43 samples; Nemotron retained the same process identity and allocation, and all Qwen allocations were absent after completion.

The run completed 18 model calls with 26,613 input and 859 output tokens, all accounted for. There were no execution errors, timeouts or truncated completions. The native two-fact case produced the sole invalid-action rejection: an EOS-terminated bare tool name/arguments object lacked the required native envelope. It was neither truncated nor a multiple-call output.

The [compact outcome artifact](artifacts/qwen3-protocol-comparison-2026-09-28.json) records exact model revisions, protocol identities, sampling, limits and evidence hashes. Raw requests, responses, token IDs, stores and journals remain under `/home/nvidia/dml-debug/qwen3-protocol-comparison-20260928` on the station. The primary checker needed one documented deepcopy correction for mutable history aliasing. Its original failed receipt remains retained; the corrected checker replayed the same evidence without model reruns. Independent replay also passed all 18 responses. No source/runtime changes, merge or candidate promotion accompany this publication.

A next investigation could separately predeclare, on fresh fixtures, a native-envelope grammar that permits either one native call or the unchanged final JSON, preserving model choice and the same model. This is a future experiment, not repair of the current outputs or a promotion. These results do not establish that a different model is required.
