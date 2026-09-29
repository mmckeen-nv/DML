# Qwen3 native-envelope diagnostic — 2026-09-28

The fresh native-envelope diagnostic passed **1/3 cases**. All three episodes produced syntactically admitted output, but two failed the unchanged factual verifier. The 3/3 threshold was not met; no candidate is promoted and no new M7 campaign ran. The historical Nemotron campaign remains 8/9 with M7 open. Both primary and independent evidence replays passed all nine calls.

| Fresh development case | Outcome | Calls |
| --- | --- | --- |
| Two distinct facts | Fabricated an unseen current value and cited old evidence; failed | 2 |
| Dependent retrieval | Passed | 3 |
| Supersession and readback | Mutation and readback succeeded; final added an unsupported claim about old evidence; failed | 4 |

The grammar admitted either one native tool-call envelope or the original final-action JSON. It did not force tools, force a final, repair output, or add factual constraints. This removed the observed envelope syntax failure in this run, while the original verifier still rejected unsupported claims. The supersession answer contained a correct current claim as well as an extra unsupported claim citing superseded evidence. Syntactic validity and a successful mutation therefore did not establish a passing answer.

These were fresh, predeclared development fixtures, run once each with separate authority stores and unchanged original scorers. Their data differs from the [preceding protocol comparison](qwen3-protocol-comparison-2026-09-28.md), so the outcomes are not a paired estimate of the grammar's effect. No fixtures, retrieval results, action branches or answers were selected using model outcomes.

The same pinned Qwen3-8B Q4_K_M weights, native template, CUDA backend and CPU Torch sampler were reused. Repository source remained `37471b68a53cb052876409b5d02b5f08ffb175a8`; the grammar and runner were station-only. Original parser, tool authorization, retrieval, strict final contract and budgets remained in force: six calls, 256 output tokens per call, 1,536 total output tokens, 32,768 input tokens and 300 seconds per episode.

Preparation passed 48 focused tests, 17 independent grammar controls, three separate non-model reachability checks, and three compile checks independently reconstructed without forwards or generation. Live execution completed nine calls with 13,239 input and 577 output tokens, all known. There were no action rejections, execution errors, timeouts or truncated completions. GPU monitoring observed all three workers; Nemotron remained unchanged and Qwen allocations were released after completion.

The [compact outcome artifact](artifacts/qwen3-native-envelope-2026-09-28.json) binds the model, protocol, grammar, declaration, frozen execution and monitor receipts. Raw evidence remains under `/home/nvidia/dml-debug/qwen3-native-envelope-20260928` on the station. Both replay and final-review hashes are included. This diagnostic provides no M7 qualification or production-readiness credit.

The next proposed investigation is a separately admitted stronger-model comparison on fresh cases. No specific model is approved by this result, and M7 remains open.
