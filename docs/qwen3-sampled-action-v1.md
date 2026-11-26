# Qwen3 sampled action profile

`qwen3-action-json-nonthinking-sampled-bf16-v1` is an explicit experimental
alternative to `qwen3-action-json-nonthinking-bf16-v1`. The existing greedy
profile and its identity remain unchanged.

Attempt 12 recovery3 had two model planning failures: premature finalization
without supersession, and an incomplete answer after two peer writes. Both
followed model-selected retrieval caps of one. Retrieval, persisted records and
verification behaved as specified. This candidate tests a different decoding
policy; it does not claim that greedy decoding caused those failures.

The [pinned Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
recommends the temperature, top-p, top-k and min-p values below for non-thinking
sampling. The single-candidate limits and fixed seed are this project's
reproducibility settings:

| Setting | Fixed value |
| --- | --- |
| Sampling | enabled |
| Temperature | 0.7 |
| Top-p | 0.8 |
| Top-k | 20 |
| Min-p | 0 |
| Beams / return sequences | 1 / 1 |
| CPU seed | 0 at the start of each generation |

The runtime identity binds the sampling settings and seed rule. CPU RNG state
is restored after success or failure. A module lock serializes this profile's
sampled generations; it does not isolate arbitrary unrelated RNG users in the
same process. Production qualification runs one isolated CPU worker at a time.
There is no user-selectable seed, seed search, candidate selection or retry.

The existing syntax mask runs before sampling warpers. It still admits every
public tool and final branch; argument values and the decision to stop remain
model choices. Raw output tokens and text retain the existing evidence and
accounting contract. The snapshot, tokenizer, non-thinking transport, policy,
tools, task corpus, verifiers, budgets and acceptance gates are unchanged.

Qualification requires reviewed source, fresh source CI, zero-generation model
admission, a declaration frozen before generation, one complete nine-task
campaign and two sequential evidence replays. All failures and costs remain
part of the result. Synthetic test passes establish runtime behavior, not live
task success or milestone completion.
