# Qwen2 BF16 sampled action profile

`qwen2-action-json-sampled-bf16-v1` is a separate experimental evaluation profile
for the pinned [Qwen2.5-3B-Instruct snapshot](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1).
Its runtime identity starts `dml-qwen2-bf16-action-runtime-v1:`. Existing Qwen2
float32 and Qwen3 profiles keep their original identities and behavior.

This is one larger-model comparison following premature finalization and missing
evidence in the earlier campaign. It does not establish that parameter count
caused the failures or that a larger model will repair them. Architecture, trained
weights and tokenizer differ from Qwen3. The old Qwen2 transport is reused exactly,
without Qwen3's empty thinking prefix; full policy, message attributes and tool
definitions remain visible. The task corpus, verifiers, tools, syntax grammar,
budgets and acceptance gates are unchanged.

The source package retains the original BF16 shards, configuration, tokenizer and
license. It admits no remote code, quantization or payload conversion. The pinned
model's Qwen Research License Agreement remains part of the evaluation package;
this candidate is not a production-model selection. Snapshot and runtime
validation separately bind the exact learned tensors and explicit tied output
head alias. Admission and observed memory use remain prerequisites; the declared
context limit does not prove every input length fits this host.

| Generation setting | Fixed value |
| --- | --- |
| Sampling | enabled |
| Temperature / top-p / top-k / min-p | 0.7 / 0.8 / 20 / 0 |
| Repetition penalty | 1.0 |
| Beams / return sequences | 1 / 1 |
| CPU seed | 0 at each call |
| Stop token | verified tokenizer `<\|im_end\|>` only; 151645 for the pinned snapshot |

The comparison keeps the previously reviewed sampling values, neutral repetition
and single-EOS boundary. The [pinned vendor generation configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/generation_config.json)
instead specifies repetition penalty **1.05** and EOS choices **151645 and 151643**.
Those settings are deliberately not copied; no claim of vendor-identical decoding
is made. Other sampling filters are neutral and the exact effective configuration
is fingerprinted with the runtime.

The request-owned syntax mask runs before sampling. Every public tool, arbitrary
permitted argument and final branch remains available; the model chooses retrieval
limits, mutation references and whether to finish. No task-keyword rules, automatic
retrieval widening, forced lifecycle calls or final-answer repairs are applied.
All generated token IDs and text remain in the evidence and usage accounting.

A module lock serializes this profile's CPU sampling, and caller CPU RNG state is
restored on success or failure. This does not isolate unrelated RNG users in the
same process. Qualification runs one isolated model worker at a time. There is one
candidate with no seed search, candidate reranking or implicit retry.

Qualification requires independent source acceptance, exact-source CI, verified
model provenance, zero-generation admission, a declaration frozen before the full
nine-task campaign and two sequential evidence replays. Every failure and cost
remains visible. Tiny fixture tests establish implementation behavior only; they
do not establish live task success or milestone completion.
