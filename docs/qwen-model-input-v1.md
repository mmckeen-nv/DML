# Qwen exact-input companion v1

This candidate Python companion supplies instruction-trained execution for the
[milestone 7 agent harness](live-agent-outcomes-2026-09-20.md). It is separate from
the existing [GPT-2 contract](model-input-contract-v1.md). The memory profile still
has nine HTTP routes and no generation endpoint. Neither companion changes DML's
alpha status or sets `production_ready` to true.

`daystrom_dml.services.qwen_model_input.LocalQwenInputConsumer` exposes the same
`compile`, `execute`, `close`, and context-manager operations as the GPT-2 consumer.
Its snapshot schema is `dml-qwen-model-snapshot-v1`; episode execution selects it
explicitly with `--consumer-profile qwen2-instruct-v1`. The default remains
`gpt2-v1`. Unknown profiles reject before creating an episode directory; choosing
the wrong profile for a bundle fails admission. There is no architecture probing
or automatic fallback.

## Admission and provenance

The bundle contains exactly five regular files: `snapshot.json`, `config.json`,
`model.safetensors`, `tokenizer.json`, and `chat_template.jinja`. Admission verifies
their hashes and the pinned runtime, retains a private copy, and checks the exact
configuration and template. Symlinks, extra files, unknown fields, remote code,
pickle weights, quantized imports, and unsupported architectures are rejected.
Only the built-in `Qwen2ForCausalLM` executes, on CPU with finite float32 weights,
eager attention, and greedy decoding. Each generation call owns a fresh dynamic
KV cache, which is discarded after that call. Callers cannot supply, receive,
persist, or reuse a cache; a retained model cache is rejected. The explicit cache
policy participates in runtime identity. This transient inference state does not
implement native-KV checkpoint restore or cross-call compatibility.

The model has a bounded layer count, context window, and parameter inventory.
Every learned tensor must have its declared name, shape, and dtype. The output
head is the explicit tied embedding alias. Meta-device construction avoids
allocating a second randomly initialized full model; strict assignment and loaded
state checks precede execution. Deterministic rotary buffers are initialized on
CPU. Model, tokenizer, template, runtime, and output-decoding policy participate
in identity and drift checks.

The offline preparer pins the official
[Qwen2.5-1.5B-Instruct snapshot](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
Its original BF16 learned values convert exactly to float32. Preparation retains
the original and normalized artifact/tensor digests and license outside the
five-file bundle. The preparer performs no download, training, remote-code import,
or generated-answer substitution. Snapshot admission establishes the bytes and
execution boundary; training provenance and campaign qualification require their
separate retained evidence.

## Complete framing and output

The fixed template places the complete tool definitions and each complete message
inside JSON bodies with explicit ChatML framing. Message content, original role,
name, tool calls, arguments, and tool-result identifiers remain recoverable.
Tool messages use a user-role envelope while retaining their original `role` in
the JSON body. Every `<` inside serialized data is escaped as `\u003c`, preventing
literal ChatML markers in memory or tool data from becoming framing tokens.
Escaping changes representation; parsing the JSON restores the original fields.

The real tokenizer compiles that complete framing, including the final assistant
prefix. Input IDs plus the reserved output must fit the verified context window.
Execution authenticates the immutable artifact and sends its exact IDs to the
model. It does not render again, truncate history, add hidden instructions, or
repair generated JSON. The GPT-2 request and compiled-input schemas remain intact.

Qwen's model rows include padding beyond the tokenizer vocabulary. Actual input
and output IDs must belong to the verified tokenizer vocabulary; unused rows are
rejected. Output decoding omits only one verified final `<|im_end|>` token from
the text. That token remains in raw output IDs and billable output counts. Internal
EOS, other control tokens, whitespace, fences, and extra prose remain visible to
the strict episode parser. This decoding policy is part of the separate runtime
identity; it does not alter GPT-2 decoding.

## Qualification limits

Tiny random Qwen fixtures test admission, framing, token accounting, dispatch,
and adversarial rejection. They do not establish trained-model task outcomes.
The live campaign separately freezes trained weights, source, corpus, runtime,
thread settings, limits, and acceptance gates before generation. Every attempted
task and failure is retained. Independent verifiers measure semantic results;
model-authored success claims cannot pass a task.

The campaign evidence checker establishes consistency and declared coverage,
not execution authenticity by itself. Its result requires independent provenance,
source, and execution review plus passing published-source CI. Fair baseline
advantage, long-horizon quality, production readiness, and durable reconstruction
after parent-process death remain separate release obligations.
