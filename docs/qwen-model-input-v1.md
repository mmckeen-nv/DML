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

## Separately pinned Coder preparation candidate

The separately pinned preparation API was reconstructed after the September 20
session froze. Historical review applies to the earlier isolated bytes; the
reconstructed revision has its own independent review and
[validation](artifacts/agent-episode-coder-recovery-validation-2026-09-22.json).
All **780 mandatory CPU cases passed with zero skips**, including six Coder
controls; both strict lint scopes and maintained mypy passed, with **434 unchanged
source hashes**. The five actual pinned source payloads have been acquired and
hash-verified. Preparation/admission and a fresh declared live campaign remain
separate gates; this source change does not establish stronger model capability.

The existing `prepare_qwen_snapshot(raw_directory, destination)` continues to
accept only the original general model. The separate
`prepare_qwen_coder_snapshot(raw_directory, destination)` pins
`Qwen/Qwen2.5-Coder-1.5B-Instruct` revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a` and provenance schema
`dml-pretrained-qwen2-coder-instruct-v1`. Neither API accepts caller-defined trust
pins or auto-detects a model. The shared finite BF16-to-float32 exact-round-trip
normalization, complete learned-tensor checks, five-file bundle, native template,
strict admission bounds and CPU execution architecture are unchanged.

After source validation, preparation from an already
acquired exact pinned raw directory uses a fresh destination:

```python
from daystrom_dml.services.qwen_pretrained_snapshot import prepare_qwen_coder_snapshot

provenance = prepare_qwen_coder_snapshot("/path/pinned-coder-raw", "/path/new-coder-prepared")
```

The raw directory must contain exactly the pinned `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `LICENSE` and `model.safetensors`. Preparation performs
no download. The destination contains `bundle/`, `provenance.json` and the license;
existing destinations, missing/extra files and wrong source sizes or hashes reject.
The new model identity needs its own strict admission and reviewed pre-generation
campaign freeze. Prior models, campaign evidence and failures remain preserved;
the policy, tools, runtime architecture, limits and qualification gates are not
relaxed for the alternate fixed source.

## Complete framing and output

Renderer v2 places each message's original content in native ChatML framing.
An initial system message's content stays at the start of the first system
message. Every message's complete non-content fields, including its original
role, name, tool calls, arguments and tool-result identifiers, are retained in an
index-aligned metadata array, together with the complete tool definitions.
That first-system data section uses `dml-qwen-chatml-fields-v2` and explicitly
states that metadata is data: user and tool attributes do not gain system
authority. The template supplies no task truth or action grammar; the conversation's
policy retains responsibility for response syntax.

Content escapes `&` to `&amp;` before escaping `<` to `&lt;`. Reverse those
substitutions in the opposite order (`&lt;`, then `&amp;`) to recover the exact
original string, including literal entity spellings. The metadata is complete
JSON with `<` escaped as `\u003c`. Caller data therefore cannot emit ChatML
framing tokens, close the metadata section or become a tool-response delimiter.
Tool content uses a user-role envelope with native `<tool_response>` wrappers;
the metadata preserves its original `tool` role and all other fields. Content
and indexed metadata together recover every supplied message field.

The exact v2 template has SHA-256
`24bd5fcdd71018049c672f96778e9162c6d00337e22f8f5f9b358c52705d9348`.
Admission rejects a different template even when its manifest is rehashed to
match. The earlier full-message JSON template used by live attempts 1 and 2
remains historical evidence with its own identity; it is not accepted by this
renderer revision. A freshly prepared bundle must retain the same pinned learned
weights, configuration and tokenizer while separately recording the new template
and derived identity. A new campaign freeze is required before execution.

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

Tiny random Qwen fixtures test admission, reversible full-field framing, token
accounting, dispatch and adversarial rejection. The v2 rendering correction
received independent **9.6/10** [source acceptance](artifacts/agent-episode-native-review-2026-09-20.json)
after **715 mandatory CPU tests passed with zero failures or skips**, with all
**429 source hashes unchanged** in the [validation record](artifacts/agent-episode-native-validation-2026-09-20.json). They do not establish trained-model task outcomes.
The live campaign separately freezes trained weights, source, corpus, runtime,
thread settings, limits, and acceptance gates before generation. Every attempted
task and failure is retained. Independent verifiers measure semantic results;
model-authored success claims cannot pass a task.

The campaign evidence checker establishes consistency and declared coverage,
not execution authenticity by itself. Its result requires independent provenance,
source, and execution review plus passing published-source CI. Fair baseline
advantage, long-horizon quality, production readiness, and durable reconstruction
after parent-process death remain separate release obligations.
