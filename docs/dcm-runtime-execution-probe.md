# Runtime-native DCM execution-state probe

`dcm-kv-probe` is an experimental, fail-closed probe for runtimes that expose all of the following capabilities:

- stable slot affinity;
- prompt/KV reuse with runtime-native token counters;
- checkpoint save and restore;
- slot erase;
- physical checkpoint deletion; and
- an exact runtime version and endpoint identity.

The probe executes:

```text
preflight → erase → cold completion → hot completion → checkpoint save
→ erase → checkpoint restore → deterministic completion → physical purge
→ registry-removal verification → final slot erase
```

It uses `ExecutionCheckpointController`, so checkpoint state is bound to the exact model, tokenizer, positional configuration, rendered context packet, manifest, runtime, endpoint, tenant, and session. Unsupported capabilities fail before the first slot mutation.

## llama.cpp invocation

The checkpoint directory must already exist and must be the same directory configured for the llama.cpp server's slot save/restore operations. Supply real model and tokenizer digests; placeholder identities defeat restore authorization and are rejected by policy even if they are syntactically valid.

```bash
dcm-kv-probe \
  --endpoint-url http://127.0.0.1:18080 \
  --runtime-id llama-local \
  --runtime-version <exact-llama.cpp-version> \
  --checkpoint-directory /absolute/shared/checkpoint/path \
  --model-id <exact-model-id> \
  --model-digest sha256:<model-file-or-revision-digest> \
  --tokenizer-digest sha256:<tokenizer-digest> \
  --positional-config-json '{"n_ctx":65536,"rope":"exact-runtime-value"}' \
  --model-limit-tokens 65536 \
  --artifact ./dcm-kv-execution-probe.json
```

A successful artifact contains digests, counters, byte counts, timings, equivalence status, and purge evidence. It does **not** contain:

- prompt or output text;
- output token IDs;
- checkpoint names;
- tenant or session identifiers;
- arbitrary runtime metadata; or
- checkpoint bytes.

Failure artifacts contain only the exception type and a digest of the reason. The process exits nonzero when the probe does not pass.

## Cleanup contract

A checkpoint is physically deleted before its registry authorization is removed. On restore or validation failure, the probe attempts bounded checkpoint purge and always erases the selected slot. If cleanup fails, the complete probe fails. Adapters that cannot prove physical deletion are rejected during preflight.

The optional metadata registry defaults to a platform-native temporary directory and is removed when the command exits. Explicit registry paths use `pathlib` and the repository's cross-platform atomic registry implementation.

## vLLM boundary

Ordinary vLLM OpenAI-compatible serving is **not** represented as checkpoint-capable by this probe.

vLLM Automatic Prefix Caching automatically reuses identical prefix blocks, but it does not expose an external request-bound slot checkpoint/save/erase/restore/purge lifecycle through the ordinary OpenAI API. It is useful acceleration, not proof of controller-authorized restoration.

vLLM's disaggregated-prefill KV connectors can transfer KV state between configured prefill and decode instances. The feature and request-level `kv_transfer_params` path are documented as experimental and require server-side connector configuration. They are not treated as a universal checkpoint API, and DML must fail closed until a version-pinned adapter can prove identity binding, native reuse metrics, lifecycle ownership, and physical purge.

Authoritative references:

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM Disaggregated Prefilling (experimental)](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)

Do not infer runtime-native KV reuse from wall-clock latency alone.
