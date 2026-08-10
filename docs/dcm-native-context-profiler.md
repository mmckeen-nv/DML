# Native-context lifecycle profiler

`dcm_native_context_profile.py` converts a payload-free span manifest into a
deterministic retain/compress/freeze/thaw/hot-swap plan. It is a planning and
measurement tool: it does not mutate a serving runtime, inspect prompt text, or
claim that vLLM can already restore paged KV or Mamba state.

## Contract

The input contains:

- model and runtime identity;
- the model-native, currently served, and desired hot-token limits;
- ordered, contiguous span metadata (IDs, token positions, token counts, SHA-256
  content digests, authority, priority, residency, and age);
- optional precomputed summary digests and token counts; and
- optional exact span IDs requested for this generation.

Unknown fields are rejected. Raw prompt or transcript fields are not accepted.
Failure artifacts contain only an error class and a digest of the reason.
Successful artifacts include span IDs and digests, lifecycle actions, residency
and tier-transfer estimates, stable-prefix/recompute boundaries, feasibility,
and a digest covering the deterministic profile.

Compression is only planned when the caller provides an already-computed
summary digest and token count. Protected authority and `exact_required` spans
are never summarized or pressure-evicted.

## Run the representative profile

From the repository root:

```bash
python3 dml_core/scripts/dcm_native_context_profile.py \
  --input docs/examples/dcm/native-context-nemotron-262k-served-65k.input.json \
  --artifact docs/examples/dcm/native-context-nemotron-262k-served-65k.profile.json
```

The checked-in example records the first proven mismatch profile:

- model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`;
- authoritative native limit: 262,144 tokens;
- live-proof served limit: 65,536 tokens;
- target hot set: 60,000 tokens; and
- representative logical generation: 192,864 tokens.

The profile is intentionally metadata-only. Its state-byte estimate is a
scenario input, not a measured Nemotron/vLLM KV or Mamba-state allocation.

## Interpretation boundaries

A successful profile means the proposed hot set fits the configured hot and
served limits. It does **not** prove runtime paging, restoration, continuation,
or output equivalence. Those claims require a runtime adapter and a live
cold/hot/save/erase/restore/purge experiment. The profile's stable-prefix and
recompute boundary identify where such an adapter must invalidate and rebuild
runtime state after compression, thaw, or hot swapping.
