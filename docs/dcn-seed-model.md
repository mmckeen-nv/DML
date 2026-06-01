# DCN seed-model profile

Status: initial operator recommendation for governed DCN trials.

## Local inventory

Observed on the current macOS host:

- Hardware: Apple M4, 16 GiB unified memory, 10-core Apple GPU / Metal 4.
- Serving runtime available: `ollama` at `/usr/local/bin/ollama`.
- Local Ollama models:
  - `llama3:8b` — 4.7 GB, installed.
  - `gemma3:4b` — 3.3 GB, installed.
  - `qwen3-embedding:0.6b` — 639 MB, installed; embedding model, not a DCN seed LLM.
- Repo defaults already use:
  - `llm_backend: ollama`
  - `model_name: llama3:8b`
  - `embedding_model: ollama:qwen3-embedding:0.6b`

## Recommendation

Use `llama3:8b` as the first DCN seed model.

Rationale:

1. It is already installed locally and matches the repository's existing Ollama default path.
2. It avoids introducing a new runtime dependency while DCN promotion controls are still maturing.
3. Its size is appropriate for 16 GiB Apple Silicon where DCN should act as a bounded policy/planning seed, not as an unconstrained frontier replacement.
4. It keeps the first `active_learn` experiments reproducible: same local runtime, same named model, same deterministic fixture gates.

## Rejected-for-now options

- `gemma3:4b`: useful fallback for latency or memory pressure, but smaller than the current default and less aligned with existing config/docs.
- `qwen3-embedding:0.6b`: keep for embeddings only; not suitable as the DCN reasoning seed model.
- vLLM/large CUDA profiles: defer for this Mac lane. Existing historical docs mention NVIDIA/GB10/CUDA paths, but this lane is an Apple M4 host and should not rely on stale Linux GPU assumptions.
- Frontier/provider-backed seed models: defer until the DCN eval harness has model-comparison fixtures and cost/latency/audit capture. The seed should stay local for the first governed learning trials.

## Proposed profile

```yaml
llm_backend: ollama
model_name: llama3:8b
embedding_model: ollama:qwen3-embedding:0.6b
strict_llm_required: true
strict_embedding_required: true
```

## Learning-loop bridge

Use the seed model as an offline candidate proposer, not as live policy authority. The current bridge is:

```bash
dml dcn seed-trial --input sanitized-feedback.json --output dcn-seed-trial-artifact.json
```

For an active local seed-model path, run:

```bash
dml dcn seed-propose --input sanitized-feedback.json --output dcn-seed-proposal.json
dml dcn seed-loop --input sanitized-feedback.json --output dcn-seed-loop-artifact.json
```

`seed-propose` uses the local seed model to propose allowlisted procedural candidates and unsupported schema pressure. `seed-loop` immediately validates that model proposal through `seed-trial`. Both commands are non-promoting and keep `active_learn` behind the existing checkpoint/eval/hygiene promotion gates.

The artifact is non-promoting. It contains accepted procedural overlay candidates, rejected updates, unsupported policy-pressure reports, and a candidate policy snapshot. Unsupported pressure is the path for schema growth: if the current allowlist is too small for useful learning, record the missing typed capability instead of silently broadening runtime authority.

## Promotion guardrails

Before using this as a DCN `active_learn` seed, require:

1. DCN eval smoke artifact readiness: `artifact.readiness.ready == true`.
2. Hygiene smoke evidence.
3. Operator checkpoint before promotion.
4. Sanitized promotion audit only; never store raw transcripts, prompt scaffolds, tool logs, or secrets in promotion evidence.
5. A separate seed-model comparison benchmark before replacing `llama3:8b` with any larger or remote model.
