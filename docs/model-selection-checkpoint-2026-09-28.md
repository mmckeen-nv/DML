# Model selection checkpoint — 2026-09-28

Saved at the user's request. Research is stopped; no new model download, load,
training, qualification or campaign was started. Branch:
`codex/remote-vllm-candidate`. No merge.

## Retained execution status

- Historical Nemotron M7 campaign remains **8/9** and unchanged. M7 is open.
- [Qwen protocol comparison](qwen3-protocol-comparison-2026-09-28.md): baseline
  **1/3**, native **2/3**; both replays passed all 18 responses. Published in
  `d7cbb57`.
- [Native-envelope diagnostic](qwen3-native-envelope-2026-09-28.md): **1/3**;
  both replays passed all nine responses. Syntax succeeded, but one final
  invented an unseen value and another added an unsupported claim after
  successful supersession/read-back. Published in `5e935f5`.
- Exact executed repository source remains
  `37471b68a53cb052876409b5d02b5f08ffb175a8`. Station-only diagnostic code and
  all raw evidence remain under `/home/nvidia/dml-debug/` in
  `qwen3-protocol-comparison-20260928` and `qwen3-native-envelope-20260928`.
  Neither diagnostic was promoted to a qualified candidate.

## Read-only stronger-model shortlist

Observed station capacity: GB300, 75,588 MiB (73.8 GiB) free GPU memory with
Nemotron still loaded; 317 GiB available disk as reported by `df -h`.
Weight totals below are sums of official Hugging Face safetensors file sizes,
not measured inference allocations. Actual runtime/cache fit is unqualified.

| Candidate | Observed repository revision | Weight bytes | Rationale |
| --- | --- | ---: | --- |
| Qwen/Qwen3.5-27B, BF16 | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 55,563,022,432 | First quality-oriented hypothesis: larger dense model without low-bit weight quantization. About 51.75 GiB of weight files. |
| Qwen/Qwen3.6-35B-A3B-FP8 | `95a723d08a9490559dae23d0cff1d9466213d989` | 37,463,662,160 | Alternative with more memory headroom; MoE and FP8 introduce different tradeoffs. About 34.89 GiB of weight files. |
| Qwen/Qwen3-30B-A3B-Instruct-2507, BF16 | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` | 61,066,575,656 | Non-thinking-only model; potentially simpler alignment with short output budgets. |

These revisions were inspected, not downloaded or attested as loaded. Vendor
benchmarks motivate investigation but do not demonstrate DML accuracy or
performance under its original 256-output-token-per-call budget. Both newer
models support disabling thinking; neither has been tested here in that mode.
Qwen3.5/3.6 use a different native function/parameter protocol with
`qwen3_coder` parsing; the old Qwen3 JSON-envelope grammar must not be reused
as though it were their native protocol. Their serving runtime also requires
separate admission, version pinning and raw-token evidence checks.

Primary references inspected:

- [Qwen3.5-27B official model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.6-35B-A3B official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Official Qwen3.6 FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [Qwen3-30B-A3B-Instruct-2507 official model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [vLLM Qwen3.5/3.6 serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

The independent grader reviewed the shortlist and agreed that Qwen3.5-27B
BF16 is a reasonable first hypothesis, with no proof of DML superiority or
live qualification implied.

## User's smaller-model direction

The user proposed Llama 3 8B Instruct, especially with fine-tuning. Preserve
this as an explicit alternative. The existing failures concern one pinned
Qwen3-8B Q4_K_M configuration and do **not** establish that 8B models cannot
perform DML tasks. Fine-tuning for evidence completeness, exact citations,
authentic error recovery and timely valid finals is plausible but untested.

Distinguish the exact [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
checkpoint from [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
before any implementation; neither has been selected or qualified. A larger
reference model could help separate task/protocol problems from smaller-model
training gaps. It is not evidence that a large production model is required.

Any future fine-tuning data must be separate from the evaluation corpus and
held-out qualification fixtures. Train general behaviors across diverse
synthetic examples, not memorized campaign answers or previously exposed
evaluation cases. Keep the original gates, budgets, retrieval, authorization,
final-answer contract and failure retention unchanged. Use fresh predeclared
tests, pinned weights/template/runtime, implementation subagents and an
independent grader before any new campaign. No next run is scheduled.
