# Desktop Codex handoff: DML bare-metal testing

## Current status

The latest native-tools-v6 budget-guidance development qualification passed 3/6 cases; both recovery cases still failed (0/2). Each recovered a verified conflict, committed supersession and read back, then repeated retrieval on the last permitted call despite visible budget metadata. The baseline supersession case produced incomplete native syntax at its output cap. No new campaign was frozen or launched; M7 remains open. Exact-source CI passed 21/21 jobs on `0351ba9`; primary and independent replay passed all 21 calls, preserving every failure. Independent campaign approval is withheld. See [v6 qualification outcome](docs/artifacts/native-vllm-v6-budget-qualification-outcome-2026-09-28.json).

The newer native-tools-v5 recovery development qualification failed both declared cases (0/2), despite both offline evidence replays passing. The source case truncated before exercising recovery; the replacement case recovered a verified conflict, retrieved fresh state, committed a model-owned supersession and read back, but reached the step limit without a final answer. No new nine-task campaign was frozen or launched. Exact-source [CI 36468583308](https://github.com/mmckeen-nv/DML/actions/runs/36468583308) passed all 21 jobs on `68fdc74`; it does not override failed live qualification. See [v5 recovery qualification](docs/artifacts/native-vllm-v5-recovery-qualification-outcome-2026-09-28.json).

M7 remains open: the frozen remote `nemotron-remote-vllm-native-tools-v4` candidate completed all nine tasks once, with eight passes. `supersede_then_answer` failed with `tool_error`; `read_both_commits` passed. Both sequential evidence replays completed and matched byte for byte. Execution source: `e970252d1b2afff582fd741f137cb4334e95a06c`; recorded usage: 30,713 input + 1,363 output tokens; no unknown usage or effects; stable serving epoch.

The model committed a supersession, received its acknowledgment and readback, then repeated the stale mutation. The gateway rejected `ReceiptLifecycleConflict` with no additional effects; no final answer followed. No discarded valid tool call, missing feedback or replay defect was found. See the [compact verified outcome](docs/artifacts/native-vllm-m7-outcome-2026-09-28.json).

M1–6 remain closed, M7–11 open, M12–13 deferred. No merge or production-readiness claim is made. Preserve the completed run and all earlier attempts; do not selectively retry tasks or change acceptance gates. Raw evidence stays on the station. See the [current milestone plan](docs/bare-metal-milestone-plan-2026-09-28.md) and [authoritative ledger](docs/production-remaining-work-2026-09-18.md).

## Paste into desktop Codex

> Read the current milestone plan and ledger first. Preserve the frozen remote candidate source and raw evidence. Use the completed primary and independent evidence reviews and compact outcome; retain the failed named task. No further campaign execution is authorized by this status update. M7 cannot close while `supersede_then_answer` fails. No new generation, selective retry, changed gate or merge is implied by this handoff.

## Historical CPU setup instructions

The Qwen-specific instructions below are retained from the original testing handoff. They describe a different candidate and are not the current remote campaign configuration or a launch instruction. Current remote profile mechanics are documented in [remote-vllm-candidate.md](docs/remote-vllm-candidate.md).

## 1. Clone and inventory

```bash
git clone --branch test/bare-metal-m7 --single-branch https://github.com/mmckeen-nv/DML.git
cd DML
git rev-parse HEAD
git status --short
```

Use Linux x86_64 and Python 3.12 for the first comparison. Record OS/kernel, Python, CPU features/core count, available RAM/disk, filesystem, and GPU/driver inventory if present. Proposed capacity: at least 16 GiB RAM and 40 GiB free persistent disk, with four inference threads. These are planning allowances; measure actual admission and generation behavior. The prior 8 GiB workspace admitted the model at about 5.46 GB process RSS; that was not a full-window or generation-peak guarantee.

Keep a dedicated persistent run root OUTSIDE the checkout, for example a directory beside it. Record absolute paths in the new manifest. Prevent host sleep during testing. A terminal multiplexer alone is not proof the host or evidence survives disconnects.

## 2. Install the matching CPU runtime

Run from repository root. Ensure Python 3.12 venv support, GCC/G++, CMake and Ninja are installed using this host's package manager. Install the CPU Torch wheel first; the verifier requires `torch==2.8.0+cpu`, not a CUDA wheel.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_SUBPROCESS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON"
python -m pip install --no-cache-dir --no-binary=llama-cpp-python '.[agent-action,agent-gguf,dev]'
export DML_SKIP_VENV_REEXEC=1
export PYTHONPATH="$PWD/dml_core"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

These mirror the CPU lane's package pins/build configuration; record all installed package/native-library hashes on the actual host. Do not copy a hardware-specific compiled wheel from the prior workspace. The native consumer rejects a backend compiled with GPU or RPC support, even with zero GPU layers. GPU/MPS/ARM experiments require a separately declared and tested profile; never bypass these checks.

Run the focused integration tests without downloading model weights:

```bash
python -m pytest -q \
  dml_core/tests/test_qwen3_gguf_input.py \
  dml_core/tests/test_qwen3_gguf_model_snapshot.py \
  dml_core/tests/test_qwen3_gguf_pretrained_snapshot.py \
  dml_core/tests/test_qwen3_gguf_profile.py
```

Require these modules and the real pinned-native ABI case to run, not skip. Then run relevant retained model/action/evidence regressions and the maintained lint/type/CI checks from `.github/workflows/ci.yml`. Record failures; do not call historical CI a substitute for host validation.

## 3. Acquire and prepare exact model files

Authoritative pins live in:
- `dml_core/daystrom_dml/services/qwen3_gguf_model_snapshot.py`
- `dml_core/daystrom_dml/services/qwen3_gguf_pretrained_snapshot.py`

Download official HTTPS artifacts at immutable revisions; verify complete SHA256 before preparation. Keep TLS verification enabled. Preserve failed attempts, and do not silently substitute a quantization, tokenizer or template.

| Runtime filename | Official source |
|---|---|
| model.gguf | Qwen/Qwen3-8B-GGUF @ 7c41481f57cb95916b40956ab2f0b139b296d974 / Qwen3-8B-Q4_K_M.gguf |
| config.json, tokenizer.json, tokenizer_config.json, LICENSE | Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 / same filenames |

GGUF size: 5,027,783,488 bytes; SHA256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. Each metadata file's hash is in SOURCE_FILE_PINS.

The stage must contain exactly those five files as real exclusive regular files, not symlinks or hardlinks. Use actual copies from any download cache. The stage and destination must be on the same Linux filesystem; destination must not exist and its parent must exist. Preparation verifies hashes, writes the existing DML Qwen3 template and manifest, privately verifies the bundle, then renames the stage. Never pre-create the destination or substitute the vendor chat template.

Python API (there is no preparation CLI):

```python
from daystrom_dml.services.qwen3_gguf_pretrained_snapshot import prepare_qwen3_gguf_snapshot
receipt = prepare_qwen3_gguf_snapshot(stage_path, bundle_path, context_window=32768)
# Persist the returned receipt as JSON in your external run root.
```

Before preparation or admission, set `TMPDIR` to an existing spacious directory on persistent disk outside the checkout. The verifier makes a private copy of the roughly 5 GB model there; verify free space on that filesystem, not just the run-root filesystem. A small or RAM-backed `/tmp` can fail despite sufficient run-root capacity.

Perform fresh zero-generation consumer admission and close it; record model identity, token-row parity, actual peak memory and backend identity. The new host/runtime changes identity, so old cloud admission/specification cannot authorize this run.

## 4. Establish durable execution and freeze before generation

Desktop Codex must implement/review the local lifecycle wrapper: unique run ID, exclusive-start guard, source/runtime/model hashes, start receipt, progress, private durable per-task evidence and terminal receipt. Test a synthetic job surviving controller disconnect/reconnect; verify duplicate start is rejected and worker termination retains an interrupted result. A host-managed service may own the job; disable automatic campaign restart. A host failure is recorded as interruption, not an implicit resume.

Existing portable producer and replay implementations:
- `dml_core/scripts/agent_episodes.py`
- `dml_core/scripts/agent_campaign_evidence.py`
- `dml_core/tests/test_agent_campaign_evidence.py` for current checker contracts.

Create a fresh specification compatible with SPEC_VERSION_V2 and the exact checker; use current source/runtime/model/corpus/selection hashes, model identity, snapshot inventory and limits. Historical specifications under docs/artifacts illustrate old schemas/profiles only; do not reuse their absolute paths or identities. The independent grader must review the actual declaration and wrapper before model tasks execute. No ready-made host-bound specification or launch approval ships in this branch.

Preserve the original nine-task/eight-intent selection, corpus, logical prompts, retrieval, grammar and fixed sampling: temperature 0.7, top_p 0.8, top_k 20, min_p 0, seed 0, one candidate. Budget: six steps, 256 output tokens/call, 1,536 output and 32,768 cumulative input tokens/task, 300 seconds/task. Do not turn synthetic diagnostics into task tuning.

## 5. Full campaign and independent replay

Only after the preceding checks, bind these arguments in the reviewed durable wrapper. Set DML_BUNDLE and DML_RUN to actual absolute external paths; DML_RUN is fresh and unique.

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.agent_episodes \
  --consumer-profile qwen3-8b-gguf-action-json-sampled-v1 \
  --snapshot-directory "$DML_BUNDLE" \
  --work-directory "$DML_RUN/episodes" \
  --output "$DML_RUN/campaign.json" \
  --max-steps 6 --output-tokens 256 \
  --max-input-tokens 32768 --max-output-tokens 1536 \
  --max-transcript-bytes 262144 --max-event-bytes 4194304 \
  --max-episode-bytes 16777216 --wall-time-seconds 300
```

Do not use --scenario or --task filters. Exit 1 may mean a fully completed campaign containing failed tasks; inspect the actual artifact and receipt, do not automatically rerun. Harness errors and unknown costs must remain visible.

Set DML_SPEC_SHA to the SHA256 of the independently reviewed specification, then run the existing data-only checker:

```bash
python -m scripts.agent_campaign_evidence \
  --spec "$DML_RUN/spec.json" --spec-sha256 "$DML_SPEC_SHA" \
  --campaign "$DML_RUN/campaign.json" --snapshot-directory "$DML_BUNDLE" \
  --source-root "$PWD" --output "$DML_RUN/primary-evidence.json"
```

The independent grader then invokes the same checker into a separate evidence file, verifies actual source/provenance/freeze/execution authenticity and compares results. Checker success alone does not prove authentic trained-model execution.

M7 needs both named tasks to pass plus every original gate (GATES in the checker), including actual model-owned supersession and predecessor feedback. Do not introduce an all-nine-success requirement or waive failed original gates. Retain every failure.

## Reporting and branch boundaries

Keep all model weights, source downloads, virtual environments and raw run directories outside tracked Git paths. Before any commit, inspect `git diff --stat` and `git status --short`; never `git add .` over generated evidence. Publish a compact summary (target under 10 KB) with task results, token/latency totals, unknowns, hashes, source/runtime/model IDs and grader decision. No raw campaign/log/weight uploads.

Update the current milestone ledger without rewriting historical reviews. M7 evaluation durability does not close M10 product replay/export/retention. No merge, deployment or production-readiness claim is authorized by a passing smoke test.

References: [PR118](https://github.com/mmckeen-nv/DML/pull/118), [CI346](https://github.com/mmckeen-nv/DML/actions/runs/36373086151), [milestone plan](docs/bare-metal-milestone-plan-2026-09-28.md).

