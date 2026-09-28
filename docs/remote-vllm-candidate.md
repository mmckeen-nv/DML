# Remote Nemotron candidate

`nemotron-remote-vllm-action-json-v2` is the corrected experimental candidate for
`http://192.168.50.91:8000/v1`, served model `nvidia/nemotron-3-super`.
It inherits the original v2 action parser, tool authorization, recovery guidance,
task corpus, retrieval, predecessor feedback, budgets and acceptance gates.
It does not qualify the local Qwen3 profile or establish production readiness.

The retained `nemotron-remote-vllm-action-v1` candidate uses the vendor template.
Its nine retained requests exposed tools and permitted their JSON grammar branches,
but the rendered template also instructed XML tool calls, conflicting with DML's
JSON action protocol. All nine outputs were finals; no tool calls were discarded.
This observation motivates a new renderer; it does not establish causation or
permit rewriting/retrying that campaign.

## Token boundary and provenance

The snapshot holds a frozen remote manifest and copies of the observed tokenizer,
configuration and chat template. Local tokenization must equal `/tokenize` before
each dispatch. Generation uses `/v1/completions` with explicit input token IDs,
`add_special_tokens=false`, request-derived structured JSON output, and returned
token IDs. The client checks the echoed prompt, model name, IDs, decoded text,
usage, token budgets and immutable request binding. Replays run without HTTP.
For v2, the frozen DML template preserves original content using reversible escaping
and retains every non-content message field and complete tool schema in an indexed
JSON data sidecar. It uses native ChatML framing and the observed Nemotron
`<think></think>` nonthinking suffix, with no vendor XML function instructions.
This is a new candidate-specific rendered prompt; the logical DML messages,
policy, tools and task prompts remain unchanged. Local encoding of that exact
rendered string must equal `/tokenize` with `prompt` and `add_special_tokens=false`.
The server does not apply a second chat template. V1 retains its original private
argument-mapping rendering and OpenAI string-argument tokenizer request.

Sampling is temperature 0.7, top-p 0.8, top-k 20,
min-p 0, seed 0, one candidate, repetition penalty 1, and zero presence/frequency
penalties. No retry, response repair or reranking is performed.

HTTP evidence cannot independently attest engine tensors/attention masks, loaded
weights, tokenizer configuration, sampler ordering, request-owned KV isolation,
or deterministic GPU execution. A request for structured output is not proof of
the server's internal token masks. Replays independently validate accepted actions.
A client timeout does not prove server cancellation or zero consumption; unknown
usage remains unknown and fails the completeness gate.

The historical v1 server lacked a revision argument; its cached revision remains
an observation rather than an attested loaded revision. A new pinned runtime may
supply a reviewed `dml-remote-runtime-attestation-v1` receipt binding the endpoint,
model, exact revision, final manifest hash and absolute evidence-file hashes.
`--runtime-attestation` requires the independently reviewed SHA256 too. The freezer
rehashes every referenced file and includes them in the execution freeze. This is
host-observed startup and pinned-byte provenance, not cryptographic proof of GPU
tensors. Without that receipt, `loaded_model_revision_attested` stays false.

## One-time execution

Keep the snapshot and all raw HTTP/events/task/campaign evidence on the execution
host, outside Git. Use the isolated Python environment and patched SQLite runtime.
The maintained `freeze_remote_campaign.py` freezes the full ordered corpus,
source, installed package files, native SQLite library, snapshot and limits.
It does not start generation. An independent reviewer must inspect the actual
specification and lifecycle evidence before launch.

`durable_campaign.py launch RUN --unit UNIQUE_NAME` starts a user systemd service
with no restart, a permanent launch claim and exclusive worker lock. Each validated
episode frame is fsynced before dispatch acknowledgment. Progress, terminal status
and failure-inclusive task reports survive controller disconnect. ExecStopPost
records supervisor death. After a host reboot, `reconcile RUN` records interruption
without resuming. Physical reboot/power-loss durability is not established by
synthetic process tests. Verify that host sleep cannot interrupt the run.

The frozen limits are six steps, 256 output tokens per call, 1,536 cumulative output
and 32,768 cumulative input tokens per task, 300 seconds per task, 262,144 transcript
bytes, 4 MiB event bytes and 16 MiB episode bytes. No task or scenario filter is used.
Exit 1 can be a completed campaign containing task failures; it is not a retry signal.

Run `scripts.agent_campaign_evidence` twice sequentially against the same frozen
specification/campaign: one primary replay and one independent reviewer replay.
Preserve failures. All original `GATES` remain unchanged; the named tasks
`supersede_then_answer` and `read_both_commits` must pass for M7 acceptance.
No source CI, active-engine attestation, merge or production-readiness claim follows
from local tests or a successful checker exit. Publish only a compact summary.

## Qualification and explicit v2 freeze

Run source tests without endpoint access or generation:

```bash
python -m pytest -q dml_core/tests/test_remote_vllm_action_input.py \
  dml_core/tests/test_remote_vllm_json_renderer.py \
  dml_core/tests/test_remote_episode_integration.py \
  dml_core/tests/test_remote_vllm_synthetic.py \
  scripts/test_durable_campaign.py scripts/test_freeze_remote_campaign.py
```

The `remote-vllm-contracts` CI job requires these tests to pass with zero skips.
Before generation, prepare the separate three-case synthetic qualification suite
with the frozen candidate, exact limits and `nemotron-remote-vllm-action-json-v2`,
then independently review its declaration. It does not use evaluation tasks,
force tool calls, narrow the final-action branch, repair responses or retry cases.
Passing synthetic diagnostics does not authorize an evaluation campaign by itself.

After host/runtime qualification and independent review, create a fresh freeze
without launching generation (replace paths and the reviewed digest):

```bash
python scripts/freeze_remote_campaign.py \
  --candidate-root /absolute/new-candidate \
  --source-root /absolute/DML-qualify \
  --sqlite-library /absolute/sqlite/lib/libsqlite3.so \
  --consumer-profile nemotron-remote-vllm-action-json-v2 \
  --runtime-attestation /absolute/runtime-attestation.json \
  --runtime-attestation-sha256 REVIEWED_SHA256
```

Omitting the profile selects the manifest's declared profile, with v1 as the
legacy default only when the manifest has no profile. An explicit mismatch is
rejected. V2 requires its exact frozen renderer and tracks renderer, synthetic
qualification and freezer sources. The producer command is regression-tested
through the real CLI so profile selection and JSON numeric limit types survive
serialization. Source CI qualification remains false in producer artifacts;
separate exact-commit CI evidence and reviewer decisions must support that claim.


The separate `nemotron-remote-vllm-native-tools-v1` candidate uses the pinned
vendor native chat template and automatic tool selection. Its system policy
changes only the action transport instructions: one native function call for a
tool action, or the original complete final-action JSON object for a final.
The original final, grounding, retrieval and lifecycle policy tail remains.
This candidate removes all-turn JSON grammar; it does not claim the old grammar
guarantee or inherit qualification from the JSON profiles. Corpus, limits,
authorized tools, argument validation, retrieval and acceptance gates remain.

Evidence retains genuine raw output token text separately from the deterministic
DML action projection, raw server assistant message, projection error and digest.
Exactly one native call maps through the existing action parser and tool gateway.
Native tool-call IDs are explicitly bound to gateway IDs; authentic tool-result
bytes return under the native ID. Parallel calls, mixed prose/calls, unsupported
raw syntax, malformed arguments and prose finals fail without repair or retry.
Offline replay verifies the raw native message/projection binding and history.
No per-token grammar-mask or terminal-prefix guarantee is asserted.

Freeze this candidate using the same command above with
`--consumer-profile nemotron-remote-vllm-native-tools-v1` and a separately
reviewed manifest/runtime receipt. The freeze tracks the native consumer and
`dml_core/scripts/native_dml_synthetic.py`. The remote CI lane includes native
adapter, transport, synthetic and freezer controls without endpoint generation.
Native synthetic qualification is distinct from both the earlier native lookup
diagnostic and the unchanged nine-task evaluation corpus.
