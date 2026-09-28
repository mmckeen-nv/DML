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


`nemotron-remote-vllm-native-tools-v2` separately admits assistant commentary
alongside exactly one native tool call. Commentary is retained verbatim in raw
evidence and assistant history but never becomes a tool argument, tool result,
claim, citation or authority. Its profile-specific policy and runtime identity
differ from v1; v1 retains mixed-content rejection for reproducible replay.
Standalone prose finals still fail the original full final-action JSON contract.
Both versions retain the same authorized action parser, tool gateway, limits,
corpus and acceptance gates. V2 requires its own reviewed qualification and
freeze; no prior failed v1 result is reinterpreted as success.


`nemotron-remote-vllm-native-tools-v3` appends generic completion guidance to
the v2 system policy. It asks the model to track acknowledged state, avoid
repeating committed operations with old references, retrieve for remaining
information gaps or required readback, and emit the original final JSON only
when requested operations, required verification and evidence conditions hold.
The exact guidance text and hash are bound into the new runtime identity.
Native transport, automatic tool choice, sampling, final schema, authority,
budgets and acceptance gates remain unchanged; no final branch is forced.
V1 and v2 policy identities and replay behavior remain intact. The reused four
synthetic cases are development regression fixtures, not held-out evidence.
Prior failed attempts remain retained; a new declared run is a new candidate
qualification attempt, not an automatic retry or reinterpretation.


`nemotron-remote-vllm-native-tools-v4` adds generic guidance that an answer
being available does not satisfy separately requested operations or verification.
It separately admits strictly bound native reasoning as non-authoritative
metadata, with the original final JSON and tool validation unchanged. API
reasoning is retained in raw evidence and aliased exactly to the pinned template's
`reasoning_content` history field; it never becomes final content or tool arguments.
V1–v3 identities remain preserved.

Synthetic declaration v3 corrects the dependency fixture: ten directory pointers
and a separate destination are predeclared, with a real gateway preflight on an
isolated fixture before generation. The first declared lookup must expose the
key and exclude the destination/answer; the destination lookup must succeed.
Actual model retrieval results remain unchanged. Qualification additionally
requires no earlier destination/answer exposure before the key-dependent call.
A failed separation check aborts generation; no model-result filtering, forced
queries, extra model budget or automatic retry is introduced.

The dependency fixture declares its construction against the existing synthetic
16-bin lexical embedder: the destination key and answer occupy a bin absent from
the initial lookup query. This is controlled fixture construction, not evidence
of production retrieval quality or selection based on model outcomes. Actual
model-query leakage still disqualifies dependency evidence. Native v4 alone opts
into the digest-bound `native-reasoning-metadata-v1` request policy; old requests
retain their exact default schema, and other profiles reject that marker.

## Native v5 lifecycle recovery candidate

The newer native-tools-v5 recovery development qualification failed both declared cases (0/2), despite both offline evidence replays passing. The source case truncated before exercising recovery; the replacement case recovered a verified conflict, retrieved fresh state, committed a model-owned supersession and read back, but reached the step limit without a final answer. No new nine-task campaign was frozen or launched. Exact-source [CI 36468583308](https://github.com/mmckeen-nv/DML/actions/runs/36468583308) passed all 21 jobs on `68fdc74`; it does not override failed live qualification. See [v5 recovery qualification](artifacts/native-vllm-v5-recovery-qualification-outcome-2026-09-28.json). Eight calls used 17,335 input and 1,163 output tokens; no HTTP/model errors or timeouts occurred. Both replayed all eight raw calls and authority/history paths, including one nonempty reasoning response. The unchanged 256-token output cap and six-step budget remain binding.

`nemotron-remote-vllm-native-tools-v5` separately versions recovery from a
verified stale record decision. V4's completed 8/9 campaign remains unchanged;
its repeated stale mutation and terminal failure are not retroactively recovered.
The model, native template, sampling, system guidance, retrieval, authorization,
final claims/citations contract, corpus, budgets and acceptance gates stay fixed.

Recovery is limited to supersession source/replacement or retirement target
full-record digest mismatches found in a validated, same-scope snapshot under
write ownership before any save attempt in that invocation. An invocation-owned
proof must bind the exact exception, request, operation key, scope, expected and
observed digests, and snapshot revision. An exception name or `effects:none`
label alone is insufficient: `ReceiptLifecycleConflict` also occurs while
validating existing or post-commit receipts.

The rejected action and authentic error remain failure evidence. Structured tool
feedback explicitly says that the attempted operation did not execute; it grants
no new record authority and does not count as a successful operation. The model
chooses its next action within the remaining original steps, input/output tokens,
transcript size and wall time. No automatic mutation retry, operation-ID
replacement, action substitution, error suppression or final-answer repair is
permitted. Earlier profiles preserve their terminal behavior and identities.

Update/promotion conflicts, eligibility failures, authorization violations,
missing records, receipt-validation errors, unresolved dispatches, integrity
failures and unknown commit effects remain terminal. Unproven lifecycle errors
cannot authorize recovery. This candidate requires independently reviewed
regressions, predeclared noncorpus live recovery evidence, exact-source CI and a
new approved freeze before another complete nine-task campaign.

The recovery development suite declares two external peer-update cases, one for
each supersession record. A real peer update occurs only after the model has
retrieved the original references; it preserves the claim value and record
eligibility while changing the immutable record digest. The peer receipt and
initial fixture remain separate evidence. A separately labeled verifier baseline
applies only that authenticated, predeclared update; the original verifier is
unchanged. Qualification requires an actual classified conflict followed by the
model's own retrieval, supersession, read-back and valid cited final. No conflict
means the recovery branch is unqualified. Retirement is covered by controlled
journal regressions, not claimed as live-qualified by these two cases.
