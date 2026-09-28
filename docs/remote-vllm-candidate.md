# Remote Nemotron candidate

`nemotron-remote-vllm-action-v1` is a separate experimental model/runtime candidate for
`http://192.168.50.91:8000/v1`, served model `nvidia/nemotron-3-super`.
It inherits the original v2 action parser, tool authorization, recovery guidance,
task corpus, retrieval, predecessor feedback, budgets and acceptance gates.
It does not qualify the local Qwen3 profile or establish production readiness.

## Token boundary and provenance

The snapshot holds a frozen remote manifest and copies of the observed tokenizer,
configuration and chat template. Local tokenization must equal `/tokenize` before
each dispatch. Generation uses `/v1/completions` with explicit input token IDs,
`add_special_tokens=false`, request-derived structured JSON output, and returned
token IDs. The client checks the echoed prompt, model name, IDs, decoded text,
usage, token budgets and immutable request binding. Replays run without HTTP.

The prompt uses the model's frozen chat template with `enable_thinking=false`
and a generation prefix. Sampling is temperature 0.7, top-p 0.8, top-k 20,
min-p 0, seed 0, one candidate, repetition penalty 1, and zero presence/frequency
penalties. No retry, response repair or reranking is performed.

HTTP evidence cannot independently attest engine tensors/attention masks, loaded
weights, tokenizer configuration, sampler ordering, request-owned KV isolation,
or deterministic GPU execution. A request for structured output is not proof of
the server's internal token masks. Replays independently validate accepted actions.
A client timeout does not prove server cancellation or zero consumption; unknown
usage remains unknown and fails the completeness gate.

The existing server was started without a revision argument. The cached revision
is an observation, not a verified loaded revision. Record active revision as
unattested, and retain the image/container identity, start time, launch arguments,
server configuration, observed file hashes and both client/server package versions.

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
