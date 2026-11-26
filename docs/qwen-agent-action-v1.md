# Qwen agent-action JSON companion v1

This candidate extends the [Qwen exact-input companion](qwen-model-input-v1.md)
with an explicit **`qwen2-action-json-v1`** profile for the bounded
[milestone 7 harness](live-agent-outcomes-2026-09-20.md). At this documentation
freeze, its original bounded source received independent **9.6/10** acceptance
and passed CI 329. A subsequently discovered worker-cleanup blocker rejected
that public source's completion state at 9.2/10. The scoped cleanup (**9.6/10**)
and grounding (**9.7/10**) repairs are now integrated; final mandatory validation
passed **774 tests with zero failures, errors or skips** across **20 modules**,
with **433 unchanged source hashes**. Both strict Ruff scopes and maintained mypy
passed. The [combined final review](artifacts/agent-episode-grounding-cleanup-review-2026-09-20.json) accepted **9.6/10 with no source blockers**. Publication/CI for the repaired source and genuine trained-model campaign acceptance remain required. This continues
serial source gate 20; milestone 7 remains open and `production_ready` is false.

`daystrom_dml.services.qwen_action_input.LocalQwenActionInputConsumer` exposes
`compile`, `execute`, `close` and context-manager operations. The episode CLI
selects it explicitly with `--consumer-profile qwen2-action-json-v1`. The default
remains `gpt2-v1`; generic `gpt2-v1` and `qwen2-instruct-v1` generation and the strict
action parser remain unchanged. Unknown profiles reject instead of falling back.

The current candidate also clarifies the shared harness `AGENT_POLICY`: retrieve
all task-required records, avoid identical read loops, and complete explicitly
requested lifecycle actions before claiming completion. It removes fictitious
query/value/citation-ID examples and distinguishes producing a final answer from
performing a requested mutation. Public tool descriptions receive the same generic
clarification; no task-specific expected value or forced action sequence is added. Thus the shared harness
prompt changes even though generic consumers' generation paths and the parser
remain unchanged. The next campaign must declare both policy and grammar changes;
any difference in its result cannot be attributed to grammar alone.

## Admission and separate dependencies

The `agent-action` extra separately pins the existing model-input dependencies
and **`xgrammar==0.2.7`** plus **`apache-tvm-ffi==0.1.12`**. Missing or differing
grammar distributions fail admission; they are not silently installed or ignored.
The generic model-input extra does not acquire a grammar dependency.

The supported installation path for this optional profile is the tested
**Linux / Python 3.12 / CPU** environment. From the repository root, install the
CPU PyTorch wheel first, then the extra:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[agent-action]'
```

A fresh ordinary PyPI resolution may select a CUDA-flavored Torch build, which
correctly fails the existing exact `2.8.0+cpu` runtime admission. Do not weaken
that check or infer profile portability from the core project's broader platform
CI. XGrammar's Linux x86_64 packaging installs Triton transitively; this profile
still explicitly uses CPU token-mask execution and does not dispatch GPU work.

The action consumer verifies the same five-file Qwen snapshot, model weights,
configuration, tokenizer and native v2 template as the generic Qwen consumer.
It retains CPU/float32, eager attention and greedy decoding. Its additional
runtime identity, `dml-qwen-action-runtime-v1`, binds the base runtime identity,
exact grammar versions, the public action-schema digest and a separate
`public_tools_sha256` over the complete public tool definitions (including their
descriptions), declared key-order policy,
stop/special-token policy, compiler settings, request capacity and prohibition
on post-generation repair. Runtime policy drift rejects execution.

## Public syntax and model-owned decisions

`dml-agent-action-grammar-v1` derives its JSON schema from the exact public tool
definitions advertised in the request. Unknown, duplicate, modified or oversized
tool lists reject admission. The schema admits only the advertised tool subset,
plus a final-answer branch. No corpus, hidden expected answer, memory-store state,
verified success result or target value is an input to grammar construction.

The declared generation order is `schema_version`, `kind`, then `name` and
`arguments` for a tool action, or `answer` for a final action. Claims use `key`,
`value`, then `evidence_ids`; argument fields follow their public schemas. This
order is part of the profile identity. Extra keys are excluded. Any parser-valid
action has a representation in this order; the unchanged parser still owns
semantic uniqueness, byte limits and complete-action admission.

The model chooses whether to invoke a tool or finish, which advertised tool to
invoke, every argument, claim key/value, citation and final answer. The grammar
does not force retrieval, supersession, a task-specific sequence or a correct
answer. Empty claims and other semantically unsuccessful outputs can still be
syntactically valid. Independent receipt/state and task verifiers retain sole
responsibility for establishing the task result. Bounded public syntax alone
cannot close live qualification.

## Immutable request ownership

Compilation validates and detaches the complete request before producing the
same immutable exact-input artifact used by the harness. The consumer retains
canonical request bytes indexed by the artifact digest. It inserts no hidden
prompt, schema text or expected answer into the model input.

Execution authenticates the artifact's consumer identity, runtime identity and
signature, then checks that its request digest matches the retained immutable
request. The grammar is built from that artifact's advertised tools. Compiling
A, then B, then executing A uses A's request; there is no last-request fallback.
The same authenticated artifact may be executed again with fresh generation
state.

At most **64 compiled request bindings** are admitted over one consumer lifetime.
Exhaustion rejects the next compile without eviction or rebinding. Successful
execution does not silently release a binding that a caller may reuse. Closing
clears all bindings and prevents subsequent compilation or execution.

## Token masking, bounds and failure behavior

Each execution constructs a fresh single-threaded grammar compiler with its
internal cache disabled, plus a fresh matcher bound to the authenticated input
prefix. Grammar preparation is inside the supervised execution deadline. Qwen's
dynamic KV cache is also fresh per generation call and cannot be supplied,
returned, persisted or reused across calls.

At each generation step, the grammar masks inadmissible logits before greedy
selection. Only admitted tokenizer-vocabulary IDs are eligible; all special
control tokens except the verified final EOS are excluded. The processor requires
one finite CPU float32 logits row, an unchanged CPU integer input prefix, and a
valid mask layout. It checks every selected token against the same matcher,
including the final generated token. Invalid prefixes, bypassed masks, out-of-range
IDs, nonfinite logits, tokens after termination and an empty admissible-token set
fail execution.

The exact compiled input IDs and original output reservation remain authoritative.
Every actual generated ID counts toward the bound and measured output cost,
including EOS. The existing Qwen decoder omits only a verified trailing EOS from
the displayed text; raw IDs retain it. A complete JSON action exactly at the
token bound can pass without EOS. An incomplete bounded prefix remains intact
and is rejected by the existing action parser. No token is injected to finish
JSON; no output is repaired, stripped of fences or substituted after generation.
Execution failure or interruption retains its observed or unknown accounting
through the existing episode contract.

## Supervised temporary-file ownership

The Linux/CPU episode supervisor now creates a private directory for each worker,
records its device/inode identity and redirects child temporary files into it.
After confirming worker termination, the surviving parent removes that owned
tree, including read-only private model copies left by a killed child. Permission
recovery rejects links and paths outside the resolved owned root. Failure to
confirm termination or reclaim the tree becomes a visible `runner_error`; it
cannot be reported as successful cleanup. User-supplied model bundles and memory
authority are outside that temporary tree.

This repairs the observed surviving-supervisor leak. It does not claim automatic
reclamation or durable event reconstruction after the supervisor itself dies.
The episode child deadline still excludes subsequent cleanup and publication;
the contract does not invent a constant-time filesystem deletion bound. Broader
platform cleanup support requires separately demonstrated qualification.

## Replay and qualification

Raw events retain the complete request, compiled identity and actual generated
IDs. The independent campaign checker reconstructs the constrained runtime
identity and request-owned grammar, checks admitted output IDs against that
grammar, and retains the existing strict action parsing, semantic verifiers,
causal-event and failure-inclusive accounting checks. Grammar replay establishes
consistency; it cannot authenticate the producer or prove the logits that ran.
Independent source, trained-weight provenance and execution review remain required.

The separate profile passed its own source review at **9.6/10**, exceeding the
required 9.5/10. The [final-source validation](artifacts/agent-episode-action-validation-2026-09-20.json)
has SHA-256 `5ddd1eaf58fdfdc5d423a7f7243912896ad227e4091ca5463879dbdc445ba032`.
The [strict admission record](artifacts/agent-episode-qwen-action-admission-2026-09-20.json),
SHA-256 `6a9f65a96bb79e907953c8ebb78c616a82b3a72cedc4100e66604c181e6f52a3`,
records **zero generations** and does not qualify a semantic task.
Those records belong to the preceding action-profile source, which passed CI 329.
The [combined repaired-source validation](artifacts/agent-episode-grounding-cleanup-validation-2026-09-20.json)
and [433-file source manifest](artifacts/agent-episode-grounding-cleanup-source-2026-09-20.json)
retain the current 774-case result. Scoped
[cleanup](artifacts/agent-episode-worker-cleanup-review-2026-09-20.json) and
[grounding](artifacts/agent-episode-grounding-review-2026-09-20.json) reviews are supplemented by [combined final review](artifacts/agent-episode-grounding-cleanup-review-2026-09-20.json) at **9.6/10 with no source blockers**. New-source CI and actual live acceptance remain required. A new
pre-generation declaration
must bind its runtime, source, trained snapshot, public corpus, limits and unchanged
milestone acceptance gates before a genuine campaign. Prior failed campaigns
remain retained with their original profiles and costs. Syntax constraints do not
establish model semantic competence, baseline advantage, long-horizon quality or
overall production readiness.
