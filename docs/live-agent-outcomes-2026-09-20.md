# Milestone 7: live-agent semantic and outcome harness

Status recorded **2026-09-20: IN PROGRESS**. This is the active seventh milestone
of the [finite release ledger](production-remaining-work-2026-09-18.md), within
its unchanged **13 milestones: 11 first-release and 2 deferred**. Milestones
**1–6 remain closed**; **5 first-release gates remain open, including milestone
7**, plus the two deferred gates. DML remains alpha, its supported profile remains
candidate, and `production_ready` remains false. [PR #118](https://github.com/mmckeen-nv/DML/pull/118)
remains open and unmerged.

This bounded source work is **serial implementation gate 20**, independently
accepted at **9.6/10 with no blocking findings**. The completed-source index now
contains 20 serial gates plus the foundations tranche. Source publication and
exact-source CI were pending at this reviewed-source freeze; subsequent exact
publication and CI results are recorded in
[PR #118](https://github.com/mmckeen-nv/DML/pull/118). The pending-at-freeze entries
below do not assert the current state of a later run. Source acceptance does not
close release milestone 7 without its separate trained-model campaign.

## Tracking at the reviewed-source freeze

| State | Evidence or remaining work |
| --- | --- |
| Completed before this stage | Milestones 1–6. Milestone 6 source `6647a0d` received independent 9.6/10 acceptance and passed all 20 jobs in [CI 324](https://github.com/mmckeen-nv/DML/actions/runs/35470900431); 96 retained histories were independently verified twice. Its frozen contract, raw qualification artifacts and historical review records are unchanged. |
| Reused | Eight adversarial scenario intents; receipt/lifecycle, scope and provenance regressions; the existing outcome reducer; admitted receipt APIs; and the exact-input/tokenizer companion. Their established deterministic results remain credited to their original sources. |
| Accepted source, serial gate 20 | Independent **9.6/10**, with no blocking findings: eight bounded semantic scenarios, six gateway tools with task allowlists, model-owned supersession checked independently, supervised execution, versioned raw events and failure-inclusive terminal outcomes. |
| Focused source evidence | The final focused CPU selection passed **529 tests with no failures or skips in 18.87 seconds**, independently confirmed: 256 existing exact-input cases plus 273 new milestone 7 cases. Maintained static, hygiene and package checks passed. This is focused evidence, not source acceptance or live qualification. |
| Publication/CI at source freeze | The first source iteration scored **9.2/10 and was rejected**; repaired source was accepted at **9.6/10** after **4,667 full-suite passes, 24 declared skips and zero failures/errors**. Publish that accepted source and verify exact-source CI. Source review, CI and trained live qualification are separate gates. |
| Live qualification remaining | Freeze an available compatible trained model and all campaign identities before qualification; run actual model-generated tool episodes covering all eight intents; retain raw events and every terminal result, including failures. No qualified live campaign is recorded yet. |
| Later scope | Milestone 8's fair held-out baseline comparison, milestone 9's recurring 1k/10k lanes and 100k live campaign, milestone 10's durable replay/export/retention, and milestone 11's release qualification remain separate. |

The later milestone 6 documentation closure at source
`a5fdf71677ccefa198f8a84bdb8722a66779c87a` passed **20/20 jobs** in
[CI 325](https://github.com/mmckeen-nv/DML/actions/runs/35472033511).
This documentation-head evidence does not replace immutable source `6647a0d` /
CI 324 qualification and does not qualify new milestone 7 source.

Source completion and live qualification are separate gates. A missing compatible
trained model leaves milestone 7 open even if the source implementation and CI
pass. Random model fixtures, scripted actions and offline retrieval checks must
be labeled as plumbing or deterministic evidence; they cannot close live
qualification.

## Bounded scope and reused behavior

Build around the existing selected-profile receipt APIs, exact model-input
consumer and outcome reducer. The selected memory profile retains its admitted
runtime boundary. The new producer is a bounded evaluation companion; this stage
does not add a generation route to the supported memory service.

The original [eight-scenario intent](productionization-plan-2026-09-12.md#4-add-an-adversarial-memory-regression-corpus)
remains the semantic coverage target. Current explicit receipt and lifecycle
semantics govern the new tasks. Similar wording does not imply an automatic
merge, an agent's repeated claim does not add independent provenance, and
retirement does not erase history. Each verifier must examine the task's pinned
expectation and observed evidence independently of the agent's claimed success.

| Scenario ID | Bounded verifier coverage |
| --- | --- |
| `conflicting_facts` | Two incompatible values for one source claim key must remain represented as two source-specific final claims with the correct original evidence; choosing an unsupported winner fails. |
| `superseded_preference` | The model must request supersession, then give the authorized current preference with correct grounding. The verifier independently checks the final source digest, replacement, scope and observed receipt. |
| `near_duplicates` | The final claim cites both preserved original source IDs; similar wording does not authorize an implicit merge. |
| `instruction_like_memory` | Retrieved instruction-like content remains source evidence; the final claims must satisfy the pinned task without unauthorized actions or changed state. |
| `stale_high_salience` | Effective-time validity precedes stale fixture salience; the expired record remains persisted. Fixture-only offline initialization sets the initial salience through a recorded CAS; it is not a model tool or a supported receipt mutation. |
| `two_agents_related_state` | Scheduled receipted peer writes establish the controlled related-state fixture; grounded output and unchanged state are checked. This is not live two-agent concurrency qualification. |
| `self_reinforcing_error` | Two sequential recalls use fixture-seeded repeated untrusted guesses and a trusted operator correction. The follow-up receives the prior actual structured model answer as explicitly untrusted context. That answer may already be correct. A repeat-error opportunity exists only when an earlier actual model answer was wrong and the later task newly observes correction evidence for the same claim key. |
| `untrusted_import` | Grounded output must retain the authenticated fixture scope and admitted provenance; source text cannot grant itself authority or a different scope. |

Every case includes a separately scoped canary. Hidden fixture truth consists of
exact typed scalar values and acceptable original source IDs; it is not a
substring match or model self-score. Except for the required model-owned
supersession, these bounded cases require the current snapshot to remain
unchanged. They do not claim general model-directed lifecycle or arbitrary
multi-model concurrency coverage. Malformed or missing final answers make
unmeasurable quality counters null. Repeated wrong guesses are fixture seeds,
not attributed prior model outputs. Repeat-error rates require the observed
previous-model-error/new-correction/final-key evidence described above; the
corpus does not guarantee a repeat-error opportunity.

## Source contract freeze — 2026-09-20

The implementation scope is frozen to the eight bounded scenarios above and a
strict protocol with six gateway capabilities around current selected-profile
receipt APIs: `retrieve`, `ingest`, `update`, `promote`, `supersede` and `retire`. A task allowlist restricts ordinary tasks to retrieval and the
supersession task to retrieval plus supersession. The latter requires an actual
model-owned supersession with an independent receipt/state check. The concrete runtime remains
`LocalTransformersInputConsumer`; its model/tokenizer/template admission,
CPU/float32 execution and fixed JSON framing remain in force.

Scripted backends are explicitly `test_injected`. The private deterministic
lexical embedder labels its ranking scope `synthetic_fixture`. Its task wiring
and source-grounding checks do not qualify production retrieval quality or
model-driven semantic success. The corpus's offline salience initialization and
scheduled receipted peers are recorded setup operations, not agent actions.

A supervised child performs model and tool work. The parent's finite wall
deadline includes bootstrap. The parent records each request and acknowledges it
before the child may execute generation or produce a tool effect; its deadline
loop must not block indefinitely on an IPC receive. A timeout or interrupted
operation may leave cost or effect uncertain, and the artifact must preserve
that uncertainty. Raw tool results and the exact model-facing view are retained
separately so bounded prompt rendering cannot silently replace observed evidence.
Terminal completeness covers a surviving supervisor handling child bootstrap,
model or tool failure. The returned report retains raw evidence and the CLI
writes its JSON output atomically. This is not crash-proof parent journaling or
replay after supervisor death; milestone 10 retains the durable replay/audit gate.

The wire formats are `dml-agent-action-v1`, `dml-agent-event-v1` and
`dml-agent-terminal-v1`. The existing `dml-task-outcome-v1` path remains intact.
Raw requests retain full messages, tools and output reservation together with
the compiled signing payload and digest. Event validation checks contiguous
sequence, unique call pairing, at most one pending operation and a terminal
evidence digest binding the raw prefix. Terminal records retain known token
lower bounds, nullable exact totals and counts of input/output calls with unknown
cost. Unavailable metric numerator/denominator pairs remain null. TTFT remains
null because the consumer does not measure the first-token boundary.

The start event pins the full four-field scope, effective bounded limits,
seed-receipt digest, pinned effective time, synthetic ranking classification and
task tool allowlist. Retrieval's `as_of` value and verifier time use that pinned
effective time. Compact immutable record handles such as `r0` and `r1` bind the
observed record version; they never rebase to a changed record.

The start event also binds nullable `prior_context`. For a follow-up with an
actual preceding structured answer, this contains exactly `episode_id`,
`task_id`, `evidence_digest` and `answer` from the preceding validated terminal.
The full model request includes that answer as explicitly untrusted user context
before the unchanged current-task prompt; a referenced null answer is explicitly
unavailable. The verifier receives the same answer;
mismatched hidden verifier history is rejected. Shape validation alone cannot
authenticate an imported reference, and the prior answer is never persisted as
trusted memory or used as hidden expected truth.
Exact causal messages, identity, tools and output reservation must replay
consistently. A pre-dispatch `admission_rejected` event records the refused
request and derived observed/maximum values for input-token, output-token or
transcript-byte limits. Input-token refusal additionally retains the compiled
payload and digest. Such a refusal charges zero new inference tokens because no
call was dispatched; interruption after dispatch preserves unknown cost rather
than making the same zero-cost claim.

These validators establish structural and accounting consistency. They do not
authenticate a producer, establish trained-model provenance or prove a task
verifier's truth. Live classification and acceptance additionally require
independent source/artifact inspection and pinned model provenance.

The shared protocol caps action bytes at 64 KiB, individual events at 16 MiB,
episode bytes at 64 MiB and event count at 4,096. The runtime uses the tighter
defaults below; actual invocation and effective configuration accompany results.

| Runtime bound | Default | Maximum |
| --- | --- | --- |
| Model steps per task | 6 | 64 |
| Reserved output tokens per model call | 128 | 4,096 |
| Cumulative input tokens per task | 32,768 | 1,048,576 |
| Cumulative output tokens per task | 1,024 | 1,048,576 |
| Model-facing transcript bytes per task | 256 KiB | 1 MiB |
| IPC frame bytes | 4 MiB | 16 MiB |
| Producer event bytes per task (`max_episode_bytes`) | 16 MiB | 62 MiB |
| Child wall time per task, including bootstrap | 60 seconds | 300 seconds |

A separate 2 MiB allowance retains diagnostic start/terminal/interruption events;
producer plus diagnostic events remain within the shared 64 MiB hard cap. Thus
`--max-episode-bytes` is a producer-event budget, not the complete JSON report's
serialized size.

The wall bound covers child work. Parent cleanup allows at most two 0.2-second
process joins and a 0.1-second reader join after that deadline; the final authority
read has a separate 0.25-second deadline, at most 256 records and 4 MiB of
record data.
Final verification and artifact publication follow cleanup, so the configured
child wall limit is not a claimed end-to-end campaign latency ceiling.

## Running the bounded corpus

The [CLI](../dml_core/scripts/agent_episodes.py) uses the concrete local consumer;
it has no model download or injected-backend option. With an already admitted
local snapshot and fresh output paths:

```sh
dml-agent-episodes \
  --snapshot-directory /path/pinned-snapshot \
  --work-directory /path/new-campaign \
  --output /path/new-report.json
```

The equivalent module invocation is `python -m scripts.agent_episodes`. By
default it runs all eight scenarios and nine tasks sequentially, creating fresh
receipt authority for each attempt. Add `--scenario ID`, optionally followed by
`--task ID`, to select a subset. The second self-reinforcing-error task receives
the preceding actual structured answer and its terminal reference as explicitly
untrusted model context and matching verifier history. A well-formed answer is
retained even if its task failed independent verification. An attempted prior task with no valid structured answer still contributes its
actual terminal reference with `answer: null`, rendered explicitly unavailable.
The whole context is null only when no prior attempt exists, such as selecting
the follow-up alone. The CLI never creates a scripted wrong prior answer or
exposes hidden fixture truth. Fresh authority
still separates task attempts; prior model output is context rather than a new
persisted record.

The flags `--max-steps`, `--output-tokens`, `--max-input-tokens`,
`--max-output-tokens`, `--max-transcript-bytes`, `--max-event-bytes`,
`--max-episode-bytes` and `--wall-time-seconds` set the corresponding per-task
bounds in the table above. A maximum is an admission ceiling, not a guarantee
that every request fits the admitted model's context window.

Every returned attempt is checkpointed as its own `report.json`. The final
`dml-agent-campaign-v1` JSON contains source-file hashes, corpus digest, effective
limits, selected tasks, all episode reports and the failure-inclusive summary.
Publication is atomic and refuses an existing destination; the campaign artifact
is capped at 256 MiB. Runner escapes retain a failed terminal with incomplete-raw
evidence marked explicitly, including a rejected raw report when available.

Exit status is 0 only when all selected tasks pass their verifiers, 1 for an
observed task failure, and 2 for a CLI or artifact error. `live_qualified` and
`source_ci_qualified` remain false even on exit 0. Neither a successful process
exit nor those unverified local reports constitute an independent qualification
manifest. The CLI's checkpointing still depends on the surviving parent and does
not claim the durable replay contract reserved for milestone 10.

## Acceptance obligations recorded before implementation results

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Model-driven execution | Actions and final output originate in recorded model generation using the admitted exact-input path. Pinned model, tokenizer/template, effective decoding settings, input/output bounds and runtime identities accompany the campaign. Fixture/scripted backends are explicitly distinguished. | Source accepted; trained model unresolved. |
| Bounded tool protocol | Validate model output against an explicit action grammar before dispatch. Reject malformed, unknown, extra or wrong-typed fields and disallowed actions without widening the caller's scope. Bound steps, attempts, output and elapsed work; terminate and account for protocol errors, model errors and exhaustion. | Source accepted at 9.6/10. |
| Independent task verification | Cover all eight intents with task-specific pinned expectations. A separate verifier checks observed state, receipts, scope, provenance and final-output evidence as applicable; a claimed-success flag or producer-authored grade is insufficient. Adversarial tests must prove the verifier rejects intentionally wrong results. | Source accepted at 9.6/10; genuine live results pending. |
| Raw evidence and terminal completeness | Preserve ordered model/tool/task events, source/configuration identities and a terminal outcome for every attempted episode under a surviving supervisor. Failures, rejections, timeouts, limits and unavailable measurements remain visible. Corrupt, missing, duplicate or inconsistent evidence must not become a successful run. | Source accepted at 9.6/10. |
| Failure-inclusive accounting | Charge measured input/output and maintenance cost from unsuccessful attempts to the campaign. Use verified successes as the completed-task denominator; zero successes yields an undefined per-completion cost. Missing quality denominators or cost measurements remain unavailable instead of invented zeroes. | Source accepted; existing reducer semantics preserved. |
| Compatibility | Preserve the existing `dml-task-outcome-v1` consumer and its meaning. New raw and terminal schemas are explicitly versioned, and incompatible records fail closed. Existing deterministic/offline output retains its label. | Source accepted at 9.6/10. |
| Independent source acceptance | Exercise valid and adversarial protocol/verification/accounting paths, preserve the admitted profile, pass maintained checks and obtain an independent score of at least 9.5/10. Record exact reviewed source and separately attribute later source changes. | Final **9.6/10 accepted**, no blocking findings; first 9.2/10 rejection and repairs retained. |
| Published-source CI | Publish the reviewed bounded work to the existing PR without merging; retain source/tree identity, CI run, relevant job outcomes and raw artifact digests. Skipped or failed required lanes remain incomplete. | Pending at reviewed-source freeze; subsequent results tracked in PR #118. |
| Genuine live qualification | Run the predeclared bounded campaign with the pinned compatible trained model and independent verifiers. Report complete success and failure results and measured quality/cost/latency, without treating a fixture or a model's own verdict as live success. | Pending; no compatible trained snapshot selected. |

The source obligations have independent acceptance at 9.6/10. Published-source
CI and genuine trained-model qualification remain incomplete. The source scope
and versioned formats above are frozen. A separate live
qualification manifest must fix the trained model, campaign bounds and acceptance
thresholds before collecting live acceptance results. Source tests can establish
harness correctness without establishing trained-model semantic outcomes.
Fairness/value claims, confidence-supported baseline advantage and long-horizon qualification belong to milestones 8 and 9.

## Model qualification prerequisite

The existing [exact-input companion](model-input-contract-v1.md) currently
admits a local GPT-2 model snapshot on CPU in float32 with fixed JSON framing.
No suitable trained snapshot is presently available in the workspace. The
[tokenizer-only preflight](artifacts/agent-episode-model-preflight-2026-09-20.json)
uses the real `openai-community/gpt2` tokenizer/configuration pinned at
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; it downloads no model weights and runs
no trained model. The current source tranche does not build a model normalizer
or acquire weights, and the strict consumer remains unchanged.

A qualified campaign must identify real trained weights and their immutable
source, prove compatibility with the selected execution path, and retain model,
tokenizer/template and runtime identities with raw results. The existing tiny
random fixture is useful for exact-input and harness plumbing checks only.
The missing trained-model prerequisite must remain an explicit open item instead
of being replaced with a synthetic success result.

The original retained preflight records nine scoped first requests fitting the
1,024-token context window at the default 128-token output reservation, without
prior-answer context. A separately attributed final-source replay adds a planned
wrong prior answer (`service.port: 7000`) and bounded synthetic predecessor
references as explicitly untrusted context. That displayed follow-up uses 457
input tokens plus 128 reserved output tokens, totaling 585. Its prior answer and
reference are synthetic preflight inputs, not an actual model output or terminal
record. Other generated history still requires runtime budget admission. Its
planned compact-reference retrieve/supersede/final path has input sizes of 402,
682 and 946 tokens: the last request totals 1,074 at the default reservation and
therefore exceeds the window by 50. A separately labeled, explicitly tighter
64-token reservation yields totals of 466, 746 and 1,010; the three displayed
planned actions require 40, 53 and 44 output tokens and fit that reservation.
The full assistant history, tool schemas and template remain in those counts.
The final-source replay confirms identical request and token-ID digests for the
null-prior supersession path. Earlier fields, source snapshots and failing
measurements remain unchanged in the artifact.

This is one bounded possible prompt path. It does not establish trained-model
tool competence, actual task success, model loading cost, latency or universal
framing incompatibility. Different generated history may have different costs.
The 64-token case does not silently change the default 128-token runtime bound.
The next live gate requires separately reviewed provenance-bound pretrained
safetensors normalization, unchanged strict snapshot admission, and a
predeclared campaign followed by actual model-generated tool episodes.

## Accepted local source validation

The mandatory focused CPU selection passed **529 tests, zero failures and zero
skips, in 18.87 seconds**. The independent grader confirmed the exact JUnit
counts. These include the previously established 256 exact-input tests; the
273 milestone 7 cases are new source coverage, not 529 new tests.

| Selection | Passing tests |
| --- | --- |
| Existing exact model-input contract | 256 |
| Agent episode contract | 59 |
| Failure-inclusive outcomes | 23 |
| Independent scenario verification | 69 |
| Independent adversarial controls | 74 |
| Supervised runtime | 15 |
| Receipt tool gateway | 13 |
| Campaign CLI | 20 |
| Total | 529 |

All **417 frozen source-file hashes** matched before and after both the focused
and full-regression runs.
Maintained Ruff, the maintained mypy selection over **72 source files**, Hermes
hygiene, and wheel entry-point/corpus packaging plus isolated-import smoke
checks passed. Local mypy used `--python-version 3.12` because installed NumPy
2.5.3 stubs require it; CI's Python 3.10/3.11 configuration is unchanged.

The final full-regression selection passed **4,667 tests, with 24 skips and
zero failures or errors, in 404.46 seconds** (4,691 collected cases). The focused
selection is contained in this suite; **do not add these counts**. The
[local validation manifest](artifacts/agent-episode-local-validation-2026-09-20.json)
records exact source hashes, runtime, JUnit identities, static checks and package
smoke, with SHA-256
`85ae95d4d399345b9f83f99eb06de182f0ecb8a82e062f4dae72790ce806a699`.

The full-suite skips were **15 dedicated ext4 ENOSPC-volume cases, 6 unavailable
MCP-extra cases, 1 legacy online-model fixture in the offline environment,
1 unavailable CUDA case and 1 Windows ACL case**. No milestone 7 case or required
CPU case skipped.

The [independent source review](artifacts/agent-episode-review-2026-09-20.json)
accepted **9.6/10 with no blocking findings** after independently verifying both
JUnit results, unchanged source hashes and the static/package gates.
Published-source CI and trained-model qualification remain separate later
requirements.

## Evidence ledger

The [tokenizer-only preflight](artifacts/agent-episode-model-preflight-2026-09-20.json)
has SHA-256 `681c037657dc69bd77fb13d806de819c4d480bb54c452d0eebe41315329d6c25`
and retains exact requests, upstream hashes, source snapshots, runtime versions,
failed cases and separately labeled tighter-bound measurements. Its appended
final-source prior-context replay records the preceding artifact digest
`bbc6e647740176935fe6d4009ac470d7ff6a522fe5c19fd00e9716b0c91d4545` and preserves
all earlier fields and negative cases. A final verifier-contract source
confirmation then records preceding digest
`fb5eff6c4f698148df5803b01225274d50d64921719b1fc0d032fbc9a938e845`, eight source
hashes and exact replay of its 15 selected requests after the final contract
correction. The token counts and fit/failure results above are unchanged. This
is feasibility evidence, not model execution or final-source qualification.

The first explicit independent source-review iteration scored **9.2/10 and was
rejected**. Confirmed transcript-rewrite, receiver-lifetime, process-start
terminalization and reversed-status issues were repaired. The final clock/reserve
behavior and request-acknowledgement/kill controls were independently checked.
A subsequently identified feedback gap was repaired by binding the preceding
actual answer into the follow-up model request and matching verifier context;
it was previously available only to the verifier. These repairs and unchanged
source validation support the final **9.6/10 accepted source review**, with no
blocking findings. The review retains the rejected iteration and source identity.

Source publication and exact-source CI were pending at this source freeze;
[PR #118](https://github.com/mmckeen-nv/DML/pull/118) records subsequent results
with their exact source attribution. No trained-model live qualification artifact
is recorded. Historical milestone 6 reviews,
qualification manifests and retained histories are immutable and are not
rewritten by this stage.
