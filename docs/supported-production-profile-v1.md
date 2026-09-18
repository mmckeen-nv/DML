# Supported production profile v1: frozen candidate boundary

Profile ID: **`dml-receipted-local-v1`**. Select it explicitly with
`production_profile: dml-receipted-local-v1`. This is the support target being
qualified for the first release. Its maturity is **candidate**,
`production_ready` remains **false**, and no API becomes stable through profile
selection. The [finite release ledger](production-remaining-work-2026-09-18.md)
governs graduation. The [profile hardening record](production-profile-hardening-2026-09-18.md)
records this boundary's review and evidence separately.

The profile provides receipted memory mutations and scoped context construction
on one host, using local storage and cooperating trusted callers. It deliberately
does not select every candidate feature in the repository. Existing deployments
without this opt-in retain their existing behavior; the broad
[production contract](contracts/production-v1.md) also describes those historical
and separately qualified features.

## Admitted interfaces

Only the following adapter operations form this profile's public runtime surface.
The returned receipts acknowledge historical commits, rather than current memory
liveness. A retry must preserve the complete request, full scope and key.

| Python interface on `DMLAdapter` | Contract |
| --- | --- |
| `ingest_memory_receipted` | Append one memory with its decision and scoped receipt; no merge, implicit eviction or RAG mirror. |
| `retire_memory_receipted` | Apply a scoped retirement guarded by the complete current record digest; history remains retained. |
| `supersede_memory_receipted` | Link an exact source to an exact same-scope replacement, suppressing only the source. |
| `update_memory_receipted` | Change text and vector in the same declared embedding space, preserving scope, trust, lifecycle and provenance. |
| `promote_memories_receipted` | Explicit first-level derivation from 1–32 exact source records; no recursive or automatic promotion. |
| `retrieve_context` | Construct a detached scoped context report with pinned store revision and effective retrieval inputs. |
| `inspect_memory_retention` | Inspect known structured copies at one verified authority revision without erasure or payload disclosure. |
| `durability_status` | Report process-local durability degradation; this is not a continuous journal integrity check. |
| `production_profile_status` | Report profile selection and qualification status. |
| `close` | Shut down owned runtime resources. |

There is no separate adapter receipt-lookup API in this profile. Repeating a
receipted mutation resolves its historical receipt before new model work. Direct
access to internal stores, journals, service objects or mutable adapter attributes
is outside the public profile boundary. Python callers are trusted code; the
profile is not a sandbox against callers that modify internals.

`production_profile_status()` reports `profile_id`, `status`,
`production_ready`, `validated`, `authority` (`journal_schema_version` and
`outbox_enabled`) and `embedding_identity`. A selected, admitted instance reports
`status: candidate`, `validated: true` and `production_ready: false`.
`validated` means this instance passed profile admission, not that the outstanding
release qualification passed. This detailed Python status is distinct from the
minimal public HTTP health response.

The HTTP profile exposes exactly these nine method/path pairs:

| Method | Path | Operation |
| --- | --- | --- |
| GET | `/health` | Minimal public health and maturity status; no paths or embedding identity. |
| GET | `/api/contracts` | Contract and maturity inventory. |
| POST | `/api/remember/receipt` | Receipted ingestion. |
| POST | `/api/memory/retire/receipt` | Receipted retirement. |
| POST | `/api/memory/supersede/receipt` | Receipted supersession. |
| POST | `/api/memory/update/receipt` | Receipted content update. |
| POST | `/api/memory/promote/receipt` | Receipted first-level promotion. |
| POST | `/api/memory/retention/inspect` | Scoped retention inspection. |
| POST | `/api/recall` | Scoped context retrieval. |

Other provider routes are outside the selected profile, including generation,
chat, embeddings compatibility, resume, frontend assets and all DCN routes. The
profile does not expose legacy remember/batch, model orchestration, checkpoint or
projection-backend operations. Existing offline journal migration tools remain
operator procedures, not concurrent runtime mutation APIs.

Python ingestion defaults `kind` to `memory`; HTTP ingestion defaults it to `note`.
Cross-interface retries must specify the same kind as well as the same scope,
request and key. HTTP recall accepts `query`, the four scope fields and `top_k`;
its response records the resolved effective time. The richer Python retrieval
arguments do not automatically become accepted HTTP fields.

## Authority and persistence formats

The authoritative database is **`storage_dir/dml_state.sqlite3`**.
`persistence.path` names the legacy file setting; it does not relocate this
journal. Use one absolute, dedicated storage directory and retain its identity
and coordination files. Profile initialization and refresh do not load auxiliary
legacy RAG files or copy legacy snapshots into this authority.

| Configured authority | Admitted journal format | Meaning |
| --- | --- | --- |
| `persistence.outbox: false` | Schema 2 | Memory, lifecycle changes, historical receipts and checksummed decisions commit in one SQLite transaction. |
| `persistence.outbox: true` | Schema 3 for a fresh journal; schema 4 for an explicitly migrated journal | The same authority also commits versioned outbox events; migrated history has an explicit baseline. |

`persistence.journal` and `persistence.receipts` must be true.
`persistence.enable` must be false: that switch enables the legacy JSONL path,
whereas the selected journal remains authoritative and durable. Its periodic
interval is zero. A configured format cannot silently adopt a different existing authority.
Unknown versions, mismatched identities, corruption and missing initialized
authority fail explicitly. An intact identity marker detects deletion across
restart; deleting both authority and identifying evidence is outside that
detection guarantee.

These are the existing schema-2/3/4 formats and versioned mutation receipts. This
profile adds no journal migration or new receipt serialization. See the
[ingestion](receipt-hardening-2026-09-12.md),
[retirement](retirement-hardening-2026-09-14.md),
[supersession](supersession-hardening-2026-09-14.md),
[content-update](content-update-hardening-2026-09-16.md),
[promotion](promotion-hardening-2026-09-17.md),
[outbox](outbox-hardening-2026-09-14.md) and
[migration](outbox-migration-hardening-2026-09-14.md) contracts for their exact
request, receipt and decision invariants. Historical receipts remain valid after
later lifecycle changes. Promotion sources are immutable snapshots in the proof;
subsequent source edits do not cascade into the derived memory.

The journal uses SQLite WAL and `synchronous=FULL`, along with the existing
cooperating-writer and initialization locks. Memory, receipt and decision atomicity
comes from that one authority. External projection delivery and consumers are
outside the runtime profile and cannot determine receipt success. Schema-3/4
outbox retention is still part of the authority even when no consumer is running.
No cross-file atomicity or distributed transaction is claimed.

Keep the database, `dml_state.sqlite3.identity.json`,
`dml_state.sqlite3.init.lock` and `.dml_store.lock` coordination files together.
During incident preservation also retain WAL, SHM and any incomplete migration
marker (`dml_state.sqlite3.migration.json`) and lock-owner diagnostic sidecar
(`.dml_store.lock.json`). A read-only SQLite preflight may use WAL/SHM or lock
sidecars; it is not a guarantee that opening an existing authority creates no
temporary coordination files. Stop writers for migration or cutover. Use the existing side-by-side
migration services, retain the original source, verify the destination, and
account explicitly for writes made after cutover before rollback. Populated
legacy vectors without embedding provenance cannot acquire that provenance
through a format migration. There is no in-place downgrade.

For backup, stop writers and use SQLite's backup facility to obtain a consistent
full database plus its identity sidecar. Copying only a live main database while
WAL is active is insufficient. Lattice-only JSON exports and semantic checkpoints
omit receipt authority and are outside this profile. A missing or corrupt store
requires recovery into a separate location from verified evidence; never delete
identity or migration markers to make a damaged store look new. Complete
backup/restore qualification and the tested release runbook remain milestones 4,
5 and 11.

## Scope and caller authority

Scope is the exact `tenant_id/client_id/session_id/instance_id` tuple, including
null optional members. Every supported memory request supplies an explicit,
nonempty tenant. A key is unique within that complete tuple. Wrong-scope lifecycle
targets are indistinguishable from missing targets.

HTTP profile startup requires a configured service bearer credential and captures
its authentication configuration for that app's lifetime. Changing the environment
after startup does not silently disable or replace that app's authentication;
restart deliberately to rotate credentials. Token holders are trusted operators
allowed to select scopes. This is **not per-token tenant authorization**. The
Python surface likewise assumes trusted cooperating service code. Tenant isolation
inside requests does not prevent a trusted caller from deliberately selecting a
different tenant.

Configure `DML_API_TOKEN` and/or `DML_ADMIN_TOKEN` with an ASCII RFC 6750 bearer
token: letters, digits, `-`, `.`, `_`, `~`, `+` or `/`, optionally followed by `=`
padding. Empty, whitespace, control-character and non-ASCII values reject before
adapter construction. Only `/health` is public. `/api/contracts` and all
memory routes require one of the captured credentials; the profile has no separate
administrative route class. Profile validation errors are sanitized and do not
echo rejected request inputs.

Trust metadata and the model/preprocessing declaration are operator assertions.
Retrieval never grants instruction authority to memory content. Checksums detect
accidental inconsistency; a writer able to alter all data and checksums is outside
the threat model. Hostile filesystem writers and hostile in-process code are not
admitted callers.

## Configuration and dependencies

Start with [the explicit example](examples/production-profile-v1.yaml), set its
absolute storage path, and replace its illustrative embedding identity with the
identity of the actual immutable model and preprocessing configuration you deploy.
The placeholder identity is not evidence that any model has been pinned. Profile
selection validates the effective configuration after configuration sources have
been combined; incompatible or unknown profile settings fail before authority
initialization rather than silently enabling another feature.

YAML and direct Python settings use their declared types. Profile environment
configuration accepts canonical values, including lowercase `true`/`false` for
booleans, integer/finite-number forms and `null` only for nullable settings.
Unknown profile configuration keys and incompatible effective values reject.
Documented operational environment variables, such as provider credentials, are
handled separately. Preserve the resolved configuration with qualification
evidence; merely copying this YAML does not prove environment overrides are absent.

The model runner is explicitly `model_name: dummy` with `llm_backend: dummy`.
No generation, summarization or model-provider input is admitted through this
runner. Embeddings use a native or operator-supplied backend with a declared
immutable identity; random embedders and unavailable-model random fallback are
rejected. Admission does not probe a remote provider or attest model weights.
Model-load and embedding-operation failures remain explicit; successful startup
alone does not prove future provider availability. The first commit binds that
identity and vector dimension. Later writes
and embedding-based reads reject an incompatible identity even when dimensions
match. This is an operator-declared contract, not attestation of model weights.

Disable RAG persistence/mirroring, ANN, DPM, DCN, agentic routing and automatic
promotion, STM, survival-ledger additions, quality mutation on retrieval,
background processing, periodic persistence and semantic checkpoints. Runtime
lattice ranking explicitly uses the NumPy vector backend; it cannot acquire the
optional CUDA vector backend through automatic backend selection. An embedding
provider may use operator-selected CPU or GPU execution, which remains separately
unqualified. Native-KV and fabric routing are outside the profile. No optional projection,
RAG, generation or cognition service is a correctness dependency of its admitted
memory operations.

The source of declared dependency floors is the root `pyproject.toml`:

| Dependency group | Declared direct requirements |
| --- | --- |
| Base package | `typer>=0.9`, `fastapi>=0.100`, `requests>=2.31`, `numpy>=1.24`, `PyYAML>=6.0`, `pydantic>=1.10`, `pydantic-settings>=2.0`, `prometheus_client>=0.16`, `structlog>=23.2`, `httpx>=0.26`, `pypdf>=4.0`, `python-multipart>=0.0.6` |
| HTTP serving (`server`) | Base requirements plus `uvicorn>=0.23`, `websockets>=11.0`; repeated base entries remain as declared. |
| Model-backed embeddings (`embeddings`) | `sentence-transformers>=2.2` when using the built-in SentenceTransformer backend; an external/injected backend has its own declared dependencies. |
| Build | `setuptools>=61`, `wheel`, `pybind11>=2.10`, `numpy>=1.24` |

The package permits Python `>=3.10`; this selected profile admits only CPython
3.10–3.13 on the runtime systems below. A declared floor is not evidence that every
version combination works. The provider uses the resolved Pydantic 2 interface;
the base `pydantic>=1.10` line does not qualify Pydantic 1 for this HTTP profile.
Installed transitive dependencies, SQLite, embedding implementation/model and
operating-system details must be captured with each qualification run. No lockfile
or fleet-wide dependency qualification is introduced here. Test doubles used in
focused cases establish boundary behavior, not live embedding quality.

The local implementation environment observed during this gate was CPython
3.12.14 on Linux x86_64, SQLite 3.53.1, NumPy 2.5.3, Pydantic 2.13.5, FastAPI
0.141.1, PyYAML 6.0.3 and HTTPX 0.28.1. These are observations of one environment,
not new package minima, exact installation pins or passing results for every
provider/dependency combination. Final test evidence is recorded separately in
the hardening record.

## Platform target and evidence boundaries

| Runtime target | Existing automated portability coverage | Qualification status |
| --- | --- | --- |
| Linux, CPython 3.10–3.13 | Ubuntu portability jobs at 3.10 and 3.13; full-suite jobs at 3.10 and 3.11; CPU evidence job at 3.11. | Candidate. No filesystem, mount, device or power-loss qualification follows from the runner name. |
| macOS, CPython 3.10–3.13 | Portability selection at 3.10 and 3.13. | Candidate. The selection is not full-suite or filesystem qualification. |
| Windows, CPython 3.10–3.13 | Portability selection at 3.10 and 3.13. | Candidate. The selection is not full-suite or filesystem qualification. |

Intermediate admitted Python versions are not implicitly tested on every target.
The profile checks its runtime admission boundary; admission is not a production
readiness verdict. The required storage environment is one host and a local
filesystem honoring SQLite WAL, file locking and atomic publication assumptions.
Network/shared filesystems, multi-host coordination, noncooperating writers and
unqualified mount/device combinations have no profile support claim. Runtime
platform checks do not prove the filesystem or mount is suitable.

Existing subprocess-kill, SQLite quota, corruption and caught-I/O-failure tests
are distinct evidence classes. Process-kill success does not establish physical
power-loss durability, arbitrary device failure behavior or all-component crash
atomicity. Those guarantees must be expressly qualified before release claims are
made. No cloud instance type, filesystem version, storage controller or container
volume class is certified by this gate.

## Enforced protocol limits and remaining capacity work

| Boundary | Limit or behavior |
| --- | --- |
| Scope members, idempotency key and ingestion kind | Nonempty strings; at most 256 UTF-8 bytes each. Optional scope members may be null. |
| Lifecycle reason | Nonempty; at most 1,024 UTF-8 bytes. |
| Canonical mutation request | Finite JSON; at most 1 MiB. |
| Lifecycle preconditions | Nonnegative integer memory IDs and complete-record, 64-character lowercase SHA-256 digests. |
| Explicit promotion | 1–32 distinct level-0 sources; canonical source proof and output each bounded to 1 MiB. |
| Scoped retrieval | `top_k` is an integer from 1 through 10; HTTP query text is at most 1 MiB of UTF-8. |
| Embedding identity declaration | Nonempty immutable declaration, at most 1,024 UTF-8 bytes; compatible finite vector dimension required. |
| Memory capacity | Explicit configured capacity; append and promotion reject exhaustion without eviction. Retirement retains capacity and content. |
| Context budget | Bounded rendered context under the existing estimator; it is not the final model/tokenizer/chat-template budget contract. |
| History and retained evidence | Full history is verified on ordinary operations; history length, cumulative receipt/outbox bytes and verification cost are not bounded by the per-request limits. |

No supported throughput, concurrency ceiling, maximum store age/bytes, retention
period, retrieval-quality threshold or latency/SLO follows from these protocol
limits. The existing 256-client cases are test workloads, not a guaranteed service
capacity. Growing-store 1k/10k/100k live campaigns, fair baselines and complete
decision replay remain explicit release obligations. Context is still data for
an external caller: any eventual final model input must pass milestone 3's pinned
tokenizer and complete-framing budget gate before inference.

## Failure and retry contract

An acknowledged receipt survives a successful authoritative commit even if runtime
hydration subsequently fails. Health exposes degradation; a later verified refresh
can rebuild runtime state. A disconnected or timed-out caller must treat the
outcome as uncertain until the identical request and key resolve it. Never replace
the key merely because an acknowledgement was lost.

| Result | Caller action |
| --- | --- |
| HTTP 200 receipt, including replay | Preserve the receipt as a historical commit result. Do not infer continued liveness. |
| HTTP 400 invalid/unsupported mutation | Correct the request or profile configuration; do not blindly retry unchanged input. |
| HTTP 401 authentication failure | Supply the app's configured service credential; do not retry by changing tenant. |
| HTTP 404 missing/wrong-scope memory | Verify scope and target through the authorized caller; no mutation is acknowledged. |
| HTTP 409 `idempotency_conflict` | The scoped key belongs to a different canonical request. Resolve the intent; do not overwrite its history. |
| HTTP 409 `receipt_lifecycle_conflict` | Inspect current state and make a fresh explicit decision; do not silently replace expected digests. |
| HTTP 409 `receipt_capacity_exceeded` | Capacity refused append/promotion; no implicit eviction occurs. Resolve capacity deliberately. |
| HTTP 409 `revision_conflict` | Retry the identical request, full scope and key after competing work settles. |
| HTTP 422 `profile_validation_failed` | Correct shape/types/fields. Profile requests require an explicit tenant and reject unsupported fields. |
| HTTP 503 `receipt_ownership_unavailable` | Ownership timed out; retry the identical request/key with bounded caller backoff. |
| HTTP 503 `receipt_outcome_unavailable` | Storage cannot establish a trustworthy outcome. Recover authority, then retry the identical request/key. |
| HTTP 503 `receipt_not_committed` | Reconciliation found no receipt at that check. Retry the same request/key; it is not permission to invent another intent. |
| HTTP 503 `embedding_unavailable` | Recover the declared embedding service/identity, then retry the identical request/key. |
| Corrupt/missing authority or unsupported journal version | Stop affected work and preserve evidence for recovery. Never substitute an empty store. |

Retention inspection also returns `retention_inspection_unsupported` (409) when
the retained structure cannot support its report and `retention_outcome_unavailable`
(503) when trustworthy inspection is unavailable. Inspection is read-only; retry
after resolving the cause. Error codes and transport failures do not certify
physical erasure or expose an untrusted store as empty. Direct Python calls raise
the corresponding explicit exceptions rather than HTTP responses.

Recall reports `invalid_or_unsupported_retrieval_request` (400),
`retrieval_outcome_unavailable` (503) or `embedding_unavailable` (503); it never
translates failure into a successful empty result. A read retry has no mutation
idempotency key. Unexpected mutation failures also report
`receipt_outcome_unavailable` with `retry_same_key: true`, preserving the uncertain
commit rule, including failures to serialize an acknowledgement. Profile HTTP
input errors use `profile_validation_failed`; they do
not echo submitted text, metadata or credentials.

## Release closure

Milestone 1 freezes this admitted product boundary and verifies its enforcement.
It does not close model-input budget binding, crash/filesystem qualification,
persisted-format coverage, mixed-operation concurrency, live-agent semantics,
fair baseline value, continuous workloads, replay/retention or release support.
Those are milestones 3–11. Remaining legacy orchestration extraction and native-KV
restore identity are the two deferred milestones. Physical erasure, recursive
promotion, cascading invalidation and extra deployment profiles remain excluded.

Freeze acceptance requires independent review and exact-source passing evidence;
the profile's production-readiness flag remains false afterward. Release promotion
requires the ledger's separate qualification and release process.
