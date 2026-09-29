# Supported-profile freeze and admission boundary

This sixteenth serial gate closes milestone 1 at the reviewed-source gate of the
[finite production release ledger](production-remaining-work-2026-09-18.md).
The target is [the explicit `dml-receipted-local-v1` profile](supported-production-profile-v1.md).
The repository remains alpha, the profile remains candidate and
`production_ready` remains false. A frozen support target is distinct from a
qualified production release.

## Reviewed boundary

The opt-in profile fixes the admitted adapter and HTTP operations, receipt journal
formats, trusted caller scope model, configuration restrictions, runtime admission,
dependency declarations, protocol limits and retry outcomes. It keeps experimental
and legacy services outside admitted operations' correctness dependencies.

The authoritative runtime supports journal schema 2 without outbox and schema 3
or explicitly migrated schema 4 with outbox. Schema/flag mismatch rejects rather
than silently adopting an existing store. Profile startup and refresh ignore
auxiliary RAG persistence and do not copy legacy snapshots. Memory, receipt and
decision transactions retain their existing serialized formats.

The profile exposes five explicit receipted mutations, scoped context retrieval,
retention inspection, durability/profile status and shutdown. The provider profile
has an exact nine-route surface, strict explicit-tenant requests and startup-bound
service authentication. A token authorizes a trusted service caller; it does not
authorize a subset of tenants. Legacy remember/generation, DCN, resume, native KV,
semantic checkpoints and projection-backend APIs remain outside this boundary.

Configuration rejects incompatible features and embedding fallback. Model-load
and embedding-operation failures remain explicit; admission does not probe remote
provider availability. Embedding identity remains an operator assertion; dummy runner
configuration admits no generation. Runtime admission covers CPython 3.10–3.13
on Linux/macOS/Windows, separately from demonstrated test portability and future
filesystem/dependency qualification. The example is explicit and requires the
operator to replace its illustrative path/model identity.

## Evidence and acceptance

Independent final review accepted this bounded gate at **9.6/10**, with no
remaining blockers. The final independent focused run passed **441 tests**:
394 new profile cases (133 contract, 144 configuration, 45 integration, 9 API and
63 adversarial) plus 47 receipt compatibility cases, with **0 skips**, in **4.56
seconds**. The root compatibility
selection additionally passed **197 existing receipt, transaction and foundations
tests** in 9.39 seconds; these selections overlap and are not an additive total.

The final root full suite passed **3,342 tests**, with **9 skips** and **3 warnings**,
in **163.26 seconds**. Maintained Ruff and strict checks of the new surfaces passed,
as did mypy over **63 maintained source files**, Hermes hygiene and the diff check.
The [review record](artifacts/production-profile-review-2026-09-18.json) records the
accepted scope and evidence. Milestone 1 is therefore closed at the reviewed-source
gate. Exact-commit publication and CI remain **pending at this source snapshot**;
their outcomes will be recorded in PR #118. This is not a merged release or a
claim that those remote jobs have already passed.

The focused gate selection is:

```sh
python -m pytest \
  dml_core/tests/test_production_profile.py \
  dml_core/tests/test_production_profile_config.py \
  dml_core/tests/test_production_profile_integration.py \
  dml_core/tests/test_production_profile_api.py \
  dml_core/tests/test_production_profile_adversarial.py
```

CI includes those files in the existing Linux/macOS/Windows portability selection
and records a separate `production-profile.xml` JUnit result in the production
evidence artifact. The existing full-suite jobs also discover the new tests.
Each reported result must identify the tested source, runtime and selection;
configured jobs are not evidence that they have passed.

The independent review checked admission and exclusion before state changes,
strict effective configuration, authority selection and auxiliary-state isolation,
receipt replay, scope and credential handling, runtime status and compatibility of
unselected legacy behavior. Deterministic injected embedders in boundary tests do
not establish live model quality, model identity attestation or production value.

## Qualification limits and accounting

This gate adds no all-component crash atomicity, durable recovery log, power-loss
qualification, tenant ACL, exact final model-input budget, bounded retained history,
live-agent campaign or benchmark advantage. Existing SQLite WAL/FULL and
cooperating-writer assumptions remain. OS admission and successful portability
tests do not identify or certify filesystems, mounts or devices.

There are still 13 audited milestones in total. Milestones 1 and 2 are closed at
their reviewed-source gates, leaving **11 remaining: 9 first-release milestones
and 2 deferred**. The next milestone is **3: exact model-input and tokenizer budget
binding**.
That accounting change does not promote maturity. The release qualification gate
still requires exact-source passing evidence and explicit support documentation
for the entire qualified profile.
