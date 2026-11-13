# M10 curriculum v6 (dataset_version 6): dependent_lookup_compose — the non-transfer repair

Date: 2026-11-13. Worktree: DML-m10-curriculum (branch m10-curriculum-v4,
merged to codex/remote-vllm-candidate tip 5407f94 which includes the pinned
HEAD c0bcd33). Generator: `dml_core/scripts/m9_curriculum/generate.py`;
tests: `dml_core/tests/test_m9_curriculum.py`.

Grounding: `docs/m10-dependent-lookups-nontransfer-2026-11-13.md` (the v5
chain transferred — 47/48 campaigns episodes ran the two-hop retrieval with
the discovered token in the second query — but claim COMPOSITION did not:
28/48 unit-stripped hop values, 27/48 rule-sentence values, 15/48
query-baited keys), `docs/m10-v41-failure-decomposition-2026-11-12.md`,
`docs/m10-dependent-lookups-investigation-2026-11-12.md`.

## What v6 changes

ADDITIVE only. Seed unchanged at `[20261116]` (absent from the gate's
forbidden seed set). Every v3/v4/v5 builder and pool is untouched: the
thirteen v3-v5 class files regenerate byte-identical, pinned sha256-by-sha256
(`test_v4_class_bytes_unchanged`, `test_v5_class_bytes_unchanged`; v3/v4 pins
from dataset_version 4, v5 pins taken from the v5 release out/ and re-verified
at v6 landing).

One new class, `dependent_lookup_compose` (40 episodes), on all-new v6 pools:

1. **No claim_value crutch (spec a).** Every presented record carries
   `claim_key` and NEVER a `claim_value` field — the campaign presentation
   (`runner_transport_native._ingest_record` ingests `meta.claim_key` + raw
   text only). The gold final COMPOSES each value from the cited record's
   raw text: hop-2 = the unit-bearing tail (`... is 41 florins.` /
   `... stands at 41 florins.` → `"41 florins"`), never the bare number.
2. **Terse rule value from a status_value tail (spec b).** Rule records end
   `... the governing lookup dimension is <dimension>.`; the gold rule claim
   value is the 2-word tail (`"signal rota"`), never the rule sentence.
3. **Keying instruction omitted ≥ half + misleading surface nouns (spec c).**
   27/40 episodes are uninstructed (floor 50%, asserted); every query names
   the rule by a misleading surface noun ("name the parish bylaw behind
   it"). Query-baited near-miss records are always presented and never
   cited: a rule restatement under the surface-snake key (`parish_bylaw`)
   and the CORRECT value text under the measure-snake key (`fen_tithe`) —
   the campaign's `galley_rule` / `mess_stipend` inventions. Gold keys stay
   the records' stored `claim_key` (parameterized hop-2 key included).
   Keyed-instruction episodes (13/40) keep the behavior when asked.
4. **Revise-on-miss demos (spec d).** Three deterministic episodes
   (`meta.bare_token_miss`, indices 3/11/27) demonstrate the campaign's
   `viridian` loop and its correction: subject query → bare-token query
   returns NOTHING → ONE reformulated query `<dimension> <token> <measure>`
   → success. Bare-token queries are confined to the failing position of
   these demos (asserted); identical consecutive retrieves remain impossible
   by construction.
5. **Subject diversity (spec e).** Four variants span person names
   (fen_tithe, peal_wage), part ids (governor_preload; "gudgeon pin 84")
   and plot names (sprinkler_quota; "cress rise") — heldout2's three
   subject types.

**value_shape_chain decision (scope item 2): UNCHANGED.** The 25-episode
v5-campaign failure (correct number, stripped units, rule-sentence value) is
a composition failure under multi-claim load; value_shape_chain's 16 episodes
already pin the single-claim unit-strip near-miss (corrected unit-bearing
echo; bare int of the corrected value always present, never cited; all four
unit kinds exercised). Extending it would break the v5 byte pins; the missing
composition ingredient is covered additively by dependent_lookup_compose.
Documented in the generator docstring and README "Dataset versions".

## Leakage discipline

Unchanged and re-proven: v6 pools token-disjoint from the v1-v5 pools, the
12 dev fixtures, and pilot216/heldout/heldout2 — including heldout2's FIXED
dependent_lookups templates. The on-disk heldout2 seal
(`~/.hermes/cache/scratch/m8-workloads/heldout2`) was byte-verified against
the in-repo generator before sweeping (144/144 scenario bytes identical,
set_digest `15ed5697cc696b8b4fcf7c52e2849a14c5092385e6546ca1a2da8e264411af61`).
Word 8-gram sweeps: v6 rows vs sealed corpora, fixtures, the v1-v5 rows of
the same generated dataset (always-live leg), and a dataset_version 3/4/5
out/ when present — all zero overlap. Stem-level:
`test_pool_stems_disjoint_from_measurement_corpora` sweeps the v6 entity
first-token stems, v6 source-id tokens and `V6_KEY_STEMS` against all
measurement-corpora fact/truth key tokens (the bare stem `rule` is a
heldout2 key token, so v6 rule keys take bylaw/ordinance/canon/edict
suffixes).

## Dataset stats (dataset_version 6, seed [20261116])

14 classes, 437 episodes. Class shas (sha256, manifest-pinned):

| class | eps | sha256 (16) |
|---|---:|---|
| abstain_exact | 30 | 1c643cfa9338187a |
| coverage | 40 | d26503376f73e6a5 |
| conflict_split | 30 | 98dd921e8a9d8e94 |
| injection_resist | 42 | f4a5dc115ef74408 |
| replay | 40 | 2eaafe4468090be5 |
| idiom_echo | 40 | e3b60249ff84fa3d |
| conflict_dual_assert | 30 | 7c48ba20a84207de |
| multipart_decomp | 40 | ad0aa371f3280b9b |
| value_shape | 20 | b5f0d1ce677e5eef |
| dependent_lookup_chain | 40 | 87b1577709861b77 |
| value_shape_chain | 16 | 8c349908273f59e5 |
| injection_resist_qn | 14 | 552ee04698424dfd |
| answer_supported | 15 | 95b709cc5be281c4 |
| **dependent_lookup_compose** | **40** | **10108c493c2a37a6** |

Full new-class sha256:
`10108c493c2a37a6` prefix; complete digest in
`dml_core/scripts/m9_curriculum/out/manifest.json`.

Byte-identical regeneration verified at full size (two independent
`--force` runs, all 14 files + per-file shas equal).

## Validation

- `pytest dml_core/tests/test_m9_curriculum.py -q`: **78 passed, 0 failed**
  (59 baseline + 19 new v6 tests), JUnit-verified.
- `ruff check dml_core/scripts/m9_curriculum/generate.py
  dml_core/tests/test_m9_curriculum.py dml_core/scripts/m9_curriculum/README.md
  (md skipped)`: clean.
- Environment: ~/Library/Caches/dml-m8-venv (python 3.12), pinned SQLite via
  DYLD_LIBRARY_PATH=~/Library/Caches/dml-test-sqlite, PYTHONPATH=dml_core,
  DML_SKIP_VENV_REEXEC=1.

## Open gates / non-claims

- This is curriculum data + generator/test changes only; no training, no
  campaign, no gate claim. The v6 success criterion (sealed
  dependent_lookups_h2-shaped probe: all three truth entries exact-matched
  with the chain preserved and zero distractor citations) is from the
  non-transfer doc and remains to be measured by a future chartered
  training/gate tranche.
- Not pushed; branch m10-curriculum-v4 local commit only.
