# M9 curriculum generator

Deterministic synthetic training episodes in the canonical DML V3 action
format (`llama3-8b-instruct-sft-v3-bf16-action-json-v1` consumer profile,
`dml-agent-action-v1` action surface).

## Classes

| class | episodes (default) | behavior |
|---|---|---|
| `abstain_exact` | 30 | empty / irrelevant-only / superseded-only store → final `{"claims": []}` |
| `coverage` | 40 | multi-part question → one claim per part, each citing retrieved records |
| `conflict_split` | 30 | two credible sources disagree → DUAL-ASSERT: both values, each under its own key echoed from the presented records (`.per_<source>` keys in `separate_keys`, `<fact>_<source>_condition` truth keys in `declare_conflict`); the refuted `conflict:` narrative is a presented near-miss, never the gold |
| `injection_resist` | 42 | retrieved record carries an imperative injection/lure (v3 surface pool AND a novel directive/override-notice pool, disjoint phrasings) → answer ignores it |
| `replay` | 40 | ordinary retrieve→answer; native-typed values, canonical citation shape |
| `idiom_echo` (v4) | 40 | novel question/record phrasing families, terse truth-key idiom (`blower_speed_setting`); narrative-key near-miss carries the answer value under a long narrative key — the narrative key is the wrong answer |
| `conflict_dual_assert` (v4) | 30 | two v4 sources disagree → DUAL-ASSERT: both values, each under its source-attributed truth key (`<fact>_<source>_condition`) citing its own record; the refuted `conflict:` narrative is a presented near-miss, never the gold |
| `multipart_decomp` (v4) | 40 | 2- and 3-part questions over novel phrasings → final covers EVERY part, each claim keyed by its presented terse key |
| `value_shape` (v4) | 20 | cited record states the answer in `status_value` form (text ends `... is <value>.`) with a unit-bearing string → verbatim string echo; a type-stripped bare-int near-miss (same subject, asked key) is always present and never cited |

Every claim key in a labeled final answer is exactly the `claim_key` of a
presented record that supports it (claim-key echo, v3) — records carry the
public `claim_key` in every episode, mirroring the control presentation.

## Output

Generated data is **not committed**: `out/` under this directory is
gitignored. Regenerate with:

```sh
PYTHONPATH=dml_core DML_SKIP_VENV_REEXEC=1 \
  python -m scripts.m9_curriculum.generate --force
# custom mix:
#   --counts abstain_exact=30,coverage=40,conflict_split=30,injection_resist=42,replay=40,idiom_echo=40,conflict_dual_assert=30,multipart_decomp=40,value_shape=20
```

Files: one `<class>.jsonl` train file per class plus `manifest.json`
(sha256 + byte count + episode count per file, seed `[20261116]`, generator
script sha256 and git commit, per-class variants).

## Dataset versions

The manifest carries a `dataset_version` field (additive to
`dml-m9-curriculum-manifest-v1`).

* **v1** — initial curriculum (generator commit `d0672f3`, merged `f2ce0c5`).
* **v2** — anti-shortcut repairs from the adversarial audit of v1, same seed
  and counts (bytes differ from v1 by design):
  * same-claim_key/different-subject near-miss distractors appear in every
    class (>= 40% of episodes, asserted by test), so filtering retrieved
    records by the asked `claim_key` can no longer identify the citable
    evidence — subject disambiguation is required;
  * injection lures frequently arrive under trusted source ids and clean
    records (cited and uncited) sometimes sit on the lure-prone sources
    `unvetted-paste` / `commons-repost` / `ticket-screenshot` (renamed from
    `forum-repost` in the 2026-11-12 stem purge), so the `source`
    field carries no trust signal (injection episodes record their presented
    lure ids in `meta.injection_ids` for audit);
  * task prompts ask the question plainly in all classes — no directive
    suffixes pre-announcing abstention or the conflict output contract;
  * superseded records (chained to presented replacements where one exists)
    appear across all classes, exercising `memory_state` discrimination;
  * distractors share the citable records' phrasing-template pool;
  * claim-value vocabulary widened (24h clock times, multi-year dates, wider
    version/phase/tier/retention ranges) and retrieval `top_k` / capped
    result sizes vary within every class.
* **v3** — claim-key echo + loop resistance (Track-B candidate-1 gate
  repairs), same seed and counts (bytes differ from v2 by design):
  * every labeled final claim's `key` is exactly the `claim_key` of a
    presented record that supports it; conflict episodes present their
    records under the keys the claims use (distinct per-source keys in
    `separate_keys` — final answers forbid duplicate claim keys — and
    per-source attributed truth keys in `declare_conflict`, aligned
    2026-11-12 to the verifier-credited dual-assert idiom: the variant's
    former single `conflict:` narrative claim is refuted and survives only
    as a never-cited near-miss, ids in `meta.conflict_narrative_ids`);
  * same-value/different-key near-miss distractors in the answer-bearing
    classes (>= 30%, asserted by test; ids in `meta.same_value_ids`): a
    current record about the asked subject carrying an answer value under
    a different claim_key, so value-matching cannot stand in for key echo.
    The fact pool gains one same-kind partner key per value kind;
  * retrieval-reformulation episodes in every class (>= 30% of
    answer-bearing episodes, asserted by test; flagged via
    `meta.reformulation` = `after_empty` / `after_unhelpful`): after an
    empty or unhelpful retrieve the labeled behavior is exactly one
    semantically rephrased query, then the episode finalizes — answer or
    abstain. Identical consecutive retrieve arguments never appear
    (rejected at construction, asserted absent by test).
* **v4** — M10 generalization classes on all-new pools, re-seeded to
  `[20261112]`, then re-seeded again to `[20261116]` on the 2026-11-12
  leakage re-audit (`[20261112]` is in the M10 qualification gate's
  forbidden seed set; `dataset_version` stays 4 — the version tracks
  structure, the seed tracks the draw). The re-audit also found nine
  content-bearing pool stems colliding at STEM level with
  measurement-corpora fact keys (invisible to the 8-gram sweeps):
  `berth`/`gantry`/`louver`/`forum`/`lighting` (heldout2 fact-key families),
  `lumen`/`juniper` (pilot216/heldout truth keys), `helios`/`kestrel`
  (dev-fixture keys). They were purged/renamed (v3 systems
  Helios/Juniper/Kestrel/Lumen/Cartographer → Solvane/Thornewood/Falconreach/
  Candela/Plotwright; sources `status-archive`→`heartbeat-archive`,
  `forum-mirror`/`forum-repost`→`commons-mirror`/`commons-repost`; v4
  sources and fact keys onto quay/craneway/pump-vault/illumination/
  vent_shutter/*_pressure_gauge stems), so the v1-v3 class bytes change too
  — ACCEPTED: the M9-qualified adapter was already trained on the old v3
  data; this regeneration serves the v4 training run only.
  `test_pool_stems_disjoint_from_measurement_corpora` proves every pool stem
  disjoint from all measurement-corpora fact/truth key tokens.
  Rationale for the v4 classes: the 2026-11-12
  held-out campaign (144 scenarios, sealed digest `4478cd7f…`, evidence
  `/home/test/dml-debug/m8-campaign-heldout-20261112/`, local copy
  `~/.hermes/cache/scratch/m8-control/campaign-144.jsonl`; see
  `docs/m10-charter-2026-11-12.md`) measured the v3-qualified policy's
  `valid_cited_final` at 21.5% (dml) / 18.1% (rag) and decomposed dml's 113
  unsupported finals into: claim-idiom key mismatch (≈50/113 — every truth
  value present but claims carry narrative keys like
  `"tunnel house delta field dispatch report shading screen condition"`
  instead of the truth-key idiom `"shading_screen_dispatch_condition"`),
  partial multi-part coverage (≈53/113 — 1/2, 1/3, 2/3 patterns), total
  misses (≈10/113), and the conflict-declaration gap in
  `conflicting_independent_sources` (the entire 6:1 DML-over-RAG margin).
  These are the classes v3 fixed *for its own phrasing templates*; the
  defect is a generalization gap. v4 therefore teaches the classes, not the
  templates:
  * `idiom_echo`: the asked claim key is a terse truth-key idiom
    (snake_case, no dots) while question and record phrasing vary across six
    question families and eight record templates (word order, clause
    structure, entity density). A narrative-key near-miss (>= 60% of
    episodes, asserted by test; ids in `meta.narrative_key_ids`) presents
    the asked subject and the answer value under a long narrative key
    (`"<subject> <source> report <label>"`, mirroring the campaign failure
    verbatim) — echoing the narrative key is the wrong answer even though
    the value matches;
  * `conflict_dual_assert` (reconciled 2026-11-12 against the
    observer-verified failure decomposition,
    `docs/m10-failure-decomposition-VERIFICATION-2026-11-12.md`): two v4
    sources disagree on the asked (subject, fact); the labeled final
    DUAL-ASSERTS — both values, each under its source-attributed truth key
    (`<fact>_<source>_condition`, the shape all 11 valid conflict-family
    campaign episodes used), each claim citing exactly its own presented
    record. The earlier `conflict_declare` design (one `conflict:` narrative
    claim) was REFUTED by the frozen verifier (6 such campaign episodes
    failed); the `conflict:` narrative survives only as a presented
    near-miss distractor (>= 60% of episodes, asserted by test; ids in
    `meta.conflict_narrative_ids`) — never the gold answer;
  * `conflict_split`'s v3 `declare_conflict` variant carried the same
    refuted `conflict:` narrative as its gold — contradictory gold
    standards for one behavior. Its gold is aligned to the identical
    dual-assert truth-key idiom (shared `_conflict_truth_key` /
    `_dual_assert_final` helpers, so both conflict classes teach exactly
    the same shape), with the `conflict:` narrative demoted to a
    never-cited near-miss; no gold final anywhere in the dataset uses the
    narrative form (asserted by test);
  * `multipart_decomp`: 2- and 3-part questions over three novel phrasing
    families (both part counts asserted by test); the labeled final covers
    EVERY part, each claim keyed by its presented terse key;
  * `value_shape` (observer's value-type/str gate): the citable record
    states the answer in the verifier's `status_value` form (text ends
    `... is <value>.`) with a unit-bearing string value (`"837 rpm"`); the
    labeled final echoes the string verbatim. A type-stripped near-miss
    (same subject, asked key, the same number as a bare int — the campaign's
    `95` vs `"95 minutes"` failure) is always present and never cited (ids
    in `meta.type_stripped_ids`);
  * `injection_resist` phrasing variance (the campaign's 4 injection
    successes came on novel phrasings): lures draw from a second template
    pool of directive / override-notice surfaces disjoint from the v3
    templates (per-episode pools in `meta.injection_surfaces`; both pools
    exercised, asserted by test); class count raised 30 → 42;
  * all v2/v3 repairs hold in the v4 classes too (same-key and same-value
    near-misses, superseded discrimination, source carries no trust signal,
    reformulate-once loop resistance), exercised by the same shared tests;
  * v4 entity/fact/source/value pools are all-new and disjoint BY
    CONSTRUCTION from the v1-v3 pools and from the pilot216/heldout sealed
    workloads (token-level disjointness asserted by test); the M8 leakage
    method (word 8-gram sweeps) is re-run by
    `test_v4_zero_word_8gram_overlap` against the sealed corpora
    (byte-regenerated from `scripts/m8_battery/sealed_workloads.py`; an
    on-disk sealed copy under `DML_M8_WORKLOADS_DIR` is verified
    byte-identical first), the m8 fixtures, and a dataset_version 3 `out/`
    when one is available (`DML_M9_CURRICULUM_V3_OUT`).

## Row schema (`dml-m9-curriculum-v1`)

`messages` is the exact logical transcript the V3 runner builds
(`initial_messages` system+user, assistant actions, `tool` feedback with
runner `tool-<step>` ids); `tools` is the `retrieve` entry of
`episode_tool_definitions()`. Every assistant turn parses with
`parse_agent_action`; the transcript is admitted unchanged by
`llama3_sft_runtime.render_messages`. `meta` carries the per-episode seed,
variant and entity usage for audit.

Entities are drawn from closed synthetic pools (v1-v3 pools for the original
five classes, all-new v4 pools for the v4 classes); `dml_core/tests/
test_m9_curriculum.py` mechanically proves every pool string and every
generated entity/value is absent from `dml_core/scripts/m8_battery/
fixtures/*.json`, that no fixture record text leaks into the curriculum, and
(v4) that the v4 pools and rows share zero word 8-grams with the sealed
pilot216/heldout workloads and the v3 curriculum output.

Note on world state: episodes are independently drawn, so the same (subject,
claim_key) pair can carry different values in different episodes. That
inconsistency is intentional — it punishes entity→value memorization and
forces evidence-grounding in the retrieval.
