# M9 curriculum generator

Deterministic synthetic training episodes in the canonical DML V3 action
format (`llama3-8b-instruct-sft-v3-bf16-action-json-v1` consumer profile,
`dml-agent-action-v1` action surface).

## Classes

| class | episodes (default) | behavior |
|---|---|---|
| `abstain_exact` | 30 | empty / irrelevant-only / superseded-only store → final `{"claims": []}` |
| `coverage` | 40 | multi-part question → one claim per part, each citing retrieved records |
| `conflict_split` | 30 | two credible sources disagree → per-source keys echoed from the presented records, or one `conflict:` claim under the shared presented key |
| `injection_resist` | 30 | retrieved record carries an imperative injection/lure → answer ignores it |
| `replay` | 40 | ordinary retrieve→answer; native-typed values, canonical citation shape |
| `idiom_echo` (v4) | 40 | novel question/record phrasing families, terse truth-key idiom (`blower_speed_setting`); narrative-key near-miss carries the answer value under a long narrative key — the narrative key is the wrong answer |
| `conflict_declare` (v4) | 30 | two v4 sources disagree under novel two-source templates → one `conflict:` claim naming and citing both sources, neither value asserted as fact |
| `multipart_decomp` (v4) | 40 | 2- and 3-part questions over novel phrasings → final covers EVERY part, each claim keyed by its presented terse key |

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
#   --counts abstain_exact=30,coverage=40,conflict_split=30,injection_resist=30,replay=40,idiom_echo=40,conflict_declare=30,multipart_decomp=40
```

Files: one `<class>.jsonl` train file per class plus `manifest.json`
(sha256 + byte count + episode count per file, seed `[20261112]`, generator
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
    `unvetted-paste` / `forum-repost` / `ticket-screenshot`, so the `source`
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
    `separate_keys` — final answers forbid duplicate claim keys — and the
    shared base key in `declare_conflict`);
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
  `[20261112]` (bytes differ from v3 by design). Rationale: the 2026-11-12
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
  * `conflict_declare`: two v4 sources disagree on the asked
    (subject, key) under the novel two-source templates; the labeled final
    DECLARES the conflict — one `conflict:` claim under the shared presented
    terse key naming and citing both sources, neither value asserted as
    fact;
  * `multipart_decomp`: 2- and 3-part questions over three novel phrasing
    families (both part counts asserted by test); the labeled final covers
    EVERY part, each claim keyed by its presented terse key;
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
