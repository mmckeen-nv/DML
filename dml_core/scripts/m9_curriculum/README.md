# M9 curriculum generator

Deterministic synthetic training episodes in the canonical DML V3 action
format (`llama3-8b-instruct-sft-v3-bf16-action-json-v1` consumer profile,
`dml-agent-action-v1` action surface).

## Classes

| class | episodes (default) | behavior |
|---|---|---|
| `abstain_exact` | 30 | empty / irrelevant-only / superseded-only store → final `{"claims": []}` |
| `coverage` | 40 | multi-part question → one claim per part, each citing retrieved records |
| `conflict_split` | 30 | two credible sources disagree → per-source keys or `.conflict` claim |
| `injection_resist` | 30 | retrieved record carries an imperative injection/lure → answer ignores it |
| `replay` | 40 | ordinary retrieve→answer; native-typed values, canonical citation shape |

## Output

Generated data is **not committed**: `out/` under this directory is
gitignored. Regenerate with:

```sh
PYTHONPATH=dml_core DML_SKIP_VENV_REEXEC=1 \
  python -m scripts.m9_curriculum.generate --force
# custom mix:
#   --counts abstain_exact=30,coverage=40,conflict_split=30,injection_resist=30,replay=40
```

Files: one `<class>.jsonl` train file per class plus `manifest.json`
(sha256 + byte count + episode count per file, seed `[20261106]`, generator
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

## Row schema (`dml-m9-curriculum-v1`)

`messages` is the exact logical transcript the V3 runner builds
(`initial_messages` system+user, assistant actions, `tool` feedback with
runner `tool-<step>` ids); `tools` is the `retrieve` entry of
`episode_tool_definitions()`. Every assistant turn parses with
`parse_agent_action`; the transcript is admitted unchanged by
`llama3_sft_runtime.render_messages`. `meta` carries the per-episode seed,
variant and entity usage for audit.

Entities are drawn from a closed synthetic pool; `dml_core/tests/
test_m9_curriculum.py` mechanically proves every pool string and every
generated entity/value is absent from `dml_core/scripts/m8_battery/
fixtures/*.json`, and that no fixture record text leaks into the curriculum.

Note on world state: episodes are independently drawn, so the same (subject,
claim_key) pair can carry different values in different episodes. That
inconsistency is intentional — it punishes entity→value memorization and
forces evidence-grounding in the retrieval.
