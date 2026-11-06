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
