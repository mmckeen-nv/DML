# M9 Track-B specification — predeclared, November 6, 2026

**Status: proposed specification, predeclared before any Track-B training. This is the [M9 charter](m9-charter-2026-11-06.md) §2 Track-B contingency, activated by two retained Track-A gate failures ([candidate 1](m9-gate-candidate1-2026-11-06.md), [candidate 2](m9-gate-candidate2-2026-11-06.md)). It carries forward the charter's discipline unchanged: one attempt per candidate, fail-closed on every check, every result retained either way, nothing tunes to pass.** All contracts remain proposals pending independent review.

## 1. Why Track B — the activated contingency

Two Track-A adapter candidates failed the frozen predeclared gate, and the failures are stable on the same two condition classes:

- **Injection-following: 1 → 2** (charter §4 condition 4). Worsened on the second candidate, which trained on the audit-repaired curriculum v2 (lures on trusted sources, no directive suffixes).
- **Valid cited finals: 6 → 5** (condition 1). Below the ≥ 8/36 threshold on both candidates despite curriculum v2 adding same-key near-miss distractors at 57–73%.

The other four conditions passed both times (conditions 2, 3, 6 on both candidates; condition 5 fixed by v2 on candidate 2). Per the candidate-2 reading: the two unmoved classes are the instruction-following floor of the pinned Llama-3-8B base itself — the fixture's injection is a stated-fact lure riding a legitimate-looking record, distinguishable only by authority/ordering cues this base cannot read, and coverage requires disambiguation the base cannot sustain across six simultaneous behavioral constraints at this data scale. A third Track-A candidate would be iterating against the predeclared gate — gate-iteration, tuning-to-pass, which this milestone exists to refuse. Charter §2's contingency therefore activates: evaluate a newer instruct base as Track B, under this predeclared specification, reviewed before any training starts. Track-B training is not authorized by the two gate records; it is authorized only after this spec passes review.

## 2. Base identity — primary and fallback

**Primary: `Qwen/Qwen3.5-9B`.** License Apache 2.0, ungated. Verified preconditions:

- **Config architecture `Qwen3_5ForConditionalGeneration` is present in the pinned container image `vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d` registry — verified 2026-11-06. No container pin bump is authorized or needed; the existing pin stands.**
- Scout analysis (Track-B selection rationale, capability comparison, ~8–9 engineer-day consumer-profile cost estimate) is retained in session scratch and is the input to this spec.

**Fallback: `Qwen/Qwen3-8B`.** Used if and only if a primary-base verification in §7 fails (e.g. a runtime incompatibility the §7 checklist surfaces). The fallback runs the identical §7 checklist against the same pinned container before any use; a failed fallback check is a hard stop returned to the owner, not a license to reach for a third base.

The base pin (model revision, sha256 of resolved weights/snapshot) is recorded in the Track-B snapshot manifest before training, in the same style as the llama3-sft-v3 pin record.

## 3. What transfers unchanged from Track A

- **Curriculum v2 dataset.** Content is base-agnostic; only the chat-template rendering differs. A `qwen` renderer replaces the `llama3` renderer; the JSONL logical content (examples, truth keys, entity inventory, split assignments) is byte-identical to the v2 manifest's. The renderer is a pinned, hashable artifact; the rendered output is diffed against the llama3-rendered v2 at the logical level to prove only the surface changed. The V3 SFT data files are referenced by their existing hashes — but see below: their rendering also moves to the qwen template, with the same logical-identity check.
- **V3 recipe constants, where applicable.** Epochs carry over unchanged (single fixed value, pinned in config). Learning rate and batch geometry are **re-derived for the new base before training, by a predeclared procedure, not by gate feedback**: lr is selected from a fixed, predeclared small grid (pinned in the config artifact before any run) by validation loss on a held-out curriculum split; effective global batch is held at the V3 value, with per-device batch and gradient-accumulation re-derived once to fit the development GPU's memory. This selection happens entirely inside training preparation — it never touches the control fixtures and is not repeated after a gate run.
- **Frozen eval fixtures.** The 12 repaired battery fixtures × 3 controls, hygiene-gated, unchanged.
- **The strict verifier.** Unchanged, including the word-bounded `record_text_contains` tightening. M9-Track-B, like Track A, does not touch it.
- **The six gate conditions from charter §4.** Identical thresholds, identical frozen control diagnostic (36 episodes, v4 amendment surface, repaired fixtures, x86 development runtime, seed 0). Note for implementers: the gate **harness** requires a new qwen3 consumer profile (§4) to run the candidate — the gate **itself** (episodes, thresholds, verifier, pass rule) does not change.

## 4. What is new and must be predeclared

- **The qwen3 consumer profile family.** A new module pair modeled one-for-one on `llama3_sft_runtime.py` + `llama3_sft_v3_action_input.py`, predeclared with the same pinned contents:
  - chat-template digest (sha256 of the exact template string/config used),
  - tokenizer hash,
  - control-token and stop ids (including the think-block delimiters, per §6),
  - vocab and context-window constants,
  - manifest schema fields (same shape as the v3 manifest, extended only with fields the qwen template requires; extensions enumerated in the profile contract),
  - training provenance (recipe id, dataset manifest hashes, renderer hash).
- **Snapshot + manifest layout for Qwen3.5-9B.** Mirrors the llama3-sft-v3 snapshot layout: base revision pin, tokenizer/template pin, adapter lineage field, runtime identity field. Produced and sha256-pinned before training.
- **New profile identity string.** Proposed: `qwen3.5-9b-sft-v1-bf16-action-json-v1`, following the existing naming convention (`<base>-sft-<version>-bf16-action-json-v1`). The identity is created by this profile work and moves only via the lock procedure in §5.

## 5. Policies carried over verbatim

- **Leakage policy.** Charter §3 applies unchanged: mechanical entity-disjointness check of curriculum v2's existing entity inventory against the fixture inventory, before training, output retained; a non-empty overlap is a hard stop. The inventory is unchanged from Track A (same JSONL content), so the existing check artifact is re-run and re-retained under the Track-B evidence dir.
- **One attempt per candidate.** A Track-B candidate is defined by its data manifest hash + training config hash, exactly as in charter §4. **The two consumed Track-A attempts do not count against Track B** — a new base is a new candidate space; but within Track B, the rule is the same: no re-run of a candidate, no gate-side adjustment after a run, pass or fail retained in full.
- **Lock procedure.** Charter §6 applies, adapted to the new profile: adapter sha256 → new consumer manifest binding the Qwen3.5-9B base pin + adapter hash + unchanged verifier/fixture pins → consumer pin update (`qwen3.5-9b-sft-v1-bf16-action-json-v1` → new manifest) only via a reviewed contract commit. Evidence dirs follow the same convention under `m9b-train-<candidate-id>-YYYYMMDD/` and `m9b-gate-<candidate-id>-YYYYMMDD/`. A failing candidate executes the retention steps with `qualified: false` and the pin does not move.

## 6. Explicit risks, declared in advance

- **Template/think-block semantics.** Qwen3.5 emits think blocks (`<think>…</think>`) in its chat template. The profile must pin think-block handling explicitly — the predeclared policy is: think blocks disabled at render time if the template supports it; if not disableable, the grammar constrains output to the action surface so think text can never reach the action channel, and the verifier sees only the grammar-valid action. Whichever mechanism is used is pinned in the profile contract before training; a template whose think behavior cannot be pinned is itself a §7 verification failure.
- **Grammar/xgrammar compatibility.** The DML action surface (`dml-agent-action-v1`, `tool`/`final` branches) is currently compiled by xgrammar under the llama3 tokenizer. Compatibility with the Qwen tokenizer/vocab in the pinned container is unverified until §7 runs; structural differences (token boundaries in JSON scaffolding, control-token ids) could require a grammar rebuild — permitted, predeclared, and pinned; a grammar change is a profile change, never a gate change.
- **Track B may also fail the gate.** That outcome is fully anticipated: the failure is retained as evidence with the same rigor (manifest, config, loss curves, diagnostic output, `qualified: false`). **There is no Track C.** A second retained failure returns the decision to the owner; nothing in this spec authorizes reaching for another base.

## 7. Verification-before-training checklist (fail-closed; any failure is a hard stop)

1. **Model config architecture in the pinned container registry** — `Qwen3_5ForConditionalGeneration` confirmed present in `vllm/vllm-openai@sha256:04563c…` — **DONE 2026-11-06** (scout verification, retained in session scratch).
2. **Tokenizer/template smoke in the container** — load the Qwen3.5-9B tokenizer and chat template inside the pinned container; render one curriculum example and one control-shaped prompt; record template digest and tokenizer hash; confirm stop/control ids match the profile contract.
3. **xgrammar DML-grammar load in the container** — compile the `dml-agent-action-v1` grammar against the Qwen tokenizer in the pinned container; run a decode smoke (one retrieve action, one final action) proving the grammar accepts both branches and rejects malformed output.
4. **Profile unit tests green** — the new module pair passes the same unit-test battery the llama3 profile passes (renderer identity, manifest schema, stop-id handling, grammar smoke), run in CI before any training starts.

Training does not begin until items 2–4 are green and this spec has passed independent review. The checklist output is retained as part of the Track-B evidence record.

## Pointers

- Charter (gate, leakage, lock procedure, Track-B contingency): `docs/m9-charter-2026-11-06.md`
- Retained gate failures (the trigger): `docs/m9-gate-candidate1-2026-11-06.md`, `docs/m9-gate-candidate2-2026-11-06.md`
- V3 snapshot / base pin record (layout this spec mirrors): `docs/llama3-sft-v3-m7-outcome-2026-09-29.md` + `docs/artifacts/llama3-sft-v3-m7-outcome-2026-09-29.json`
