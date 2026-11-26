# M7 recovery handoff

## Recovery after the execution-environment disconnect — 2026-09-22

M7 remains **IN PROGRESS**. Public Coder-recovery source `1c4ccd5` passed
[CI 331](https://github.com/mmckeen-nv/DML/actions/runs/35697661942), all 20 jobs.
The current workspace can access the public repository and the separate reviewed
retrieval-metadata checkout, but the previous main workspace is absent.

Before the disconnect, attempt 6 was observed to finish all nine tasks with 8/9
verifier successes, nine measurable finals across eight intents and 15,534 known
tokens. Both sequential replays matched and the independent completion review
rejected M7 solely because the model never performed the required supersession.
Those are retained observations in the [PR recovery handoff](https://github.com/mmckeen-nv/DML/pull/118),
not a replacement for raw evidence. The campaign, both replay artifacts, trained
provenance/admission artifacts, integrated validation receipt and unpublished
27-file commit are currently unavailable. Do not reconstruct them from prose,
claim their bytes remain verified locally, or reclassify attempt 6 as qualifying.
Attempt 5 remains separately interrupted/unreplayable with unknown full costs.

The unavailable local commit was `97df9a569baf0c4c1a81370f91094a13b777946f`, tree
`97d7aed92e0cea92e467404bcf173069a58039ba`. This recovery does not recreate or claim
that exact tree. The surviving three-file metadata change and its original
[independent 9.6/10 review](artifacts/agent-episode-retrieval-metadata-review-2026-09-22.json)
remain byte-matched. Its 234-case independent JUnit also survives with zero skips,
failures or errors. A fresh clone differs from public source only in those three
reviewed source files before recovery documentation is added.

The change exposes only requested retrieval cap, returned count and equality to
the cap. It asserts no hidden-record total or completeness guarantee and does not
force model actions, add reads, change task truth or relax acceptance gates.
Fresh [integrated validation](artifacts/agent-episode-metadata-recovery-validation-2026-09-22.json)
passed **784 mandatory cases with zero skips/errors/failures**, all 21 modules,
both strict lint scopes and maintained mypy, with 434 unchanged source hashes.
[Independent supervisor review](artifacts/agent-episode-metadata-recovery-supervisor-review-2026-09-22.json)
accepts this recovered source for publication at **9.6/10**, with no source blockers.
Published-source CI, recreated pinned model assets, strict admission and
independent pre-generation review remain required before a new campaign can qualify. Attempt 7 has not been frozen or launched. No attempt ID or
successful result will be substituted for missing historical evidence.

Milestones 1–6 remain closed; M7 is active; M8–11 remain open; M12–13 remain
deferred. This continues serial source gate 20. PR #118 stays unmerged and DML
remains alpha with `production_ready=false`.

## Current paths

- Checkout: `/workspace/scratch/c79340404ab9/DML-core`
- Runtime being restored: `/workspace/scratch/c79340404ab9/dml-venv/bin/python`
- Recovery helpers and fresh validation: `/workspace/scratch/c79340404ab9/m7-resume`
- Surviving reviewed checkout: `/workspace/scratch/c79340404ab9/dml-retrieval-worktree`

Keep complete evidence in this checkout and publish it promptly. Reacquire model
bytes only from immutable pins; never recreate raw episode evidence from a summary.
