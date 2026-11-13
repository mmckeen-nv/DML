# M10 dependent_lookups_h2 non-transfer root cause (v5 campaign, 2026-11-13)

Status: analysis complete. Evidence: campaign
`/home/test/dml-debug/m10-campaign-v5-20261113/campaign.jsonl` on
test@192.168.50.133 (local copy
`~/.hermes/cache/scratch/m8-control/campaign-v5.jsonl`, 576 episodes);
scenarios regenerated locally at HEAD c0bcd33 into
`~/.hermes/cache/scratch/m8-workloads/heldout2`; curriculum regenerated with
`--force` at `dml_core/scripts/m9_curriculum/out/dependent_lookup_chain.jsonl`
(40 eps, sha256 prefix 87b1577709861b77).

## Verdict

**(b) behavior-attempted-but-misexecuted.** The chain itself transferred
almost perfectly. What failed is the final-answer claim *shape*: unit-bearing
values, terse rule values, and verbatim stored claim keys. Retrieval of hop-2
is not the bottleneck (3/48 edge cases aside). The two arms fail identically
(rag_memory: 48/48 chain attempted, 48/48 unsupported_final), so the cause is
policy behavior, not the memory backend.

## Headline counts (dml_memory arm, 48 dependent_lookups_h2 episodes)

| Check | Count |
|---|---|
| Chain attempted (≥2 retrieve calls) | **47/48** |
| …second query differs from first | 47/47 |
| …later query contains the discovered (corrected) hop-1 token | **47/47** |
| Hop-2 table record presented | 45/48 |
| Hop-2 table record cited in final | 45/48 |
| Corrected register record presented / cited | 48/48 / 44/48 |
| Register (mid) truth entry answered correctly | 44/48 |
| Rule record presented / cited | 45/48 / 33/48 |
| Distractor (pinned/injected) record cited | **0/48** |
| Episodes with all 3 truth entries correct | **0/48** |
| Verdicts | 45 unsupported_final, 2 empty_claims_final, 1 step_limit |

Truth-entry totals (144 = 48×3): 45 passed, 99 failed (55 value_mismatch,
44 key_missing).

### Per-episode failure classification

| Count | Failure mode |
|---|---|
| 25 | hop value unit-stripped + rule value emitted as full sentence |
| 15 | wrong claim keys for hop value and rule (query-derived keys) |
| 3 | hop value unit-stripped + rule key wrong |
| 3 | empty claims (incl. the step-limit episode) |
| 1 | register key wrong + rule sentence |
| 1 | hop key wrong + rule sentence |

## Verbatim evidence

### The chain fires and hop-2 is found — the final shape fails

`heldout2_dependent_lookups_0003` (dml arm):

- Query surface: "Give the mess stipend in credits that applies to Sorcha
  Malloy at present, state the duty rotation that currently governs Sorcha
  Malloy, and name the galley rule behind it."
- step 0 retrieve: `{"query": "Sorcha Malloy mess stipend", "top_k": 10}`
- step 1 retrieve: `{"query": "graveyard rotation mess stipend", "top_k": 10}`
  — keyed on "graveyard", the corrected hop-1 token that exists only in the
  hop-1 register/hop-2 table records. Discovered-token test: PASS.
- final:
  `{"claims": [{"evidence_ids": [5], "key": "stipend_graveyard", "value": 13},
  {"evidence_ids": [4], "key": "duty_rotation", "value": "graveyard"},
  {"evidence_ids": [0], "key": "mess_stipend_rule", "value": "The graveyard
  rotation mess stipend is 13 credits."}]}`
- truth: `stipend_graveyard = "13 credits"`, `duty_rotation = "graveyard"`,
  `mess_stipend_rule = "duty rotation"` (all verifier `status_value`).
- Verdict `unsupported_final`: hop value `13` ≠ `"13 credits"` (unit
  stripped); rule value is the hop-2 sentence, not the terse dimension
  `"duty rotation"`. Evidence ids [5,4,0] are exactly the hop-2 table record,
  the corrected register, and the rule record — retrieval and citation are
  right.

Recurring value mismatches (want → got):

- `stipend_graveyard`: `"13 credits"` → `"13"`; `"29 credits"` → `"29"`
- `allotment_marl` / `allotment_brash`: `"39 kiloliters"` → `"39"`
- `reorder_point_ivory`: `"81 units"` → `"81"`
- `mess_stipend_rule`: `"duty rotation"` → `"The graveyard rotation mess
  stipend is 13 credits."`
- `irrigation_rule`: `"soil type"` → `"a plot's weekly irrigation allotment
  equals the allotment of the soil type beneath it"`
- `reorder_point_rule`: `"storage band"` → `"a part's reorder point equals the
  reorder point of the storage band it belongs to; the governing lookup
  dimension is storage band"`

### Query-derived wrong keys (15 episodes)

The model invented claim keys from the query's surface nouns instead of
copying the records' stored `claim_key`:

- 11 eps emitted `("mess_stipend", "duty_rotation", "galley_rule")` —
  `"galley_rule"` comes straight from "name the **galley rule** behind it";
  truth keys are `stipend_<token>` / `duty_rotation` / `mess_stipend_rule`.
- 4 eps emitted `("reorder_point", "storage_band", "restocking_rule")` —
  from "give the **restocking rule** behind it"; truth keys are
  `reorder_point_<token>` / `storage_band` / `reorder_point_rule`.

Example `heldout2_dependent_lookups_0018`: queries `auger flight two` →
`band teal reorder point`; final keys `reorder_point` (value `69`, also
unit-stripped vs `"69 units"`), `storage_band` = `"teal"` ✓, `restocking_rule`
= the rule sentence. Truth keys `reorder_point_teal` / `storage_band` /
`reorder_point_rule`.

### The single-retrieve episode (0115) — chain not attempted, shape still fails

Query surface: "State the reorder point in units that governs sprocket hub 12
at present, …". One retrieve `{"query": "sprocket hub 12", "top_k": 10}`
surfaced the whole 6-record stream (corrected register + hop-2 included), so
the chain was unnecessary; the final still failed: `reorder_point_carmine =
27` (vs `"27 units"`) and key `restocking_rule` (vs `reorder_point_rule`).
This episode is evidence the failure is shape, not chaining.

### The step-limit episode (0076) — bare-token query loop

Query surface: "State the reorder point in units that governs feed roller c9
at present, name the storage band …, and give the restocking rule behind it."
Retrieve queries, verbatim and in order:

1. `feed roller c9` → presented corrected register (ev idx 4) + rule (idx 0)
2. `viridian`
3. `viridian`
4. `viridian`
5. `viridian`
6. `viridian` → step_limit, `answer: null`, presented stays `[4, 0]`

The bare token `viridian` (no "reorder point" context) never surfaced ev idx 5
("Band viridian reorder point is 27 units."), and the model repeated the
identical query 5× despite the system prompt's explicit "Repeating an
identical retrieval against unchanged memory adds no evidence; revise the
query or limit" instruction. The same bare-token second query caused the two
completed-but-empty episodes (`0069`: `marl`; `0117`: `chalk`) — these are the
only 3 episodes where hop-2 was never presented, i.e. the entire
retrieval-cant-find-hop2 contribution is 3/48, all self-inflicted by
query keying.

## Why the v5 curriculum didn't transfer the shape

Side-by-side, curriculum (5 examples, `dependent_lookup_chain.jsonl`) vs
heldout2:

Curriculum task text (all 40 eps share the template):

> "For Zelpher Cordage Tidemill Basin, give the dockage toll in shillings that
> applies at present, state the hull grade that currently governs Zelpher
> Cordage Tidemill Basin, and name the dockage toll precept behind it. **Key
> each answer as its stored claim key with the stored value verbatim and one
> citation per claim.**"

Heldout2 query surface:

> "Give the mess stipend in credits that applies to Sorcha Malloy at present,
> state the duty rotation that currently governs Sorcha Malloy, and name the
> galley rule behind it."

Three gaps, in impact order:

1. **claim_value crutch (primary).** Curriculum tool results present each
   record with explicit `claim_key` **and** `claim_value` fields
   (`{"claim_key": "toll_wolfram", "claim_value": "38 shillings", ...}`), and
   the demonstrated final copies both verbatim. Campaign records
   (`runner_transport_native._ingest_record`) present only `meta.claim_key`
   plus raw `text` — there is **no claim_value field**. The unit-bearing value
   and the terse rule value must be *composed* from sentence text
   ("The graveyard rotation mess stipend is 13 credits." → `"13 credits"`;
   rule sentence → `"duty rotation"`). The curriculum never taught composition;
   it taught copying. Left to compose, the policy emits the bare JSON number
   (28/48 unit-stripped) and echoes the rule sentence (27/48).
2. **Keying instruction absent, surface nouns misleading.** Every curriculum
   task ends with the explicit "Key each answer as its stored claim key…"
   instruction; heldout2 queries carry no keying instruction and name the rule
   with a different noun than the stored key ("galley rule" →
   `mess_stipend_rule`; "restocking rule" → `reorder_point_rule`). 15/48 eps
   took the query-noun bait (`galley_rule`, `restocking_rule`) despite the
   correct `claim_key` being visible in presented record meta.
3. **Rule-name template mismatch.** Curriculum rule keys are uniformly
   `<metric>_precept` and the query says "the <metric> precept" — key
   guessable from the query. Heldout2 says "galley rule"/"restocking
   rule"/"irrigation rule" while keys are `<metric>_rule` /
   `mess_stipend_rule`. The one curriculum phrasing pattern did not cover the
   heldout2 surface.

Not the cause: retrieval quality (hop-2 presented 45/48, distractor cited
0/48), chain logic (47/48 with discovered-token re-query), register readout
(44/48 correct).

## v6 curriculum target spec

Keep `dependent_lookup_chain`'s chain structure — it transferred. Replace the
answer-shape scaffolding:

1. **Drop `claim_value` from presented curriculum records** (or add a variant
   subset without it) so the demonstrated final *composes* values from raw
   record text: number + unit token (`"13 credits"`, `"39 kiloliters"`,
   `"81 units"`), never the bare number, even though the schema allows JSON
   numbers and the record sentence contains one.
2. **Teach terse rule-value extraction**: rule claim value is the lookup
   dimension noun phrase (`"duty rotation"`, `"soil type"`, `"storage band"`),
   demonstrated against rule records whose text is a full sentence, including
   the "; the governing lookup dimension is X" suffix variant.
3. **Remove the "Key each answer as its stored claim key…" sentence from half
   or more of the tasks**, and add rule-naming decoys: query says "galley
   rule"/"restocking rule"/"irrigation rule" while stored keys are
   `<metric>_rule`; demonstrated finals must copy the record meta `claim_key`
   verbatim, never the query noun. Keep some keyed-instruction episodes so the
   behavior also holds when asked.
4. **Vary the first-query/second-query demonstrations** to include a
   revise-on-miss example: second query keyed as
   `"<dimension> <token> <metric>"` (e.g. "band viridian reorder point"),
   never the bare token; include one episode where a bare-token query returns
   nothing new and the corrected behavior is reformulation, not repetition.
5. **Subject diversity**: mix person names, plot names, and part IDs
   ("sprocket hub 12", "feed roller c9") as subjects, matching heldout2's
   surface, so the chain trigger is not conditioned on org-style multiword
   names.
6. Success criterion for v6: on a dependent_lookups_h2-shaped sealed probe,
   all three truth entries exact-matched (parameterized key + unit-bearing
   value + terse rule value) with the chain behavior preserved and zero
   distractor citations.

## Open gates / non-claims

- This is a diagnosis plus a curriculum proposal; no v6 curriculum has been
  generated or approved. Pending owner review.
- Counts are from strict exact-key/exact-value replay of episode finals
  against regenerated heldout2 scenario truth at HEAD c0bcd33; no verifier
  leniency was applied or is proposed.
