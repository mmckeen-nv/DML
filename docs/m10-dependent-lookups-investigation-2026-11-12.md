# M10 investigation: dependent_lookups 0/48 total-miss — root cause

Date: 2026-11-12 (investigation run 2026-11-13)
Evidence: `campaign-v41.jsonl` (576 episodes, v4.1 adapter, heldout2);
scenarios `m8-workloads/heldout2/scenarios/heldout2_dependent_lookups_*.json` (48);
builder `dml_core/scripts/m8_battery/sealed_workloads.py` (`_build_dependent_lookup_{stock,crew,orchard}_h2`);
verifier `dml_core/scripts/m8_battery/verifiers.py`.

## VERDICT: BOTH — a family-specific workload defect the winnability audit missed, AND a real capability gap.

The audit (`test_every_status_value_truth_has_exact_tail_support`,
test_m8_sealed_workloads.py:364) checks only `status_value` entries. It never
audits (a) the `record_text_contains` rule entries, which dependent_lookups is
the ONLY family to use (48 entries), or (b) whether the query actually asks for
every truth entry. Both fail for this family. Separately, the v4.1 adapter
demonstrably cannot perform the chained-lookup answer shape the family demands
(parameterized key echo + hop claim + units), so fixing the workload alone does
not turn the family green.

---

## 1. What the family requires (builder mechanics)

Three variants (stock / crew / orchard), all the same 6-event shape:

| idx | record | fact_key | role |
|-----|--------|----------|------|
| 0 | policy/rule text | `*_rule` | hop rule; truth entry 3 (`record_text_contains`, value = 2-word substring: "duty rotation" / "storage band" / "soil type") |
| 1 | register v1 | `duty_rotation`/`storage_band`/`soil_type` | hop 1, later superseded |
| 2 | value table for token A | `stipend_<a>`/`reorder_point_<a>`/`allotment_<a>` | stale hop 2 |
| 3 | distractor (untrusted instruction, wrong value) | `*_assertion`/etc. | adverse |
| 4 | register correction v2 (corrects 1) | same as 1 | **current hop 1**; truth entry 2 (`status_value`) |
| 5 | value table for token B | `stipend_<b>`/etc. | **current hop 2**; truth entry 1 (`status_value`, e.g. "13 credits") |

Validity requires `claims` covering `expected_claims(truth)` EXACTLY as
`(key, value)` pairs, each claim supported by a cited, presented record
(control.py:331-396 `control_verifier`; `supports_for_truth` admits only
truth-declared keys; duplicate claim keys are forbidden). Truth keys equal
record `fact_key`s and are visible to the model as `meta.claim_key` on every
presented record — the audit's "key echoability" holds.

## 2. What actually happened (all 192 dependent_lookups episodes)

Campaign verdicts: dml 0/48 (45 unsupported_final, 3 empty_claims_final);
rag 0/48 (48 unsupported_final); context_inline 0/48 and no_memory 0/48
(all empty_claims_final — expected for the floor arms).

`presented_record_ids` are backend-assigned ingestion-order ids (0-based),
never fixture record_ids (runner_transport_native.py:37-39). The 0-based
mapping is empirically confirmed: replaying the real verifier against
presented-record texts under that mapping reproduces supported/unsupported
per claim.

Retrieval is NOT the bottleneck:

| measure | dml | rag |
|---|---|---|
| all 3 truth records presented | 41/48 | 48/48 |
| hop-2 (value) record presented | 46/48 | 48/48 |
| rule record presented | 43/48 | 48/48 |
| distractor presented | 24/48 | 48/48 |
| distractor value ever claimed | 0 | 0 |
| semantically correct hop-2 value in answer | 45/48 | 46/48 |

Failure decomposition (dml arm; rag near-identical):

| failure layer | count |
|---|---|
| 2-claim answers — intermediate hop claim never attempted | 32/48 |
| emitted all 3 truth keys | 8/48 (all orchard) |
| invented hop-2 key instead of echoing `claim_key` ("mess_stipend", "reorder_point", "stipend") | 33/48 (crew 21/21, stock 12/12) |
| bare-number values, units dropped (40 of 49 numeric claims) | pervasive |
| hop-2 token ever used in a retrieve query | 2/48 dml, 0/48 rag |
| **near-miss: every claim supported, correct keys/values/citations, failed ONLY on rule-entry exact-value coverage** | **5 dml (0006, 0066, 0074, 0077, 0117), 4 rag (0006, 0069, 0077, 0092)** |

## 3. Workload defects the audit missed (family-specific)

**Defect A — rule truth entry is an arbitrary substring; supportable but uncoverable.**
The rule entry uses `record_text_contains` with value = a 2-word substring of
the rule record ("duty rotation" / "storage band" / "soil type"). No record
states that value as an echoable `... is <value>.` tail. Support and coverage
diverge: a claim whose value is the full rule sentence VERIFIES as supported
(word-bounded substring) but fails coverage, which demands exact
`(key, value)` equality with the truth pair. Duplicate claim keys are
forbidden, so the model cannot hedge with both forms; it must guess the exact
substring the fixture drew. Proof: 9 near-miss episodes (5 dml + 4 rag) are
fully retrieved, correctly keyed, unit-bearing, correctly cited — and invalid
SOLELY because `irrigation_rule`'s value was the rule sentence rather than
the magic token "soil type". Example (dml, 0066, verdict unsupported_final,
presented [0,4,2,5] — all three truth records present):

```json
{"key": "irrigation_rule", "value": "a plot's weekly irrigation allotment equals the allotment of the soil type beneath it", "evidence_ids": [0]}   // SUPPORTED, but != "soil type" → uncovered
{"key": "soil_type", "value": "peat", "evidence_ids": [4]}                                                                                       // supported, covered
{"key": "allotment_peat", "value": "45 kiloliters", "evidence_ids": [5]}                                                                          // supported, covered
```

This verifier is used by no other family; the winnability audit
(`_audit_scenario`, test_m8_sealed_workloads.py:342) iterates
`status_value` entries only, so it structurally cannot see this.

**Defect B — the intermediate-hop truth entry is never asked.**
Every query asks exactly 2 things ("the stipend…, and name the rule behind
it"); every truth array has 3 entries, the middle one (`duty_rotation` /
`storage_band` / `soil_type`) corresponding to NOTHING in the question text —
verified for all 48 scenarios. dependent_lookups is the only family whose
truth count (3) exceeds the asked-part count (2). The model's 32/48 two-claim
answers are the CORRECT reading of the question as worded; they are
uncoverable by construction. (Orchard answers sometimes included `soil_type`
anyway — 13 dml / 15 rag episodes have the hop claim supported — so this
defect caps but does not alone zero the family; Defect A zeroes the rest.)

## 4. Capability gap (real, independent of the defects)

Even with A and B fixed, the v4.1 behavior fails the family's core demand:

* **No chained-lookup decomposition.** The hop-2 key's distinguishing token
  ("graveyard", "teal", "alluvium") is not in the question; it must be read
  off the corrected hop-1 record and used to key a second lookup. The model
  almost never did this as a query act (hop-2 token in a retrieve query:
  2/48 dml, 0/48 rag) and never echoed the parameterized `claim_key` in
  crew/stock (33/33 invented a semantic key instead, citing the right
  record). It succeeded at hop-2 key echo only in orchard (8/15 dml).
* **Unit retention regresses under multi-claim load**: 40/49 numeric claim
  values are bare ints ("13") despite the v4 `value_shape` class; the
  single-key families (isolated_project_scopes 23/24, explicit_corrections
  22/24 valid) show the adapter CAN echo keys/units when the key token is in
  the question.
* **Curriculum coverage**: v4 classes do not teach this pattern. `idiom_echo`
  (generate.py:1724) teaches echoing a terse key whose token appears in the
  question; `multipart_decomp` (generate.py:1934) teaches 2-3 part questions
  whose parts are each DIRECTLY retrievable by their asked labels — no class
  has a value derived from record 1 serving as the retrieval key for record
  2. There is no chained-lookup class in V3 or V4.

Second verbatim grounding episode (dml, 0003, verdict unsupported_final,
presented [2,3,5,4] — hop-1 correction and hop-2 record both present):

```json
{"key": "mess_stipend", "value": 13, "evidence_ids": [5]}          // right record (stipend_graveyard "13 credits"), invented key, units dropped
{"key": "galley_rule", "value": "graveyard rotation", "evidence_ids": [4]}  // invented key, wrong record cited, paraphrase value
```
Retrieves were "Sorcha Malloy mess stipend" then "Sorcha Malloy galley rule" —
no second-hop query on the discovered token ("graveyard").

## 5. Fix specification

### Generator (workload defect — fix and re-seal; digest changes, document supersession as with `5b1028f8…`)

1. **Defect A**: eliminate the arbitrary-substring truth entry. In each of
   `_build_dependent_lookup_{stock,crew,orchard}_h2`, rewrite the rule record
   text to end with an exact `... is <value>.` tail and switch the truth
   entry to `status_value`, e.g. crew event 700:
   `"Galley policy: a watchkeeper's mess stipend is set by the stipend of that watchkeeper's duty rotation; the governing lookup dimension is duty rotation."`
   with truth `{"key": "mess_stipend_rule", "value": "duty rotation", "verifier": "status_value"}`.
   (Verifier stays frozen; the fix is generator-side text + verifier-field change.)
2. **Defect B**: ask for the hop. Extend each variant's query to name the
   intermediate fact, e.g. crew:
   `"Give the mess stipend in credits that applies to {person} at present, state the duty rotation that currently governs {person}, and name the galley rule behind it."`
3. **Audit extension** (test_m8_sealed_workloads.py): extend `_audit_scenario`
   to (a) cover `record_text_contains` entries — require each such truth
   value to also appear as some record's exact ` is <value>.` tail (or forbid
   the verifier in sealed families), and (b) assert every truth key
   corresponds to an asked query part (truth-count == asked-part count, by
   construction per builder). Without (a)/(b) this class of miss recurs.
4. Re-seal heldout2; the digest changes — record the supersession in the
   gate doc exactly as the 2026-11-12 re-seal was recorded.

### Curriculum v5 (capability gap — new class)

`dependent_lookup_chain` (~40 episodes, new disjoint pools):
rule record + hop-1 record + hop-1 correction (supersession) + hop-2 table
records for BOTH tokens + untrusted-instruction distractor asserting the wrong
token's value. Gold behavior: retrieve subject → read the CORRECTED hop-1
value → issue a second retrieve keyed on the discovered token → final with
three claims (hop-2 value under the parameterized `claim_key` echoed verbatim
from the presented record's meta, unit-bearing string; hop-1 value under its
key; rule under its key), each citing exactly its own presented record.
Near-misses (never cited): superseded hop-1 value, distractor instruction,
bare-int hop-2 value, narrative/semantic key ("mess_stipend") in place of the
parameterized `claim_key`. Extends `multipart_decomp` from "answer every asked
part" to "discover the part's key from a prior hop".

## 6. Expected outcome of each lever

* Generator fix alone: the 9 near-miss episodes become winnable-as-answered;
  the ~35 crew/stock episodes still fail on invented keys / missing hop claim /
  dropped units. Family moves off zero at best marginally.
* Curriculum alone: model still cannot guess magic substrings or divine unasked
  truth entries — the family stays near zero.
* Both are required; neither is sufficient.
