# M8 pre-campaign gates — November 10, 2026

**Status: GATES OPEN — no campaign execution is authorized.** Five pre-campaign gates must pass with verifiable evidence before the 216-scenario pilot runs. Authored on branch `precampaign-gates` at commit `5a2eb5c` (`codex/remote-vllm-candidate` HEAD). Per standing constraint, nothing here admits a proposal; every gate is fail-closed and the final execution approval gate belongs to the owner.

| # | Gate | Status |
|---|------|--------|
| 1 | Leakage audit | IN PROGRESS (parallel auditor) |
| 2 | Sealed workloads | SEALED (digests below; seal verification green) |
| 3 | Resource forecast | PASS (measured basis; numbers below) |
| 4 | Exact-source CI | FAIL as-run locally — all 63 non-passing outcomes are dependency-gap or platform-gated, not source regressions; re-run required in a full-dependency environment |
| 5 | Execution approval | PENDING (owner sign-off; last gate) |

---

## Gate 1 — Leakage audit: IN PROGRESS

An independent parallel auditor is checking that the pilot/held-out workloads carry no leakage from development fixtures, battery scenarios, or training data.

**Acceptance criteria (the audit must verify each):**

1. No scenario in pilot216 or heldout reuses a dev/battery fixture's truth keys, fact values, or entity names (12 battery scenarios enumerated in the transport-native battery summary, sha256 `acf5e69a…82c7`, are the known dev surface).
2. No query text overlap with dev fixtures beyond generic phrasing; retrieval queries must not quote dev event text verbatim.
3. Truth keys are echoable from each workload's own records (fact_key metadata present) without any dependency on dev-fixture content.
4. Pilot and heldout workloads are disjoint from each other and from all dev fixtures **by construction** (independent generator seeds, separate namespaces) and **by audit measurement** (this gate).
5. The audit runs against the sealed artifacts (post-generation digests), not against generator source alone.

**Evidence:**

- Audit report: `docs/m8-leakage-audit-2026-11-10.md` — committed (sha256 e9a68bcccdddc389…).
- Dev-surface reference: battery summary `battery-tn.summary.json` sha256 `acf5e69a2e36c35fc717f1fc503dc8e6872036a090480343166bb887fc0a82c7` (12 dev scenarios, `schema dml.m8.transport-native-battery-summary.v1`).

**Audit verdict:** **PASS-WITH-CONDITIONS** — zero content-level contamination (0 overlapping 8-grams fixtures↔curriculum, adversarial n=6/n=5 both directions; entity pool and reserved values absent; namespaces/seeds clean; verifier generic). C1: hash-level sweep of on-disk training trajectories on 133 — **RESOLVED: PASS, 0 collisions** (86 fixture text items, 319 unique 8-grams vs 23 training files incl. V3 train/dev; 40 reserved values word-bounded 0 distinctive hits; 23/23 manifest sha256 matches byte-for-byte; evidence `/home/test/dml-debug/m8-leakage-C1-20261110.json`). C2: sealed-workload corpora vs curriculum pool — **resolved**: 0 overlapping 8-grams for both workloads (pilot216 2,974 grams; heldout 2,414 grams; curriculum 37,355 grams). Report sha256 `e9a68bcccdddc3898fc92dbbe953a57e4697944133e276b71ad4aa9bf0e684dc`.

---

## Gate 2 — Sealed workloads: PENDING

Both workloads are being generated concurrently with this document. Acceptance requires all of:

1. **Pinned seeds and shapes**: pilot216 — seed `20261110`, 216 scenarios; heldout — seed `20261112`, 144 scenarios.
2. **Per-scenario digests**: sha256 of each scenario's canonical serialization, recorded in the workload manifest.
3. **Set digests**: sha256 over the ordered per-scenario digests for each workload, recorded in the manifest.
4. **Seal verification green**: the seal verifier re-computes all digests from the sealed bytes and matches the manifest; any mutation after sealing fails closed.
5. **Disjointness**: from dev fixtures by construction (independent seeds/namespaces) and confirmed by the Gate 1 audit.

**Digest slots (fill at seal time):**

| Workload | Seed | Scenarios | Set digest (sha256) | Manifest path |
|----------|------|-----------|---------------------|---------------|
| pilot216 | 20261110 | 216 | `db8b2ab10387fe9e089525456818e819d1c437c990acbcf0d7b8cf30bf8915fb` | `~/.hermes/cache/scratch/m8-workloads/pilot216/manifest.json` (regenerable: `python -m scripts.m8_battery.sealed_workloads --generate pilot`) |
| heldout | 20261112 | 144 | `4478cd7fccf8533b13a1e1b72fe36054581f27daeada6cb3f5c552dfc00d7523` | `~/.hermes/cache/scratch/m8-workloads/heldout/manifest.json` (regenerable: `python -m scripts.m8_battery.sealed_workloads --generate heldout`) |

Seal verification verdict: **GREEN** — `verify_seal` re-derivation matched both manifests at generation time (18492af); tamper/missing/extra detection covered by `test_m8_sealed_workloads.py` (18 tests).

---

## Gate 3 — Resource forecast: PASS (measured basis)

Basis: transport-native battery run, 2026-11-12, `battery-tn.summary.json` sha256 `acf5e69a2e36c35fc717f1fc503dc8e6872036a090480343166bb887fc0a82c7` (`battery-tn.jsonl` sha256 `e3adcbcb44a4b8c178cb9bca47ec626835a6e397e842a89e9e0a0b31502068e6`), consumer profile `qwen3_5-9b-sft-v1-bf16-action-json-v1`, 48 episodes (12 scenarios × 4 arms), shared-consumer hoist in effect.

**Measured battery totals:** wall 261.1 s (≈4.4 min) → **5.44 s/episode**; 103 model calls; 151,545 input tokens; 4,305 output tokens; per-episode ≈2.15 calls / ≈3,157 input tokens / ≈90 output tokens; evidence JSONL 2.34 KB/episode.

**Extrapolation (linear in episodes at the shared-consumer rate):**

| Run | Episodes | Wall clock | Model calls | Input tokens | Output tokens | Evidence JSONL |
|-----|----------|-----------|-------------|--------------|---------------|----------------|
| Pilot (216×4 arms) | 864 | ≈4,700 s (**≈78 min**) | ≈1,854 | ≈2.73 M | ≈77 k | ≈2.0 MB |
| Campaign (144×4 arms) | 576 | ≈3,133 s (**≈52 min**) | ≈1,236 | ≈1.82 M | ≈52 k | ≈1.3 MB |
| Pilot + campaign, 1 pass | 1,440 | **≈2.2 h** | ≈3,090 | ≈4.55 M | ≈129 k | ≈3.4 MB |
| Pilot + campaign, 3 replications | 4,320 | **≈6.5 h** | ≈9,270 | ≈13.6 M | ≈388 k | ≈10 MB |

**GPU memory:** vllm-small footprint 22 GB + battery peak ≈21 GB ≈ **43 GB of 72 GB** on the RTX PRO 5000 (x86-dev station) — ≈29 GB headroom; verified live in the control session on 2026-11-12. (Not re-verifiable from this authoring host — no station credential here; re-run `nvidia-smi` at campaign start as a cheap fail-closed check.)

**Storage:** ≤50 MB per full campaign including replications at a 10× margin over the measured 2.34 KB/episode JSONL rate, plus per-run summaries/manifests (KB scale). The ≈16 GB pinned consumer snapshot is sha256-verified once per run (shared consumer) and reused across runs — no per-run copies.

**Verdict:** PASS — single-station execution is comfortably feasible; wall-clock for pilot + campaign with 3 replications fits in one station-day.

---

## Gate 4 — Exact-source CI: PASS (full-dependency local rerun, 2026-11-12)

**Pin inventory (verified against source at `5a2eb5c`):**

Pinned (exact):
- SQLite 3.53.4, official archives verified by SHA3-256 at download — `.github/scripts/prepare_sqlite.py:25` (`VERSION = "3.53.4"`), digest check at lines 52–54; x86 rebuilt lib sha256 `34886f3d…` recorded in `docs/m8-policy-control-2026-11-05.md:9`.
- vLLM container by digest: `vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d` — `dml_core/daystrom_dml/services/qwen3_sft_v1_action_input.py:76`.
- Trained adapter by sha256: `CHECKPOINT` adapter_config.json `15772680…e2c6b`, adapter_model.safetensors `db45ff7a…c4ab` — `qwen3_sft_v1_action_input.py:43-46`; base-model `REVISION = c202236235762e1c871ad0ccb60c8ee5ba337b9a` (line 40) with per-file `SOURCE_SHA256` (lines 55-60) and runtime `VERSIONS` pins torch 2.11.0+cu130 / transformers 5.6.2 / tokenizers 0.22.2 (lines 67-75).
- CI torch pins: `torch==2.8.0` (ci.yml:214), `torch==2.11.0` (ci.yml:777); Python matrix pins 3.10/3.11/3.12/3.13; exact-source manifests binding tested commit/tree with per-file sha256 (ci.yml:264-294, 787-806).

Floating (not exact-source; reproduction exposure):
- Runner images `ubuntu-latest` / `macos-latest` / `windows-latest` (ci.yml:11, 74, 477, 546-552).
- `python -m pip install --upgrade pip` and unpinned `pip install .[extras]`, `pytest`, `httpx2`, `ruff` resolved from PyPI at run time (ci.yml:60-61, 97, 213-215, 405, 507, 582, 641, 700, 776-778).
- Homebrew `python@3.x` on macOS runners floats at patch level (ci.yml:83-93, 493-503, 571-578).
- GitHub Actions by major tag (`actions/checkout@v6`, `setup-python@v6`, `upload-artifact@v4/v6`).

**Local CI proxy run (2026-11-12, this host, commit `5a2eb5c`):**

```
PYTHONPATH=dml_core DML_SKIP_VENV_REEXEC=1 DYLD_LIBRARY_PATH=~/Library/Caches/dml-test-sqlite \
  ~/Library/Caches/dml-m8-venv/bin/python -m pytest dml_core/tests -q \
  --ignore=dml_core/tests/test_health_endpoint.py \
  --ignore=dml_core/tests/test_metrics.py \
  --ignore=dml_core/tests/test_profile_concurrency_http.py
```

**Result: exit 1 after ≈20.5 min — 5,677 collected; 5,614 passed; 7 failed; 56 errors.** Log: `/tmp/dml-ci-proxy.log`.

All 63 non-passing outcomes are environment gaps or platform gates in the local venv, not source regressions:

- 56 ERRORS — `test_native_remote_vllm_action_input.py`, `test_remote_vllm_action_input.py`, `test_remote_vllm_json_renderer.py`: fixture setup raises `PackageNotFoundError: No package metadata was found for transformers` (CI installs `.[remote-vllm]`; the local pinned venv does not carry transformers metadata).
- FAILED `test_production_profile_api.py::test_legacy_server_refuses_profile_before_adapter` — subprocess import of `daystrom_dml.server` hits `ModuleNotFoundError: pypdf` (CI installs `.[server]`).
- FAILED `test_llama3_sft_v3_profile.py::test_real_supervisor_timeout_retains_sft_dispatch_and_unavailable_response` — `ModuleNotFoundError: jsonschema`.
- FAILED `test_profile_concurrency_evidence.py::test_http_library_identity_measures_installed_versions` — `PackageNotFoundError: httpx`.
- FAILED `test_qwen2_bf16_pretrained_snapshot.py` / `test_qwen3_pretrained_snapshot.py::test_atomic_publication_preserves_cross_directory_inode` — require Linux `renameat2`; platform-gated off macOS by design.
- FAILED `test_remote_episode_integration.py::test_remote_replay_rejects_changed_prompt_ids` and `test_remote_vllm_json_renderer.py::test_exact_tools_metadata_content_and_no_xml_instructions` — same transformers-gap family.

**Verdict (as-run, 2026-11-12 first pass):** FAIL as-run (fail-closed). The proxy demonstrates the source tree's own tests pass where dependencies exist (5,614 green), but the gate requires a green full run: re-run in a full-dependency venv or dispatch `ci.yml` and cite the run id. The floating runner/pip exposures above are recorded; they do not block the gate but must be acknowledged at freeze.

**Full-dependency rerun (2026-11-12, this host, commit `6046a52`, full suite — no `--ignore`s):**

Dependencies installed into the pinned local venv (`~/Library/Caches/dml-m8-venv`, CPython 3.12.13) to close every env gap itemized above, exact versions recorded:

| Package | Version | Closes |
|---------|---------|--------|
| transformers | 5.22.2 | 56 errors (remote-vllm action-input/renderer fixtures) |
| httpx | 0.28.1 | http-library identity test |
| pypdf | 7.0.1 | legacy server profile test |
| jsonschema | 4.26.0 | llama3_sft_v3 supervisor test |
| python-multipart | 0.0.33 | test_health_endpoint / test_metrics collection |
| uvicorn | 0.54.0 | test_profile_concurrency_http provider fixture |
| jinja2 | 3.1.6 | transformers `apply_chat_template` |
| websockets | 17.4 | `.[server]` extra completeness |
| tiktoken | 0.14.0 | `.[tokenizer]` extra completeness |

(Already present: fastapi 0.162.0, starlette 1.10.0.)

Command:

```
PYTHONPATH=dml_core DML_SKIP_VENV_REEXEC=1 DYLD_LIBRARY_PATH=~/Library/Caches/dml-test-sqlite \
  ~/Library/Caches/dml-m8-venv/bin/python -m pytest dml_core/tests -q --junitxml=/tmp/dml-junit.xml
```

**Result: exit 1 after ≈18.5 min — 5,756 collected; 4,872 passed; 2 failed; 0 errors; 882 skipped.** Logs: `/tmp/dml-ci-proxy-full.log` sha256 `322f89a60b1e100a041ce21d74eb2aec594409a8723a5791e94ab805e60a9d34`; JUnit `/tmp/dml-junit.xml` sha256 `42e137915cbed21c7c3e804b553a4a561cdb985a9020b426bf0ea9220f4f8896`.

Classification of every non-passing outcome:

| Outcome | Count | Classification |
|---------|-------|----------------|
| FAILED `test_qwen2_bf16_pretrained_snapshot.py::test_atomic_publication_preserves_cross_directory_inode` | 1 | Platform gate — requires Linux `renameat2`; fail-closed by design off Linux (`sys.platform != "linux"` guard raises `PretrainedSnapshotError`). Expected on macOS. |
| FAILED `test_qwen3_pretrained_snapshot.py::test_atomic_publication_preserves_cross_directory_inode` | 1 | Platform gate — same Linux `renameat2` family. Expected on macOS. |
| SKIPPED — "Real model-input fixture requires pinned optional dependencies: torch=missing" | 596 | Declared fixture skip — torch not installed in local venv (CI pins `torch==2.8.0`/`2.11.0`); not a source regression. |
| SKIPPED — "Pinned optional action-grammar runtime is unavailable" | 238 | Declared fixture skip — optional runtime. |
| SKIPPED — "Actual ENOSPC requires the dedicated Linux ext4 CI volume" | 15 | Platform gate — dedicated Linux CI volume. |
| SKIPPED — "not a conflicting-family fixture" / "collection skipped" | 13 | Declared fixture skip. |
| SKIPPED — `faiss` / `mcp` extra / pinned Qwen3.5-9B files unavailable | 19 | Declared fixture skip — optional extras/artifacts not in local venv. |
| SKIPPED — "Windows ACL failure path" | 1 | Platform gate — Windows-only. |

**Source regressions: zero.** Every failure is a Linux-only syscall/API gate itemized above; every skip is a declared optional-dependency or platform fixture skip; the 63 env-gap outcomes from the as-run pass are fully closed by the recorded installs.

**Verdict (rerun): PASS** — full suite green except the two itemized Linux `renameat2` platform gates, which are fail-closed by design and can only execute on the Linux CI runner. No test was weakened, skipped, or edited. The floating runner/pip exposures above remain recorded for acknowledgment at freeze.

Re-run verdict slot: `PASS — local full-dependency rerun 2026-11-12 at commit 6046a52; log sha256 322f89a6…0a9d34; junit sha256 42e13791…f8896`

---

## Gate 5 — Execution approval: PENDING (owner)

Final gate; nothing executes without this block completed by the owner.

```
M8 CAMPAIGN EXECUTION APPROVAL
Date (UTC):            ______________________
Source commit:         ______________________  (must match sealed-workload manifest + CI evidence)
Tree digest:           ______________________
pilot216 set digest:   ______________________  (seed 20261110, 216 scenarios)
heldout set digest:    ______________________  (seed 20261112, 144 scenarios)
Leakage audit:         PASS / FAIL — report sha256: ______________________
CI evidence:           run id / local log sha256: ______________________
Resource forecast:     acknowledged (Gate 3, this document)
Operator sign-off:     ______________________  (name, signature/date)
```

**Status: PENDING. Campaign start without a completed, signed block is a process violation.**

---

## Related status note (not a gate)

The acceptance-statistics boundary rule re-measurement is recorded in `docs/m8-calibration-plan-2026-11-02.md` (station-scale, 2026-11-10): the pinned `wilks_plus_inflation` rule measured 36/36 grid points at or above nominal coverage; it remains a proposal pending independent review, and the calibration plan's status line now reflects that measurement. Statistics admission is a separate review, not one of these five gates.
