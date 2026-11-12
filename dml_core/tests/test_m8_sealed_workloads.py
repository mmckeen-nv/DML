"""Sealed M8 campaign workloads: determinism, schema, seal, isolation, dry-run.

No model, no tokenizer, no network. Generation is exercised twice into
separate directories and compared byte-for-byte; the seal is tampered with to
prove detection; both corpora are validated against schema.py and accepted by
the transport-native runner's --dry-run path unchanged.
"""

import hashlib
import json
from pathlib import Path
import re

import pytest

from scripts.m8_battery import runner_transport_native as battery
from scripts.m8_battery import schema
from scripts.m8_battery import sealed_workloads as sealed

DEV_FIXTURES = Path(__file__).parent.parent / "scripts" / "m8_battery" / "fixtures"
EXPECTED_FAMILY_COUNTS = {
    # dev mix 1:2:1:2:4:2 over 12 fixtures, scaled x18 (pilot) and x12 (heldout)
    "pilot": {
        "changing_facts": 18,
        "explicit_corrections": 36,
        "conflicting_independent_sources": 18,
        "isolated_project_scopes": 36,
        "dependent_lookups": 72,
        "stable_memory_adverse": 36,
    },
    "heldout": {
        "changing_facts": 12,
        "explicit_corrections": 24,
        "conflicting_independent_sources": 12,
        "isolated_project_scopes": 24,
        "dependent_lookups": 48,
        "stable_memory_adverse": 24,
    },
}
WORKLOAD_SIZES = {"pilot": 216, "heldout": 144}


def _scenario_files(directory):
    scenarios_dir = Path(directory) / "scenarios"
    return {path.name: path.read_bytes() for path in sorted(scenarios_dir.glob("*.json"))}


def _word_ngrams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def _corpus_ngrams(paths):
    grams = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        texts = [event["text"] for event in raw["events"]] + [raw["query"]]
        texts += [claim["value"] for claim in raw["truth"] if type(claim.get("value")) is str]
        for text in texts:
            grams |= _word_ngrams(text)
    return grams


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_generation_is_byte_identical_across_runs(tmp_path, workload):
    first = tmp_path / "first"
    second = tmp_path / "second"
    sealed.generate_workload(workload, first)
    sealed.generate_workload(workload, second)
    first_files = _scenario_files(first)
    second_files = _scenario_files(second)
    assert len(first_files) == WORKLOAD_SIZES[workload]
    assert first_files == second_files
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    for manifest in (first_manifest, second_manifest):
        manifest.pop("generated_at_commit")
    assert first_manifest == second_manifest


def test_serialization_is_canonical():
    scenario = sealed.generate_scenarios(20261110, 1, "pilot216_")[0]
    payload = sealed.serialize_scenario(scenario)
    assert payload == (json.dumps(scenario, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("utf-8")
    assert payload == sealed.serialize_scenario(json.loads(payload))


# --- schema validity -----------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_generated_scenarios_validate_against_schema(tmp_path, workload):
    out = tmp_path / workload
    sealed.generate_workload(workload, out)
    scenarios = schema.load_scenarios(out / "scenarios")
    assert len(scenarios) == WORKLOAD_SIZES[workload]
    prefix = sealed.WORKLOADS[workload]["namespace_prefix"]
    assert all(scenario.scenario_id.startswith(prefix) for scenario in scenarios)


# --- seal ----------------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_seal_verification_roundtrip(tmp_path, workload):
    out = tmp_path / workload
    expected = sealed.generate_workload(workload, out)
    manifest = sealed.verify_seal(out)
    assert manifest["set_digest"] == expected["set_digest"]
    assert manifest["scenario_count"] == WORKLOAD_SIZES[workload]
    shas = [entry["sha256"] for entry in manifest["scenarios"]]
    assert manifest["set_digest"] == hashlib.sha256("\n".join(sorted(shas)).encode("utf-8")).hexdigest()


def test_seal_detects_tampered_scenario_byte(tmp_path):
    out = tmp_path / "pilot"
    manifest = sealed.generate_workload("pilot", out)
    target = out / "scenarios" / f"{manifest['scenarios'][0]['scenario_id']}.json"
    payload = bytearray(target.read_bytes())
    index = payload.index(b"5000")
    payload[index] = ord("6")
    target.write_bytes(bytes(payload))
    with pytest.raises(sealed.SealError, match="sha256 mismatch"):
        sealed.verify_seal(out)


def test_seal_detects_missing_and_extra_files(tmp_path):
    out = tmp_path / "heldout"
    manifest = sealed.generate_workload("heldout", out)
    scenarios_dir = out / "scenarios"
    missing = scenarios_dir / f"{manifest['scenarios'][0]['scenario_id']}.json"
    missing.unlink()
    with pytest.raises(sealed.SealError, match="missing scenario file"):
        sealed.verify_seal(out)
    sealed.generate_workload("heldout", out)
    (scenarios_dir / "heldout_extra_unlisted_0000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(sealed.SealError, match="extra scenario file"):
        sealed.verify_seal(out)


# --- namespace and content isolation --------------------------------------------


def test_namespaces_disjoint_between_sets_and_dev_fixtures(tmp_path):
    pilot = tmp_path / "pilot"
    heldout = tmp_path / "heldout"
    sealed.generate_workload("pilot", pilot)
    sealed.generate_workload("heldout", heldout)
    pilot_ids = {path.stem for path in (pilot / "scenarios").glob("*.json")}
    heldout_ids = {path.stem for path in (heldout / "scenarios").glob("*.json")}
    dev_ids = {path.stem for path in DEV_FIXTURES.glob("*.json")}
    assert len(dev_ids) == 12
    assert pilot_ids.isdisjoint(heldout_ids)
    assert pilot_ids.isdisjoint(dev_ids)
    assert heldout_ids.isdisjoint(dev_ids)


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_zero_word_8gram_overlap_with_dev_fixtures(tmp_path, workload):
    out = tmp_path / workload
    sealed.generate_workload(workload, out)
    generated = _corpus_ngrams(sorted((out / "scenarios").glob("*.json")))
    dev = _corpus_ngrams(sorted(DEV_FIXTURES.glob("*.json")))
    overlap = generated & dev
    assert not overlap, f"shared 8-grams with dev fixtures: {sorted(overlap)[:5]}"


def test_no_generated_event_text_repeats_a_dev_event_text(tmp_path):
    out = tmp_path / "pilot"
    sealed.generate_workload("pilot", out)
    dev_texts = set()
    for path in DEV_FIXTURES.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        dev_texts.update(event["text"] for event in raw["events"])
        dev_texts.add(raw["query"])
    for path in (out / "scenarios").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for text in [event["text"] for event in raw["events"]] + [raw["query"]]:
            assert text not in dev_texts


# --- family balance --------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_family_balance_mirrors_dev_mix(tmp_path, workload):
    out = tmp_path / workload
    manifest = sealed.generate_workload(workload, out)
    assert manifest["family_counts"] == EXPECTED_FAMILY_COUNTS[workload]
    assert sum(manifest["family_counts"].values()) == WORKLOAD_SIZES[workload]


def test_family_counts_helper_matches_mix():
    assert sealed.family_counts(12) == {
        "changing_facts": 1,
        "explicit_corrections": 2,
        "conflicting_independent_sources": 1,
        "isolated_project_scopes": 2,
        "dependent_lookups": 4,
        "stable_memory_adverse": 2,
    }
    assert sealed.family_counts(216)["dependent_lookups"] == 72
    assert sealed.family_counts(144)["stable_memory_adverse"] == 24


# --- runner acceptance -------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOAD_SIZES))
def test_runner_dry_run_accepts_generated_set(tmp_path, capsys, workload):
    out = tmp_path / workload
    sealed.generate_workload(workload, out)
    argv = [
        "--scenarios-dir",
        str(out / "scenarios"),
        "--workdir-root",
        str(tmp_path / "work"),
        "--output",
        str(tmp_path / "episodes.jsonl"),
        "--dry-run",
    ]
    assert battery.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] == "fixtures_valid"
    assert len(report["scenarios"]) == WORKLOAD_SIZES[workload]
    assert all(len(entry["event_stream_digest"]) == 64 for entry in report["scenarios"])
