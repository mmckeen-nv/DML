from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_MANIFEST = ROOT / "notes" / "read-lane-promotion-manifest.md"

ALLOWED_PREFIXES = (
    "notes/",
    "specs/",
    "scripts/",
    "tests/",
    "examples/dpm/config.active-read.json",
    "examples/dpm/config.observe-only.json",
    "examples/dpm/config.disabled.json",
    "PROJECT.md",
)

EXCLUDED_MARKERS = (
    "runtime/",
    "out/",
    "scratch/",
    ".pytest_cache/",
    "__pycache__/",
    "active-write",
)


def _manifest_packet_entries() -> list[str]:
    text = PROMOTION_MANIFEST.read_text().splitlines()
    entries: list[str] = []
    in_packet = False
    excluded = False
    for line in text:
        if line.startswith("## Promotion packet contents"):
            in_packet = True
            continue
        if not in_packet:
            continue
        if line.startswith("Excluded from rehearsal promotion packet:"):
            excluded = True
            continue
        if excluded:
            break
        stripped = line.strip()
        if stripped.startswith("- `") and stripped.endswith("`"):
            entries.append(stripped[3:-1])
    return entries


def test_ingress_preflight_manifest_packet_contains_only_reviewable_read_lane_entries():
    packet_entries = _manifest_packet_entries()
    assert packet_entries, "manifest packet entries missing"

    for entry in packet_entries:
        assert entry.startswith(ALLOWED_PREFIXES), f"non-reviewable packet entry: {entry}"
        assert all(marker not in entry for marker in EXCLUDED_MARKERS), f"excluded entry leaked into packet: {entry}"


def test_ingress_preflight_manifest_declares_rejected_candidates_before_rehearsal():
    manifest = PROMOTION_MANIFEST.read_text()

    required_rejections = [
        "- `examples/dpm/config.active-write.json` as an enabled target",
        "- `runtime/`",
        "- `out/`",
        "- `scratch/`",
        "- Python cache artifacts",
        "- no active-write semantics are introduced during promotion",
    ]
    for line in required_rejections:
        assert line in manifest, f"missing rejection rule: {line}"
