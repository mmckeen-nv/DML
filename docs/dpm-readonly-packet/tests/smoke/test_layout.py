from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_CACHE_DIRS = {'.pytest_cache', '__pycache__'}
EXCLUDED_CACHE_SUFFIXES = {'.pyc', '.pyo'}
PYTEST_SELF_CACHE_ROOTS = {
    ROOT / 'scripts',
    ROOT / 'tests' / 'smoke',
    ROOT / 'tests' / 'unit',
    ROOT / 'tests',
}


def _packet_cache_leaks() -> list[Path]:
    leaks: list[Path] = []
    for path in ROOT.rglob('*'):
        if path.is_dir() and path.name in EXCLUDED_CACHE_DIRS:
            if path.name == '__pycache__' and path.parent in PYTEST_SELF_CACHE_ROOTS:
                continue
            leaks.append(path)
            continue
        if path.is_file() and path.suffix in EXCLUDED_CACHE_SUFFIXES:
            if any(parent.name == '__pycache__' and parent.parent in PYTEST_SELF_CACHE_ROOTS for parent in path.parents):
                continue
            leaks.append(path)
    return sorted(leaks)


def test_dpm_layout_exists():
    required = [
        ROOT / 'PROJECT.md',
        ROOT / 'specs',
        ROOT / 'notes',
        ROOT / 'tests' / 'smoke',
        ROOT / 'scripts',
        ROOT / 'specs' / 'continuity-metadata-foundation.md',
        ROOT / 'notes' / 'next-step.md',
        ROOT / 'notes' / 'packet-2-boundary-contract.md',
    ]
    for path in required:
        assert path.exists(), f'missing: {path}'

    excluded = [
        ROOT / 'runtime',
        ROOT / 'out',
    ]
    for path in excluded:
        assert not path.exists(), f'excluded generated path present in read-only packet: {path}'

    assert not _packet_cache_leaks(), f'excluded cache artifact present in read-only packet: {_packet_cache_leaks()}'
