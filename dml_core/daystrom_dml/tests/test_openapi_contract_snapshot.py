import json
from pathlib import Path
from typing import Any, cast

from daystrom_dml.provider_server import create_app
from daystrom_dml.tests.test_provider_server import DummyAdapter


def test_openapi_snapshot_contains_core_daystrom_paths():
    snapshot_path = Path(__file__).resolve().parents[3] / "docs" / "contracts" / "openapi-paths-v1.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    actual_paths = set(create_app(adapter_factory=lambda: cast(Any, DummyAdapter())).openapi()["paths"])

    missing = sorted(set(expected["required_paths"]) - actual_paths)

    assert expected["schema_version"] == "daystrom-openapi-paths-v1"
    assert missing == []
