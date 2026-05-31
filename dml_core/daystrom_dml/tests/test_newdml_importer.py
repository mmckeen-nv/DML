from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_importer():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "import_newdml_archive.py"
    spec = importlib.util.spec_from_file_location("import_newdml_archive", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


import_newdml_archive = _load_importer()


def test_legacy_meta_defaults_to_quarantine():
    item = {
        "id": 42,
        "timestamp": 123.0,
        "level": 0,
        "salience": 0.8,
        "fidelity": 1.0,
        "meta": {"doc_path": "useful.pdf"},
    }

    meta = import_newdml_archive._legacy_meta(
        item,
        Path("/tmp/newdml-archive.tar"),
        "Useful legacy memory text for continuity.",
        target_state="quarantined",
    )

    assert meta["memory_state"] == "quarantined"
    assert meta["namespace"] == "legacy_archive"
    assert meta["source"] == "old_openclaw_newdml_archive"
    assert meta["legacy_id"] == 42
    assert meta["legacy_doc_path"] == "useful.pdf"
    assert meta["quality_score"] > 0.0


def test_noise_filter_rejects_csv_hash_rows():
    item = {"meta": {"doc_path": "lcwa_gov_pdf_metadata.csv"}}
    text = (
        "Windows,-,1,612,792,94,5217,"
        "917222099f206146c1fb5096833cbd2629a1dd6f975f52eabe2569df92567de3,"
        "6a8e7e3f9f05d68550a0905a95a5d4a234aabadc96544732228a342b9a5cc931"
    )

    assert import_newdml_archive._looks_like_metadata_noise(item, text)
