"""Declaration binding and interruption reconciliation without generation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).with_name("durable_campaign.py")
spec = importlib.util.spec_from_file_location("durable_campaign", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_worker_refuses_replaced_freeze_before_launch(tmp_path):
    command = [sys.executable, "-c", "from pathlib import Path;Path('should-not-exist').touch()"]
    freeze = {"files": {}, "command": command, "cwd": str(tmp_path), "synthetic": True}
    module.publish(tmp_path / "freeze.json", freeze)
    module.publish(tmp_path / "launch.json", {"freeze_sha256": "0" * 64})
    result = subprocess.run([sys.executable, str(SCRIPT), "worker", str(tmp_path)], capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / "should-not-exist").exists()
    assert not (tmp_path / "start.json").exists()


def test_single_use_worker_retains_terminal_and_cannot_resume(tmp_path):
    freeze = {"files": {str(SCRIPT): module.digest(SCRIPT)}, "command": [sys.executable, "-c", "pass"],
              "cwd": str(tmp_path), "synthetic": True}
    module.publish(tmp_path / "freeze.json", freeze)
    module.publish(tmp_path / "launch.json", {"freeze_sha256": module.digest(tmp_path / "freeze.json")})
    command = [sys.executable, str(SCRIPT), "worker", str(tmp_path)]
    assert subprocess.run(command).returncode == 0
    original = (tmp_path / "terminal.json").read_bytes()
    assert json.loads(original)["status"] == "completed_synthetic"
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (tmp_path / "terminal.json").read_bytes() == original


def test_reconcile_prior_boot_records_interruption_without_generation(tmp_path):
    module.publish(tmp_path / "start.json", {"boot_id": "previous-boot"})
    assert subprocess.run([sys.executable, str(SCRIPT), "reconcile", str(tmp_path)]).returncode == 0
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "interrupted"
    assert terminal["usage_unknown"] and not terminal["resumable"]
    original = (tmp_path / "terminal.json").read_bytes()
    module.finalize(tmp_path)
    assert (tmp_path / "terminal.json").read_bytes() == original


def test_publish_never_overwrites_terminal(tmp_path):
    path = tmp_path / "terminal.json"
    module.publish(path, {"status": "interrupted"})
    with pytest.raises(FileExistsError):
        module.publish(path, {"status": "completed"})
    assert json.loads(path.read_text())["status"] == "interrupted"
