"""Validate installable artifacts, independently of editable import paths."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def test_cpu_wheel_contains_runtime_packages_and_defaults(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, DML_BUILD_CUDA="0")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=root, env=env, check=True, capture_output=True, text=True,
    )
    wheel, = tmp_path.glob("*.whl")
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in (
            "daystrom_dml/config.yaml",
            "daystrom_dml/journal.py",
            "daystrom_dml/provider_client.py",
            "scripts/dml_journal.py",
            "scripts/performance_benchmark.py",
            "daystrom_dml/context/vllm_bridge/__init__.py",
            "daystrom_dml/context/vllm_bridge/connector.py",
            "daystrom_dml/context/vllm_bridge/policy.py",
            "daystrom_dml/contracts/schemas/context-packet-v1.schema.json",
            "daystrom_dml/provider_web/provider.js",
            "scripts/dcm_workload_benchmark.py",
            "scripts/dcm_kv_probe.py",
        ):
            assert name in archive.namelist()
