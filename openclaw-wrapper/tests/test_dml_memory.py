import importlib.util
import io
import fcntl
import json
import os
import sys
import tempfile
import time
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path


class _StubMemoryKindValue:
    def __init__(self, value: str):
        self.value = value


class _StubMemoryKind:
    ACTION = _StubMemoryKindValue("action")
    OBSERVATION = _StubMemoryKindValue("observation")
    NOTE = _StubMemoryKindValue("note")
    PLAN = _StubMemoryKindValue("plan")
    ERROR = _StubMemoryKindValue("error")
    ARTIFACT_REF = _StubMemoryKindValue("artifact_ref")


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dml_memory.py"

    daystrom_pkg = types.ModuleType("daystrom_dml")
    daystrom_schema = types.ModuleType("daystrom_dml.agent_schema")
    daystrom_adapter = types.ModuleType("daystrom_dml.dml_adapter")

    daystrom_schema.MemoryKind = _StubMemoryKind

    class _Adapter:  # pragma: no cover - just to satisfy import
        pass

    daystrom_adapter.DMLAdapter = _Adapter

    prev = {
        "daystrom_dml": sys.modules.get("daystrom_dml"),
        "daystrom_dml.agent_schema": sys.modules.get("daystrom_dml.agent_schema"),
        "daystrom_dml.dml_adapter": sys.modules.get("daystrom_dml.dml_adapter"),
    }
    sys.modules["daystrom_dml"] = daystrom_pkg
    sys.modules["daystrom_dml.agent_schema"] = daystrom_schema
    sys.modules["daystrom_dml.dml_adapter"] = daystrom_adapter

    try:
        spec = importlib.util.spec_from_file_location("dml_memory", module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules["dml_memory"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in prev.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


mod = _load_module()


def _write_state_records(storage_dir: str, records: list[dict]) -> Path:
    state_path = Path(storage_dir) / "dml_state.jsonl"
    payload_lines = [json.dumps(record, separators=(",", ":"), sort_keys=True) for record in records]
    checksum = mod.hashlib.sha256("\n".join(payload_lines).encode("utf-8")).hexdigest()
    header = {
        "type": "daystrom_dml.memory",
        "version": 1,
        "created_at": "2026-05-21T00:00:00+00:00",
        "count": len(payload_lines),
        "checksum": checksum,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(header, separators=(",", ":"), sort_keys=True) + "\n" + "\n".join(payload_lines),
        encoding="utf-8",
    )
    return state_path


class _DummyAdapter:
    def __init__(self, raise_gt: bool = False, sleep_gt_s: float = 0.0, retrieval_report: dict | None = None):
        self.raise_gt = raise_gt
        self.sleep_gt_s = sleep_gt_s
        self.retrieval_report = retrieval_report
        self.ingests: list[tuple[str, dict | None, bool]] = []
        self.preferences: list[tuple[str, dict]] = []
        self.persist_calls = 0

    def ingest(self, text: str, meta: dict | None = None, *, persist: bool = True) -> None:
        self.ingests.append((text, meta, persist))

    def record_personality_preference(self, text: str, **kwargs: object) -> dict:
        self.preferences.append((text, kwargs))
        return {"status": "recorded", "node_id": "pref.test"}

    def _persist_all(self) -> None:
        self.persist_calls += 1

    def retrieve_context(self, query: str, **_: object) -> dict:
        if self.retrieval_report is not None:
            report = dict(self.retrieval_report)
            report["query_seen"] = query
            return report
        return {"status": "ok", "query_seen": query}

    def query_database(self, query: str, mode: str = "hybrid") -> dict:
        if self.sleep_gt_s > 0:
            time.sleep(self.sleep_gt_s)
        if self.raise_gt:
            raise RuntimeError("ground truth backend unavailable")
        return {"mode": mode, "query": query, "hits": 2}

    def close(self) -> None:
        return None


class _FakeOllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", dim: int = 1024):
        self.base_url = base_url
        self._dim = dim



class TestAdapterConstruction(unittest.TestCase):
    def test_foreground_adapter_leaves_working_subsystems_enabled_by_default(self):
        calls = []

        class FakeDMLAdapter:
            def __init__(self, *args, **kwargs):
                calls.append({"args": args, "kwargs": kwargs})

        original_adapter = mod.DMLAdapter
        try:
            mod.DMLAdapter = FakeDMLAdapter
            adapter = mod._adapter("/tmp/dml-store", None, False)
        finally:
            mod.DMLAdapter = original_adapter

        self.assertIsInstance(adapter, FakeDMLAdapter)
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]["kwargs"]
        overrides = kwargs["config_overrides"]
        self.assertNotIn("background_processing_enabled", overrides)
        self.assertNotIn("skip_rag_state_import", overrides)
        self.assertNotIn("start_aging_loop", kwargs)

class TestGpuOnlyBackendProof(unittest.TestCase):
    def test_backend_proof_reports_ollama_embedder_surface(self):
        adapter = types.SimpleNamespace(
            config={
                "embedding_model": "ollama:qwen3-embedding:0.6b",
                "embedding_device": "cuda",
                "llm_backend": "ollama",
                "model_name": "llama3:8b",
            },
            embedder=_FakeOllamaEmbedder(),
            runner=types.SimpleNamespace(is_dummy=False, _backend=object()),
            storage_dir=Path("/tmp/dml-proof"),
        )
        report = mod._backend_proof(adapter)
        self.assertEqual(report["embedder_backend"], "ollama")
        self.assertEqual(report["embedder_target_device"], "ollama-managed")
        self.assertTrue(report["embedder_ready"])
        self.assertEqual(report["embedding_device_cfg"], "cuda")

    def test_assert_gpu_only_accepts_ollama_embedder_when_cuda_config_is_explicit(self):
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            return original_import(name, *args, **kwargs)

        adapter = types.SimpleNamespace(
            config={"embedding_model": "ollama:qwen3-embedding:0.6b", "embedding_device": "cuda"},
            embedder=_FakeOllamaEmbedder(),
            runner=types.SimpleNamespace(is_dummy=False, _backend=object()),
            storage_dir=Path("/tmp/dml-proof"),
        )
        import builtins
        builtins_import = builtins.__import__
        builtins.__import__ = fake_import
        try:
            mod._assert_gpu_only(adapter)
        finally:
            builtins.__import__ = builtins_import

    def test_assert_gpu_only_rejects_ollama_embedder_without_explicit_cuda_config(self):
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            return original_import(name, *args, **kwargs)

        adapter = types.SimpleNamespace(
            config={"embedding_model": "ollama:qwen3-embedding:0.6b", "embedding_device": "cpu"},
            embedder=_FakeOllamaEmbedder(),
            runner=types.SimpleNamespace(is_dummy=False, _backend=object()),
            storage_dir=Path("/tmp/dml-proof"),
        )
        import builtins
        builtins_import = builtins.__import__
        builtins.__import__ = fake_import
        try:
            with self.assertRaises(RuntimeError):
                mod._assert_gpu_only(adapter)
        finally:
            builtins.__import__ = builtins_import


class TestGroundTruthHardening(unittest.TestCase):
    def test_attach_ground_truth_records_error_without_raise_when_not_strict(self):
        report = {"status": "ok"}
        mod._attach_ground_truth(report, adapter=_DummyAdapter(raise_gt=True), query="q", mode="hybrid", strict=False)
        self.assertEqual(report["ground_truth_status"], "error")
        self.assertIn("ground truth backend unavailable", report["ground_truth_error"])
        self.assertIsNone(report["ground_truth"])

    def test_attach_ground_truth_raises_when_strict(self):
        with self.assertRaises(RuntimeError):
            mod._attach_ground_truth(
                {"status": "ok"},
                adapter=_DummyAdapter(raise_gt=True),
                query="q",
                mode="hybrid",
                strict=True,
            )

    def test_attach_ground_truth_timeout_sets_timeout_status(self):
        report = {"status": "ok"}
        started = time.perf_counter()
        mod._attach_ground_truth(
            report,
            adapter=_DummyAdapter(sleep_gt_s=0.2),
            query="q",
            mode="hybrid",
            strict=False,
            timeout_ms=5,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertEqual(report["ground_truth_status"], "timeout")
        self.assertIn("timed out", report["ground_truth_error"])
        # Timeout handling should return promptly instead of waiting for the full slow call.
        self.assertLess(elapsed_ms, 100.0)

    def test_cmd_retrieve_emits_json_when_ground_truth_fails(self):
        original_adapter = mod._adapter
        try:
            mod._adapter = lambda *_args, **_kwargs: _DummyAdapter(raise_gt=True)
            args = Namespace(
                storage_dir="/tmp/does-not-matter",
                config_path=None,
                require_gpu=False,
                query="How do I export USD?",
                query_expand=False,
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                top_k=3,
                with_ground_truth=True,
                ground_truth_mode="hybrid",
                ground_truth_policy="always",
                confidence_threshold=0.46,
                reform_memory=True,
                strict_ground_truth=False,
                ground_truth_timeout_ms=1800,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_retrieve(args)
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["ground_truth_status"], "error")
            self.assertEqual(payload["ground_truth_reason"], "policy_always")
            self.assertIn("memory_reformed_chunks", payload)
        finally:
            mod._adapter = original_adapter

    def test_cmd_retrieve_surfaces_conflicted_items(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter(
            retrieval_report={
                "status": "ok",
                "items": [
                    {
                        "id": "7",
                        "text": "Current claim says deploy_mode is manual.",
                        "meta": {
                            "tenant_id": "openclaw",
                            "namespace": "ops",
                            "conflict_key": "deploy_mode",
                            "claim_value": "manual",
                            "conflict_state": "conflicted",
                            "conflict_scope": {
                                "tenant_id": "openclaw",
                                "namespace": "ops",
                                "conflict_key": "deploy_mode",
                            },
                            "conflicts_with": [{"id": 3, "claim_value": "automatic"}],
                        },
                    }
                ],
            }
        )
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            args = Namespace(
                storage_dir="/tmp/does-not-matter",
                config_path=None,
                require_gpu=False,
                query="deploy mode",
                query_expand=False,
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                top_k=3,
                include_quarantined=False,
                with_ground_truth=False,
                ground_truth_mode="hybrid",
                ground_truth_policy="never",
                confidence_threshold=0.46,
                reform_memory=False,
                strict_ground_truth=False,
                ground_truth_timeout_ms=1800,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_retrieve(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["conflict_count"], 1)
            self.assertIn("=== Memory Conflicts ===", payload["raw_context"])
            self.assertEqual(payload["conflicts"][0]["scope"]["conflict_key"], "deploy_mode")
        finally:
            mod._adapter = original_adapter


class TestContinuityResume(unittest.TestCase):
    def test_cmd_resume_filters_active_continuity_and_emits_compact_checkpoint(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter(
            retrieval_report={
                "status": "ok",
                "items": [
                    {
                        "text": "Unrelated imported manual text.",
                        "meta": {"source": "archive_import", "namespace": "old_import"},
                    },
                    {
                        "text": "\n".join(
                            [
                                "[source:rolling_thread_checkpoint]",
                                "thread: main",
                                "updated_at: 2026-05-21T12:00:00Z",
                                "state: executing",
                                "task: activate continuity loop",
                                "next_action: run smoke tests",
                            ]
                        ),
                        "meta": {
                            "source": "rolling_thread_checkpoint",
                            "namespace": "active_continuity",
                            "memory_state": "active",
                        },
                    },
                    {
                        "text": "thread: old\nnext_action: ignore",
                        "meta": {
                            "source": "rolling_thread_checkpoint",
                            "namespace": "active_continuity",
                            "memory_state": "quarantined",
                        },
                    },
                ],
            }
        )
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            args = Namespace(
                storage_dir="/tmp/does-not-matter",
                config_path=None,
                require_gpu=False,
                query="resume continuity",
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                top_k=12,
                fallback_items=3,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_resume(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["action"], "resume")
            self.assertEqual(payload["continuity_items"], 1)
            self.assertFalse(payload["fallback_used"])
            self.assertEqual(payload["latest_checkpoint"]["next_action"], "run smoke tests")
            self.assertIn("=== Active Continuity Resume ===", payload["raw_context"])
            self.assertIn("activate continuity loop", payload["raw_context"])
            self.assertNotIn("archive_import", payload["raw_context"])
        finally:
            mod._adapter = original_adapter

    def test_cmd_resume_uses_newest_active_checkpoint_not_retrieval_order(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter(
            retrieval_report={
                "status": "ok",
                "items": [
                    {
                        "text": "\n".join(
                            [
                                "thread: old-session",
                                "updated_at: 2026-05-21T12:00:00Z",
                                "state: stale",
                                "task: old work",
                                "next_action: ignore old checkpoint",
                            ]
                        ),
                        "timestamp": 1.0,
                        "meta": {
                            "source": "rolling_thread_checkpoint",
                            "namespace": "active_continuity",
                            "memory_state": "active",
                        },
                    },
                    {
                        "text": "\n".join(
                            [
                                "thread: current-session",
                                "updated_at: 2026-05-21T13:00:00Z",
                                "state: executing",
                                "task: multi-session hardening",
                                "next_action: run multi-session smoke",
                            ]
                        ),
                        "timestamp": 2.0,
                        "meta": {
                            "source": "rolling_thread_checkpoint",
                            "namespace": "active_continuity",
                            "memory_state": "active",
                        },
                    },
                ],
            }
        )
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            args = Namespace(
                storage_dir="/tmp/does-not-matter",
                config_path=None,
                require_gpu=False,
                query="resume continuity",
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                top_k=12,
                fallback_items=3,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_resume(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["latest_checkpoint"]["thread"], "current-session")
            self.assertEqual(payload["latest_checkpoint"]["next_action"], "run multi-session smoke")
            self.assertLess(
                payload["raw_context"].find("current-session"),
                payload["raw_context"].find("old-session"),
            )
        finally:
            mod._adapter = original_adapter


class TestSessionAndHandoffCommands(unittest.TestCase):
    def test_cmd_session_reuses_stable_session_id_for_label(self):
        with tempfile.TemporaryDirectory(prefix="dml-session-test-") as tmp:
            args = Namespace(
                storage_dir=tmp,
                lock_timeout_ms=0,
                label="openclaw-main",
                tenant_id="openclaw",
                session_id=None,
                rotate=False,
            )
            first = io.StringIO()
            with redirect_stdout(first):
                self.assertEqual(mod.cmd_session(args), 0)
            first_payload = json.loads(first.getvalue())

            second = io.StringIO()
            with redirect_stdout(second):
                self.assertEqual(mod.cmd_session(args), 0)
            second_payload = json.loads(second.getvalue())

            self.assertTrue(first_payload["created"])
            self.assertFalse(second_payload["created"])
            self.assertEqual(first_payload["session_id"], second_payload["session_id"])
            self.assertTrue(Path(first_payload["registry_path"]).exists())

    def test_cmd_handoff_emits_structured_continuity_ingest(self):
        original_cmd_ingest = mod.cmd_ingest
        captured = {}

        def fake_ingest(args):
            captured["args"] = args
            return 0

        try:
            mod.cmd_ingest = fake_ingest
            args = Namespace(
                storage_dir="/tmp/does-not-matter",
                config_path=None,
                require_gpu=False,
                lock_timeout_ms=0,
                audit_actor="unit",
                tenant_id="openclaw",
                client_id=None,
                session_id="session-a",
                instance_id=None,
                thread="session-a",
                state="executing",
                task="build handoff",
                next_action="run smoke",
                note="checkpoint note",
                intent=None,
                selected_path=None,
                updated_at="2026-05-21T20:00:00Z",
            )

            self.assertEqual(mod.cmd_handoff(args), 0)
            ingest_args = captured["args"]
            meta = json.loads(ingest_args.meta)

            self.assertIn("[source:rolling_thread_checkpoint]", ingest_args.text)
            self.assertIn("next_action: run smoke", ingest_args.text)
            self.assertEqual(ingest_args.kind, "plan")
            self.assertFalse(ingest_args.chunk)
            self.assertEqual(meta["namespace"], "active_continuity")
            self.assertEqual(meta["session_id"], "session-a")
            self.assertEqual(meta["next_action"], "run smoke")
            self.assertTrue(meta["no_merge"])
            self.assertEqual(meta["merge_policy"], "never")
        finally:
            mod.cmd_ingest = original_cmd_ingest


class TestHealthCommand(unittest.TestCase):
    def _write_state(self, storage_dir: str, *, text: str = "checkpoint", tenant_id: str | None = None) -> Path:
        state_path = Path(storage_dir) / "dml_state.jsonl"
        meta = {
            "source": "rolling_thread_checkpoint",
            "namespace": "active_continuity",
            "summary": "thread: main | next: run tests",
        }
        if tenant_id is not None:
            meta["tenant_id"] = tenant_id
        record = {
            "id": 0,
            "text": text,
            "embedding": [0.1, 0.2, 0.3],
            "timestamp": 1.0,
            "salience": 0.5,
            "fidelity": 1.0,
            "level": 0,
            "summary_of": [0],
            "meta": meta,
        }
        payload_line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        checksum = mod.hashlib.sha256(payload_line.encode("utf-8")).hexdigest()
        header = {
            "type": "daystrom_dml.memory",
            "version": 1,
            "created_at": "2026-05-21T00:00:00+00:00",
            "count": 1,
            "checksum": checksum,
        }
        state_path.write_text(
            json.dumps(header, separators=(",", ":"), sort_keys=True) + "\n" + payload_line,
            encoding="utf-8",
        )
        return state_path

    def test_cmd_health_reports_valid_state_without_backend_probe(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-health-test-") as tmp:
            self._write_state(tmp, tenant_id="openclaw")
            args = Namespace(
                storage_dir=tmp,
                config_path=None,
                require_gpu=False,
                probe_backend=False,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_health(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["contract_version"], "dml-agent-memory-v1")
            self.assertTrue(payload["state"]["checksum_ok"])
            self.assertTrue(payload["state"]["count_ok"])
            self.assertEqual(payload["state"]["embedding_dimensions"], [3])
            self.assertEqual(payload["state"]["active_continuity_count"], 1)
            self.assertEqual(payload["state"]["unscoped_count"], 0)
            self.assertEqual(payload["state"]["records_by_tenant"], {"openclaw": 1})
            self.assertEqual(payload["state"]["active_continuity_by_tenant"], {"openclaw": 1})

    def test_schema_command_reports_supported_state_version(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-schema-test-") as tmp:
            self._write_state(tmp, tenant_id="openclaw")
            args = Namespace(storage_dir=tmp)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_schema(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["state_schema"]["version"], 1)
            self.assertFalse(payload["state_schema"]["migration_required"])

    def test_health_flags_unsupported_state_version(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-schema-bad-") as tmp:
            state_path = self._write_state(tmp, tenant_id="openclaw")
            lines = state_path.read_text(encoding="utf-8").splitlines()
            header = json.loads(lines[0])
            header["version"] = 999
            state_path.write_text(
                json.dumps(header, separators=(",", ":"), sort_keys=True) + "\n" + "\n".join(lines[1:]),
                encoding="utf-8",
            )
            args = Namespace(
                storage_dir=tmp,
                config_path=None,
                require_gpu=False,
                probe_backend=False,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_health(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "degraded")
            self.assertIn("unsupported_state_version: 999", payload["errors"])

    def test_report_summarizes_health_conflicts_and_curation(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-report-") as tmp:
            old_ts = time.time() - 90 * 86400
            _write_state_records(
                tmp,
                [
                    {
                        "id": 1,
                        "text": "old low fidelity memory",
                        "embedding": [0.1, 0.2],
                        "timestamp": old_ts,
                        "salience": 0.1,
                        "fidelity": 0.2,
                        "level": 0,
                        "summary_of": [1],
                        "meta": {"tenant_id": "alpha", "source": "unit", "namespace": "scratch", "memory_state": "active"},
                    },
                    {
                        "id": 2,
                        "text": "claim one",
                        "embedding": [0.2, 0.3],
                        "timestamp": time.time(),
                        "salience": 0.8,
                        "fidelity": 0.9,
                        "level": 0,
                        "summary_of": [2],
                        "meta": {
                            "tenant_id": "alpha",
                            "source": "agent-a",
                            "namespace": "ops",
                            "conflict_key": "deploy_mode",
                            "claim_value": "manual",
                        },
                    },
                    {
                        "id": 3,
                        "text": "claim two",
                        "embedding": [0.3, 0.4],
                        "timestamp": time.time(),
                        "salience": 0.8,
                        "fidelity": 0.9,
                        "level": 0,
                        "summary_of": [3],
                        "meta": {
                            "tenant_id": "alpha",
                            "source": "agent-b",
                            "namespace": "ops",
                            "conflict_key": "deploy_mode",
                            "claim_value": "automatic",
                            "conflict_state": "conflicted",
                        },
                    },
                ],
            )
            args = Namespace(
                storage_dir=tmp,
                tenant_id="alpha",
                conflict_limit=10,
                curation_min_age_days=30.0,
                curation_max_fidelity=0.35,
                curation_limit=10,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_report(args), 0)

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["state"]["record_count"], 3)
            self.assertEqual(payload["conflicts"]["count"], 1)
            self.assertEqual(payload["curation"]["candidate_count"], 1)
            self.assertIn("unresolved_conflicts=1", payload["warnings"])

    def test_cmd_health_fails_missing_state_file(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-health-missing-") as tmp:
            args = Namespace(
                storage_dir=tmp,
                config_path=None,
                require_gpu=False,
                probe_backend=False,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_health(args)

            self.assertEqual(rc, 1)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "fail")
            self.assertIn("state_file_missing", payload["errors"])

    def test_backup_verify_and_restore_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-backup-test-") as tmp:
            storage = Path(tmp) / "store"
            backup_dir = Path(tmp) / "backups"
            storage.mkdir()
            state_path = self._write_state(str(storage), text="original memory")

            backup_args = Namespace(
                storage_dir=str(storage),
                backup_dir=str(backup_dir),
                label="unit",
                keep=20,
                lock_timeout_ms=0,
                audit_actor="unit-test",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_backup(backup_args), 0)
            backup_payload = json.loads(buf.getvalue())
            backup_path = Path(backup_payload["backup"]["backup_dir"])
            self.assertTrue((backup_path / "dml_state.jsonl").exists())
            self.assertTrue((backup_path / "backup_manifest.json").exists())

            state_path.write_text("broken\n", encoding="utf-8")
            verify_args = Namespace(storage_dir=str(storage))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_verify(verify_args), 1)
            self.assertEqual(json.loads(buf.getvalue())["status"], "fail")

            restore_args = Namespace(
                storage_dir=str(storage),
                backup=str(backup_path),
                backup_dir=str(backup_dir),
                keep=20,
                lock_timeout_ms=0,
                audit_actor="unit-test",
                no_pre_restore_backup=False,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_restore(restore_args), 0)
            restored = json.loads(buf.getvalue())
            self.assertEqual(restored["status"], "ok")
            self.assertIsNotNone(restored["pre_restore_backup"])
            self.assertIn("original memory", state_path.read_text(encoding="utf-8"))

            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_verify(verify_args), 0)
            self.assertEqual(json.loads(buf.getvalue())["status"], "ok")

    def test_export_verify_and_import_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-export-test-") as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            exports = Path(tmp) / "exports"
            source.mkdir()
            target.mkdir()
            self._write_state(str(source), text="portable exported memory", tenant_id="openclaw")
            mod._append_audit_event(
                str(source),
                operation="unit",
                status="ok",
                actor="unit-test",
                details={"case": "export"},
            )

            export_args = Namespace(
                storage_dir=str(source),
                output_dir=str(exports),
                label="unit",
                audit_actor="unit-test",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_export(export_args), 0)
            export_payload = json.loads(buf.getvalue())
            bundle = Path(export_payload["export"]["bundle_path"])
            self.assertTrue(bundle.exists())
            self.assertTrue(export_payload["export"]["bundle_sha256"])

            verify_export_args = Namespace(bundle=str(bundle))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_verify_export(verify_export_args), 0)
            self.assertEqual(json.loads(buf.getvalue())["status"], "ok")

            import_args = Namespace(
                storage_dir=str(target),
                bundle=str(bundle),
                backup_dir=str(Path(tmp) / "target-backups"),
                keep=20,
                no_pre_import_backup=False,
                lock_timeout_ms=0,
                audit_actor="unit-test",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_import_bundle(import_args), 0)
            import_payload = json.loads(buf.getvalue())
            self.assertEqual(import_payload["status"], "ok")
            self.assertTrue((target / "dml_state.jsonl").exists())
            self.assertTrue((target / "dml_audit.jsonl").exists())
            self.assertIn("portable exported memory", (target / "dml_state.jsonl").read_text(encoding="utf-8"))

            verify_args = Namespace(storage_dir=str(target))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_verify(verify_args), 0)
            self.assertEqual(json.loads(buf.getvalue())["status"], "ok")

    def test_backup_reports_blocked_when_store_write_lock_is_held(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-lock-test-") as tmp:
            storage = Path(tmp) / "store"
            backup_dir = Path(tmp) / "backups"
            storage.mkdir()
            self._write_state(str(storage), text="locked memory")
            lock_path = mod._lock_file_path(str(storage))
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    mod._lock_metadata_path(str(storage)).write_text(
                        json.dumps({"operation": "unit-test", "pid": os.getpid()}),
                        encoding="utf-8",
                    )
                    args = Namespace(
                        storage_dir=str(storage),
                        backup_dir=str(backup_dir),
                        label="blocked",
                        keep=20,
                        lock_timeout_ms=0,
                        audit_actor="unit-test",
                    )
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = mod.cmd_backup(args)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            self.assertEqual(rc, 2)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(payload["error"], "store_write_lock_held")
            self.assertEqual(payload["lock"]["holder"]["operation"], "unit-test")
            audit_events = mod._tail_audit_events(str(storage), limit=1)
            self.assertEqual(audit_events[0]["operation"], "backup")
            self.assertEqual(audit_events[0]["status"], "blocked")

    def test_audit_tail_reports_events_without_raw_text(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-audit-test-") as tmp:
            secret_text = "secret raw memory text should not appear in audit"
            event = mod._append_audit_event(
                tmp,
                operation="ingest",
                status="ok",
                actor="unit-test",
                details={"text_sha256": mod._text_digest(secret_text), "scope": {"tenant_id": "alpha"}},
            )
            self.assertTrue(Path(event["path"]).exists())
            buf = io.StringIO()
            args = Namespace(storage_dir=tmp, limit=5)
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_audit_tail(args), 0)

            payload_text = buf.getvalue()
            self.assertNotIn(secret_text, payload_text)
            payload = json.loads(payload_text)
            self.assertEqual(payload["audit"]["event_count"], 1)
            self.assertEqual(payload["events"][0]["operation"], "ingest")
            self.assertEqual(payload["events"][0]["details"]["scope"]["tenant_id"], "alpha")

    def test_conflicts_command_lists_scoped_claim_groups(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-conflict-list-") as tmp:
            _write_state_records(
                tmp,
                [
                    {
                        "id": 1,
                        "text": "Deploy mode is automatic.",
                        "embedding": [0.1, 0.2],
                        "timestamp": 1.0,
                        "salience": 0.5,
                        "fidelity": 1.0,
                        "level": 0,
                        "summary_of": [1],
                        "meta": {
                            "tenant_id": "alpha",
                            "namespace": "ops",
                            "source": "agent-a",
                            "conflict_key": "deploy_mode",
                            "claim_value": "automatic",
                        },
                    },
                    {
                        "id": 2,
                        "text": "Deploy mode is manual.",
                        "embedding": [0.2, 0.3],
                        "timestamp": 2.0,
                        "salience": 0.5,
                        "fidelity": 1.0,
                        "level": 0,
                        "summary_of": [2],
                        "meta": {
                            "tenant_id": "alpha",
                            "namespace": "ops",
                            "source": "agent-b",
                            "conflict_key": "deploy_mode",
                            "claim_value": "manual",
                            "conflict_state": "conflicted",
                        },
                    },
                ],
            )
            args = Namespace(
                storage_dir=tmp,
                tenant_id="alpha",
                client_id=None,
                session_id=None,
                instance_id=None,
                namespace="ops",
                conflict_key="deploy_mode",
                limit=10,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_conflicts(args), 0)

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["conflict_group_count"], 1)
            values = payload["conflicts"][0]["values"]
            self.assertIn("automatic", values)
            self.assertIn("manual", values)

    def test_resolve_conflict_accepts_one_value_and_suppresses_others(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-conflict-resolve-") as tmp:
            _write_state_records(
                tmp,
                [
                    {
                        "id": 1,
                        "text": "Deploy mode is automatic.",
                        "embedding": [0.1, 0.2],
                        "timestamp": 1.0,
                        "salience": 0.5,
                        "fidelity": 1.0,
                        "level": 0,
                        "summary_of": [1],
                        "meta": {
                            "tenant_id": "alpha",
                            "namespace": "ops",
                            "source": "agent-a",
                            "conflict_key": "deploy_mode",
                            "claim_value": "automatic",
                        },
                    },
                    {
                        "id": 2,
                        "text": "Deploy mode is manual.",
                        "embedding": [0.2, 0.3],
                        "timestamp": 2.0,
                        "salience": 0.5,
                        "fidelity": 1.0,
                        "level": 0,
                        "summary_of": [2],
                        "meta": {
                            "tenant_id": "alpha",
                            "namespace": "ops",
                            "source": "agent-b",
                            "conflict_key": "deploy_mode",
                            "claim_value": "manual",
                            "conflict_state": "conflicted",
                            "conflicts_with": [{"id": 1, "claim_value": "automatic"}],
                        },
                    },
                ],
            )
            args = Namespace(
                storage_dir=tmp,
                tenant_id="alpha",
                client_id=None,
                session_id=None,
                instance_id=None,
                namespace="ops",
                conflict_key="deploy_mode",
                accept_value="manual",
                lock_timeout_ms=0,
                audit_actor="unit-test",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_resolve_conflict(args), 0)

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["accepted"], 1)
            self.assertEqual(payload["suppressed"], 1)
            records = mod._iter_state_records(tmp)
            by_value = {record["meta"]["claim_value"]: record["meta"] for record in records}
            self.assertEqual(by_value["manual"]["memory_state"], "active")
            self.assertNotIn("conflict_state", by_value["manual"])
            self.assertEqual(by_value["automatic"]["memory_state"], "suppressed")
            self.assertEqual(by_value["automatic"]["conflict_resolution"]["accepted_value"], "manual")
            audit_events = mod._tail_audit_events(tmp, limit=1)
            self.assertEqual(audit_events[0]["operation"], "resolve-conflict")
            self.assertEqual(audit_events[0]["status"], "ok")

    def test_curate_dry_run_reports_candidates_without_raw_text(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-curate-dry-") as tmp:
            old_ts = time.time() - 90 * 86400
            raw_text = "old low fidelity raw memory should not appear"
            _write_state_records(
                tmp,
                [
                    {
                        "id": 1,
                        "text": raw_text,
                        "embedding": [0.1, 0.2],
                        "timestamp": old_ts,
                        "salience": 0.1,
                        "fidelity": 0.2,
                        "level": 0,
                        "summary_of": [1],
                        "meta": {"tenant_id": "alpha", "source": "unit", "namespace": "scratch", "memory_state": "active"},
                    },
                    {
                        "id": 2,
                        "text": "new memory",
                        "embedding": [0.3, 0.4],
                        "timestamp": time.time(),
                        "salience": 0.8,
                        "fidelity": 0.9,
                        "level": 0,
                        "summary_of": [2],
                        "meta": {"tenant_id": "alpha", "source": "unit", "namespace": "scratch", "memory_state": "active"},
                    },
                ],
            )
            args = Namespace(
                storage_dir=tmp,
                tenant_id="alpha",
                namespace="scratch",
                source=None,
                state=None,
                min_age_days=30.0,
                max_fidelity=0.35,
                limit=10,
                action="suppressed",
                reason="unit",
                include_continuity=False,
                apply=False,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_curate(args), 0)

            payload_text = buf.getvalue()
            payload = json.loads(payload_text)
            self.assertEqual(payload["status"], "dry-run")
            self.assertEqual(payload["candidate_count"], 1)
            self.assertEqual(payload["candidates"][0]["id"], 1)
            self.assertNotIn(raw_text, payload_text)
            records = mod._iter_state_records(tmp)
            self.assertEqual(records[0]["meta"]["memory_state"], "active")

    def test_curate_apply_suppresses_candidates_and_audits(self):
        with tempfile.TemporaryDirectory(prefix="dml-wrapper-curate-apply-") as tmp:
            old_ts = time.time() - 90 * 86400
            _write_state_records(
                tmp,
                [
                    {
                        "id": 1,
                        "text": "old low fidelity memory",
                        "embedding": [0.1, 0.2],
                        "timestamp": old_ts,
                        "salience": 0.1,
                        "fidelity": 0.2,
                        "level": 0,
                        "summary_of": [1],
                        "meta": {"tenant_id": "alpha", "source": "unit", "namespace": "scratch", "memory_state": "active"},
                    },
                    {
                        "id": 2,
                        "text": "continuity should be protected",
                        "embedding": [0.3, 0.4],
                        "timestamp": old_ts,
                        "salience": 0.1,
                        "fidelity": 0.1,
                        "level": 0,
                        "summary_of": [2],
                        "meta": {
                            "tenant_id": "alpha",
                            "source": "rolling_thread_checkpoint",
                            "namespace": "active_continuity",
                            "memory_state": "active",
                        },
                    },
                ],
            )
            args = Namespace(
                storage_dir=tmp,
                tenant_id="alpha",
                namespace=None,
                source=None,
                state=None,
                min_age_days=30.0,
                max_fidelity=0.35,
                limit=10,
                action="suppressed",
                reason="unit",
                include_continuity=False,
                apply=True,
                lock_timeout_ms=0,
                audit_actor="unit-test",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mod.cmd_curate(args), 0)

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["changed"], 1)
            records = mod._iter_state_records(tmp)
            states = {record["id"]: record["meta"]["memory_state"] for record in records}
            self.assertEqual(states[1], "suppressed")
            self.assertEqual(states[2], "active")
            audit_events = mod._tail_audit_events(tmp, limit=1)
            self.assertEqual(audit_events[0]["operation"], "curate")
            self.assertEqual(audit_events[0]["details"]["changed"], 1)


class TestIngestBatching(unittest.TestCase):
    def test_cmd_ingest_defers_persistence_until_chunks_finish(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter()
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            storage_dir = tempfile.mkdtemp(prefix="dml-wrapper-ingest-test-")
            args = Namespace(
                storage_dir=storage_dir,
                config_path=None,
                require_gpu=False,
                lock_timeout_ms=0,
                audit_actor="unit-test",
                tenant_id="tenant-ingest",
                client_id="client-ingest",
                session_id="session-ingest",
                instance_id="instance-ingest",
                text=" ".join(f"Durable memory sentence {idx} with useful context." for idx in range(80)),
                kind="note",
                meta=None,
                chunk=True,
                chunk_chars=24,
                chunk_overlap=0,
                filter_noise=False,
                summary_policy="llm",
                summary_max_chars=220,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_ingest(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertGreater(len(adapter.ingests), 1)
            self.assertTrue(all(call[2] is False for call in adapter.ingests))
            self.assertEqual(adapter.persist_calls, 1)
            self.assertGreater(payload["llm_summaries_allowed"], 0)
            _, first_meta, _ = adapter.ingests[0]
            self.assertEqual(first_meta["tenant_id"], "tenant-ingest")
            self.assertEqual(first_meta["client_id"], "client-ingest")
            self.assertEqual(first_meta["session_id"], "session-ingest")
            self.assertEqual(first_meta["instance_id"], "instance-ingest")
        finally:
            mod._adapter = original_adapter

    def test_cmd_ingest_marks_scoped_claim_conflict(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter()
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            storage_dir = tempfile.mkdtemp(prefix="dml-wrapper-conflict-test-")
            _write_state_records(
                storage_dir,
                [
                    {
                        "id": 42,
                        "text": "Deploy mode is automatic.",
                        "embedding": [0.1, 0.2, 0.3],
                        "timestamp": 1.0,
                        "salience": 0.5,
                        "fidelity": 1.0,
                        "level": 0,
                        "summary_of": [42],
                        "meta": {
                            "tenant_id": "tenant-conflict",
                            "session_id": "session-conflict",
                            "namespace": "ops",
                            "source": "agent-a",
                            "conflict_key": "deploy_mode",
                            "claim_value": "automatic",
                        },
                    }
                ],
            )
            args = Namespace(
                storage_dir=storage_dir,
                config_path=None,
                require_gpu=False,
                lock_timeout_ms=0,
                audit_actor="unit-test",
                tenant_id="tenant-conflict",
                client_id=None,
                session_id="session-conflict",
                instance_id=None,
                text="Deploy mode is manual.",
                kind="note",
                meta=json.dumps(
                    {
                        "source": "agent-b",
                        "namespace": "ops",
                        "conflict_key": "deploy_mode",
                        "claim_value": "manual",
                    }
                ),
                chunk=False,
                chunk_chars=620,
                chunk_overlap=90,
                filter_noise=False,
                summary_policy="skip",
                summary_max_chars=220,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_ingest(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["conflicts_detected"], 1)
            _, meta, _ = adapter.ingests[0]
            self.assertEqual(meta["conflict_state"], "conflicted")
            self.assertEqual(meta["conflicts_with"][0]["id"], 42)
            self.assertEqual(meta["conflicts_with"][0]["claim_value"], "automatic")
        finally:
            mod._adapter = original_adapter

    def test_cmd_ingest_records_dpm_preference_when_marked_explicit(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter()
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            storage_dir = tempfile.mkdtemp(prefix="dml-wrapper-dpm-test-")
            args = Namespace(
                storage_dir=storage_dir,
                config_path=None,
                require_gpu=False,
                lock_timeout_ms=0,
                audit_actor="unit-test",
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                text="I prefer wrapper smoke status updates to be concise.",
                kind="note",
                meta=json.dumps({"source": "wrapper-smoke", "dpm_preference": True}),
                chunk=False,
                chunk_chars=620,
                chunk_overlap=90,
                filter_noise=False,
                summary_policy="auto",
                summary_max_chars=220,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_ingest(args)

            self.assertEqual(rc, 0)
            self.assertEqual(len(adapter.preferences), 1)
            text, kwargs = adapter.preferences[0]
            self.assertIn("wrapper smoke status", text)
            self.assertEqual(kwargs["source_id"], "wrapper-smoke")
            self.assertTrue(kwargs["explicit"])
        finally:
            mod._adapter = original_adapter

    def test_cmd_ingest_auto_summary_uses_deterministic_continuity_summary(self):
        original_adapter = mod._adapter
        adapter = _DummyAdapter()
        try:
            mod._adapter = lambda *_args, **_kwargs: adapter
            storage_dir = tempfile.mkdtemp(prefix="dml-wrapper-summary-test-")
            args = Namespace(
                storage_dir=storage_dir,
                config_path=None,
                require_gpu=False,
                lock_timeout_ms=0,
                audit_actor="unit-test",
                tenant_id="openclaw",
                client_id=None,
                session_id=None,
                instance_id=None,
                text="\n".join(
                    [
                        "[source:rolling_thread_checkpoint]",
                        "thread: main",
                        "state: executing",
                        "task: reduce intake summarization cost",
                        "next_action: run smoke tests",
                    ]
                ),
                kind="plan",
                meta=json.dumps(
                    {
                        "source": "rolling_thread_checkpoint",
                        "namespace": "active_continuity",
                        "thread": "main",
                        "state": "executing",
                        "task": "reduce intake summarization cost",
                        "next_action": "run smoke tests",
                    }
                ),
                chunk=False,
                chunk_chars=620,
                chunk_overlap=90,
                filter_noise=False,
                summary_policy="auto",
                summary_max_chars=220,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.cmd_ingest(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["cheap_summaries"], 1)
            self.assertEqual(payload["llm_summaries_allowed"], 0)
            _, meta, _ = adapter.ingests[0]
            self.assertEqual(meta["summary_source"], "deterministic")
            self.assertEqual(meta["tenant_id"], "openclaw")
            self.assertIn("task: reduce intake summarization cost", meta["summary"])
            self.assertIn("next: run smoke tests", meta["summary"])
        finally:
            mod._adapter = original_adapter


if __name__ == "__main__":
    unittest.main()
