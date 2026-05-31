from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from daystrom_dml import server  # noqa: E402


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a TestClient with visualizer auto-launch disabled."""

    monkeypatch.setattr(server, "VISUALIZER_URL", "http://example.com")
    return TestClient(server.app)


def test_ingest_endpoint_invokes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | None]] = []

        def ingest(self, text: str, meta: dict | None = None) -> None:
            self.calls.append((text, meta))

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        response = client.post(
            "/ingest",
            json={"text": "capture this", "meta": {"source": "unit-test"}},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert stub.calls == [("capture this", {"source": "unit-test"})]


def test_reinforce_endpoint_invokes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict | None]] = []

        def reinforce(
            self, prompt: str, response: str, meta: dict | None = None
        ) -> None:
            self.calls.append((prompt, response, meta))

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        response = client.post(
            "/reinforce",
            json={"text": "keep this", "meta": {"tags": ["test"]}},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert stub.calls == [("", "keep this", {"tags": ["test"]})]


def test_dpm_overlay_endpoint_invokes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def personality_overlay(self, **kwargs) -> dict:
            self.calls.append(kwargs)
            return {"overlay": {"rendered_text": "matrix active"}}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        response = client.get(
            "/dpm/overlay",
            params={
                "prompt": "hello",
                "thread_id": "thread:one",
                "project_id": "project:one",
                "relationship_id": "relationship:one",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "overlay": {"overlay": {"rendered_text": "matrix active"}}}
    assert stub.calls == [
        {
            "prompt": "hello",
            "thread_id": "thread:one",
            "project_id": "project:one",
            "relationship_id": "relationship:one",
        }
    ]


def test_dpm_preference_endpoint_records_and_governs_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.record_calls: list[tuple] = []

        def personality_graph(self) -> dict:
            return {"nodes": [{"id": "pref.concise"}]}

        def record_personality_preference(self, text: str, **kwargs) -> dict:
            self.record_calls.append((text, kwargs))
            return {"status": "recorded", "node_id": "pref.concise"}

        def suppress_personality_preference(self, node_id: str, *, reason: str) -> dict:
            return {"status": "suppressed", "node_id": node_id, "reason": reason}

        def delete_personality_preference(self, node_id: str) -> dict:
            return {"status": "deleted", "node_id": node_id}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        graph_response = client.get("/dpm/graph")
        record_response = client.post(
            "/dpm/preference",
            json={
                "text": "I prefer concise updates.",
                "scope": "project",
                "source_id": "turn:test",
                "explicit": True,
                "meta": {"project_id": "project:test"},
            },
        )
        suppress_response = client.post(
            "/dpm/preference/pref.concise/suppress",
            json={"reason": "user_changed_preference"},
        )
        delete_response = client.delete("/dpm/preference/pref.concise")

    assert graph_response.json() == {"status": "ok", "graph": {"nodes": [{"id": "pref.concise"}]}}
    assert record_response.json()["status"] == "recorded"
    assert suppress_response.json()["status"] == "suppressed"
    assert delete_response.json()["status"] == "deleted"
    assert stub.record_calls == [
        (
            "I prefer concise updates.",
            {
                "scope": "project",
                "source_id": "turn:test",
                "explicit": True,
                "meta": {"project_id": "project:test"},
            },
        )
    ]


def test_dpm_preference_endpoint_reports_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def record_personality_preference(self, text: str, **kwargs) -> None:
            return None

    monkeypatch.setattr(server, "adapter", StubAdapter())

    with _client(monkeypatch) as client:
        response = client.post("/dpm/preference", json={"text": "I prefer concise updates."})

    assert response.status_code == 200
    assert response.json() == {"status": "inactive", "result": None}


def test_query_endpoint_uses_context_and_records_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubRunner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "final answer"

    class StubAdapter:
        def __init__(self) -> None:
            self.runner = StubRunner()
            self.metrics_enabled = True
            self.enable_stm_controller = False
            self.retrieval_prompts: list[str] = []
            self.reinforcements: list[tuple[str, str]] = []

        def query_database(self, prompt: str) -> dict:
            self.retrieval_prompts.append(prompt)
            return {
                "mode": "literal",
                "context": "System context",
                "tokens": 3,
                "latency_ms": 17,
            }

        def reinforce(self, prompt: str, response: str) -> None:
            self.reinforcements.append((prompt, response))

        def stats(self) -> dict:
            return {"memories": 5}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    token_inputs: list[str] = []
    monkeypatch.setattr(
        server.utils,
        "estimate_tokens",
        lambda text: token_inputs.append(text) or 10,
    )

    recorded: list[tuple[int, int]] = []

    def record_tokens(consumed: int, saved: int) -> None:
        recorded.append((consumed, saved))

    monkeypatch.setattr(server, "record_tokens", record_tokens)

    with _client(monkeypatch) as client:
        response = client.post("/query", json={"prompt": "Explain warp drive"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "mode": "literal",
        "context": "System context",
        "response": "final answer",
        "stats": {"memories": 5},
    }
    assert stub.retrieval_prompts == ["Explain warp drive"]
    assert stub.runner.prompts == ["System context\n\nExplain warp drive"]
    assert stub.reinforcements == [("Explain warp drive", "final answer")]
    assert token_inputs == ["Explain warp drive"]
    assert recorded == [(13, 7)]


def test_rag_compare_translates_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubAdapter:
        def compare_responses(
            self, prompt: str, *, top_k: int | None = None, max_new_tokens: int
        ) -> dict:
            raise requests.RequestException("network down")

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post("/rag/compare", json={"prompt": "Check"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "NIM backend is unreachable" in detail
    assert queue_calls == []


def test_rag_compare_success_queues_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.config = {"top_k": 4}
            self.calls: list[tuple[str, int | None, int]] = []

        def compare_responses(
            self,
            prompt: str,
            *,
            top_k: int | None = None,
            max_new_tokens: int,
            allow_reinforce: bool = True,
        ) -> dict:
            self.calls.append((prompt, top_k, max_new_tokens, allow_reinforce))
            return {"candidates": ["ok"], "dml": {"entries": [{"id": 42, "summary": "matched"}]}}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    token_inputs: list[str] = []
    monkeypatch.setattr(
        server.utils,
        "estimate_tokens",
        lambda text: token_inputs.append(text) or 12,
    )

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post(
            "/rag/compare",
            json={"prompt": "Assemble", "top_k": 3, "max_new_tokens": 1024},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "candidates": ["ok"],
        "dml": {"entries": [{"id": 42, "summary": "matched"}]},
        "prompt_tokens_est": 12,
    }
    assert stub.calls == [("Assemble", 3, 1024, False)]
    assert token_inputs == ["Assemble"]
    assert queue_calls == [
        (
            ("Assemble",),
            {
                "top_k": 3,
                "mode": "auto",
                "metadata": {
                    "source": "rag_compare",
                    "activated_node_ids": [42],
                    "activated_nodes": [{"id": 42, "summary": "matched"}],
                },
            },
        )
    ]


def test_rag_retrieve_collects_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubRagStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def report_all(self, prompt: str, top_k: int) -> dict:
            self.calls.append((prompt, top_k))
            return {"faiss": ["result"]}

    class StubAdapter:
        def __init__(self) -> None:
            self.config = {"top_k": 7}
            self.rag_store = StubRagStore()
            self.reports: list[str] = []

        def retrieval_report(self, prompt: str) -> dict:
            self.reports.append(prompt)
            return {"mode": "literal"}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post("/rag/retrieve", json={"prompt": "Gather data"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "prompt": "Gather data",
        "rag_backends": {"faiss": ["result"]},
        "dml": {"mode": "literal"},
    }
    assert stub.rag_store.calls == [("Gather data", 7)]
    assert stub.reports == ["Gather data"]
    assert queue_calls == [
        (
            ("Gather data",),
            {
                "top_k": 7,
                "mode": "auto",
                "metadata": {"source": "rag_retrieve"},
            },
        )
    ]


def test_pipeline_page_serves_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        response = client.get("/pipeline")

    assert response.status_code == 200
    assert "Daystrom Inference Pipeline" in response.text


def test_inference_prepare_builds_frontier_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def retrieve_context(self, prompt: str, **kwargs) -> dict:
            return {
                "raw_context": "=== Retrieved Context ===\nSURVIVAL-ANCHOR-123 is active.",
                "context_tokens": 12,
                "items": [{"id": "1", "summary": "SURVIVAL-ANCHOR-123"}],
                "survival_ledger_included": True,
            }

    monkeypatch.setattr(server, "adapter", StubAdapter())
    monkeypatch.delenv("DML_NVIDIA_API_KEY", raising=False)

    with _client(monkeypatch) as client:
        response = client.post(
            "/inference/prepare",
            json={
                "prompt": "What anchor is active?",
                "direct_input_tokens_estimate": 1000,
                "direct_output_tokens_estimate": 900,
                "frontier_max_tokens": 420,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_configured"] is False
    assert payload["frontier_request"]["model"] == "azure/openai/gpt-5.2-codex"
    assert "SURVIVAL-ANCHOR-123" in payload["frontier_prompt"]
    assert payload["telemetry"]["input_tokens_saved_estimate"] > 0
    assert payload["telemetry"]["output_tokens_saved_estimate"] == 480


def test_flappy_bird_scenario_seeds_canned_code_agent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | None]] = []

        def ingest(self, text: str, meta: dict | None = None) -> None:
            self.calls.append((text, meta))

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)
    server.SEEDED_INFERENCE_SCENARIOS.discard("flappy-bird-code-agent")

    with _client(monkeypatch) as client:
        response = client.post("/inference/scenarios/flappy-bird")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "flappy-bird-canned-demo"
    assert payload["traditional_turns"] == 16
    assert payload["dml_turns"] == 3
    assert payload["direct_input_tokens_estimate"] > 0
    assert "Flappy Bird" in payload["prompt"]
    assert "Traditional agent turn 16" in payload["direct_prompt"]
    assert len(stub.calls) == payload["memory_count"]
    assert any(call[1]["kind"] == "survival_ledger" for call in stub.calls)


def test_inference_run_requires_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def retrieve_context(self, prompt: str, **kwargs) -> dict:
            return {"raw_context": "Context", "context_tokens": 2, "items": []}

    monkeypatch.setattr(server, "adapter", StubAdapter())
    monkeypatch.delenv("DML_NVIDIA_API_KEY", raising=False)

    with _client(monkeypatch) as client:
        response = client.post("/inference/run", json={"prompt": "Spend?"})

    assert response.status_code == 400
    assert "DML_NVIDIA_API_KEY" in response.json()["detail"]


def test_inference_direct_run_requires_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DML_NVIDIA_API_KEY", raising=False)

    with _client(monkeypatch) as client:
        response = client.post("/inference/direct/run", json={"prompt": "Build code directly"})

    assert response.status_code == 400
    assert "DML_NVIDIA_API_KEY" in response.json()["detail"]
