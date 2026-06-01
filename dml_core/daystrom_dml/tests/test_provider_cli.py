from __future__ import annotations

import json

import httpx

from daystrom_dml import provider_cli


def _transport(handler):
    return httpx.MockTransport(handler)


def test_provider_cli_status_prints_health(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "provider": "daystrom-dml"})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["status"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_provider_cli_recall_context_only(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/recall"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["query"] == "hello"
        return httpx.Response(200, json={"raw_context": "context block", "items": []})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["recall", "--query", "hello", "--context-only"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "context block"


def test_provider_cli_dcn_eval_smoke_is_readiness_gate(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/dcn/eval/smoke"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "component": "daystrom-cognition-network",
                "mode": "offline_fixture_smoke",
                "report": {"passed": True, "summary": {"case_count": 3, "blocked_polluting_items": 1}},
            },
        )

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["dcn", "eval-smoke"])

    rendered = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(rendered)
    assert payload["mode"] == "offline_fixture_smoke"
    assert payload["report"]["passed"] is True


def test_provider_cli_dcn_eval_smoke_fails_closed(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/dcn/eval/smoke"
        return httpx.Response(200, json={"status": "failed", "report": {"passed": False}})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["dcn", "readiness"])

    assert rc == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_provider_cli_install_app_writes_profile(tmp_path, capsys) -> None:
    output = tmp_path / "hermes-dml.json"
    rc = provider_cli.main(
        [
            "install-app",
            "--app",
            "hermes",
            "--storage-dir",
            "/tmp/dml-store",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["app"] == "hermes"
    assert written["environment"]["HERMES_MEMORY_PROVIDER"] == "daystrom-dml"
    assert written["commands"]["dcn_eval_smoke"] == "dml dcn eval-smoke"
    assert written["endpoints"]["dcn_eval_smoke"] == "http://127.0.0.1:8765/api/dcn/eval/smoke"
    assert json.loads(capsys.readouterr().out)["written_to"] == str(output)
