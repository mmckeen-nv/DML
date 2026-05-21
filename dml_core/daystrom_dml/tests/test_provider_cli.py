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
    assert json.loads(capsys.readouterr().out)["written_to"] == str(output)
