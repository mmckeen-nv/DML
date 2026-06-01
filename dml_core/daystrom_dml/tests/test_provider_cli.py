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


def test_provider_cli_dcn_eval_smoke_is_readiness_gate(tmp_path, capsys, monkeypatch) -> None:
    real_client = httpx.Client
    artifact_path = tmp_path / "dcn-eval-artifact.json"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/dcn/eval/smoke"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "component": "daystrom-cognition-network",
                "mode": "offline_fixture_smoke",
                "report": {"passed": True, "summary": {"case_count": 9, "blocked_polluting_items": 3}},
                "artifact": {
                    "schema_version": "dcn-eval-artifact-v1",
                    "artifact_hash": "artifact123",
                    "summary": {"case_count": 9},
                    "readiness": {"ready": True, "failed_gates": [], "gate_count": 15},
                },
            },
        )

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["dcn", "eval-smoke", "--output", str(artifact_path), "--artifact-only"])

    rendered = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(rendered)
    assert payload["mode"] == "offline_fixture_smoke"
    assert payload["report"]["passed"] is True
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "dcn-eval-artifact-v1"
    assert written["artifact_hash"] == "artifact123"
    assert written["readiness"]["ready"] is True


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
    assert written["commands"]["dcn_eval_smoke"] == "dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only"
    assert written["commands"]["dcn_seed_trial"] == "dml dcn seed-trial --input sanitized-feedback.json --output dcn-seed-trial-artifact.json"
    assert written["commands"]["dcn_seed_propose"] == "dml dcn seed-propose --input sanitized-feedback.json --output dcn-seed-proposal.json"
    assert written["commands"]["dcn_seed_loop"] == "dml dcn seed-loop --input sanitized-feedback.json --output dcn-seed-loop-artifact.json"
    assert written["endpoints"]["dcn_eval_smoke"] == "http://127.0.0.1:8765/api/dcn/eval/smoke"
    assert json.loads(capsys.readouterr().out)["written_to"] == str(output)


def test_provider_cli_dcn_observe_posts_scoped_request(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/dcn/observe"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["content"] == "continue sprint"
        assert payload["scope"]["tenant_id"] == "openclaw"
        assert payload["scope"]["session_id"] == "s1"
        assert payload["metadata"] == {"compaction_resume": True}
        assert payload["constraints"]["allow_tools"] is False
        return httpx.Response(200, json={"status": "ok", "plan": {"decision_id": "d1"}})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main([
        "dcn",
        "observe",
        "--text",
        "continue sprint",
        "--session-id",
        "s1",
        "--metadata",
        '{"compaction_resume": true}',
        "--no-tools",
    ])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["plan"]["decision_id"] == "d1"


def test_provider_cli_dcn_packet_posts_scoped_request(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/dcn/cognitive-packet"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["content"] == "continue provider memory work"
        assert payload["constraints"]["max_memory_tokens"] == 333
        return httpx.Response(200, json={"status": "ok", "packet": {"packet_version": "daystrom-cognitive-packet-v1"}})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main(["dcn", "packet", "--text", "continue provider memory work", "--max-memory-tokens", "333"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["packet"]["packet_version"] == "daystrom-cognitive-packet-v1"


def test_provider_cli_dcn_seed_trial_writes_non_promoting_artifact(tmp_path, capsys) -> None:
    input_path = tmp_path / "seed-input.json"
    output_path = tmp_path / "seed-artifact.json"
    input_path.write_text(
        json.dumps({
            "candidate_updates": [
                {"task_type": "debugging", "field": "verification_requirement", "value": "strict"}
            ],
            "unsupported_policy_pressure": [
                {"task_type": "debugging", "needed_capability": "tool_sequence_policy"}
            ],
        }),
        encoding="utf-8",
    )

    rc = provider_cli.main(["dcn", "seed-trial", "--input", str(input_path), "--output", str(output_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dcn-seed-trial-artifact-v1"
    assert payload["non_promoting"] is True
    assert payload["summary"]["accepted_update_count"] == 1
    assert payload["summary"]["unsupported_policy_pressure_count"] == 1
    assert written["artifact_hash"] == payload["artifact_hash"]


def test_provider_cli_dcn_seed_propose_and_loop_write_active_artifacts(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "feedback.json"
    proposal_path = tmp_path / "proposal.json"
    loop_path = tmp_path / "loop.json"
    input_path.write_text(json.dumps({"feedback": [{"decision_id": "d1", "outcome": "verified"}]}), encoding="utf-8")

    def fake_propose(payload, *, model, ollama_base_url, timeout):
        assert payload["feedback"][0]["decision_id"] == "d1"
        assert model == "llama3:8b"
        return {
            "schema_version": "dcn-seed-proposal-v1",
            "proposal_hash": "p1",
            "non_promoting": True,
            "candidate_updates": [],
            "unsupported_policy_pressure": [],
        }

    def fake_loop(payload, *, model, ollama_base_url, timeout):
        return {
            "schema_version": "dcn-seed-loop-artifact-v1",
            "artifact_hash": "l1",
            "non_promoting": True,
            "proposal": {"proposal_hash": "p1"},
            "trial": {"artifact_hash": "t1"},
        }

    monkeypatch.setattr(provider_cli, "propose_seed_updates", fake_propose)
    monkeypatch.setattr(provider_cli, "run_seed_loop", fake_loop)

    assert provider_cli.main(["dcn", "seed-propose", "--input", str(input_path), "--output", str(proposal_path)]) == 0
    assert json.loads(capsys.readouterr().out)["proposal_hash"] == "p1"
    assert json.loads(proposal_path.read_text(encoding="utf-8"))["proposal_hash"] == "p1"

    assert provider_cli.main(["dcn", "seed-loop", "--input", str(input_path), "--output", str(loop_path)]) == 0
    assert json.loads(capsys.readouterr().out)["artifact_hash"] == "l1"
    assert json.loads(loop_path.read_text(encoding="utf-8"))["schema_version"] == "dcn-seed-loop-artifact-v1"


def test_provider_cli_dcn_feedback_and_audit_tail(capsys, monkeypatch) -> None:
    real_client = httpx.Client
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/dcn/feedback":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["decision_id"] == "d1"
            assert payload["outcome"] == "verified"
            assert payload["signals"] == {"tests_passed": True}
            return httpx.Response(200, json={"status": "ok", "accepted": True})
        assert request.url.path == "/api/dcn/audit"
        assert request.url.params["limit"] == "7"
        return httpx.Response(200, json={"status": "ok", "count": 1, "entries": [{"decision_id": "d1"}]})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    assert provider_cli.main(["dcn", "feedback", "--decision-id", "d1", "--outcome", "verified", "--signals", '{"tests_passed": true}']) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert provider_cli.main(["dcn", "audit-tail", "--limit", "7"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    assert seen == ["/api/dcn/feedback", "/api/dcn/audit"]


def test_provider_cli_dcn_policy_show_export_import(tmp_path, capsys, monkeypatch) -> None:
    real_client = httpx.Client
    snapshot_path = tmp_path / "policy.json"
    snapshot = {"schema_version": "dcn-procedural-learning-v1", "base_policy_ref": "dcn-policy-v0", "mutable_overlay": {}}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/dcn/policy" and request.method == "GET":
            return httpx.Response(200, json={"status": "ok", "policy_version": "dcn-policy-v0"})
        if request.url.path == "/api/dcn/policy/export":
            return httpx.Response(200, json={"status": "ok", "snapshot": snapshot})
        assert request.url.path == "/api/dcn/policy/import"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["snapshot"] == snapshot
        return httpx.Response(200, json={"status": "ok", "imported": True, "profiles": 0})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    assert provider_cli.main(["dcn", "policy", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["policy_version"] == "dcn-policy-v0"
    assert provider_cli.main(["dcn", "policy", "export", "--output", str(snapshot_path), "--snapshot-only"]) == 0
    assert json.loads(capsys.readouterr().out)["snapshot"] == snapshot
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == snapshot
    assert provider_cli.main(["dcn", "policy", "import", "--input", str(snapshot_path)]) == 0
    assert json.loads(capsys.readouterr().out)["imported"] is True
    assert seen == ["/api/dcn/policy", "/api/dcn/policy/export", "/api/dcn/policy/import"]


def test_provider_cli_dcn_policy_checkpoint_list_and_rollback(capsys, monkeypatch) -> None:
    real_client = httpx.Client
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/dcn/policy/checkpoint":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {"label": "before-active-learn"}
            return httpx.Response(200, json={"status": "ok", "checkpoint_id": "cp1", "label": "before-active-learn"})
        if request.url.path == "/api/dcn/policy/checkpoints":
            return httpx.Response(200, json={"status": "ok", "count": 1, "checkpoints": [{"checkpoint_id": "cp1"}]})
        assert request.url.path == "/api/dcn/policy/rollback"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {"checkpoint_id": "cp1"}
        return httpx.Response(200, json={"status": "ok", "rolled_back": True, "checkpoint_id": "cp1"})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    assert provider_cli.main(["dcn", "policy", "checkpoint", "--label", "before-active-learn"]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_id"] == "cp1"
    assert provider_cli.main(["dcn", "policy", "checkpoints"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    assert provider_cli.main(["dcn", "policy", "rollback", "--checkpoint-id", "cp1"]) == 0
    assert json.loads(capsys.readouterr().out)["rolled_back"] is True
    assert seen == [
        ("POST", "/api/dcn/policy/checkpoint"),
        ("GET", "/api/dcn/policy/checkpoints"),
        ("POST", "/api/dcn/policy/rollback"),
    ]


def test_provider_cli_dcn_promote_and_promotions(capsys, monkeypatch) -> None:
    real_client = httpx.Client
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/dcn/mode/promote":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {
                "target_mode": "active_learn",
                "checkpoint_id": "cp1",
                "hygiene_evidence": {"passed": True, "artifact_hash": "abc123"},
                "operator": "pytest",
                "reason": "promotion test",
            }
            return httpx.Response(200, json={"status": "ok", "promoted": True, "runtime_mode": "active_learn", "audit": {"promotion_id": "p1"}})
        assert request.url.path == "/api/dcn/mode/promotions"
        assert request.url.params["limit"] == "3"
        return httpx.Response(200, json={"status": "ok", "runtime_mode": "active_learn", "count": 1, "entries": [{"promotion_id": "p1"}]})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    assert provider_cli.main([
        "dcn",
        "promote",
        "--mode",
        "active_learn",
        "--checkpoint-id",
        "cp1",
        "--hygiene-evidence",
        '{"passed": true, "artifact_hash": "abc123"}',
        "--operator",
        "pytest",
        "--reason",
        "promotion test",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["promoted"] is True
    assert provider_cli.main(["dcn", "promotions", "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"][0]["promotion_id"] == "p1"
    assert seen == [("POST", "/api/dcn/mode/promote"), ("GET", "/api/dcn/mode/promotions")]


def test_provider_cli_dcn_promote_returns_nonzero_when_provider_refuses(capsys, monkeypatch) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/dcn/mode/promote"
        return httpx.Response(200, json={"status": "failed", "promoted": False, "reason": "hygiene_evidence_required"})

    monkeypatch.setattr(provider_cli.httpx, "Client", lambda **kwargs: real_client(transport=_transport(handler), **kwargs))
    rc = provider_cli.main([
        "dcn",
        "promote",
        "--checkpoint-id",
        "cp1",
        "--hygiene-evidence",
        '{"passed": false}',
    ])

    assert rc == 1
    assert json.loads(capsys.readouterr().out)["promoted"] is False
