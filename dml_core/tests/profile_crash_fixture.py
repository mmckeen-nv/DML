"""Parent-owned synthetic profile histories and bounded process-death protocol.

No production receipt builders, digest helpers, or lifecycle selectors construct
expected results. A barrier is test instrumentation, never acknowledgement.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import sqlite3
import subprocess
import sys
from tempfile import TemporaryFile
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import IdempotencyConflict, JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal

SCHEMAS = (2, 3, 4)
OPERATIONS = ("ingest", "update", "promote", "supersede", "retire")
SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}
CLOCK = 1700000000.0
VECTOR = [1.0, 0.0, 0.0, 0.0]
WAIT = 30
METHODS = {
    "ingest": "ingest_memory_receipted", "update": "update_memory_receipted",
    "promote": "promote_memories_receipted", "supersede": "supersede_memory_receipted",
    "retire": "retire_memory_receipted",
}
# Independent inventory: do not derive case counts/outcomes from runtime constants.
RECEIPT_POINTS = ("after_begin", "after_records", "after_state", "after_snapshot",
                  "after_receipt", "after_decision", "before_commit", "after_commit")
OUTBOX_POINTS = (*RECEIPT_POINTS[:-2], "after_outbox", *RECEIPT_POINTS[-2:])
COMMITTED_POINTS = frozenset({"after_commit", "before_hydration", "before_response", "after_ack"})


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


class ProfileEmbedder:
    model_name = "synthetic-profile-recovery-v1"

    def __init__(self, unavailable=False):
        self.unavailable = unavailable
        self.calls = []
        self.barrier = lambda _point: None

    def embed(self, text):
        self.barrier("during_embedding")
        if self.unavailable:
            raise AssertionError("Historical receipt retry contacted embedding backend")
        self.calls.append(text)
        return np.array(VECTOR, dtype=np.float32)


def supported_config(directory, schema):
    return {
        "production_profile": "dml-receipted-local-v1", "storage_dir": str(directory),
        "model_name": "dummy", "llm_backend": "dummy", "embedding_model": ProfileEmbedder.model_name,
        "strict_embedding_required": True,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True, "receipts": True,
            "outbox": schema in (3, 4), "receipt_embedding_identity": "synthetic-profile-revision-v1",
            "snapshot_interval": 1},
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False, "include_in_preamble": False},
        "ann_min_items": 0, "checkpoint_interval_seconds": 0, "skip_rag_state_import": True,
        "survival_ledger_enabled": False, "background_processing_enabled": False,
        "enable_stm_controller": False, "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False, "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False, "metrics_enabled": False,
    }


def make_adapter(directory, schema, unavailable=False):
    return DMLAdapter(config_path=Path(directory).parent / "absent-recovery-config.yaml",
        config_overrides=supported_config(directory, schema),
        embedder=ProfileEmbedder(unavailable=unavailable), start_aging_loop=False)


@dataclass
class Request:
    operation: str
    args: tuple
    kwargs: dict

    @property
    def method(self):
        return METHODS[self.operation]


@dataclass
class Scenario:
    directory: Path
    schema: int
    before: dict

    @property
    def records(self):
        return self.before["state"]["items"]


def invoke(adapter, request):
    # Replace this module's reference, not the process-global time module.
    with patch("daystrom_dml.services.receipt_ingestion.time", SimpleNamespace(time=lambda: CLOCK)):
        return getattr(adapter, request.method)(*request.args, **deepcopy(request.kwargs))


def sql_observation(directory):
    """Read logical SQL rows directly; no JournalStateStore decoding/oracle reuse."""
    path = Path(directory) / "dml_state.sqlite3"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        schema = connection.execute("PRAGMA user_version").fetchone()[0]
        revision, metadata = connection.execute("SELECT revision,metadata FROM state WHERE id=1").fetchone()
        state = {**json.loads(metadata), "items": [], "lineage": []}
        for bucket, raw in connection.execute("SELECT bucket,payload FROM records ORDER BY bucket,position"):
            state[bucket].append(json.loads(raw))
        return {
            "schema": schema, "store_id": connection.execute("SELECT store_id FROM identity WHERE id=1").fetchone()[0],
            "revision": revision, "state": state,
            "snapshot": json.loads(connection.execute("SELECT payload FROM snapshot WHERE id=1").fetchone()[0]),
            "receipts": [json.loads(row[0]) for row in connection.execute("SELECT payload FROM receipts ORDER BY revision")],
            "decisions": [json.loads(row[0]) for row in connection.execute("SELECT payload FROM decisions ORDER BY revision")],
            "outbox": ([json.loads(row[0]) for row in connection.execute("SELECT payload FROM outbox ORDER BY revision")]
                       if schema in (3, 4) else []),
        }


def prepare(directory, schema):
    directory = Path(directory)
    if schema == 4:
        source = JournalStateStore(directory.parent / (directory.name + "-migration-source") / "dml_state.sqlite3",
                                   receipt_mode=True)
        upgrade_outbox_journal(source.path, directory / "dml_state.sqlite3")
    adapter = make_adapter(directory, schema)
    acknowledgements = []
    try:
        for key, text, scope in (
            ("source", "The owner prefers blue notebooks", SCOPE),
            ("replacement", "The owner prefers green notebooks", SCOPE),
            ("foreign", "PRIVATE_FOREIGN_RECOVERY_SENTINEL", {**SCOPE, "tenant_id": "foreign"}),
        ):
            request = Request("ingest", (text,), {**scope, "idempotency_key": key,
                "meta": {"source_trust": "trusted"}})
            acknowledgements.append(invoke(adapter, request))
    finally:
        adapter.close()
    before = sql_observation(directory)
    assert before["receipts"] == acknowledgements
    # Schema 4 publishes one explicit migration-baseline event before appends.
    assert before["revision"] == (4 if schema == 4 else 3)
    expected_records = []
    for ident, text, scope in (
        (0, "The owner prefers blue notebooks", SCOPE),
        (1, "The owner prefers green notebooks", SCOPE),
        (2, "PRIVATE_FOREIGN_RECOVERY_SENTINEL", {**SCOPE, "tenant_id": "foreign"}),
    ):
        expected_records.append({"schema_version": 1, "id": ident, "text": text,
            "timestamp": CLOCK, "salience": 1.0, "fidelity": 1.0, "level": 0,
            "meta": {"source_trust": "trusted", **scope, "kind": "memory", "no_merge": True},
            "summary_of": [], "children": [ident], "embedding": VECTOR})
    assert encoded(before["state"]["items"]) == encoded(expected_records)
    assert before["state"]["lineage"] == [] and before["state"]["next_id"] == 3
    assert before["state"]["embedding_contract"] == {
        "schema_version": "dml-embedding-contract-v1", "dimension": 4,
        "identity": {"backend": "profile_crash_fixture.ProfileEmbedder",
            "revision": "synthetic-profile-revision-v1", "model": ProfileEmbedder.model_name, "mode": "native"},
    }
    return Scenario(directory, schema, before)


def request_for(operation, records, key="target"):
    owner_records = [record for record in records if all(record["meta"].get(k) == v for k, v in SCOPE.items())]
    source, replacement = owner_records[:2]
    kwargs = {**SCOPE, "idempotency_key": key}
    if operation == "ingest":
        return Request(operation, ("A newly remembered notebook fact",),
                       {**kwargs, "meta": {"source_trust": "trusted"}})
    if operation == "promote":
        return Request(operation, ([{"memory_id": source["id"], "expected_memory_digest": digest(source)}],),
                       {**kwargs, "text": "An explicit notebook interpretation", "reason": "Owner approved derivation"})
    kwargs.update(expected_memory_digest=digest(source), reason="Owner approved correction")
    if operation == "update":
        kwargs["text"] = "The owner uses blue notebooks"
    if operation == "supersede":
        kwargs.update(replacement_memory_id=replacement["id"], expected_replacement_digest=digest(replacement))
    return Request(operation, (source["id"],), kwargs)


def expected_transition(before, request):
    """Complete transition oracle from submitted intent and parent-owned records."""
    state = deepcopy(before["state"])
    args, kw = request.args, request.kwargs
    scope = {name: kw.get(name) for name in SCOPE}
    operation = request.operation
    if operation == "ingest":
        ident = state.get("next_id", 0)
        meta = {**kw.get("meta", {}), **scope, "kind": kw.get("kind") or "memory", "no_merge": True}
        record = {"schema_version": 1, "id": ident, "text": args[0], "timestamp": CLOCK,
            "salience": 1.0, "fidelity": 1.0, "level": 0, "meta": meta,
            "summary_of": [], "embedding": VECTOR, "children": [ident]}
        canonical = {"schema_version": "dml-append-request-v1", "text": args[0],
                     "scope": scope, "kind": meta["kind"], "meta": meta}
        state["items"].append(record)
        state["next_id"] = ident + 1
    elif operation == "promote":
        sources = sorted(args[0], key=lambda entry: entry["memory_id"])
        records = [{record["id"]: record for record in state["items"]}[source["memory_id"]] for source in sources]
        ident = state["next_id"]
        source_ids = [record["id"] for record in records]
        meta = {name: deepcopy(value) for name, value in records[0]["meta"].items()
                if name not in {"summary", "source", "source_id", "source_ids", "provenance", "content_update_decision"}}
        meta["promotion_decision"] = {"schema_version": "dml-promotion-decision-v1", "reason": kw["reason"],
            "sources": [{"memory_digest": digest(record), "memory": deepcopy(record)} for record in records]}
        record = {"schema_version": 1, "id": ident, "text": kw["text"], "embedding": VECTOR,
            "timestamp": min(record["timestamp"] for record in records), "salience": min(record["salience"] for record in records),
            "fidelity": min(record["fidelity"] for record in records), "level": 1, "meta": meta,
            "summary_of": source_ids, "children": source_ids}
        canonical = {"schema_version": "dml-promotion-request-v1", "sources": sources,
                     "text": kw["text"], "reason": kw["reason"], "scope": scope}
        state["items"].append(record)
        state["next_id"] = ident + 1
    else:
        record = next(record for record in state["items"] if record["id"] == args[0])
        canonical = {"schema_version": "dml-" + operation + "-request-v1", "memory_id": args[0],
            "expected_memory_digest": kw["expected_memory_digest"], "reason": kw["reason"], "scope": scope}
        decision = {"prior_memory_digest": kw["expected_memory_digest"], "reason": kw["reason"]}
        if operation == "update":
            record.update(text=kw["text"], embedding=VECTOR)
            record["meta"]["content_update_decision"] = {"schema_version": "dml-content-update-decision-v1", **decision}
            canonical["text"] = kw["text"]
        elif operation == "supersede":
            record["meta"].update(memory_state="superseded", superseded_by=kw["replacement_memory_id"],
                supersession_decision={"schema_version": "dml-supersession-decision-v1", **decision,
                    "replacement_memory_id": kw["replacement_memory_id"], "replacement_memory_digest": kw["expected_replacement_digest"]})
            canonical.update(replacement_memory_id=kw["replacement_memory_id"], expected_replacement_digest=kw["expected_replacement_digest"])
        elif operation == "retire":
            record["meta"].update(memory_state="deleted", retirement_decision={"schema_version": "dml-retirement-decision-v1", **decision})
        else:
            raise AssertionError(operation)
    receipt = {"schema_version": 1, "store_id": before["store_id"], "revision": before["revision"] + 1,
        "scope": scope, "key": kw["idempotency_key"], "request_digest": digest(canonical), "result": {"memory": record}}
    return {"state": state, "receipt": receipt,
            "operation": ("append" if operation == "ingest" else operation) + "-receipt-v1"}


def assert_transition(before, after, request):
    expected = expected_transition(before, request)
    assert after["schema"] == before["schema"]
    assert after["store_id"] == before["store_id"]
    assert after["revision"] == before["revision"] + 1
    assert encoded(after["state"]) == encoded(expected["state"])
    assert encoded(after["snapshot"]) == encoded(expected["state"])
    assert encoded(after["receipts"]) == encoded([*before["receipts"], expected["receipt"]])
    assert after["decisions"][:-1] == before["decisions"]
    decision = after["decisions"][-1]
    assert decision["schema_version"] == before["schema"]
    assert decision["revision"] == after["revision"]
    assert decision["operation"] == expected["operation"]
    assert decision["state_digest"] == digest(expected["state"])
    assert decision["record_count"] == sum(len(expected["state"][bucket]) for bucket in ("items", "lineage"))
    assert decision["deleted"] == []
    assert decision["changed"] == [{"bucket": "items", "id": str(expected["receipt"]["result"]["memory"]["id"]),
                                     "digest": digest(expected["receipt"]["result"]["memory"])}]
    binding = {name: expected["receipt"][name] for name in ("scope", "key", "request_digest")}
    binding["digest"] = digest(expected["receipt"])
    assert decision["receipt"] == binding
    if before["schema"] in (3, 4):
        assert after["outbox"][:-1] == before["outbox"]
        event = after["outbox"][-1]
        assert event["state"] == expected["state"]
        assert event["receipt"] == binding
        assert event["source_revision"] == after["revision"]
        assert event["source_store_id"] == before["store_id"]
        assert event["operation"] == expected["operation"]
        assert event["source_digest"] == digest(expected["state"])
    else:
        assert after["outbox"] == []
    return expected["receipt"]


def assert_reads(adapter, observed):
    expected_ids = {record["id"] for record in observed["state"]["items"]
        if all(record["meta"].get(name) == value for name, value in SCOPE.items())
        and record["meta"].get("memory_state", "active") == "active"}
    report = adapter.retrieve_context("notebooks", top_k=10, as_of=CLOCK, **SCOPE)
    assert {int(record["id"]) for record in report["items"]} == expected_ids
    assert "PRIVATE_FOREIGN_RECOVERY_SENTINEL" not in report["raw_context"]
    assert adapter.retrieve_context("notebooks", top_k=10, as_of=CLOCK,
        **{**SCOPE, "session_id": "different"})["items"] == []
    groups = {"current_items": observed["state"]["items"], "current_lineage": observed["state"]["lineage"],
        "journal_snapshot": [record for bucket in ("items", "lineage") for record in observed["snapshot"][bucket]],
        "receipts": [receipt["result"]["memory"] for receipt in observed["receipts"]],
        "outbox_states": [record for event in observed["outbox"] for bucket in ("items", "lineage") for record in event["state"][bucket]]}
    for target in observed["state"]["items"]:
        if not all(target["meta"].get(name) == value for name, value in SCOPE.items()):
            continue
        counts = {}
        for surface, records in groups.items():
            scoped = [record for record in records if all(record["meta"].get(name) == value for name, value in SCOPE.items())]
            counts[surface] = {
                "direct_records": sum(record["id"] == target["id"] for record in scoped),
                "embedded_source_records": sum(entry["memory"]["id"] == target["id"] for record in scoped
                    for entry in record["meta"].get("promotion_decision", {}).get("sources", [])),
            }
        retention = adapter.inspect_memory_retention(target["id"], **SCOPE)
        assert retention["source"] == {"store_id": observed["store_id"], "journal_schema_version": observed["schema"],
                                       "revision": observed["revision"], "state_digest": digest(observed["state"])}
        assert retention["surfaces"] == counts
        assert retention["known_reference_count"] == sum(sum(count.values()) for count in counts.values())
        assert retention["erasure_proven"] is False


def assert_recovered(scenario, request, committed):
    observed = sql_observation(scenario.directory)
    if committed:
        assert_transition(scenario.before, observed, request)
    else:
        assert encoded(observed) == encoded(scenario.before)
    adapter = make_adapter(scenario.directory, scenario.schema, unavailable=committed)
    try:
        receipt = invoke(adapter, request)
        if committed:
            assert adapter.embedder.calls == []
        expected = assert_transition(scenario.before, sql_observation(scenario.directory), request)
        assert encoded(receipt) == encoded(expected)
        # Historical replay cannot invoke the backend even after an originally absent operation committed.
        adapter.embedder.unavailable = True
        assert invoke(adapter, request) == receipt
        changed = deepcopy(request)
        changed.kwargs["meta"] = {"source_trust": "untrusted"} if request.operation == "ingest" else changed.kwargs.get("meta")
        if request.operation != "ingest":
            changed.kwargs.pop("meta")
            changed.kwargs["reason"] += " changed"
        try:
            invoke(adapter, changed)
        except IdempotencyConflict:
            pass
        else:
            raise AssertionError("Changed request reused committed key")
        stable = sql_observation(scenario.directory)
        adapter.embedder.unavailable = False
        assert_reads(adapter, stable)
        assert sql_observation(scenario.directory) == stable
    finally:
        adapter.close()
    # A second fresh adapter must observe the same reconstructed state.
    reopened = make_adapter(scenario.directory, scenario.schema)
    try:
        assert_reads(reopened, stable)
        assert sql_observation(scenario.directory) == stable
    finally:
        reopened.close()
    return receipt


def _child_main():
    directory, schema, point, request_path = sys.argv[1:]
    raw = json.loads(Path(request_path).read_bytes())
    request = Request(raw["operation"], tuple(raw["args"]), raw["kwargs"])
    adapter = make_adapter(Path(directory), int(schema))

    def barrier(here):
        if here == point:
            print(encoded({"barrier": here}), flush=True)
            # Only the parent terminates this process. EOF/release is a test failure.
            sys.stdin.read(1)
            raise AssertionError("Parent did not kill process at reached boundary")

    adapter._journal._fault_hook = barrier
    adapter.embedder.barrier = barrier
    original_import = adapter.store.import_state

    def hydrate(payload):
        barrier("before_hydration")
        return original_import(payload)

    adapter.store.import_state = hydrate
    receipt = invoke(adapter, request)
    barrier("before_response")
    print(encoded({"ack": receipt}), flush=True)
    barrier("after_ack")
    raise AssertionError("Requested crash boundary was not reached")


def run_interrupted(scenario, request, point):
    request_path = scenario.directory.parent / (scenario.directory.name + "-request.json")
    request_path.write_bytes(encoded({"operation": request.operation, "args": request.args, "kwargs": request.kwargs}).encode("utf-8"))
    env = {key: value for key, value in os.environ.items() if not key.startswith("DML_")}
    test_dir = Path(__file__).resolve().parent
    env.update(DML_SKIP_VENV_REEXEC="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(test_dir), str(test_dir.parent), env.get("PYTHONPATH", "")])
    messages = Queue()
    acknowledgement = None
    with TemporaryFile(mode="w+b") as errors:
        process = subprocess.Popen([sys.executable, "-u", "-c", "from profile_crash_fixture import _child_main; _child_main()",
            str(scenario.directory), str(scenario.schema), point, str(request_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8",
            env=env, cwd=scenario.directory.parent)

        def read_messages():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = Thread(target=read_messages, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    line = messages.get(timeout=WAIT)
                except Empty as exc:
                    raise AssertionError(f"Child did not reach {point} within {WAIT}s") from exc
                if line is None:
                    errors.seek(0)
                    raise AssertionError("Child exited before barrier: " + errors.read().decode("utf-8", errors="replace")[-10000:])
                message = json.loads(line)
                if "ack" in message:
                    assert point == "after_ack" and acknowledgement is None
                    acknowledgement = message["ack"]
                else:
                    assert message == {"barrier": point}
                    break
            process.kill()  # SIGKILL on POSIX; TerminateProcess on Windows.
            process.wait(timeout=WAIT)
            if os.name == "posix":
                assert process.returncode == -signal.SIGKILL
            else:
                assert process.returncode != 0
            assert (acknowledgement is not None) == (point == "after_ack")
            return acknowledgement
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=WAIT)
            process.stdin.close()
            reader.join(WAIT)
            process.stdout.close()
            assert not reader.is_alive(), "Child output reader did not terminate"
