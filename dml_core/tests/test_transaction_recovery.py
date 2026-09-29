"""Independent ownership histories at the adapter transaction boundary.

Events and child-process pipes establish the interleavings; timeouts only bound
failures. Legacy rollback compensates caught exceptions while ownership is held.
It cannot make separate lattice/RAG replacements atomic after process death.
Receipt journals instead publish their records and receipt in one SQLite commit.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sqlite3
import sys
from threading import Event, Thread, current_thread
from time import monotonic

import pytest

import daystrom_dml.dml_adapter as adapter_module
import daystrom_dml.store_lock as lock_module
from daystrom_dml.dml_adapter import DMLAdapter, PersistenceCommitError
from daystrom_dml.embeddings import RandomEmbedder
from daystrom_dml.journal import JournalIntegrityError
from daystrom_dml.persistence import PersistenceFormatError
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal


LEGACY = ["json", "jsonl", "j1"]
RECEIPTS = ["r2", "r3", "r4"]
WAIT = 15


def _make(directory, profile):
    return DMLAdapter(
        config_overrides={
            "storage_dir": str(directory), "model_name": "dummy", "embedding_model": None,
            "checkpoint_interval_seconds": 0, "metrics_enabled": False,
            "persistence": {"enable": profile == "jsonl", "path": "state.jsonl",
                            "interval_sec": 0, "journal": profile not in {"json", "jsonl"},
                            "receipts": profile in RECEIPTS, "outbox": profile in {"r3", "r4"}},
            "rag_store": {"enable": False},
            "dpm": {"enable": False, "include_in_context": False},
            "theta_merge": 2.0, "eta": 0.0, "gamma": 0.0, "kappa": 0.0,
            "similarity_threshold": 0.0, "enable_quality_on_retrieval": False,
        }, embedder=RandomEmbedder(dim=4), start_aging_loop=False,
    )


@pytest.fixture
def factory(tmp_path):
    instances = []

    def build(profile="jsonl", directory=None):
        directory = directory or tmp_path / profile
        if profile == "r4" and not (directory / "dml_state.sqlite3").exists():
            source = _make(tmp_path / "s2", "r2")
            source.close(persist=False)
            upgrade_outbox_journal(source._journal.path, directory / "dml_state.sqlite3")
        instance = _make(directory, profile)
        instances.append(instance)
        return instance

    yield build
    for instance in reversed(instances):
        instance.close(persist=False)


def _ingest(adapter, text, *, persist=True):
    adapter.ingest(text, meta={"no_merge": True, "oracle": text}, persist=persist)


def _append(adapter, text):
    return adapter.ingest_memory_receipted(
        text, tenant_id="owner", idempotency_key=text, meta={"oracle": text})


def _records(adapter):
    records = adapter.store.items()
    assert len({item.id for item in records}) == len(records)
    assert len({item.text for item in records}) == len(records), "duplicate acknowledged record"
    return {item.text: item.meta["oracle"] for item in records}


def _rag_texts(adapter):
    return sorted(item["text"] for item in adapter.rag_store.export_state()["documents"])


def _expect(adapter, *texts):
    # Expectations come from submitted operations, never a second runtime copy.
    assert _records(adapter) == {text: text for text in texts}


@contextmanager
def _thread(action, *, name="contender"):
    result = Queue()

    def run():
        try:
            result.put((True, action()))
        except BaseException as exc:
            result.put((False, exc))

    worker = Thread(target=run, name=name, daemon=True)
    worker.start()
    try:
        yield result
    finally:
        worker.join(WAIT)
        assert not worker.is_alive(), f"{name} did not leave transaction ownership"


def _outcome(result):
    succeeded, value = result.get(timeout=WAIT)
    if not succeeded:
        raise value
    return value


def _watch_contention(monkeypatch, *, name="contender"):
    blocked = Event()
    acquire = lock_module.acquire_file_lock

    def observed(handle):
        try:
            return acquire(handle)
        except BlockingIOError:
            if current_thread().name == name:
                blocked.set()
            raise

    monkeypatch.setattr(lock_module, "acquire_file_lock", observed)
    return blocked


@pytest.mark.parametrize("profile", LEGACY + RECEIPTS)
def test_nested_owner_excludes_another_thread_on_same_adapter(factory, monkeypatch, profile):
    adapter = factory(profile)
    blocked = _watch_contention(monkeypatch)
    entered = Event()
    start = Event()
    released = Event()

    def contender():
        assert start.wait(WAIT)
        with adapter.mutation_transaction("thread-contender"):
            assert released.is_set(), "thread inherited another thread's nesting privilege"
            entered.set()

    # Keep thread cleanup outside ownership so an assertion cannot strand it.
    with _thread(contender) as competing:
        with adapter.mutation_transaction("outer"):
            with pytest.raises(RuntimeError, match="inner"):
                with adapter.mutation_transaction("inner"):
                    raise RuntimeError("inner")
            start.set()
            assert blocked.wait(WAIT), "competing thread never attempted the real file lock"
            assert not entered.is_set()
            released.set()
    _outcome(competing)
    assert entered.is_set()


@pytest.mark.parametrize("profile", LEGACY)
def test_stale_peer_refreshes_before_batch_body_and_preserves_all_records(factory, profile):
    writer, peer = factory(profile), factory(profile)
    _ingest(writer, "seed")
    _expect(peer)
    with peer.atomic_batch("peer-batch"):
        _expect(peer, "seed")
        _ingest(peer, "peer-one", persist=False)
        _ingest(peer, "peer-two", persist=False)
    with writer.mutation_transaction("observe-peer"):
        _expect(writer, "seed", "peer-one", "peer-two")
    reopened = factory(profile)
    _expect(reopened, "seed", "peer-one", "peer-two")
    assert _rag_texts(reopened) == ["peer-one", "peer-two", "seed"]


@pytest.mark.parametrize("profile", LEGACY + RECEIPTS)
def test_next_owner_validates_authority_before_entering_body(factory, profile):
    adapter = factory(profile)
    (_append if profile in RECEIPTS else _ingest)(adapter, "seed")
    path = adapter._active_state_path()
    if adapter._journal is None:
        original = path.read_bytes()
        path.write_bytes(b"invalid-authority")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            original = connection.execute("SELECT checksum FROM snapshot WHERE id=1").fetchone()[0]
            connection.execute("UPDATE snapshot SET checksum='invalid-authority' WHERE id=1")
    try:
        with pytest.raises((json.JSONDecodeError, PersistenceFormatError, JournalIntegrityError)):
            with adapter.mutation_transaction("invalid-authority"):
                pytest.fail("mutation body ran without a verified durable authority")
        with lock_module.store_write_lock(path.parent, operation="after-invalid", timeout_ms=0):
            pass
        if adapter._journal is None:
            assert path.read_bytes() == b"invalid-authority"
        else:
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("SELECT checksum FROM snapshot WHERE id=1").fetchone() == (
                    "invalid-authority",)
    finally:
        if adapter._journal is None:
            path.write_bytes(original)
        else:
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE snapshot SET checksum=? WHERE id=1", (original,))
    with adapter.mutation_transaction("repaired-authority"):
        _expect(adapter, "seed")


@pytest.mark.parametrize("profile", LEGACY)
def test_compensation_keeps_ownership_until_waiting_peer_can_refresh(factory, monkeypatch, profile):
    writer, peer = factory(profile), factory(profile)
    _ingest(writer, "seed")
    blocked = _watch_contention(monkeypatch)
    compensation = Event()
    release = Event()
    commit = writer.lattice_persistence.commit
    write_text = adapter_module.atomic_write_text
    commits = []

    def watched_commit(**kwargs):
        commits.append(True)
        if len(commits) == 2:
            compensation.set()
            assert release.wait(WAIT), "parent did not release compensation"
        return commit(**kwargs)

    def fail_rag(path, content, *args, **kwargs):
        if Path(path) == writer.rag_state_path and current_thread().name == "failing":
            raise OSError("RAG replacement failed after lattice publication")
        return write_text(path, content, *args, **kwargs)

    monkeypatch.setattr(writer.lattice_persistence, "commit", watched_commit)
    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_rag)

    def failing():
        with pytest.raises(PersistenceCommitError, match="RAG state"):
            _ingest(writer, "rolled-back")

    def succeeds():
        with peer.atomic_batch("surviving-peer"):
            _expect(peer, "seed")
            _ingest(peer, "survivor", persist=False)

    with _thread(failing, name="failing") as failure:
        try:
            assert compensation.wait(WAIT)
            with _thread(succeeds) as success:
                try:
                    assert blocked.wait(WAIT), "peer must contend during compensation"
                finally:
                    release.set()
            _outcome(success)
        finally:
            release.set()
    _outcome(failure)
    assert len(commits) == 2
    _expect(writer, "seed")
    reopened = factory(profile)
    _expect(reopened, "seed", "survivor")
    assert _rag_texts(reopened) == ["seed", "survivor"]


def _child_main():
    directory, profile, scenario = sys.argv[1:]
    adapter = _make(Path(directory), profile)

    def signal_parent(message):
        print(message, flush=True)

    def await_parent():
        if sys.stdin.readline() != "release\n":
            raise RuntimeError("parent channel closed")

    if scenario == "contend":
        acquire = lock_module.acquire_file_lock
        reported = False

        def observed(handle):
            nonlocal reported
            try:
                return acquire(handle)
            except BlockingIOError:
                if not reported:
                    signal_parent("blocked")
                    reported = True
                raise

        lock_module.acquire_file_lock = observed
        signal_parent("ready")
        await_parent()
        with adapter.mutation_transaction("spawned-outer"):
            with adapter.mutation_transaction("spawned-inner"):
                _expect(adapter, "seed")
                _ingest(adapter, "child")
        signal_parent("done")
    elif scenario == "legacy-published":
        with adapter.atomic_batch("interrupted-batch"):
            _ingest(adapter, "interrupted", persist=False)
            adapter._persist_dml_state()
            signal_parent("published")
            await_parent()
    else:
        def fault(point):
            if point == scenario:
                signal_parent(point)
                await_parent()

        adapter._journal._fault_hook = fault
        _append(adapter, "interrupted")
    adapter.close(persist=False)


@contextmanager
def _child(adapter, profile, scenario):
    env = os.environ.copy()
    test_dir = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join(
        [str(test_dir), str(test_dir.parent), env.get("PYTHONPATH", "")])
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "from test_transaction_recovery import _child_main; _child_main()",
         str(adapter.storage_dir), profile, scenario],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=WAIT)


def _message(process, expected):
    with _thread(process.stdout.readline, name="child-output") as result:
        try:
            actual = _outcome(result).strip()
            assert actual == expected, f"child reported {actual!r}, expected {expected!r}"
        except BaseException:
            if process.poll() is None:
                process.kill()
            raise


@pytest.mark.parametrize("profile", LEGACY)
def test_spawned_peer_cannot_inherit_parent_nesting_privilege(factory, profile):
    adapter = factory(profile)
    _ingest(adapter, "seed")
    with _child(adapter, profile, "contend") as process:
        _message(process, "ready")
        with adapter.mutation_transaction("parent-outer"):
            with adapter.mutation_transaction("parent-inner"):
                process.stdin.write("release\n")
                process.stdin.flush()
                _message(process, "blocked")
        stdout, stderr = process.communicate(timeout=WAIT)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "done"
    _expect(factory(profile), "seed", "child")


def _forked_owner(adapter, channel):
    """Use the inherited adapter deliberately: thread-local depth crosses fork."""
    acquire = lock_module.acquire_file_lock

    def observed(handle):
        channel.send("lock-attempt")
        return acquire(handle)

    lock_module.acquire_file_lock = observed
    try:
        with adapter.mutation_transaction("forked-owner"):
            with adapter.mutation_transaction("forked-inner"):
                channel.send("entered")
                _ingest(adapter, "child")
    except RuntimeError as exc:
        channel.send(("refused", str(exc)))
    finally:
        channel.close()


@pytest.mark.skipif(os.name != "posix", reason="fork is a POSIX ownership hazard")
def test_forked_child_must_construct_a_fresh_transaction_owner(factory):
    adapter = factory("j1")
    _ingest(adapter, "seed")
    context = multiprocessing.get_context("fork")
    parent_channel, child_channel = context.Pipe(duplex=False)
    process = context.Process(target=_forked_owner, args=(adapter, child_channel))
    try:
        with adapter.mutation_transaction("parent-owner"):
            with adapter.mutation_transaction("parent-inner"):
                process.start()
                child_channel.close()
                assert parent_channel.poll(WAIT), "forked child did not report ownership refusal"
                assert parent_channel.recv() == (
                    "refused", "Inherited transaction coordinator cannot be used after fork; "
                    "create a fresh adapter in the child process")
                process.join(WAIT)
                assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(WAIT)
        parent_channel.close()
        child_channel.close()
    _expect(adapter, "seed")
    _expect(factory("j1"), "seed")


@pytest.mark.parametrize("profile", LEGACY)
def test_process_death_releases_owner_but_does_not_compensate_legacy_files(factory, profile):
    adapter = factory(profile)
    _ingest(adapter, "seed")
    with _child(adapter, profile, "legacy-published") as process:
        _message(process, "published")
        with pytest.raises(TimeoutError):
            with lock_module.store_write_lock(adapter._active_state_path().parent,
                                               operation="probe-dead-owner", timeout_ms=0):
                pytest.fail("child must still own the writer lock")
        process.kill()
        process.wait(timeout=WAIT)
    with adapter.mutation_transaction("next-owner"):
        _expect(adapter, "seed", "interrupted")
        assert _rag_texts(adapter) == ["seed"]
        _ingest(adapter, "survivor")
    reopened = factory(profile)
    _expect(reopened, "seed", "interrupted", "survivor")
    assert _rag_texts(reopened) == ["seed", "survivor"]


@pytest.mark.parametrize("point,published", [("before_commit", False), ("after_commit", True)],
                         ids=["pre", "post"])
def test_receipt_process_death_recovers_one_atomic_record_and_receipt(factory, point, published):
    adapter = factory("r3")
    initial = _append(adapter, "seed")
    with _child(adapter, "r3", point) as process:
        _message(process, point)
        process.kill()
        process.wait(timeout=WAIT)
    with adapter.mutation_transaction("verify-after-death"):
        _expect(adapter, *(["seed", "interrupted"] if published else ["seed"]))
        assert adapter._journal.revision == initial["revision"] + int(published)
    recovered = _append(adapter, "interrupted")
    assert recovered["revision"] == initial["revision"] + 1
    assert _append(adapter, "interrupted") == recovered
    _expect(factory("r3"), "seed", "interrupted")


@pytest.mark.parametrize("profile", RECEIPTS)
def test_lost_receipt_ack_is_not_compensated_and_waiting_reader_observes_commit(
        factory, monkeypatch, profile):
    writer, reader = factory(profile), factory(profile)
    committed, release = Event(), Event()
    blocked = _watch_contention(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("receipt failure must not use legacy compensation")

    monkeypatch.setattr(writer, "_rollback_mutation", forbidden)

    def fault(point):
        if point == "after_commit":
            committed.set()
            assert release.wait(WAIT)
            raise OSError("lost durable receipt acknowledgement")

    writer._journal._fault_hook = fault

    def observes():
        with reader.mutation_transaction("receipt-reader"):
            _expect(reader, "committed")

    with _thread(lambda: _append(writer, "committed"), name="receipt-writer") as write:
        try:
            assert committed.wait(WAIT)
            with _thread(observes) as read:
                try:
                    assert blocked.wait(WAIT)
                finally:
                    release.set()
            _outcome(read)
        finally:
            release.set()
    receipt = _outcome(write)
    assert _append(reader, "committed") == receipt
    _expect(factory(profile), "committed")


def test_mixed_adapter_clients_preserve_independent_committed_history(factory):
    """Bounded coordinator contention, not HTTP throughput or fairness qualification.

    Set DML_TRANSACTION_CLIENTS=256 for the one-off expanded qualification run.
    """
    clients = int(os.environ.get("DML_TRANSACTION_CLIENTS", "16"))
    assert 4 <= clients <= 256
    adapters = [factory("j1") for _ in range(clients)]
    start = Event()

    def write(index):
        assert start.wait(WAIT)
        adapter, text = adapters[index], f"client-{index:03}"
        if index % 4 == 0:
            _ingest(adapter, text)
        elif index % 4 == 1:
            with adapter.atomic_batch("mixed-batch"):
                _ingest(adapter, text, persist=False)
        elif index % 4 == 2:
            with adapter.mutation_transaction("mixed-owned"):
                adapter.ingest_memory(text, tenant_id="owner",
                                      meta={"no_merge": True, "oracle": text})
        else:
            with pytest.raises(RuntimeError, match="cancelled"):
                with adapter.atomic_batch("mixed-abort"):
                    _ingest(adapter, f"aborted-{index:03}", persist=False)
                    raise RuntimeError("cancelled")
            _ingest(adapter, text)
        return text

    tasks, outcomes, stop = Queue(), Queue(), Event()
    for index in range(clients):
        tasks.put(index)

    def consume():
        while not stop.is_set():
            try:
                index = tasks.get_nowait()
            except Empty:
                return
            try:
                outcomes.put((True, write(index)))
            except BaseException as exc:
                outcomes.put((False, exc))
                stop.set()

    workers = [Thread(target=consume, name=f"mixed-{index}", daemon=True)
               for index in range(min(clients, 32))]
    for worker in workers:
        worker.start()
    acknowledged = []
    try:
        deadline = monotonic() + 120
        start.set()
        for _ in range(clients):
            succeeded, value = outcomes.get(timeout=max(0, deadline - monotonic()))
            if not succeeded:
                raise value
            acknowledged.append(value)
    finally:
        stop.set()
        start.set()
        deadline = monotonic() + WAIT
        for worker in workers:
            worker.join(max(0, deadline - monotonic()))
        assert not any(worker.is_alive() for worker in workers), "mixed owner failed to release"
    assert sorted(acknowledged) == [f"client-{index:03}" for index in range(clients)]
    reopened = factory("j1")
    _expect(reopened, *[f"client-{index:03}" for index in range(clients)])
    assert _rag_texts(reopened) == [f"client-{index:03}" for index in range(clients) if index % 4 != 2]
    assert reopened._journal.revision == clients
