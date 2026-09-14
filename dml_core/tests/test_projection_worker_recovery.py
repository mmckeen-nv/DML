"""Process-death oracle for scheduled projection delivery and restart recovery."""
from __future__ import annotations

import os
import time

import pytest

from daystrom_dml.journal import FAULT_POINTS, JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection, prepare_source, projection_status, reconcile
from daystrom_dml.services.projection_worker import ProjectionWorker
from test_projection import pair as projection_pair, query, run_child, write

pair = projection_pair


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_killed_worker_recovers_from_durable_authority_without_pending_queue(pair, point):
    source, target = pair
    reconcile(source, target)
    prior = target.read()
    write(source, "two", text="Receipt survives the background worker")
    acknowledged = prepare_source(source)
    child = run_child('''
import os, signal, sys, threading
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import SQLiteProjection
from daystrom_dml.services.projection_worker import ProjectionWorker
source_path, target_path, point = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix': os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
source = JournalStateStore(Path(source_path))
target = SQLiteProjection(Path(target_path), fault_hook=crash)
worker = ProjectionWorker(source, target, poll_interval=0.1, retry_initial=0.02, retry_max=0.1)
worker.start()
threading.Event().wait(15)
raise AssertionError('worker did not reach the fault hook')
''', source.path, target.path, point)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    assert prepare_source(source) == acknowledged
    reopened = SQLiteProjection(target.path)
    if point != "after_commit":
        assert reopened.read() == prior
    else:
        assert projection_status(source, reopened)["matches_pinned_source"]

    # A fresh process-equivalent service has no retry queue to restore. Its first
    # pass derives pending work entirely from the verified source/target cursors.
    source = JournalStateStore(source.path)
    worker = ProjectionWorker(source, reopened, poll_interval=0.05, retry_initial=0.02, retry_max=0.1)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not projection_status(source, reopened)["matches_pinned_source"]:
            assert time.monotonic() < deadline, worker.status()
            time.sleep(0.01)
        assert len(query(source, reopened)["results"]) == 2
        assert prepare_source(source) == acknowledged
        # Receipt replay acknowledges the same committed revision after recovery.
        assert write(source, "two", text="Receipt survives the background worker")["revision"] == 2
        assert prepare_source(source) == acknowledged
    finally:
        assert worker.close(timeout=5)
