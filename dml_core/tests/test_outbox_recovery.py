"""Actual process-death oracles for transactional source events and delivery."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from daystrom_dml.journal import (
    FAULT_POINTS, INIT_FAULT_POINTS,
    JournalIntegrityError, JournalSchemaError, JournalStateStore,
)
from test_projection import run_child, write

CRASH = '''
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
            threading.Event().wait(5)
            os._exit(80)
        else:
            os._exit(79)
'''


@pytest.fixture
def authority(tmp_path):
    source = JournalStateStore(tmp_path / "authority" / "source.sqlite3",
                               receipt_mode=True, outbox_mode=True)
    write(source)
    return source


def assert_killed(child):
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr


@pytest.mark.parametrize("point", INIT_FAULT_POINTS)
def test_outbox_initialization_kill_has_explicit_reopen_state(tmp_path, point):
    path = tmp_path / "source.sqlite3"
    code = '''
import os, signal, sys, threading
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
source_path, point = sys.argv[1:]
''' + CRASH + '''
JournalStateStore(Path(source_path), receipt_mode=True, outbox_mode=True, fault_hook=crash)
raise AssertionError('initialization fault hook not reached')
'''
    assert_killed(run_child(code, path, point))
    if point in ("init_after_connect", "init_after_begin", "init_after_schema"):
        damaged = path.read_bytes()
        with pytest.raises(JournalSchemaError):
            JournalStateStore(path, receipt_mode=True, outbox_mode=True)
        assert path.read_bytes() == damaged
    else:
        source = JournalStateStore(path, receipt_mode=True, outbox_mode=True)
        assert source.schema_version == 3
        assert source.outbox_events()["events"] == []
        assert source.outbox_events()["head_revision"] == 0


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_consumer_kill_never_acknowledges_a_partial_event(authority, tmp_path, point):
    from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer, deliver_outbox

    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "inbox.sqlite3")
    before = consumer.read()
    source_before = authority.outbox_events()
    code = '''
import os, signal, sys, threading
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer, deliver_outbox
source_path, consumer_path, point = sys.argv[1:]
''' + CRASH + '''
source = JournalStateStore(Path(source_path))
consumer = SQLiteOutboxConsumer(Path(consumer_path), fault_hook=crash)
deliver_outbox(source, consumer)
raise AssertionError('consumer fault hook not reached')
'''
    assert_killed(run_child(code, authority.path, consumer.path, point))
    assert authority.outbox_events() == source_before
    reopened = SQLiteOutboxConsumer(consumer.path)
    state = reopened.read()
    if point != "after_commit":
        assert state == before
    else:
        assert state["cursor"]["source_revision"] == 1
        assert state["last_event"] == source_before["events"][0]
    deliver_outbox(authority, reopened)
    assert reopened.read()["last_event"] == source_before["events"][0]
    assert reopened.read()["event_checksums"] == [source_before["events"][0]["checksum"]]
    assert authority.outbox_events() == source_before


@pytest.mark.parametrize("point", (*INIT_FAULT_POINTS, *FAULT_POINTS))
def test_consumer_construction_kill_is_complete_or_requires_new_location(tmp_path, point):
    from daystrom_dml.services.outbox_delivery import OutboxDeliveryError, SQLiteOutboxConsumer

    path = tmp_path / "consumer" / "inbox.sqlite3"
    code = '''
import os, signal, sys, threading
from pathlib import Path
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
consumer_path, point = sys.argv[1:]
''' + CRASH + '''
SQLiteOutboxConsumer(Path(consumer_path), fault_hook=crash)
raise AssertionError('consumer construction hook not reached')
'''
    assert_killed(run_child(code, path, point))
    if point in ("init_before_connect", "after_commit"):
        consumer = SQLiteOutboxConsumer(path)
        assert consumer.read()["cursor"] is None
        assert consumer.read()["event_checksums"] == []
    else:
        # Neither a bare initialized journal nor a torn schema is a consumer.
        # Preserve it and choose a separate location for explicit recovery.
        with pytest.raises((JournalIntegrityError, OutboxDeliveryError)):
            SQLiteOutboxConsumer(path)
        assert path.exists()
        fresh = SQLiteOutboxConsumer(tmp_path / "recovered" / "inbox.sqlite3")
        assert fresh.read()["cursor"] is None
