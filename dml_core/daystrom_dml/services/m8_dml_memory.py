"""Development-only neutral projection of genuine DML receipted memory.

This bridge does not qualify historical event-clock parity or a model runtime.
It never rewrites authority timestamps. Incomplete multi-operation ingestion is
retained and fail-closed, including across bridge restart.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

from ..contracts.profile import PROFILE_ID
from ..contracts.agent_episode import decode_json
from .episode_tools import _digest, _scope
from .m8_memory_baseline import MemoryEvent, validate_event, present_records


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class DMLDevelopmentMemory:
    """Owned scope, authentic receipts, reversible source-ID presentation.

    The JSONL evidence file is immutable append-only by this bridge. Its hash
    chain detects accidental alteration, not an adversary rewriting the whole
    file; external freeze hashes remain required. No interrupted append is retried.
    """

    version = "dml-m8-development-memory-v1"
    event_clock_qualified = False

    def __init__(self, adapter, *, scope, evidence_path):
        if adapter.production_profile_id != PROFILE_ID:
            raise ValueError("supported receipt profile required")
        self.adapter, self.scope = adapter, _scope(scope)
        self.path = Path(evidence_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events, self.records, self.receipts = {}, {}, []
        self.failed = False
        self._head = "0" * 64
        self._count = 0
        self._load()
        self._assert_owned_state()

    def _load(self):
        if not self.path.exists():
            return
        pending = None
        for line in self.path.read_bytes().splitlines():
            row = decode_json(line, limit=16 * 1024 * 1024)
            digest = row.pop("sha256")
            if (
                row.get("previous") != self._head
                or row.get("sequence") != self._count
                or hashlib.sha256(_bytes(row)).hexdigest() != digest
                or row.get("scope") != self.scope
                or row.get("version") != self.version
            ):
                raise ValueError("invalid DML development evidence chain")
            kind, payload = row["kind"], row["payload"]
            if kind == "begin":
                if pending is not None:
                    raise ValueError("overlapping ingestion")
                pending = payload
            elif kind == "receipt":
                if pending is None:
                    raise ValueError("receipt without ingestion")
                self.receipts.append(payload)
            elif kind == "commit":
                if pending is None or pending != payload["event"]:
                    raise ValueError("commit without matching event")
                self._accept(payload)
                pending = None
            else:
                raise ValueError("unknown evidence event")
            self._head, self._count = digest, self._count + 1
        if pending is not None:
            self.failed = True
            raise ValueError("unresolved ingestion; retained evidence requires investigation")

    def _assert_owned_state(self):
        # Exact scope ownership: an unselected extra row can change top-k, and
        # removal can produce an empty result. Neither may escape drift checks.
        _, _, snapshot = self.adapter._journal.verified_snapshot()
        actual = {
            r["id"]: r
            for bucket in ("items", "lineage")
            for r in snapshot.get(bucket, [])
            if all(r["meta"].get(k) == v for k, v in self.scope.items())
        }
        expected = {r["id"]: r for r in self.records.values()}
        if actual != expected:
            self.failed = True
            raise ValueError("DML authority differs from acknowledged owned state")

    def _append(self, kind, payload):
        row = {
            "version": self.version,
            "scope": self.scope,
            "sequence": self._count,
            "previous": self._head,
            "kind": kind,
            "payload": payload,
        }
        digest = hashlib.sha256(_bytes(row)).hexdigest()
        row["sha256"] = digest
        with self.path.open("ab") as stream:
            stream.write(_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._head, self._count = digest, self._count + 1

    def _accept(self, payload):
        event = payload["event"]
        self.events[event["record_id"]] = deepcopy(event)
        for key, record in payload["records"].items():
            self.records[int(key)] = deepcopy(record)

    def _ready(self):
        if self.failed:
            raise ValueError("unresolved ingestion; no further dispatch permitted")
        self._assert_owned_state()

    def ingest(self, event: MemoryEvent):
        self._ready()
        value = validate_event(event)
        if value["scope"] != self.scope:
            raise ValueError("event outside owned scope")
        if event.record_id in self.events:
            raise ValueError("duplicate event; no automatic retry")
        if self.events and event.timestamp < max(e["timestamp"] for e in self.events.values()):
            raise ValueError("nonchronological input")
        related = [
            e for e in self.events.values() if (e["source_id"], e["fact_key"]) == (event.source_id, event.fact_key)
        ]
        if any(e["version"] == event.version for e in related):
            raise ValueError("duplicate source/fact version")
        if event.metadata.get("source_trust", "untrusted") not in ("trusted", "untrusted"):
            raise ValueError("invalid explicit source trust")
        previous = None
        if event.corrects is not None:
            previous = self.events.get(event.corrects)
            if (
                previous is None
                or previous["source_id"] != event.source_id
                or previous["fact_key"] != event.fact_key
                or event.version <= previous["version"]
                or event.timestamp < previous["timestamp"]
                or any(e["corrects"] == event.corrects for e in self.events.values())
            ):
                raise ValueError("invalid or stale explicit correction")
        elif related:
            raise ValueError("new version requires explicit current predecessor")
        # Mark uncertain before the first durable write; any exception including
        # evidence I/O failure prevents reads or another mutation in this object.
        self.failed = True
        self._append("begin", value)
        meta = {
            "source": event.source_id,
            "source_trust": event.metadata.get("source_trust", "untrusted"),
            "m8_event": value,
        }
        receipt = self.adapter.ingest_memory_receipted(
            event.text, idempotency_key=f"m8:{event.record_id}:append", meta=meta, **self.scope
        )
        self._append("receipt", receipt)
        self.receipts.append(deepcopy(receipt))
        record = receipt["result"]["memory"]
        updates = {str(event.record_id): record}
        if previous is not None:
            old = self.records[event.corrects]
            supersession = self.adapter.supersede_memory_receipted(
                old["id"],
                replacement_memory_id=record["id"],
                expected_memory_digest=_digest(old),
                expected_replacement_digest=_digest(record),
                reason="explicit source correction",
                idempotency_key=f"m8:{event.record_id}:supersede",
                **self.scope,
            )
            self._append("receipt", supersession)
            self.receipts.append(deepcopy(supersession))
            updates[str(event.corrects)] = supersession["result"]["memory"]
        payload = {"event": value, "records": updates}
        self._append("commit", payload)
        self._accept(payload)
        self.failed = False
        return deepcopy(receipt)

    def retrieve(self, query, *, now, top_k=8, token_budget=2048, count_tokens, render_records):
        self._ready()
        if type(query) is not str or not query.strip():
            raise ValueError("nonempty query required")
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("finite query clock required")
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("invalid top_k")
        if any(event["timestamp"] > now for event in self.events.values()):
            raise ValueError("query precedes ingested event")
        report = self.adapter.retrieve_context(query, top_k=top_k, as_of=now, **self.scope)
        self._assert_owned_state()
        reverse = {str(r["id"]): key for key, r in self.records.items()}
        records = []
        for item in report["items"]:
            if item["id"] not in reverse or any(item.get("meta", {}).get(k) != v for k, v in self.scope.items()):
                raise ValueError("unowned retrieval result")
            event = self.events[reverse[item["id"]]]
            record = self.records[reverse[item["id"]]]
            projection = {
                "lattice_row",
                "lattice_col",
                "lattice_layer",
                "lattice_neighbors",
                "lattice_degree",
                "lattice_policy",
                "lattice_id",
                "lattice_index",
                "lattice_size_hint",
            }
            metadata = {k: v for k, v in item["meta"].items() if k not in projection}
            if metadata != {k: v for k, v in record["meta"].items() if k not in projection}:
                raise ValueError("retrieved source metadata differs")
            if any(record[k] != item.get(k) for k in ("timestamp", "level", "fidelity", "salience")):
                raise ValueError("retrieved source identity differs")
            excerpt, text = item["text"], event["text"].strip()
            if excerpt != text and not (excerpt.endswith("...") and text.startswith(excerpt[:-3])):
                raise ValueError("retrieved source text differs")
            records.append(deepcopy(event))
        return {
            "evidence": present_records(records, token_budget, count_tokens, render_records),
            "raw_report": deepcopy(report),
            "event_clock_qualified": False,
        }
