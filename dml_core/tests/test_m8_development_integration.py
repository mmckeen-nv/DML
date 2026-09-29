"""Three real memory paths with scripted protocol inputs, never model evidence."""

from dataclasses import replace
import hashlib
import os
import time
import numpy as np
import pytest
from daystrom_dml.services.agent_episode import _fixture_adapter
from daystrom_dml.services.m8_memory_baseline import MemoryEvent, PersistentRAGBaseline, ContextBuffer
from daystrom_dml.services.m8_dml_memory import DMLDevelopmentMemory
from daystrom_dml.services import m8_protocol as protocol
from daystrom_dml.services.m8_accounting import DevelopmentAccounting

SCOPE = dict(tenant_id="m8-control", client_id=None, session_id="isolated-project", instance_id=None)


class ByteTokenizer:
    chat_template = "explicit-development-tokenizer-double"

    def encode(self, text, *, add_special_tokens=False, truncation=False, padding=False):
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "".join("[" + m["role"] + "]" + m["content"] for m in messages) + "[assistant]"


def test_three_arm_correction_feedback_and_visible_citation_contract(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("DML_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    tokenizer = ByteTokenizer()
    monkeypatch.setattr(
        protocol, "PINNED_TEMPLATE_SHA256", hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    )
    adapter = _fixture_adapter(tmp_path / "dml")
    dml = DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=tmp_path / "receipts.jsonl")
    rag = PersistentRAGBaseline(tmp_path / "rag.sqlite", embedding_identity="unit-test-vector-v1", dimensions=2)
    context = ContextBuffer(
        scope=SCOPE,
        token_budget=4096,
        count_tokens=lambda s: protocol.count_tokens(tokenizer, s),
        render_records=protocol.render_records,
    )
    first = MemoryEvent(
        10, "Project status is amber.", SCOPE, "bulletin", "status", 1, 1000, metadata={"source_trust": "trusted"}
    )
    second = replace(first, record_id=11, text="Project status is green.", version=2, timestamp=1001, corrects=10)
    try:
        for event in (first, second):
            dml.ingest(event)
            rag.ingest(event, np.array([1.0, 0.0], dtype=np.float32))
            context.ingest(event)
        options = dict(
            top_k=8,
            token_budget=2048,
            count_tokens=lambda s: protocol.count_tokens(tokenizer, s),
            render_records=protocol.render_records,
        )
        evidence = {
            "context_only": context.present(),
            "persistent_rag": rag.retrieve([1.0, 0.0], scope=SCOPE, now=time.time(), **options),
            "dml": dml.retrieve("Project status", now=time.time(), **options)["evidence"],
        }
        assert [r["record_id"] for r in evidence["context_only"].records] == [10, 11]
        for arm in ("persistent_rag", "dml"):
            assert [r["record_id"] for r in evidence[arm].records] == [11]
        # Scripted actions verify plumbing only; these are not model-chosen
        # answers, a live qualification, or a held-out benchmark workload.
        for arm, shown in evidence.items():
            ledger = DevelopmentAccounting(clock=lambda: 0)
            messages = protocol.initial_messages(
                arm, "What is project status?", records=shown.records if arm == "context_only" else (), scope=SCOPE
            )
            compiled = protocol.compile_request(tokenizer, arm, messages, scope=SCOPE)
            ledger.admit("call-1", compiled.input_ids)
            if arm != "context_only":
                raw = protocol.data_json(
                    {
                        "schema_version": "dml-agent-action-v1",
                        "kind": "tool",
                        "name": "retrieve",
                        "arguments": {"query": "Project status", "top_k": 8},
                    }
                )
                ledger.complete("call-1", tokenizer.encode(raw))
                messages += protocol.tool_feedback(
                    raw, arm=arm, call_id="retrieve-1", records=shown.records, scope=SCOPE
                )
                compiled = protocol.compile_request(
                    tokenizer, arm, messages, scope=SCOPE, remaining_input_tokens=32768 - ledger.input_tokens
                )
                ledger.admit("call-2", compiled.input_ids)
            answer = {
                "schema_version": "dml-agent-action-v1",
                "kind": "final",
                "answer": {"claims": [{"key": "status", "value": "green", "evidence_ids": [11]}]},
            }
            checked = protocol.validate_final(
                answer,
                shown.records,
                scope=SCOPE,
                supports=lambda claim, record: (
                    claim["key"] == "status"
                    and claim["value"] == "green"
                    and record["text"] == "Project status is green."
                ),
            )
            assert checked == answer["answer"]
            ledger.complete(
                "call-1" if arm == "context_only" else "call-2", tokenizer.encode(protocol.data_json(answer))
            )
            assert not ledger.snapshot()["usage_unknown"]
            bad = replace(second, record_id=99)
            with pytest.raises(ValueError):
                protocol.validate_final(answer, [bad.__dict__], scope=SCOPE, supports=lambda c, r: True)
    finally:
        adapter.close()
        rag.close()
