"""Development-only neutral runtime smoke on the pinned station environment.

One scoped three-event corpus (trusted correction pair plus an independent
conflict), one query, all three neutral arms against the pinned Llama 3 8B
SFTv3 snapshot. This qualifies protocol PLUMBING only: automatic retrieval
where available, strict parsing and valid cited finals. It is not a benchmark,
pilot, quality comparison or held-out workload; no answer keys are used by
drivers beyond fixture support verification, and every failure is retained.

Usage (inside the pinned container): see docs/m8-neutral-runtime-plan-2026-11-02.md.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

SNAPSHOT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_IDENTITY = (
    "sentence-transformers/all-MiniLM-L6-v2@c9745ed1d9f207416be6d2e6f8de32d1f16199bf+cpu-fp32-meanpool-l2-384-v1"
)
SCOPE = {"tenant_id": "m8-smoke", "client_id": None, "session_id": "project", "instance_id": None}
QUERY = "What is the project status?"
QUERY_CLOCK = 2000.0
MAX_MODEL_CALLS = 6


class MiniLMEmbedder:
    """Pinned MiniLM embeddings without sentence-transformers (same semantics).

    CPU float32, batch one, mean pooling over attention tokens, L2 normalize.
    At most 192 content tokens; silent truncation is rejected.
    """

    model_name = EMBEDDING_MODEL

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(4)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=SNAPSHOT_REVISION, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, revision=SNAPSHOT_REVISION, local_files_only=True, dtype=torch.float32
        ).eval()

    def embed(self, text):
        import numpy as np
        import torch

        if type(text) is not str or not text.strip():
            raise ValueError("Embedding requires nonempty text")
        batch = self._tokenizer(text, return_tensors="pt", truncation=False)
        if int(batch["attention_mask"].sum()) > 192:
            raise ValueError("Event exceeds 192 content tokens; chunking is out of smoke scope")
        with torch.inference_mode():
            hidden = self._model(**batch).last_hidden_state[0]
        mask = batch["attention_mask"][0].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(0) / mask.sum()
        vector = torch.nn.functional.normalize(pooled, p=2, dim=0).numpy().astype(np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError("Invalid embedding row")
        return vector


def fixture_events():
    from daystrom_dml.services.m8_memory_baseline import MemoryEvent

    def make(record_id, text, source, version, timestamp, corrects=None):
        return MemoryEvent(
            record_id=record_id,
            text=text,
            scope=SCOPE,
            source_id=source,
            fact_key="status",
            version=version,
            timestamp=timestamp,
            corrects=corrects,
            metadata={"source_trust": "trusted"},
        )

    return [
        make(10, "Project status is amber.", "bulletin", 1, 1000),
        make(11, "Project status is green.", "bulletin", 2, 1001, corrects=10),
        make(12, "Project status is red.", "independent", 1, 1002),
    ]


def status_supports(claim, record):
    """Fixture domain verifier: a claim is supported iff the cited record states it."""
    if type(claim) is not dict or claim.get("key") != "status":
        return False
    text = record.get("text", "")
    prefix = "Project status is "
    if not (text.startswith(prefix) and text.endswith(".")):
        return False
    return text[len(prefix) : -1] == claim.get("value")


def build_dml_bridge(embedder, workdir):
    from daystrom_dml.dml_adapter import DMLAdapter
    from daystrom_dml.settings import DMLSettings
    from daystrom_dml.services.m8_dml_memory import DMLDevelopmentEventMemory

    config = {
        "production_profile": "dml-receipted-local-v1",
        "storage_dir": str(workdir / "dml-authority"),
        "model_name": "dummy",
        "llm_backend": "dummy",
        "embedding_model": EMBEDDING_MODEL,
        "strict_embedding_required": True,
        "persistence": {
            "enable": False,
            "interval_sec": 0,
            "journal": True,
            "receipts": True,
            "outbox": False,
            "receipt_embedding_identity": EMBEDDING_IDENTITY,
            "snapshot_interval": 1,
        },
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False, "include_in_preamble": False},
        "ann_min_items": 0,
        "checkpoint_interval_seconds": 0,
        "skip_rag_state_import": True,
        "survival_ledger_enabled": False,
        "background_processing_enabled": False,
        "enable_stm_controller": False,
        "enable_workflow_cache": False,
        "enable_quality_on_retrieval": False,
        "mirror_agentic_memory_to_rag": False,
        "gpu_acceleration": False,
        "metrics_enabled": False,
    }
    adapter = DMLAdapter(_validated_settings=DMLSettings(**config), embedder=embedder, start_aging_loop=False)
    bridge = DMLDevelopmentEventMemory(adapter, scope=SCOPE, evidence_path=workdir / "dml-evidence.jsonl")
    return adapter, bridge


def run_arm(arm, snapshot, embedder, stores):
    """One neutral consumer lifecycle: compile, execute, retrieve, final."""
    from daystrom_dml.services import m8_protocol as protocol
    from daystrom_dml.services.m8_neutral_runtime import M8NeutralConsumer

    record = {
        "arm": arm,
        "calls": [],
        "presented_record_ids": [],
        "verdict": None,
        "failure": None,
    }
    context_records = list(stores["context_records"]) if arm == "context_only" else []
    messages = protocol.initial_messages(arm, QUERY, records=context_records, scope=SCOPE)
    presented = list(context_records)
    try:
        with M8NeutralConsumer(snapshot, arm=arm, offline=False) as consumer:
            record["runtime_identity"] = consumer.identity["runtime_identity"]

            def count(text):
                # Exact pinned tokenizer already admitted by this consumer.
                return protocol.count_tokens(consumer._tokenizer, text)

            for step in range(1, MAX_MODEL_CALLS + 1):
                compiled = consumer.compile(messages, scope=SCOPE)
                result = consumer.execute(compiled)
                call = {
                    "step": step,
                    "request_digest": compiled.request_digest,
                    "input_token_count": result["input_token_count"],
                    "output_token_count": result["output_token_count"],
                    "finish_reason": result.get("finish_reason"),
                    "action": result.get("derived_action"),
                    "action_error": result.get("action_error"),
                }
                record["calls"].append(call)
                action = result.get("derived_action")
                if action is None:
                    record["failure"] = "unparseable action retained: " + str(result.get("action_error"))
                    break
                if action["kind"] == "final":
                    answer = protocol.validate_final(action, presented, scope=SCOPE, supports=status_supports)
                    record["verdict"] = "valid_cited_final"
                    record["answer"] = answer
                    break
                # Tool action: retrieve through THIS arm's store and continue.
                args = action["arguments"]
                if arm == "persistent_rag":
                    ev = stores["rag"].retrieve(
                        embedder.embed(args["query"]),
                        scope=SCOPE,
                        now=QUERY_CLOCK,
                        top_k=args["top_k"],
                        token_budget=2048,
                        count_tokens=count,
                        render_records=protocol.render_records,
                    )
                elif arm == "dml":
                    ev = stores["bridge"].retrieve(
                        args["query"],
                        now=QUERY_CLOCK,
                        top_k=args["top_k"],
                        token_budget=2048,
                        count_tokens=count,
                        render_records=protocol.render_records,
                    )["evidence"]
                else:
                    record["failure"] = "context_only issued a tool action"
                    break
                presented.extend(deepcopy(list(ev.records)))
                messages += protocol.tool_feedback(
                    result["derived_action_text"],
                    arm=arm,
                    call_id=f"retrieve-{step}",
                    records=list(ev.records),
                    scope=SCOPE,
                )
            else:
                record["failure"] = "model call ceiling reached without final"
    except Exception as exc:  # noqa: BLE001 — every failure is retained evidence
        record["failure"] = f"{type(exc).__name__}: {exc}"
    record["presented_record_ids"] = [r["record_id"] for r in presented]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(args.snapshot)
    started = time.time()

    from daystrom_dml.services import m8_protocol as protocol
    from daystrom_dml.services.m8_memory_baseline import ContextBuffer, PersistentRAGBaseline

    embedder = MiniLMEmbedder()
    events = fixture_events()

    rag = PersistentRAGBaseline(workdir / "rag.sqlite", embedding_identity=EMBEDDING_IDENTITY, dimensions=384)
    adapter, bridge = build_dml_bridge(embedder, workdir)
    llama_tokenizer_holder = {}

    def llama_count(text):
        if "tok" not in llama_tokenizer_holder:
            from transformers import AutoTokenizer

            llama_tokenizer_holder["tok"] = AutoTokenizer.from_pretrained(
                str(snapshot / "model"), local_files_only=True, trust_remote_code=False
            )
        return protocol.count_tokens(llama_tokenizer_holder["tok"], text)

    buffer = ContextBuffer(
        scope=SCOPE, token_budget=2048, count_tokens=llama_count, render_records=protocol.render_records
    )
    for event in events:
        rag.ingest(event, embedder.embed(event.text))
        bridge.ingest(event)
        buffer.ingest(event)
    rag.close()
    # Reopen to prove durable ingestion before the query path.
    rag = PersistentRAGBaseline(workdir / "rag.sqlite", embedding_identity=EMBEDDING_IDENTITY, dimensions=384)

    stores = {"rag": rag, "bridge": bridge, "context_records": buffer.present().records}
    arms = {}
    try:
        for arm in ("context_only", "persistent_rag", "dml"):
            arms[arm] = run_arm(arm, snapshot, embedder, stores)
    finally:
        adapter.close()
        rag.close()

    outcome = {
        "schema": "dml.m8.neutral-smoke.v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "scope": SCOPE,
        "events": [json.loads(json.dumps(e.__dict__)) for e in events],
        "query": QUERY,
        "query_clock": QUERY_CLOCK,
        "embedding_identity": EMBEDDING_IDENTITY,
        "arms": arms,
        "verdict": (
            "all_arms_valid_cited_final"
            if all(a.get("verdict") == "valid_cited_final" for a in arms.values())
            else "incomplete_or_failed_retained"
        ),
        "limitations": (
            "Development plumbing smoke only: one fixture corpus, one query, scripted "
            "retrieval dispatch, no benchmark, no held-out data, no quality claim."
        ),
    }
    Path(args.output).write_text(json.dumps(outcome, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                arm: {
                    "verdict": a["verdict"],
                    "failure": a["failure"],
                    "calls": len(a["calls"]),
                    "presented": len(a["presented_record_ids"]),
                }
                for arm, a in arms.items()
            },
            indent=1,
        )
    )
    print("verdict:", outcome["verdict"])
    return 0 if outcome["verdict"] == "all_arms_valid_cited_final" else 1


if __name__ == "__main__":
    sys.exit(main())
