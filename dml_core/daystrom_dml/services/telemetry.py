"""Deterministic, payload-free evidence for the context actually returned."""
import hashlib
import json


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def retrieval_decision(*, prompt, scope, revision, as_of, entries, context,
                       top_k, kinds, embedding, suppressed, replayable):
    evidence = {
        "schema_version": 1, "policy_version": "scoped-exact-v1",
        "query_digest": digest(prompt), "scope": scope, "store_revision": revision,
        "effective_time": as_of, "top_k": top_k, "kinds": kinds,
        "query_embedding_digest": digest(embedding.tolist()),
        "returned_ids": [entry["id"] for entry in entries],
        "source_digests": [digest(entry.get("meta", {})) for entry in entries],
        "suppressed": suppressed, "context_digest": digest(context),
        "replay_inputs_complete": False,
        "deterministic_scoped_ranking": replayable,
    }
    # Content/embedding model identities and the original query are deliberately
    # not guessed. Callers retain their request/config to reconstruct a turn.
    return {**evidence, "decision_digest": digest(evidence)}
