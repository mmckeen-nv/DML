"""High level adapter orchestrating the Daystrom Memory Lattice."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .embeddings import Embedder, create_embedder
from .gpt_runner import GPTRunner
from .memory_store import MemoryStore
from .summarizer import DummySummarizer, LLMSummarizer, Summarizer
from .retrievers import LiteralRetriever, LiteralResult
from .multi_rag import MultiRAGStore
from . import utils

LOGGER = logging.getLogger(__name__)

# Limit the amount of data returned by the knowledge endpoint to keep the
# payload responsive even when the lattice contains thousands of memories.
KNOWLEDGE_MAX_ENTRIES = 200
KNOWLEDGE_ENTRY_PREVIEW_CHARS = 320

STARFLEET_BANNER = "\n".join(
    [
        "Initializing Daystrom Memory Lattice v1.0",
        "Semantic coherence field stabilized.",
        "Cognitive resonance online.",
    ]
)


class DMLAdapter:
    """Facade used by the CLI and service to interact with the DML."""

    def __init__(
        self,
        config_path: str | os.PathLike | None = None,
        *,
        config_overrides: Optional[Dict] = None,
        embedder: Optional[Embedder] = None,
        summarizer: Optional[Summarizer] = None,
        runner: Optional[GPTRunner] = None,
        start_aging_loop: bool = True,
    ) -> None:
        self.config = self._load_config(config_path)
        if config_overrides:
            self.config.update(config_overrides)
        self.config.setdefault("dml_top_k", 0)
        self.runner = runner or GPTRunner(self.config["model_name"])
        self.embedder = embedder or create_embedder(self.config.get("embedding_model"))
        if summarizer is not None:
            self.summarizer = summarizer
        elif self.runner.is_dummy:
            self.summarizer = DummySummarizer()
        else:
            self.summarizer = LLMSummarizer(self.runner)
        storage_dir = self.config.get("storage_dir") or "data"
        self.storage_dir = Path(storage_dir)
        if not self.storage_dir.is_absolute():
            self.storage_dir = Path(__file__).resolve().parent.parent / self.storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dml_state_path = self.storage_dir / "dml_store.json"
        self.rag_state_path = self.storage_dir / "rag_store.json"
        self._persist_lock = RLock()
        self.literal_retriever = LiteralRetriever(
            self.embedder,
            self.summarizer,
            context_window=int(self.config.get("literal_context", 1)),
        )
        self.rag_store = MultiRAGStore(self.embedder)
        self.store = MemoryStore(
            self.summarizer,
            beta_a=float(self.config["beta_a"]),
            beta_r=float(self.config["beta_r"]),
            eta=float(self.config["eta"]),
            gamma=float(self.config["gamma"]),
            kappa=float(self.config["kappa"]),
            tau_s=float(self.config["tau_s"]),
            theta_merge=float(self.config["theta_merge"]),
            K=int(self.config["K"]),
            capacity=int(self.config["capacity"]),
            start_aging_loop=start_aging_loop,
        )
        self._load_persisted_state()
        LOGGER.info("Daystrom Memory Lattice initialised with %d capacity", self.store.capacity)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._persist_all()
        self.store.close()

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def ingest(self, text: str, meta: Optional[Dict] = None) -> None:
        if not text:
            return
        embedding = self.embedder.embed(text)
        salience = self._estimate_salience(text)
        self.store.ingest(text, embedding, salience=salience, meta=meta)
        self.rag_store.add_document(text, meta=meta)
        self._persist_all()

    def build_preamble(self, prompt: str, top_k: Optional[int] = None) -> str:
        items = self._retrieve_items(prompt, top_k)
        _, preamble, _ = self._prepare_context(prompt, items)
        return preamble

    def reinforce(self, prompt: str, response: str, meta: Optional[Dict] = None) -> None:
        if not response:
            return
        combined = f"Prompt: {prompt}\nResponse: {response}".strip()
        summary = self.summarizer.summarize(combined, max_len=256)
        embedding = self.embedder.embed(summary)
        salience = self._estimate_salience(summary) + 0.1
        self.store.ingest(summary, embedding, salience=salience, meta=meta)
        self._persist_dml_state()

    def run_generation(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        context = self.build_preamble(prompt)
        augmented_prompt = f"{context}\n\n{prompt}"
        response = self.runner.generate(augmented_prompt, max_new_tokens=max_new_tokens)
        self.reinforce(prompt, response)
        return response

    def retrieval_report(self, prompt: str, *, top_k: Optional[int] = None) -> Dict:
        start = time.perf_counter()
        items = self._retrieve_items(prompt, top_k)
        entries, preamble, tokens_used = self._prepare_context(prompt, items)
        fidelities = [entry["fidelity"] for entry in entries]
        avg_fidelity = float(np.mean(fidelities) if fidelities else 0.0)
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return {
            "entries": entries,
            "preamble": preamble,
            "tokens": tokens_used,
            "avg_fidelity": avg_fidelity,
            "latency_ms": latency_ms,
        }

    def compare_responses(
        self,
        prompt: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> Dict:
        step_counter = 0
        pipeline_trace: List[Dict[str, Any]] = []

        base_response, base_usage, base_latency = self._generate_with_metrics(
            prompt, max_new_tokens=max_new_tokens
        )
        step_counter += 1
        base_sequence = step_counter
        pipeline_trace.append(
            {
                "step": step_counter,
                "stage": "base",
                "label": "Base model",
            }
        )

        rag_top_k = self.config.get("top_k", 6) if top_k is None else top_k
        rag_reports = self.rag_store.report_all(prompt, top_k=rag_top_k)
        rag_results: List[Dict] = []
        for report in rag_reports:
            if not report.get("available", True):
                rag_results.append(
                    {
                        **report,
                        "response": "",
                        "usage": None,
                        "context_tokens": report.get("tokens", 0),
                        "sequence": None,
                        "generation_latency_ms": 0,
                        "retrieval_latency_ms": report.get("latency_ms", 0),
                    }
                )
                continue
            rag_context = self._format_rag_context(report.get("context") or "")
            rag_prompt = self._compose_prompt(prompt, rag_context)
            rag_response, rag_usage, rag_latency = self._generate_with_metrics(
                rag_prompt, max_new_tokens=max_new_tokens
            )
            step_counter += 1
            rag_entry = {
                "id": report.get("id"),
                "label": report.get("label"),
                "strategy": report.get("strategy"),
                "response": rag_response,
                "usage": rag_usage,
                "context": rag_context,
                "context_tokens": report.get("tokens"),
                "documents": report.get("documents"),
                "sequence": step_counter,
                "retrieval_latency_ms": report.get("latency_ms", 0),
                "generation_latency_ms": rag_latency,
                "available": True,
                "error": report.get("error"),
            }
            rag_results.append(rag_entry)
            pipeline_trace.append(
                {
                    "step": step_counter,
                    "stage": "rag",
                    "id": rag_entry["id"],
                    "label": rag_entry.get("label"),
                }
            )
        dml_report = self.retrieval_report(prompt, top_k=top_k)
        dml_context = self._format_dml_context(dml_report["entries"])
        dml_prompt = self._compose_prompt(prompt, dml_context)
        dml_response, dml_usage, dml_latency = self._generate_with_metrics(
            dml_prompt, max_new_tokens=max_new_tokens
        )
        self.reinforce(prompt, dml_response)
        step_counter += 1
        dml_sequence = step_counter
        pipeline_trace.append(
            {
                "step": step_counter,
                "stage": "dml",
                "label": "Daystrom memory lattice",
            }
        )

        dml_reference = dml_response or ""
        evaluations = []
        if dml_reference.strip():
            for entry in rag_results:
                if not entry.get("available"):
                    entry["grade"] = {
                        "score": 0.0,
                        "grade": "N/A",
                        "explanation": entry.get("error") or "Backend unavailable",
                    }
                    continue
                grade = self._grade_response(entry.get("response", ""), dml_reference)
                entry["grade"] = grade
                evaluations.append(
                    {
                        "backend_id": entry.get("id"),
                        "score": grade.get("score", 0.0),
                        "grade": grade.get("grade"),
                    }
                )
        else:
            for entry in rag_results:
                entry["grade"] = {
                    "score": 0.0,
                    "grade": "N/A",
                    "explanation": "DML response unavailable for comparison.",
                }

        return {
            "prompt": prompt,
            "base": {
                "response": base_response,
                "usage": base_usage,
                "sequence": base_sequence,
                "generation_latency_ms": base_latency,
            },
            "rag_backends": rag_results,
            "dml": {
                "response": dml_response,
                "usage": dml_usage,
                "context": dml_context,
                "context_tokens": utils.estimate_tokens(dml_context),
                "avg_fidelity": dml_report["avg_fidelity"],
                "entries": dml_report["entries"],
                "sequence": dml_sequence,
                "retrieval_latency_ms": dml_report.get("latency_ms", 0),
                "generation_latency_ms": dml_latency,
            },
            "rag_token_breakdown": [
                {
                    "id": entry.get("id"),
                    "label": entry.get("label"),
                    "strategy": entry.get("strategy"),
                    "tokens": entry.get("context_tokens", 0),
                    "sequence": entry.get("sequence"),
                    "retrieval_latency_ms": entry.get("retrieval_latency_ms", 0),
                }
                for entry in rag_results
                if entry.get("available")
            ],
            "pipeline_trace": pipeline_trace,
            "evaluations": evaluations,
        }

    def knowledge_report(self) -> Dict:
        """Expose summaries of the RAG corpus and DML memory lattice."""

        rag_summary = self.rag_store.catalog_summary()

        dml_items = []
        dml_total_tokens = 0
        store_items = self.store.items()
        truncated = len(store_items) > KNOWLEDGE_MAX_ENTRIES
        for index, item in enumerate(store_items):
            text = (item.text or "").strip()
            tokens = utils.estimate_tokens(text)
            dml_total_tokens += tokens
            if index < KNOWLEDGE_MAX_ENTRIES:
                dml_items.append(
                    {
                        "id": item.id,
                        "level": item.level,
                        "fidelity": item.fidelity,
                        "tokens": tokens,
                        "summary": self._trim_summary(text),
                        "meta": item.meta or {},
                    }
                )

        return {
            "rag": rag_summary,
            "dml": {
                "entries": dml_items,
                "total_tokens": dml_total_tokens,
                "count": len(store_items),
                "truncated": truncated,
                "display_limit": KNOWLEDGE_MAX_ENTRIES,
            },
        }

    def _generate_with_metrics(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> tuple[str, Optional[dict], int]:
        start = time.perf_counter()
        response = self.runner.generate(prompt, max_new_tokens=max_new_tokens)
        usage = self.runner.last_usage
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return response, usage, latency_ms

    def _grade_response(self, candidate: str, reference: str) -> Dict[str, Any]:
        candidate = (candidate or "").strip()
        reference = (reference or "").strip()
        if not candidate:
            return {
                "score": 0.0,
                "grade": "F",
                "explanation": "No response generated by the backend.",
            }
        if not reference:
            return {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "Reference answer missing for comparison.",
            }
        try:
            cand_vec = np.asarray(self.embedder.embed(candidate), dtype=np.float32)
            ref_vec = np.asarray(self.embedder.embed(reference), dtype=np.float32)
            score = float(utils.cosine_similarity(cand_vec, ref_vec))
        except Exception:
            return {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "Failed to compute similarity for grading.",
            }
        grade = self._score_to_grade(score)
        return {
            "score": score,
            "grade": grade,
            "explanation": f"Cosine similarity to DML response: {score:.2f}",
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.6:
            return "C"
        if score >= 0.45:
            return "D"
        return "F"

    def _trim_summary(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= KNOWLEDGE_ENTRY_PREVIEW_CHARS:
            return text
        truncated = text[: KNOWLEDGE_ENTRY_PREVIEW_CHARS - 1].rstrip()
        return f"{truncated}…"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_persisted_state(self) -> None:
        with contextlib.suppress(Exception):
            if self.dml_state_path.exists():
                data = json.loads(self.dml_state_path.read_text(encoding="utf-8"))
                self._ensure_embedding_compatibility(data)
                self.store.import_state(data)
        with contextlib.suppress(Exception):
            if self.rag_state_path.exists():
                data = json.loads(self.rag_state_path.read_text(encoding="utf-8"))
                self.rag_store.import_state(data)

    def _persist_all(self) -> None:
        self._persist_dml_state()
        self._persist_rag_state()

    def _persist_dml_state(self) -> None:
        with self._persist_lock:
            data = self.store.export_state()
            tmp = self.dml_state_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.dml_state_path)
            except Exception:
                LOGGER.exception("Failed to persist DML state to %s", self.dml_state_path)

    def _persist_rag_state(self) -> None:
        with self._persist_lock:
            data = self.rag_store.export_state()
            tmp = self.rag_state_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.rag_state_path)
            except Exception:
                LOGGER.exception("Failed to persist RAG state to %s", self.rag_state_path)

    def _ensure_embedding_compatibility(self, payload: Dict) -> None:
        items = payload.get("items") if isinstance(payload, dict) else None
        if not items:
            return
        first = items[0] or {}
        stored_embedding = first.get("embedding")
        if stored_embedding is None:
            return
        try:
            stored_dim = int(np.asarray(stored_embedding, dtype=np.float32).size)
        except Exception:
            return
        try:
            probe = self.embedder.embed("Daystrom persistence probe")
            current_dim = int(np.asarray(probe, dtype=np.float32).size)
        except Exception:
            LOGGER.debug("Unable to determine embedder dimensions for persistence compatibility check.")
            return
        if stored_dim == current_dim or current_dim == 0:
            return
        LOGGER.warning(
            "Embedding dimension changed from %s to %s; re-embedding persisted memories.",
            stored_dim,
            current_dim,
        )
        for entry in items:
            text = entry.get("text") or ""
            try:
                new_embedding = self.embedder.embed(text)
                entry["embedding"] = utils.ensure_serializable(new_embedding)
            except Exception:
                entry["embedding"] = []

    def query_database(self, prompt: str, mode: str = "auto") -> Dict:
        """Retrieve context-aware snippets from the external corpus."""

        if mode not in {"semantic", "literal", "hybrid", "auto"}:
            raise ValueError(f"Unsupported mode: {mode}")
        selected_mode = mode if mode != "auto" else self._classify_mode(prompt)
        start = time.perf_counter()
        query_embedding = self.embedder.embed(prompt)
        items = self.store.items()
        dml_limit = self._resolve_dml_top_k(None)
        if dml_limit is None:
            top_k = max(len(items), int(self.config.get("top_k", 6)))
        else:
            top_k = dml_limit
        literal_results = []
        semantic_results = []
        if selected_mode in {"literal", "hybrid"}:
            literal_results = self.literal_retriever.retrieve(
                prompt, items, query_embedding, top_k=top_k
            )
        if selected_mode in {"semantic", "hybrid"}:
            semantic_results = self._semantic_retrieve(query_embedding, top_k=top_k)
        if selected_mode == "literal" and not literal_results:
            # fall back to semantic snippets if no literal hits
            semantic_results = self._semantic_retrieve(query_embedding, top_k=top_k)
        alpha = self._alpha_for_mode(selected_mode)
        combined = self._blend_results(literal_results, semantic_results, alpha, top_k=top_k)
        context_blocks = []
        sources: List[str] = []
        for entry in combined:
            source = entry.get("source") or "unknown"
            if source not in sources and entry.get("source"):
                sources.append(source)
            block_lines = [f"Source: {source}"]
            for segment in entry.get("context", []):
                block_lines.append(segment)
            context_blocks.append("\n".join(block_lines))
        context = "\n\n".join(context_blocks).strip()
        latency_ms = (time.perf_counter() - start) * 1000.0
        token_count = utils.estimate_tokens(context)
        return {
            "mode": selected_mode,
            "context": context,
            "source_docs": sources,
            "tokens": token_count,
            "latency_ms": int(latency_ms),
        }

    def stats(self) -> Dict:
        items = self.store.items()
        return {
            "count": len(items),
            "levels": {level: sum(1 for it in items if it.level == level) for level in range(self.store.K + 1)},
            "avg_fidelity": float(np.mean([it.fidelity for it in items]) if items else 0.0),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _format_rag_context(self, context: str) -> str:
        if not context:
            return ""
        return "=== RAG Retrieval ===\n" + context.strip()

    def _format_dml_context(self, entries: List[Dict]) -> str:
        if not entries:
            return ""
        lines = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        for entry in entries:
            lines.append(
                f"- L{entry['level']} (f={entry['fidelity']:.2f}): {entry['summary']}"
            )
        return "\n".join(lines)

    def _compose_prompt(self, prompt: str, context: str) -> str:
        blocks: List[str] = []
        if context:
            blocks.append(context.strip())
        blocks.append("=== User Prompt ===")
        blocks.append(prompt.strip())
        return "\n\n".join(blocks)

    def _load_config(self, path: str | os.PathLike | None) -> Dict:
        default_path = Path(__file__).with_name("config.yaml")
        env_override = os.environ.get("DML_CONFIG_PATH") or os.environ.get("DML_CONFIG")
        if path is not None:
            config_file = Path(path)
        elif env_override:
            config_file = Path(env_override)
        else:
            config_file = default_path
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data

    def _estimate_salience(self, text: str) -> float:
        tokens = utils.estimate_tokens(text)
        return float(max(0.1, min(1.0, tokens / 200.0)))

    def _retrieve_items(self, prompt: str, top_k: Optional[int]) -> List[MemoryStore.MemoryItem]:
        limit = self._resolve_dml_top_k(top_k)
        prompt_embedding = self.embedder.embed(prompt)
        return self.store.retrieve(prompt_embedding, top_k=limit)

    def _resolve_dml_top_k(self, requested: Optional[int]) -> Optional[int]:
        if requested is None:
            raw_value = self.config.get("dml_top_k")
        else:
            raw_value = requested
        if raw_value is None:
            return None
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid dml_top_k value %r; defaulting to unlimited retrieval.",
                raw_value,
            )
            return None
        if parsed <= 0:
            return None
        return parsed

    def _prepare_context(
        self, prompt: str, items: List[MemoryStore.MemoryItem]
    ) -> tuple[List[Dict], str, int]:
        budget = int(self.config.get("token_budget", 600))
        consumed = 0
        lines: List[str] = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        entries: List[Dict] = []
        for item in items:
            summary = self.summarizer.summarize(item.text, max_len=180)
            tokens = utils.estimate_tokens(summary)
            if consumed + tokens > budget:
                break
            consumed += tokens
            lines.append(f"- L{item.level} (f={item.fidelity:.2f}): {summary}")
            entries.append(
                {
                    "id": item.id,
                    "summary": summary,
                    "level": item.level,
                    "fidelity": float(item.fidelity),
                    "salience": float(item.salience),
                    "meta": item.meta or {},
                    "tokens": tokens,
                }
            )
        lines.append("=== User Prompt ===")
        lines.append(prompt)
        preamble = "\n".join(lines)
        return entries, preamble, consumed

    def _classify_mode(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        literal_signals = {
            "show",
            "exact",
            "line",
            "code",
            "api",
            "function",
            "table",
            "timestamp",
            "error",
            "log",
            "fetch",
        }
        semantic_signals = {
            "summary",
            "summarize",
            "average",
            "overview",
            "trend",
            "explain",
            "insight",
            "compare",
            "why",
        }
        literal_hits = sum(token in prompt_lower for token in literal_signals)
        semantic_hits = sum(token in prompt_lower for token in semantic_signals)
        if literal_hits > semantic_hits + 1:
            return "literal"
        if semantic_hits > literal_hits + 1:
            return "semantic"
        if literal_hits and semantic_hits:
            return "hybrid"
        # fall back to heuristic using question type
        if any(prompt_lower.startswith(prefix) for prefix in {"how", "why", "what"}):
            return "semantic"
        return "literal"

    def _alpha_for_mode(self, mode: str) -> float:
        if mode == "semantic":
            return 0.8
        if mode == "literal":
            return 0.2
        if mode == "hybrid":
            return 0.5
        return 0.5

    def _semantic_retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict]:
        items = self.store.retrieve(query_embedding, top_k=top_k)
        results: List[Dict] = []
        for item in items:
            summary = self.summarizer.summarize(item.text, max_len=220)
            source = item.meta.get("doc_path") if item.meta else None
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
            results.append(
                {
                    "text": summary,
                    "context": [summary],
                    "semantic_score": similarity,
                    "literal_score": 0.0,
                    "source": source,
                }
            )
        return results

    def _blend_results(
        self,
        literal_results: List[LiteralResult],
        semantic_results: List[Dict],
        alpha: float,
        *,
        top_k: int,
    ) -> List[Dict]:
        blended: List[Dict] = []
        for res in literal_results:
            blended.append(
                {
                    "context": res.context,
                    "semantic_score": res.semantic_score,
                    "literal_score": res.literal_score,
                    "source": res.source,
                }
            )
        blended.extend(semantic_results)
        for entry in blended:
            semantic_score = entry.get("semantic_score", 0.0)
            literal_score = entry.get("literal_score", 0.0)
            entry["final_score"] = alpha * semantic_score + (1 - alpha) * literal_score
        blended.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        seen: set[str] = set()
        deduped: List[Dict] = []
        for entry in blended:
            key = "|".join(entry.get("context", []))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
            if len(deduped) >= top_k:
                break
        return deduped

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        try:
            self.close()
        except Exception:
            pass

