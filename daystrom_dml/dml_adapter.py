"""High level adapter orchestrating the Daystrom Memory Lattice."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .embeddings import Embedder, create_embedder
from .gpt_runner import GPTRunner
from .memory_store import MemoryStore
from .summarizer import DummySummarizer, LLMSummarizer, Summarizer
from .retrievers import LiteralRetriever, LiteralResult
from .rag_store import SimpleRAGStore
from . import utils

LOGGER = logging.getLogger(__name__)

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
        self.runner = runner or GPTRunner(self.config["model_name"])
        self.embedder = embedder or create_embedder(self.config.get("embedding_model"))
        if summarizer is not None:
            self.summarizer = summarizer
        elif self.runner.is_dummy:
            self.summarizer = DummySummarizer()
        else:
            self.summarizer = LLMSummarizer(self.runner)
        self.literal_retriever = LiteralRetriever(
            self.embedder,
            self.summarizer,
            context_window=int(self.config.get("literal_context", 1)),
        )
        self.rag_store = SimpleRAGStore(self.embedder)
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
        LOGGER.info("Daystrom Memory Lattice initialised with %d capacity", self.store.capacity)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
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

    def build_preamble(self, prompt: str, top_k: Optional[int] = None) -> str:
        if top_k is None:
            config_top_k = self.config.get("top_k", 6)
        else:
            config_top_k = top_k
        try:
            top_k_int = int(config_top_k)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid top_k value %r, falling back to 0", config_top_k)
            top_k_int = 0
        top_k = max(0, top_k_int)
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

    def run_generation(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        context = self.build_preamble(prompt, top_k=self.config.get("top_k", 6))
        augmented_prompt = f"{context}\n\n{prompt}"
        response = self.runner.generate(augmented_prompt, max_new_tokens=max_new_tokens)
        self.reinforce(prompt, response)
        return response

    def retrieval_report(self, prompt: str, *, top_k: Optional[int] = None) -> Dict:
        items = self._retrieve_items(prompt, top_k)
        entries, preamble, tokens_used = self._prepare_context(prompt, items)
        fidelities = [entry["fidelity"] for entry in entries]
        avg_fidelity = float(np.mean(fidelities) if fidelities else 0.0)
        return {
            "entries": entries,
            "preamble": preamble,
            "tokens": tokens_used,
            "avg_fidelity": avg_fidelity,
        }

    def compare_responses(
        self,
        prompt: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> Dict:
        base_response = self.runner.generate(prompt, max_new_tokens=max_new_tokens)
        base_usage = self.runner.last_usage

        rag_top_k = self.config.get("top_k", 6) if top_k is None else top_k
        rag_report = self.rag_store.report(prompt, top_k=rag_top_k)
        rag_context = self._format_rag_context(rag_report["context"])
        rag_prompt = self._compose_prompt(prompt, rag_context)
        rag_response = self.runner.generate(rag_prompt, max_new_tokens=max_new_tokens)
        rag_usage = self.runner.last_usage

        dml_report = self.retrieval_report(prompt, top_k=top_k)
        dml_context = self._format_dml_context(dml_report["entries"])
        dml_prompt = self._compose_prompt(prompt, dml_context)
        dml_response = self.runner.generate(dml_prompt, max_new_tokens=max_new_tokens)
        dml_usage = self.runner.last_usage
        self.reinforce(prompt, dml_response)

        combined_blocks = [block for block in (rag_context, dml_context) if block]
        combined_context = "\n\n".join(combined_blocks)
        combined_prompt = self._compose_prompt(prompt, combined_context)
        integrated_response = self.runner.generate(combined_prompt, max_new_tokens=max_new_tokens)
        integrated_usage = self.runner.last_usage

        return {
            "prompt": prompt,
            "base": {
                "response": base_response,
                "usage": base_usage,
            },
            "rag": {
                "response": rag_response,
                "usage": rag_usage,
                "context": rag_context,
                "context_tokens": rag_report["tokens"],
                "documents": rag_report["documents"],
            },
            "dml": {
                "response": dml_response,
                "usage": dml_usage,
                "context": dml_context,
                "context_tokens": utils.estimate_tokens(dml_context),
                "avg_fidelity": dml_report["avg_fidelity"],
                "entries": dml_report["entries"],
            },
            "integrated": {
                "response": integrated_response,
                "usage": integrated_usage,
                "context": combined_context,
                "context_tokens": utils.estimate_tokens(combined_context),
            },
        }

    def knowledge_report(self) -> Dict:
        """Expose summaries of the RAG corpus and DML memory lattice."""

        rag_catalog = self.rag_store.catalog()

        dml_items = []
        dml_total_tokens = 0
        for item in self.store.items():
            text = (item.text or "").strip()
            tokens = utils.estimate_tokens(text)
            dml_total_tokens += tokens
            dml_items.append(
                {
                    "id": item.id,
                    "level": item.level,
                    "fidelity": item.fidelity,
                    "tokens": tokens,
                    "summary": text,
                    "meta": item.meta or {},
                }
            )

        return {
            "rag": {
                "documents": rag_catalog["documents"],
                "total_tokens": rag_catalog["total_tokens"],
                "count": rag_catalog["count"],
            },
            "dml": {
                "entries": dml_items,
                "total_tokens": dml_total_tokens,
                "count": len(dml_items),
            },
        }

    def query_database(self, prompt: str, mode: str = "auto") -> Dict:
        """Retrieve context-aware snippets from the external corpus."""

        if mode not in {"semantic", "literal", "hybrid", "auto"}:
            raise ValueError(f"Unsupported mode: {mode}")
        selected_mode = mode if mode != "auto" else self._classify_mode(prompt)
        top_k = int(self.config.get("top_k", 6))
        start = time.perf_counter()
        query_embedding = self.embedder.embed(prompt)
        items = self.store.items()
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
        config_file = Path(path) if path else default_path
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data

    def _estimate_salience(self, text: str) -> float:
        tokens = utils.estimate_tokens(text)
        return float(max(0.1, min(1.0, tokens / 200.0)))

    def _retrieve_items(self, prompt: str, top_k: Optional[int]) -> List[MemoryStore.MemoryItem]:
        if top_k is None:
            config_top_k = self.config.get("top_k", 6)
        else:
            config_top_k = top_k
        try:
            top_k_int = int(config_top_k)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid top_k value %r, falling back to 0", config_top_k)
            top_k_int = 0
        top_k_int = max(0, top_k_int)
        prompt_embedding = self.embedder.embed(prompt)
        return self.store.retrieve(prompt_embedding, top_k=top_k_int)

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

