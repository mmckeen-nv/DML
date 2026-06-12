"""High level adapter orchestrating the Daystrom Memory Lattice."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from .config import load_config
from .checkpoint import CheckpointManager
from .embeddings import Embedder, create_embedder
from .gpt_runner import GPTRunner
from .memory_store import MemoryItem, MemoryStore
from .metrics import record_retrieval, update_memory_gauge
from .multi_rag import MultiRAGStore, RAGBackendDescriptor
from .persistence import load_state as load_persisted_memories
from .persistence import save_state as save_persisted_memories
from .personality_matrix import PersonalityMatrix, overlay_token_count
from .summarizer import DummySummarizer, LLMSummarizer, Summarizer
from .retrievers import LiteralRetriever
from .router import decide_mode
from .rag_store import PersistentRAGStore
from .stm.controller import STMController
from .stm.policy import LTMWritePolicy, MemoryWrite
from .stm.schema import STMState
from . import utils
from .agent_schema import AgenticMemorySchema, MemoryKind, MemoryPhase
from .promotion_pipeline import PromotionPipeline, MemoryEntry
from .policy_router import PolicyRouter, TaskType, RouterDecision

LOGGER = logging.getLogger(__name__)


class _PersistentRAGBackendAdapter:
    """Adapt PersistentRAGStore to the MultiRAGStore backend protocol."""

    identifier = "persistent-rag"
    label = "Persistent RAG"
    description = "Configured persistent RAG backend"

    def __init__(self, store: PersistentRAGStore) -> None:
        self._store = store

    def add_document(self, text: str, embedding: np.ndarray, tokens: int, meta: Optional[Dict[str, Any]] = None) -> None:
        self._store.add(text, embedding.tolist(), meta=meta)

    def clear(self) -> None:
        return

    def retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
        results = self._store.search(query_embedding.tolist(), top_k=top_k)
        for result in results:
            result.setdefault("meta", {})
            result.setdefault("tokens", utils.estimate_tokens(result.get("text", "")))
        return results


DEFAULT_DML_TOP_K = 8
MAX_RETRIEVAL_TOP_K = 10
SURVIVAL_LEDGER_KIND = "survival_ledger"
SURVIVAL_ANCHOR_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b")

# Limit the amount of data returned by the knowledge endpoint to keep the
# payload responsive even when the lattice contains thousands of memories.
KNOWLEDGE_MAX_ENTRIES = 200
KNOWLEDGE_ENTRY_PREVIEW_CHARS = 320

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
        overrides = dict(config_overrides or {})
        self.settings = load_config(config_path, overrides=overrides)
        self.config = self.settings.as_dict()
        self.config.setdefault("dml_top_k", DEFAULT_DML_TOP_K)
        self.enable_workflow_cache = bool(
            self.config.get("enable_workflow_cache", False)
        )
        self.enable_quality_on_retrieval = bool(
            self.config.get("enable_quality_on_retrieval", False)
        )
        self.metrics_enabled = bool(self.settings.metrics_enabled)
        self.survival_ledger_enabled = bool(
            self.config.get("survival_ledger_enabled", True)
        )
        self.survival_ledger_max_anchors = max(
            8, int(self.config.get("survival_ledger_max_anchors", 80) or 80)
        )
        self.survival_ledger_summary_chars = max(
            160, int(self.config.get("survival_ledger_summary_chars", 1200) or 1200)
        )
        self.llm_backend = str(self.config.get("llm_backend", "auto"))
        strict_embedding_required = bool(self.config.get("strict_embedding_required", False))
        strict_llm_required = bool(self.config.get("strict_llm_required", False))
        self.runner = runner or GPTRunner(
            self.config["model_name"],
            backend=self.llm_backend,
            device=self.config.get("llm_device"),
            dtype=self.config.get("llm_dtype", "auto"),
            load_in_4bit=bool(self.config.get("load_in_4bit", False)),
            load_in_8bit=bool(self.config.get("load_in_8bit", False)),
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
            use_fast_tokenizer=bool(self.config.get("use_fast_tokenizer", True)),
            temperature=float(self.config.get("llm_temperature", 0.2)),
            top_p=float(self.config.get("llm_top_p", 1.0)),
        )
        if strict_llm_required and self.runner.is_dummy:
            raise RuntimeError(
                f"Failed to initialize LLM backend for model {self.config['model_name']!r}; local completion fallback is disabled"
            )
        self.embedder = embedder or create_embedder(
            self.config.get("embedding_model"),
            device=self.config.get("embedding_device"),
            allow_random_fallback=not strict_embedding_required,
        )
        self._query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_embedding_cache_size = max(
            0, int(self.config.get("query_embedding_cache_size", 64) or 0)
        )
        if summarizer is not None:
            self.summarizer = summarizer
        elif self.runner.is_dummy:
            self.summarizer = DummySummarizer()
        else:
            self.summarizer = LLMSummarizer(self.runner)
        self.storage_dir = self.settings.storage_dir.expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.personality_matrix = PersonalityMatrix(
            getattr(self.settings, "dpm", None),
            storage_dir=self.storage_dir,
        )
        persistence_settings = getattr(self.settings, "persistence", None)
        persistence_path = getattr(persistence_settings, "path", None) if persistence_settings else None
        if persistence_path:
            persistence_path = Path(persistence_path).expanduser()
        else:
            persistence_path = Path("dml_state.jsonl")
        if not persistence_path.is_absolute():
            legacy_path = (self.storage_dir / persistence_path).resolve()
            persistence_path = self._resolve_storage_path(persistence_path).resolve()
            if legacy_path != persistence_path and legacy_path.exists() and not persistence_path.exists():
                persistence_path.parent.mkdir(parents=True, exist_ok=True)
                persistence_path.write_bytes(legacy_path.read_bytes())
                LOGGER.info(
                    "Copied legacy DML persistence state from %s to %s",
                    legacy_path,
                    persistence_path,
                )
        self._persistence_path = persistence_path
        interval_value = getattr(persistence_settings, "interval_sec", 0) if persistence_settings else 0
        try:
            self._persistence_interval = max(0, int(interval_value))
        except (TypeError, ValueError):
            self._persistence_interval = 0
        self._persistence_enabled = bool(persistence_settings and getattr(persistence_settings, "enable", False))
        self._persistence_stop_event = Event()
        self._persistence_thread: Optional[threading.Thread] = None
        self.dml_state_path = self.storage_dir / "dml_store.json"
        self.rag_state_path = self.storage_dir / "rag_store.json"
        self.checkpoint_dir = self.storage_dir / "checkpoints"
        self._persist_lock = RLock()
        literal_cfg = getattr(self.settings, "literal", None)
        literal_tokens = 160
        literal_snippets = 8
        if literal_cfg is not None:
            literal_tokens = int(getattr(literal_cfg, "max_snippet_tokens", literal_tokens))
            literal_snippets = int(getattr(literal_cfg, "max_snippets", literal_snippets))
        self.literal_snippet_cap = max(1, literal_snippets)
        self.literal_token_cap = max(16, literal_tokens)
        char_window = max(64, int(self.literal_token_cap * 4))
        self.literal_retriever = LiteralRetriever(
            self.embedder,
            self.summarizer,
            context_window=int(self.config.get("literal_context", 1)),
            max_snippet_chars=char_window,
        )
        rag_settings = getattr(self.settings, "rag_store", None)
        self.persistent_rag_store: Optional[PersistentRAGStore] = None
        if rag_settings and getattr(rag_settings, "enable", False):
            index_path = Path(rag_settings.path).expanduser()
            meta_path = Path(rag_settings.meta_path).expanduser()
            if not index_path.is_absolute():
                index_path = self._resolve_storage_path(index_path)
            if not meta_path.is_absolute():
                meta_path = self._resolve_storage_path(meta_path)
            try:
                self.persistent_rag_store = PersistentRAGStore(
                    enable=True,
                    index_path=index_path,
                    meta_path=meta_path,
                    dim=int(rag_settings.dim),
                    backend=str(rag_settings.backend),
                )
            except Exception:
                LOGGER.exception("Failed to initialise persistent RAG store.")
                self.persistent_rag_store = None
            else:
                with contextlib.suppress(Exception):
                    self.persistent_rag_store.load()
        else:
            self.persistent_rag_store = None

        # Initialize Agentic Mode components
        self.agentic_mode_enabled = bool(self.config.get("dml.agentic_mode.enabled", False))
        self.agentic_router: Optional[PolicyRouter] = None

        if self.agentic_mode_enabled:
            router_enabled = bool(self.config.get("dml.router.enabled", False))
            router_profile = self.config.get("dml.router.profile")
            router_log = self.config.get("dml.router.log_level", "info")

            self.agentic_router = PolicyRouter(
                enabled=router_enabled,
                profile=router_profile,
                log_level=router_log,
            )

            LOGGER.info("Agentic Mode enabled with router: %s", router_enabled)

            # Initialize promotion pipeline
            self.agentic_promotion = PromotionPipeline(
                commitment_threshold=float(self.config.get("dml.commitment_threshold", 0.75)),
                allow_action_observation=True,
                strict_mode=True,
            )

            LOGGER.info("Agentic promotion pipeline initialized")

        backends: List[RAGBackendDescriptor] = []
        if self.persistent_rag_store is not None:
            backends.append(
                RAGBackendDescriptor(
                    identifier="persistent-rag",
                    label="Persistent RAG",
                    description="Configured persistent RAG backend",
                    factory=lambda store=self.persistent_rag_store: _PersistentRAGBackendAdapter(store),
                )
            )
        self.rag_store = MultiRAGStore(self.embedder, backends=backends)
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
            enable_quality_on_retrieval=self.enable_quality_on_retrieval,
            similarity_threshold=float(self.config.get("similarity_threshold", 0.0)),
        )
        self.enable_stm_controller = bool(self.config.get("enable_stm_controller", False))
        self.stm_controller: Optional[STMController] = None
        self._stm_states: Dict[str, STMState] = {}
        self._stm_lock = RLock()
        self._dml_queue: List[Dict[str, Any]] = []
        self._idle_threshold = self.config.get("idle_threshold_seconds", 300)  # 5 minutes default
        self._last_activity_time = time.time()
        self._background_processing_enabled = self.config.get("background_processing_enabled", True)

        if self.enable_stm_controller:
            policy = LTMWritePolicy(
                mode=str(self.config.get("ltm_write_policy", "balanced")),
                confidence_threshold=float(self.config.get("commitment_threshold", 0.75)),
            )
            self.stm_controller = STMController(
                policy=policy,
                stm_max_commitments=int(self.config.get("stm_max_commitments", 8)),
                stm_max_entities=int(self.config.get("stm_max_entities", 8)),
                top_k=int(self.config.get("ltm_top_k", self.config.get("dml_top_k", DEFAULT_DML_TOP_K))),
                extract_max_tokens=int(self.config.get("stm_extract_max_tokens", 256)),
            )
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if int(self.settings.checkpoint_interval_seconds) > 0:
            self.checkpoint_manager = CheckpointManager(
                self.checkpoint_dir,
                self._gather_checkpoint_state,
                interval_seconds=int(self.settings.checkpoint_interval_seconds),
                retention=int(self.settings.checkpoint_retention),
            )
        self._load_persisted_state()
        if self._persistence_enabled and self._persistence_interval > 0:
            self._start_persistence_loop()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))
        LOGGER.info("Daystrom Memory Lattice initialised with %d capacity", self.store.capacity)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._persist_all()
        self._stop_persistence_loop()
        if self.checkpoint_manager:
            self.checkpoint_manager.close()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))
        self.store.close()

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def ingest(
        self,
        text: str,
        meta: Optional[Dict] = None,
        *,
        persist: bool = True,
    ) -> None:
        """Full DML ingest - generates embeddings and adds to all systems"""
        if not text:
            return
        embedding = self.embedder.embed(text)
        salience = self._estimate_salience(text)
        item, merged = self.store.ingest(text, embedding, salience=salience, meta=meta)
        if merged and meta and str(meta.get("conflict_state") or "").strip():
            item.meta.update(
                {
                    key: value
                    for key, value in meta.items()
                    if key.startswith("conflict") or key in {"claim_key", "claim_value"}
                }
            )
        rag_text = item.text if merged else text
        rag_embedding = item.embedding if merged else embedding
        rag_meta: Dict[str, Any] = dict(meta or {})
        rag_meta.setdefault("memory_id", item.id)
        if merged:
            rag_meta["memory_merges"] = int(item.meta.get("merges", 0))
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                self.persistent_rag_store.add(rag_text, rag_embedding, meta=rag_meta)
        self.rag_store.add_document(rag_text, meta=rag_meta)
        if persist:
            self._persist_all()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def ingest_fast(self, text: str, meta: Optional[Dict] = None) -> None:
        """Fast ingest - adds to RAG only, queues for background DML processing"""
        if not text:
            return
        rag_meta: Dict[str, Any] = dict(meta or {})
        rag_meta.setdefault("source", "fast_ingest")
        rag_meta.setdefault("queued_for_dml", True)
        self.rag_store.add_document(text, meta=rag_meta)
        # Queue for background DML processing
        self._enqueue_for_dml(text, meta)
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def ingest_agentic(
        self,
        text: str,
        kind: Union[MemoryKind, str],
        meta: Optional[Dict] = None,
    ) -> None:
        """
        Agentic ingest - validates schema and routes through promotion pipeline.

        Args:
            text: Memory content.
            kind: Type of memory (action, observation, plan, error).
            meta: Optional metadata with phase, tool, provenance, etc.
        """
        if not text:
            return

        # Add agentic metadata. Public callers historically pass both
        # MemoryKind enum members and raw string values; normalize once so the
        # validation path, MemoryEntry shape, and promotion routing agree.
        agentic_meta = dict(meta or {})
        kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
        try:
            kind_enum: Optional[MemoryKind] = kind if isinstance(kind, MemoryKind) else MemoryKind(kind_value)
        except ValueError:
            kind_enum = None
        agentic_meta["kind"] = kind_value

        # Validate schema in agentic mode
        if self.agentic_mode_enabled:
            schema = AgenticMemorySchema(strict=True)
            is_valid, errors = schema.validate(agentic_meta)

            if not is_valid:
                LOGGER.warning(f"Agentic memory rejected: {errors}")
                return

        # Add to memory store (existing behavior)
        embedding = self.embedder.embed(text)
        salience = self._estimate_salience(text)
        item, merged = self.store.ingest(text, embedding, salience=salience, meta=agentic_meta)
        rag_text = item.text if merged else text
        rag_embedding = item.embedding if merged else embedding
        rag_meta: Dict[str, Any] = dict(agentic_meta)
        rag_meta.setdefault("memory_id", item.id)

        # Add to RAG store
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                self.persistent_rag_store.add(rag_text, rag_embedding, meta=rag_meta)
        self.rag_store.add_document(rag_text, meta=rag_meta)

        # Add to promotion pipeline in agentic mode
        if self.agentic_mode_enabled and self.agentic_router:
            memory_entry = MemoryEntry(
                text=text,
                embedding=embedding,
                timestamp=time.time(),
                meta=agentic_meta,
                kind=kind_value,
                phase=agentic_meta.get("phase"),
                tool=agentic_meta.get("tool"),
                outcome=agentic_meta.get("outcome"),
            )

            # Route through promotion pipeline
            if kind_enum in [MemoryKind.ACTION, MemoryKind.OBSERVATION, MemoryKind.ERROR]:
                self.agentic_promotion.ingest_to_scratch(memory_entry)
                LOGGER.debug(f"Added {kind_value} to scratch store")
            else:
                # Plans and artifacts go directly to verified
                self.agentic_promotion.verified.add(memory_entry)
                LOGGER.debug(f"Added {kind_value} to verified store")

        self._maybe_update_survival_ledger(text, agentic_meta)
        self._persist_all()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def get_context(self, query: str, max_tokens: int = 1000) -> str:
        """Return formatted retrieval context, truncated to an approximate token cap."""
        report = self.retrieve_context(query)
        raw_context = str(report.get("raw_context", ""))
        estimated_tokens = len(raw_context) // 4
        if estimated_tokens > max_tokens:
            return raw_context[: max_tokens * 4]
        return raw_context

    def memory_count(self) -> int:
        """Return the number of currently stored DML memories.

        Kept as a small compatibility API for legacy scripts and wrappers that
        predate direct access to ``adapter.store.items()``.
        """
        return len(self.store.items())

    def _enqueue_for_dml(self, text: str, meta: Optional[Dict] = None) -> None:
        """Queue documents for background DML processing"""
        self._dml_queue.append({"text": text, "meta": meta})
        self._schedule_idle_processing()

    def _schedule_idle_processing(self) -> None:
        """Schedule background DML queue processing"""
        if self._background_processing_enabled:
            # Simple trigger - check next time _persist_all is called or add timer
            if not hasattr(self, '_background_timer') or not self._background_timer.is_alive():
                self._background_timer = threading.Timer(
                    1.0,  # Check immediately (in real implementation, use idle detection)
                    self._process_dml_queue
                )
                self._background_timer.start()

    def _process_dml_queue(self) -> None:
        """Process queued documents for DML when idle"""
        if not self._dml_queue or not self._background_processing_enabled:
            return

        # Process a batch
        batch_size = self.config.get("dml_batch_size", 10)
        documents = self._dml_queue[:batch_size]
        self._dml_queue = self._dml_queue[batch_size:]

        LOGGER.info(f"Processing {len(documents)} documents from DML queue")

        for doc in documents:
            try:
                self.ingest(doc["text"], meta=doc["meta"])
            except Exception as e:
                LOGGER.error(f"Failed to process DML queue document: {e}")

        # Check if more to process or if we should schedule next run
        if self._dml_queue:
            # Reschedule
            self._schedule_idle_processing()
        else:
            # Clear timer
            if hasattr(self, '_background_timer'):
                self._background_timer.cancel()

    def build_preamble(self, prompt: str, top_k: Optional[int] = None) -> str:
        items = self._retrieve_items(prompt, top_k)
        _, preamble, _ = self._prepare_context(prompt, items)
        if getattr(self.settings.dpm, "include_in_preamble", True):
            overlay = self.personality_overlay(prompt=prompt)
            block = self.personality_matrix.render_context_block(overlay) if overlay else ""
            if block:
                preamble = f"{block}\n\n{preamble}" if preamble else block
        return preamble

    def personality_overlay(
        self,
        *,
        prompt: str = "",
        thread_id: Optional[str] = None,
        project_id: Optional[str] = None,
        relationship_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the active Daystrom Personality Matrix overlay, if enabled."""

        return self.personality_matrix.build_overlay(
            prompt=prompt,
            thread_id=thread_id,
            project_id=project_id,
            relationship_id=relationship_id,
        )

    def record_personality_preference(
        self,
        text: str,
        *,
        scope: str = "relationship",
        source_id: str = "turn:current",
        explicit: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record an explicit preference into the DPM graph in active-write mode."""

        return self.personality_matrix.record_preference(
            text,
            scope=scope,
            source_id=source_id,
            explicit=explicit,
            meta=meta,
        )

    def personality_graph(self) -> Optional[Dict[str, Any]]:
        """Return the current DPM preference graph, if one exists."""

        return self.personality_matrix.graph()

    def suppress_personality_preference(
        self, node_id: str, *, reason: str = "suppressed_by_user"
    ) -> Optional[Dict[str, Any]]:
        """Suppress a DPM preference node in active-write mode."""

        return self.personality_matrix.suppress_preference(node_id, reason=reason)

    def delete_personality_preference(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Delete a DPM preference node in active-write mode."""

        return self.personality_matrix.delete_preference(node_id)

    def reinforce(self, prompt: str, response: str, meta: Optional[Dict] = None) -> None:
        prompt_text = (prompt or "").strip()
        response_text = self._clean_context_fragment(response)
        if not response_text:
            return
        response_summary = self.summarizer.summarize(response_text, max_len=220).strip()
        if not response_summary:
            response_summary = response_text[:220].strip()
        lines: List[str] = []
        if prompt_text:
            lines.append(f"Prompt: {prompt_text}")
        else:
            lines.append("Prompt: (empty)")
        lines.append(f"Answer summary: {response_summary}")
        memory_text = "\n".join(lines)
        embedding = self.embedder.embed(memory_text)
        salience = self._estimate_salience(response_summary) + 0.1
        memory_meta = dict(meta or {})
        if prompt_text:
            memory_meta.setdefault("prompt", prompt_text)
        if response_text:
            memory_meta.setdefault("response_excerpt", response_text[:500])
        self.store.ingest(memory_text, embedding, salience=salience, meta=memory_meta)
        explicit_dpm_preference = bool(memory_meta.get("dpm_preference"))
        self.record_personality_preference(
            prompt_text,
            scope=str(memory_meta.get("dpm_scope") or "relationship"),
            source_id=str(memory_meta.get("source") or "turn:current"),
            explicit=explicit_dpm_preference,
            meta=memory_meta,
        )
        self._persist_dml_state()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def run_generation(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        session_id: Optional[str] = None,
    ) -> str:
        if self.enable_stm_controller and self.stm_controller:
            result = self._run_generation_with_controller(
                prompt,
                max_new_tokens=max_new_tokens,
                session_id=session_id,
            )
            return result["response"]
        context = self.build_preamble(prompt)
        augmented_prompt = f"{context}\n\n{prompt}"
        response = self.runner.generate(augmented_prompt, max_new_tokens=max_new_tokens)
        self.reinforce(prompt, response)
        return response

    def generate_with_controller(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not (self.enable_stm_controller and self.stm_controller):
            return {
                "response": self.run_generation(prompt, max_new_tokens=max_new_tokens),
                "context": self.build_preamble(prompt),
                "mode": "semantic",
            }
        return self._run_generation_with_controller(
            prompt,
            max_new_tokens=max_new_tokens,
            session_id=session_id,
        )

    def get_stm_state(self, session_id: Optional[str] = None) -> STMState:
        if not self.enable_stm_controller:
            raise RuntimeError("STM controller is disabled.")
        session_key = (session_id or "default").strip() or "default"
        return self._get_stm_state(session_key)

    def _run_generation_with_controller(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        session_key = (session_id or "default").strip() or "default"
        controller = self.stm_controller
        if controller is None:
            raise RuntimeError("STM controller not configured.")
        stm = self._get_stm_state(session_key)
        retrieval_plan = controller.decide_retrieval(stm, prompt)
        ltm_items = []
        if retrieval_plan.use_ltm:
            ltm_items = self._retrieve_ltm_items(prompt, top_k=retrieval_plan.top_k)
        stm_summary = controller.build_stm_summary(stm) if retrieval_plan.use_stm else ""
        ltm_block = self._format_ltm_entries(ltm_items)
        structured_prompt = self._compose_structured_prompt(
            prompt=prompt,
            stm_summary=stm_summary,
            ltm_block=ltm_block,
        )
        response = self.runner.generate(
            structured_prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(self.config.get("llm_temperature", 0.2)),
            top_p=float(self.config.get("llm_top_p", 1.0)),
        )
        extraction = controller.extract_structured_updates(
            user_msg=prompt,
            model_msg=response,
            generator=self.runner.generate,
        )
        contradictions = controller.detect_contradictions(stm, extraction.commitments)
        if contradictions:
            reconcile = controller.slow_path_reconcile(
                contradictions=contradictions, user_msg=prompt
            )
            extraction.commitments = []
            response = reconcile.response
        new_commitments = controller.update_stm_from_turn(
            stm,
            user_msg=prompt,
            model_msg=response,
            extraction=extraction,
        )
        if not contradictions:
            writes = controller.decide_ltm_writes(new_commitments)
            self._apply_ltm_writes(writes)
        context = self._compose_context_summary(stm_summary, ltm_block)
        return {
            "response": response,
            "context": context,
            "mode": retrieval_plan.mode,
        }

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
        allow_reinforce: bool = True,
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

        dml_report = self.retrieval_report(prompt, top_k=top_k)
        dml_context = self._format_dml_context(dml_report["entries"])
        dml_prompt = self._compose_prompt(prompt, dml_context)
        dml_response, dml_usage, dml_latency = self._generate_with_metrics(
            dml_prompt, max_new_tokens=max_new_tokens
        )
        if allow_reinforce and not self.runner.is_dummy:
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
        evaluations: List[Dict[str, Any]] = []

        reference_available = bool(dml_reference.strip())
        if reference_available:
            dml_grade = {
                "score": 1.0,
                "grade": self._score_to_grade(1.0),
                "explanation": "Reference response for grading.",
            }
            evaluations.append(
                {
                    "backend_id": "dml",
                    "score": 1.0,
                    "grade": dml_grade["grade"],
                }
            )
        else:
            dml_grade = {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "DML response unavailable for comparison.",
            }
            evaluations.append(
                {
                    "backend_id": "dml",
                    "score": 0.0,
                    "grade": dml_grade["grade"],
                }
            )

        answer_key = self._answer_key_for_prompt(prompt)
        base_accuracy = self._evaluate_answer_accuracy(base_response, answer_key)
        dml_accuracy = self._evaluate_answer_accuracy(dml_response, answer_key)

        rag_top_k = self.config.get("top_k", 6) if top_k is None else top_k
        rag_reports = self.rag_store.report_all(prompt, top_k=rag_top_k)

        # Ensure FAISS results are surfaced before Chroma when both are available.
        preferred_order = {"faiss": 0, "chroma": 1}
        indexed_reports = list(enumerate(rag_reports))
        indexed_reports.sort(
            key=lambda item: (
                preferred_order.get(item[1].get("id"), len(preferred_order) + item[0]),
                item[0],
            )
        )

        rag_results: List[Dict[str, Any]] = []
        for original_index, report in indexed_reports:
            if not report.get("available", True):
                entry = {
                    **report,
                    "response": "",
                    "usage": None,
                    "context_tokens": report.get("tokens", 0),
                    "sequence": None,
                    "generation_latency_ms": 0,
                    "retrieval_latency_ms": report.get("latency_ms", 0),
                    "accuracy": self._evaluate_answer_accuracy("", answer_key),
                }
                if reference_available:
                    entry["grade"] = {
                        "score": 0.0,
                        "grade": "N/A",
                        "explanation": report.get("error") or "Backend unavailable",
                    }
                else:
                    entry["grade"] = {
                        "score": 0.0,
                        "grade": "N/A",
                        "explanation": "DML response unavailable for comparison.",
                    }
                rag_results.append(entry)
                continue

            rag_context = self._format_rag_context(report.get("context") or "")
            rag_prompt = self._compose_prompt(prompt, rag_context)
            rag_response, rag_usage, rag_latency = self._generate_with_metrics(
                rag_prompt, max_new_tokens=max_new_tokens
            )
            step_counter += 1
            entry = {
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
                "embedding_latency_ms": report.get("embedding_latency_ms"),
                "index_latency_ms": report.get("index_latency_ms"),
                "generation_latency_ms": rag_latency,
                "available": True,
                "error": report.get("error"),
                "accuracy": self._evaluate_answer_accuracy(rag_response, answer_key),
            }

            if reference_available:
                grade = self._grade_response(rag_response, dml_reference)
                entry["grade"] = grade
                evaluations.append(
                    {
                        "backend_id": entry.get("id"),
                        "score": grade.get("score", 0.0),
                        "grade": grade.get("grade"),
                    }
                )
            else:
                entry["grade"] = {
                    "score": 0.0,
                    "grade": "N/A",
                    "explanation": "DML response unavailable for comparison.",
                }

            rag_results.append(entry)
            pipeline_trace.append(
                {
                    "step": step_counter,
                    "stage": "rag",
                    "id": entry["id"],
                    "label": entry.get("label"),
                }
            )

        return {
            "prompt": prompt,
            "base": {
                "response": base_response,
                "usage": base_usage,
                "sequence": base_sequence,
                "generation_latency_ms": base_latency,
                "accuracy": base_accuracy,
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
                "grade": dml_grade,
                "accuracy": dml_accuracy,
            },
            "answer_key": self._public_answer_key(answer_key),
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

    def _answer_key_for_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        normalized = str(prompt or "").lower()
        keys: List[Dict[str, Any]] = []

        def add_fuel() -> None:
            keys.extend(
                [
                    {"label": "41,200 tonnes deuterium slush", "patterns": [r"41,?200", r"deuterium"]},
                    {"label": "7,900 tonnes helium-3", "patterns": [r"7,?900", r"helium[- ]?3|he[- ]?3"]},
                    {"label": "320 kilograms antimatter catalyst", "patterns": [r"320", r"antimatter"]},
                    {"label": "8,400 tonnes argon", "patterns": [r"8,?400", r"argon"]},
                    {"label": "19,000 tonnes shield ice", "patterns": [r"19,?000", r"shield ice|ice"]},
                ]
            )

        def add_inventory_medical() -> None:
            keys.extend(
                [
                    {"label": "Quartermaster Jun Park controls inventory", "patterns": [r"jun park", r"inventory|quartermaster"]},
                    {"label": "Dr. Mateo Velasquez owns medical stores/care", "patterns": [r"velasquez|velásquez|mateo", r"medical|hibernation|surgical"]},
                    {"label": "1,200 trauma kits", "patterns": [r"1,?200", r"trauma kit"]},
                    {"label": "44 surgical nanofiber packs", "patterns": [r"\b44\b", r"surgical nanofiber"]},
                    {"label": "18 organ scaffold cartridges", "patterns": [r"\b18\b", r"organ scaffold"]},
                    {"label": "900 hibernation stabilizer vials", "patterns": [r"\b900\b", r"hibernation stabilizer"]},
                    {"label": "6,400 antiviral courses", "patterns": [r"6,?400", r"antiviral"]},
                ]
            )

        def add_landing_site() -> None:
            keys.extend(
                [
                    {"label": "Morrow Basin", "patterns": [r"morrow basin"]},
                    {"label": "shelter ceramics", "patterns": [r"shelter ceramic"]},
                    {"label": "landing beacon anchors", "patterns": [r"beacon anchor|landing beacon"]},
                ]
            )

        def add_failure_modes() -> None:
            keys.extend(
                [
                    {"label": "Fusion Injector Flutter", "patterns": [r"fusion injector flutter"]},
                    {"label": "Garden Fungal Bloom", "patterns": [r"garden fungal bloom"]},
                    {"label": "Optical Bus Desync", "patterns": [r"optical bus desync"]},
                    {"label": "Crawler Adhesion Loss", "patterns": [r"crawler adhesion loss"]},
                    {"label": "Hibernation Pod Cascade", "patterns": [r"hibernation pod cascade"]},
                ]
            )

        if "fuel" in normalized or "reserves" in normalized:
            add_fuel()
        if "inventory" in normalized or "medical" in normalized or "stores" in normalized:
            add_inventory_medical()
        if "shelter ceramics" in normalized or "beacon anchors" in normalized or "landing site" in normalized:
            add_landing_site()
        if "failure mode" in normalized or "failure modes" in normalized:
            add_failure_modes()

        if not keys:
            return None
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for fact in keys:
            label = str(fact["label"])
            if label in seen:
                continue
            seen.add(label)
            deduped.append(fact)
        return {"name": "Asteria deterministic answer key", "facts": deduped}

    def _public_answer_key(self, answer_key: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not answer_key:
            return None
        return {
            "name": answer_key.get("name"),
            "facts": [fact.get("label") for fact in answer_key.get("facts", [])],
        }

    def _evaluate_answer_accuracy(
        self, response: str, answer_key: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not answer_key:
            return {
                "scored": False,
                "score": None,
                "grade": "N/A",
                "matched": [],
                "missing": [],
                "explanation": "No deterministic answer key matched this prompt.",
            }
        text = str(response or "").lower()
        matched: List[str] = []
        missing: List[str] = []
        for fact in answer_key.get("facts", []):
            patterns = fact.get("patterns") or []
            found = all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
            label = str(fact.get("label") or "")
            if found:
                matched.append(label)
            else:
                missing.append(label)
        total = len(matched) + len(missing)
        score = (len(matched) / total) if total else 0.0
        return {
            "scored": True,
            "score": score,
            "grade": self._score_to_grade(score),
            "matched": matched,
            "missing": missing,
            "required": total,
            "explanation": f"Matched {len(matched)} of {total} required Asteria facts.",
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
        state_loaded = False
        if self._persistence_enabled:
            try:
                items = load_persisted_memories(self._persistence_path)
            except FileNotFoundError:
                pass
            except Exception:
                LOGGER.exception(
                    "Failed to load durable DML state from %s", self._persistence_path
                )
            else:
                payload = {"items": [item.to_dict() for item in items]}
                report = self._ensure_embedding_compatibility(payload)
                if report.get("status") in {"migrated", "partial"}:
                    LOGGER.warning(
                        "Loaded durable DML state from %s with embedding compatibility migration status=%s report=%s",
                        self._persistence_path,
                        report.get("status"),
                        report.get("report_path"),
                    )
                self.store.import_state(payload)
                state_loaded = True
        if not state_loaded:
            with contextlib.suppress(Exception):
                if self.dml_state_path.exists():
                    data = json.loads(self.dml_state_path.read_text(encoding="utf-8"))
                    report = self._ensure_embedding_compatibility(data)
                    if report.get("status") in {"migrated", "partial"}:
                        LOGGER.warning(
                            "Loaded JSON DML state from %s with embedding compatibility migration status=%s report=%s",
                            self.dml_state_path,
                            report.get("status"),
                            report.get("report_path"),
                        )
                    self.store.import_state(data)
        if not bool(self.config.get("skip_rag_state_import", False)):
            with contextlib.suppress(Exception):
                if self.rag_state_path.exists():
                    data = json.loads(self.rag_state_path.read_text(encoding="utf-8"))
                    self.rag_store.import_state(data)

    def _persist_all(self) -> None:
        self._persist_dml_state()
        self._persist_rag_state()

    def _start_persistence_loop(self) -> None:
        if self._persistence_thread and self._persistence_thread.is_alive():
            return
        if self._persistence_interval <= 0:
            return
        self._persistence_stop_event.clear()
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop,
            name="dml-persistence",
            daemon=True,
        )
        self._persistence_thread.start()

    def _stop_persistence_loop(self) -> None:
        self._persistence_stop_event.set()
        thread = self._persistence_thread
        if thread and thread.is_alive():
            thread.join(timeout=max(2.0, float(self._persistence_interval)))
        self._persistence_thread = None

    def _persistence_loop(self) -> None:
        while not self._persistence_stop_event.wait(self._persistence_interval):
            try:
                self._persist_dml_state()
            except Exception:
                LOGGER.exception("Failed to persist DML state during background save.")

    def _gather_checkpoint_state(self) -> Dict[str, Any]:
        """Collect a combined state payload for checkpointing."""

        return {
            "timestamp": time.time(),
            "dml": self.store.export_state(),
            "rag": self.rag_store.export_state(),
            "stats": self.stats(),
        }

    def _persist_dml_state(self) -> None:
        if self._persistence_enabled:
            with self._persist_lock:
                items = self.store.items()
                try:
                    save_persisted_memories(items, self._persistence_path)
                except Exception:
                    LOGGER.exception(
                        "Failed to persist DML state to %s", self._persistence_path
                    )
            return
        with self._persist_lock:
            data = self.store.export_state()
            tmp = self.dml_state_path.with_suffix(".tmp")
            try:
                self.dml_state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.dml_state_path)
            except Exception:
                LOGGER.exception("Failed to persist DML state to %s", self.dml_state_path)

    def _persist_rag_state(self) -> None:
        with self._persist_lock:
            if self.persistent_rag_store is not None:
                try:
                    self.persistent_rag_store.persist()
                except Exception:
                    LOGGER.exception("Failed to persist persistent RAG index.")
            data = self.rag_store.export_state()
            tmp = self.rag_state_path.with_suffix(".tmp")
            try:
                self.rag_state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.rag_state_path)
            except Exception:
                LOGGER.exception("Failed to persist RAG state to %s", self.rag_state_path)

    def _embedding_compatibility_report_path(self) -> Path:
        return self.storage_dir / "embedding_compatibility_report.json"

    def _read_embedding_compatibility_report(self) -> Optional[Dict[str, Any]]:
        report_path = self._embedding_compatibility_report_path()
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.debug("Failed to read embedding compatibility report from %s", report_path, exc_info=True)
            return None

    def _write_embedding_compatibility_report(self, report: Dict[str, Any]) -> None:
        report_path = self.storage_dir / "embedding_compatibility_report.json"
        out_dir = Path(os.environ.get("DAYSTROM_DML_OUT_DIR", str(self.storage_dir.parent / "out")))
        mirror_json = out_dir / "dml-migration-progress.json"
        mirror_md = out_dir / "dml-migration-progress.md"
        tmp_path = report_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            tmp_path.replace(report_path)
        except Exception:
            LOGGER.debug("Failed to write embedding compatibility report to %s", report_path, exc_info=True)
        try:
            mirror_json.parent.mkdir(parents=True, exist_ok=True)
            mirror_tmp = mirror_json.with_suffix(".tmp")
            mirror_tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mirror_tmp.replace(mirror_json)
        except Exception:
            LOGGER.debug("Failed to mirror embedding compatibility report to %s", mirror_json, exc_info=True)
        try:
            summary = "\n".join([
                "# DML Migration Progress",
                "",
                f"- status: {report.get('status')}",
                f"- phase: {report.get('phase')}",
                f"- detail: {report.get('phase_detail')}",
                f"- total_items: {report.get('total_items')}",
                f"- checked: {report.get('checked')}",
                f"- remaining_items: {report.get('remaining_items')}",
                f"- mismatched: {report.get('mismatched')}",
                f"- reembedded: {report.get('reembedded')}",
                f"- failed: {report.get('failed')}",
                f"- target_dim: {report.get('target_dim')}",
                f"- progress_pct: {report.get('progress_pct')}",
                f"- current_item_index: {report.get('current_item_index')}",
                f"- current_item_preview: {report.get('current_item_preview')}",
                f"- last_completed_item_index: {report.get('last_completed_item_index')}",
                f"- last_completed_item_preview: {report.get('last_completed_item_preview')}",
                f"- started_at: {report.get('started_at')}",
                f"- updated_at: {report.get('updated_at')}",
                f"- elapsed_ms: {report.get('elapsed_ms')}",
                f"- storage_report_path: {report_path}",
            ]) + "\n"
            mirror_md.parent.mkdir(parents=True, exist_ok=True)
            mirror_md_tmp = mirror_md.with_suffix(".tmp")
            mirror_md_tmp.write_text(summary, encoding="utf-8")
            mirror_md_tmp.replace(mirror_md)
        except Exception:
            LOGGER.debug("Failed to mirror embedding compatibility markdown report to %s", mirror_md, exc_info=True)

    def _ensure_embedding_compatibility(self, payload: Dict, *, max_items: Optional[int] = None) -> Dict[str, Any]:
        cached_report = self._read_embedding_compatibility_report()
        items = payload.get("items") if isinstance(payload, dict) else None
        total_items = len(items) if items else 0
        if (
            max_items is None
            and cached_report
            and cached_report.get("status") == "ok"
            and int(cached_report.get("total_items") or 0) == total_items
            and int(cached_report.get("remaining_items") or 0) == 0
            and int(cached_report.get("mismatched") or 0) == 0
            and int(cached_report.get("failed") or 0) == 0
        ):
            LOGGER.info(
                "Skipping embedding compatibility scan for %s persisted memories in %s; cached report already indicates compatibility.",
                total_items,
                self.storage_dir,
            )
            cached_report["phase_detail"] = "reused cached compatibility proof; full scan skipped"
            cached_report["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_embedding_compatibility_report(cached_report)
            return cached_report
        started_wall = datetime.now(timezone.utc)
        report: Dict[str, Any] = {
            "status": "skipped",
            "phase": "init",
            "phase_detail": "initializing",
            "total_items": 0,
            "checked": 0,
            "remaining_items": 0,
            "mismatched": 0,
            "reembedded": 0,
            "failed": 0,
            "target_dim": 0,
            "last_checked_index": 0,
            "last_completed_item_index": 0,
            "last_completed_item_preview": None,
            "progress_pct": 0.0,
            "current_item_index": 0,
            "current_item_preview": None,
            "started_at": started_wall.isoformat(),
            "updated_at": started_wall.isoformat(),
            "phase_started_at": started_wall.isoformat(),
            "elapsed_ms": 0.0,
            "report_path": str(self.storage_dir / "embedding_compatibility_report.json"),
            "max_items": max_items,
            "truncated": False,
        }
        started = time.perf_counter()

        def _flush_report(
            status: Optional[str] = None,
            phase: Optional[str] = None,
            phase_detail: Optional[str] = None,
        ) -> None:
            now_iso = datetime.now(timezone.utc).isoformat()
            if status is not None:
                report["status"] = status
            if phase is not None and phase != report.get("phase"):
                report["phase"] = phase
                report["phase_started_at"] = now_iso
            if phase_detail is not None:
                report["phase_detail"] = phase_detail
            total_items = int(report.get("total_items") or 0)
            checked = int(report.get("checked") or 0)
            report["last_checked_index"] = checked
            report["remaining_items"] = max(total_items - checked, 0)
            report["progress_pct"] = round((checked / total_items) * 100.0, 2) if total_items > 0 else 0.0
            report["updated_at"] = now_iso
            report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            self._write_embedding_compatibility_report(report)

        items = payload.get("items") if isinstance(payload, dict) else None
        if not items:
            _flush_report(status="no-items", phase="done", phase_detail="no persisted items found")
            return report
        report["total_items"] = len(items)
        try:
            probe = self.embedder.embed("Daystrom persistence probe")
            current_dim = int(np.asarray(probe, dtype=np.float32).size)
        except Exception:
            LOGGER.debug("Unable to determine embedder dimensions for persistence compatibility check.")
            _flush_report(status="probe-failed", phase="probe", phase_detail="failed to determine current embedding dimension")
            return report
        if current_dim == 0:
            _flush_report(status="zero-dimension-probe", phase="probe", phase_detail="embedding probe returned zero dimensions")
            return report

        report["target_dim"] = current_dim
        _flush_report(status="running", phase="scan", phase_detail="scanning persisted memories for incompatible embedding dimensions")
        LOGGER.warning(
            "Starting embedding compatibility migration for %s persisted memories in %s (target_dim=%s).",
            len(items),
            self.storage_dir,
            current_dim,
        )

        mismatched = 0
        reembedded = 0
        failed = 0
        for idx, entry in enumerate(items, start=1):
            if max_items is not None and report["checked"] >= max_items:
                report["truncated"] = True
                _flush_report(
                    status="partial",
                    phase="paused",
                    phase_detail=f"paused after bounded migration chunk of {max_items} items",
                )
                break
            if not isinstance(entry, dict):
                continue
            text = entry.get("text") or ""
            report["checked"] += 1
            report["current_item_index"] = idx
            report["current_item_preview"] = text[:80] if text else None
            stored_embedding = entry.get("embedding")
            try:
                stored_dim = int(np.asarray(stored_embedding, dtype=np.float32).size)
            except Exception:
                stored_dim = 0
            if stored_dim == current_dim:
                report["last_completed_item_index"] = idx
                report["last_completed_item_preview"] = text[:80] if text else None
                _flush_report(
                    status="running",
                    phase="scan",
                    phase_detail=f"scanned item {idx}/{len(items)}; embedding already compatible",
                )
                continue
            mismatched += 1
            report["mismatched"] = mismatched
            _flush_report(
                status="running",
                phase="reembed",
                phase_detail=f"re-embedding item {idx}/{len(items)} due to dimension mismatch ({stored_dim} -> {current_dim})",
            )
            try:
                new_embedding = self.embedder.embed(text)
                new_dim = int(np.asarray(new_embedding, dtype=np.float32).size)
                if new_dim == current_dim:
                    entry["embedding"] = utils.ensure_serializable(new_embedding)
                    reembedded += 1
                    report["reembedded"] = reembedded
                    report["last_completed_item_index"] = idx
                    report["last_completed_item_preview"] = text[:80] if text else None
                    _flush_report(
                        status="running",
                        phase="reembed",
                        phase_detail=f"re-embedded item {idx}/{len(items)} successfully",
                    )
                else:
                    entry["embedding"] = []
                    failed += 1
                    report["failed"] = failed
                    report["last_completed_item_index"] = idx
                    report["last_completed_item_preview"] = text[:80] if text else None
                    _flush_report(
                        status="running",
                        phase="reembed",
                        phase_detail=f"re-embed for item {idx}/{len(items)} returned wrong dimension ({new_dim} != {current_dim})",
                    )
            except Exception:
                entry["embedding"] = []
                failed += 1
                report["failed"] = failed
                report["last_completed_item_index"] = idx
                report["last_completed_item_preview"] = text[:80] if text else None
                _flush_report(
                    status="running",
                    phase="reembed",
                    phase_detail=f"re-embed for item {idx}/{len(items)} failed; stored empty embedding placeholder",
                )

        report["mismatched"] = mismatched
        report["reembedded"] = reembedded
        report["failed"] = failed
        report["current_item_index"] = report["checked"]
        report["current_item_preview"] = None
        _flush_report(
            status="ok" if mismatched == 0 else ("migrated" if failed == 0 else "partial"),
            phase="done",
            phase_detail=(
                f"completed compatibility migration: checked={report['checked']} mismatched={mismatched} reembedded={reembedded} failed={failed}"
            ),
        )

        if mismatched:
            LOGGER.warning(
                "Completed embedding compatibility migration for %s persisted memories in %s: mismatched=%s reembedded=%s failed=%s target_dim=%s elapsed_ms=%s report=%s",
                report["checked"],
                self.storage_dir,
                mismatched,
                reembedded,
                failed,
                current_dim,
                report["elapsed_ms"],
                report["report_path"],
            )
        else:
            LOGGER.info(
                "Embedding compatibility check found no mismatches for %s persisted memories in %s (target_dim=%s, elapsed_ms=%s).",
                report["checked"],
                self.storage_dir,
                current_dim,
                report["elapsed_ms"],
            )
        return report

    def query_database(self, prompt: str, mode: str = "auto") -> Dict:
        """Retrieve context-aware snippets from the external corpus."""

        if mode not in {"semantic", "literal", "hybrid", "auto"}:
            raise ValueError(f"Unsupported mode: {mode}")
        selected_mode = mode if mode != "auto" else decide_mode(prompt)
        start = time.perf_counter()
        query_embedding = self._embed_query(prompt)
        items = self.store.items()
        dml_limit = self._resolve_dml_top_k(None)
        top_k = min(len(items), dml_limit)
        literal_results: List[Dict[str, Any]] = []
        semantic_results: List[Dict[str, Any]] = []
        if selected_mode in {"literal", "hybrid"}:
            literal_results = self._literal_retrieve(
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
        latency = time.perf_counter() - start
        latency_ms = latency * 1000.0
        if self.metrics_enabled:
            record_retrieval(selected_mode, latency_ms)
        token_count = utils.estimate_tokens(context)
        return {
            "mode": selected_mode,
            "context": context,
            "source_docs": sources,
            "tokens": token_count,
            "latency_ms": int(latency_ms),
        }

    def create_checkpoint(self) -> Path:
        """Persist a combined snapshot of the lattice and RAG stores."""

        if self.checkpoint_manager:
            return self.checkpoint_manager.checkpoint()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manager = CheckpointManager(
            self.checkpoint_dir,
            self._gather_checkpoint_state,
            interval_seconds=0,
            retention=int(self.settings.checkpoint_retention),
            start=False,
        )
        try:
            return manager.checkpoint()
        finally:
            manager.close()

    def stats(self) -> Dict:
        items = self.store.items()
        return {
            "count": len(items),
            "levels": {level: sum(1 for it in items if it.level == level) for level in range(self.store.K + 1)},
            "avg_fidelity": float(np.mean([it.fidelity for it in items]) if items else 0.0),
        }

    def run_maintenance(self, sample_ratio: float = 0.1) -> None:
        """Run a maintenance pass to assess quality without slowing retrieval."""

        self.store.maintenance_pass(sample_ratio=sample_ratio)

    # ------------------------------------------------------------------
    # Multi-tenant helpers used by the DML memory service
    # ------------------------------------------------------------------
    def ingest_memory(
        self,
        text: str,
        *,
        tenant_id: str,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        kind: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        enriched_meta: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "session_id": session_id,
            "instance_id": instance_id,
            "kind": kind or "memory",
        }
        if meta:
            enriched_meta.update(meta)
        embedding = self.embedder.embed(text)
        salience = self._estimate_salience(text)
        item, merged = self.store.ingest(text, embedding, salience=salience, meta=enriched_meta)
        rag_text = item.text if merged else text
        rag_embedding = item.embedding if merged else embedding
        rag_meta: Dict[str, Any] = dict(enriched_meta)
        rag_meta.setdefault("memory_id", item.id)
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                self.persistent_rag_store.add(rag_text, rag_embedding, meta=rag_meta)
        self.rag_store.add_document(rag_text, meta=rag_meta)
        self._maybe_update_survival_ledger(text, enriched_meta)
        self._persist_all()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))
        return item

    def collect_instance_scratch(
        self,
        tenant_id: str,
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
    ) -> List[MemoryStore.MemoryItem]:
        return self.store.list_scratch(
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
        )

    def record_agent_workflow(
        self, task_description: str, steps: List[str], outcome: str
    ) -> Optional[str]:
        """Optionally store a successful agent workflow as a reusable template."""

        if not self.enable_workflow_cache:
            return None

        text = self._format_workflow_text(task_description, steps, outcome)
        embedding = self.embedder.embed(text)
        item = self.store.add(
            text=text,
            embedding=embedding,
            meta={
                "kind": "workflow",
                "task_description": task_description,
                "steps_count": len(steps),
                "outcome": outcome,
            },
        )
        return str(item.id)

    def suggest_workflows_for_task(
        self, task_description: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve reusable workflow templates related to a new task."""

        if not self.enable_workflow_cache:
            return []
        try:
            limit = max(1, int(top_k))
        except (TypeError, ValueError):
            limit = 3

        query_embedding = self.embedder.embed(task_description)
        candidates = self.store.retrieve_by_kind(
            query_embedding=query_embedding, kind="workflow", top_k=limit
        )

        results: List[Dict[str, Any]] = []
        for item in candidates:
            results.append(
                {
                    "id": item.id,
                    "summary": item.cached_summary(max_len=200),
                    "task_description": (item.meta or {}).get("task_description", ""),
                    "outcome": (item.meta or {}).get("outcome", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _format_rag_context(self, context: str) -> str:
        if not context:
            return ""
        return "=== RAG Retrieval ===\n" + context.strip()

    def _get_stm_state(self, session_key: str) -> STMState:
        with self._stm_lock:
            state = self._stm_states.get(session_key)
            if state is None:
                state = STMState()
                self._stm_states[session_key] = state
            return state

    def _retrieve_ltm_items(self, prompt: str, top_k: int) -> List[MemoryStore.MemoryItem]:
        items = self._retrieve_items(prompt, top_k)
        filtered: List[MemoryStore.MemoryItem] = []
        for item in items:
            meta = item.meta or {}
            kind = str(meta.get("kind") or "memory")
            if kind in {"scratch", "stm"}:
                continue
            filtered.append(item)
        return filtered

    def retrieve_context(
        self,
        prompt: str,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        phase: Optional[str | MemoryPhase] = None,
        dpm_thread_id: Optional[str] = None,
        dpm_project_id: Optional[str] = None,
        dpm_relationship_id: Optional[str] = None,
        include_quarantined: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve context with agentic-aware routing.

        Returns retrieval report with selected kinds, scores, and tokens.
        """
        start = time.perf_counter()
        decision: Optional[RouterDecision] = None
        phase_enum = self._coerce_memory_phase(phase)
        if self.agentic_mode_enabled and self.agentic_router:
            decision = self.agentic_router.decide(
                meta={"prompt": prompt[:100]},  # Sample for task detection
                phase=phase_enum,
                token_pressure=0.0,
            )

            if decision:
                LOGGER.debug(f"Router selected: {decision.selected_profile}")

        final_top_k = top_k
        if final_top_k is None and decision and decision.overrides.top_k:
            final_top_k = decision.overrides.top_k
        if final_top_k is None:
            final_top_k = self.config.get("dml_top_k", DEFAULT_DML_TOP_K)
        final_top_k = max(1, min(MAX_RETRIEVAL_TOP_K, self._resolve_dml_top_k(final_top_k)))

        final_kinds = kinds
        if final_kinds is None and decision and decision.overrides.allowed_kinds:
            final_kinds = decision.overrides.allowed_kinds
        if final_kinds is None and phase_enum in {MemoryPhase.EXECUTE, MemoryPhase.DEBUG}:
            final_kinds = ["action", "observation", "error"]

        query_embedding = self._embed_query(prompt)
        scoped = any(value is not None for value in (tenant_id, client_id, session_id, instance_id))
        if scoped:
            items = self.store.retrieve_filtered(
                query_embedding,
                tenant_id=tenant_id,
                client_id=client_id,
                session_id=session_id,
                instance_id=instance_id,
                kinds=final_kinds,
                top_k=final_top_k,
            )
            items = self._filter_retrievable_items(items, include_quarantined=include_quarantined)
            if not items:
                items = self._recent_context_items(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    session_id=session_id,
                    instance_id=instance_id,
                    kinds=final_kinds,
                    phase=phase_enum,
                    top_k=final_top_k,
                    include_quarantined=include_quarantined,
                )
            if not items:
                items = self.store.retrieve_filtered(
                    query_embedding,
                    tenant_id=None,
                    client_id=None,
                    session_id=None,
                    instance_id=None,
                    kinds=final_kinds,
                    top_k=final_top_k,
                )
                items = self._filter_retrievable_items(items, include_quarantined=include_quarantined)
            if not items:
                items = self._recent_context_items(
                    tenant_id=None,
                    client_id=None,
                    session_id=None,
                    instance_id=None,
                    kinds=final_kinds,
                    phase=phase_enum,
                    top_k=final_top_k,
                    require_unscoped=True,
                    include_quarantined=include_quarantined,
                )
        else:
            items = self._retrieve_items(
                prompt,
                final_top_k,
                phase=phase_enum.value if phase_enum else None,
                kinds=final_kinds,
                include_quarantined=include_quarantined,
            )
            if not items:
                items = self._recent_context_items(
                    tenant_id=None,
                    client_id=None,
                    session_id=None,
                    instance_id=None,
                    kinds=final_kinds,
                    phase=phase_enum,
                    top_k=final_top_k,
                    include_quarantined=include_quarantined,
                )

        ledger_item = self._survival_ledger_for_scope(
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
        )
        ledger_included = False
        if ledger_item is not None:
            items = [ledger_item] + [item for item in items if item.id != ledger_item.id]
            ledger_included = True

        entries, context, tokens_used = self._compact_context_items(items)
        personality_overlay = None
        if getattr(self.settings.dpm, "include_in_context", True):
            personality_overlay = self.personality_overlay(
                prompt=prompt,
                thread_id=dpm_thread_id or session_id,
                project_id=dpm_project_id,
                relationship_id=dpm_relationship_id or tenant_id,
            )
            personality_block = (
                self.personality_matrix.render_context_block(personality_overlay)
                if personality_overlay
                else ""
            )
            if personality_block:
                context = f"{personality_block}\n\n{context}" if context else personality_block
                tokens_used += overlay_token_count(personality_overlay)
        latency_ms = int((time.perf_counter() - start) * 1000.0)

        report = {
            "raw_context": context,
            "context_tokens": tokens_used,
            "top_k": final_top_k,
            "kinds": final_kinds,
            "phase": phase_enum.value if phase_enum else None,
            "include_quarantined": include_quarantined,
            "items": entries,
            "survival_ledger_included": ledger_included,
            "personality_overlay": personality_overlay,
            "latency_ms": latency_ms,
        }

        if self.metrics_enabled:
            record_retrieval("context", latency_ms=latency_ms)

        return report

    def _maybe_update_survival_ledger(self, text: str, meta: Optional[Dict[str, Any]]) -> None:
        if not self.survival_ledger_enabled or not text:
            return
        metadata = dict(meta or {})
        if metadata.get("kind") == SURVIVAL_LEDGER_KIND:
            return
        tenant_id = metadata.get("tenant_id")
        session_id = metadata.get("session_id")
        if not tenant_id or not session_id:
            return
        if not self._is_survival_ledger_event(text, metadata):
            return
        anchors = self._extract_survival_anchors(text, metadata)
        if not anchors:
            return
        self._upsert_survival_ledger(
            anchors,
            tenant_id=str(tenant_id),
            client_id=metadata.get("client_id"),
            session_id=str(session_id),
            instance_id=metadata.get("instance_id"),
            source_meta=metadata,
        )

    def _is_survival_ledger_event(self, text: str, meta: Dict[str, Any]) -> bool:
        if any(
            key in meta
            for key in (
                "compaction_cycle",
                "virtual_tokens",
                "survival_ledger",
                "long_horizon",
                "continuity_checkpoint",
            )
        ):
            return True
        marker_text = " ".join(
            str(value)
            for key, value in meta.items()
            if key in {"source", "tool", "kind", "phase", "task_id", "episode_id"}
        )
        haystack = f"{text} {marker_text}".lower()
        return any(
            marker in haystack
            for marker in (
                "compaction",
                "compact survival ledger",
                "survival ledger",
                "long-horizon",
                "long horizon",
                "continuity checkpoint",
                "handoff",
            )
        )

    def _extract_survival_anchors(self, text: str, meta: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        values.extend(SURVIVAL_ANCHOR_RE.findall(text or ""))
        for key in (
            "anchor",
            "anchors",
            "decision",
            "blocker",
            "next_step",
            "objective",
            "invariant",
        ):
            value = meta.get(key)
            if isinstance(value, str):
                values.extend(SURVIVAL_ANCHOR_RE.findall(value))
            elif isinstance(value, list):
                for entry in value:
                    values.extend(SURVIVAL_ANCHOR_RE.findall(str(entry)))
        seen = set()
        ordered: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _survival_scope_key(
        self,
        *,
        tenant_id: Optional[str],
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        return (
            str(tenant_id) if tenant_id is not None else None,
            str(client_id) if client_id is not None else None,
            str(session_id) if session_id is not None else None,
            str(instance_id) if instance_id is not None else None,
        )

    def _survival_ledger_for_scope(
        self,
        *,
        tenant_id: Optional[str],
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
    ) -> Optional[MemoryItem]:
        if not self.survival_ledger_enabled or not tenant_id or not session_id:
            return None
        target = self._survival_scope_key(
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
        )
        candidates: List[MemoryItem] = []
        for item in self.store.items():
            meta = item.meta or {}
            if meta.get("kind") != SURVIVAL_LEDGER_KIND:
                continue
            scope = self._survival_scope_key(
                tenant_id=meta.get("tenant_id"),
                client_id=meta.get("client_id"),
                session_id=meta.get("session_id"),
                instance_id=meta.get("instance_id"),
            )
            if scope == target:
                candidates.append(item)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.timestamp)

    def _upsert_survival_ledger(
        self,
        anchors: List[str],
        *,
        tenant_id: str,
        client_id: Optional[str],
        session_id: str,
        instance_id: Optional[str],
        source_meta: Dict[str, Any],
    ) -> None:
        existing = self._survival_ledger_for_scope(
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
        )
        previous: List[str] = []
        if existing is not None:
            previous = list((existing.meta or {}).get("anchors") or [])
        combined = self._merge_survival_anchors(previous, anchors)
        text = self._render_survival_ledger_text(
            tenant_id=tenant_id,
            session_id=session_id,
            anchors=combined,
            source_meta=source_meta,
        )
        summary = text[: self.survival_ledger_summary_chars].rstrip()
        if len(text) > len(summary):
            summary = summary.rstrip() + "..."
        now = time.time()
        meta: Dict[str, Any] = {
            "kind": SURVIVAL_LEDGER_KIND,
            "source": "dml_survival_ledger",
            "tenant_id": tenant_id,
            "client_id": client_id,
            "session_id": session_id,
            "instance_id": instance_id,
            "anchors": combined,
            "anchor_count": len(combined),
            "summary": summary,
            "survival_ledger": True,
            "updated_at": now,
            "last_step_id": source_meta.get("step_id"),
            "last_task_id": source_meta.get("task_id"),
            "last_virtual_tokens": source_meta.get("virtual_tokens"),
            "last_compaction_cycle": source_meta.get("compaction_cycle"),
            "no_merge": True,
        }
        embedding = self.embedder.embed(text)
        if existing is None:
            self.store.ingest(
                text,
                embedding,
                salience=2.0,
                fidelity=1.0,
                level=0,
                meta=meta,
            )
            return
        with self.store._lock:
            existing.text = text
            existing.embedding = np.asarray(embedding, dtype=np.float32)
            existing.timestamp = now
            existing.salience = max(float(existing.salience), 2.0)
            existing.fidelity = 1.0
            existing.meta = meta
            self.store._cache_summary(existing, text)
            self.store._invalidate_embedding_cache()

    def _merge_survival_anchors(self, previous: List[str], incoming: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in list(previous) + list(incoming):
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        if len(ordered) > self.survival_ledger_max_anchors:
            ordered = ordered[-self.survival_ledger_max_anchors :]
        return ordered

    def _render_survival_ledger_text(
        self,
        *,
        tenant_id: str,
        session_id: str,
        anchors: List[str],
        source_meta: Dict[str, Any],
    ) -> str:
        anchor_text = " ".join(anchors)
        parts = [
            f"Survival ledger anchors: {anchor_text}.",
            f"Scope tenant={tenant_id} session={session_id}.",
        ]
        if source_meta.get("virtual_tokens") is not None:
            parts.append(f"Last virtual token count: {source_meta.get('virtual_tokens')}.")
        if source_meta.get("compaction_cycle") is not None:
            parts.append(f"Last compaction cycle: {source_meta.get('compaction_cycle')}.")
        parts.append("Use this ledger as high-priority continuity state for long-horizon agent recovery.")
        return " ".join(parts)

    def _coerce_memory_phase(self, phase: Optional[str | MemoryPhase]) -> Optional[MemoryPhase]:
        if phase is None:
            return None
        if isinstance(phase, MemoryPhase):
            return phase
        try:
            return MemoryPhase(str(phase).strip().lower())
        except ValueError:
            LOGGER.debug("Ignoring unknown memory phase %r", phase)
            return None

    def _embed_query(self, prompt: str) -> np.ndarray:
        key = re.sub(r"\s+", " ", str(prompt or "").strip()).lower()
        if self._query_embedding_cache_size > 0 and key in self._query_embedding_cache:
            embedding = self._query_embedding_cache.pop(key)
            self._query_embedding_cache[key] = embedding
            return embedding
        embedding = self.embedder.embed(prompt)
        if self._query_embedding_cache_size > 0 and key:
            self._query_embedding_cache[key] = embedding
            while len(self._query_embedding_cache) > self._query_embedding_cache_size:
                self._query_embedding_cache.popitem(last=False)
        return embedding

    def _compact_context_items(
        self, items: List[MemoryItem]
    ) -> tuple[List[Dict[str, Any]], str, int]:
        if not items:
            return [], "", 0
        budget = int(self.config.get("token_budget", 600))
        item_limit = self._context_item_limit(len(items))
        summary_chars = self._context_summary_chars()
        consumed = 0
        lines: List[str] = ["=== Retrieved Context ==="]
        entries: List[Dict[str, Any]] = []
        for item in items[:item_limit]:
            meta = item.meta or {}
            max_len = summary_chars
            if meta.get("kind") == SURVIVAL_LEDGER_KIND:
                max_len = max(summary_chars, self.survival_ledger_summary_chars)
            summary = item.cached_summary(max_len=max_len)
            tokens = utils.estimate_tokens(summary)
            if consumed + tokens > budget:
                if entries:
                    break
                approx_chars = max(32, budget * 4)
                summary = summary[:approx_chars].rstrip()
                if len(item.cached_summary(max_len=max_len)) > len(summary):
                    summary = summary.rstrip() + "..."
                tokens = min(max(1, utils.estimate_tokens(summary)), budget)
            consumed += tokens
            source = meta.get("source", "unknown")
            timestamp = time.strftime("%Y-%m-%d", time.gmtime(item.timestamp))
            lines.append(f"- ({timestamp}) [source={source}]\n  {summary}")
            entries.append(
                {
                    "id": str(item.id),
                    "text": summary,
                    "summary": summary,
                    "meta": meta,
                    "timestamp": float(item.timestamp),
                    "level": item.level,
                    "fidelity": float(item.fidelity),
                    "salience": float(item.salience),
                    "tokens": tokens,
                }
            )
        if not entries:
            return [], "", 0
        return entries, "\n".join(lines), consumed

    @staticmethod
    def _is_quarantined_or_suppressed(item: MemoryItem) -> bool:
        meta = item.meta or {}
        state = str(meta.get("memory_state") or meta.get("lifecycle_state") or "").strip().lower()
        namespace = str(meta.get("namespace") or "").strip().lower()
        if state in {"quarantine", "quarantined", "suppressed", "deleted"}:
            return True
        return namespace in {"quarantine", "quarantined"}

    def _filter_retrievable_items(
        self, items: List[MemoryItem], *, include_quarantined: bool = False
    ) -> List[MemoryItem]:
        if include_quarantined:
            return list(items)
        return [item for item in items if not self._is_quarantined_or_suppressed(item)]

    def _recent_context_items(
        self,
        *,
        tenant_id: Optional[str],
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
        kinds: Optional[List[str]],
        phase: Optional[MemoryPhase],
        top_k: int,
        require_unscoped: bool = False,
        include_quarantined: bool = False,
    ) -> List[MemoryItem]:
        allowed_kinds = set(kinds or [])
        if not allowed_kinds and phase in {MemoryPhase.EXECUTE, MemoryPhase.DEBUG}:
            allowed_kinds = {"action", "observation", "error"}
        candidates: List[MemoryItem] = []
        for item in self.store.items():
            if not include_quarantined and self._is_quarantined_or_suppressed(item):
                continue
            meta = item.meta or {}
            if require_unscoped and any(
                meta.get(scope_key) is not None
                for scope_key in ("tenant_id", "client_id", "session_id", "instance_id")
            ):
                continue
            if tenant_id is not None and meta.get("tenant_id") != tenant_id:
                continue
            if client_id is not None and meta.get("client_id") != client_id:
                continue
            if session_id is not None and meta.get("session_id") != session_id:
                continue
            if instance_id is not None and meta.get("instance_id") != instance_id:
                continue
            if phase is not None:
                item_phase = meta.get("phase")
                if item_phase is not None and str(item_phase).strip().lower() != phase.value:
                    continue
            item_kind = str(meta.get("kind") or "memory").lower()
            if allowed_kinds and item_kind not in allowed_kinds:
                continue
            candidates.append(item)
        candidates.sort(key=lambda item: item.timestamp, reverse=True)
        return candidates[: max(1, top_k)]

    def _format_ltm_entries(self, items: List[MemoryStore.MemoryItem]) -> str:
        if not items:
            return ""
        lines: List[str] = ["=== Retrieved Memory ==="]
        for item in items:
            meta = item.meta or {}
            source = meta.get("source", "unknown")
            confidence = meta.get("confidence")
            scope = meta.get("scope", "session")
            timestamp = time.strftime("%Y-%m-%d", time.gmtime(item.timestamp))
            summary = item.cached_summary(max_len=220)
            conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
            lines.append(
                f"- ({timestamp}) [source={source} | conf={conf_text} | scope={scope}]\n  {summary}"
            )
        return "\n".join(lines)

    def _compose_structured_prompt(
        self,
        *,
        prompt: str,
        stm_summary: str,
        ltm_block: str,
    ) -> str:
        system_rules = "\n".join(
            [
                "=== System Rules ===",
                "- Respect commitments; do not contradict unless you call it out.",
                "- If uncertain, say so and ask questions.",
                "- Use retrieved memory as grounding and cite provenance inline.",
                "- Do NOT invent facts.",
                "- Keep responses concise and helpful.",
            ]
        )
        blocks: List[str] = [system_rules]
        if stm_summary:
            blocks.append("=== STM Summary ===\n" + stm_summary.strip())
        if ltm_block:
            blocks.append(ltm_block.strip())
        blocks.append("=== User ===\n" + prompt.strip())
        return "\n\n".join(blocks)

    def _compose_context_summary(self, stm_summary: str, ltm_block: str) -> str:
        blocks = []
        if stm_summary:
            blocks.append("=== STM Summary ===\n" + stm_summary.strip())
        if ltm_block:
            blocks.append(ltm_block.strip())
        return "\n\n".join(blocks)

    def _apply_ltm_writes(self, writes: List[MemoryWrite]) -> None:
        for write in writes:
            meta = dict(write.meta)
            if write.expires_at is not None:
                meta["expires_at"] = write.expires_at.isoformat()
            self.ingest(write.text, meta=meta)

    def _format_dml_context(self, entries: List[Dict]) -> str:
        if not entries:
            return ""
        lines = ["=== Daystrom Memory Lattice ==="]
        for entry in entries:
            summary = self._clean_context_fragment(entry["summary"])
            lines.append(
                f"- L{entry['level']} (f={entry['fidelity']:.2f}): {summary}"
            )
        return "\n".join(lines)

    @staticmethod
    def _clean_context_fragment(value: object) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\n?\[[^\]]*completion truncated\]", "", text)
        return text.strip()

    @classmethod
    def _clean_meta(cls, meta: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, str):
                cleaned[key] = cls._clean_context_fragment(value)
            else:
                cleaned[key] = value
        return cleaned

    def _format_workflow_text(
        self, task_description: str, steps: List[str], outcome: str
    ) -> str:
        lines = [
            f"Task: {task_description.strip()}",
            "Steps:",
        ]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
        lines.append(f"Outcome: {outcome.strip()}")
        return "\n".join(lines)

    def _compose_prompt(self, prompt: str, context: str) -> str:
        blocks: List[str] = []
        if context:
            blocks.append(
                "Answer the user directly and naturally. "
                "Treat the notes below as private grounding, not as something to announce. "
                "Do not mention DML, RAG, retrieved context, background notes, or say "
                "'according to' unless the user explicitly asks for provenance. "
                "Use only the relevant details. "
                "If the context is insufficient, say what is missing."
            )
            blocks.append("=== Private Grounding Notes ===")
            blocks.append(self._silent_context_for_model(context))
        blocks.append("=== User Prompt ===")
        blocks.append(prompt.strip())
        return "\n\n".join(blocks)

    @staticmethod
    def _silent_context_for_model(context: str) -> str:
        """Strip retrieval branding from context before it reaches the generator."""

        text = str(context or "").strip()
        if not text:
            return ""
        text = text.split("=== User Prompt ===", 1)[0].strip()
        replacements = {
            "=== Daystrom Memory Lattice ===": "",
            "=== RAG Retrieval ===": "",
            "=== Retrieved Context ===": "",
            "=== Retrieved Memory ===": "",
        }
        for needle, replacement in replacements.items():
            text = text.replace(needle, replacement)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _resolve_storage_path(self, candidate: Path) -> Path:
        path = candidate.expanduser()
        if path.is_absolute():
            return path
        relative = path
        if relative.parts and relative.parts[0] == self.storage_dir.name:
            if len(relative.parts) > 1:
                relative = Path(*relative.parts[1:])
            else:
                relative = Path(relative.name)
        return (self.storage_dir / relative).expanduser()

    def _estimate_salience(self, text: str) -> float:
        tokens = utils.estimate_tokens(text)
        return float(max(0.1, min(1.0, tokens / 200.0)))

    def _fallback_truncate(self, text: str, *, max_len: int) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."

    def _summary_for_item(self, item: MemoryStore.MemoryItem, *, max_len: int) -> str:
        summary = ""
        if item.meta:
            summary = str(item.meta.get("summary") or "").strip()
        if summary:
            return self._fallback_truncate(summary, max_len=max_len)
        return self._fallback_truncate(item.text, max_len=max_len)

    def _retrieve_items(
        self,
        prompt: str,
        top_k: Optional[int] = None,
        phase: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        include_quarantined: bool = False,
    ) -> List[MemoryStore.MemoryItem]:
        """Retrieve items with phase-aware filtering."""
        limit = self._resolve_dml_top_k(top_k)
        prompt_embedding = self._embed_query(prompt)
        semantic_items = self.store.retrieve(prompt_embedding, top_k=limit)
        literal_items: List[MemoryStore.MemoryItem] = []
        with contextlib.suppress(Exception):
            literal_items = [
                result.item
                for result in self.literal_retriever.retrieve(
                    prompt,
                    self.store.items(),
                    prompt_embedding,
                    top_k=limit,
                )
                if result.literal_score > 0.0
            ]
        items = []
        seen_ids: set[int] = set()
        for item in [*literal_items, *semantic_items]:
            raw_item_id = getattr(item, "id", id(item))
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                item_id = id(item)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            items.append(item)
            if len(items) >= limit:
                break
        items = self._filter_retrievable_items(items, include_quarantined=include_quarantined)

        # Apply phase-aware filtering
        if phase in ["execute", "debug"]:
            filtered = []
            for item in items:
                meta = item.meta or {}
                item_kind = str(meta.get("kind", "memory")).lower()
                item_phase = str(meta.get("phase") or "").strip().lower()
                if item_phase and item_phase != phase:
                    continue
                if item_kind in ["action", "observation", "error"]:
                    filtered.append(item)
            items = filtered

        # Apply kind filtering
        if kinds:
            filtered = []
            for item in items:
                meta = item.meta or {}
                item_kind = str(meta.get("kind", "memory")).lower()
                if any(kind.lower() in item_kind for kind in kinds):
                    filtered.append(item)
            items = filtered

        return items

    def _resolve_dml_top_k(self, requested: Optional[int]) -> int:
        """Resolve a safe retrieval cap.

        Non-positive or invalid values fall back to the configured default to
        avoid unbounded scans of the lattice during retrieval.
        """

        if requested is None:
            raw_value = self.config.get("dml_top_k", DEFAULT_DML_TOP_K)
        else:
            raw_value = requested
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid dml_top_k value %r; defaulting to %d.",
                raw_value,
                DEFAULT_DML_TOP_K,
            )
            return DEFAULT_DML_TOP_K
        if parsed <= 0:
            return DEFAULT_DML_TOP_K
        return parsed

    def _context_item_limit(self, available: int) -> int:
        if available <= 0:
            return 0
        try:
            configured = int(self.config.get("dml_context_max_items", 4) or 4)
        except (TypeError, ValueError):
            configured = 4
        return max(1, min(available, configured))

    def _context_summary_chars(self) -> int:
        try:
            configured = int(self.config.get("dml_context_summary_chars", 140) or 140)
        except (TypeError, ValueError):
            configured = 140
        return max(64, min(320, configured))

    def _prepare_context(
        self, prompt: str, items: List[MemoryStore.MemoryItem]
    ) -> tuple[List[Dict], str, int]:
        budget = int(self.config.get("token_budget", 600))
        item_limit = self._context_item_limit(len(items))
        summary_chars = self._context_summary_chars()
        consumed = 0
        lines: List[str] = ["=== Daystrom Memory Lattice ==="]
        entries: List[Dict] = []
        for item in items[:item_limit]:
            summary = self._clean_context_fragment(item.cached_summary(max_len=summary_chars))
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
                    "meta": self._clean_meta(item.meta or {}),
                    "tokens": tokens,
                }
            )
        lines.append("=== User Prompt ===")
        lines.append(prompt)
        preamble = "\n".join(lines)
        return entries, preamble, consumed

    def _classify_mode(self, prompt: str) -> str:
        """Backward compatible wrapper around the intent router."""
        return decide_mode(prompt)

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
            summary = item.cached_summary(max_len=220)
            source = item.meta.get("doc_path") if item.meta else None
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
            results.append(
                {
                    "text": summary,
                    "context": [summary],
                    "semantic_score": similarity,
                    "literal_score": 0.0,
                    "source": source,
                    "origin": "semantic",
                }
            )
        return results

    def _literal_retrieve(
        self,
        prompt: str,
        items: List[Any],
        query_embedding: np.ndarray,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        store_matches: List[Dict[str, Any]] = []
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                store_matches = self.persistent_rag_store.search(query_embedding, top_k=top_k)
        formatted: List[Dict[str, Any]] = []
        formatted.extend(self._format_rag_matches(store_matches))
        fallback = self.literal_retriever.retrieve(prompt, items, query_embedding, top_k=top_k)
        for result in fallback:
            meta = result.item.meta if getattr(result, "item", None) else {}
            formatted.append(
                {
                    "context": list(result.context),
                    "semantic_score": float(result.semantic_score),
                    "literal_score": float(result.literal_score),
                    "source": result.source,
                    "meta": dict(meta or {}),
                    "text": result.snippet,
                    "origin": "literal",
                }
            )
        formatted.sort(
            key=lambda result: (
                float(result.get("literal_score", 0.0)),
                float(result.get("semantic_score", 0.0)),
            ),
            reverse=True,
        )
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for result in formatted:
            key = "|".join(str(segment) for segment in result.get("context", []))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
            if len(deduped) >= top_k:
                break
        return deduped

    def _format_rag_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        max_snippet_chars = getattr(self.literal_retriever, "max_snippet_chars", 320)

        def _clip_segment(value: Any) -> str:
            segment = str(value or "").strip()
            if not segment:
                return ""
            if len(segment) <= max_snippet_chars:
                return segment
            return segment[: max_snippet_chars - 3] + "..."

        for match in matches:
            meta = dict(match.get("meta") or {})
            context_segments: List[str] = []
            raw_context = meta.get("context")
            if isinstance(raw_context, list):
                context_segments.extend(
                    _clip_segment(segment)
                    for segment in raw_context
                    if isinstance(segment, str) and segment.strip()
                )
            for key in ("context_before", "preceding"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    clipped = _clip_segment(value)
                    if clipped:
                        context_segments.append(clipped)
            text = _clip_segment(match.get("text"))
            if text:
                context_segments.append(text)
            for key in ("context_after", "following"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    clipped = _clip_segment(value)
                    if clipped:
                        context_segments.append(clipped)
            deduped: List[str] = []
            seen: set[str] = set()
            for segment in context_segments:
                if not segment or segment in seen:
                    continue
                seen.add(segment)
                deduped.append(segment)
            if not deduped and text:
                deduped = [text]
            source = self.literal_retriever._resolve_source(meta) if meta else None
            formatted.append(
                {
                    "id": match.get("id"),
                    "text": text,
                    "meta": meta,
                    "literal_score": 0.0,
                    "semantic_score": float(match.get("score", 0.0)),
                    "source": source,
                    "context": deduped,
                    "origin": "semantic",
                }
            )
        return formatted

    def _blend_results(
        self,
        literal_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        alpha: float,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        def _normalise(entry: Dict[str, Any]) -> Dict[str, Any]:
            data = dict(entry)
            context_value = data.get("context") or []
            if isinstance(context_value, str):
                context_list = [context_value]
            else:
                context_list = [
                    str(segment).strip()
                    for segment in context_value
                    if isinstance(segment, str) and segment.strip()
                ]
            data["context"] = context_list
            data["semantic_score"] = float(data.get("semantic_score", 0.0))
            data["literal_score"] = float(data.get("literal_score", 0.0))
            origin = str(data.get("origin") or "semantic")
            data["origin"] = origin
            if origin == "literal":
                data["context"] = self._clip_literal_context(data["context"])
            return data

        blended: List[Dict[str, Any]] = []
        blended.extend(_normalise(res) for res in literal_results)
        blended.extend(_normalise(res) for res in semantic_results)
        for entry in blended:
            semantic_score = entry.get("semantic_score", 0.0)
            literal_score = entry.get("literal_score", 0.0)
            blended_score = alpha * semantic_score + (1 - alpha) * literal_score
            if literal_score > 0.0:
                blended_score = max(blended_score, literal_score)
            entry["final_score"] = blended_score
        blended.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for entry in blended:
            key = "|".join(entry.get("context", []))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
            if len(deduped) >= top_k:
                break
        constrained = self._apply_token_budgets(deduped)
        return constrained

    def _clip_literal_context(self, segments: List[str]) -> List[str]:
        if not segments:
            return []
        tokens_used = 0
        clipped: List[str] = []
        for segment in segments[: self.literal_snippet_cap]:
            text = str(segment or "").strip()
            if not text:
                continue
            tokens = max(1, utils.estimate_tokens(text))
            if tokens_used + tokens > self.literal_token_cap:
                remaining = self.literal_token_cap - tokens_used
                if remaining <= 0:
                    break
                approx_chars = max(32, remaining * 4)
                truncated = text[:approx_chars]
                if len(text) > approx_chars:
                    truncated = truncated.rstrip() + "..."
                clipped.append(truncated)
                tokens_used = self.literal_token_cap
                break
            clipped.append(text)
            tokens_used += tokens
        return clipped

    def _apply_token_budgets(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        total_budget = int(self.config.get("token_budget", 600))
        total_budget = max(1, total_budget)
        budgets = self.config.get("budgets", {}) or {}

        def _pct(key: str, default: float) -> float:
            raw = budgets.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        semantic_limit = max(0, int(total_budget * _pct("semantic_pct", 0.7)))
        literal_limit = max(0, int(total_budget * _pct("literal_pct", 0.2)))
        free_limit = max(0, total_budget - semantic_limit - literal_limit)

        consumed = {"semantic": 0, "literal": 0}
        free_used = 0
        total_used = 0
        allowed: List[Dict[str, Any]] = []
        for entry in entries:
            origin = str(entry.get("origin") or "semantic")
            if origin not in consumed:
                origin = "semantic"
            context_segments = entry.get("context") or []
            if not isinstance(context_segments, list):
                context_segments = [str(context_segments)]
            tokens = entry.get("tokens")
            if not isinstance(tokens, (int, float)):
                tokens = sum(max(1, utils.estimate_tokens(seg)) for seg in context_segments)
            tokens = int(max(1, tokens))
            if total_used + tokens > total_budget:
                continue
            limit = semantic_limit if origin == "semantic" else literal_limit
            if consumed[origin] + tokens <= limit:
                consumed[origin] += tokens
                total_used += tokens
                entry["tokens"] = tokens
                allowed.append(entry)
                continue
            if free_used + tokens <= free_limit:
                free_used += tokens
                total_used += tokens
                entry["tokens"] = tokens
                allowed.append(entry)

        if not allowed and entries:
            first = entries[0]
            first_tokens = sum(
                max(1, utils.estimate_tokens(seg)) for seg in first.get("context", [])
            )
            first["tokens"] = int(max(1, first_tokens))
            return [first]
        return allowed

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        try:
            self.close()
        except Exception:
            pass
