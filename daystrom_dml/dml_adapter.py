"""High level adapter orchestrating the Daystrom Memory Lattice."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

from .embeddings import Embedder, create_embedder
from .gpt_runner import GPTRunner
from .memory_store import MemoryStore
from .summarizer import DummySummarizer, LLMSummarizer, Summarizer
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

    def build_preamble(self, prompt: str, top_k: Optional[int] = None) -> str:
        top_k = top_k or int(self.config.get("top_k", 6))
        prompt_embedding = self.embedder.embed(prompt)
        items = self.store.retrieve(prompt_embedding, top_k=top_k)
        budget = int(self.config.get("token_budget", 600))
        consumed = 0
        lines: List[str] = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        for item in items:
            summary = self.summarizer.summarize(item.text, max_len=180)
            tokens = utils.estimate_tokens(summary)
            if consumed + tokens > budget:
                break
            consumed += tokens
            lines.append(f"- L{item.level} (f={item.fidelity:.2f}): {summary}")
        lines.append("=== User Prompt ===")
        lines.append(prompt)
        return "\n".join(lines)

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
    def _load_config(self, path: str | os.PathLike | None) -> Dict:
        default_path = Path(__file__).with_name("config.yaml")
        config_file = Path(path) if path else default_path
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data

    def _estimate_salience(self, text: str) -> float:
        tokens = utils.estimate_tokens(text)
        return float(max(0.1, min(1.0, tokens / 200.0)))

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        try:
            self.close()
        except Exception:
            pass

