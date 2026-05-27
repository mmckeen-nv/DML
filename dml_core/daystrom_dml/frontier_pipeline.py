"""Frontier-model compression and routing pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import utils


DraftGenerator = Callable[[str, int], str]


@dataclass(frozen=True)
class FrontierPipelineConfig:
    """Controls how DML prepares work for a frontier model."""

    top_k: int = 8
    local_max_tokens: int = 256
    frontier_max_tokens: int = 512
    min_context_tokens_for_verify: int = 24
    include_local_draft: bool = True


class FrontierCompressionPipeline:
    """Prepare compact DML context and optional local drafts for frontier LLMs.

    This is intentionally a controller in front of a frontier endpoint, not a
    frontier client. The caller owns the final OpenAI/Claude/Codex API call.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        config: Optional[FrontierPipelineConfig] = None,
        draft_generator: Optional[DraftGenerator] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or FrontierPipelineConfig()
        self.draft_generator = draft_generator

    def prepare(
        self,
        prompt: str,
        *,
        tenant_id: str = "openclaw",
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        top_k: Optional[int] = None,
        local_max_tokens: Optional[int] = None,
        frontier_max_tokens: Optional[int] = None,
        include_local_draft: Optional[bool] = None,
        direct_input_tokens_estimate: Optional[int] = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        final_top_k = int(top_k or self.config.top_k)
        local_limit = int(local_max_tokens or self.config.local_max_tokens)
        frontier_limit = int(frontier_max_tokens or self.config.frontier_max_tokens)
        use_draft = self.config.include_local_draft if include_local_draft is None else bool(include_local_draft)

        retrieval_started = time.perf_counter()
        report = self.adapter.retrieve_context(
            prompt,
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
            top_k=final_top_k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        dml_context = str(report.get("raw_context") or "")
        context_tokens = int(report.get("context_tokens") or utils.estimate_tokens(dml_context))

        local_prompt = self._local_draft_prompt(prompt, dml_context)
        local_draft = ""
        local_latency_ms: Optional[float] = None
        if use_draft and self.draft_generator is not None and dml_context.strip():
            draft_started = time.perf_counter()
            local_draft = (self.draft_generator(local_prompt, local_limit) or "").strip()
            local_latency_ms = (time.perf_counter() - draft_started) * 1000.0

        mode = self._select_mode(
            dml_context=dml_context,
            context_tokens=context_tokens,
            local_draft=local_draft,
        )
        frontier_prompt = self._frontier_prompt(
            prompt=prompt,
            dml_context=dml_context,
            local_draft=local_draft,
            mode=mode,
        )
        frontier_input_tokens = utils.estimate_tokens(frontier_prompt)
        direct_input_tokens = int(direct_input_tokens_estimate or 0)
        if direct_input_tokens <= 0:
            direct_input_tokens = frontier_input_tokens
        saved_input_tokens = max(0, direct_input_tokens - frontier_input_tokens)

        return {
            "mode": mode,
            "prompt": prompt,
            "frontier_prompt": frontier_prompt,
            "frontier_max_tokens": frontier_limit,
            "local_prompt": local_prompt,
            "local_draft": local_draft,
            "dml_context": dml_context,
            "dml_report": report,
            "telemetry": {
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "local_draft_latency_ms": round(local_latency_ms, 2) if local_latency_ms is not None else None,
                "dml_context_tokens": context_tokens,
                "local_prompt_tokens": utils.estimate_tokens(local_prompt),
                "local_draft_tokens": utils.estimate_tokens(local_draft),
                "frontier_input_tokens": frontier_input_tokens,
                "direct_input_tokens_estimate": direct_input_tokens,
                "input_tokens_saved_estimate": saved_input_tokens,
                "input_savings_pct_estimate": round(
                    (saved_input_tokens / max(1, direct_input_tokens)) * 100.0,
                    1,
                ),
                "retrieved_items": len(report.get("items") or []),
                "survival_ledger_included": bool(report.get("survival_ledger_included")),
            },
        }

    def _select_mode(self, *, dml_context: str, context_tokens: int, local_draft: str) -> str:
        if not dml_context.strip():
            return "frontier_full"
        if local_draft.strip():
            return "frontier_verify_local_draft"
        return "frontier_with_dml_context"

    def _local_draft_prompt(self, prompt: str, dml_context: str) -> str:
        return (
            "Draft a concise answer using only the DML context. Preserve exact IDs, "
            "decisions, blockers, and next steps. Mark uncertainty plainly.\n\n"
            f"DML context:\n{dml_context}\n\nUser request:\n{prompt}\n\nDraft:"
        )

    def _frontier_prompt(
        self,
        *,
        prompt: str,
        dml_context: str,
        local_draft: str,
        mode: str,
    ) -> str:
        if mode == "frontier_full":
            return (
                "Answer the user request. No reliable DML context was available, so reason from the request directly.\n\n"
                f"User request:\n{prompt}\n\nFinal answer:"
            )
        if local_draft.strip():
            return (
                "You are the frontier verifier/finalizer. Use the DML context as the authority. "
                "Use the local draft only as a cheap first pass; correct any wrong IDs, missing facts, "
                "or unsupported claims. Answer concisely.\n\n"
                f"DML context:\n{dml_context}\n\nLocal draft:\n{local_draft}\n\nUser request:\n{prompt}\n\nFinal answer:"
            )
        return (
            "You are the frontier finalizer. Use the compact DML context as authoritative memory. "
            "Answer concisely and preserve exact IDs when present.\n\n"
            f"DML context:\n{dml_context}\n\nUser request:\n{prompt}\n\nFinal answer:"
        )
