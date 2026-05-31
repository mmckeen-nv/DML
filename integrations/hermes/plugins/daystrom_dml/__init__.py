"""Daystrom DML memory provider for Hermes/Citizen Snips.

This provider intentionally uses the Hermes-owned Daystrom DML launcher,
source tree, venv, and runtime store. It does not route model inference
through DML; it only contributes memory recall and DPM/personality overlay
context, then mirrors completed turns back into DML.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get, load_config

logger = logging.getLogger(__name__)

_DEFAULT_INTEGRATION_DIR = Path("/Users/markmckeen/.hermes/hermes-agent/integrations/daystrom-dml")


def _clean_text(value: str, limit: int = 4000) -> str:
    text = " ".join((value or "").split())
    return text[:limit].rstrip()


_MEMORY_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>",
    re.IGNORECASE,
)
_DAYSTROM_BLOCK_RE = re.compile(
    r"^=== Daystrom (?:Personality Matrix Overlay|DML Active Continuity|DML Retrieved Memory) ===[\s\S]*?(?=^=== |\Z)",
    re.MULTILINE,
)
_ROLE_PREFIX_RE = re.compile(r"^(?:(?:user|assistant):\s*){2,}", re.IGNORECASE)
_ANY_ROLE_PREFIX_RE = re.compile(r"\b(?:(?:user|assistant):\s*){2,}", re.IGNORECASE)
_SCAFFOLD_RE = re.compile(
    r"^(?:i(?:'|’)ll|i am|i’m|let me|reading|checking|inspecting)\b",
    re.IGNORECASE,
)
_SMOKE_TEST_RE = re.compile(
    r"\b(?:smoke[- ]?test|self[- ]?test|test record|pre[- ]?fix|completed snips_?2 turn|completed citizen snips turn)\b",
    re.IGNORECASE,
)
_TOOL_EXHAUST_RE = re.compile(
    r"\b(?:chunk id:|wall time:|process exited|original token count|output:|"
    r"functions\.exec_command|multi_tool_use\.parallel|apply_patch|namespace web|"
    r"^total output lines:|\[truncated\])\b",
    re.IGNORECASE | re.MULTILINE,
)
_DURABLE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("preference", re.compile(r"\b(?:i|we|mark)\s+(?:prefers?|likes?|wants?|needs?|expects?)\b|\bplease\s+(?:prefer|use|keep|make|remember|avoid)\b", re.IGNORECASE)),
    ("identity", re.compile(r"\b(?:assistant identity|citizen snips|snips_?2|you are citizen snips)\b", re.IGNORECASE)),
    ("constraint", re.compile(r"\b(?:do not|don't|never|always|must|should|explicit(?:ly)? instructed|current-turn instructions?)\b", re.IGNORECASE)),
    ("artifact", re.compile(r"\b(?:changed|updated|added|implemented|fixed|created|wrote|patched|configured|wired|compiled)\b.*(?:/[^\s]+|\b[\w.-]+\.(?:py|yaml|yml|json|md|sh|txt)\b)", re.IGNORECASE)),
    ("validation", re.compile(r"\b(?:test|pytest|py_compile|validation|smoke script|compile)\b.*\b(?:pass(?:ed)?|fail(?:ed)?|error|blocked|ran|run)\b", re.IGNORECASE)),
    ("blocker", re.compile(r"\b(?:blocker|blocked|failure|bug|regression|root cause|workaround|resolved|fix)\b", re.IGNORECASE)),
    ("checkpoint", re.compile(r"\b(?:current task|active task|next action|remaining|checkpoint|handoff|resume|continuity)\b", re.IGNORECASE)),
)
_SENSITIVE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization|bearer)\b\s*[:=]\s*\S+"
)


def _strip_injected_context(value: str) -> str:
    """Remove API-time memory injection from text before writing DML handoffs."""
    text = _MEMORY_CONTEXT_RE.sub(" ", value or "")
    text = _DAYSTROM_BLOCK_RE.sub(" ", text)
    text = _ANY_ROLE_PREFIX_RE.sub("", text)
    return text


def _sentenceish_fragments(text: str, *, limit: int = 4) -> List[str]:
    cleaned = _clean_text(_strip_injected_context(text), 2400)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    fragments: List[str] = []
    seen: set[str] = set()
    for part in parts:
        line = _ROLE_PREFIX_RE.sub("", part).strip(" -\t")
        if not line:
            continue
        if _TOOL_EXHAUST_RE.search(line):
            continue
        key = re.sub(r"\W+", " ", line).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        fragments.append(line)
        if len(fragments) >= limit:
            break
    return fragments


def _fit_sentence_boundary(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit)].rstrip()
    sentence_cut = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if sentence_cut >= max(40, int(limit * 0.55)):
        return cut[: sentence_cut + 1].rstrip()
    word_cut = cut.rfind(" ")
    if word_cut >= max(20, int(limit * 0.45)):
        return cut[:word_cut].rstrip()
    return cut.rstrip(" .,;:-")


def _redact_sensitive(text: str) -> str:
    return _SENSITIVE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text or "")


def _classify_turn_memory(user_content: str, assistant_content: str) -> Dict[str, Any]:
    raw_user = _strip_injected_context(user_content)
    raw_assistant = _strip_injected_context(assistant_content)
    combined = _clean_text(f"{raw_user}\n{raw_assistant}", 5000)
    if not combined:
        return {"keep": False, "score": 0.0, "memory_class": "empty", "reasons": ["empty"]}

    score = 0.0
    classes: List[str] = []
    reasons: List[str] = []
    for memory_class, pattern in _DURABLE_PATTERNS:
        if pattern.search(combined):
            classes.append(memory_class)
            score += 0.24
    if classes:
        reasons.append("durable_signal")

    user_fragments = _sentenceish_fragments(raw_user, limit=3)
    assistant_fragments = _sentenceish_fragments(raw_assistant, limit=4)
    if user_fragments:
        score += 0.08
    if assistant_fragments:
        score += 0.06

    noise_hits = 0
    if _SMOKE_TEST_RE.search(combined):
        noise_hits += 3
        reasons.append("smoke_or_self_test")
    if _MEMORY_CONTEXT_RE.search(user_content or "") or _MEMORY_CONTEXT_RE.search(assistant_content or ""):
        noise_hits += 2
        reasons.append("injected_memory_context")
    if _DAYSTROM_BLOCK_RE.search(user_content or "") or _DAYSTROM_BLOCK_RE.search(assistant_content or ""):
        noise_hits += 2
        reasons.append("daystrom_context_block")
    if _TOOL_EXHAUST_RE.search(combined):
        noise_hits += 1
        reasons.append("tool_output_boilerplate")
    role_prefix_count = len(re.findall(r"\b(?:user|assistant):\s*\|?\s*(?:user|assistant):", combined, flags=re.IGNORECASE))
    if role_prefix_count:
        noise_hits += min(3, role_prefix_count)
        reasons.append("repeated_role_prefix")
    scaffold_lines = sum(1 for line in combined.splitlines() if _SCAFFOLD_RE.match(line.strip()))
    if scaffold_lines:
        noise_hits += min(2, scaffold_lines)
        reasons.append("assistant_scaffolding")
    score -= noise_hits * 0.16

    memory_class = classes[0] if classes else "low_signal"
    if not classes:
        reasons.append("no_durable_signal")
    if "smoke_or_self_test" in reasons:
        keep = False
    else:
        keep = bool(classes) and score >= 0.28

    summary_parts: List[str] = []
    if user_fragments:
        summary_parts.append("User signal: " + _fit_sentence_boundary(" ".join(user_fragments), 360))
    if assistant_fragments and memory_class in {"artifact", "validation", "blocker", "checkpoint"}:
        summary_parts.append("Assistant outcome: " + _fit_sentence_boundary(" ".join(assistant_fragments), 520))
    summary = _redact_sensitive(_fit_sentence_boundary(" ".join(summary_parts), 900))
    if not summary:
        keep = False
    return {
        "keep": keep,
        "score": round(max(0.0, min(1.0, score)), 3),
        "memory_class": memory_class,
        "reasons": reasons[:8],
        "summary": summary,
    }


def _handoff_fragment(role: str, content: str, *, limit: int = 900) -> str:
    text = _strip_injected_context(content)
    lines: List[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = _ROLE_PREFIX_RE.sub("", raw).strip()
        if not line:
            continue
        # Drop obvious progress/scaffolding chatter from the checkpoint. The
        # actual completed result is still captured via sync_turn/ingest.
        if role == "assistant" and _SCAFFOLD_RE.match(line):
            continue
        key = re.sub(r"\W+", " ", line).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return _clean_text(" ".join(lines), limit)


class DaystromDMLProvider(MemoryProvider):
    """Hermes external-memory provider backed by the local Daystrom DML store."""

    def __init__(self) -> None:
        self._cfg = self._load_provider_config()
        integration_dir = Path(self._cfg.get("integration_dir") or _DEFAULT_INTEGRATION_DIR)
        self.integration_dir = integration_dir
        self.launcher = Path(self._cfg.get("launcher") or integration_dir / "bin" / "hermes-dml-memory")
        self.venv_python = Path(self._cfg.get("venv_python") or integration_dir / ".venv-dml" / "bin" / "python")
        self.source_dir = Path(self._cfg.get("source_dir") or integration_dir / "source")
        self.store_dir = Path(self._cfg.get("storage_dir") or integration_dir / "stores" / "hermes-runtime-store")
        self.config_path = Path(self._cfg.get("config_path") or integration_dir / "source" / "openclaw-wrapper" / "config" / "dml_gpu_only.yaml")
        self.tenant_id = str(self._cfg.get("tenant_id") or "openclaw")
        self.client_id = self._cfg.get("client_id") or "snips2"
        self.top_k = int(self._cfg.get("top_k") or 6)
        self.timeout = float(self._cfg.get("timeout_seconds") or 20)
        self.max_context_chars = int(self._cfg.get("max_context_chars") or 5000)
        self.sync_turns = bool(self._cfg.get("sync_turns", True))
        self.enable_personality = bool(self._cfg.get("enable_personality", True))
        self.enable_memory = bool(self._cfg.get("enable_memory", True))
        self.no_require_gpu = bool(self._cfg.get("no_require_gpu", True))
        self._session_id = ""
        self._thread_id = ""
        self._chat_id = ""
        self._relationship_id = str(self._cfg.get("relationship_id") or "relationship:mark-snips2")
        self._project_id = str(self._cfg.get("project_id") or "project:snips2")
        self._last_sync_key = ""
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "daystrom_dml"

    def _load_provider_config(self) -> Dict[str, Any]:
        try:
            cfg = load_config()
            block = cfg_get(cfg, "memory", "daystrom_dml") or {}
            return block if isinstance(block, dict) else {}
        except Exception:
            return {}

    def is_available(self) -> bool:
        return self.launcher.exists() and os.access(self.launcher, os.X_OK) and self.store_dir.exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = self._shape_session_id(session_id, kwargs)
        self._thread_id = str(kwargs.get("thread_id") or "")
        self._chat_id = str(kwargs.get("chat_id") or "")
        # Validate best-effort. Never block agent startup on DML rough edges.
        try:
            self._run_cli(["health"], timeout=min(self.timeout, 15))
        except Exception as exc:
            logger.warning("Daystrom DML health check failed during init: %s", exc)

    def _shape_session_id(self, session_id: str, kwargs: Dict[str, Any]) -> str:
        thread = str(kwargs.get("thread_id") or "").strip()
        chat = str(kwargs.get("chat_id") or "").strip()
        if thread:
            return f"snips2-discord-{thread}"
        if chat:
            return f"snips2-discord-{chat}"
        return session_id or "snips2-hermes-default"

    def system_prompt_block(self) -> str:
        return (
            "Daystrom DML memory/personality provider is active for Citizen Snips. "
            "Use injected DML context as potentially relevant recall, subordinate to current user instructions. "
            "The DML inference/frontier pipeline is intentionally not active."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = _clean_text(query, 1200)
        if not query:
            return ""
        effective_session = self._session_id or session_id or "snips2-hermes-default"
        blocks: List[str] = []
        if self.enable_personality:
            overlay = self._personality_overlay(query)
            block = self._format_personality_overlay(overlay)
            if block:
                blocks.append(block)
        if self.enable_memory:
            # Resume preserves active continuity even when retrieval confidence is low.
            resume_block = self._resume_block(effective_session)
            if resume_block:
                blocks.append(resume_block)
            retrieve_block = self._retrieve_block(query, effective_session)
            if retrieve_block:
                blocks.append(retrieve_block)
        return self._fit("\n\n".join(blocks), self.max_context_chars)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self.sync_turns:
            return
        user = _clean_text(user_content, 1800)
        assistant = _clean_text(assistant_content, 2200)
        if not user or not assistant:
            return
        hygiene = _classify_turn_memory(user_content, assistant_content)
        if not hygiene.get("keep"):
            logger.debug("Daystrom DML sync_turn skipped low-value turn: %s", hygiene)
            return
        summary = str(hygiene.get("summary") or "").strip()
        key = f"{hash(summary)}:{hygiene.get('memory_class')}"
        with self._lock:
            if key == self._last_sync_key:
                return
            self._last_sync_key = key
        effective_session = self._session_id or session_id or "snips2-hermes-default"
        text = f"Citizen Snips durable turn memory. {summary}"
        meta = {
            "source": "hermes-memory-provider",
            "phase": "completed-turn",
            "provider": "daystrom_dml",
            "memory_class": hygiene.get("memory_class"),
            "hygiene_score": hygiene.get("score"),
            "hygiene_reasons": hygiene.get("reasons"),
            "summary_source": "hermes_daystrom_hygiene_v1",
        }
        if self._thread_id:
            meta["thread_id"] = self._thread_id
        if self._chat_id:
            meta["chat_id"] = self._chat_id
        try:
            self._run_cli([
                "ingest",
                "--kind", "observation",
                "--session-id", effective_session,
                "--tenant-id", self.tenant_id,
                "--client-id", self.client_id,
                "--summary-policy", "cheap",
                "--filter-noise",
                "--meta", json.dumps(meta, separators=(",", ":")),
                "--text", text,
            ], timeout=self.timeout)
        except Exception as exc:
            logger.debug("Daystrom DML sync_turn failed: %s", exc)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Retrieval is synchronous and capped by timeout in prefetch(); no background worker yet.
        return None

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        try:
            tail = []
            for msg in messages[-16:]:
                # Skip internal/ephemeral messages if the caller ever marks
                # them explicitly. API-time memory injection is also stripped
                # from content below before writing a DML checkpoint.
                if any(str(key).startswith("_") for key in msg):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, str) and role in {"user", "assistant"}:
                    fragment = _handoff_fragment(role, content, limit=700)
                    if fragment:
                        tail.append(f"{role}: {fragment}")
            tail = tail[-4:]
            if not tail:
                return ""
            state = _clean_text(" | ".join(tail), 2500)
            self._run_cli([
                "handoff",
                "--thread", self._thread_id or self._session_id or "snips2-hermes",
                "--state", state,
                "--task", "Preserve Citizen Snips DML memory/personality continuity across Hermes compression.",
                "--next-action", "Resume from Daystrom DML and retrieve relevant context before continuing.",
                "--session-id", self._session_id or "snips2-hermes-default",
                "--tenant-id", self.tenant_id,
                "--client-id", self.client_id,
            ], timeout=self.timeout)
            return "Daystrom DML handoff checkpoint was written before compression."
        except Exception:
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def _run_cli(self, args: List[str], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        cmd = [str(self.launcher)]
        if self.no_require_gpu:
            cmd.append("--no-require-gpu")
        cmd.extend(["--storage-dir", str(self.store_dir), "--config-path", str(self.config_path)])
        cmd.extend(args)
        env = os.environ.copy()
        env.setdefault("DAYSTROM_DPM_RELATIONSHIP_ID", self._relationship_id)
        env.setdefault("DAYSTROM_DPM_PROJECT_ID", self._project_id)
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout or self.timeout,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"DML command failed rc={proc.returncode}: {proc.stderr.strip() or proc.stdout[:500]}")
        return self._parse_json_output(proc.stdout)

    @staticmethod
    def _parse_json_output(stdout: str) -> Dict[str, Any]:
        text = stdout or ""
        start = text.find("{")
        if start < 0:
            return {"raw": text}
        return json.loads(text[start:])

    def _resume_block(self, session_id: str) -> str:
        try:
            data = self._run_cli(["resume", "--session-id", session_id, "--tenant-id", self.tenant_id], timeout=self.timeout)
        except Exception:
            return ""
        raw = str(data.get("raw_context") or "").strip()
        if not raw:
            return ""
        return "=== Daystrom DML Active Continuity ===\n" + self._fit(raw, 1600)

    def _retrieve_block(self, query: str, session_id: str) -> str:
        try:
            data = self._run_cli([
                "retrieve",
                "--query", query,
                "--session-id", session_id,
                "--tenant-id", self.tenant_id,
                "--top-k", str(self.top_k),
                "--ground-truth-policy", "never",
                "--no-reform-memory",
            ], timeout=self.timeout)
        except Exception:
            return ""
        raw = str(data.get("raw_context") or "").strip()
        if not raw:
            return ""
        return "=== Daystrom DML Retrieved Memory ===\n" + self._fit(raw, 2800)

    def _personality_overlay(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not (self.venv_python.exists() and self.source_dir.exists()):
            return None
        code = r'''
import json, os, sys
from pathlib import Path
source = Path(os.environ["DAYSTROM_DML_SOURCE"])
sys.path.insert(0, str(source / "dml_core"))
from daystrom_dml.dml_adapter import DMLAdapter
adapter = DMLAdapter(
    config_path=os.environ.get("DAYSTROM_DML_CONFIG"),
    config_overrides={
        "storage_dir": os.environ["DAYSTROM_DML_STORE"],
        "dml.agentic_mode.enabled": True,
        "strict_llm_required": False,
        "dpm": {
            "enable": True,
            "mode": os.environ.get("DAYSTROM_DPM_MODE", "active-write"),
            "preference_graph_path": str(Path(os.environ["DAYSTROM_DML_STORE"]) / "dpm_preference_graph.json"),
            "relationship_id": os.environ.get("DAYSTROM_DPM_RELATIONSHIP_ID", "relationship:mark-snips2"),
            "project_id": os.environ.get("DAYSTROM_DPM_PROJECT_ID", "project:snips2"),
            "token_budget": int(os.environ.get("DAYSTROM_DPM_TOKEN_BUDGET", "80")),
        },
    },
)
overlay = adapter.personality_overlay(
    prompt=os.environ.get("DAYSTROM_DML_PROMPT", ""),
    thread_id=os.environ.get("DAYSTROM_DML_THREAD_ID") or None,
    relationship_id=os.environ.get("DAYSTROM_DPM_RELATIONSHIP_ID", "relationship:mark-snips2"),
    project_id=os.environ.get("DAYSTROM_DPM_PROJECT_ID", "project:snips2"),
)
print(json.dumps(overlay or {}))
'''
        env = os.environ.copy()
        env.update({
            "DAYSTROM_DML_SOURCE": str(self.source_dir),
            "DAYSTROM_DML_STORE": str(self.store_dir),
            "DAYSTROM_DML_CONFIG": str(self.config_path),
            "DAYSTROM_DML_PROMPT": prompt,
            "DAYSTROM_DML_THREAD_ID": self._thread_id,
            "DAYSTROM_DPM_RELATIONSHIP_ID": self._relationship_id,
            "DAYSTROM_DPM_PROJECT_ID": self._project_id,
        })
        try:
            proc = subprocess.run(
                [str(self.venv_python), "-c", code],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(self.timeout, 12),
                env=env,
            )
            if proc.returncode != 0:
                logger.debug("Daystrom DPM overlay failed: %s", proc.stderr.strip())
                return None
            start = proc.stdout.find("{")
            if start < 0:
                return None
            data = json.loads(proc.stdout[start:])
            return data if isinstance(data, dict) and data else None
        except Exception as exc:
            logger.debug("Daystrom DPM overlay exception: %s", exc)
            return None

    def _format_personality_overlay(self, overlay: Optional[Dict[str, Any]]) -> str:
        if not overlay:
            return ""
        body = overlay.get("overlay") if isinstance(overlay.get("overlay"), dict) else {}
        rendered = str(body.get("rendered_text") or body.get("persona_summary") or "").strip()
        directives = body.get("style_directives") if isinstance(body.get("style_directives"), list) else []
        do_not = body.get("do_not_do") if isinstance(body.get("do_not_do"), list) else []
        lines = ["=== Daystrom Personality Matrix Overlay ==="]
        if rendered:
            lines.append(rendered)
        for directive in directives[:4]:
            if directive and str(directive) not in rendered:
                lines.append(f"- {directive}")
        for item in do_not[:3]:
            lines.append(f"- Constraint: {item}")
        lines.append("- Constraint: Explicit current-turn user instructions override this overlay.")
        return self._fit("\n".join(lines), 1200)

    @staticmethod
    def _fit(text: str, limit: int) -> str:
        fitted = _fit_sentence_boundary(text, limit)
        if len((text or "").strip()) <= limit:
            return fitted
        return fitted + "\n... [truncated]"


def register(ctx) -> None:
    ctx.register_memory_provider(DaystromDMLProvider())
