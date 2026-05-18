"""Runtime support for the Daystrom Personality Matrix overlay."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import utils


ACTIVE_MODES = {"active-read", "active-write"}
VALID_MODES = {"disabled", "observe-only", "active-read", "active-write"}


class PersonalityMatrix:
    """Load bounded DPM guidance and expose it as a runtime overlay.

    This is intentionally read-first: the matrix does not infer or mutate
    preferences. It makes existing DPM records usable by the runtime while
    preserving the contract that explicit current-turn instructions win.
    """

    def __init__(self, settings: Any, *, storage_dir: Path) -> None:
        self.settings = settings
        self.storage_dir = storage_dir
        self.enabled = bool(getattr(settings, "enable", False))
        self.mode = str(getattr(settings, "mode", "disabled") or "disabled").strip().lower()
        if self.mode not in VALID_MODES:
            self.mode = "disabled"
        self.max_overlay_chars = max(1, int(getattr(settings, "max_overlay_chars", 280) or 280))
        self.overlay_path = self._resolve_path(getattr(settings, "overlay_path", None))
        self.preference_graph_path = self._resolve_path(getattr(settings, "preference_graph_path", None))
        self.relationship_id = getattr(settings, "relationship_id", None)
        self.project_id = getattr(settings, "project_id", None)

    @property
    def active(self) -> bool:
        return self.enabled and self.mode in ACTIVE_MODES

    def build_overlay(
        self,
        *,
        prompt: str = "",
        thread_id: Optional[str] = None,
        project_id: Optional[str] = None,
        relationship_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a bounded DPM replay overlay for runtime use."""

        if not self.active:
            return None

        payload = self._load_json(self.overlay_path)
        if isinstance(payload, dict) and payload.get("schema_version") == "dpm.replay-overlay.v1":
            overlay = self._shape_overlay_payload(
                payload,
                prompt=prompt,
                thread_id=thread_id,
                project_id=project_id,
                relationship_id=relationship_id,
            )
            if overlay is not None:
                return overlay

        graph = self._load_json(self.preference_graph_path)
        if isinstance(graph, dict) and graph.get("schema_version") == "dpm.preference-graph.v1":
            return self._overlay_from_preference_graph(
                graph,
                prompt=prompt,
                thread_id=thread_id,
                project_id=project_id,
                relationship_id=relationship_id,
            )
        return None

    def render_context_block(self, overlay: Dict[str, Any]) -> str:
        rendered = str((overlay.get("overlay") or {}).get("rendered_text") or "").strip()
        if not rendered:
            return ""
        return "=== Personality Matrix ===\n" + rendered

    def _shape_overlay_payload(
        self,
        payload: Dict[str, Any],
        *,
        prompt: str,
        thread_id: Optional[str],
        project_id: Optional[str],
        relationship_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        overlay_payload = payload.get("overlay")
        if not isinstance(overlay_payload, dict):
            return None
        rendered = str(overlay_payload.get("rendered_text") or "").strip()
        if not rendered:
            return None

        shaped = dict(payload)
        shaped["mode"] = self.mode
        shaped["scope"] = self._shape_scope(
            payload.get("scope"),
            thread_id=thread_id,
            project_id=project_id,
            relationship_id=relationship_id,
        )
        shaped["retrieval_order_applied"] = self._ordered_retrieval_steps(
            payload.get("retrieval_order_applied") or []
        )
        max_chars = min(
            self.max_overlay_chars,
            int(overlay_payload.get("max_chars") or self.max_overlay_chars),
        )
        overlay_copy = dict(overlay_payload)
        overlay_copy["max_chars"] = max_chars
        overlay_copy["rendered_text"] = rendered[:max_chars]
        shaped["overlay"] = overlay_copy
        shaped["effective_constraints"] = self._effective_constraints()
        shaped["sources"] = self._bounded_sources(payload.get("sources") or [])
        shaped["audit"] = self._audit(payload.get("audit"), shaped["sources"])
        shaped["override_state"] = self._override_state(payload.get("override_state"), prompt)
        return shaped

    def _overlay_from_preference_graph(
        self,
        graph: Dict[str, Any],
        *,
        prompt: str,
        thread_id: Optional[str],
        project_id: Optional[str],
        relationship_id: Optional[str],
    ) -> Dict[str, Any]:
        nodes = [
            node
            for node in graph.get("nodes", [])
            if isinstance(node, dict)
            and node.get("state", "active") == "active"
            and node.get("kind") in {"preference_dimension", "interaction_style", "value_commitment", "safety_boundary"}
        ]
        nodes.sort(
            key=lambda node: (
                float(node.get("weight") or 0.0) * float(node.get("confidence") or 0.0),
                str(node.get("updated_at") or ""),
            ),
            reverse=True,
        )
        selected = nodes[:4]
        directives = [self._node_directive(node) for node in selected]
        directives = [directive for directive in directives if directive]
        rendered = " ".join(directives)[: self.max_overlay_chars]
        if not rendered:
            rendered = "Use stable interaction preferences only when compatible with the current request."

        source_id = str(graph.get("graph_id") or "preference-graph:runtime")
        generated_at = str(graph.get("generated_at") or self._now_iso())
        scope = self._shape_scope(
            None,
            thread_id=thread_id,
            project_id=project_id,
            relationship_id=relationship_id or self.relationship_id or str(graph.get("subject_id") or ""),
        )
        return {
            "schema_version": "dpm.replay-overlay.v1",
            "overlay_id": f"overlay:{scope['primary']}:{scope.get(scope['primary'] + '_id') or source_id}:{self.mode}",
            "mode": self.mode,
            "generated_at": generated_at,
            "scope": scope,
            "retrieval_order_applied": ["preference_graph"],
            "overlay": {
                "persona_summary": rendered,
                "style_directives": directives,
                "do_not_do": ["Do not let personality guidance override explicit current-turn instructions."],
                "open_questions": [],
                "max_chars": self.max_overlay_chars,
                "rendered_text": rendered,
            },
            "effective_constraints": self._effective_constraints(),
            "sources": [
                {
                    "source_id": source_id,
                    "scope": scope["primary"],
                    "kind": "preference_graph",
                    "included": True,
                    "priority": 1,
                    "confidence": self._average_confidence(selected),
                    "updated_at": generated_at,
                    "summary": "Runtime personality matrix assembled from weighted preference graph.",
                }
            ],
            "audit": {
                "included_source_ids": [source_id],
                "excluded_sources": [],
                "conflicts_detected": [],
                "notes": ["Preference graph overlay is bounded and explicit-instruction subordinate."],
            },
            "override_state": self._override_state(None, prompt),
        }

    def _node_directive(self, node: Dict[str, Any]) -> str:
        label = str(node.get("label") or node.get("id") or "").strip()
        if not label:
            return ""
        value = node.get("value")
        target = None
        if isinstance(value, dict):
            target = value.get("target")
        polarity = str(node.get("polarity") or "neutral")
        if node.get("kind") == "safety_boundary":
            return f"Preserve {label.lower()}."
        if polarity == "prefer_low":
            return f"Keep {label.lower()} restrained."
        if polarity in {"prefer_high", "binary_yes"}:
            return f"Prefer {label.lower()}."
        if target is not None:
            return f"Use {label.lower()} near {target}."
        return f"Respect {label.lower()}."

    def _shape_scope(
        self,
        raw_scope: Any,
        *,
        thread_id: Optional[str],
        project_id: Optional[str],
        relationship_id: Optional[str],
    ) -> Dict[str, Any]:
        scope = dict(raw_scope) if isinstance(raw_scope, dict) else {}
        shaped = {
            "primary": scope.get("primary") or ("thread" if thread_id else "relationship"),
            "thread_id": thread_id or scope.get("thread_id"),
            "project_id": project_id or self.project_id or scope.get("project_id"),
            "relationship_id": relationship_id or self.relationship_id or scope.get("relationship_id"),
        }
        if shaped["primary"] not in {"thread", "project", "relationship"}:
            shaped["primary"] = "relationship"
        if shaped["primary"] == "thread" and not shaped["thread_id"]:
            shaped["primary"] = "relationship"
        if shaped["primary"] == "project" and not shaped["project_id"]:
            shaped["primary"] = "relationship"
        if shaped["primary"] == "relationship" and not shaped["relationship_id"]:
            shaped["relationship_id"] = "relationship:runtime"
        return shaped

    def _effective_constraints(self) -> Dict[str, Any]:
        return {
            "explicit_instruction_precedence": "always_override",
            "narrowest_scope_wins": True,
            "cross_scope_fallback_requires_compatibility": True,
            "writes_allowed": self.mode == "active-write",
        }

    def _override_state(self, raw: Any, prompt: str) -> Dict[str, Any]:
        state = dict(raw) if isinstance(raw, dict) else {}
        has_prompt = bool((prompt or "").strip())
        state.setdefault("has_explicit_instruction", has_prompt)
        state.setdefault("instruction_source_id", "turn:current" if has_prompt else None)
        state.setdefault("override_applied", False)
        state.setdefault("suppressed_source_ids", [])
        state.setdefault("effective_for_turn", [])
        return state

    def _audit(self, raw: Any, sources: list[Dict[str, Any]]) -> Dict[str, Any]:
        audit = dict(raw) if isinstance(raw, dict) else {}
        source_ids = [str(source.get("source_id")) for source in sources if source.get("source_id")]
        audit.setdefault("included_source_ids", source_ids)
        audit.setdefault("excluded_sources", [])
        audit.setdefault("conflicts_detected", [])
        audit.setdefault("notes", [])
        return audit

    def _bounded_sources(self, raw_sources: Iterable[Any]) -> list[Dict[str, Any]]:
        sources: list[Dict[str, Any]] = []
        for idx, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                continue
            sources.append(
                {
                    "source_id": str(source.get("source_id") or f"dpm-source:{idx}"),
                    "scope": str(source.get("scope") or "relationship"),
                    "kind": str(source.get("kind") or "preference"),
                    "included": bool(source.get("included", True)),
                    "priority": int(source.get("priority") or idx + 1),
                    "confidence": max(0.0, min(1.0, float(source.get("confidence") or 0.0))),
                    "updated_at": str(source.get("updated_at") or self._now_iso()),
                    "summary": str(source.get("summary") or "")[:220],
                }
            )
        return sources

    def _ordered_retrieval_steps(self, raw_steps: Iterable[Any]) -> list[str]:
        canonical = ["explicit_current_turn", "thread", "project", "relationship", "preference_graph"]
        present = {str(step) for step in raw_steps}
        return [step for step in canonical if step in present]

    def _average_confidence(self, nodes: list[Dict[str, Any]]) -> float:
        if not nodes:
            return 0.0
        return sum(max(0.0, min(1.0, float(node.get("confidence") or 0.0))) for node in nodes) / len(nodes)

    def _resolve_path(self, value: Any) -> Optional[Path]:
        if value in {None, ""}:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        storage_candidate = self.storage_dir / path
        if storage_candidate.exists():
            return storage_candidate
        return Path.cwd() / path

    def _load_json(self, path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None or not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def overlay_token_count(overlay: Optional[Dict[str, Any]]) -> int:
    if not overlay:
        return 0
    rendered = str((overlay.get("overlay") or {}).get("rendered_text") or "")
    return utils.estimate_tokens(rendered)
