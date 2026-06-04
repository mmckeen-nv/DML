"""Runtime support for the Daystrom Personality Matrix overlay."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import utils


ACTIVE_MODES = {"active-read", "active-write"}
VALID_MODES = {"disabled", "observe-only", "active-read", "active-write"}
PREFERENCE_PATTERNS = (
    re.compile(r"\b(?:i|we)\s+(?:prefer|like|want|need)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bplease\s+(?:prefer|use|keep|make|be)\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:always|usually)\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:do not|don't)\s+(.+)", re.IGNORECASE),
)


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
        self.token_budget = max(1, int(getattr(settings, "token_budget", 80) or 80))
        self.overlay_path = self._resolve_path(getattr(settings, "overlay_path", None))
        self.preference_graph_path = self._resolve_path(getattr(settings, "preference_graph_path", None))
        self.relationship_id = getattr(settings, "relationship_id", None)
        self.project_id = getattr(settings, "project_id", None)

    @property
    def active(self) -> bool:
        return self.enabled and self.mode in ACTIVE_MODES

    @property
    def write_enabled(self) -> bool:
        return self.enabled and self.mode == "active-write"

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
        graph = self._load_json(self.preference_graph_path)
        graph_overlay = None
        if isinstance(graph, dict) and graph.get("schema_version") == "dpm.preference-graph.v1" and self._graph_has_active_nodes(graph):
            graph_overlay = self._overlay_from_preference_graph(
                graph,
                prompt=prompt,
                thread_id=thread_id,
                project_id=project_id,
                relationship_id=relationship_id,
            )

        if isinstance(payload, dict) and payload.get("schema_version") == "dpm.replay-overlay.v1":
            overlay = self._shape_overlay_payload(
                payload,
                prompt=prompt,
                thread_id=thread_id,
                project_id=project_id,
                relationship_id=relationship_id,
            )
            if graph_overlay is not None and self._graph_is_at_least_as_fresh(graph, payload):
                return graph_overlay
            if overlay is not None:
                return overlay

        if graph_overlay is not None:
            return graph_overlay
        return None

    def render_context_block(self, overlay: Dict[str, Any]) -> str:
        rendered = str((overlay.get("overlay") or {}).get("rendered_text") or "").strip()
        if not rendered:
            return ""
        return "=== Personality Matrix ===\n" + rendered

    def graph(self) -> Optional[Dict[str, Any]]:
        """Return the current preference graph, if present."""

        return self._load_json(self._graph_path())

    def record_preference(
        self,
        text: str,
        *,
        scope: str = "relationship",
        source_id: str = "turn:current",
        explicit: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist an explicit preference signal into the DPM graph.

        Active-write mode is required. When ``explicit`` is false, only clear
        preference-shaped text is recorded.
        """

        if not self.write_enabled:
            return None
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None
        signal = self._extract_preference_signal(cleaned, explicit=explicit)
        if signal is None:
            return None

        graph = self._load_or_create_graph()
        now = self._now_iso()
        node_id = self._preference_node_id(signal["label"])
        nodes = graph.setdefault("nodes", [])
        node = self._find_node(nodes, node_id)
        if node is None:
            node = self._new_preference_node(node_id, signal, scope=scope, now=now)
            nodes.append(node)
        self._reinforce_node(node, signal, source_id=source_id, now=now, meta=meta)

        graph["generated_at"] = now
        audit = graph.setdefault("audit", {})
        audit["source_count"] = int(audit.get("source_count") or 0) + 1
        included = audit.setdefault("included_sources", [])
        if source_id not in included:
            included.append(source_id)
        audit.setdefault("excluded_sources", [])
        audit.setdefault("conflicts_detected", [])
        notes = audit.setdefault("notes", [])
        notes.append(f"Recorded active-write preference signal for {node_id}.")
        del notes[:-8]

        self._save_graph(graph)
        return {"status": "recorded", "node_id": node_id, "graph_path": str(self._graph_path())}

    def suppress_preference(self, node_id: str, *, reason: str = "suppressed_by_user") -> Optional[Dict[str, Any]]:
        if not self.write_enabled:
            return None
        graph = self._load_json(self._graph_path())
        if not isinstance(graph, dict):
            return {"status": "missing", "node_id": node_id}
        node = self._find_node(graph.get("nodes", []), node_id)
        if node is None:
            return {"status": "missing", "node_id": node_id}
        now = self._now_iso()
        node["state"] = "suppressed"
        node["updated_at"] = now
        constraints = node.setdefault("constraints", {})
        constraints["suppression_reason"] = reason
        audit = graph.setdefault("audit", {})
        audit.setdefault("excluded_sources", []).append({"source_id": node_id, "reason": reason})
        audit.setdefault("notes", []).append(f"Suppressed preference node {node_id}.")
        graph["generated_at"] = now
        self._save_graph(graph)
        return {"status": "suppressed", "node_id": node_id, "graph_path": str(self._graph_path())}

    def delete_preference(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not self.write_enabled:
            return None
        graph = self._load_json(self._graph_path())
        if not isinstance(graph, dict):
            return {"status": "missing", "node_id": node_id}
        nodes = [node for node in graph.get("nodes", []) if not (isinstance(node, dict) and node.get("id") == node_id)]
        removed = len(nodes) != len(graph.get("nodes", []))
        graph["nodes"] = nodes
        graph["edges"] = [
            edge for edge in graph.get("edges", [])
            if not (isinstance(edge, dict) and (edge.get("from") == node_id or edge.get("to") == node_id))
        ]
        if not removed:
            return {"status": "missing", "node_id": node_id}
        graph["generated_at"] = self._now_iso()
        graph.setdefault("audit", {}).setdefault("notes", []).append(f"Deleted preference node {node_id}.")
        self._save_graph(graph)
        return {"status": "deleted", "node_id": node_id, "graph_path": str(self._graph_path())}

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
        overlay_copy["style_directives"] = self._dedupe_text_items(overlay_copy.get("style_directives") or [])
        overlay_copy["do_not_do"] = self._dedupe_text_items(
            [
                *(overlay_copy.get("do_not_do") or []),
                "Current-turn instructions override the DPM overlay.",
            ]
        )
        overlay_copy["rendered_text"] = self._fit_overlay_budget(self._dedupe_rendered_text(rendered), char_limit=max_chars)
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
        directives = self._dedupe_text_items(self._node_directive(node) for node in selected)
        constraints = ["Current-turn instructions override the DPM overlay."]
        rendered = self._fit_overlay_budget(self._structured_overlay_text(directives, constraints))
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
                "do_not_do": constraints,
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

    def _node_phrase(self, node: Dict[str, Any]) -> str:
        """Return the most human-readable phrase for a preference node.

        Early active-write extraction often derives short labels from the
        first few words of a full preference sentence, e.g. "Snips 2 Should
        Preserve The".  The provenance note usually preserves the full user
        preference and is a better replay surface for DPM overlays.
        """
        provenance = node.get("provenance")
        if isinstance(provenance, list):
            for entry in reversed(provenance):
                if isinstance(entry, dict):
                    note = str(entry.get("note") or "").strip(" .")
                    if note:
                        return note
        return str(node.get("label") or node.get("id") or "").strip(" .")

    def _node_directive(self, node: Dict[str, Any]) -> str:
        phrase = self._node_phrase(node)
        if not phrase:
            return ""
        value = node.get("value")
        target = None
        if isinstance(value, dict):
            target = value.get("target")
        polarity = str(node.get("polarity") or "neutral")
        rendered = phrase[:1].upper() + phrase[1:]
        if re.search(r"\bshould\b", phrase, re.IGNORECASE):
            return f"{rendered}."
        label = phrase.lower()
        if node.get("kind") == "safety_boundary":
            return f"Preserve {label}."
        if polarity == "prefer_low":
            return f"Keep {label} restrained."
        if polarity in {"prefer_high", "binary_yes"}:
            return f"Prefer {label}."
        if target is not None:
            return f"Use {label} near {target}."
        return f"Respect {label}."

    def _fit_overlay_budget(self, text: str, *, char_limit: Optional[int] = None) -> str:
        overlay_chars = self.max_overlay_chars if char_limit is None else min(self.max_overlay_chars, max(1, int(char_limit)))
        chars = max(1, min(overlay_chars, self.token_budget * 4))
        rendered = self._cut_at_boundary(text or "", chars)
        while utils.estimate_tokens(rendered) > self.token_budget and len(rendered) > 8:
            rendered = self._cut_at_boundary(rendered, max(8, int(len(rendered) * 0.85)))
        return rendered

    def _cut_at_boundary(self, text: str, chars: int) -> str:
        text = (text or "").strip()
        if len(text) <= chars:
            return text
        cut = text[: max(1, chars)].rstrip()
        sentence_cut = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if sentence_cut >= max(24, int(chars * 0.5)):
            return cut[: sentence_cut + 1].rstrip()
        word_cut = cut.rfind(" ")
        if word_cut >= max(12, int(chars * 0.4)):
            return cut[:word_cut].rstrip()
        return cut.rstrip(" .,;:-")

    def _dedupe_text_items(self, items: Iterable[Any]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = re.sub(r"\W+", " ", text).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _dedupe_rendered_text(self, text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        deduped = self._dedupe_text_items(parts)
        return " ".join(deduped)

    def _structured_overlay_text(self, directives: list[str], constraints: list[str]) -> str:
        parts = ["Identity: Citizen Snips."]
        if directives:
            parts.append("Preferences: " + " ".join(directives))
            parts.append("Style: Follow stable preferences when they fit the request.")
        parts.append("Constraints: " + " ".join(constraints))
        return " ".join(parts)

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

    def _extract_preference_signal(self, text: str, *, explicit: bool) -> Optional[Dict[str, Any]]:
        lowered = text.lower()
        phrase = ""
        for pattern in PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                phrase = match.group(1)
                break
        if explicit and not phrase:
            phrase = text
        if not explicit and not phrase:
            return None
        phrase = self._normalize_preference_phrase(phrase)
        if not phrase:
            return None
        negative = "do not" in lowered or "don't" in lowered or "avoid" in lowered
        label = self._label_from_phrase(phrase)
        return {
            "label": label,
            "phrase": phrase[:220],
            "polarity": "prefer_low" if negative else "prefer_high",
            "target": 0.15 if negative else 0.85,
        }

    def _normalize_preference_phrase(self, phrase: str) -> str:
        phrase = (phrase or "").strip(" .,:;!?")
        phrase = re.sub(r"^(?:prefer|use|keep|make|be)\s+", "", phrase, flags=re.IGNORECASE).strip(" .,:;!?")
        return phrase

    def _graph_has_active_nodes(self, graph: Dict[str, Any]) -> bool:
        return any(
            isinstance(node, dict)
            and node.get("state", "active") == "active"
            and node.get("kind") in {"preference_dimension", "interaction_style", "value_commitment", "safety_boundary"}
            for node in graph.get("nodes", [])
        )

    def _graph_is_at_least_as_fresh(self, graph: Optional[Dict[str, Any]], overlay: Dict[str, Any]) -> bool:
        if not isinstance(graph, dict):
            return False
        graph_ts = str(graph.get("generated_at") or "")
        overlay_ts = str(overlay.get("generated_at") or "")
        if not graph_ts:
            return False
        if not overlay_ts:
            return True
        return graph_ts >= overlay_ts

    def _label_from_phrase(self, phrase: str) -> str:
        words = re.findall(r"[a-z0-9]+", phrase.lower())[:5]
        if not words:
            return "Preference"
        return " ".join(words).title()

    def _preference_node_id(self, label: str) -> str:
        slug = "-".join(re.findall(r"[a-z0-9]+", label.lower()))[:80]
        return f"pref.{slug or 'preference'}"

    def _find_node(self, nodes: list[Any], node_id: str) -> Optional[Dict[str, Any]]:
        for node in nodes:
            if isinstance(node, dict) and node.get("id") == node_id:
                return node
        return None

    def _new_preference_node(
        self,
        node_id: str,
        signal: Dict[str, Any],
        *,
        scope: str,
        now: str,
    ) -> Dict[str, Any]:
        normalized_scope = scope if scope in {"thread", "project", "relationship", "global"} else "relationship"
        return {
            "id": node_id,
            "kind": "interaction_style",
            "label": signal["label"],
            "scope": normalized_scope,
            "state": "active",
            "weight": 0.55,
            "confidence": 0.55,
            "polarity": signal["polarity"],
            "value_type": "scalar",
            "value": {"target": signal["target"], "allowed_range": [0.0, 1.0]},
            "evidence": {
                "support_count": 0,
                "contradiction_count": 0,
                "last_supported_at": now,
                "last_contradicted_at": None,
            },
            "provenance": [],
            "constraints": {
                "overridden_by_explicit_instruction": True,
                "ttl_days": 120,
                "requires_review_if_confidence_below": 0.55,
            },
            "updated_at": now,
        }

    def _reinforce_node(
        self,
        node: Dict[str, Any],
        signal: Dict[str, Any],
        *,
        source_id: str,
        now: str,
        meta: Optional[Dict[str, Any]],
    ) -> None:
        evidence = node.setdefault("evidence", {})
        same_polarity = str(node.get("polarity") or "") == signal["polarity"]
        if same_polarity:
            evidence["support_count"] = int(evidence.get("support_count") or 0) + 1
            evidence["last_supported_at"] = now
            node["weight"] = min(1.0, float(node.get("weight") or 0.5) + 0.05)
            node["confidence"] = min(1.0, float(node.get("confidence") or 0.5) + 0.04)
        else:
            evidence["contradiction_count"] = int(evidence.get("contradiction_count") or 0) + 1
            evidence["last_contradicted_at"] = now
            node["state"] = "conflicted"
            node["confidence"] = max(0.0, float(node.get("confidence") or 0.5) - 0.08)

        provenance = node.setdefault("provenance", [])
        provenance.append(
            {
                "type": "current_turn_preference",
                "source_id": source_id,
                "observed_at": now,
                "note": signal["phrase"][:220],
                "meta": dict(meta or {}),
            }
        )
        del provenance[:-8]
        node["updated_at"] = now

    def _load_or_create_graph(self) -> Dict[str, Any]:
        graph = self._load_json(self._graph_path())
        if isinstance(graph, dict) and graph.get("schema_version") == "dpm.preference-graph.v1":
            graph.setdefault("nodes", [])
            graph.setdefault("edges", [])
            graph.setdefault("audit", {})
            return graph
        now = self._now_iso()
        relationship_id = self.relationship_id or "relationship:runtime"
        return {
            "schema_version": "dpm.preference-graph.v1",
            "graph_id": f"preference-graph:{relationship_id}",
            "subject_id": relationship_id,
            "generated_at": now,
            "default_policy": {
                "explicit_instruction_precedence": "always_override",
                "conflict_mode": "preserve_and_audit",
                "decay_policy": "recency_weighted",
            },
            "nodes": [],
            "edges": [],
            "audit": {
                "source_count": 0,
                "included_sources": [],
                "excluded_sources": [],
                "conflicts_detected": [],
                "notes": ["Graph created by DPM active-write runtime."],
            },
        }

    def _graph_path(self) -> Path:
        if self.preference_graph_path is not None:
            return self.preference_graph_path
        return self.storage_dir / "dpm_preference_graph.json"

    def _save_graph(self, graph: Dict[str, Any]) -> None:
        path = self._graph_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.preference_graph_path = path

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
