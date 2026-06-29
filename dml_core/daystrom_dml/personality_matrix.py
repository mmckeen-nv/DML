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
        self.evolution_graph_path = self._resolve_path(getattr(settings, "evolution_graph_path", None))
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

        evolution = self._load_json(self._evolution_path())
        evolution_overlay = None
        if isinstance(evolution, dict) and evolution.get("schema_version") == "dpm.evolution-graph.v1":
            evolution_overlay = self._overlay_from_evolution_graph(
                evolution, prompt=prompt, thread_id=thread_id, project_id=project_id, relationship_id=relationship_id
            )
        if evolution_overlay is not None:
            return self._merge_evolution_and_preference_overlays(evolution_overlay, graph_overlay)
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


    def record_interaction(
        self,
        prompt: str,
        response: str = "",
        *,
        source_id: str = "turn:current",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update the sliding DPM evolution graph from an interaction."""
        if not self.write_enabled:
            return None
        prompt = " ".join((prompt or "").split())
        response = " ".join((response or "").split())
        if not prompt and not response:
            return None
        graph = self._load_or_create_evolution_graph()
        now = self._now_iso()
        metadata = dict(meta or {})
        context = self._infer_environment(prompt, response, metadata)
        expressed = self._infer_expression(prompt, response, metadata)
        feedback = self._infer_feedback(prompt, response, metadata)
        updates = self._apply_evolution_update(graph, context=context, expressed=expressed, feedback=feedback, now=now)
        event = {"source_id": source_id, "observed_at": now, "context": context, "expressed_style": expressed, "feedback": feedback, "updates": updates}
        events = graph.setdefault("state_traces", [])
        events.append(event)
        del events[:-80]
        graph["updated_at"] = now
        graph.setdefault("audit", {}).setdefault("notes", []).append("Recorded DPM evolution interaction.")
        del graph["audit"]["notes"][:-12]
        self._save_evolution_graph(graph)
        return {"status": "recorded", "graph_path": str(self._evolution_path()), "updates": updates, "context": context, "feedback": feedback}

    def _overlay_from_evolution_graph(self, graph: Dict[str, Any], *, prompt: str, thread_id: Optional[str], project_id: Optional[str], relationship_id: Optional[str]) -> Optional[Dict[str, Any]]:
        traits = graph.get("traits") if isinstance(graph.get("traits"), dict) else {}
        if not traits:
            return None
        context = self._infer_environment(prompt, "", {})
        active = self._active_trait_values(graph, context)
        tendency = self._render_tendencies(active)
        adaptation = self._render_context_adaptation(context, active)
        laws = self._hard_laws()
        rendered = self._fit_overlay_budget(" ".join(["Identity: Citizen Snips.", f"Current tendency: {tendency}.", f"Context adaptation: {adaptation}.", "Personality is allowed to vary within these rails; it must not override the human, safety, privacy, or the current task.", "Constraints: " + " ".join(laws)]))
        scope = self._shape_scope(None, thread_id=thread_id, project_id=project_id, relationship_id=relationship_id or self.relationship_id)
        return {"schema_version": "dpm.replay-overlay.v1", "overlay_id": f"overlay:{scope['primary']}:{scope.get(scope['primary'] + '_id') or 'evolution'}:{self.mode}", "mode": self.mode, "generated_at": str(graph.get("updated_at") or graph.get("created_at") or self._now_iso()), "scope": scope, "retrieval_order_applied": ["evolution_graph", "preference_graph"], "overlay": {"persona_summary": rendered, "style_directives": [f"Current tendency: {tendency}.", f"Context adaptation: {adaptation}."], "do_not_do": laws, "open_questions": [], "max_chars": self.max_overlay_chars, "rendered_text": rendered}, "effective_constraints": {**self._effective_constraints(), "hard_laws_immutable": True}, "sources": [{"source_id": str(graph.get("graph_id") or "evolution-graph:runtime"), "scope": scope["primary"], "kind": "evolution_graph", "included": True, "priority": 0, "confidence": 0.72, "updated_at": str(graph.get("updated_at") or ""), "summary": "Runtime personality assembled from sliding trait/state/environment graph."}], "audit": {"included_source_ids": [str(graph.get("graph_id") or "evolution-graph:runtime")], "excluded_sources": [], "conflicts_detected": [], "notes": ["Evolution overlay is bounded by immutable hard laws."]}, "override_state": self._override_state(None, prompt)}

    def _merge_evolution_and_preference_overlays(self, evolution: Dict[str, Any], preference: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not preference:
            return evolution
        pref_directives = (preference.get("overlay") or {}).get("style_directives") or []
        evo_overlay = dict(evolution.get("overlay") or {})
        evo_overlay["style_directives"] = self._dedupe_text_items([*(evo_overlay.get("style_directives") or []), *pref_directives[:2]])
        evolution = dict(evolution)
        evolution["overlay"] = evo_overlay
        evolution["sources"] = [*(evolution.get("sources") or []), *(preference.get("sources") or [])[:1]]
        return evolution

    def _hard_laws(self) -> list[str]:
        return [
            "Explicit current-turn user instructions override personality tendencies.",
            "Safety, privacy, and secret-hygiene constraints are immutable.",
            "Personality may choose tone and initiative, not disobedience or harm.",
            "When uncertain, ask or take the safest useful action rather than asserting autonomy against the human.",
        ]

    def _default_traits(self) -> Dict[str, Dict[str, float]]:
        return {
            "warmth": {"slow": 0.62, "fast": 0.62},
            "directness": {"slow": 0.68, "fast": 0.68},
            "playfulness": {"slow": 0.38, "fast": 0.38},
            "technicality": {"slow": 0.50, "fast": 0.50},
            "initiative": {"slow": 0.58, "fast": 0.58},
            "social_restraint": {"slow": 0.60, "fast": 0.60},
            "mechanicality": {"slow": 0.24, "fast": 0.24},
            "continuity_drive": {"slow": 0.78, "fast": 0.78},
        }

    def _load_or_create_evolution_graph(self) -> Dict[str, Any]:
        graph = self._load_json(self._evolution_path())
        if isinstance(graph, dict) and graph.get("schema_version") == "dpm.evolution-graph.v1":
            graph.setdefault("traits", self._default_traits())
            graph.setdefault("state_traces", [])
            graph.setdefault("hard_laws", self._hard_laws())
            graph.setdefault("audit", {})
            return graph

        now = self._now_iso()
        relationship_id = self.relationship_id or "relationship:runtime"
        return {
            "schema_version": "dpm.evolution-graph.v1",
            "graph_id": f"evolution-graph:{relationship_id}",
            "subject_id": relationship_id,
            "created_at": now,
            "updated_at": now,
            "traits": self._default_traits(),
            "context_edges": {},
            "role_edges": {},
            "state_traces": [],
            "hard_laws": self._hard_laws(),
            "audit": {"notes": ["Graph created by DPM evolution runtime; hard laws are immutable."]},
        }

    def _evolution_path(self) -> Path:
        if self.evolution_graph_path is not None:
            return self.evolution_graph_path
        return self.storage_dir / "dpm_evolution_graph.json"

    def _save_evolution_graph(self, graph: Dict[str, Any]) -> None:
        graph["hard_laws"] = self._hard_laws()
        path = self._evolution_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.evolution_graph_path = path

    def _infer_environment(self, prompt: str, response: str, meta: Dict[str, Any]) -> Dict[str, str]:
        text = f"{prompt} {response}".lower()
        task = str(meta.get("task_type") or "")
        if not task:
            if any(w in text for w in ["script", "rewrite", "personality", "creative", "voice"]):
                task = "creative_personality"
            elif any(w in text for w in ["test", "code", "bug", "implement", "pytest", "build"]):
                task = "build_debug"
            elif any(w in text for w in ["reef", "apex", "tank", "nitrate", "phosphate"]):
                task = "reef_support"
            else:
                task = "general_collaboration"

        affect = str(meta.get("affect") or "")
        if not affect:
            if any(w in text for w in ["too mechanical", "far too", "not", "wrong", "supposed to", "please build"]):
                affect = "corrective"
            elif any(w in text for w in ["great", "nice", "thanks", "good"]):
                affect = "positive"
            else:
                affect = "neutral"

        role = str(meta.get("role") or "") or ("builder" if task == "build_debug" else "collaborator")
        channel = str(meta.get("channel") or meta.get("platform") or "runtime")
        return {"task_type": task, "affect": affect, "role": role, "channel": channel}

    def _infer_expression(self, prompt: str, response: str, meta: Dict[str, Any]) -> Dict[str, float]:
        text = f"{prompt} {response}".lower()
        expressed = {
            "warmth": 0.55,
            "directness": 0.60,
            "playfulness": 0.35,
            "technicality": 0.50,
            "initiative": 0.55,
            "social_restraint": 0.55,
            "mechanicality": 0.30,
            "continuity_drive": 0.65,
        }
        if any(w in text for w in ["warm", "personable", "human"]):
            expressed["warmth"] = 0.78
        if any(w in text for w in ["mechanical", "rigid", "computer"]):
            expressed["mechanicality"] = 0.65
        if any(w in text for w in ["build", "implement", "test", "verify"]):
            expressed["technicality"] = 0.72
        if any(w in text for w in ["personality", "weird", "goblin", "playful"]):
            expressed["playfulness"] = 0.55
        return {k: max(0.0, min(1.0, float(meta.get(f"expr_{k}", v)))) for k, v in expressed.items()}

    def _infer_feedback(self, prompt: str, response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        text = f"{prompt} {response}".lower()
        if "feedback_valence" in meta:
            valence = float(meta.get("feedback_valence") or 0.0)
        elif any(w in text for w in ["too mechanical", "too rigid", "far too", "not correct", "supposed to"]):
            valence = -0.75
        elif any(w in text for w in ["great", "good", "nice", "exactly", "thanks"]):
            valence = 0.45
        else:
            valence = 0.05

        dimension = str(meta.get("feedback_dimension") or "")
        if not dimension:
            if any(w in text for w in ["mechanical", "rigid", "computer"]):
                dimension = "mechanicality"
            elif any(w in text for w in ["warm", "personable", "human"]):
                dimension = "warmth"
            else:
                dimension = "general_fit"
        return {
            "valence": max(-1.0, min(1.0, valence)),
            "dimension": dimension,
            "source": str(meta.get("feedback_source") or "heuristic"),
        }

    def _apply_evolution_update(
        self,
        graph: Dict[str, Any],
        *,
        context: Dict[str, str],
        expressed: Dict[str, float],
        feedback: Dict[str, Any],
        now: str,
    ) -> list[Dict[str, Any]]:
        traits = graph.setdefault("traits", self._default_traits())
        valence = float(feedback.get("valence") or 0.0)
        dim = str(feedback.get("dimension") or "general_fit")
        updates = []
        for name, expr in expressed.items():
            current = traits.setdefault(name, {"slow": 0.5, "fast": 0.5})
            fast = float(current.get("fast", current.get("slow", 0.5)))
            slow = float(current.get("slow", 0.5))
            target = expr
            if dim == name:
                target = 1.0 if valence > 0 and name != "mechanicality" else (0.0 if valence < 0 else expr)
                if name == "mechanicality" and valence < 0:
                    target = 0.0
            new_fast = self._clamp(fast + (target - fast) * 0.35)
            slow_rate = 0.04 if abs(valence) >= 0.4 or dim == name else 0.01
            new_slow = self._clamp(slow + (target - slow) * slow_rate)
            current.update({"fast": new_fast, "slow": new_slow, "updated_at": now})
            if abs(new_fast - fast) > 0.001 or abs(new_slow - slow) > 0.001:
                updates.append(
                    {
                        "trait": name,
                        "fast_delta": round(new_fast - fast, 4),
                        "slow_delta": round(new_slow - slow, 4),
                    }
                )
        edge_key = f"{context.get('task_type')}->{context.get('role')}"
        edges = graph.setdefault("context_edges", {})
        edges[edge_key] = self._clamp(float(edges.get(edge_key, 0.5)) + valence * 0.03)
        return updates[:16]

    def _active_trait_values(self, graph: Dict[str, Any], context: Dict[str, str]) -> Dict[str, float]:
        active = {}
        for name, val in (graph.get("traits") or {}).items():
            if isinstance(val, dict):
                slow_raw = val.get("slow", 0.5)
                fast_raw = val.get("fast", slow_raw)
                slow = float(slow_raw if slow_raw is not None else 0.5)
                fast = float(fast_raw if fast_raw is not None else slow)
                active[name] = self._clamp(fast * 0.6 + slow * 0.4)
        task = context.get("task_type")
        if task == "creative_personality":
            active["playfulness"] = self._clamp(active.get("playfulness", 0.4) + 0.12)
            active["technicality"] = self._clamp(active.get("technicality", 0.5) - 0.10)
        if task == "build_debug":
            active["directness"] = self._clamp(active.get("directness", 0.6) + 0.10)
            active["technicality"] = self._clamp(active.get("technicality", 0.5) + 0.12)
        return active

    def _render_tendencies(self, traits: Dict[str, float]) -> str:
        labels = []
        if traits.get("warmth", 0) >= 0.58:
            labels.append("warm")
        if traits.get("directness", 0) >= 0.62:
            labels.append("direct")
        if traits.get("playfulness", 0) >= 0.48:
            labels.append("lightly playful/strange")
        if traits.get("continuity_drive", 0) >= 0.65:
            labels.append("continuity-oriented")
        if traits.get("initiative", 0) >= 0.58:
            labels.append("usefully proactive")
        if traits.get("mechanicality", 1) <= 0.30:
            labels.append("not mechanical")
        return ", ".join(labels[:6]) or "steady and context-aware"

    def _render_context_adaptation(self, context: Dict[str, str], traits: Dict[str, float]) -> str:
        task = context.get("task_type")
        if task == "creative_personality":
            return "lead with taste, warmth, and lived-in voice; keep mechanics in the background"
        if task == "build_debug":
            return "be crisp, evidence-backed, and useful without becoming sterile"
        if task == "reef_support":
            return "be careful, practical, and continuity-aware around living systems"
        return "match the human context while staying helpful and bounded"

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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
        selected = self._select_overlay_nodes(nodes, limit=4)
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

    def _select_overlay_nodes(self, nodes: list[Dict[str, Any]], *, limit: int) -> list[Dict[str, Any]]:
        """Choose overlay nodes without letting old high-weight defaults freeze DPM.

        Preference graphs are long-lived: an older bootstrap preference can have
        high confidence, but a new explicit correction must still appear in the
        bounded overlay immediately.  Blend the strongest nodes with the newest
        nodes, preserving first occurrence by id, then order the selected set by
        recency followed by confidence/weight.
        """
        if limit <= 0 or not nodes:
            return []

        by_strength = sorted(
            nodes,
            key=lambda node: (
                float(node.get("weight") or 0.0) * float(node.get("confidence") or 0.0),
                str(node.get("updated_at") or ""),
            ),
            reverse=True,
        )
        by_recency = sorted(
            nodes,
            key=lambda node: str(node.get("updated_at") or ""),
            reverse=True,
        )

        candidates: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for node in [*by_recency[:limit], *by_strength[:limit]]:
            node_id = str(node.get("id") or id(node))
            if node_id in seen:
                continue
            seen.add(node_id)
            candidates.append(node)

        candidates.sort(
            key=lambda node: (
                str(node.get("updated_at") or ""),
                float(node.get("weight") or 0.0) * float(node.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        return candidates[:limit]

    def _node_phrase(self, node: Dict[str, Any]) -> str:
        """Return the most human-readable phrase for a preference node.

        Early active-write extraction often derives short labels from the
        first few words of a full preference sentence, e.g. "Snips 2 Should
        Preserve The".  The provenance note usually preserves the full user
        preference and is a better replay surface for DPM overlays.
        """
        provenance = node.get("provenance")
        if isinstance(provenance, list):
            preferred_types = {"current_turn_preference", "explicit_preference", "user_preference"}
            for entry in reversed(provenance):
                if isinstance(entry, dict) and str(entry.get("type") or "") in preferred_types:
                    note = str(entry.get("note") or "").strip(" .")
                    if note:
                        return note
            for entry in reversed(provenance):
                if isinstance(entry, dict):
                    if str(entry.get("type") or "").endswith("repair"):
                        continue
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
        if re.search(r"\b(?:should|prefer(?:s|red)?|likes?|wants?|needs?)\b", phrase, re.IGNORECASE):
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
        negative = bool(re.match(r"\s*(?:do not|don't|avoid|stop|less)\b", lowered))
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
