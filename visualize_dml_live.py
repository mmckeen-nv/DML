"""Streamlit application for live Daystrom Memory Lattice visualisation."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from daystrom_dml import utils, visualizer_bridge
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.memory_store import MemoryItem

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MODE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "literal": (63, 136, 245),  # bright blue
    "semantic": (46, 160, 67),  # vivid green
    "hybrid": (139, 94, 205),  # deep purple
}
DEFAULT_MODE = "literal"
FRAME_DELAY = 0.35
MAX_TOP_K = 8


@dataclass
class StepResult:
    """Container used by the streaming generator."""

    item: MemoryItem
    score: float
    rank: int
    summary: str
    tokens: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _rgb_to_plotly(color: Tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _mix_with_white(color: np.ndarray, intensity: float) -> np.ndarray:
    """Blend an RGB colour with white by ``intensity`` (0-1)."""

    intensity = float(max(0.0, min(1.0, intensity)))
    return color + (255.0 - color) * intensity


def _compute_node_colour(
    base_rgb: Tuple[int, int, int], fidelity: float, activation: float
) -> str:
    base = np.array(base_rgb, dtype=float)
    # Fidelity controls brightness (decay reduces fidelity and therefore brightness)
    brightness = 0.25 + 0.75 * max(0.0, min(1.0, fidelity))
    colour = base * brightness
    if activation > 0:
        colour = _mix_with_white(colour, min(1.0, activation * 0.7))
    colour = np.clip(colour, 0.0, 255.0)
    return _rgb_to_plotly(tuple(int(round(c)) for c in colour))


def _compute_position(item: MemoryItem) -> Tuple[float, float, float]:
    """Generate a deterministic spiral layout for nodes."""

    golden_angle = math.pi * (3 - math.sqrt(5))
    theta = item.id * golden_angle
    radius = 2.5 + item.level * 2.5 + item.salience * 2.0
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    z = -item.level * 3.0 + item.fidelity
    return (x, y, z)


def _shorten_text(text: str, limit: int = 140) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _build_step_summary(adapter: DMLAdapter, item: MemoryItem) -> Tuple[str, int]:
    summary = adapter.summarizer.summarize(item.text, max_len=180)
    tokens = utils.estimate_tokens(summary)
    return summary, tokens


# ---------------------------------------------------------------------------
# Retrieval streaming helpers
# ---------------------------------------------------------------------------


def _deduplicate_by_id(
    entries: Iterable[Tuple[float, MemoryItem, Dict[str, float]]],
    top_k: int,
) -> List[Tuple[float, MemoryItem, Dict[str, float]]]:
    """Take the highest scoring entry per memory id."""

    best: Dict[int, Tuple[float, MemoryItem, Dict[str, float]]] = {}
    for score, item, payload in entries:
        current = best.get(item.id)
        if current is None or score > current[0]:
            best[item.id] = (score, item, payload)
    ordered = sorted(best.values(), key=lambda row: row[0], reverse=True)
    return ordered[:top_k]


def _literal_stream(
    adapter: DMLAdapter,
    prompt: str,
    items: List[MemoryItem],
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> Iterator[StepResult]:
    literal_results = adapter.literal_retriever.retrieve(
        prompt, items, query_embedding, top_k=top_k
    )
    for rank, result in enumerate(literal_results, start=1):
        summary, tokens = _build_step_summary(adapter, result.item)
        yield StepResult(
            item=result.item,
            score=float(result.literal_score),
            rank=rank,
            summary=summary,
            tokens=tokens,
        )


def _semantic_stream(
    adapter: DMLAdapter,
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> Iterator[StepResult]:
    now = time.time()
    retrieved = adapter.store.retrieve(query_embedding, top_k=top_k)
    for rank, item in enumerate(retrieved, start=1):
        score = adapter.store._score_item(item, query_embedding, now)  # type: ignore[attr-defined]
        summary, tokens = _build_step_summary(adapter, item)
        yield StepResult(
            item=item,
            score=float(score),
            rank=rank,
            summary=summary,
            tokens=tokens,
        )


def _hybrid_stream(
    adapter: DMLAdapter,
    prompt: str,
    items: List[MemoryItem],
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> Iterator[StepResult]:
    alpha = adapter._alpha_for_mode("hybrid")  # type: ignore[attr-defined]
    literal_results = adapter.literal_retriever.retrieve(
        prompt, items, query_embedding, top_k=top_k
    )
    semantic_items = adapter.store.retrieve(query_embedding, top_k=top_k)
    combined: List[Tuple[float, MemoryItem, Dict[str, float]]] = []
    for result in literal_results:
        final_score = alpha * result.semantic_score + (1 - alpha) * result.literal_score
        combined.append(
            (
                float(final_score),
                result.item,
                {"literal": float(result.literal_score), "semantic": float(result.semantic_score)},
            )
        )
    for item in semantic_items:
        similarity = utils.cosine_similarity(item.embedding, query_embedding)
        final_score = alpha * similarity + (1 - alpha) * 0.0
        combined.append(
            (
                float(final_score),
                item,
                {"literal": 0.0, "semantic": float(similarity)},
            )
        )
    ordered = _deduplicate_by_id(combined, top_k)
    for rank, (score, item, _) in enumerate(ordered, start=1):
        summary, tokens = _build_step_summary(adapter, item)
        yield StepResult(
            item=item,
            score=float(score),
            rank=rank,
            summary=summary,
            tokens=tokens,
        )


def retrieval_stream(
    adapter: DMLAdapter,
    prompt: str,
    mode: str,
    *,
    top_k: int,
) -> Iterator[StepResult]:
    """Yield retrieval steps for the visualiser."""

    items = list(adapter.store.items())
    if not prompt.strip() or not items:
        return iter(())
    query_embedding = adapter.embedder.embed(prompt)
    actual_mode = mode
    if actual_mode == "auto":
        actual_mode = adapter._classify_mode(prompt)  # type: ignore[attr-defined]
    actual_mode = actual_mode or DEFAULT_MODE
    if actual_mode == "literal":
        generator = _literal_stream(adapter, prompt, items, query_embedding, top_k=top_k)
    elif actual_mode == "semantic":
        generator = _semantic_stream(adapter, query_embedding, top_k=top_k)
    elif actual_mode == "hybrid":
        generator = _hybrid_stream(adapter, prompt, items, query_embedding, top_k=top_k)
    else:
        generator = _semantic_stream(adapter, query_embedding, top_k=top_k)
    return generator


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _refresh_graph_state(
    graph_state: Dict,
    items: Iterable[MemoryItem],
    *,
    mode: str,
    active_item_id: int | None = None,
) -> None:
    existing_ids = set(graph_state["nodes"].keys())
    updated_ids = set()
    for item in items:
        position = _compute_position(item)
        label = _shorten_text(item.text)
        node = graph_state["nodes"].get(item.id, {})
        activation = float(node.get("activation", 0.0)) * 0.6
        if active_item_id is not None and item.id == active_item_id:
            activation = 1.0
        graph_state["nodes"][item.id] = {
            "id": item.id,
            "position": position,
            "salience": float(item.salience),
            "fidelity": float(item.fidelity),
            "level": int(item.level),
            "summary_of": list(item.summary_of),
            "label": label,
            "merges": int(item.meta.get("merges", 0)) if item.meta else 0,
            "activation": activation,
            "last_score": float(node.get("last_score", 0.0)),
            "summary": node.get("summary", ""),
        }
        updated_ids.add(item.id)
    for node_id in existing_ids - updated_ids:
        graph_state["nodes"].pop(node_id, None)
    graph_state["mode"] = mode


def _update_active_node(graph_state: Dict, step: StepResult) -> None:
    node = graph_state["nodes"].get(step.item.id)
    if not node:
        return
    node["activation"] = 1.0
    node["last_score"] = float(step.score)
    node["summary"] = step.summary
    node["salience"] = float(step.item.salience)
    node["fidelity"] = float(step.item.fidelity)
    node["level"] = int(step.item.level)


def _build_edges(nodes: Dict[int, Dict]) -> Tuple[List[float], List[float], List[float]]:
    x_coords: List[float] = []
    y_coords: List[float] = []
    z_coords: List[float] = []
    for node in nodes.values():
        start = node["position"]
        for target_id in node["summary_of"]:
            target = nodes.get(target_id)
            if target is None:
                continue
            end = target["position"]
            x_coords.extend([start[0], end[0], None])
            y_coords.extend([start[1], end[1], None])
            z_coords.extend([start[2], end[2], None])
    return x_coords, y_coords, z_coords


def build_figure(graph_state: Dict) -> go.Figure:
    nodes = graph_state["nodes"]
    mode = graph_state.get("mode", DEFAULT_MODE)
    base_color = MODE_COLORS.get(mode, MODE_COLORS[DEFAULT_MODE])
    node_x: List[float] = []
    node_y: List[float] = []
    node_z: List[float] = []
    node_size: List[float] = []
    node_color: List[str] = []
    node_text: List[str] = []
    halo_x: List[float] = []
    halo_y: List[float] = []
    halo_z: List[float] = []
    halo_size: List[float] = []
    halo_color: List[str] = []
    for node in nodes.values():
        pos = node["position"]
        activation = node.get("activation", 0.0)
        salience = node.get("salience", 0.3)
        fidelity = node.get("fidelity", 0.2)
        base_size = 10.0 + salience * 26.0
        display_size = base_size * (1.0 + 0.45 * activation)
        colour = _compute_node_colour(base_color, fidelity, activation)
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_z.append(pos[2])
        node_size.append(display_size)
        node_color.append(colour)
        tooltip = (
            f"ID {node['id']}<br>Level {node['level']}<br>Salience {salience:.2f}"
            f"<br>Fidelity {fidelity:.2f}<br>Score {node['last_score']:.3f}"\
            f"<br>{_shorten_text(node.get('summary') or node['label'], 180)}"
        )
        node_text.append(tooltip)
        if activation > 0.25:
            halo_x.append(pos[0])
            halo_y.append(pos[1])
            halo_z.append(pos[2])
            halo_size.append(display_size * (1.5 + activation))
            halo_color.append(_compute_node_colour(base_color, fidelity, 1.0))
    edge_x, edge_y, edge_z = _build_edges(nodes)
    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="rgba(200,200,240,0.18)", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )
    if halo_x:
        fig.add_trace(
            go.Scatter3d(
                x=halo_x,
                y=halo_y,
                z=halo_z,
                mode="markers",
                marker=dict(
                    size=halo_size,
                    color=halo_color,
                    opacity=0.25,
                    symbol="circle",
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.95,
                line=dict(color="rgba(255,255,255,0.35)", width=1.2),
            ),
            hoverinfo="text",
            text=node_text,
            showlegend=False,
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
        ),
        paper_bgcolor="#0b0d17",
        plot_bgcolor="#0b0d17",
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


@st.cache_resource
def get_adapter() -> DMLAdapter:
    return DMLAdapter()


def main() -> None:
    st.set_page_config(page_title="Daystrom Memory Lattice Live", layout="wide")
    st.title("Daystrom Memory Lattice — Live Retrieval Visualiser")
    left_col, centre_col, right_col = st.columns([1.2, 2.8, 1.0], gap="large")

    if "graph_state" not in st.session_state:
        st.session_state["graph_state"] = {"mode": DEFAULT_MODE, "nodes": {}}

    adapter = get_adapter()

    query_params = st.experimental_get_query_params()
    prompt_param = query_params.get("prompt", [""])[0]
    top_k_param = query_params.get("top_k", [""])[0]
    mode_param = query_params.get("mode", ["auto"])[0]
    stamp_param = query_params.get("ts", [""])[0]

    latest_payload = None
    if prompt_param:
        try:
            safe_top_k = int(top_k_param)
        except (TypeError, ValueError):
            safe_top_k = 0
        stamp_key = f"{prompt_param}|{stamp_param}" if stamp_param else f"{prompt_param}|query"
        if st.session_state.get("bridge_stamp") != stamp_key:
            latest_payload = {
                "prompt": prompt_param,
                "top_k": safe_top_k,
                "mode": mode_param or "auto",
                "reason": "Prompt received from the Playground UI.",
                "stamp": stamp_key,
            }
    else:
        bridge_payload = visualizer_bridge.latest_prompt()
        if bridge_payload and bridge_payload.get("prompt"):
            stamp_key = f"{bridge_payload.get('prompt')}|{bridge_payload.get('timestamp')}"
            if st.session_state.get("bridge_stamp") != stamp_key:
                latest_payload = {
                    "prompt": bridge_payload.get("prompt", ""),
                    "top_k": bridge_payload.get("top_k"),
                    "mode": bridge_payload.get("mode") or "auto",
                    "reason": "Prompt queued by the backend.",
                    "stamp": stamp_key,
                }

    if latest_payload and latest_payload["prompt"].strip():
        st.session_state["prompt"] = latest_payload["prompt"]
        if latest_payload.get("top_k"):
            st.session_state["auto_top_k"] = int(latest_payload["top_k"])
        st.session_state["auto_mode"] = latest_payload.get("mode") or "auto"
        st.session_state["auto_trigger"] = True
        st.session_state["auto_reason"] = latest_payload.get("reason")
        st.session_state["bridge_stamp"] = latest_payload["stamp"]

    with left_col:
        st.subheader("Controls")
        st.markdown(
            "Enter a prompt to animate how the Daystrom Memory Lattice retrieves and scores memories."
        )
        st.subheader("Prompt")
        prompt = st.text_area(
            "Describe what you want the DML to recall", value=st.session_state.get("prompt", ""), height=180
        )
        st.session_state["prompt"] = prompt
        mode_options = ["auto", "semantic", "literal", "hybrid"]
        default_mode = st.session_state.get("auto_mode", "auto")
        mode_index = mode_options.index(default_mode) if default_mode in mode_options else 0
        mode = st.radio(
            "Retrieval mode",
            options=mode_options,
            index=mode_index,
            help="Choose how the lattice should search. 'Auto' selects a mode heuristically.",
        )
        default_top_k = int(st.session_state.get("auto_top_k", 6) or 6)
        default_top_k = max(1, min(MAX_TOP_K, default_top_k))
        top_k = st.slider("Max nodes", min_value=1, max_value=MAX_TOP_K, value=default_top_k, step=1)
        st.session_state["auto_top_k"] = top_k
        run_clicked = st.button("Run retrieval", type="primary", use_container_width=True)
        st.caption("Nodes pulse as they are scored. Lower fidelity nodes dim over time.")

    chart_placeholder = centre_col.empty()
    summary_placeholder = centre_col.empty()

    with right_col:
        st.subheader("Context metrics")
        mode_metric = right_col.empty()
        latency_metric = right_col.empty()
        nodes_metric = right_col.empty()
        fidelity_metric = right_col.empty()
        token_text = right_col.empty()
        token_bar = right_col.progress(0)

    graph_state = st.session_state["graph_state"]
    _refresh_graph_state(graph_state, adapter.store.items(), mode=graph_state.get("mode", DEFAULT_MODE))
    chart_placeholder.plotly_chart(build_figure(graph_state), use_container_width=True)

    auto_trigger = st.session_state.pop("auto_trigger", False)
    auto_reason = st.session_state.pop("auto_reason", None)
    if auto_trigger:
        run_clicked = True
        if auto_reason:
            summary_placeholder.info(auto_reason)
        mode = st.session_state.get("auto_mode", mode) or mode
        top_k = int(st.session_state.get("auto_top_k", top_k) or top_k)

    if not run_clicked or not prompt.strip():
        summary_placeholder.info("Enter a prompt and click *Run retrieval* to start the live visualisation.")
        return

    actual_mode = mode
    if actual_mode == "auto":
        actual_mode = adapter._classify_mode(prompt)  # type: ignore[attr-defined]
    actual_mode = actual_mode or DEFAULT_MODE

    stream = retrieval_stream(adapter, prompt, actual_mode, top_k=top_k)
    try:
        first_step = next(stream)
    except StopIteration:
        summary_placeholder.warning("No memory nodes were retrieved for this prompt.")
        return
    steps_iter = itertools.chain([first_step], stream)

    graph_state["mode"] = actual_mode
    budget = int(adapter.config.get("token_budget", 600))
    tokens_used = 0
    retrieved_ids: List[int] = []
    start_time = time.time()
    mode_metric.metric("Mode", actual_mode.title())
    nodes_metric.metric("Nodes retrieved", "0")
    latency_metric.metric("Latency (ms)", "—")
    fidelity_metric.metric("Average fidelity", "—")
    token_text.markdown(f"**Token budget:** 0 / {budget}")
    token_bar.progress(0)

    for step in steps_iter:
        retrieved_ids.append(step.item.id)
        tokens_used += step.tokens
        _refresh_graph_state(
            graph_state,
            adapter.store.items(),
            mode=actual_mode,
            active_item_id=step.item.id,
        )
        _update_active_node(graph_state, step)
        ordered_unique = list(dict.fromkeys(retrieved_ids))
        avg_fidelity = (
            sum(graph_state["nodes"][nid]["fidelity"] for nid in ordered_unique)
            / len(ordered_unique)
        )
        chart_placeholder.plotly_chart(build_figure(graph_state), use_container_width=True)
        elapsed_ms = (time.time() - start_time) * 1000.0
        nodes_metric.metric("Nodes retrieved", f"{len(retrieved_ids)}")
        latency_metric.metric("Latency (ms)", f"{elapsed_ms:.0f}")
        fidelity_metric.metric("Average fidelity", f"{avg_fidelity:.2f}")
        token_ratio = min(1.0, tokens_used / max(1, budget))
        token_text.markdown(
            f"**Token budget:** {tokens_used} / {budget}"
        )
        token_bar.progress(int(token_ratio * 100))
        time.sleep(FRAME_DELAY)

    # Highlight final nodes with a last pulse
    for node_id in retrieved_ids:
        node = graph_state["nodes"].get(node_id)
        if node:
            node["activation"] = max(0.8, node.get("activation", 0.0))
    chart_placeholder.plotly_chart(build_figure(graph_state), use_container_width=True)

    total_time_ms = (time.time() - start_time) * 1000.0
    summary_placeholder.success(
        f"DML retrieved {len(retrieved_ids)} nodes in {total_time_ms:.0f} ms using mode {actual_mode.title()}"
    )


if __name__ == "__main__":
    main()
