"""Streamlit playground for live DML retrieval visualisation."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from daystrom_dml.dml_adapter import DMLAdapter

st.set_page_config(page_title="Daystrom Playground", layout="wide")

DEFAULT_STORAGE_DIR = Path("./data/playground")


def _create_adapter(storage_dir: Path) -> DMLAdapter:
    """Create a Daystrom adapter rooted at ``storage_dir``."""

    target = storage_dir.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    try:
        adapter = DMLAdapter(
            config_overrides={"storage_dir": str(target)},
            start_aging_loop=False,
        )
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        raise RuntimeError(f"Failed to initialise Daystrom adapter: {exc}") from exc
    return adapter


def _store_adapter(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.session_state["adapter"] = adapter
    st.session_state["storage_dir"] = str(storage_dir)


def _close_adapter() -> None:
    adapter: DMLAdapter | None = st.session_state.pop("adapter", None)
    if adapter is not None:
        adapter.close()


def _replace_adapter(storage_dir: Path) -> DMLAdapter:
    _close_adapter()
    adapter = _create_adapter(storage_dir)
    _store_adapter(adapter, storage_dir)
    st.session_state.pop("last_result", None)
    return adapter


def _resolve_storage_dir() -> Path:
    raw = st.session_state.get("storage_dir")
    if raw:
        return Path(str(raw)).expanduser()
    st.session_state["storage_dir"] = str(DEFAULT_STORAGE_DIR)
    return DEFAULT_STORAGE_DIR.expanduser()


def _clear_storage(storage_dir: Path) -> None:
    if not storage_dir.exists():
        return
    for child in storage_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:  # pragma: no cover - surfaced via Streamlit
            raise RuntimeError(f"Failed to clear storage: {exc}") from exc


def _vector_to_coords(vector: np.ndarray) -> Tuple[float, float, float]:
    if vector.size == 0:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(vector, dtype=np.float32)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return float(arr[0]), float(arr[1]), float(arr[2])


def _build_lattice_plot(items: List, highlighted: set[int]) -> go.Figure:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    colors: List[int] = []
    texts: List[str] = []
    for item in items:
        x, y, z = _vector_to_coords(item.embedding)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        colors.append(item.level)
        summary = item.text[:120].replace("\n", " ")
        texts.append(f"L{item.level} • {summary}")

    base = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        name="Memories",
        marker=dict(size=4, color=colors, colorscale="Viridis", opacity=0.5),
        text=texts,
        hoverinfo="text",
    )

    highlighted_points = []
    if highlighted:
        hx: List[float] = []
        hy: List[float] = []
        hz: List[float] = []
        htext: List[str] = []
        for item in items:
            if item.id not in highlighted:
                continue
            x, y, z = _vector_to_coords(item.embedding)
            hx.append(x)
            hy.append(y)
            hz.append(z)
            htext.append(f"Hit • L{item.level}")
        highlighted_points.append(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="markers",
                name="Retrieved",
                marker=dict(size=8, color="#FF3366", opacity=0.95),
                text=htext,
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=[base, *highlighted_points])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520,
    )
    return fig


current_storage = _resolve_storage_dir()
if "storage_dir_input" not in st.session_state:
    st.session_state["storage_dir_input"] = str(current_storage)

adapter: DMLAdapter | None = st.session_state.get("adapter")
if adapter is None:
    try:
        adapter = _replace_adapter(current_storage)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

assert adapter is not None

st.sidebar.header("Storage")
with st.sidebar:
    storage_input = st.text_input(
        "Storage directory",
        value=st.session_state.get("storage_dir_input", str(current_storage)),
        key="storage_dir_input",
    )
    if st.button("Use storage directory", use_container_width=True):
        requested_dir = Path((storage_input or "").strip() or str(DEFAULT_STORAGE_DIR)).expanduser()
        if requested_dir != current_storage:
            try:
                adapter = _replace_adapter(requested_dir)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            current_storage = requested_dir
            st.session_state["storage_dir_input"] = str(requested_dir)
            st.success(f"Switched storage to {requested_dir}")
        else:
            st.info("Storage directory unchanged.")
    if st.button("Reset lattice", use_container_width=True):
        _close_adapter()
        try:
            _clear_storage(current_storage)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        try:
            adapter = _replace_adapter(current_storage)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["storage_dir_input"] = str(current_storage)
        st.success("Cleared stored memories.")

    st.divider()
    st.subheader("Ingestion")
    uploaded = st.file_uploader(
        "Upload text or markdown",
        type=["txt", "md", "log", "json"],
        help="Each file is chunked and stored as individual memories.",
    )
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        try:
            adapter.ingest(text, meta={"doc_path": uploaded.name})
        except Exception as exc:
            st.error(f"Failed to ingest {uploaded.name}: {exc}")
        else:
            st.success(f"Ingested {uploaded.name}")
    manual_text = st.text_area(
        "Paste text to ingest",
        key="manual_ingest_text",
        placeholder="Drop snippets here to create memories without uploading a file.",
        height=160,
    )
    if st.button("Ingest text snippet", use_container_width=True):
        snippet = manual_text.strip()
        if not snippet:
            st.warning("Provide some text before ingesting.")
        else:
            try:
                adapter.ingest(snippet, meta={"doc_path": "manual-entry"})
            except Exception as exc:
                st.error(f"Failed to ingest manual snippet: {exc}")
            else:
                st.success("Manual snippet ingested.")
                st.session_state["manual_ingest_text"] = ""

st.sidebar.caption(f"Storage directory: {current_storage.resolve()}")

st.title("Daystrom Memory Lattice Playground")

prompt = st.text_area("Prompt", placeholder="Ask a question about your ingested data")
mode = st.selectbox("Retrieval mode", options=["auto", "semantic", "literal", "hybrid"], index=0)

result: Dict[str, Any] | None = None
run_query = st.button("Run retrieval", type="primary")
if run_query and not prompt.strip():
    st.warning("Enter a prompt before running retrieval.")
elif run_query:
    with st.spinner("Running retrieval..."):
        try:
            result = adapter.query_database(prompt.strip(), mode=mode)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            result = None
        else:
            st.session_state["last_result"] = result

result = result or st.session_state.get("last_result")

col_context, col_stats = st.columns([3, 1])

items = list(adapter.store.items())
highlighted_ids: set[int] = set()

if result:
    with col_context:
        st.subheader(f"Mode: {result['mode']}")
        st.write(result["context"] or "No context produced.")
        sources = result.get("source_docs", []) or []
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        else:
            st.caption("Sources: none")
    with col_stats:
        st.metric("Tokens", int(result.get("tokens", 0)))
        st.metric("Latency (ms)", int(result.get("latency_ms", 0)))
    retrieved_texts = [segment for segment in (result.get("context") or "").split("\n") if segment]
    source_docs = {doc for doc in result.get("source_docs", []) if doc}
    for item in items:
        snippet = item.text
        meta = getattr(item, "meta", {}) or {}
        doc_path = str(meta.get("doc_path") or "")
        if doc_path and doc_path in source_docs:
            highlighted_ids.add(item.id)
            continue
        if any(fragment in snippet for fragment in retrieved_texts):
            highlighted_ids.add(item.id)

budgets = adapter.config.get("budgets", {})
semantic_pct = float(budgets.get("semantic_pct", 0.7))
literal_pct = float(budgets.get("literal_pct", 0.2))
free_pct = float(budgets.get("free_pct", 0.1))

st.markdown("### Token Budget")
budget_cols = st.columns(3)
budget_cols[0].progress(min(1.0, semantic_pct), text="Semantic")
budget_cols[1].progress(min(1.0, literal_pct), text="Literal")
budget_cols[2].progress(min(1.0, free_pct), text="Free")

fig = _build_lattice_plot(items, highlighted_ids)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Memory catalogue"):
    if items:
        table_rows = []
        for item in items:
            meta = getattr(item, "meta", {}) or {}
            doc_path = meta.get("doc_path") or "—"
            table_rows.append(
                {
                    "ID": item.id,
                    "Level": item.level,
                    "Salience": f"{item.salience:.2f}",
                    "Fidelity": f"{item.fidelity:.2f}",
                    "Updated": datetime.fromtimestamp(item.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": doc_path,
                    "Preview": item.text[:100].replace("\n", " "),
                }
            )
        st.dataframe(table_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No memories ingested yet.")

with st.expander("Adapter stats"):
    st.json(adapter.stats())
