"""Streamlit playground for live DML retrieval visualisation."""
from __future__ import annotations

import os
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from daystrom_dml.dml_adapter import DMLAdapter

st.set_page_config(page_title="Daystrom Playground", layout="wide")

# Pricing defaults — roughly aligns with blended GPT-4o prompt/completion rates.
DEFAULT_BASELINE_TOKENS = 8192
DEFAULT_PRICE_PER_1K = 0.01


def _normalise_storage_dir(path: Path) -> Path:
    """Expand ``path`` and ensure it is rooted on the local filesystem."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _resolve_default_storage() -> Path:
    """Return the default storage directory for the playground."""

    env_override = os.environ.get("DML_PLAYGROUND_STORAGE") or os.environ.get(
        "DML_STORAGE_DIR"
    )
    if env_override:
        return _normalise_storage_dir(Path(env_override))
    return _normalise_storage_dir(Path("~/.dml/playground"))


DEFAULT_STORAGE_DIR = _resolve_default_storage()


def _create_adapter(storage_dir: Path) -> DMLAdapter:
    """Create a Daystrom adapter rooted at ``storage_dir``."""

    target = _normalise_storage_dir(storage_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
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
    st.session_state["storage_dir_input"] = str(storage_dir)


def _close_adapter() -> None:
    adapter: DMLAdapter | None = st.session_state.pop("adapter", None)
    if adapter is not None:
        adapter.close()


def _replace_adapter(storage_dir: Path) -> DMLAdapter:
    _close_adapter()
    adapter = _create_adapter(storage_dir)
    _store_adapter(adapter, _normalise_storage_dir(storage_dir))
    st.session_state.pop("last_result", None)
    st.session_state.pop("advanced_result", None)
    st.session_state.pop("simple_result", None)
    return adapter


def _resolve_storage_dir() -> Path:
    raw = st.session_state.get("storage_dir")
    if raw:
        return _normalise_storage_dir(Path(str(raw)))
    st.session_state["storage_dir"] = str(DEFAULT_STORAGE_DIR)
    return DEFAULT_STORAGE_DIR


def _clear_storage(storage_dir: Path) -> None:
    target = _normalise_storage_dir(storage_dir)
    if not target.exists():
        return
    for child in target.iterdir():
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


def _render_cost_savings(tokens_used: int, *, key_prefix: str) -> None:
    """Render a calculator estimating savings versus a naive baseline.

    ``tokens_used`` should reflect the tokens consumed by the latest request. The
    calculator compares that to an adjustable "no DML" baseline and a blended
    average generation price.
    """

    st.markdown("#### Cost impact")
    input_cols = st.columns(2)
    baseline_key = f"{key_prefix}_baseline_tokens"
    price_key = f"{key_prefix}_price_per_1k"
    baseline_default = int(st.session_state.get(baseline_key, DEFAULT_BASELINE_TOKENS))
    price_default = float(st.session_state.get(price_key, DEFAULT_PRICE_PER_1K))

    baseline_tokens = input_cols[0].number_input(
        "Baseline tokens without DML",
        min_value=1,
        value=baseline_default,
        step=512,
        help="Rough size of a naive context window you would have sent to the model.",
    )
    price_per_1k = input_cols[1].number_input(
        "Avg generation price per 1K tokens (USD)",
        min_value=0.0,
        value=price_default,
        step=0.001,
        format="%.4f",
        help="Use your provider's blended prompt + completion rate; defaults to GPT-4o averages.",
    )

    st.session_state[baseline_key] = int(baseline_tokens)
    st.session_state[price_key] = float(price_per_1k)

    adjusted_tokens_used = max(tokens_used, 0)
    baseline_cost = (baseline_tokens / 1000) * price_per_1k
    actual_cost = (adjusted_tokens_used / 1000) * price_per_1k
    token_delta = max(baseline_tokens - adjusted_tokens_used, 0)
    savings = max(baseline_cost - actual_cost, 0.0)
    savings_pct = (savings / baseline_cost * 100) if baseline_cost else 0.0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Baseline spend (est.)", f"${baseline_cost:,.4f}")
    metric_cols[1].metric("DML spend", f"${actual_cost:,.4f}", f"{token_delta} tokens saved")
    metric_cols[2].metric("Savings", f"${savings:,.4f}", f"{savings_pct:.1f}% vs baseline")


def _run_retrieval(adapter: DMLAdapter, prompt: str, *, mode: str) -> Dict[str, Any] | None:
    cleaned = prompt.strip()
    if not cleaned:
        return None
    try:
        result = adapter.query_database(cleaned, mode=mode)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        return None
    st.session_state["last_result"] = result
    st.session_state["last_prompt"] = cleaned
    st.session_state["last_mode"] = mode
    return result


def _try_ingest_text(adapter: DMLAdapter, text: str, source_label: str) -> tuple[str, str]:
    snippet = (text or "").strip()
    if not snippet:
        return "warning", "Provide some text before ingesting."
    try:
        adapter.ingest(snippet, meta={"doc_path": source_label})
    except Exception as exc:
        return "error", f"Failed to ingest {source_label}: {exc}"
    return "success", f"Ingested {source_label}"


def _render_simple_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.caption("Quickstart mode — minimal controls with safe defaults.")
    st.info(
        textwrap.dedent(
            """
            1. Upload a `.txt`, `.md`, `.json`, or `.log` file to populate the lattice.
            2. Paste snippets for quick notes when a file upload is overkill.
            3. Ask a question — retrieval automatically balances semantic and literal search.
            4. Switch to **Advanced** for storage management, token budgets, and the 3D lattice view.
            """
        ).strip()
    )

    st.subheader("Add knowledge")
    uploaded = st.file_uploader(
        "Upload text or markdown",
        type=["txt", "md", "log", "json"],
        help="Each file is chunked and stored as individual memories.",
        key="simple_upload",
    )
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        status, message = _try_ingest_text(adapter, text, uploaded.name)
        if status == "success":
            st.success(message)
        elif status == "warning":
            st.warning(message)
        else:
            st.error(message)

    with st.form("simple_manual_ingest"):
        manual_text = st.text_area(
            "Paste text to ingest",
            key="simple_manual_text",
            placeholder="Drop snippets here to create memories without uploading a file.",
            height=160,
        )
        submit_manual = st.form_submit_button("Ingest text snippet", use_container_width=True)
    if submit_manual:
        status, message = _try_ingest_text(adapter, manual_text, "manual-entry")
        if status == "success":
            st.success("Manual snippet ingested.")
            st.session_state["simple_manual_text"] = ""
        elif status == "warning":
            st.warning(message)
        else:
            st.error(message)

    items = list(adapter.store.items())
    st.markdown("### Lattice health")
    col_count, col_storage = st.columns(2)
    col_count.metric("Stored memories", len(items))
    col_storage.metric("Storage directory", str(storage_dir.resolve()))

    st.markdown("---")
    st.subheader("Ask the lattice")
    with st.form("simple_query_form"):
        st.text_area(
            "Prompt",
            key="simple_prompt",
            placeholder="Ask a question about your ingested data",
        )
        run_simple = st.form_submit_button("Ask", type="primary")
    if run_simple:
        prompt_value = st.session_state.get("simple_prompt", "")
        if not prompt_value.strip():
            st.warning("Enter a prompt before running retrieval.")
        else:
            result = _run_retrieval(adapter, prompt_value, mode="auto")
            if result:
                st.session_state["simple_result"] = result

    result = st.session_state.get("simple_result")
    if result:
        st.markdown(f"**Mode:** {result['mode']}")
        st.write(result.get("context") or "No context produced.")
        sources = result.get("source_docs", []) or []
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        else:
            st.caption("Sources: none")
        metrics = st.columns(2)
        metrics[0].metric("Tokens", int(result.get("tokens", 0)))
        metrics[1].metric("Latency (ms)", int(result.get("latency_ms", 0)))

        _render_cost_savings(int(result.get("tokens", 0)), key_prefix="simple")

    st.markdown("---")
    st.caption(
        "Ready for power features? Switch to **Advanced** to manage storage, inspect memory salience, and explore the 3D lattice."
    )


def _render_advanced_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.caption("Advanced mode — full control for power users and enterprise tuning.")

    prompt = st.text_area(
        "Prompt",
        key="advanced_prompt",
        placeholder="Ask a question about your ingested data",
    )
    mode = st.selectbox(
        "Retrieval mode",
        options=["auto", "semantic", "literal", "hybrid"],
        index=0,
        key="advanced_retrieval_mode",
    )
    run_query = st.button("Run retrieval", type="primary", key="advanced_run")
    if run_query:
        if not prompt.strip():
            st.warning("Enter a prompt before running retrieval.")
        else:
            result = _run_retrieval(adapter, prompt, mode=mode)
            if result:
                st.session_state["advanced_result"] = result

    result: Dict[str, Any] | None = st.session_state.get("advanced_result")
    if result is None:
        result = st.session_state.get("last_result")

    items = list(adapter.store.items())
    highlighted_ids: set[int] = set()

    if result:
        col_context, col_stats = st.columns([3, 1])
        with col_context:
            st.subheader(f"Mode: {result['mode']}")
            st.write(result.get("context") or "No context produced.")
            sources = result.get("source_docs", []) or []
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            else:
                st.caption("Sources: none")
        with col_stats:
            st.metric("Tokens", int(result.get("tokens", 0)))
            st.metric("Latency (ms)", int(result.get("latency_ms", 0)))

        _render_cost_savings(int(result.get("tokens", 0)), key_prefix="advanced")

        retrieved_texts = [
            segment for segment in (result.get("context") or "").split("\n") if segment
        ]
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

    st.markdown("### Token budget")
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
                        "Updated": datetime.fromtimestamp(item.timestamp).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Source": doc_path,
                        "Preview": item.text[:100].replace("\n", " "),
                    }
                )
            st.dataframe(table_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No memories ingested yet.")

    with st.expander("Adapter stats"):
        st.json(adapter.stats())

    st.caption(f"Storage directory: {storage_dir.resolve()}")


if hasattr(st, "on_session_end"):
    st.on_session_end(_close_adapter)

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

interface_options = ["Simple", "Advanced"]
if "ui_mode" in st.session_state:
    try:
        default_index = interface_options.index(st.session_state["ui_mode"])
    except ValueError:
        default_index = 0
else:
    default_index = 0
ui_mode = st.sidebar.radio(
    "Interface mode",
    options=interface_options,
    index=default_index,
    key="ui_mode",
)

if ui_mode == "Advanced":
    st.sidebar.header("Storage")
    storage_input = st.sidebar.text_input(
        "Storage directory",
        value=st.session_state.get("storage_dir_input", str(current_storage)),
        key="storage_dir_input",
    )
    if st.sidebar.button("Use storage directory", use_container_width=True):
        requested_raw = (storage_input or "").strip() or str(DEFAULT_STORAGE_DIR)
        requested_dir = _normalise_storage_dir(Path(requested_raw))
        if requested_dir != current_storage:
            try:
                adapter = _replace_adapter(requested_dir)
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
                st.stop()
            current_storage = requested_dir
            st.sidebar.success(f"Switched storage to {requested_dir}")
        else:
            st.sidebar.info("Storage directory unchanged.")

    if st.sidebar.button("Reset lattice", use_container_width=True):
        _close_adapter()
        try:
            _clear_storage(current_storage)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        try:
            adapter = _replace_adapter(current_storage)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        st.sidebar.success("Cleared stored memories.")

    st.sidebar.divider()
    st.sidebar.subheader("Ingestion")
    uploaded = st.sidebar.file_uploader(
        "Upload text or markdown",
        type=["txt", "md", "log", "json"],
        help="Each file is chunked and stored as individual memories.",
        key="advanced_upload",
    )
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        status, message = _try_ingest_text(adapter, text, uploaded.name)
        if status == "success":
            st.sidebar.success(message)
        elif status == "warning":
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)

    manual_text = st.sidebar.text_area(
        "Paste text to ingest",
        key="advanced_manual_text",
        placeholder="Drop snippets here to create memories without uploading a file.",
        height=160,
    )
    if st.sidebar.button("Ingest text snippet", use_container_width=True):
        status, message = _try_ingest_text(adapter, manual_text, "manual-entry")
        if status == "success":
            st.sidebar.success("Manual snippet ingested.")
            st.session_state["advanced_manual_text"] = ""
        elif status == "warning":
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
else:
    st.sidebar.info(
        "Advanced storage and ingestion controls are available when Advanced mode is selected."
    )

current_storage = _resolve_storage_dir()
st.sidebar.caption(f"Storage directory: {current_storage.resolve()}")

st.title("Daystrom Memory Lattice Playground")

if ui_mode == "Simple":
    _render_simple_mode(adapter, current_storage)
else:
    _render_advanced_mode(adapter, current_storage)
