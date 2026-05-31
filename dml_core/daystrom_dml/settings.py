"""Configuration model for the Daystrom Memory Lattice."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # Support both Pydantic v1 and v2
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - pydantic should always be present via FastAPI
    raise

try:  # pragma: no cover - optional import for Pydantic v2
    from pydantic import ConfigDict, field_validator
except ImportError:  # pragma: no cover - fallback for Pydantic v1
    ConfigDict = None  # type: ignore[misc]
    field_validator = None  # type: ignore[assignment]
    from pydantic import validator as legacy_validator
else:
    legacy_validator = None  # type: ignore[assignment]


class PersistenceSettings(BaseModel):
    """Configuration for durable memory persistence."""

    enable: bool = False
    path: Path = Path("dml_state.jsonl")
    interval_sec: int = Field(300, ge=0)

    if field_validator is not None:  # pragma: no branch - executed on Pydantic v2

        @field_validator("path", mode="before")
        def _coerce_path(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))

    else:  # pragma: no cover - Pydantic v1 compatibility

        @legacy_validator("path", pre=True)
        def _coerce_path(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))


class RAGStoreSettings(BaseModel):
    """Configuration for the lightweight persistent RAG vector store."""

    enable: bool = False
    path: Path = Path("./data/rag_index.faiss")
    meta_path: Path = Path("./data/rag_meta.json")
    backend: str = "faiss"
    dim: int = Field(384, ge=1)

    if field_validator is not None:  # pragma: no branch - executed on Pydantic v2

        @field_validator("path", "meta_path", mode="before")
        def _coerce_path(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))

    else:  # pragma: no cover - Pydantic v1 compatibility

        @legacy_validator("path", "meta_path", pre=True)
        def _coerce_path(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))


class LiteralSettings(BaseModel):
    """Configuration for literal retrieval controls."""

    max_snippet_tokens: int = Field(160, ge=16)
    max_snippets: int = Field(8, ge=1)


class BudgetSettings(BaseModel):
    """Configuration for token allocation across retrieval strategies."""

    semantic_pct: float = Field(0.7, ge=0.0, le=1.0)
    literal_pct: float = Field(0.2, ge=0.0, le=1.0)
    free_pct: float = Field(0.1, ge=0.0, le=1.0)

    def validate_totals(self) -> None:
        total = float(self.semantic_pct + self.literal_pct + self.free_pct)
        if total > 1.0 + 1e-6:  # pragma: no cover - guardrail
            raise ValueError("Token budget percentages cannot exceed 1.0")


class DPMSettings(BaseModel):
    """Configuration for the Daystrom Personality Matrix runtime overlay."""

    enable: bool = False
    mode: str = Field("disabled", description="DPM lifecycle mode: disabled, observe-only, active-read, active-write.")
    overlay_path: Path | None = None
    preference_graph_path: Path | None = None
    max_overlay_chars: int = Field(280, ge=1)
    token_budget: int = Field(80, ge=1)
    include_in_context: bool = True
    include_in_preamble: bool = True
    relationship_id: str | None = None
    project_id: str | None = None

    if field_validator is not None:  # pragma: no branch - executed on Pydantic v2

        @field_validator("overlay_path", "preference_graph_path", mode="before")
        def _coerce_optional_path(cls, value: Any) -> Path | None:
            if value in {None, ""}:
                return None
            if isinstance(value, Path):
                return value
            return Path(str(value))

    else:  # pragma: no cover - Pydantic v1 compatibility

        @legacy_validator("overlay_path", "preference_graph_path", pre=True)
        def _coerce_optional_path(cls, value: Any) -> Path | None:
            if value in {None, ""}:
                return None
            if isinstance(value, Path):
                return value
            return Path(str(value))


class DMLSettings(BaseModel):
    """Central configuration for the DML stack with env overrides."""

    beta_a: float = 0.08
    beta_r: float = 0.2
    eta: float = 0.15
    gamma: float = 0.02
    kappa: float = 0.5
    tau_s: float = 0.1
    theta_merge: float = 0.92
    K: int = Field(4, ge=1)
    capacity: int = Field(2000, ge=1)
    top_k: int = Field(6, ge=1)
    dml_top_k: int = Field(8, ge=0)
    similarity_threshold: float = Field(0.32, ge=-1.0, le=1.0)
    literal_context: int = Field(1, ge=0)
    token_budget: int = Field(600, ge=1)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_backend: str = "auto"
    llm_device: str | None = None
    llm_dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    llm_temperature: float = Field(0.2, ge=0.0)
    llm_top_p: float = Field(1.0, ge=0.0, le=1.0)
    embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None
    strict_embedding_required: bool = Field(
        False,
        description="Fail closed when the configured embedding model cannot be loaded instead of falling back to RandomEmbedder.",
    )
    strict_llm_required: bool = Field(
        False,
        description="Fail closed when the configured LLM backend/model cannot be loaded instead of falling back to DummyGPT.",
    )
    enable_stm_controller: bool = Field(False, description="Enable structured STM controller loop.")
    commitment_threshold: float = Field(0.75, ge=0.0, le=1.0)
    ltm_write_policy: str = Field("balanced", description="LTM write policy: strict, balanced, off.")
    stm_max_commitments: int = Field(8, ge=1)
    stm_max_entities: int = Field(8, ge=1)
    ltm_top_k: int = Field(6, ge=1)
    stm_extract_max_tokens: int = Field(256, ge=64)
    storage_dir: Path = Field(Path("data"), description="Root directory for persisted artefacts.")
    checkpoint_interval_seconds: int = Field(0, ge=0)
    checkpoint_retention: int = Field(3, ge=0)
    vector_index_file: str = Field("vector_index.json", description="Filename for the persistent vector index.")
    metrics_namespace: str = Field("daystrom_dml", description="Namespace prefix for exported metrics.")
    metrics_enabled: bool = Field(True, description="Toggle Prometheus metric emission.")
    enable_quality_on_retrieval: bool = Field(
        False,
        description="Run quality assessment during retrieval when True; otherwise defer to maintenance.",
    )
    enable_workflow_cache: bool = Field(
        False, description="Enable storing and suggesting reusable agent workflow templates."
    )
    gpu_acceleration: bool = Field(False, description="Enable GPU specific optimisations when available.")
    nim_default_id: str = Field("gpt-oss-20b", description="Default NIM model identifier.")
    nim_health_timeout: int = Field(60, ge=1)
    nim_health_interval: float = Field(5.0, ge=0.1)
    persistence: PersistenceSettings = PersistenceSettings()
    rag_store: RAGStoreSettings = RAGStoreSettings()
    literal: LiteralSettings = LiteralSettings()
    budgets: BudgetSettings = BudgetSettings()
    dpm: DPMSettings = DPMSettings()

    if ConfigDict is not None:  # pragma: no branch - executed on Pydantic v2
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - configuration for Pydantic v1
        class Config:
            extra = "allow"

    if field_validator is not None:  # pragma: no branch - Pydantic v2 path

        @field_validator("storage_dir", mode="before")
        def _coerce_storage_dir(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))

    else:  # pragma: no cover - Pydantic v1 compatibility

        @legacy_validator("storage_dir", pre=True)
        def _coerce_storage_dir(cls, value: Any) -> Path:
            if isinstance(value, Path):
                return value
            return Path(str(value))

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""

        if hasattr(self, "model_dump"):
            data = self.model_dump()
        else:  # pragma: no cover - used on Pydantic v1
            data = self.dict()
        data["storage_dir"] = str(self.storage_dir)
        persistence = data.get("persistence")
        if isinstance(persistence, dict) and "path" in persistence:
            persistence["path"] = str(persistence["path"])
        rag_store = data.get("rag_store")
        if isinstance(rag_store, dict):
            if "path" in rag_store:
                rag_store["path"] = str(rag_store["path"])
            if "meta_path" in rag_store:
                rag_store["meta_path"] = str(rag_store["meta_path"])
        dpm = data.get("dpm")
        if isinstance(dpm, dict):
            if dpm.get("overlay_path") is not None:
                dpm["overlay_path"] = str(dpm["overlay_path"])
            if dpm.get("preference_graph_path") is not None:
                dpm["preference_graph_path"] = str(dpm["preference_graph_path"])
        return data
