"""Configuration model for the Daystrom Memory Lattice."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # Support both Pydantic v1 and v2
    from pydantic import Field
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - pydantic should always be present via FastAPI
    raise

try:  # pragma: no cover - prefer dedicated settings package on Pydantic v2
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - fallback to legacy location
    from pydantic import BaseSettings  # type: ignore[misc]

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
    path: Path = Path("data/dml_state.jsonl")
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


class DMLSettings(BaseSettings):
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
    dml_top_k: int = Field(0, ge=0)
    literal_context: int = Field(1, ge=0)
    token_budget: int = Field(600, ge=1)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    storage_dir: Path = Field(Path("data"), description="Root directory for persisted artefacts.")
    checkpoint_interval_seconds: int = Field(0, ge=0)
    checkpoint_retention: int = Field(3, ge=0)
    vector_index_file: str = Field("vector_index.json", description="Filename for the persistent vector index.")
    metrics_namespace: str = Field("daystrom_dml", description="Namespace prefix for exported metrics.")
    metrics_enabled: bool = Field(True, description="Toggle Prometheus metric emission.")
    gpu_acceleration: bool = Field(False, description="Enable GPU specific optimisations when available.")
    nim_default_id: str = Field("gpt-oss-20b", description="Default NIM model identifier.")
    nim_health_timeout: int = Field(60, ge=1)
    nim_health_interval: float = Field(5.0, ge=0.1)
    persistence: PersistenceSettings = PersistenceSettings()

    if ConfigDict is not None:  # pragma: no branch - executed on Pydantic v2
        model_config = ConfigDict(
            env_prefix="DML_",
            env_file=".env",
            case_sensitive=False,
            env_nested_delimiter="__",
            extra="allow",
        )
    else:  # pragma: no cover - configuration for Pydantic v1
        class Config:
            env_prefix = "DML_"
            env_file = ".env"
            case_sensitive = False
            env_nested_delimiter = "__"
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
        return data
