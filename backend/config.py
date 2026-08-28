"""
Central, typed configuration for the AI Research Paper Assistant.

This module contains configuration only. Importing it does not load datasets,
models, FAISS, PDFs, embeddings, or LLM providers.

The concrete defaults below are derived from the existing project contracts:
SPECTER2 defaults, reranker defaults, RAG/search defaults, uploaded-index
layout, upload gateway limits, and LLM client configuration keys.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import Self

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    SettingsConfigDict,
)


# ---------------------------------------------------------------------------
# Stable project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppEnvironment(str, Enum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Single source of truth for backend configuration.

    Environment variables use the same uppercase names consumed by the
    existing LLM client and upload/search code. No runtime service is created
    here.
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        env_ignore_empty=True,
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    app_name: str = Field(
        default="AI Research Paper Assistant",
        min_length=1,
        max_length=200,
    )
    app_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=50,
    )
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    debug: bool = False

    # -----------------------------------------------------------------------
    # API server
    # -----------------------------------------------------------------------

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)

    # -----------------------------------------------------------------------
    # CORS
    #
    # Existing application defaults are local development frontends.
    # Production deployments should override CORS_ORIGINS explicitly.
    # -----------------------------------------------------------------------

    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:5173",
        ],
    )

    # -----------------------------------------------------------------------
    # Data paths
    #
    # These are definitions only. Directories are not created on import.
    # -----------------------------------------------------------------------

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"

    raw_dataset_path: Path = (
        PROJECT_ROOT / "data" / "raw" / "arxiv_scientific_dataset.csv"
    )
    cleaned_dataset_path: Path = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_arxiv_dataset.csv"
)
    metadata_dataset_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "paper_metadata.parquet"
    )

    # -----------------------------------------------------------------------
    # Research FAISS index
    #
    # FAISSVectorIndex uses an index base; the concrete loader resolves the
    # .index / manifest artifacts from it.
    # -----------------------------------------------------------------------

    research_index_dir: Path = (
        PROJECT_ROOT / "indexes" / "research"
    )
    research_index_base: Path = (
        PROJECT_ROOT / "indexes" / "research" / "index"
    )

    # -----------------------------------------------------------------------
    # Uploaded-paper storage/index
    #
    # paper_indexing.py currently uses indexes/uploaded/<document_id>/.
    # papers.py currently stages uploads under outputs/uploaded_papers/_staging.
    # -----------------------------------------------------------------------

    uploaded_index_dir: Path = (
        PROJECT_ROOT / "indexes" / "uploaded"
    )
    upload_root: Path = (
        PROJECT_ROOT / "outputs" / "uploaded_papers"
    )
    temp_upload_dir: Path = (
        PROJECT_ROOT / "outputs" / "uploaded_papers" / "_staging"
    )

    # -----------------------------------------------------------------------
    # Output roots
    #
    # These are centralized locations; config.py does not write to them.
    # -----------------------------------------------------------------------

    outputs_dir: Path = PROJECT_ROOT / "outputs"
    research_outputs_dir: Path = (
        PROJECT_ROOT / "outputs" / "research"
    )
    research_top10_output_dir: Path = (
        PROJECT_ROOT / "outputs" / "research" / "top10"
    )
    paper_outputs_dir: Path = (
        PROJECT_ROOT / "outputs" / "papers"
    )
    paper_summaries_output_dir: Path = (
        PROJECT_ROOT / "outputs" / "papers" / "summaries"
    )
    paper_qa_output_dir: Path = (
        PROJECT_ROOT / "outputs" / "papers" / "qa"
    )
    paper_analysis_output_dir: Path = (
        PROJECT_ROOT / "outputs" / "papers" / "analysis"
    )

    # -----------------------------------------------------------------------
    # Model/cache locations
    #
    # SPECTER2 and the reranker use Hugging Face model identifiers by default.
    # These paths are optional cache locations, not forced local model files.
    # -----------------------------------------------------------------------

    models_dir: Path = PROJECT_ROOT / "models"
    embeddings_model_dir: Path = (
        PROJECT_ROOT / "models" / "embeddings"
    )
    reranker_model_dir: Path = (
        PROJECT_ROOT / "models" / "reranker"
    )
    llm_model_dir: Path = (
        PROJECT_ROOT / "models" / "llm"
    )

    # -----------------------------------------------------------------------
    # SPECTER2
    # -----------------------------------------------------------------------

    specter2_base_model: str = "allenai/specter2_base"
    specter2_document_adapter: str = "allenai/specter2"
    specter2_query_adapter: str = "allenai/specter2_adhoc_query"

    specter2_device: str = Field(default="auto", min_length=1)
    specter2_batch_size: int = Field(default=16, ge=1, le=4096)
    specter2_max_length: int = Field(default=512, ge=1, le=8192)
    specter2_normalize: bool = True
    specter2_output_dtype: str = "float32"
    specter2_cache_dir: Path | None = None
    specter2_local_files_only: bool = False
    specter2_trust_remote_code: bool = False
    specter2_show_progress: bool = False
    specter2_deterministic: bool = True
    specter2_use_autocast: bool = False
    specter2_normalization_tolerance: float = Field(
        default=1e-4,
        gt=0,
    )

    # -----------------------------------------------------------------------
    # Retrieval / ranking
    # -----------------------------------------------------------------------

    research_candidate_k: int = Field(default=50, ge=1, le=1000)
    research_max_candidate_k: int = Field(default=100, ge=1, le=5000)

    uploaded_candidate_k: int = Field(default=20, ge=1, le=1000)
    uploaded_max_candidate_k: int = Field(default=100, ge=1, le=5000)

    final_k: int = Field(default=10, ge=1, le=100)

    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = Field(default="auto", min_length=1)
    reranker_batch_size: int = Field(default=16, ge=1, le=4096)
    reranker_max_length: int = Field(default=512, ge=1, le=8192)
    reranker_final_k: int = Field(default=10, ge=1, le=100)
    reranker_cache_dir: Path | None = None
    reranker_revision: str | None = None
    reranker_normalize_score: bool = False

    # -----------------------------------------------------------------------
    # RAG/context
    #
    # These defaults match the existing ContextBuilder/Pipeline contracts.
    # -----------------------------------------------------------------------

    rag_context_max_characters: int = Field(
        default=24_000,
        ge=1,
        le=2_000_000,
    )
    rag_context_max_evidence_items: int = Field(
        default=12,
        ge=1,
        le=1000,
    )

    # -----------------------------------------------------------------------
    # LLM client
    #
    # LLMClient.create_client_from_mapping() consumes these exact keys.
    # API keys are never logged or serialized as plain text.
    # -----------------------------------------------------------------------

    llm_provider: str | None = Field(default=None, min_length=1)
    llm_model: str | None = Field(default=None, min_length=1)
    llm_api_key: SecretStr | None = None
    llm_base_url: AnyHttpUrl | None = None

    llm_temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    llm_max_output_tokens: int | None = Field(
        default=2048,
        ge=1,
        le=1_000_000,
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=3600.0,
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=20,
    )
    llm_backoff_base_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=300.0,
    )
    llm_backoff_max_seconds: float = Field(
        default=8.0,
        ge=0.0,
        le=3600.0,
    )
    llm_jitter_seconds: float = Field(
        default=0.2,
        ge=0.0,
        le=60.0,
    )
    llm_enable_structured_output: bool = True

    # Ollama structured-output transport mode.
    #
    # ``json`` is the safest default for the current local Ollama/Qwen
    # integration: Ollama enforces valid JSON at the transport layer while
    # backend/llm/parser.py remains responsible for strict application-level
    # schema validation. ``schema`` is available for providers/models that
    # support Ollama's JSON-schema ``format`` payload.
    llm_ollama_structured_output_mode: str = Field(
        default="json",
        min_length=1,
    )

    # -----------------------------------------------------------------------
    # Upload/PDF gateway limits
    # -----------------------------------------------------------------------

    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=2048,
    )
    upload_read_chunk_size_bytes: int = Field(
        default=1024 * 1024,
        ge=4096,
        le=16 * 1024 * 1024,
    )

    allowed_file_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [".pdf"],
    )

    # -----------------------------------------------------------------------
    # Uploaded index persistence
    # -----------------------------------------------------------------------

    uploaded_index_overwrite: bool = False
    uploaded_index_include_chunk_text: bool = True
    uploaded_index_verify_reload: bool = True
    uploaded_index_max_chunk_text_chars: int = Field(
        default=2_000_000,
        ge=1,
        le=20_000_000,
    )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Validation / cross-field checks
    # -----------------------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        """Accept JSON lists or comma-separated environment values."""
        if value is None:
            return []

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []

            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "CORS_ORIGINS must be a JSON list or comma-separated "
                        "origins."
                    ) from exc

                if not isinstance(parsed, list):
                    raise ValueError("CORS_ORIGINS JSON value must be a list.")
                value = parsed
            else:
                value = raw.split(",")

        if not isinstance(value, (list, tuple, set)):
            raise ValueError(
                "CORS_ORIGINS must be a list or comma-separated string."
            )

        origins = [str(item).strip() for item in value if str(item).strip()]

        if "*" in origins and len(origins) > 1:
            raise ValueError(
                "CORS_ORIGINS cannot combine '*' with explicit origins."
            )

        for origin in origins:
            if origin == "*":
                continue
            try:
                AnyHttpUrl(origin)
            except Exception as exc:
                raise ValueError(
                    f"Invalid CORS origin: {origin!r}."
                ) from exc

        return origins

    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, value: Any) -> list[str]:
        """Normalize configured upload extensions."""
        if value is None:
            return [".pdf"]

        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "ALLOWED_FILE_EXTENSIONS must be a JSON list or "
                        "comma-separated string."
                    ) from exc
                value = parsed
            else:
                value = raw.split(",")

        if not isinstance(value, (list, tuple, set)):
            raise ValueError(
                "ALLOWED_FILE_EXTENSIONS must be a list or string."
            )

        extensions = []
        for item in value:
            extension = str(item).strip().lower()
            if not extension:
                continue
            if not extension.startswith("."):
                extension = f".{extension}"
            extensions.append(extension)

        if not extensions:
            raise ValueError(
                "At least one allowed file extension is required."
            )

        if any(extension != ".pdf" for extension in extensions):
            raise ValueError(
                "The current document pipeline supports PDF uploads only."
            )

        return list(dict.fromkeys(extensions))

    @field_validator("specter2_device", "reranker_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "Device must be one of: auto, cpu, cuda."
            )
        return normalized

    @field_validator("llm_ollama_structured_output_mode")
    @classmethod
    def validate_ollama_structured_output_mode(cls, value: str) -> str:
        """Normalize and validate the Ollama structured-output mode."""
        normalized = value.strip().lower()
        if normalized not in {"none", "json", "schema"}:
            raise ValueError(
                "LLM_OLLAMA_STRUCTURED_OUTPUT_MODE must be "
                "'none', 'json', or 'schema'."
            )
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ValueError(
                "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            )
        return normalized

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if self.research_max_candidate_k < self.research_candidate_k:
            raise ValueError(
                "RESEARCH_MAX_CANDIDATE_K must be >= RESEARCH_CANDIDATE_K."
            )

        if self.uploaded_max_candidate_k < self.uploaded_candidate_k:
            raise ValueError(
                "UPLOADED_MAX_CANDIDATE_K must be >= UPLOADED_CANDIDATE_K."
            )

        if self.research_candidate_k < self.final_k:
            raise ValueError(
                "RESEARCH_CANDIDATE_K must be >= FINAL_K."
            )

        if self.reranker_final_k != self.final_k:
            # Keep one final Top-K policy across the search pipeline.
            raise ValueError(
                "RERANKER_FINAL_K must equal FINAL_K so the final ranking "
                "contract remains consistent."
            )

        if self.llm_backoff_max_seconds < self.llm_backoff_base_seconds:
            raise ValueError(
                "LLM_BACKOFF_MAX_SECONDS must be >= "
                "LLM_BACKOFF_BASE_SECONDS."
            )

        if self.app_env is AppEnvironment.PRODUCTION and self.debug:
            raise ValueError(
                "DEBUG=true is not allowed when APP_ENV=production."
            )

        return self

    # -----------------------------------------------------------------------
    # Compatibility helpers
    # -----------------------------------------------------------------------

    @property
    def research_index_path(self) -> Path:
        """Compatibility alias for existing main/service wiring."""
        return self.research_index_base

    @property
    def uploaded_index_root(self) -> Path:
        """Compatibility alias for existing uploaded retriever wiring."""
        return self.uploaded_index_dir

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload size limit converted from MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def as_llm_mapping(self) -> dict[str, Any]:
        """
        Return only the configuration keys consumed by llm.client.

        SecretStr is deliberately unwrapped only at the runtime boundary where
        the LLM client needs the actual credential.
        """
        api_key = (
            self.llm_api_key.get_secret_value()
            if self.llm_api_key is not None
            else None
        )

        return {
            "LLM_PROVIDER": self.llm_provider,
            "LLM_MODEL": self.llm_model,
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": (
                str(self.llm_base_url)
                if self.llm_base_url is not None
                else None
            ),
            "LLM_TEMPERATURE": self.llm_temperature,
            "LLM_MAX_OUTPUT_TOKENS": self.llm_max_output_tokens,
            "LLM_TIMEOUT_SECONDS": self.llm_timeout_seconds,
            "LLM_MAX_RETRIES": self.llm_max_retries,
            "LLM_BACKOFF_BASE_SECONDS": self.llm_backoff_base_seconds,
            "LLM_BACKOFF_MAX_SECONDS": self.llm_backoff_max_seconds,
            "LLM_JITTER_SECONDS": self.llm_jitter_seconds,
            "LLM_ENABLE_STRUCTURED_OUTPUT": (
                self.llm_enable_structured_output
            ),
            "LLM_OLLAMA_STRUCTURED_OUTPUT_MODE": (
                self.llm_ollama_structured_output_mode
            ),
        }


# Cached singleton. Settings are immutable at the application level.
_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide validated settings object."""
    global _SETTINGS

    if _SETTINGS is None:
        _SETTINGS = Settings()

    return _SETTINGS


settings = get_settings()


def validate_configuration() -> Settings:
    """
    Validate configuration without checking runtime resource availability.

    This deliberately does not check whether datasets/indexes/models exist.
    Resource availability belongs to application startup/readiness.
    """
    return get_settings()


def run_self_test() -> None:
    """Fast, model-free configuration contract tests."""
    test = Settings(
        app_env="testing",
        debug=False,
        research_candidate_k=50,
        research_max_candidate_k=100,
        final_k=10,
        reranker_final_k=10,
        cors_origins=["http://localhost:5173"],
        allowed_file_extensions=[".pdf"],
        llm_provider="test-provider",
        llm_model="test-model",
        llm_api_key="test-secret",
    )

    assert test.app_env is AppEnvironment.TESTING
    assert test.research_candidate_k == 50
    assert test.final_k == 10
    assert test.max_upload_size_bytes == 50 * 1024 * 1024
    assert test.allowed_file_extensions == [".pdf"]
    assert test.cors_origins == ["http://localhost:5173"]
    assert test.llm_api_key is not None
    assert test.llm_api_key.get_secret_value() == "test-secret"

    llm_mapping = test.as_llm_mapping()
    assert llm_mapping["LLM_PROVIDER"] == "test-provider"
    assert llm_mapping["LLM_MODEL"] == "test-model"
    assert llm_mapping["LLM_API_KEY"] == "test-secret"
    assert test.llm_ollama_structured_output_mode == "json"
    assert llm_mapping["LLM_OLLAMA_STRUCTURED_OUTPUT_MODE"] == "json"

    # Invalid candidate relationship.
    try:
        Settings(
            app_env="testing",
            research_candidate_k=5,
            final_k=10,
            reranker_final_k=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid candidate/final relationship was accepted."
        )

    # Invalid device.
    try:
        Settings(app_env="testing", specter2_device="tpu")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid device was accepted.")

    # Invalid Ollama structured-output mode must fail.
    try:
        Settings(
            app_env="testing",
            llm_ollama_structured_output_mode="xml",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid Ollama structured-output mode was accepted."
        )

    # Mode normalization should be case-insensitive.
    normalized_mode_test = Settings(
        app_env="testing",
        llm_ollama_structured_output_mode=" JSON ",
    )
    assert normalized_mode_test.llm_ollama_structured_output_mode == "json"

    # Production + debug must fail.
    try:
        Settings(app_env="production", debug=True)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "DEBUG=true was accepted in production."
        )

    # Import/configuration must not create directories.
    assert isinstance(test.research_index_base, Path)
    assert isinstance(test.upload_root, Path)

    # Secret must not be present in normal repr.
    assert "test-secret" not in repr(test)

    print("backend/config.py self-test: PASSED")


if __name__ == "__main__":
    run_self_test()