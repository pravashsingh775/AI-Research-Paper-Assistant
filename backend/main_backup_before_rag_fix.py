"""
Central FastAPI application entry point for the AI Research Paper Assistant.

Responsibilities
----------------
- configuration loading
- application creation
- dependency wiring
- lifecycle management
- router registration
- CORS
- exception boundaries
- graceful resource shutdown
- health/readiness contract

Business logic remains inside the existing backend services.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger("ai_research_paper_assistant")


# ============================================================================
# Application constants
# ============================================================================

DEFAULT_APP_NAME = "AI Research Paper Assistant"
DEFAULT_APP_VERSION = "0.1.0"

DEFAULT_RESEARCH_CANDIDATE_K = 50
DEFAULT_UPLOADED_CANDIDATE_K = 20
DEFAULT_UPLOADED_MAX_CANDIDATE_K = 100
DEFAULT_FINAL_K = 10

DEFAULT_RESEARCH_INDEX_BASE = (
    Path("indexes") / "research" / "index"
)

DEFAULT_UPLOADED_INDEX_ROOT = (
    Path("indexes") / "uploaded"
)


# ============================================================================
# IMPORTANT:
# Central readiness contract
# ============================================================================

# This must exist independently of the lifespan.
#
# Why?
# ----
# /health/ready must be able to correctly report:
#
#     application created but services not initialized -> 503
#
# rather than incorrectly returning:
#
#     200 because no required services were registered.
#
# `True`  = required for application readiness
# `False` = optional / informational only
#
# Keep this as the SINGLE source of truth.
HEALTH_REQUIRED_SERVICES: dict[str, bool] = {
    "research_service": True,
    "research_analyzer": True,
    "paper_indexing_service": True,
    "rag_pipeline": True,
    "llm_client": True,

    # The current paper_indexing.py implementation exposes the indexer,
    # not a complete upload orchestration pipeline.
    "paper_upload_pipeline": False,
}


# ============================================================================
# Configuration
# ============================================================================

def _load_project_settings() -> Any:
    """
    Resolve the existing backend configuration object.

    Preference:
        backend.config.get_settings()
        backend.config.Settings()

    Compatibility fallback:
        config.get_settings()
        config.Settings()
    """
    try:
        module = import_module("backend.config")
    except ImportError:
        try:
            module = import_module("config")
        except ImportError:
            return None

    getter = getattr(module, "get_settings", None)

    if callable(getter):
        return getter()

    settings_cls = getattr(module, "Settings", None)

    if settings_cls is not None:
        return settings_cls()

    return None


def _settings_mapping(settings: Any) -> dict[str, Any]:
    """Convert an existing settings object into a dictionary."""

    if settings is None:
        return {}

    if isinstance(settings, Mapping):
        return dict(settings)

    model_dump = getattr(settings, "model_dump", None)

    if callable(model_dump):
        try:
            value = model_dump()

            if isinstance(value, Mapping):
                return dict(value)

        except Exception:
            LOGGER.debug(
                "Settings model_dump() was unavailable.",
                exc_info=True,
            )

    as_dict = getattr(settings, "dict", None)

    if callable(as_dict):
        try:
            value = as_dict()

            if isinstance(value, Mapping):
                return dict(value)

        except Exception:
            LOGGER.debug(
                "Settings dict() was unavailable.",
                exc_info=True,
            )

    try:
        return {
            key: value
            for key, value in vars(settings).items()
            if not key.startswith("_")
        }

    except Exception:
        return {}


def _setting(
    mapping: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first meaningful configured value."""

    for name in names:

        if name not in mapping:
            continue

        value = mapping[name]

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return default


def _as_bool(
    value: Any,
    default: bool = False,
) -> bool:

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _as_int(
    value: Any,
    default: int,
) -> int:

    if value is None:
        return default

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def _as_float(
    value: Any,
    default: float,
) -> float:

    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _as_path(
    value: Any,
    default: Path,
) -> Path:

    if value is None:
        return default

    return Path(str(value))


def _cors_origins(
    settings: Mapping[str, Any],
) -> list[str]:
    """
    Resolve CORS configuration.

    Wildcard origins are intentionally not combined with credentials.
    """

    value = _setting(
        settings,
        "CORS_ORIGINS",
        "cors_origins",
        default=None,
    )

    if value is None:
        raw_env = os.getenv("CORS_ORIGINS")

        if raw_env:
            value = raw_env

    if isinstance(value, str):

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray),
    ):

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return []


# ============================================================================
# Generic dependency construction
# ============================================================================

def _constructor_parameters(
    factory: Callable[..., Any],
) -> inspect.Signature:

    try:
        return inspect.signature(factory)

    except (TypeError, ValueError) as exc:

        raise RuntimeError(
            f"Cannot inspect constructor signature for {factory!r}."
        ) from exc


def _construct_from_dependencies(
    factory: Callable[..., Any],
    dependencies: Mapping[str, Any],
    *,
    label: str,
    optional_values: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Construct a project component using its actual constructor signature.

    Required constructor arguments that cannot be resolved cause a clear
    startup error instead of silently constructing an invalid object.
    """

    signature = _constructor_parameters(factory)

    optional_values = dict(optional_values or {})

    kwargs: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():

        if name in {"self", "cls"}:
            continue

        if name in dependencies:
            kwargs[name] = dependencies[name]
            continue

        if name in optional_values:
            kwargs[name] = optional_values[name]
            continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        raise RuntimeError(
            f"{label} requires constructor dependency {name!r}, "
            "but main.py could not resolve it."
        )

    try:
        return factory(**kwargs)

    except Exception as exc:

        raise RuntimeError(
            f"Failed to initialize {label}."
        ) from exc


def _import_symbol(
    module_name: str,
    *names: str,
) -> Any:

    module = import_module(module_name)

    for name in names:

        symbol = getattr(module, name, None)

        if symbol is not None:
            return symbol

    raise RuntimeError(
        f"None of the expected symbols {names!r} "
        f"exists in {module_name!r}."
    )


def _find_analysis_component(
    module_name: str,
    component_name: str,
) -> Any:
    """
    Resolve an existing analysis component.

    Preference:
        exact conventional class name
        otherwise the only public class exposing analyze()
    """

    module = import_module(module_name)

    preferred_names = (
        f"{component_name}Analyzer",
        component_name,
    )

    for name in preferred_names:

        candidate = getattr(module, name, None)

        if inspect.isclass(candidate):
            return candidate

    candidates: list[type[Any]] = []

    for name, candidate in vars(module).items():

        if name.startswith("_"):
            continue

        if not inspect.isclass(candidate):
            continue

        if callable(getattr(candidate, "analyze", None)):
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:

        raise RuntimeError(
            f"No analysis component exposing analyze() "
            f"was found in {module_name!r}."
        )

    raise RuntimeError(
        f"Multiple analysis components were found in "
        f"{module_name!r}; explicit selection is required."
    )


# ============================================================================
# Service bootstrap
# ============================================================================

def _build_services(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build all long-lived application services exactly once.

    Expensive ML/FAISS/LLM resources are initialized here and never inside
    individual API requests.
    """

    # ------------------------------------------------------------------------
    # SPECTER2
    # ------------------------------------------------------------------------

    SPECTER2Embedder = _import_symbol(
        "backend.embeddings.specter2",
        "SPECTER2Embedder",
    )

    encoder = _construct_from_dependencies(
        SPECTER2Embedder,
        {},
        label="SPECTER2 embedding encoder",
        optional_values={
            "base_model": _setting(
                settings,
                "SPECTER2_BASE_MODEL",
                "specter2_base_model",
                default="allenai/specter2_base",
            ),
            "document_adapter": _setting(
                settings,
                "SPECTER2_DOCUMENT_ADAPTER",
                "specter2_document_adapter",
                default="allenai/specter2",
            ),
            "query_adapter": _setting(
                settings,
                "SPECTER2_QUERY_ADAPTER",
                "specter2_query_adapter",
                default="allenai/specter2_adhoc_query",
            ),
            "device": _setting(
                settings,
                "SPECTER2_DEVICE",
                "specter2_device",
                default="auto",
            ),
            "batch_size": _as_int(
                _setting(
                    settings,
                    "SPECTER2_BATCH_SIZE",
                    "specter2_batch_size",
                ),
                16,
            ),
            "max_length": _as_int(
                _setting(
                    settings,
                    "SPECTER2_MAX_LENGTH",
                    "specter2_max_length",
                ),
                512,
            ),
            "normalize": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_NORMALIZE",
                    "specter2_normalize",
                ),
                True,
            ),
            "normalization_tolerance": _as_float(
                _setting(
                    settings,
                    "SPECTER2_NORMALIZATION_TOLERANCE",
                    "specter2_normalization_tolerance",
                ),
                1e-4,
            ),
            "output_dtype": _setting(
                settings,
                "SPECTER2_OUTPUT_DTYPE",
                "specter2_output_dtype",
                default="float32",
            ),
            "cache_dir": _setting(
                settings,
                "SPECTER2_CACHE_DIR",
                "specter2_cache_dir",
            ),
            "local_files_only": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_LOCAL_FILES_ONLY",
                    "specter2_local_files_only",
                ),
                False,
            ),
            "trust_remote_code": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_TRUST_REMOTE_CODE",
                    "specter2_trust_remote_code",
                ),
                False,
            ),
            "show_progress": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_SHOW_PROGRESS",
                    "specter2_show_progress",
                ),
                False,
            ),
            "deterministic": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_DETERMINISTIC",
                    "specter2_deterministic",
                ),
                True,
            ),
            "use_autocast": _as_bool(
                _setting(
                    settings,
                    "SPECTER2_USE_AUTOCAST",
                    "specter2_use_autocast",
                ),
                False,
            ),
        },
    )

    # ------------------------------------------------------------------------
    # Research FAISS index
    # ------------------------------------------------------------------------

    FAISSVectorIndex = _import_symbol(
        "backend.retrieval.faiss_index",
        "FAISSVectorIndex",
    )

    research_index_base = _as_path(
        _setting(
            settings,
            "RESEARCH_INDEX_PATH",
            "research_index_path",
            "RESEARCH_INDEX_BASE",
            "research_index_base",
        ),
        DEFAULT_RESEARCH_INDEX_BASE,
    )

    index_file = Path(f"{research_index_base}.index")
    manifest_file = Path(
        f"{research_index_base}.index.manifest.json"
    )

    if not index_file.is_file() or not manifest_file.is_file():

        raise FileNotFoundError(
            "Research FAISS index is not ready. Expected:\n"
            f"  {index_file}\n"
            f"  {manifest_file}"
        )

    model_name = getattr(
        encoder,
        "model_name",
        None,
    )

    model_version = getattr(
        encoder,
        "model_version",
        None,
    )

    try:

        research_index = FAISSVectorIndex.load(
            research_index_base,
            expected_model_name=model_name,
            expected_model_version=model_version,
        )

    except TypeError:

        # Backward compatibility with older FAISSVectorIndex.load().
        research_index = FAISSVectorIndex.load(
            research_index_base,
        )

    # ------------------------------------------------------------------------
    # Dense retriever
    # ------------------------------------------------------------------------

    DenseRetriever = _import_symbol(
        "backend.retrieval.retriever",
        "DenseRetriever",
        "Retriever",
    )

    candidate_k = _as_int(
        _setting(
            settings,
            "RESEARCH_CANDIDATE_K",
            "candidate_k",
            "DEFAULT_CANDIDATE_K",
        ),
        DEFAULT_RESEARCH_CANDIDATE_K,
    )

    max_candidate_k_value = _setting(
        settings,
        "RESEARCH_MAX_CANDIDATE_K",
        "max_candidate_k",
    )

    max_candidate_k = None

    if max_candidate_k_value is not None:

        max_candidate_k = _as_int(
            max_candidate_k_value,
            candidate_k,
        )

    dense_retriever = DenseRetriever(
        encoder,
        research_index,
        default_candidate_k=candidate_k,
        max_candidate_k=max_candidate_k,
        raise_on_empty_index=True,
    )

    # ------------------------------------------------------------------------
    # Uploaded-paper retriever
    # ------------------------------------------------------------------------

    UploadedPaperRetriever = _import_symbol(
        "backend.retrieval.retriever",
        "UploadedPaperRetriever",
    )

    uploaded_root = _as_path(
        _setting(
            settings,
            "UPLOADED_INDEX_ROOT",
            "uploaded_index_root",
        ),
        DEFAULT_UPLOADED_INDEX_ROOT,
    )

    uploaded_retriever = UploadedPaperRetriever(
        encoder,
        uploaded_root=uploaded_root,
        candidate_k=_as_int(
            _setting(
                settings,
                "UPLOADED_CANDIDATE_K",
                "uploaded_candidate_k",
            ),
            DEFAULT_UPLOADED_CANDIDATE_K,
        ),
        max_candidate_k=_as_int(
            _setting(
                settings,
                "UPLOADED_MAX_CANDIDATE_K",
                "uploaded_max_candidate_k",
            ),
            DEFAULT_UPLOADED_MAX_CANDIDATE_K,
        ),
        include_text=True,
    )

    # ------------------------------------------------------------------------
    # Paper retriever
    # ------------------------------------------------------------------------

    PaperRetriever = _import_symbol(
        "backend.retrieval.retriever",
        "PaperRetriever",
    )

    paper_retriever = PaperRetriever(
        research_retriever=dense_retriever,
        uploaded_retriever=uploaded_retriever,
    )

    # ------------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------------

    CrossEncoderReranker = _import_symbol(
        "backend.retrieval.reranker",
        "CrossEncoderReranker",
    )

    reranker_module = import_module(
        "backend.retrieval.reranker"
    )

    default_reranker_model = getattr(
        reranker_module,
        "DEFAULT_MODEL_NAME",
        None,
    )

    reranker_kwargs: dict[str, Any] = {
        "device": _setting(
            settings,
            "RERANKER_DEVICE",
            "reranker_device",
            default="auto",
        ),
        "batch_size": _as_int(
            _setting(
                settings,
                "RERANKER_BATCH_SIZE",
                "reranker_batch_size",
            ),
            16,
        ),
        "max_length": _as_int(
            _setting(
                settings,
                "RERANKER_MAX_LENGTH",
                "reranker_max_length",
            ),
            512,
        ),
        "final_k": _as_int(
            _setting(
                settings,
                "RERANKER_FINAL_K",
                "final_k",
            ),
            DEFAULT_FINAL_K,
        ),
        "cache_dir": _setting(
            settings,
            "RERANKER_CACHE_DIR",
            "reranker_cache_dir",
        ),
        "revision": _setting(
            settings,
            "RERANKER_REVISION",
            "reranker_revision",
        ),
        "normalize_score": _as_bool(
            _setting(
                settings,
                "RERANKER_NORMALIZE_SCORE",
                "reranker_normalize_score",
            ),
            False,
        ),
    }

    configured_reranker_model = _setting(
        settings,
        "RERANKER_MODEL_NAME",
        "reranker_model_name",
    )

    if configured_reranker_model is not None:

        reranker_kwargs["model_name"] = (
            configured_reranker_model
        )

    elif default_reranker_model is not None:

        reranker_kwargs["model_name"] = (
            default_reranker_model
        )

    reranker = CrossEncoderReranker(
        **reranker_kwargs
    )

    # ------------------------------------------------------------------------
    # Final ranker
    # ------------------------------------------------------------------------

    Ranker = _import_symbol(
        "backend.retrieval.ranking",
        "Ranker",
    )

    ranker = Ranker(
        top_k=_as_int(
            _setting(
                settings,
                "FINAL_K",
                "final_k",
            ),
            DEFAULT_FINAL_K,
        )
    )

    # ------------------------------------------------------------------------
    # RAG retriever
    # ------------------------------------------------------------------------

    RAGRetriever = _import_symbol(
        "backend.rag.retriever",
        "RAGRetriever",
    )

    # Candidate enrichment is intentionally not fabricated here.
    candidate_enricher = None

    rag_retriever = RAGRetriever(
        retriever=paper_retriever,
        reranker=reranker,
        ranker=ranker,
        candidate_enricher=candidate_enricher,
    )

    # ------------------------------------------------------------------------
    # Context + prompt
    # ------------------------------------------------------------------------

    ContextBuilder = _import_symbol(
        "backend.rag.context",
        "ContextBuilder",
    )

    PromptBuilder = _import_symbol(
        "backend.rag.prompts",
        "PromptBuilder",
    )

    context_builder = ContextBuilder(
        max_characters=_setting(
            settings,
            "RAG_CONTEXT_MAX_CHARACTERS",
            "rag_context_max_characters",
            default=None,
        ),
        max_evidence_items=_setting(
            settings,
            "RAG_CONTEXT_MAX_EVIDENCE_ITEMS",
            "rag_context_max_evidence_items",
            default=None,
        ),
    )

    prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------------

    from backend.llm.client import (
        create_client_from_mapping,
    )

    llm_client = create_client_from_mapping(
        settings
    )

    # ------------------------------------------------------------------------
    # LLM parser + validation
    # ------------------------------------------------------------------------

    LLMOutputParser = _import_symbol(
        "backend.llm.parser",
        "LLMOutputParser",
    )

    EvidenceValidator = _import_symbol(
        "backend.validation.evidence",
        "EvidenceValidator",
    )

    ResponseValidator = _import_symbol(
        "backend.validation.validator",
        "ResponseValidator",
    )

    output_parser = LLMOutputParser()

    evidence_validator = EvidenceValidator()

    response_validator = ResponseValidator(
        evidence_validator=evidence_validator,
    )

    # ------------------------------------------------------------------------
    # Central RAG pipeline
    # ------------------------------------------------------------------------

    RAGPipeline = _import_symbol(
        "backend.rag.pipeline",
        "RAGPipeline",
    )

    rag_pipeline = RAGPipeline(
        rag_retriever=rag_retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        output_parser=output_parser,
        evidence_validator=evidence_validator,
        validator=response_validator,
        default_candidate_k=candidate_k,
        default_final_k=_as_int(
            _setting(
                settings,
                "FINAL_K",
                "final_k",
            ),
            DEFAULT_FINAL_K,
        ),
        default_context_max_characters=_setting(
            settings,
            "RAG_CONTEXT_MAX_CHARACTERS",
            "rag_context_max_characters",
            default=None,
        ),
        default_context_max_evidence_items=_setting(
            settings,
            "RAG_CONTEXT_MAX_EVIDENCE_ITEMS",
            "rag_context_max_evidence_items",
            default=None,
        ),
        system_configuration={},
    )

    # ------------------------------------------------------------------------
    # Research search service
    # ------------------------------------------------------------------------

    ResearchSearchService = _import_symbol(
        "backend.research.search",
        "ResearchSearchService",
    )

    research_service = ResearchSearchService(
        retriever=dense_retriever,
        reranker=reranker,
        ranker=ranker,
    )

    # ------------------------------------------------------------------------
    # Research analysis
    # ------------------------------------------------------------------------

    analyzer_dependencies: dict[str, Any] = {}

    analysis_module_map = {
        "summary_analyzer": "backend.analysis.summary",
        "model_analyzer": "backend.analysis.model",
        "dataset_analyzer": "backend.analysis.dataset",
        "methodology_analyzer": "backend.analysis.methodology",
        "findings_analyzer": "backend.analysis.findings",
        "strengths_analyzer": "backend.analysis.strengths",
        "weaknesses_analyzer": "backend.analysis.weaknesses",
    }

    for dependency_name, module_name in (
        analysis_module_map.items()
    ):

        component_name = dependency_name.removesuffix(
            "_analyzer"
        )

        component_class = _find_analysis_component(
            module_name,
            component_name,
        )

        analyzer_dependencies[
            dependency_name
        ] = component_class()

    ResearchPaperAnalyzer = _import_symbol(
        "backend.research.analyzer",
        "ResearchPaperAnalyzer",
    )

    research_analyzer = ResearchPaperAnalyzer(
        **analyzer_dependencies,
        analysis_mode=_setting(
            settings,
            "RESEARCH_ANALYSIS_MODE",
            "analysis_mode",
            default="abstract",
        ),
    )

    # ------------------------------------------------------------------------
    # Uploaded-paper indexing service
    # ------------------------------------------------------------------------

    UploadedPaperIndexer = _import_symbol(
        "backend.services.paper_indexing",
        "UploadedPaperIndexer",
    )

    UploadedPaperIndexConfig = _import_symbol(
        "backend.services.paper_indexing",
        "UploadedPaperIndexConfig",
    )

    paper_indexing_service = UploadedPaperIndexer(
        encoder=encoder,
        config=UploadedPaperIndexConfig(
            root_dir=uploaded_root,
            overwrite=False,
            include_chunk_text=True,
            verify_reload=True,
        ),
        faiss_index_cls=FAISSVectorIndex,
    )

    # ------------------------------------------------------------------------
    # Optional upload pipeline
    # ------------------------------------------------------------------------

    paper_upload_pipeline = None

    try:

        paper_module = import_module(
            "backend.services.paper_indexing"
        )

        pipeline_cls = getattr(
            paper_module,
            "PaperUploadPipeline",
            None,
        )

        if pipeline_cls is not None:

            paper_upload_pipeline = pipeline_cls()

        else:

            getter = getattr(
                paper_module,
                "get_paper_pipeline",
                None,
            )

            if callable(getter):
                paper_upload_pipeline = getter()

    except Exception:

        LOGGER.warning(
            "No complete PaperUploadPipeline could be initialized. "
            "Mode-2 upload API may require dedicated upload "
            "pipeline integration.",
            exc_info=True,
        )

    return {
        "encoder": encoder,
        "research_index": research_index,
        "research_retriever": dense_retriever,
        "uploaded_retriever": uploaded_retriever,
        "paper_retriever": paper_retriever,
        "reranker": reranker,
        "ranker": ranker,
        "rag_retriever": rag_retriever,
        "context_builder": context_builder,
        "prompt_builder": prompt_builder,
        "llm_client": llm_client,
        "output_parser": output_parser,
        "evidence_validator": evidence_validator,
        "validator": response_validator,
        "rag_pipeline": rag_pipeline,
        "research_service": research_service,
        "research_analyzer": research_analyzer,
        "paper_indexing_service": paper_indexing_service,
        "paper_upload_pipeline": paper_upload_pipeline,
    }


# ============================================================================
# Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all long-lived services exactly once.

    Startup is fail-fast for core services.

    Shutdown attempts graceful close() operations without hiding the
    application lifecycle state.
    """

    settings = _load_project_settings()

    settings_map = _settings_mapping(settings)

    app.state.settings = settings
    app.state.config = settings_map

    app.state.service_name = _setting(
        settings_map,
        "APP_NAME",
        "app_name",
        "SERVICE_NAME",
        "service_name",
        default=DEFAULT_APP_NAME,
    )

    # Keep the centralized readiness contract available throughout
    # the entire application lifecycle.
    app.state.health_required_services = dict(
        HEALTH_REQUIRED_SERVICES
    )

    LOGGER.info(
        "Starting %s.",
        app.state.service_name,
    )

    services: dict[str, Any] = {}

    try:

        services = _build_services(
            settings_map
        )

        for name, service in services.items():

            setattr(
                app.state,
                name,
                service,
            )

        LOGGER.info(
            "Application startup completed. "
            "Core services initialized."
        )

        yield

    except Exception:

        LOGGER.exception(
            "Application startup failed."
        )

        raise

    finally:

        LOGGER.info(
            "Shutting down application services."
        )

        seen: set[int] = set()

        for name, service in services.items():

            if service is None:
                continue

            identity = id(service)

            if identity in seen:
                continue

            seen.add(identity)

            close = getattr(
                service,
                "close",
                None,
            )

            if not callable(close):
                continue

            try:

                result = close()

                if inspect.isawaitable(result):
                    await result

            except Exception:

                LOGGER.warning(
                    "Graceful close failed for service=%s.",
                    name,
                    exc_info=True,
                )

        LOGGER.info(
            "Application shutdown completed."
        )


# ============================================================================
# Application factory
# ============================================================================

def create_app(
    *,
    initialize_services: bool = True,
) -> FastAPI:
    """
    Create the single FastAPI application.

    Parameters
    ----------
    initialize_services:
        True for normal production execution.

        False for lightweight tests/import/OpenAPI checks.

    Important
    ---------
    The readiness contract is registered regardless of this flag.
    Therefore:

        initialize_services=False
            -> /health       = 200
            -> /health/ready = 503

    until actual application services are initialized.
    """

    settings = _load_project_settings()

    settings_map = _settings_mapping(
        settings
    )

    app = FastAPI(
        title=str(
            _setting(
                settings_map,
                "APP_NAME",
                "app_name",
                default=DEFAULT_APP_NAME,
            )
        ),
        version=str(
            _setting(
                settings_map,
                "APP_VERSION",
                "app_version",
                "VERSION",
                "version",
                default=DEFAULT_APP_VERSION,
            )
        ),
        description=(
            "AI Research Paper Assistant backend API. "
            "Research discovery and uploaded-paper analysis."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=(
            lifespan
            if initialize_services
            else None
        ),
    )

    # ------------------------------------------------------------------------
    # Application state
    # ------------------------------------------------------------------------

    app.state.settings = settings
    app.state.config = settings_map

    app.state.service_name = _setting(
        settings_map,
        "APP_NAME",
        "app_name",
        "SERVICE_NAME",
        "service_name",
        default=DEFAULT_APP_NAME,
    )

    # CRITICAL FIX:
    #
    # The readiness contract must exist even when lifespan is disabled.
    #
    # This makes:
    #
    #     create_app(initialize_services=False)
    #                  ↓
    #     /health/ready
    #                  ↓
    #               503
    #
    # rather than incorrectly returning 200.
    app.state.health_required_services = dict(
        HEALTH_REQUIRED_SERVICES
    )

    # ------------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------------

    origins = _cors_origins(
        settings_map
    )

    if origins:

        # Never combine wildcard origin with credentials.
        if "*" in origins:

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=[
                    "GET",
                    "POST",
                    "PUT",
                    "PATCH",
                    "DELETE",
                    "OPTIONS",
                ],
                allow_headers=["*"],
            )

        else:

            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=[
                    "GET",
                    "POST",
                    "PUT",
                    "PATCH",
                    "DELETE",
                    "OPTIONS",
                ],
                allow_headers=["*"],
            )

    # ------------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------------

    from backend.api import (
        health,
        papers,
        qa,
        research,
    )

    app.include_router(
        health.router
    )

    app.include_router(
        research.router
    )

    app.include_router(
        papers.router
    )

    app.include_router(
        qa.router
    )

    # ------------------------------------------------------------------------
    # Exception boundaries
    # ------------------------------------------------------------------------

    @app.exception_handler(
        RequestValidationError
    )
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:

        LOGGER.warning(
            "Request validation failed: "
            "method=%s path=%s",
            request.method,
            request.url.path,
        )

        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed.",
            },
        )

    @app.exception_handler(
        StarletteHTTPException
    )
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:

        LOGGER.exception(
            "Unhandled application exception: "
            "method=%s path=%s",
            request.method,
            request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error."
            },
        )

    return app


# ============================================================================
# ASGI application
# ============================================================================

app = create_app()


# ============================================================================
# Lightweight contract self-test
# ============================================================================

def run_self_test() -> None:
    """
    Validate application creation and router registration without loading:

        - SPECTER2
        - Transformers
        - FAISS
        - CrossEncoder
        - LLM
        - PDF processing
        - production services
    """

    from fastapi.testclient import TestClient

    test_app = create_app(
        initialize_services=False
    )

    client = TestClient(
        test_app
    )

    # ------------------------------------------------------------------------
    # OpenAPI
    # ------------------------------------------------------------------------

    openapi = client.get(
        "/openapi.json"
    )

    assert openapi.status_code == 200, (
        openapi.text
    )

    paths = openapi.json()["paths"]

    expected_routes = (
        "/health",
        "/health/ready",
        "/research/search",
        "/api/qa",
        "/api/papers/upload",
    )

    for route in expected_routes:

        assert route in paths, (
            f"Missing route: {route}"
        )

    # ------------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------------

    health_response = client.get(
        "/health"
    )

    assert health_response.status_code == 200

    health_payload = (
        health_response.json()
    )

    assert health_payload["status"] == (
        "healthy"
    )

    # ------------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------------

    # No lifespan means no production services exist.
    #
    # Because the readiness contract is registered in create_app(),
    # this MUST return 503.
    readiness = client.get(
        "/health/ready"
    )

    assert readiness.status_code == 503, (
        readiness.text
    )

    readiness_payload = (
        readiness.json()
    )

    assert readiness_payload["status"] == (
        "not_ready"
    )

    checks = readiness_payload.get(
        "checks",
        {},
    )

    # Required services must be reported unavailable.
    for service_name, required in (
        HEALTH_REQUIRED_SERVICES.items()
    ):

        if required:

            assert checks.get(
                service_name
            ) == "unavailable", (
                f"{service_name} readiness contract failed: "
                f"{checks!r}"
            )

    # ------------------------------------------------------------------------
    # Security contract
    # ------------------------------------------------------------------------

    serialized = str(
        readiness_payload
    ).lower()

    forbidden_markers = (
        "api_key",
        "password",
        "secret",
        "token",
        "c:\\users",
        "/users/",
    )

    assert not any(
        marker in serialized
        for marker in forbidden_markers
    ), (
        "Health endpoint exposed a forbidden "
        "secret/path marker."
    )

    print(
        "backend/main.py self-test: PASSED"
    )


# ============================================================================
# Direct execution
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    run_self_test()
