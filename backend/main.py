"""
Central FastAPI application entry point for the AI Research Paper Assistant.

main.py is intentionally limited to:
    configuration
    application/lifecycle management
    dependency wiring
    router registration
    CORS
    exception boundaries
    graceful resource shutdown

Business logic remains in the existing backend services.
"""

from __future__ import annotations

import inspect
import json
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

LOGGER = logging.getLogger("ai_research_paper_assistant")

MAIN_WIRING_VERSION = "2.4.0"


class _FunctionBackedUploadedPaperIndexer:
    """
    Compatibility adapter for projects whose paper_indexing.py exposes the
    functional build_uploaded_paper_index() API but not the newer
    UploadedPaperIndexer class.

    This adapter is intentionally limited to Mode-2 and does not touch the
    Mode-1 research index.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        root_dir: Path,
        overwrite: bool = False,
        include_chunk_text: bool = True,
        verify_reload: bool = True,
    ) -> None:
        if encoder is None:
            raise RuntimeError("Mode-2 encoder cannot be None.")

        self.encoder = encoder
        self.root_dir = Path(root_dir)
        self.overwrite = bool(overwrite)
        self.include_chunk_text = bool(include_chunk_text)
        self.verify_reload = bool(verify_reload)

        module = import_module("backend.services.paper_indexing")
        builder = getattr(module, "build_uploaded_paper_index", None)
        if not callable(builder):
            raise RuntimeError(
                "backend.services.paper_indexing exposes neither "
                "UploadedPaperIndexer nor build_uploaded_paper_index()."
            )
        self._builder = builder

    def build(
        self,
        *,
        document_id: str,
        chunks: Sequence[Any],
        source_document_id: Optional[str] = None,
    ) -> Any:
        """
        Call the legacy functional builder while preserving the current
        PaperUploadPipeline contract.
        """
        kwargs: dict[str, Any] = {
            "document_id": document_id,
            "chunks": chunks,
            "encoder": self.encoder,
            "root_dir": self.root_dir,
            "overwrite": self.overwrite,
            "include_chunk_text": self.include_chunk_text,
            "verify_reload": self.verify_reload,
        }

        if source_document_id is not None:
            kwargs["source_document_id"] = source_document_id

        # Only pass arguments supported by the installed project version.
        # This keeps the adapter compatible with both older and newer
        # paper_indexing.py signatures.
        try:
            signature = inspect.signature(self._builder)
            accepted = set(signature.parameters)
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in accepted
            }
        except (TypeError, ValueError):
            # Some callables do not expose a signature. In that case the
            # canonical keyword contract is used directly.
            pass

        return self._builder(**kwargs)


# ============================================================================
# Project-path and persisted-index integrity helpers
# ============================================================================

# main.py lives in <project_root>/backend/main.py.  Resolve relative storage
# paths from the project root rather than from the process working directory.
# This prevents `uvicorn`/IDE/test runners from accidentally reading a second
# indexes/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: Any, default: Path) -> Path:
    """Resolve a configured path deterministically against PROJECT_ROOT."""
    if value is None:
        return default

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _json_object(path: Path) -> Mapping[str, Any]:
    """Read one JSON object with a clear startup error."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON sidecar {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON sidecar must contain an object: {path}")
    return payload


def _validate_uploaded_index_root(root: Path) -> dict[str, Any]:
    """
    Validate persisted Mode-2 indexes without modifying them.

    Important compatibility rule:
    `index.index.manifest.json["document_ids"]` in the current uploaded-index
    format contains VECTOR/CHUNK identifiers, not the paper's folder-level
    document_id.  Therefore it must NOT be compared directly with the folder
    document_id.  The authoritative paper ID is metadata.json["document_id"].
    """
    root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Uploaded-paper index root does not exist: {root}"
        )

    papers = [
        item for item in root.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]

    validated = 0
    warnings: list[str] = []

    for paper_dir in sorted(papers, key=lambda p: p.name):
        index_path = paper_dir / "index.index"
        manifest_path = paper_dir / "index.index.manifest.json"
        metadata_path = paper_dir / "metadata.json"

        missing = [
            str(path.name)
            for path in (index_path, manifest_path, metadata_path)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Uploaded-paper index {paper_dir.name!r} is incomplete; "
                f"missing: {', '.join(missing)}."
            )

        metadata = _json_object(metadata_path)
        manifest = _json_object(manifest_path)

        document_id = metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise RuntimeError(
                f"Uploaded-paper metadata has no valid document_id: {metadata_path}"
            )

        if document_id != paper_dir.name:
            raise RuntimeError(
                "Uploaded-paper index directory/document_id mismatch: "
                f"directory={paper_dir.name!r}, metadata={document_id!r}."
            )

        chunks = metadata.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError(
                f"Uploaded-paper metadata has no valid chunks array: {metadata_path}"
            )

        vector_count = metadata.get("vector_count")
        if not isinstance(vector_count, int) or vector_count < 0:
            raise RuntimeError(
                f"Uploaded-paper metadata has invalid vector_count: {metadata_path}"
            )

        manifest_ids = manifest.get("document_ids")
        if not isinstance(manifest_ids, list):
            raise RuntimeError(
                f"Uploaded-paper manifest has no document_ids array: {manifest_path}"
            )

        # Current schema stores chunk IDs in the manifest's vector-ID array.
        chunk_ids = {
            chunk.get("chunk_id")
            for chunk in chunks
            if isinstance(chunk, Mapping) and isinstance(chunk.get("chunk_id"), str)
        }

        if manifest_ids and not all(
            isinstance(item, str) and item in chunk_ids
            for item in manifest_ids
        ):
            # Legacy indexes may use the paper ID itself. Accept that only when
            # it is explicit; otherwise reject an ambiguous/corrupt mapping.
            if not all(item == document_id for item in manifest_ids):
                raise RuntimeError(
                    "Uploaded-paper manifest contains IDs that are neither "
                    f"known chunk IDs nor the paper document_id: {manifest_path}"
                )

        if len(chunks) != vector_count:
            raise RuntimeError(
                "Uploaded-paper metadata vector_count/chunk count mismatch: "
                f"document_id={document_id!r}, "
                f"vector_count={vector_count}, chunks={len(chunks)}."
            )

        manifest_count = manifest.get("document_count")
        if isinstance(manifest_count, int) and manifest_count != vector_count:
            raise RuntimeError(
                "Uploaded-paper manifest vector count mismatch: "
                f"document_id={document_id!r}, "
                f"manifest={manifest_count}, metadata={vector_count}."
            )

        # FAISS is optional during lightweight import/self-test. In production
        # startup it is already a required dependency, so verify the persisted
        # index really contains the number of vectors advertised by metadata.
        try:
            import faiss  # type: ignore

            faiss_index = faiss.read_index(str(index_path))
            if int(faiss_index.ntotal) != vector_count:
                raise RuntimeError(
                    "Uploaded-paper FAISS/vector_count mismatch: "
                    f"document_id={document_id!r}, "
                    f"faiss={faiss_index.ntotal}, metadata={vector_count}."
                )

            expected_dimension = manifest.get("embedding_dimension")
            if (
                isinstance(expected_dimension, int)
                and expected_dimension > 0
                and int(faiss_index.d) != expected_dimension
            ):
                raise RuntimeError(
                    "Uploaded-paper FAISS dimension mismatch: "
                    f"document_id={document_id!r}, "
                    f"faiss={faiss_index.d}, manifest={expected_dimension}."
                )
        except ImportError:
            warnings.append("FAISS validation skipped because faiss is unavailable.")

        validated += 1

    return {
        "root": str(root),
        "indexed_papers": validated,
        "warnings": warnings,
    }


# ============================================================================
# Configuration
# ============================================================================

def _load_project_settings() -> Any:
    """
    Resolve the existing backend configuration object.

    No second configuration class is introduced. If config.py exposes a
    get_settings() factory it is preferred; otherwise Settings() is used.
    """
    try:
        module = import_module("backend.config")
    except ImportError:
        try:
            module = import_module("config")
        except ImportError:
            # Allows import/openapi contract tests in an environment where the
            # project package is not mounted. Production startup still fails
            # clearly when service initialization is attempted.
            return None

    getter = getattr(module, "get_settings", None)
    if callable(getter):
        return getter()

    settings_cls = getattr(module, "Settings", None)
    if settings_cls is not None:
        return settings_cls()

    return None


def _settings_mapping(settings: Any) -> dict[str, Any]:
    """Convert application Settings into the canonical runtime mapping.

    Generic application settings are copied first.  If the project Settings
    object exposes ``as_llm_mapping()``, its canonical LLM keys are then
    merged explicitly.  This preserves the configuration contract owned by
    ``backend.config`` and prevents local Ollama settings from being lost
    before ``create_client_from_mapping()`` is called.
    """
    if settings is None:
        return {}

    if isinstance(settings, Mapping):
        result: dict[str, Any] = dict(settings)
    else:
        result = {}

        model_dump = getattr(settings, "model_dump", None)
        if callable(model_dump):
            try:
                value = model_dump()
                if isinstance(value, Mapping):
                    result.update(value)
            except Exception:
                LOGGER.debug("Settings model_dump() was unavailable.", exc_info=True)

        if not result:
            as_dict = getattr(settings, "dict", None)
            if callable(as_dict):
                try:
                    value = as_dict()
                    if isinstance(value, Mapping):
                        result.update(value)
                except Exception:
                    LOGGER.debug("Settings dict() was unavailable.", exc_info=True)

        if not result:
            try:
                result.update({
                    key: value
                    for key, value in vars(settings).items()
                    if not key.startswith("_")
                })
            except Exception:
                LOGGER.debug("Could not convert Settings object to mapping.", exc_info=True)

    # Canonical LLM configuration is authoritative over generic aliases.
    as_llm_mapping = getattr(settings, "as_llm_mapping", None)
    if callable(as_llm_mapping):
        try:
            llm_mapping = as_llm_mapping()
        except Exception:
            LOGGER.exception("Failed to resolve canonical LLM configuration.")
            raise
        if not isinstance(llm_mapping, Mapping):
            raise TypeError("Settings.as_llm_mapping() must return a mapping.")
        result.update(llm_mapping)

    return result



def _setting(
    mapping: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first meaningful configured value.

    Empty/whitespace-only strings are treated as missing. This is important
    for environment/configuration overrides because an empty environment
    variable must never override a valid application default.
    """
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_path(value: Any, default: Path) -> Path:
    return _project_path(value, default)


def _cors_origins(settings: Mapping[str, Any]) -> list[str]:
    """
    Resolve CORS from existing configuration.

    An empty list means no cross-origin origins are explicitly enabled.
    Wildcard + credentials is deliberately not enabled.
    """
    value = _setting(settings, "CORS_ORIGINS", "cors_origins", default=None)

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

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return []


# ============================================================================
# Generic dependency construction
# ============================================================================

def _constructor_parameters(factory: Callable[..., Any]) -> inspect.Signature:
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
    Construct a project component from its actual callable signature.

    This is application wiring, not business logic. It prevents main.py from
    guessing constructor arguments and fails loudly when an existing component
    exposes an incompatible interface.
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
            "but main.py could not resolve it from the existing application "
            "components/configuration."
        )

    try:
        return factory(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize {label}."
        ) from exc


def _import_symbol(module_name: str, *names: str) -> Any:
    module = import_module(module_name)
    for name in names:
        symbol = getattr(module, name, None)
        if symbol is not None:
            return symbol

    raise RuntimeError(
        f"None of the expected symbols {names!r} exists in {module_name!r}."
    )


def _find_analysis_component(module_name: str, component_name: str) -> Any:
    """
    Resolve the canonical analysis component from an existing module.

    Resolution is deterministic and alias-safe:
      1. Prefer the conventional ``<component>Analyzer`` / ``<component>``
         names.
      2. Otherwise inspect public classes exposing ``analyze()``.
      3. Deduplicate aliases that point to the same class object before
         deciding whether the module is ambiguous.

    This matters for modules such as ``backend.analysis.findings`` where
    ``ResearchFindingsAnalyzer`` is intentionally an alias of
    ``FindingsAnalyzer``. Counting both names as separate implementations
    would incorrectly abort application startup.
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

    # Keep class identity, not public attribute name, as the uniqueness key.
    # A module may intentionally expose multiple aliases for one analyzer.
    candidates_by_identity: dict[int, type[Any]] = {}

    for name, candidate in vars(module).items():
        if name.startswith("_") or not inspect.isclass(candidate):
            continue
        if callable(getattr(candidate, "analyze", None)):
            candidates_by_identity.setdefault(id(candidate), candidate)

    candidates = list(candidates_by_identity.values())

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            f"No analysis component exposing analyze() was found in "
            f"{module_name!r}."
        )

    candidate_names = ", ".join(
        sorted(getattr(candidate, "__name__", repr(candidate)) for candidate in candidates)
    )
    raise RuntimeError(
        f"Multiple distinct analysis components were found in {module_name!r}: "
        f"{candidate_names}. Configure an explicit application-level selection."
    )


# ============================================================================
# Service bootstrap
# ============================================================================

def _build_services(settings: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build the long-lived application services once.

    Expensive resources are created here, never inside API requests.
    """
    # ---- Embedding ---------------------------------------------------------
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

    # ---- Research FAISS ---------------------------------------------------
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
        PROJECT_ROOT / "indexes" / "research" / "index",
    )

    if not (
        Path(f"{research_index_base}.index").is_file()
        and Path(f"{research_index_base}.index.manifest.json").is_file()
    ):
        raise FileNotFoundError(
            "Research FAISS index is not ready. Expected the configured "
            "index base plus .index and .index.manifest.json."
        )

    model_name = getattr(encoder, "model_name", None)
    model_version = getattr(encoder, "model_version", None)

    try:
        research_index = FAISSVectorIndex.load(
            research_index_base,
            expected_model_name=model_name,
            expected_model_version=model_version,
        )
    except TypeError:
        # Compatibility with the earlier load signature.
        research_index = FAISSVectorIndex.load(research_index_base)

    # ---- Research metadata sidecar -----------------------------------------
    # The search service must hydrate FAISS IDs with semantic metadata before
    # reranking. Keep this as a long-lived service so the sidecar is parsed once
    # and automatically reloaded when its mtime changes.
    try:
        ResearchMetadataStore = _import_symbol(
            "backend.retrieval.metadata_store",
            "ResearchMetadataStore",
        )
        metadata_store_module_name = "backend.retrieval.metadata_store"
    except (ImportError, RuntimeError):
        # Compatibility with the current project filename `metadata_storee.py`.
        # Prefer the canonical module whenever it exists.
        ResearchMetadataStore = _import_symbol(
            "backend.retrieval.metadata_storee",
            "ResearchMetadataStore",
        )
        metadata_store_module_name = "backend.retrieval.metadata_storee"

    research_metadata_path = _as_path(
        _setting(
            settings,
            "RESEARCH_METADATA_PATH",
            "research_metadata_path",
            "RESEARCH_METADATA_FILE",
            "research_metadata_file",
        ),
        research_index_base.parent / "metadata.json",
    )

    if not research_metadata_path.is_file():
        raise FileNotFoundError(
            "Research metadata sidecar is not ready. Expected the configured "
            f"metadata file: {research_metadata_path}"
        )

    metadata_store = ResearchMetadataStore(
        research_metadata_path,
        expected_document_count=int(getattr(research_index, "ntotal", 0)),
        strict_count=True,
    )
    # Force validation during startup rather than discovering corruption on the
    # first user request.
    metadata_store.validate()

    # ---- Dense research retriever -----------------------------------------
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
        50,
    )
    max_candidate_k = _setting(
        settings,
        "RESEARCH_MAX_CANDIDATE_K",
        "max_candidate_k",
    )
    if max_candidate_k is not None:
        max_candidate_k = _as_int(max_candidate_k, candidate_k)

    dense_retriever = DenseRetriever(
        encoder,
        research_index,
        default_candidate_k=candidate_k,
        max_candidate_k=max_candidate_k,
        raise_on_empty_index=True,
    )

    # ---- Uploaded-paper retriever -----------------------------------------
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
        PROJECT_ROOT / "indexes" / "uploaded",
    )

    uploaded_index_health = _validate_uploaded_index_root(uploaded_root)
    LOGGER.info(
        "Validated uploaded-paper indexes: root=%s papers=%d",
        uploaded_index_health["root"],
        uploaded_index_health["indexed_papers"],
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
            20,
        ),
        max_candidate_k=_as_int(
            _setting(
                settings,
                "UPLOADED_MAX_CANDIDATE_K",
                "uploaded_max_candidate_k",
            ),
            100,
        ),
        include_text=True,
    )

    # Existing PaperRetriever explicitly separates Mode 1 and Mode 2.
    PaperRetriever = _import_symbol(
        "backend.retrieval.retriever",
        "PaperRetriever",
    )

    paper_retriever = PaperRetriever(
        research_retriever=dense_retriever,
        uploaded_retriever=uploaded_retriever,
    )

    # ---- Reranker + final ranker ------------------------------------------
    CrossEncoderReranker = _import_symbol(
        "backend.retrieval.reranker",
        "CrossEncoderReranker",
    )
    reranker_module = import_module("backend.retrieval.reranker")
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
            10,
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
        reranker_kwargs["model_name"] = configured_reranker_model
    elif default_reranker_model is not None:
        reranker_kwargs["model_name"] = default_reranker_model

    reranker = CrossEncoderReranker(**reranker_kwargs)

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
            10,
        ),
    )

    # ---- Research search service -----------------------------------------
    # Build this before RAGRetriever because the current RAG retriever accepts
    # the complete research search service as its authoritative Mode-1 path.
    ResearchSearchService = _import_symbol(
        "backend.research.search",
        "ResearchSearchService",
    )

    research_service = ResearchSearchService(
        retriever=dense_retriever,
        reranker=reranker,
        ranker=ranker,
        metadata_store=metadata_store,
    )

    # ---- RAG retrieval ----------------------------------------------------
    RAGRetriever = _import_symbol(
        "backend.rag.retriever",
        "RAGRetriever",
    )

    # IMPORTANT:
    # RAGRetriever currently accepts either:
    #   1) paper_retriever
    # OR
    #   2) explicit research_retriever + uploaded_retriever
    #
    # We use the PaperRetriever facade so Mode-1 and Mode-2 remain separated.
    # ResearchSearchService remains authoritative for the complete research
    # retrieval -> reranking -> final-ranking flow.
    #
    # Do NOT pass reranker/ranker/candidate_enricher here. Those dependencies
    # belong to ResearchSearchService, not RAGRetriever.
    rag_retriever = _construct_from_dependencies(
        RAGRetriever,
        {
            "paper_retriever": paper_retriever,
            "research_search_service": research_service,
        },
        label="RAG retriever",
    )

    # ---- Context + prompt -------------------------------------------------
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

    # ---- LLM client -------------------------------------------------------
    # _settings_mapping() has already merged Settings.as_llm_mapping().
    # Pass only the canonical LLM contract to the LLM client.
    from backend.llm.client import create_client_from_mapping

    llm_keys = (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
        "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES", "LLM_BACKOFF_BASE_SECONDS",
        "LLM_BACKOFF_MAX_SECONDS", "LLM_JITTER_SECONDS",
        "LLM_ENABLE_STRUCTURED_OUTPUT",
    )
    llm_configuration = {key: settings[key] for key in llm_keys if key in settings}

    if not llm_configuration.get("LLM_PROVIDER"):
        raise RuntimeError(
            "LLM provider is missing from application configuration. "
            "Configure LLM_PROVIDER in .env/backend.config."
        )
    if not llm_configuration.get("LLM_MODEL"):
        raise RuntimeError(
            "LLM model is missing from application configuration. "
            "Configure LLM_MODEL in .env/backend.config."
        )

    llm_client = create_client_from_mapping(llm_configuration)

    # ---- Parser + validation ----------------------------------------------
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

    # ---- Central RAG pipeline ---------------------------------------------
    if getattr(rag_retriever, "research_search_service", None) is not research_service:
        raise RuntimeError(
            "RAGRetriever was not wired to the authoritative ResearchSearchService."
        )

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
            _setting(settings, "FINAL_K", "final_k"),
            10,
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

        # ---- Research analysis modules ---------------------------------------
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

    for dependency_name, module_name in analysis_module_map.items():
        component_name = dependency_name.removesuffix("_analyzer")
        component_class = _find_analysis_component(
            module_name,
            component_name,
        )
        analyzer_dependencies[dependency_name] = _construct_from_dependencies(
            component_class,
            {},
            label=f"{dependency_name} analysis component",
        )

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

    if not callable(getattr(analyzer_dependencies.get("findings_analyzer"), "analyze", None)):
        raise RuntimeError(
            "Configured findings_analyzer does not expose the required analyze() contract."
        )

    # ---- Mode-2: uploaded-paper indexing + upload pipeline ---------------
    #
    # Mode-1 remains completely unchanged above.
    #
    # Mode-2 is wired against the installed paper_indexing.py using the
    # strongest available contract:
    #
    #   1. UploadedPaperIndexer class (preferred)
    #   2. build_uploaded_paper_index() function (compatibility fallback)
    #
    # The fallback exists because older project versions expose the functional
    # builder while newer versions expose UploadedPaperIndexer. In both cases
    # PaperUploadPipeline receives one object exposing build(...), so the rest
    # of the Mode-2 pipeline remains unchanged.

    paper_indexing_module = import_module(
        "backend.services.paper_indexing"
    )

    uploaded_indexer_cls = getattr(
        paper_indexing_module,
        "UploadedPaperIndexer",
        None,
    )
    uploaded_index_config_cls = getattr(
        paper_indexing_module,
        "UploadedPaperIndexConfig",
        None,
    )

    if uploaded_indexer_cls is not None:
        indexer_kwargs: dict[str, Any] = {
            "encoder": encoder,
            "faiss_index_cls": FAISSVectorIndex,
        }

        if uploaded_index_config_cls is not None:
            config_kwargs = {
                "root_dir": uploaded_root,
                "overwrite": False,
                "include_chunk_text": True,
                "verify_reload": True,
            }

            try:
                config_signature = inspect.signature(
                    uploaded_index_config_cls
                )
                accepted = set(config_signature.parameters)
                config_kwargs = {
                    key: value
                    for key, value in config_kwargs.items()
                    if key in accepted
                }
            except (TypeError, ValueError):
                pass

            uploaded_index_config = uploaded_index_config_cls(
                **config_kwargs
            )
            indexer_kwargs["config"] = uploaded_index_config

        paper_indexing_service = _construct_from_dependencies(
            uploaded_indexer_cls,
            indexer_kwargs,
            label="uploaded-paper indexing service",
        )

        LOGGER.info(
            "Mode-2 using UploadedPaperIndexer class: index_root=%s",
            uploaded_root,
        )

    else:
        build_uploaded_paper_index = getattr(
            paper_indexing_module,
            "build_uploaded_paper_index",
            None,
        )

        if not callable(build_uploaded_paper_index):
            raise RuntimeError(
                "Mode-2 cannot start: backend.services.paper_indexing "
                "must expose either UploadedPaperIndexer or "
                "build_uploaded_paper_index()."
            )

        paper_indexing_service = _FunctionBackedUploadedPaperIndexer(
            encoder=encoder,
            root_dir=uploaded_root,
            overwrite=False,
            include_chunk_text=True,
            verify_reload=True,
        )

        LOGGER.warning(
            "Mode-2 using compatibility build_uploaded_paper_index() "
            "adapter because UploadedPaperIndexer is not available."
        )

    create_paper_upload_pipeline = _import_symbol(
        "backend.services.upload_indexing",
        "create_paper_upload_pipeline",
    )

    paper_upload_pipeline = create_paper_upload_pipeline(
        encoder=encoder,
        indexer=paper_indexing_service,
        settings=settings,
    )

    if not callable(
        getattr(paper_upload_pipeline, "process_uploaded_file", None)
    ):
        raise RuntimeError(
            "PaperUploadPipeline does not expose the required "
            "process_uploaded_file() contract."
        )

    LOGGER.info(
        "Mode-2 PDF upload pipeline registered successfully: index_root=%s",
        uploaded_root,
    )

    return {
        "encoder": encoder,
        "research_index": research_index,
        "metadata_store": metadata_store,
        "research_retriever": dense_retriever,
        "uploaded_retriever": uploaded_retriever,
        "uploaded_index_health": uploaded_index_health,
        "metadata_store_module_name": metadata_store_module_name,
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


def _validate_core_services(services: Mapping[str, Any]) -> dict[str, str]:
    """
    Validate the minimum runtime contracts required by the application.

    This runs immediately after service construction and before the application
    is marked ready. It specifically prevents the historical failure mode where
    /api/qa starts successfully but app.state.rag_pipeline is missing or
    malformed.
    """
    required = (
        "rag_pipeline",
        "rag_retriever",
        "paper_retriever",
        "uploaded_retriever",
        "llm_client",
        "paper_upload_pipeline",
        "paper_indexing_service",
        "research_service",
        "research_analyzer",
        "metadata_store",
    )

    missing = [name for name in required if services.get(name) is None]
    if missing:
        raise RuntimeError(
            "Core application services were not initialized: "
            + ", ".join(missing)
        )

    rag_pipeline = services["rag_pipeline"]
    if not callable(getattr(rag_pipeline, "run", None)):
        raise RuntimeError(
            "Core service validation failed: "
            "rag_pipeline.run() is not callable."
        )

    upload_pipeline = services["paper_upload_pipeline"]
    if not callable(getattr(upload_pipeline, "process_uploaded_file", None)):
        raise RuntimeError(
            "Core service validation failed: "
            "paper_upload_pipeline.process_uploaded_file() is not callable."
        )

    rag_retriever = services["rag_retriever"]
    if not callable(getattr(rag_retriever, "retrieve", None)):
        if not callable(getattr(rag_retriever, "search", None)):
            raise RuntimeError(
                "Core service validation failed: rag_retriever exposes neither "
                "retrieve() nor search()."
            )

    LOGGER.info(
        "Core service validation passed: rag_pipeline=%s, "
        "paper_upload_pipeline=%s, uploaded_retriever=%s",
        type(rag_pipeline).__name__,
        type(upload_pipeline).__name__,
        type(services["uploaded_retriever"]).__name__,
    )

    return {
        "rag_pipeline": type(rag_pipeline).__name__,
        "rag_retriever": type(rag_retriever).__name__,
        "uploaded_retriever": type(services["uploaded_retriever"]).__name__,
        "paper_upload_pipeline": type(upload_pipeline).__name__,
        "llm_client": type(services["llm_client"]).__name__,
    }


# ============================================================================
# Canonical pipeline state contract
# ============================================================================

def _validate_rag_pipeline_state(state: Any) -> Any:
    """
    Validate the application-level RAG pipeline dependency contract.

    This is intentionally independent of FastAPI and model initialization so
    API dependency wiring can be tested deterministically.
    """
    rag_pipeline = getattr(state, "rag_pipeline", None)
    qa_pipeline = getattr(state, "qa_pipeline", None)

    if rag_pipeline is None:
        raise RuntimeError("app.state.rag_pipeline is not initialized.")

    if not callable(getattr(rag_pipeline, "run", None)):
        raise RuntimeError("app.state.rag_pipeline.run() is not callable.")

    if qa_pipeline is not rag_pipeline:
        raise RuntimeError(
            "app.state.qa_pipeline must be an identity-preserving alias of "
            "app.state.rag_pipeline."
        )

    return rag_pipeline


# ============================================================================
# Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all long-lived application dependencies exactly once.

    Startup is fail-fast for core services. Shutdown attempts graceful close()
    calls without hiding the original lifecycle state.
    """
    settings = _load_project_settings()
    settings_map = _settings_mapping(settings)

    # Validate the canonical LLM contract before expensive ML services start.
    if settings_map:
        provider = _setting(settings_map, "LLM_PROVIDER", "llm_provider", default=None)
        model = _setting(settings_map, "LLM_MODEL", "llm_model", default=None)
        if not provider or not model:
            raise RuntimeError(
                "LLM configuration is incomplete: both LLM_PROVIDER and "
                "LLM_MODEL are required."
            )

    app.state.settings = settings
    app.state.config = settings_map
    app.state.project_root = PROJECT_ROOT
    app.state.service_name = _setting(
        settings_map,
        "APP_NAME",
        "app_name",
        "SERVICE_NAME",
        "service_name",
        default="AI Research Paper Assistant",
    )

    LOGGER.info("Starting %s.", app.state.service_name)

    app.state.health_lifecycle_ready = False

    try:
        services = _build_services(settings_map)
        service_health = _validate_core_services(services)
    except Exception:
        app.state.health_lifecycle_ready = False
        LOGGER.exception("Application startup failed during service initialization.")
        raise

    for name, service in services.items():
        setattr(app.state, name, service)

    # ------------------------------------------------------------------
    # Canonical QA/RAG pipeline state
    # ------------------------------------------------------------------
    # `app.state.rag_pipeline` is the SINGLE source of truth for the QA
    # orchestration pipeline. Do not construct a second QA pipeline here.
    #
    # Older API code may still resolve `qa_pipeline`; keep it only as an
    # identity-preserving compatibility alias.
    canonical_rag_pipeline = services["rag_pipeline"]

    if canonical_rag_pipeline is None:
        raise RuntimeError(
            "Application startup aborted: canonical rag_pipeline is missing."
        )

    if not callable(getattr(canonical_rag_pipeline, "run", None)):
        raise RuntimeError(
            "Application startup aborted: canonical rag_pipeline.run() "
            "is not callable."
        )

    app.state.rag_pipeline = canonical_rag_pipeline
    app.state.qa_pipeline = canonical_rag_pipeline
    app.state.core_service_health = service_health

    # These aliases MUST point to the exact same object. This catches the
    # historical dependency-wiring bug at startup instead of on /api/qa.
    if app.state.qa_pipeline is not app.state.rag_pipeline:
        raise RuntimeError(
            "Application startup aborted: qa_pipeline and rag_pipeline "
            "must reference the same canonical RAGPipeline instance."
        )

    # Final runtime assertion: dependency resolution for /api/qa must be able
    # to find the SAME live pipeline that was constructed by _build_services().
    runtime_rag_pipeline = getattr(app.state, "rag_pipeline", None)

    if runtime_rag_pipeline is None:
        raise RuntimeError(
            "Application startup aborted: app.state.rag_pipeline is missing."
        )

    if not callable(getattr(runtime_rag_pipeline, "run", None)):
        raise RuntimeError(
            "Application startup aborted: app.state.rag_pipeline.run() "
            "is not callable."
        )

    if runtime_rag_pipeline is not services["rag_pipeline"]:
        raise RuntimeError(
            "Application startup aborted: app.state.rag_pipeline is not "
            "the canonical RAGPipeline created by _build_services()."
        )

    if getattr(app.state, "qa_pipeline", None) is not runtime_rag_pipeline:
        raise RuntimeError(
            "Application startup aborted: app.state.qa_pipeline is not the "
            "canonical app.state.rag_pipeline instance."
        )

    _validate_rag_pipeline_state(app.state)

    # Health router consumes this explicit required/optional contract.
    app.state.health_required_services = {
        "research_service": True,
        "metadata_store": True,
        "research_analyzer": True,
        "paper_indexing_service": True,
        "rag_pipeline": True,
        "llm_client": True,
        "paper_upload_pipeline": True,
        "uploaded_index_health": True,
    }

    # Only after all core services are installed is the application ready.
    app.state.health_lifecycle_ready = True

    LOGGER.info(
        "Application startup completed. Core services initialized. "
        "RAG QA ready=%s upload pipeline ready=%s",
        callable(getattr(app.state.rag_pipeline, "run", None)),
        callable(
            getattr(
                app.state.paper_upload_pipeline,
                "process_uploaded_file",
                None,
            )
        ),
    )

    try:
        yield
    finally:
        app.state.health_lifecycle_ready = False
        LOGGER.info("Shutting down application services.")

        seen: set[int] = set()

        for name, service in services.items():
            if service is None:
                continue

            identity = id(service)
            if identity in seen:
                continue
            seen.add(identity)

            close = getattr(service, "close", None)
            if callable(close):
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

        LOGGER.info("Application shutdown completed.")


# ============================================================================
# Application
# ============================================================================

def create_app(*, initialize_services: bool = True) -> FastAPI:
    """
    Create the single FastAPI application.

    `initialize_services=False` exists only for lightweight tests/import checks.
    Production/Uvicorn uses the default True.
    """
    settings = _load_project_settings()
    settings_map = _settings_mapping(settings)

    app = FastAPI(
        title=str(
            _setting(
                settings_map,
                "APP_NAME",
                "app_name",
                default="AI Research Paper Assistant",
            )
        ),
        version=str(
            _setting(
                settings_map,
                "APP_VERSION",
                "app_version",
                "VERSION",
                "version",
                default="0.1.0",
            )
        ),
        description=(
            "AI Research Paper Assistant backend API. "
            "Research discovery and uploaded-paper analysis."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan if initialize_services else None,
    )

    # Baseline application state must exist even when
    # `initialize_services=False`. The lightweight self-test intentionally
    # skips the production lifespan, while dependency-free endpoints such as
    # `/` still need a stable service identity.
    app.state.service_name = _setting(
        settings_map,
        "APP_NAME",
        "app_name",
        "SERVICE_NAME",
        "service_name",
        default="AI Research Paper Assistant",
    )
    app.state.project_root = PROJECT_ROOT
    app.state.config = settings_map
    app.state.health_lifecycle_ready = False
    app.state.health_required_services = {
        "research_service": True,
        "metadata_store": True,
        "research_analyzer": True,
        "paper_indexing_service": True,
        "rag_pipeline": True,
        "llm_client": True,
        "paper_upload_pipeline": True,
        "uploaded_index_health": True,
    }

    origins = _cors_origins(settings_map)

    if origins:
        wildcard_cors = "*" in origins
        if wildcard_cors:
            LOGGER.warning(
                "CORS_ORIGINS contains '*'; credentials are disabled for wildcard CORS."
            )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=not wildcard_cors,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------------
    from backend.api import health, papers, qa, research

    app.include_router(health.router)
    app.include_router(research.router)
    app.include_router(papers.router)
    app.include_router(qa.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        # `/` is intentionally dependency-free and must also work for the
        # lightweight `initialize_services=False` contract test.
        return {
            "service": str(
                getattr(
                    app.state,
                    "service_name",
                    _setting(
                        settings_map,
                        "APP_NAME",
                        "app_name",
                        "SERVICE_NAME",
                        "service_name",
                        default="AI Research Paper Assistant",
                    ),
                )
            ),
            "status": "ok",
        }

    # ------------------------------------------------------------------------
    # Exception boundaries
    # ------------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        validation_errors = exc.errors()
        LOGGER.warning(
            "Request validation failed: method=%s path=%s errors=%s",
            request.method,
            request.url.path,
            validation_errors,
        )

        content: dict[str, Any] = {
            "detail": "Request validation failed.",
        }

        # DEBUG mode is opt-in. Production responses remain intentionally
        # generic so validation internals are not exposed to clients.
        debug_enabled = _as_bool(
            _setting(
                settings_map,
                "DEBUG",
                "debug",
                "APP_DEBUG",
                "app_debug",
                default=os.getenv("DEBUG"),
            ),
            False,
        )
        if debug_enabled:
            content["errors"] = validation_errors

        return JSONResponse(
            status_code=422,
            content=content,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception(
            "Unhandled application exception: method=%s path=%s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()


# ============================================================================
# Lightweight contract self-test
# ============================================================================

def run_self_test() -> None:
    """
    Validate application creation/router registration without loading ML
    services. This is intentionally separate from the production lifespan.
    """
    from fastapi.testclient import TestClient

    test_app = create_app(initialize_services=False)

    with TestClient(test_app) as client:
        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200

        paths = openapi.json()["paths"]

        assert "/health" in paths
        assert "/health/ready" in paths
        assert "/research/search" in paths
        assert "/api/qa" in paths
        assert "/api/papers/upload" in paths

        root_response = client.get("/")
        assert root_response.status_code == 200
        assert root_response.json()["status"] == "ok"

        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"

        # Readiness without lifecycle-created services must fail safely.
        readiness = client.get("/health/ready")
        assert readiness.status_code == 503
        assert readiness.json()["status"] == "not_ready"

        # The production lifespan is intentionally skipped here, so the RAG
        # dependency must not appear magically during a lightweight import test.
        assert getattr(test_app.state, "rag_pipeline", None) is None

        # Configuration contract self-test: canonical LLM settings must survive
        # the Settings -> runtime mapping boundary without starting any model.
        class _SelfTestSettings:
            def model_dump(self) -> dict[str, Any]:
                return {"llm_provider": "ollama", "llm_model": "qwen2.5:7b"}

            def as_llm_mapping(self) -> dict[str, Any]:
                return {
                    "LLM_PROVIDER": "ollama",
                    "LLM_MODEL": "qwen2.5:7b",
                    "LLM_BASE_URL": "http://127.0.0.1:11434/",
                    "LLM_API_KEY": None,
                }

        config_contract = _settings_mapping(_SelfTestSettings())
        assert config_contract["LLM_PROVIDER"] == "ollama"
        assert config_contract["LLM_MODEL"] == "qwen2.5:7b"
        assert config_contract["LLM_BASE_URL"] == "http://127.0.0.1:11434/"

        # Canonical RAG pipeline state contract: no second QA pipeline may be
        # silently substituted into application state.
        class _PipelineStub:
            def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        canonical = _PipelineStub()
        test_app.state.rag_pipeline = canonical
        test_app.state.qa_pipeline = canonical
        assert _validate_rag_pipeline_state(test_app.state) is canonical

        test_app.state.qa_pipeline = _PipelineStub()
        try:
            _validate_rag_pipeline_state(test_app.state)
        except RuntimeError as exc:
            assert "identity-preserving alias" in str(exc)
        else:
            raise AssertionError(
                "A second QA pipeline instance was incorrectly accepted."
            )

        test_app.state.qa_pipeline = canonical

        # No secrets or local paths are exposed by the public health contract.
        payload = str(readiness.json()).lower()
        forbidden = (
            "api_key",
            "password",
            "secret",
            "token",
            "c:\\users",
        )
        assert not any(item in payload for item in forbidden)

    print("backend/main.py self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()