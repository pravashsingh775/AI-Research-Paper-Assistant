"""
Centralized application exception contract for the AI Research Paper Assistant.

This module intentionally contains:
    - domain/application exception definitions
    - stable machine-readable error codes
    - safe HTTP metadata
    - safe structured serialization

It does NOT:
    - import FastAPI
    - load models or indexes
    - call an LLM
    - process PDFs
    - perform retrieval/RAG/analysis
    - perform logging
    - perform business logic

Low-level services raise these exceptions. The API/global handler decides how
to expose them over HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final


# =============================================================================
# Stable error-code constants
# =============================================================================
#
# These values are part of the backend/frontend contract. Do not rename an
# existing code casually once clients depend on it.
# =============================================================================

CONFIGURATION_ERROR: Final[str] = "CONFIGURATION_ERROR"

INVALID_DOCUMENT: Final[str] = "INVALID_DOCUMENT"
UNSUPPORTED_FILE_TYPE: Final[str] = "UNSUPPORTED_FILE_TYPE"
FILE_TOO_LARGE: Final[str] = "FILE_TOO_LARGE"
DOCUMENT_NOT_FOUND: Final[str] = "DOCUMENT_NOT_FOUND"
DOCUMENT_NOT_READY: Final[str] = "DOCUMENT_NOT_READY"
DOCUMENT_PROCESSING_FAILED: Final[str] = "DOCUMENT_PROCESSING_FAILED"

PDF_EXTRACTION_FAILED: Final[str] = "PDF_EXTRACTION_FAILED"
PDF_SECTION_PARSING_FAILED: Final[str] = "PDF_SECTION_PARSING_FAILED"
PDF_CHUNKING_FAILED: Final[str] = "PDF_CHUNKING_FAILED"

INDEX_NOT_FOUND: Final[str] = "INDEX_NOT_FOUND"
INDEX_UNAVAILABLE: Final[str] = "INDEX_UNAVAILABLE"
INDEX_LOAD_FAILED: Final[str] = "INDEX_LOAD_FAILED"
INDEX_SEARCH_FAILED: Final[str] = "INDEX_SEARCH_FAILED"

EMBEDDING_MODEL_UNAVAILABLE: Final[str] = "EMBEDDING_MODEL_UNAVAILABLE"
EMBEDDING_GENERATION_FAILED: Final[str] = "EMBEDDING_GENERATION_FAILED"

RETRIEVAL_FAILED: Final[str] = "RETRIEVAL_FAILED"
RERANKING_FAILED: Final[str] = "RERANKING_FAILED"
RANKING_FAILED: Final[str] = "RANKING_FAILED"

RAG_FAILED: Final[str] = "RAG_FAILED"
CONTEXT_CONSTRUCTION_FAILED: Final[str] = "CONTEXT_CONSTRUCTION_FAILED"
EVIDENCE_RETRIEVAL_FAILED: Final[str] = "EVIDENCE_RETRIEVAL_FAILED"

LLM_UNAVAILABLE: Final[str] = "LLM_UNAVAILABLE"
LLM_TIMEOUT: Final[str] = "LLM_TIMEOUT"
LLM_AUTHENTICATION_FAILED: Final[str] = "LLM_AUTHENTICATION_FAILED"
LLM_RESPONSE_PARSE_FAILED: Final[str] = "LLM_RESPONSE_PARSE_FAILED"
LLM_FAILED: Final[str] = "LLM_FAILED"

VALIDATION_FAILED: Final[str] = "VALIDATION_FAILED"
DOCUMENT_CONTEXT_MISMATCH: Final[str] = "DOCUMENT_CONTEXT_MISMATCH"
EVIDENCE_PROVENANCE_FAILED: Final[str] = "EVIDENCE_PROVENANCE_FAILED"

SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"
INTERNAL_SERVICE_ERROR: Final[str] = "INTERNAL_SERVICE_ERROR"


# =============================================================================
# HTTP metadata
# =============================================================================

DEFAULT_STATUS_CODE: Final[int] = 500

# Only these fields are allowed to cross the exception -> API boundary.
# In particular, no traceback, path, prompt, model path, secret, or raw
# provider exception is accepted here.
_SAFE_DETAIL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "document_id",
        "filename",
        "supported_file_types",
        "max_file_size",
        "retryable",
        "operation",
        "provider",
        "component",
        "status",
    }
)

_SENSITIVE_KEY_TOKENS: Final[tuple[str, ...]] = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "password",
    "secret",
    "token",
    "credential",
    "traceback",
    "stack",
    "prompt",
    "path",
    "filepath",
    "file_path",
    "model_path",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return (
        normalized not in _SAFE_DETAIL_KEYS
        or any(token in normalized for token in _SENSITIVE_KEY_TOKENS)
    )


def _safe_detail_value(key: str, value: Any) -> Any:
    """Return a small JSON-safe value or None for unsafe detail content."""
    if _is_sensitive_key(key):
        return None

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        # Exception details are intentionally bounded. This prevents accidental
        # serialization of large internal payloads.
        return value[:500]

    if isinstance(value, (list, tuple, set, frozenset)):
        values: list[Any] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)) or item is None:
                values.append(item)
            else:
                values.append(str(type(item).__name__))
        return values[:50]

    return None


def _sanitize_details(
    details: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Keep only explicitly approved, JSON-safe, non-sensitive details.

    Callers should still pass only intentionally safe information. This helper
    is a second defensive boundary rather than a replacement for good design.
    """
    if not details:
        return None

    sanitized: dict[str, Any] = {}

    for raw_key, raw_value in details.items():
        key = str(raw_key).strip()
        if not key or _is_sensitive_key(key):
            continue

        value = _safe_detail_value(key, raw_value)
        if value is not None:
            sanitized[key] = value

    return sanitized or None


# =============================================================================
# Base exception
# =============================================================================

class ApplicationError(Exception):
    """
    Base class for expected backend application/domain failures.

    Attributes:
        code:
            Stable machine-readable error code.
        message:
            Safe user-facing message. It must not contain implementation
            details or secrets.
        status_code:
            Suggested HTTP status for the API boundary. It is metadata only;
            this module does not create HTTP responses.
        details:
            Optional, explicitly sanitized structured information.
        retryable:
            Whether retrying the same operation may reasonably succeed.

    The original low-level exception should be preserved through normal Python
    exception chaining:

        raise RetrievalError() from exc

    The cause is deliberately not included in ``to_dict()``.
    """

    __slots__ = (
        "_code",
        "_message",
        "_status_code",
        "_details",
        "_retryable",
    )

    code: str
    message: str
    status_code: int
    details: dict[str, Any] | None
    retryable: bool

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = DEFAULT_STATUS_CODE,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        normalized_code = str(code).strip().upper()
        normalized_message = str(message).strip()

        if not normalized_code:
            raise ValueError("ApplicationError code cannot be empty.")

        if not normalized_message:
            raise ValueError("ApplicationError message cannot be empty.")

        if not 400 <= int(status_code) <= 599:
            raise ValueError(
                "ApplicationError status_code must be between 400 and 599."
            )

        self._code = normalized_code
        self._message = normalized_message[:1000]
        self._status_code = int(status_code)
        self._details = _sanitize_details(details)
        self._retryable = bool(retryable)

        # Exception.__init__ controls str(exc). Use only the safe message.
        super().__init__(self._message)

    @property
    def code(self) -> str:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def details(self) -> dict[str, Any] | None:
        if self._details is None:
            return None
        return dict(self._details)

    @property
    def retryable(self) -> bool:
        return self._retryable

    def to_dict(self) -> dict[str, Any]:
        """
        Return the API-safe error object.

        The shape intentionally matches the requested frontend contract:

            {
                "error": {
                    "code": "...",
                    "message": "...",
                    "details": ...
                }
            }

        ``retryable`` is included inside details only when meaningful.
        """
        details = self.details

        if self.retryable:
            details = dict(details or {})
            details["retryable"] = True

        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": details,
            }
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"status_code={self.status_code!r}, "
            f"retryable={self.retryable!r})"
        )


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(ApplicationError):
    """Application configuration is invalid or unusable."""

    def __init__(
        self,
        message: str = "The backend configuration is invalid.",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=CONFIGURATION_ERROR,
            message=message,
            status_code=500,
            details=details,
        )


# =============================================================================
# Document / upload errors
# =============================================================================

class DocumentError(ApplicationError):
    """Base class for expected uploaded-document failures."""

    pass


class InvalidDocumentError(DocumentError):
    """Uploaded document is invalid or cannot be accepted."""

    def __init__(
        self,
        message: str = "The uploaded document is invalid.",
        *,
        filename: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged = dict(details or {})
        if filename:
            merged.setdefault("filename", filename)

        super().__init__(
            code=INVALID_DOCUMENT,
            message=message,
            status_code=400,
            details=merged,
        )


class UnsupportedFileTypeError(DocumentError):
    """Uploaded file type is not supported by the current pipeline."""

    def __init__(
        self,
        *,
        filename: str | None = None,
        supported_file_types: list[str] | tuple[str, ...] = (".pdf",),
    ) -> None:
        details: dict[str, Any] = {
            "supported_file_types": list(supported_file_types),
        }
        if filename:
            details["filename"] = filename

        super().__init__(
            code=UNSUPPORTED_FILE_TYPE,
            message="The uploaded file type is not supported.",
            status_code=400,
            details=details,
        )


class FileTooLargeError(DocumentError):
    """Uploaded file exceeds the configured size limit."""

    def __init__(
        self,
        *,
        filename: str | None = None,
        max_file_size: int | str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if filename:
            details["filename"] = filename
        if max_file_size is not None:
            details["max_file_size"] = max_file_size

        super().__init__(
            code=FILE_TOO_LARGE,
            message="The uploaded file exceeds the allowed size limit.",
            status_code=413,
            details=details,
        )


class DocumentNotFoundError(DocumentError):
    """Requested document does not exist."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            code=DOCUMENT_NOT_FOUND,
            message="The requested paper was not found.",
            status_code=404,
            details={"document_id": document_id},
        )


class DocumentNotReadyError(DocumentError):
    """Document exists but its ingestion/indexing pipeline is not complete."""

    def __init__(
        self,
        document_id: str,
        *,
        status: str = "processing",
    ) -> None:
        super().__init__(
            code=DOCUMENT_NOT_READY,
            message="The requested paper is not ready yet.",
            status_code=409,
            details={
                "document_id": document_id,
                "status": status,
            },
        )


class DocumentProcessingError(DocumentError):
    """General document-ingestion/indexing failure."""

    def __init__(
        self,
        message: str = "The paper could not be processed.",
        *,
        document_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        code: str = DOCUMENT_PROCESSING_FAILED,
    ) -> None:
        merged = dict(details or {})
        if document_id:
            merged.setdefault("document_id", document_id)

        super().__init__(
            code=code,
            message=message,
            status_code=503 if retryable else 500,
            details=merged,
            retryable=retryable,
        )


# =============================================================================
# PDF processing
# =============================================================================

class PDFExtractionError(DocumentProcessingError):
    """PDF text/metadata extraction failed."""

    def __init__(
        self,
        message: str = "The PDF could not be extracted.",
        *,
        document_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            retryable=retryable,
            code=PDF_EXTRACTION_FAILED,
        )


class SectionParsingError(DocumentProcessingError):
    """PDF section parsing failed."""

    def __init__(
        self,
        message: str = "The paper sections could not be parsed.",
        *,
        document_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code=PDF_SECTION_PARSING_FAILED,
        )


class ChunkingError(DocumentProcessingError):
    """Semantic chunking failed."""

    def __init__(
        self,
        message: str = "The paper could not be chunked for retrieval.",
        *,
        document_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code=PDF_CHUNKING_FAILED,
        )


# =============================================================================
# Index errors
# =============================================================================

class IndexErrorBase(ApplicationError):
    """
    Base class for application vector-index failures.

    Named ``IndexErrorBase`` to avoid shadowing Python's built-in ``IndexError``.
    """

    pass


class IndexNotFoundError(IndexErrorBase):
    """Required vector index does not exist."""

    def __init__(
        self,
        message: str = "The required research index was not found.",
    ) -> None:
        super().__init__(
            code=INDEX_NOT_FOUND,
            message=message,
            status_code=503,
            retryable=False,
        )


class IndexUnavailableError(IndexErrorBase):
    """Index exists conceptually but cannot currently be used."""

    def __init__(
        self,
        message: str = "The research index is currently unavailable.",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            code=INDEX_UNAVAILABLE,
            message=message,
            status_code=503,
            retryable=retryable,
        )


class IndexLoadError(IndexErrorBase):
    """Vector index failed to load or validate."""

    def __init__(
        self,
        message: str = "The research index could not be loaded.",
    ) -> None:
        super().__init__(
            code=INDEX_LOAD_FAILED,
            message=message,
            status_code=503,
        )


class IndexSearchError(IndexErrorBase):
    """Vector-index search operation failed."""

    def __init__(
        self,
        message: str = "The research index search failed.",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            code=INDEX_SEARCH_FAILED,
            message=message,
            status_code=503,
            retryable=retryable,
        )


# =============================================================================
# Embedding errors
# =============================================================================

class EmbeddingError(ApplicationError):
    """Base class for embedding-layer failures."""

    pass


class EmbeddingModelUnavailableError(EmbeddingError):
    """Embedding model cannot be loaded or is unavailable."""

    def __init__(
        self,
        message: str = "The scientific embedding model is unavailable.",
    ) -> None:
        super().__init__(
            code=EMBEDDING_MODEL_UNAVAILABLE,
            message=message,
            status_code=503,
            retryable=True,
        )


class EmbeddingGenerationError(EmbeddingError):
    """Embedding generation failed for a request/document."""

    def __init__(
        self,
        message: str = "Scientific embedding generation failed.",
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            code=EMBEDDING_GENERATION_FAILED,
            message=message,
            status_code=503 if retryable else 500,
            retryable=retryable,
        )


# =============================================================================
# Retrieval / ranking
# =============================================================================

class RetrievalError(ApplicationError):
    """Dense retrieval failed."""

    def __init__(
        self,
        message: str = "Scientific paper retrieval failed.",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            code=RETRIEVAL_FAILED,
            message=message,
            status_code=503,
            retryable=retryable,
        )


class RerankingError(ApplicationError):
    """Candidate reranking failed."""

    def __init__(
        self,
        message: str = "Paper relevance reranking failed.",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            code=RERANKING_FAILED,
            message=message,
            status_code=503,
            retryable=retryable,
        )


class RankingError(ApplicationError):
    """Final ranking failed."""

    def __init__(
        self,
        message: str = "Final paper ranking failed.",
    ) -> None:
        super().__init__(
            code=RANKING_FAILED,
            message=message,
            status_code=500,
        )


# =============================================================================
# RAG
# =============================================================================

class RAGError(ApplicationError):
    """Base class for RAG pipeline failures."""

    def __init__(
        self,
        message: str = "The research-answering pipeline failed.",
        *,
        retryable: bool = False,
        code: str = RAG_FAILED,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=503 if retryable else 500,
            retryable=retryable,
        )


class ContextConstructionError(RAGError):
    """Evidence context construction failed."""

    def __init__(
        self,
        message: str = "Research evidence context could not be constructed.",
    ) -> None:
        super().__init__(
            message=message,
            code=CONTEXT_CONSTRUCTION_FAILED,
        )


class EvidenceRetrievalError(RAGError):
    """RAG evidence retrieval failed."""

    def __init__(
        self,
        message: str = "Evidence retrieval failed.",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            message=message,
            retryable=retryable,
            code=EVIDENCE_RETRIEVAL_FAILED,
        )


# =============================================================================
# Document provenance / isolation
# =============================================================================

class DocumentContextMismatchError(ApplicationError):
    """Evidence belongs to a different document than requested."""

    def __init__(
        self,
        *,
        document_id: str,
    ) -> None:
        super().__init__(
            code=DOCUMENT_CONTEXT_MISMATCH,
            message="The retrieved evidence does not belong to the requested paper.",
            status_code=409,
            details={"document_id": document_id},
        )


class EvidenceProvenanceError(ApplicationError):
    """Evidence provenance could not be trusted or validated."""

    def __init__(
        self,
        message: str = "The evidence provenance could not be validated.",
        *,
        document_id: str | None = None,
    ) -> None:
        details = (
            {"document_id": document_id}
            if document_id
            else None
        )

        super().__init__(
            code=EVIDENCE_PROVENANCE_FAILED,
            message=message,
            status_code=500,
            details=details,
        )


# =============================================================================
# LLM
# =============================================================================

class LLMError(ApplicationError):
    """Base class for language-model/provider failures."""

    def __init__(
        self,
        message: str = "The language model service failed.",
        *,
        status_code: int = 502,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        code: str = LLM_FAILED,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=status_code,
            details=details,
            retryable=retryable,
        )


class LLMUnavailableError(LLMError):
    """LLM provider/service is unavailable."""

    def __init__(
        self,
        message: str = "The language model service is currently unavailable.",
        *,
        provider: str | None = None,
    ) -> None:
        details = {"provider": provider} if provider else None

        super().__init__(
            message=message,
            status_code=503,
            retryable=True,
            details=details,
            code=LLM_UNAVAILABLE,
        )


class LLMTimeoutError(LLMError):
    """LLM request exceeded its configured timeout."""

    def __init__(
        self,
        message: str = "The language model request timed out.",
        *,
        provider: str | None = None,
    ) -> None:
        details = {"provider": provider} if provider else None

        super().__init__(
            message=message,
            status_code=504,
            retryable=True,
            details=details,
            code=LLM_TIMEOUT,
        )


class LLMAuthenticationError(LLMError):
    """LLM provider rejected authentication."""

    def __init__(
        self,
        message: str = "Language model provider authentication failed.",
        *,
        provider: str | None = None,
    ) -> None:
        details = {"provider": provider} if provider else None

        super().__init__(
            message=message,
            status_code=502,
            retryable=False,
            details=details,
            code=LLM_AUTHENTICATION_FAILED,
        )


class LLMResponseParseError(LLMError):
    """Expected structured LLM output could not be parsed safely."""

    def __init__(
        self,
        message: str = "The language model returned an invalid response format.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=502,
            retryable=False,
            code=LLM_RESPONSE_PARSE_FAILED,
        )


# =============================================================================
# Validation
# =============================================================================

class ValidationError(ApplicationError):
    """
    Final validation-system failure.

    This is intentionally different from a valid ``insufficient_evidence``
    result. Lack of evidence is a scientifically meaningful application state,
    not automatically an HTTP 500 error.
    """

    def __init__(
        self,
        message: str = "The generated research response could not be validated.",
    ) -> None:
        super().__init__(
            code=VALIDATION_FAILED,
            message=message,
            status_code=500,
        )


# =============================================================================
# Generic service boundary
# =============================================================================

class ServiceError(ApplicationError):
    """Base class for an unexpected service-layer failure."""

    def __init__(
        self,
        message: str = "The requested backend service is unavailable.",
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=SERVICE_UNAVAILABLE if retryable else INTERNAL_SERVICE_ERROR,
            message=message,
            status_code=503 if retryable else 500,
            details=details,
            retryable=retryable,
        )


# =============================================================================
# Explicit public API
# =============================================================================

__all__ = [
    # Base
    "ApplicationError",

    # Configuration
    "ConfigurationError",

    # Documents
    "DocumentError",
    "InvalidDocumentError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "DocumentNotFoundError",
    "DocumentNotReadyError",
    "DocumentProcessingError",

    # PDF
    "PDFExtractionError",
    "SectionParsingError",
    "ChunkingError",

    # Index
    "IndexErrorBase",
    "IndexNotFoundError",
    "IndexUnavailableError",
    "IndexLoadError",
    "IndexSearchError",

    # Embeddings
    "EmbeddingError",
    "EmbeddingModelUnavailableError",
    "EmbeddingGenerationError",

    # Retrieval
    "RetrievalError",
    "RerankingError",
    "RankingError",

    # RAG
    "RAGError",
    "ContextConstructionError",
    "EvidenceRetrievalError",

    # Provenance/isolation
    "DocumentContextMismatchError",
    "EvidenceProvenanceError",

    # LLM
    "LLMError",
    "LLMUnavailableError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
    "LLMResponseParseError",

    # Validation
    "ValidationError",

    # Generic service
    "ServiceError",

    # Stable error codes
    "CONFIGURATION_ERROR",
    "INVALID_DOCUMENT",
    "UNSUPPORTED_FILE_TYPE",
    "FILE_TOO_LARGE",
    "DOCUMENT_NOT_FOUND",
    "DOCUMENT_NOT_READY",
    "DOCUMENT_PROCESSING_FAILED",
    "PDF_EXTRACTION_FAILED",
    "PDF_SECTION_PARSING_FAILED",
    "PDF_CHUNKING_FAILED",
    "INDEX_NOT_FOUND",
    "INDEX_UNAVAILABLE",
    "INDEX_LOAD_FAILED",
    "INDEX_SEARCH_FAILED",
    "EMBEDDING_MODEL_UNAVAILABLE",
    "EMBEDDING_GENERATION_FAILED",
    "RETRIEVAL_FAILED",
    "RERANKING_FAILED",
    "RANKING_FAILED",
    "RAG_FAILED",
    "CONTEXT_CONSTRUCTION_FAILED",
    "EVIDENCE_RETRIEVAL_FAILED",
    "LLM_UNAVAILABLE",
    "LLM_TIMEOUT",
    "LLM_AUTHENTICATION_FAILED",
    "LLM_RESPONSE_PARSE_FAILED",
    "LLM_FAILED",
    "VALIDATION_FAILED",
    "DOCUMENT_CONTEXT_MISMATCH",
    "EVIDENCE_PROVENANCE_FAILED",
    "SERVICE_UNAVAILABLE",
    "INTERNAL_SERVICE_ERROR",
]


# =============================================================================
# Model-free self-test
# =============================================================================

def _run_self_test() -> None:
    """Run lightweight contract tests without importing application services."""
    error = DocumentNotFoundError("paper_123")

    assert isinstance(error, ApplicationError)
    assert error.code == DOCUMENT_NOT_FOUND
    assert error.status_code == 404
    assert error.message == "The requested paper was not found."
    assert error.details == {"document_id": "paper_123"}
    assert error.retryable is False

    payload = error.to_dict()
    assert payload == {
        "error": {
            "code": "DOCUMENT_NOT_FOUND",
            "message": "The requested paper was not found.",
            "details": {"document_id": "paper_123"},
        }
    }

    too_large = FileTooLargeError(
        filename="paper.pdf",
        max_file_size=50 * 1024 * 1024,
    )
    assert too_large.status_code == 413
    assert too_large.details == {
        "filename": "paper.pdf",
        "max_file_size": 50 * 1024 * 1024,
    }

    timeout = LLMTimeoutError(provider="example-provider")
    assert timeout.status_code == 504
    assert timeout.retryable is True
    assert timeout.details == {
        "provider": "example-provider",
    }

    secret_attempt = ApplicationError(
        code="TEST_ERROR",
        message="Safe message",
        status_code=500,
        details={
            "document_id": "paper_123",
            "api_key": "sk-secret-value",
            "authorization": "Bearer secret",
            "file_path": r"C:\secret\paper.pdf",
            "traceback": "private traceback",
        },
    )
    assert secret_attempt.details == {
        "document_id": "paper_123",
    }

    try:
        raise RuntimeError("private low-level failure")
    except RuntimeError as cause:
        wrapped = RetrievalError()
        wrapped.__cause__ = cause

        assert wrapped.__cause__ is cause
        assert "private low-level failure" not in wrapped.to_dict()["error"]["message"]

    # Empty/invalid base metadata should fail early.
    try:
        ApplicationError(code="", message="x")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty error code was accepted.")

    try:
        ApplicationError(code="X", message="x", status_code=200)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-error HTTP status was accepted.")

    print("backend/exceptions.py self-test: PASSED")


if __name__ == "__main__":
    _run_self_test()