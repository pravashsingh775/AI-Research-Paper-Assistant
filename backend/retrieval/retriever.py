"""
Production retrieval orchestration for the AI Research Paper Assistant.

Responsibilities
----------------
Mode 1:
    query -> existing EmbeddingEncoder -> existing FAISSVectorIndex
    -> validated document-level candidates

Mode 2:
    query -> existing EmbeddingEncoder -> uploaded-paper FAISS index
    -> chunk IDs -> persisted chunk metadata -> validated chunk candidates

This module deliberately does NOT:
    - implement SPECTER2
    - implement FAISS
    - create/rebuild uploaded indexes
    - parse PDFs or chunk documents
    - rerank
    - perform final ranking
    - build RAG context
    - call an LLM

The uploaded-paper path is explicitly document-scoped. It never falls back
to the research index or another uploaded paper.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Set, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_K = object()

DEFAULT_RESEARCH_INDEX_BASE = Path("indexes") / "research" / "index"
DEFAULT_UPLOADED_INDEX_ROOT = Path("indexes") / "uploaded"

_SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ---------------------------------------------------------------------------
# Existing project collaborators
# ---------------------------------------------------------------------------

try:
    from backend.embeddings.encoder import EmbeddingEncoder
except ImportError:  # pragma: no cover - supports backend-local execution
    try:
        from ..embeddings.encoder import EmbeddingEncoder
    except ImportError:
        EmbeddingEncoder = Any  # type: ignore[misc,assignment]

try:
    from backend.retrieval.faiss_index import FAISSVectorIndex
except ImportError:  # pragma: no cover
    try:
        from .faiss_index import FAISSVectorIndex
    except ImportError:
        FAISSVectorIndex = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RetrievalError(RuntimeError):
    """Base retrieval failure."""


class InvalidQueryError(RetrievalError, ValueError):
    """Invalid user query."""


class InvalidCandidateKError(RetrievalError, ValueError):
    """Invalid candidate depth."""


class InvalidDocumentIDError(RetrievalError, ValueError):
    """Invalid or unsafe uploaded-paper document ID."""


class EmptyIndexError(RetrievalError):
    """Requested vector index contains no vectors."""


class IndexNotFoundError(RetrievalError, FileNotFoundError):
    """Requested vector index does not exist."""


class MetadataNotFoundError(RetrievalError, FileNotFoundError):
    """Uploaded-paper metadata does not exist."""


class EncoderError(RetrievalError):
    """Query embedding generation/validation failed."""


class VectorSearchError(RetrievalError):
    """Vector search failed."""


class RetrievalResultError(RetrievalError):
    """Underlying search result violated the retriever contract."""


class DocumentMetadataError(RetrievalError):
    """Chunk metadata could not be resolved or was inconsistent."""


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalResult:
    """One validated Mode-1 dense-retrieval candidate.

    Optional semantic metadata is preserved for downstream reranking.
    """

    document_id: str
    score: float
    rank: int
    index_position: int
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def retrieval_score(self) -> float:
        """Compatibility alias consumed by reranking/ranking layers."""
        return self.score


@dataclass(frozen=True)
class ChunkRetrievalResult:
    """
    One validated Mode-2 uploaded-paper evidence chunk.

    `score` is the raw FAISS similarity score. It is not a probability.
    """

    document_id: str
    chunk_id: str
    score: float
    rank: int
    index_position: int
    section_id: str
    section_type: str
    section_heading: Optional[str]
    section_level: int
    start_page: Optional[int]
    end_page: Optional[int]
    source_pages: tuple[int, ...]
    parent_section_id: Optional[str]
    chunk_index: int
    token_count: int
    char_count: int
    overlap_with_previous: int
    chunking_method: str
    text: Optional[str]
    metadata: Mapping[str, Any]

    @property
    def paper_id(self) -> str:
        """Stable paper-level identity for downstream ranking/aggregation."""
        return self.document_id

    @property
    def page(self) -> Optional[int]:
        """Convenience source page for single-page chunks."""
        if self.start_page is not None:
            return self.start_page
        return self.source_pages[0] if self.source_pages else None

    @property
    def section(self) -> str:
        """Human-friendly section label."""
        return self.section_heading or self.section_type


@dataclass(frozen=True)
class _ChunkMetadataStore:
    by_position: Mapping[int, Mapping[str, Any]]
    by_chunk_id: Mapping[str, Mapping[str, Any]]
    document_id: str
    vector_count: int
    embedding_dimension: int
    vector_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Minimal dependency protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class QueryEncoderProtocol(Protocol):
    def embed_query(self, text: str) -> Any:
        """Embed one semantic query."""


@runtime_checkable
class VectorIndexProtocol(Protocol):
    @property
    def ntotal(self) -> int:
        """Number of indexed vectors."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""

    def search(
        self,
        query_embedding: Any,
        top_k: int,
        *,
        exclude_ids: Optional[set[str]] = None,
    ) -> Sequence[Any]:
        """Search the vector index."""


# ---------------------------------------------------------------------------
# Optional document metadata enrichment
# ---------------------------------------------------------------------------

DocumentMetadataResolver = Callable[[str], Any]


def _coerce_metadata_mapping(
    resolved: Any,
    *,
    document_id: str,
) -> Mapping[str, Any]:
    """Normalize mapping- or dataclass-style metadata into a mapping.

    The research metadata repository returns ``PaperMetadata`` objects, while
    lightweight integrations may return dictionaries.  The retriever accepts
    both without coupling itself to the repository implementation.
    """
    if isinstance(resolved, Mapping):
        return dict(resolved)

    # Native project PaperMetadata-style objects.
    object_document_id = getattr(resolved, "document_id", None)
    if object_document_id is not None and str(object_document_id) != document_id:
        raise DocumentMetadataError(
            f"Metadata document_id mismatch: requested={document_id!r}, "
            f"resolved={object_document_id!r}."
        )

    title = getattr(resolved, "title", None)
    summary = getattr(resolved, "summary", None)
    if isinstance(title, str) and isinstance(summary, str):
        return {
            "document_id": document_id,
            "title": title,
            "summary": summary,
        }

    raise DocumentMetadataError(
        f"Metadata resolver returned an unsupported value for "
        f"document {document_id!r}; expected a Mapping or metadata object "
        "with document_id/title/summary."
    )


def _normalize_optional_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _extract_title_summary(
    metadata: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    title = _normalize_optional_text(
        metadata.get("title") or metadata.get("paper_title")
    )
    summary = _normalize_optional_text(
        metadata.get("summary")
        or metadata.get("abstract")
        or metadata.get("paper_abstract")
    )
    return title, summary


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Generic dense retrieval orchestration layer.

    The injected encoder and index remain the authoritative implementations.
    """

    def __init__(
        self,
        encoder: QueryEncoderProtocol,
        index: VectorIndexProtocol,
        *,
        default_candidate_k: int = 50,
        max_candidate_k: Optional[int] = None,
        raise_on_empty_index: bool = True,
        metadata_resolver: Optional[DocumentMetadataResolver] = None,
    ) -> None:
        self._validate_encoder(encoder)
        self._validate_index(index)

        self._default_candidate_k = self._validate_candidate_k(
            default_candidate_k,
            field_name="default_candidate_k",
        )

        if max_candidate_k is not None:
            max_candidate_k = self._validate_candidate_k(
                max_candidate_k,
                field_name="max_candidate_k",
            )
            if max_candidate_k < self._default_candidate_k:
                raise InvalidCandidateKError(
                    "max_candidate_k cannot be smaller than "
                    f"default_candidate_k ({self._default_candidate_k})."
                )

        self._encoder = encoder
        self._index = index
        self._max_candidate_k = max_candidate_k
        self._raise_on_empty_index = bool(raise_on_empty_index)
        if metadata_resolver is not None and not callable(metadata_resolver):
            raise TypeError("metadata_resolver must be callable or None.")
        self._metadata_resolver = metadata_resolver

    @property
    def encoder(self) -> QueryEncoderProtocol:
        return self._encoder

    @property
    def index(self) -> VectorIndexProtocol:
        return self._index

    @property
    def default_candidate_k(self) -> int:
        return self._default_candidate_k

    @property
    def max_candidate_k(self) -> Optional[int]:
        return self._max_candidate_k

    @property
    def metadata_resolver(self) -> Optional[DocumentMetadataResolver]:
        return self._metadata_resolver

    @property
    def indexed_count(self) -> int:
        try:
            count = int(self._index.ntotal)
        except Exception as exc:
            raise VectorSearchError(
                "Unable to read the vector-index document count."
            ) from exc

        if count < 0:
            raise VectorSearchError(
                f"Vector index reported invalid negative count: {count}."
            )
        return count

    def retrieve(
        self,
        query: str,
        candidate_k: int | object = _DEFAULT_CANDIDATE_K,
        *,
        exclude_ids: Optional[Set[str]] = None,
        filters: Any = None,
    ) -> list[RetrievalResult]:
        """
        Mode-1 generic/research dense retrieval.

        Existing callers continue to use:
            retrieve(query, candidate_k=...)
        """
        normalized_query = self._validate_query(query)
        requested_k = self._resolve_candidate_k(candidate_k)
        exclusions = self._validate_exclusions(exclude_ids)

        if filters is not None:
            raise NotImplementedError(
                "Metadata filters are not implemented by DenseRetriever. "
                "Use a dedicated metadata/filter layer."
            )

        indexed_count = self.indexed_count
        if indexed_count == 0:
            message = (
                "Cannot retrieve because the vector index is empty. "
                "Build or load a populated index first."
            )
            if self._raise_on_empty_index:
                raise EmptyIndexError(message)
            logger.warning(message)
            return []

        if requested_k > indexed_count:
            logger.warning(
                "Requested candidate_k=%d but index contains %d vectors; "
                "retrieval will return at most %d.",
                requested_k,
                indexed_count,
                indexed_count,
            )

        query_embedding = self._embed_query(normalized_query)
        search_k = self._calculate_search_k(
            requested_k=requested_k,
            indexed_count=indexed_count,
            exclusion_count=len(exclusions),
        )

        raw_results = self._search_index(
            query_embedding=query_embedding,
            top_k=search_k,
            exclude_ids=exclusions,
        )

        return self._validate_and_structure_results(
            raw_results,
            requested_k=requested_k,
            exclude_ids=exclusions,
        )

    # Backward-friendly explicit alias.
    search = retrieve

    def _embed_query(self, query: str) -> Any:
        start = time.perf_counter()
        try:
            embedding = self._encoder.embed_query(query)
        except Exception as exc:
            logger.exception("Query embedding generation failed.")
            raise EncoderError(
                "Failed to generate the query embedding. "
                "Check the configured embedding encoder/model."
            ) from exc

        if embedding is None:
            raise EncoderError(
                "Embedding encoder returned None for a valid query."
            )

        self._validate_embedding_for_index(embedding, self._index)

        logger.debug(
            "Query embedding generated in %.4fs.",
            time.perf_counter() - start,
        )
        return embedding

    @staticmethod
    def _validate_embedding_for_index(
        embedding: Any,
        index: Any,
    ) -> None:
        try:
            import numpy as np

            vector = np.asarray(embedding)
        except Exception as exc:
            raise EncoderError(
                "Query embedding could not be converted to a numeric array."
            ) from exc

        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]

        if vector.ndim != 1:
            raise EncoderError(
                f"Query embedding must be one-dimensional; got shape "
                f"{vector.shape}."
            )

        if vector.size == 0:
            raise EncoderError("Query embedding is empty.")

        if vector.dtype.kind not in "fc":
            raise EncoderError(
                f"Query embedding must use a floating-point dtype; "
                f"got {vector.dtype}."
            )

        if not np.isfinite(vector).all():
            raise EncoderError(
                "Query embedding contains NaN or infinite values."
            )

        dimension = getattr(index, "dimension", None)
        if dimension is not None:
            try:
                expected = int(dimension)
            except Exception as exc:
                raise VectorSearchError(
                    "Vector index exposes an invalid embedding dimension."
                ) from exc

            if expected <= 0:
                raise VectorSearchError(
                    f"Vector index reported invalid dimension: {expected}."
                )

            if vector.shape[0] != expected:
                raise EncoderError(
                    "Query embedding dimension does not match the selected "
                    f"FAISS index: embedding={vector.shape[0]}, index={expected}."
                )

    def _search_index(
        self,
        *,
        query_embedding: Any,
        top_k: int,
        exclude_ids: set[str],
    ) -> Sequence[Any]:
        try:
            return self._index.search(
                query_embedding,
                top_k,
                exclude_ids=exclude_ids,
            )
        except TypeError as exc:
            message = str(exc)
            if "exclude_ids" not in message:
                raise
            try:
                return self._index.search(query_embedding, top_k)
            except Exception as inner:
                raise VectorSearchError(
                    "Dense vector search failed after compatibility fallback."
                ) from inner
        except Exception as exc:
            logger.exception("Vector search failed.")
            raise VectorSearchError(
                "Dense vector search failed. Check index/embedding "
                "compatibility and index integrity."
            ) from exc

    def _validate_and_structure_results(
        self,
        raw_results: Sequence[Any],
        *,
        requested_k: int,
        exclude_ids: set[str],
    ) -> list[RetrievalResult]:
        if raw_results is None:
            raise RetrievalResultError(
                "Vector index returned None instead of a result sequence."
            )

        try:
            results_list = list(raw_results)
        except TypeError as exc:
            raise RetrievalResultError(
                "Vector index returned a non-iterable result."
            ) from exc

        if len(results_list) > requested_k and not exclude_ids:
            raise RetrievalResultError(
                "Vector index returned more candidates than requested: "
                f"requested={requested_k}, returned={len(results_list)}."
            )

        structured: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        seen_positions: set[int] = set()

        for response_position, item in enumerate(results_list):
            document_id, score, index_position, metadata = self._parse_result(
                item,
                response_position=response_position,
            )

            if not metadata and self._metadata_resolver is not None:
                try:
                    resolved = self._metadata_resolver(document_id)
                except Exception as exc:
                    logger.exception(
                        "Document metadata resolution failed for %s.", document_id
                    )
                    raise DocumentMetadataError(
                        f"Could not resolve metadata for document {document_id!r}."
                    ) from exc
                if resolved is not None:
                    metadata = dict(
                        _coerce_metadata_mapping(
                            resolved,
                            document_id=document_id,
                        )
                    )

            title, summary = _extract_title_summary(metadata)

            if document_id in exclude_ids:
                continue

            if document_id in seen_ids:
                raise RetrievalResultError(
                    f"Duplicate document ID returned: {document_id!r}."
                )

            if index_position in seen_positions:
                raise RetrievalResultError(
                    f"Duplicate index position returned: {index_position}."
                )

            seen_ids.add(document_id)
            seen_positions.add(index_position)
            structured.append(
                RetrievalResult(
                    document_id=document_id,
                    score=score,
                    rank=0,
                    index_position=index_position,
                    title=title,
                    summary=summary,
                    metadata=metadata,
                )
            )

            if len(structured) >= requested_k:
                break

        structured.sort(
            key=lambda result: (-result.score, result.index_position)
        )

        return [
            RetrievalResult(
                document_id=result.document_id,
                score=result.score,
                rank=rank,
                index_position=result.index_position,
                    title=result.title,
                    summary=result.summary,
                    metadata=result.metadata,
            )
            for rank, result in enumerate(structured, start=1)
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def object_metadata_resolver(
        resolver: Callable[[str], Any],
    ) -> DocumentMetadataResolver:
        """Adapt an object-returning metadata repository to this contract.

        Example:
            metadata_resolver = DenseRetriever.object_metadata_resolver(
                metadata_store.get
            )
        """
        if not callable(resolver):
            raise TypeError("resolver must be callable.")

        def resolve(document_id: str) -> Optional[Mapping[str, Any]]:
            resolved = resolver(document_id)
            if resolved is None:
                return None
            return _coerce_metadata_mapping(
                resolved,
                document_id=document_id,
            )

        return resolve

    @staticmethod
    def mapping_metadata_resolver(
        metadata: Mapping[str, Mapping[str, Any]],
    ) -> DocumentMetadataResolver:
        """Create a constant-time resolver from an in-memory ID->metadata map."""
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping keyed by document ID.")
        snapshot = dict(metadata)

        def resolve(document_id: str) -> Optional[Mapping[str, Any]]:
            value = snapshot.get(document_id)
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise DocumentMetadataError(
                    f"Metadata entry for {document_id!r} is not a mapping."
                )
            return value

        return resolve

    @staticmethod
    def _validate_encoder(encoder: QueryEncoderProtocol) -> None:
        if encoder is None:
            raise TypeError("encoder cannot be None.")
        if not callable(getattr(encoder, "embed_query", None)):
            raise TypeError(
                "encoder must expose callable embed_query(text)."
            )

    @staticmethod
    def _validate_index(index: VectorIndexProtocol) -> None:
        if index is None:
            raise TypeError("index cannot be None.")
        if not callable(getattr(index, "search", None)):
            raise TypeError(
                "index must expose callable search(query_embedding, top_k, ...)."
            )
        if not hasattr(index, "ntotal"):
            raise TypeError("index must expose an ntotal property.")

    @staticmethod
    def _validate_candidate_k(
        candidate_k: int,
        *,
        field_name: str = "candidate_k",
    ) -> int:
        if (
            not isinstance(candidate_k, int)
            or isinstance(candidate_k, bool)
            or candidate_k <= 0
        ):
            raise InvalidCandidateKError(
                f"{field_name} must be a positive integer; "
                f"got {candidate_k!r}."
            )
        return candidate_k

    def _resolve_candidate_k(self, candidate_k: int | object) -> int:
        if candidate_k is _DEFAULT_CANDIDATE_K:
            return self._default_candidate_k
        if candidate_k is None:
            raise InvalidCandidateKError(
                "candidate_k cannot be None; omit it to use the configured "
                f"default ({self._default_candidate_k})."
            )

        value = self._validate_candidate_k(candidate_k)
        if (
            self._max_candidate_k is not None
            and value > self._max_candidate_k
        ):
            raise InvalidCandidateKError(
                f"candidate_k={value} exceeds max_candidate_k="
                f"{self._max_candidate_k}."
            )
        return value

    @staticmethod
    def _validate_query(query: str) -> str:
        if query is None:
            raise InvalidQueryError(
                "Query cannot be None."
            )
        if not isinstance(query, str):
            raise InvalidQueryError(
                f"Query must be a string; got {type(query).__name__}."
            )

        normalized = query.strip()
        if not normalized:
            raise InvalidQueryError(
                "Query cannot be empty or whitespace-only."
            )
        return normalized

    @staticmethod
    def _validate_exclusions(
        exclude_ids: Optional[Set[str]],
    ) -> set[str]:
        if exclude_ids is None:
            return set()
        if isinstance(exclude_ids, (str, bytes)):
            raise RetrievalResultError(
                "exclude_ids must be an iterable of IDs, not one raw string."
            )

        try:
            values = set(exclude_ids)
        except TypeError as exc:
            raise RetrievalResultError(
                "exclude_ids must be an iterable collection."
            ) from exc

        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise RetrievalResultError(
                    "Every exclude ID must be a non-empty string."
                )
        return values

    @staticmethod
    def _calculate_search_k(
        *,
        requested_k: int,
        indexed_count: int,
        exclusion_count: int,
    ) -> int:
        if exclusion_count == 0:
            return min(requested_k, indexed_count)
        return indexed_count

    @staticmethod
    def _parse_result(
        item: Any,
        *,
        response_position: int,
    ) -> tuple[str, float, int, Mapping[str, Any]]:
        if item is None:
            raise RetrievalResultError(
                f"Vector index returned None at position {response_position}."
            )

        if isinstance(item, Mapping):
            document_id = item.get("document_id")
            score = item.get("score")
            index_position = item.get("index_position")
            raw_metadata = item.get("metadata")
            if raw_metadata is None:
                raw_metadata = item.get("meta")
        else:
            document_id = getattr(item, "document_id", None)
            score = getattr(item, "score", None)
            index_position = getattr(item, "index_position", None)
            raw_metadata = getattr(item, "metadata", None)
            if raw_metadata is None:
                raw_metadata = getattr(item, "meta", None)

        if raw_metadata is None:
            metadata: Mapping[str, Any] = {}
        elif isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            raise RetrievalResultError(
                f"Invalid metadata for document {document_id!r}; expected a mapping."
            )

        if not isinstance(document_id, str) or not document_id.strip():
            raise RetrievalResultError(
                f"Invalid document ID at result position "
                f"{response_position}: {document_id!r}."
            )

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RetrievalResultError(
                f"Invalid numeric score for document {document_id!r}: {score!r}."
            )

        score_float = float(score)
        if not math.isfinite(score_float):
            raise RetrievalResultError(
                f"Non-finite similarity score for document "
                f"{document_id!r}: {score_float!r}."
            )

        if (
            isinstance(index_position, bool)
            or not isinstance(index_position, int)
            or index_position < 0
        ):
            raise RetrievalResultError(
                f"Invalid index position for document "
                f"{document_id!r}: {index_position!r}."
            )

        return document_id.strip(), score_float, index_position, metadata


# ---------------------------------------------------------------------------
# Uploaded-paper / Mode-2 retriever
# ---------------------------------------------------------------------------

class UploadedPaperRetriever:
    """
    Document-scoped Mode-2 retrieval.

    For every call, `document_id` resolves to exactly:
        <uploaded_root>/<document_id>/

    The selected index is loaded directly. There is intentionally no fallback
    to the research index and no cross-document filtering after a global search.
    """

    def __init__(
        self,
        encoder: QueryEncoderProtocol,
        *,
        uploaded_root: Path | str = DEFAULT_UPLOADED_INDEX_ROOT,
        candidate_k: int = 20,
        max_candidate_k: Optional[int] = 100,
        include_text: bool = True,
        index_loader: Any = None,
    ) -> None:
        if encoder is None:
            raise TypeError("encoder cannot be None.")
        if not callable(getattr(encoder, "embed_query", None)):
            raise TypeError(
                "encoder must expose callable embed_query(text)."
            )

        self._encoder = encoder
        self._uploaded_root = Path(uploaded_root)
        self._candidate_k = DenseRetriever._validate_candidate_k(
            candidate_k,
            field_name="candidate_k",
        )

        if max_candidate_k is not None:
            max_candidate_k = DenseRetriever._validate_candidate_k(
                max_candidate_k,
                field_name="max_candidate_k",
            )
            if max_candidate_k < self._candidate_k:
                raise InvalidCandidateKError(
                    "max_candidate_k cannot be smaller than candidate_k."
                )

        self._max_candidate_k = max_candidate_k
        self._include_text = bool(include_text)

        # Dependency injection is useful for tests and keeps this service from
        # knowing how FAISS is implemented.
        self._index_loader = index_loader or self._default_index_loader

    @property
    def encoder(self) -> QueryEncoderProtocol:
        return self._encoder

    @property
    def uploaded_root(self) -> Path:
        return self._uploaded_root

    def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        *,
        candidate_k: Optional[int] = None,
        return_candidate_pool: bool = False,
    ) -> list[ChunkRetrievalResult]:
        """
        Retrieve evidence chunks ONLY from the selected uploaded document.

        `top_k` is the normal retrieval-stage return count. When
        `return_candidate_pool=True`, the full validated `candidate_k` pool is
        returned so a downstream reranker can score the entire pool before the
        final ranking layer selects Top-K.
        """
        normalized_query = DenseRetriever._validate_query(query)
        safe_document_id = self._validate_document_id(document_id)
        requested_top_k = DenseRetriever._validate_candidate_k(
            top_k,
            field_name="top_k",
        )

        effective_candidate_k = (
            self._candidate_k if candidate_k is None else
            DenseRetriever._validate_candidate_k(
                candidate_k,
                field_name="candidate_k",
            )
        )

        if effective_candidate_k < requested_top_k:
            effective_candidate_k = requested_top_k

        if (
            self._max_candidate_k is not None
            and effective_candidate_k > self._max_candidate_k
        ):
            raise InvalidCandidateKError(
                f"candidate_k={effective_candidate_k} exceeds "
                f"max_candidate_k={self._max_candidate_k}."
            )

        index_directory = self._resolve_document_directory(
            safe_document_id
        )

        logger.info(
            "Starting document retrieval: document_id=%s "
            "query_chars=%d top_k=%d candidate_k=%d",
            safe_document_id,
            len(normalized_query),
            requested_top_k,
            effective_candidate_k,
        )

        # Resolve metadata BEFORE search so a malformed/missing paper index
        # fails clearly instead of producing opaque downstream errors.
        metadata = self._load_metadata(
            index_directory,
            document_id=safe_document_id,
        )

        if metadata.vector_count == 0:
            raise EmptyIndexError(
                f"Uploaded paper {safe_document_id!r} has no indexed chunks."
            )

        index = self._load_index(
            index_directory,
            document_id=safe_document_id,
            metadata=metadata,
        )

        # Mode-2 has two distinct identity levels:
        #   paper/document ID -> owning uploaded paper
        #   chunk ID          -> one FAISS vector
        # Validate the persisted manifest against metadata and the loaded
        # binary index before performing any search.
        self._validate_faiss_manifest(
            index_directory,
            document_id=safe_document_id,
            metadata=metadata,
            index=index,
        )

        indexed_count = self._read_index_count(index)
        if indexed_count == 0:
            raise EmptyIndexError(
                f"Uploaded paper {safe_document_id!r} has an empty vector index."
            )

        if indexed_count != metadata.vector_count:
            raise DocumentMetadataError(
                f"Index/metadata count mismatch for {safe_document_id!r}: "
                f"index={indexed_count}, metadata={metadata.vector_count}."
            )

        if effective_candidate_k > indexed_count:
            logger.warning(
                "Requested candidate_k=%d exceeds uploaded-paper vector "
                "count=%d; using %d.",
                effective_candidate_k,
                indexed_count,
                indexed_count,
            )
            effective_candidate_k = indexed_count

        # Query is embedded exactly once.
        try:
            query_embedding = self._encoder.embed_query(normalized_query)
        except Exception as exc:
            logger.exception(
                "Uploaded-paper query embedding failed."
            )
            raise EncoderError(
                "Failed to generate the uploaded-paper query embedding."
            ) from exc

        DenseRetriever._validate_embedding_for_index(
            query_embedding,
            index,
        )

        try:
            raw_results = index.search(
                query_embedding,
                effective_candidate_k,
            )
        except Exception as exc:
            logger.exception(
                "Uploaded-paper FAISS search failed: document_id=%s",
                safe_document_id,
            )
            raise VectorSearchError(
                f"Vector search failed for uploaded paper "
                f"{safe_document_id!r}."
            ) from exc

        candidates = self._validate_chunk_search_results(
            raw_results,
            document_id=safe_document_id,
            metadata=metadata,
            requested_k=effective_candidate_k,
        )

        # Candidate retrieval is deliberately not reranking.
        #
        # Normal callers receive top_k for backward compatibility. A downstream
        # reranker can explicitly request the complete candidate_k pool.
        returned = candidates if return_candidate_pool else candidates[:requested_top_k]

        logger.info(
            "Document retrieval completed: document_id=%s candidates=%d "
            "(candidate_pool=%s)",
            safe_document_id,
            len(returned),
            return_candidate_pool,
        )
        return returned

    # Concise aliases for callers that prefer a generic search name.
    search_for_document = retrieve_for_document

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------

    def _resolve_document_directory(self, document_id: str) -> Path:
        root = self._uploaded_root

        try:
            root_resolved = root.resolve(strict=False)
            target = (root / document_id).resolve(strict=False)
        except OSError as exc:
            raise IndexNotFoundError(
                "Could not safely resolve the uploaded-paper index path."
            ) from exc

        if target.parent != root_resolved:
            raise InvalidDocumentIDError(
                "document_id resolves outside the uploaded-index root."
            )

        if not target.is_dir():
            raise IndexNotFoundError(
                f"Uploaded-paper index directory not found for "
                f"document_id={document_id!r}: {target}"
            )

        return target

    @staticmethod
    def _validate_document_id(document_id: str) -> str:
        if not isinstance(document_id, str):
            raise InvalidDocumentIDError(
                "document_id must be a string."
            )

        value = document_id.strip()

        # Explicitly reject traversal and platform-specific path syntax.
        if (
            not value
            or not _SAFE_DOCUMENT_ID.fullmatch(value)
            or value in {".", ".."}
            or ".." in value
            or "/" in value
            or "\\" in value
            or ":" in value
            or Path(value).is_absolute()
        ):
            raise InvalidDocumentIDError(
                "document_id contains unsafe filesystem characters or "
                "path traversal syntax."
            )

        return value

    # ------------------------------------------------------------------
    # Existing FAISS loading contract
    # ------------------------------------------------------------------

    def _default_index_loader(
        self,
        index_directory: Path,
        *,
        document_id: str,
        metadata: _ChunkMetadataStore,
    ) -> Any:
        index_base = index_directory / "index"

        if not (
            Path(f"{index_base}.index").is_file()
            and Path(f"{index_base}.index.manifest.json").is_file()
        ):
            raise IndexNotFoundError(
                f"Uploaded-paper FAISS files are missing for "
                f"{document_id!r}: expected {index_base}.index and "
                f"{index_base}.index.manifest.json."
            )

        model_name = self._encoder_model_name()
        model_version = self._encoder_model_version()

        try:
            return FAISSVectorIndex.load(
                index_base,
                expected_model_name=model_name,
                expected_model_version=model_version,
                expected_document_count=metadata.vector_count,
            )
        except TypeError:
            # Older project versions may not expose expected_document_count.
            try:
                return FAISSVectorIndex.load(
                    index_base,
                    expected_model_name=model_name,
                    expected_model_version=model_version,
                )
            except FileNotFoundError as exc:
                raise IndexNotFoundError(
                    f"Uploaded-paper FAISS index not found for "
                    f"{document_id!r}."
                ) from exc
            except Exception as exc:
                raise VectorSearchError(
                    f"Failed to load uploaded-paper FAISS index for "
                    f"{document_id!r}: {exc}"
                ) from exc
        except FileNotFoundError as exc:
            raise IndexNotFoundError(
                f"Uploaded-paper FAISS index not found for "
                f"{document_id!r}."
            ) from exc
        except Exception as exc:
            raise VectorSearchError(
                f"Failed to load uploaded-paper FAISS index for "
                f"{document_id!r}: {exc}"
            ) from exc

    def _load_index(
        self,
        index_directory: Path,
        *,
        document_id: str,
        metadata: _ChunkMetadataStore,
    ) -> Any:
        try:
            index = self._index_loader(
                index_directory,
                document_id=document_id,
                metadata=metadata,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise VectorSearchError(
                f"Could not resolve uploaded-paper index for "
                f"{document_id!r}."
            ) from exc

        if index is None:
            raise IndexNotFoundError(
                f"Index loader returned no index for {document_id!r}."
            )

        validate = getattr(index, "validate", None)
        if callable(validate):
            try:
                validate()
            except Exception as exc:
                raise VectorSearchError(
                    f"Uploaded-paper FAISS index validation failed for "
                    f"{document_id!r}: {exc}"
                ) from exc

        return index

    @staticmethod
    def _read_index_count(index: Any) -> int:
        try:
            count = int(index.ntotal)
        except Exception as exc:
            raise VectorSearchError(
                "Uploaded-paper index does not expose a valid ntotal."
            ) from exc

        if count < 0:
            raise VectorSearchError(
                f"Uploaded-paper index reported invalid ntotal={count}."
            )
        return count

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(
        self,
        index_directory: Path,
        *,
        document_id: str,
    ) -> _ChunkMetadataStore:
        metadata_path = index_directory / "metadata.json"

        if not metadata_path.is_file():
            raise MetadataNotFoundError(
                f"Uploaded-paper metadata not found for {document_id!r}: "
                f"{metadata_path}"
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentMetadataError(
                f"Could not read uploaded-paper metadata for "
                f"{document_id!r}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise DocumentMetadataError(
                "Uploaded-paper metadata root must be a JSON object."
            )

        if payload.get("document_id") != document_id:
            raise DocumentMetadataError(
                "Uploaded-paper metadata document_id does not match "
                "the requested document."
            )

        id_contract = payload.get("id_contract")
        if id_contract is not None:
            if not isinstance(id_contract, dict):
                raise DocumentMetadataError(
                    "Uploaded-paper metadata id_contract must be an object."
                )
            if id_contract.get("vector_id") != "chunk_id":
                raise DocumentMetadataError(
                    "Uploaded-paper metadata id_contract must define "
                    "vector_id='chunk_id'."
                )
            if id_contract.get("mapping_key") != "chunks[].chunk_id":
                raise DocumentMetadataError(
                    "Uploaded-paper metadata id_contract has an invalid "
                    "mapping_key."
                )

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise DocumentMetadataError(
                "Uploaded-paper metadata must contain a 'chunks' array."
            )

        vector_count = payload.get("vector_count")
        dimension = payload.get("embedding_dimension")

        if (
            isinstance(vector_count, bool)
            or not isinstance(vector_count, int)
            or vector_count < 0
        ):
            raise DocumentMetadataError(
                "Uploaded-paper metadata contains invalid vector_count."
            )

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise DocumentMetadataError(
                "Uploaded-paper metadata contains invalid embedding_dimension."
            )

        if len(chunks) != vector_count:
            raise DocumentMetadataError(
                f"Metadata chunk count={len(chunks)} does not match "
                f"vector_count={vector_count}."
            )

        vector_ids_raw = payload.get("vector_ids")
        if vector_ids_raw is None:
            vector_ids_raw = [
                record.get("chunk_id")
                for record in chunks
                if isinstance(record, dict)
            ]

        if not isinstance(vector_ids_raw, list):
            raise DocumentMetadataError(
                "Uploaded-paper metadata vector_ids must be an array."
            )
        if len(vector_ids_raw) != vector_count:
            raise DocumentMetadataError(
                "Uploaded-paper metadata vector_ids count does not match "
                "vector_count."
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in vector_ids_raw
        ):
            raise DocumentMetadataError(
                "Uploaded-paper metadata vector_ids must contain non-empty "
                "strings."
            )

        by_position: dict[int, Mapping[str, Any]] = {}
        by_chunk_id: dict[str, Mapping[str, Any]] = {}

        for expected_position, record in enumerate(chunks):
            if not isinstance(record, dict):
                raise DocumentMetadataError(
                    f"Chunk metadata record {expected_position} is not an object."
                )

            required = (
                "vector_position",
                "chunk_id",
                "document_id",
                "section_id",
                "section_type",
                "chunk_index",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise DocumentMetadataError(
                    f"Chunk metadata record {expected_position} is missing "
                    f"fields: {missing}."
                )

            position = record["vector_position"]
            if position != expected_position:
                raise DocumentMetadataError(
                    "Chunk metadata vector positions are not contiguous."
                )

            if record["chunk_index"] != expected_position:
                raise DocumentMetadataError(
                    "Chunk metadata chunk_index does not match vector position."
                )

            if record["document_id"] != document_id:
                raise DocumentMetadataError(
                    "Chunk metadata contains evidence from another document."
                )

            chunk_id = record["chunk_id"]
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise DocumentMetadataError(
                    f"Invalid chunk_id at position {expected_position}."
                )
            if vector_ids_raw[expected_position] != chunk_id:
                raise DocumentMetadataError(
                    "Uploaded-paper metadata vector_ids are not aligned with "
                    f"chunks at vector position {expected_position}."
                )
            if chunk_id in by_chunk_id:
                raise DocumentMetadataError(
                    f"Duplicate chunk_id in metadata: {chunk_id!r}."
                )

            by_position[position] = record
            by_chunk_id[chunk_id] = record

        return _ChunkMetadataStore(
            by_position=by_position,
            by_chunk_id=by_chunk_id,
            document_id=document_id,
            vector_count=vector_count,
            embedding_dimension=dimension,
            vector_ids=tuple(str(value) for value in vector_ids_raw),
        )

    # ------------------------------------------------------------------
    # FAISS manifest / paper-ID <-> chunk-ID contract
    # ------------------------------------------------------------------

    def _validate_faiss_manifest(
        self,
        index_directory: Path,
        *,
        document_id: str,
        metadata: _ChunkMetadataStore,
        index: Any,
    ) -> None:
        """Validate paper identity and vector-level chunk-ID alignment."""
        manifest_path = index_directory / "index.index.manifest.json"
        if not manifest_path.is_file():
            raise IndexNotFoundError(
                f"Uploaded-paper FAISS manifest not found for "
                f"{document_id!r}: {manifest_path}"
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentMetadataError(
                f"Could not read uploaded-paper FAISS manifest for "
                f"{document_id!r}: {exc}"
            ) from exc

        if not isinstance(manifest, dict):
            raise DocumentMetadataError(
                "Uploaded-paper FAISS manifest root must be a JSON object."
            )

        manifest_paper_id = manifest.get("paper_document_id")
        if manifest_paper_id is None:
            manifest_paper_id = manifest.get("document_id")

        if manifest_paper_id != document_id:
            raise DocumentMetadataError(
                "Uploaded-paper FAISS manifest paper/document ID mismatch: "
                f"requested={document_id!r}, manifest={manifest_paper_id!r}."
            )

        manifest_count = manifest.get("vector_count")
        if manifest_count is None:
            manifest_count = manifest.get("document_count")
        if (
            isinstance(manifest_count, bool)
            or not isinstance(manifest_count, int)
            or manifest_count < 0
        ):
            raise DocumentMetadataError(
                "Uploaded-paper FAISS manifest contains invalid vector count."
            )
        if manifest_count != metadata.vector_count:
            raise DocumentMetadataError(
                "FAISS manifest/metadata vector-count mismatch: "
                f"manifest={manifest_count}, metadata={metadata.vector_count}."
            )

        manifest_dimension = manifest.get("embedding_dimension")
        if manifest_dimension is not None:
            if (
                isinstance(manifest_dimension, bool)
                or not isinstance(manifest_dimension, int)
                or manifest_dimension <= 0
            ):
                raise DocumentMetadataError(
                    "Uploaded-paper FAISS manifest contains invalid "
                    "embedding_dimension."
                )
            if manifest_dimension != metadata.embedding_dimension:
                raise DocumentMetadataError(
                    "FAISS manifest/metadata embedding-dimension mismatch: "
                    f"manifest={manifest_dimension}, "
                    f"metadata={metadata.embedding_dimension}."
                )

        manifest_vector_ids = manifest.get("vector_ids")
        manifest_document_ids = manifest.get("document_ids")

        if manifest_vector_ids is None and manifest_document_ids is None:
            raise DocumentMetadataError(
                "Uploaded-paper FAISS manifest must contain vector_ids or "
                "document_ids."
            )

        if manifest_vector_ids is not None:
            if not isinstance(manifest_vector_ids, list):
                raise DocumentMetadataError(
                    "Uploaded-paper FAISS manifest vector_ids must be an array."
                )
            if len(manifest_vector_ids) != metadata.vector_count:
                raise DocumentMetadataError(
                    "FAISS manifest vector-ID count does not match metadata."
                )
            if list(manifest_vector_ids) != list(metadata.vector_ids):
                raise DocumentMetadataError(
                    "FAISS manifest vector IDs do not match metadata chunk IDs "
                    "in exact vector order."
                )

        # `document_ids` is the legacy/native FAISS field used by the existing
        # project index implementation. In Mode-2 it MUST contain chunk IDs,
        # not the owning paper ID.
        if manifest_document_ids is not None:
            if not isinstance(manifest_document_ids, list):
                raise DocumentMetadataError(
                    "Uploaded-paper FAISS manifest document_ids must be an array."
                )
            if len(manifest_document_ids) != metadata.vector_count:
                raise DocumentMetadataError(
                    "FAISS manifest document-ID count does not match metadata."
                )
            if list(manifest_document_ids) != list(metadata.vector_ids):
                raise DocumentMetadataError(
                    "FAISS manifest document_ids do not match metadata chunk "
                    "IDs. A paper/document ID was likely written into a "
                    "vector-level field."
                )

        contract = manifest.get("id_contract")
        if contract is not None:
            if not isinstance(contract, dict):
                raise DocumentMetadataError(
                    "Uploaded-paper FAISS manifest id_contract must be an object."
                )
            if contract.get("vector_id") != "chunk_id":
                raise DocumentMetadataError(
                    "Uploaded-paper FAISS manifest id_contract must define "
                    "vector_id='chunk_id'."
                )

        actual_ids = getattr(index, "document_ids", None)
        if actual_ids is not None:
            try:
                actual_ids_list = list(actual_ids)
            except TypeError as exc:
                raise VectorSearchError(
                    "Loaded FAISS index exposes invalid document_ids."
                ) from exc
            if actual_ids_list != list(metadata.vector_ids):
                raise DocumentMetadataError(
                    "Loaded FAISS index vector IDs do not match persisted "
                    "chunk metadata."
                )

    # ------------------------------------------------------------------
    # Result validation and traceability
    # ------------------------------------------------------------------

    def _validate_chunk_search_results(
        self,
        raw_results: Sequence[Any],
        *,
        document_id: str,
        metadata: _ChunkMetadataStore,
        requested_k: int,
    ) -> list[ChunkRetrievalResult]:
        if raw_results is None:
            raise RetrievalResultError(
                "Uploaded-paper FAISS search returned None."
            )

        try:
            items = list(raw_results)
        except TypeError as exc:
            raise RetrievalResultError(
                "Uploaded-paper FAISS search returned a non-iterable result."
            ) from exc

        if len(items) > requested_k:
            raise RetrievalResultError(
                "Uploaded-paper FAISS returned more results than requested."
            )

        seen_chunk_ids: set[str] = set()
        seen_positions: set[int] = set()
        parsed: list[tuple[str, float, int, Mapping[str, Any]]] = []

        for response_position, item in enumerate(items):
            chunk_id, score, index_position = self._parse_chunk_result(
                item,
                response_position=response_position,
            )

            if chunk_id in seen_chunk_ids:
                raise RetrievalResultError(
                    f"Duplicate chunk_id returned by FAISS: {chunk_id!r}."
                )

            if index_position in seen_positions:
                raise RetrievalResultError(
                    f"Duplicate FAISS index position returned: "
                    f"{index_position}."
                )

            # Never silently repair a chunk-ID mismatch by vector position.
            # Position is a consistency check, not an ID substitution mechanism.
            record = metadata.by_chunk_id.get(chunk_id)

            if record is None:
                raise DocumentMetadataError(
                    f"FAISS result chunk_id={chunk_id!r} cannot be resolved "
                    "to uploaded-paper metadata."
                )

            if record["document_id"] != document_id:
                raise DocumentMetadataError(
                    "Resolved chunk metadata belongs to another document."
                )

            if record["chunk_id"] != chunk_id:
                raise DocumentMetadataError(
                    f"FAISS ID/metadata mismatch at position "
                    f"{index_position}: FAISS={chunk_id!r}, "
                    f"metadata={record['chunk_id']!r}."
                )

            if record["vector_position"] != index_position:
                raise DocumentMetadataError(
                    f"FAISS position/metadata mismatch for chunk "
                    f"{chunk_id!r}: FAISS={index_position}, "
                    f"metadata={record['vector_position']}."
                )

            seen_chunk_ids.add(chunk_id)
            seen_positions.add(index_position)
            parsed.append(
                (chunk_id, score, index_position, record)
            )

        # Stable deterministic ordering: score descending, then FAISS position.
        parsed.sort(key=lambda item: (-item[1], item[2]))

        results: list[ChunkRetrievalResult] = []
        for rank, (chunk_id, score, index_position, record) in enumerate(
            parsed[:requested_k],
            start=1,
        ):
            results.append(
                self._build_chunk_result(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank,
                    index_position=index_position,
                    record=record,
                )
            )

        return results

    @staticmethod
    def _parse_chunk_result(
        item: Any,
        *,
        response_position: int,
    ) -> tuple[str, float, int]:
        if isinstance(item, Mapping):
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                chunk_id = item.get("document_id")
            score = item.get("score")
            if score is None:
                score = item.get("retrieval_score")
            index_position = item.get("index_position")
        else:
            chunk_id = getattr(item, "chunk_id", None)
            if chunk_id is None:
                chunk_id = getattr(item, "document_id", None)
            score = getattr(item, "score", None)
            if score is None:
                score = getattr(item, "retrieval_score", None)
            index_position = getattr(item, "index_position", None)

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RetrievalResultError(
                f"Uploaded-paper FAISS result at position "
                f"{response_position} has invalid chunk_id/document_id."
            )

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RetrievalResultError(
                f"Uploaded-paper FAISS result for {chunk_id!r} has "
                f"invalid score={score!r}."
            )

        score_float = float(score)
        if not math.isfinite(score_float):
            raise RetrievalResultError(
                f"Uploaded-paper FAISS result for {chunk_id!r} has "
                f"non-finite score={score_float!r}."
            )

        if (
            isinstance(index_position, bool)
            or not isinstance(index_position, int)
            or index_position < 0
        ):
            raise RetrievalResultError(
                f"Uploaded-paper FAISS result for {chunk_id!r} has "
                f"invalid index_position={index_position!r}."
            )

        return chunk_id.strip(), score_float, index_position

    def _build_chunk_result(
        self,
        *,
        document_id: str,
        chunk_id: str,
        score: float,
        rank: int,
        index_position: int,
        record: Mapping[str, Any],
    ) -> ChunkRetrievalResult:
        def required_text(key: str) -> str:
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                raise DocumentMetadataError(
                    f"{key} for chunk {chunk_id!r} must be a non-empty string."
                )
            return value.strip()

        def optional_int(
            key: str,
            default: Optional[int] = None,
        ) -> Optional[int]:
            value = record.get(key, default)
            if value is None:
                return None
            if isinstance(value, bool):
                raise DocumentMetadataError(
                    f"{key} for chunk {chunk_id!r} must be an integer."
                )
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DocumentMetadataError(
                    f"{key} for chunk {chunk_id!r} is not a valid integer."
                ) from exc
            if parsed < 0:
                raise DocumentMetadataError(
                    f"{key} for chunk {chunk_id!r} cannot be negative."
                )
            return parsed

        vector_position = optional_int("vector_position")
        if vector_position != index_position:
            raise DocumentMetadataError(
                f"Chunk {chunk_id!r} vector_position={vector_position} does not "
                f"match FAISS index_position={index_position}."
            )

        section_id = required_text("section_id")
        section_type = required_text("section_type")

        chunk_index = optional_int("chunk_index")
        if chunk_index is None or chunk_index < 0:
            raise DocumentMetadataError(
                f"chunk_index for chunk {chunk_id!r} must be non-negative."
            )

        source_pages_raw = record.get("source_pages") or []
        if not isinstance(source_pages_raw, list):
            raise DocumentMetadataError(
                f"source_pages for chunk {chunk_id!r} is not an array."
            )

        source_pages_list: list[int] = []
        for page in source_pages_raw:
            if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
                raise DocumentMetadataError(
                    f"source_pages for chunk {chunk_id!r} must contain "
                    "positive integers."
                )
            source_pages_list.append(page)
        source_pages = tuple(source_pages_list)

        text_value = record.get("text") if self._include_text else None
        if text_value is not None and not isinstance(text_value, str):
            raise DocumentMetadataError(
                f"text for chunk {chunk_id!r} is not a string."
            )

        metadata_value = record.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise DocumentMetadataError(
                f"metadata for chunk {chunk_id!r} is not an object."
            )

        return ChunkRetrievalResult(
            document_id=document_id,
            chunk_id=chunk_id,
            score=score,
            rank=rank,
            index_position=index_position,
            section_id=section_id,
            section_type=section_type,
            section_heading=(
                str(record["section_heading"])
                if record.get("section_heading") is not None
                else None
            ),
            section_level=(
                optional_int("section_level", 1) or 1
            ),
            start_page=optional_int("start_page"),
            end_page=optional_int("end_page"),
            source_pages=source_pages,
            parent_section_id=(
                str(record["parent_section_id"])
                if record.get("parent_section_id") is not None
                else None
            ),
            chunk_index=chunk_index,
            token_count=optional_int("token_count", 0) or 0,
            char_count=optional_int("char_count", 0) or 0,
            overlap_with_previous=optional_int("overlap_with_previous", 0) or 0,
            chunking_method=str(
                record.get("chunking_method", "unknown")
            ),
            text=text_value,
            metadata=dict(metadata_value),
        )

    # ------------------------------------------------------------------
    # Encoder metadata compatibility
    # ------------------------------------------------------------------

    def _encoder_model_name(self) -> Optional[str]:
        value = getattr(self._encoder, "model_name", None)
        return value if isinstance(value, str) else None

    def _encoder_model_version(self) -> Optional[str]:
        value = getattr(self._encoder, "model_version", None)
        return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Unified service facade
# ---------------------------------------------------------------------------

class PaperRetriever:
    """
    Optional facade exposing explicit Mode-1 and Mode-2 methods.

    It does not introduce an ambiguous `mode="1"/"2"` switch.
    """

    def __init__(
        self,
        *,
        research_retriever: DenseRetriever,
        uploaded_retriever: UploadedPaperRetriever,
    ) -> None:
        self.research = research_retriever
        self.uploaded = uploaded_retriever

    def retrieve(
        self,
        query: str,
        candidate_k: int | object = _DEFAULT_CANDIDATE_K,
        *,
        exclude_ids: Optional[Set[str]] = None,
        filters: Any = None,
    ) -> list[RetrievalResult]:
        return self.research.retrieve(
            query,
            candidate_k,
            exclude_ids=exclude_ids,
            filters=filters,
        )

    def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        *,
        candidate_k: Optional[int] = None,
        return_candidate_pool: bool = False,
    ) -> list[ChunkRetrievalResult]:
        return self.uploaded.retrieve_for_document(
            query,
            document_id,
            top_k,
            candidate_k=candidate_k,
            return_candidate_pool=return_candidate_pool,
        )


Retriever = DenseRetriever


# ---------------------------------------------------------------------------
# Deterministic model-free self-test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MockSearchResult:
    document_id: str
    score: float
    index_position: int


class _MockEncoder:
    def __init__(self, vector: Any) -> None:
        self.vector = vector
        self.calls = 0
        self.model_name = "mock"
        self.model_version = "test"

    def embed_query(self, text: str) -> Any:
        self.calls += 1
        return self.vector


class _MockIndex:
    def __init__(
        self,
        results: Sequence[_MockSearchResult],
        ntotal: int,
        dimension: int = 3,
    ) -> None:
        self.results = list(results)
        self.ntotal = ntotal
        self.dimension = dimension
        self.calls = 0
        self.last_top_k: Optional[int] = None

    def search(
        self,
        query_embedding: Any,
        top_k: int,
        *,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[_MockSearchResult]:
        self.calls += 1
        self.last_top_k = top_k
        excluded = exclude_ids or set()

        candidates = [
            item
            for item in self.results
            if item.document_id not in excluded
        ]
        candidates.sort(
            key=lambda item: (-float(item.score), item.index_position)
        )
        return candidates[:top_k]


class _MockUploadedIndex(_MockIndex):
    def __init__(
        self,
        results: Sequence[_MockSearchResult],
        ntotal: int,
        dimension: int = 3,
    ) -> None:
        super().__init__(results, ntotal, dimension)
        self.document_ids = [item.document_id for item in self.results]

    def validate(self) -> None:
        return None


def run_self_test() -> None:
    """Run unit-style tests without SPECTER2, FAISS, or a real corpus."""
    encoder = _MockEncoder([0.1, 0.2, 0.3])

    index = _MockIndex(
        [
            _MockSearchResult("paper-c", 0.80, 2),
            _MockSearchResult("paper-a", 0.95, 0),
            _MockSearchResult("paper-b", 0.90, 1),
        ],
        ntotal=3,
    )

    retriever = DenseRetriever(
        encoder,
        index,
        default_candidate_k=2,
    )

    results = retriever.retrieve(
        "  deep learning for medical image segmentation  "
    )

    assert encoder.calls == 1
    assert index.calls == 1
    assert index.last_top_k == 2
    assert [r.document_id for r in results] == ["paper-a", "paper-b"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].title is None

    metadata_resolver = DenseRetriever.mapping_metadata_resolver(
        {
            "paper-a": {
                "title": "Medical image segmentation with transformers",
                "abstract": "A transformer method for image segmentation.",
            }
        }
    )
    enriched_retriever = DenseRetriever(
        encoder,
        index,
        default_candidate_k=1,
        metadata_resolver=metadata_resolver,
    )
    enriched = enriched_retriever.retrieve("segmentation")
    assert enriched[0].title == "Medical image segmentation with transformers"
    assert enriched[0].summary == "A transformer method for image segmentation."
    assert enriched[0].retrieval_score == enriched[0].score

    @dataclass(frozen=True)
    class _PaperMetadata:
        document_id: str
        title: str
        summary: str

    object_resolver = DenseRetriever.object_metadata_resolver(
        lambda document_id: _PaperMetadata(
            document_id=document_id,
            title="Object title",
            summary="Object summary",
        )
    )
    object_retriever = DenseRetriever(
        encoder,
        index,
        default_candidate_k=1,
        metadata_resolver=object_resolver,
    )
    object_result = object_retriever.retrieve("object metadata")
    assert object_result[0].title == "Object title"
    assert object_result[0].summary == "Object summary"

    excluded = retriever.retrieve(
        "related paper query",
        candidate_k=2,
        exclude_ids={"paper-a"},
    )
    assert [r.document_id for r in excluded] == ["paper-b", "paper-c"]

    for invalid_query in ("   ", None, 123):
        try:
            retriever.retrieve(invalid_query)  # type: ignore[arg-type]
        except InvalidQueryError:
            pass
        else:
            raise AssertionError("Invalid query was accepted.")

    for invalid_k in (0, -1, None):
        try:
            retriever.retrieve("valid", candidate_k=invalid_k)
        except InvalidCandidateKError:
            pass
        else:
            raise AssertionError("Invalid candidate_k was accepted.")

    empty = DenseRetriever(
        encoder,
        _MockIndex([], ntotal=0),
    )
    try:
        empty.retrieve("valid")
    except EmptyIndexError:
        pass
    else:
        raise AssertionError("Empty index was not rejected.")

    bad_score = DenseRetriever(
        encoder,
        _MockIndex(
            [_MockSearchResult("paper-x", float("nan"), 0)],
            ntotal=1,
        ),
    )
    try:
        bad_score.retrieve("valid")
    except RetrievalResultError:
        pass
    else:
        raise AssertionError("NaN score was not rejected.")

    duplicate = DenseRetriever(
        encoder,
        _MockIndex(
            [
                _MockSearchResult("paper-x", 0.9, 0),
                _MockSearchResult("paper-x", 0.8, 1),
            ],
            ntotal=2,
        ),
    )
    try:
        duplicate.retrieve("valid", candidate_k=2)
    except RetrievalResultError:
        pass
    else:
        raise AssertionError("Duplicate result was not rejected.")

    # Uploaded-paper candidate-pool contract:
    # candidate_k must survive until a downstream reranker when explicitly
    # requested, while ordinary callers retain top_k behavior.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        document_id = "paper-1"
        paper_dir = root / document_id
        paper_dir.mkdir(parents=True)

        chunks = []
        for pos, (chunk_id, section_type, content, page) in enumerate(
            (
                ("c0", "abstract", "alpha", 1),
                ("c1", "method", "beta", 2),
                ("c2", "results", "gamma", 3),
            )
        ):
            chunks.append(
                {
                    "vector_position": pos,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "section_id": f"s{pos}",
                    "section_type": section_type,
                    "section_heading": section_type.title(),
                    "chunk_index": pos,
                    "text": content,
                    "source_pages": [page],
                    "metadata": {},
                }
            )

        (paper_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "vector_count": 3,
                    "embedding_dimension": 3,
                    "id_contract": {
                        "document_id": "paper/document owner ID",
                        "vector_id": "chunk_id",
                        "mapping_field": "chunks[].vector_position",
                        "mapping_key": "chunks[].chunk_id",
                    },
                    "vector_ids": ["c0", "c1", "c2"],
                    "chunks": chunks,
                }
            ),
            encoding="utf-8",
        )

        (paper_dir / "index.index.manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "2",
                    "paper_document_id": document_id,
                    "document_id": document_id,
                    "vector_count": 3,
                    "document_count": 3,
                    "embedding_dimension": 3,
                    "document_ids": ["c0", "c1", "c2"],
                    "vector_ids": ["c0", "c1", "c2"],
                    "id_contract": {
                        "document_id": "paper/document owner ID",
                        "vector_id": "chunk_id",
                    },
                }
            ),
            encoding="utf-8",
        )

        uploaded_index = _MockUploadedIndex(
            [
                _MockSearchResult("c0", 0.90, 0),
                _MockSearchResult("c1", 0.80, 1),
                _MockSearchResult("c2", 0.70, 2),
            ],
            ntotal=3,
            dimension=3,
        )

        uploaded = UploadedPaperRetriever(
            encoder,
            uploaded_root=root,
            candidate_k=3,
            index_loader=lambda *_args, **_kwargs: uploaded_index,
        )

        pool = uploaded.retrieve_for_document(
            "what method was used?",
            document_id,
            top_k=1,
            candidate_k=3,
            return_candidate_pool=True,
        )
        assert len(pool) == 3
        assert [item.chunk_id for item in pool] == ["c0", "c1", "c2"]

        final = uploaded.retrieve_for_document(
            "what method was used?",
            document_id,
            top_k=1,
            candidate_k=3,
        )
        assert len(final) == 1
        assert final[0].chunk_id == "c0"

        # Regression: a paper/document ID must never be accepted as a FAISS
        # vector ID in place of a chunk ID.
        broken_manifest = json.loads(
            (paper_dir / "index.index.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        broken_manifest["document_ids"] = [document_id, "c1", "c2"]
        try:
            (paper_dir / "index.index.manifest.json").write_text(
                json.dumps(broken_manifest),
                encoding="utf-8",
            )
            try:
                uploaded.retrieve_for_document(
                    "what method was used?",
                    document_id,
                    top_k=1,
                )
            except DocumentMetadataError:
                pass
            else:
                raise AssertionError(
                    "Paper/document ID was incorrectly accepted as a FAISS "
                    "vector ID."
                )
        finally:
            (paper_dir / "index.index.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2",
                        "paper_document_id": document_id,
                        "document_id": document_id,
                        "vector_count": 3,
                        "document_count": 3,
                        "embedding_dimension": 3,
                        "document_ids": ["c0", "c1", "c2"],
                        "vector_ids": ["c0", "c1", "c2"],
                        "id_contract": {
                            "document_id": "paper/document owner ID",
                            "vector_id": "chunk_id",
                        },
                    }
                ),
                encoding="utf-8",
            )

    # Uploaded-paper document-ID validation.
    for bad_id in ("../paper", "..\\paper", "/absolute", r"C:\paper", ""):
        try:
            UploadedPaperRetriever(
                encoder,
                uploaded_root=Path("indexes/uploaded"),
            )._validate_document_id(bad_id)
        except InvalidDocumentIDError:
            pass
        else:
            raise AssertionError(
                f"Unsafe document ID accepted: {bad_id!r}"
            )

    print("Retriever self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()