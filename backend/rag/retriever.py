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
    - perform model-based reranking
    - build RAG context
    - call an LLM

Uploaded-paper retrieval does perform a lightweight, deterministic
query-aware prioritization pass over the already retrieved FAISS candidates.
It is not a second retrieval system and never changes the raw FAISS score.

The uploaded-paper path is explicitly document-scoped. It never falls back
to the research index or another uploaded paper.

RAG boundary contract
---------------------
RAGRetriever.retrieve() always returns RAGRetrievalResult with a canonical
``candidates`` tuple. This is the strict boundary consumed by RAGPipeline;
``results`` and sequence behavior remain backward-compatible.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Set, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_K = object()

DEFAULT_RESEARCH_INDEX_BASE = Path("indexes") / "research" / "index"
DEFAULT_UPLOADED_INDEX_ROOT = Path("indexes") / "uploaded"

_SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# ---------------------------------------------------------------------------
# Query-aware uploaded-paper retrieval
# ---------------------------------------------------------------------------
#
# FAISS remains the primary semantic retriever.  The uploaded-paper path also
# applies a deterministic lexical/answer-shape signal to the already retrieved
# candidates.  This fixes a real failure mode observed in production:
# an exact answer-bearing chunk was retrieved, but several neighboring chunks
# with slightly higher cosine scores were delivered first to the LLM.
#
# No second embedding model, external service, or cross-document search is
# introduced.  Document isolation remains absolute.
_RETRIEVAL_DENSE_WEIGHT = 0.55
_RETRIEVAL_LEXICAL_WEIGHT = 0.25
_RETRIEVAL_INTENT_WEIGHT = 0.20
_RETRIEVAL_MAX_LEXICAL_TERMS = 32

# A high intent score means the chunk explicitly looks like the requested
# count/list/type answer. Such candidates receive a deterministic priority
# bucket before the continuous hybrid score. This prevents an exact
# answer-bearing passage from being buried by semantically similar neighbors.
_RETRIEVAL_ANSWER_PRIORITY_THRESHOLD = 0.75
_RETRIEVAL_ANSWER_BOOST = 0.35

_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "two", "was", "were", "what", "when",
        "where", "which", "who", "why", "with",
    }
)

_LIST_QUESTION_RE = re.compile(
    r"\b(?:what|which)\b.{0,100}\b(?:two|three|four|five|types?|forms?|"
    r"kinds?|categories|classes|methods?|approaches?|factors?|steps?|"
    r"components?)\b",
    re.IGNORECASE,
)

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _normalize_retrieval_text(value: Any) -> str:
    """Normalize OCR/PDF text without changing word boundaries."""
    if value is None:
        return ""
    text_value = str(value).replace("\u00ad", " ")
    return re.sub(r"\s+", " ", text_value).strip().lower()


def _retrieval_tokens(value: str) -> tuple[str, ...]:
    normalized = _normalize_retrieval_text(value)
    tokens = re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", normalized)
    return tuple(
        token
        for token in tokens
        if len(token) > 1 and token not in _QUERY_STOPWORDS
    )


def _query_term_set(query: str) -> tuple[str, ...]:
    terms = list(dict.fromkeys(_retrieval_tokens(query)))
    return tuple(terms[:_RETRIEVAL_MAX_LEXICAL_TERMS])


def _lexical_relevance(query: str, text: str) -> float:
    """Return a deterministic lexical relevance signal in [0, 1]."""
    query_terms = set(_query_term_set(query))
    if not query_terms:
        return 0.0

    text_norm = _normalize_retrieval_text(text)
    text_terms = set(_retrieval_tokens(text_norm))
    if not text_terms:
        return 0.0

    overlap = len(query_terms & text_terms) / len(query_terms)

    # Reward preserved adjacent query phrases.
    query_tokens = _retrieval_tokens(query)
    phrase_hits = 0
    if len(query_tokens) >= 2:
        for left, right in zip(query_tokens, query_tokens[1:]):
            if f"{left} {right}" in text_norm:
                phrase_hits += 1

    phrase_score = (
        phrase_hits / max(1, len(query_tokens) - 1)
        if query_tokens
        else 0.0
    )

    return min(1.0, 0.72 * overlap + 0.28 * phrase_score)


def _answer_intent_score(query: str, text: str) -> float:
    """
    Detect whether a chunk looks like the answer-bearing passage for a
    count/list question.

    This is intentionally conservative: it boosts candidates that explicitly
    state the requested count/type, but does not manufacture an answer.
    """
    query_norm = _normalize_retrieval_text(query)
    text_norm = _normalize_retrieval_text(text)

    if not query_norm or not text_norm:
        return 0.0

    if not _LIST_QUESTION_RE.search(query_norm):
        return 0.0

    score = 0.15

    requested_count = next(
        (
            number
            for word, number in _COUNT_WORDS.items()
            if re.search(rf"\b{word}\b", query_norm)
        ),
        None,
    )

    requested_type = next(
        (
            word
            for word in (
                "forms",
                "types",
                "kinds",
                "categories",
                "classes",
                "methods",
                "approaches",
                "factors",
                "steps",
                "components",
            )
            if re.search(rf"\b{word}\b", query_norm)
        ),
        None,
    )

    if requested_count and requested_type:
        count_word = next(
            word
            for word, number in _COUNT_WORDS.items()
            if number == requested_count
        )

        # Strongest signal: explicit count + requested type.
        if re.search(
            rf"\b(?:there\s+are|there\s+is|are|is|have|has)\s+"
            rf"{count_word}\b.{{0,80}}\b{requested_type}\b",
            text_norm,
        ):
            score += 0.55

        # Still useful when the source uses a less direct construction.
        elif re.search(
            rf"\b{count_word}\b.{{0,100}}\b{requested_type}\b",
            text_norm,
        ):
            score += 0.35

    if re.search(
        r"\b(?:forms?|types?|kinds?|categories|classes)\b.{0,100}"
        r"\b(?:are|include|consist|classified)\b",
        text_norm,
    ):
        score += 0.15

    # Explicit "two general forms" is a very strong answer-bearing pattern.
    if re.search(
        r"\b(?:two|three|four|five)\s+general\s+forms?\b",
        text_norm,
    ):
        score += 0.45

    return min(1.0, score)


def _normalize_dense_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize finite dense scores without changing their order."""
    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return [1.0] * len(scores)

    span = high - low
    return [(score - low) / span for score in scores]


def _hybrid_candidate_score(
    *,
    dense_score: float,
    lexical_score: float,
    intent_score: float,
) -> float:
    return (
        _RETRIEVAL_DENSE_WEIGHT * dense_score
        + _RETRIEVAL_LEXICAL_WEIGHT * lexical_score
        + _RETRIEVAL_INTENT_WEIGHT * intent_score
    )



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
    """One validated Mode-1 dense-retrieval candidate."""

    document_id: str
    score: float
    rank: int
    index_position: int


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


@dataclass(frozen=True)
class RAGRetrievalResult:
    """Canonical result returned by RAGRetriever to RAGPipeline.

    ``candidates`` is the authoritative field consumed by the pipeline.
    ``results`` is a compatibility alias for older integrations.

    The object is intentionally sequence-compatible so legacy code that
    previously treated RAGRetriever.retrieve() as a list keeps working.
    """

    candidates: tuple[Any, ...]
    scope: str
    task_type: str
    query: str
    document_id: Optional[str] = None
    document_ids: tuple[str, ...] = ()

    @property
    def results(self) -> tuple[Any, ...]:
        """Backward-compatible alias for ``candidates``."""
        return self.candidates

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, item):
        return self.candidates[item]


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
            document_id, score, index_position = self._parse_result(
                item,
                response_position=response_position,
            )

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
            )
            for rank, result in enumerate(structured, start=1)
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

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
    ) -> tuple[str, float, int]:
        if item is None:
            raise RetrievalResultError(
                f"Vector index returned None at position {response_position}."
            )

        if isinstance(item, Mapping):
            document_id = item.get("document_id")
            score = item.get("score")
            index_position = item.get("index_position")
        else:
            document_id = getattr(item, "document_id", None)
            score = getattr(item, "score", None)
            index_position = getattr(item, "index_position", None)

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

        return document_id, score_float, index_position


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

    def document_exists(self, document_id: str) -> bool:
        """Return whether a safely named uploaded-paper directory exists.

        This is a non-throwing availability probe for API layers such as
        ``/api/qa``. It performs the same document-ID and root containment
        checks used by retrieval, so API availability checks cannot silently
        disagree with the actual retrieval path.
        """
        try:
            safe_document_id = self._validate_document_id(document_id)
            self._resolve_document_directory(safe_document_id)
        except (InvalidDocumentIDError, IndexNotFoundError, OSError):
            return False
        return True

    def validate_document_available(self, document_id: str) -> Path:
        """Validate and return the authoritative uploaded-paper directory."""
        safe_document_id = self._validate_document_id(document_id)
        return self._resolve_document_directory(safe_document_id)

    def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        *,
        candidate_k: Optional[int] = None,
    ) -> list[ChunkRetrievalResult]:
        """
        Retrieve evidence chunks ONLY from the selected uploaded document.

        `top_k` is the number returned by this retrieval stage. `candidate_k`
        may be larger when the caller wants a broader candidate set for a
        downstream reranker.
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

        # FAISS semantic similarity is still the primary signal.  We now add a
        # deterministic lexical/answer-shape pass over the retrieved chunks so
        # an exact answer-bearing passage is not buried beneath neighboring
        # passages that merely discuss the same topic.
        candidates = self._prioritize_answer_bearing_candidates(
            normalized_query,
            candidates,
        )

        candidates = candidates[:requested_top_k]

        logger.info(
            "Document retrieval completed: document_id=%s candidates=%d",
            safe_document_id,
            len(candidates),
        )
        return candidates

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
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position != expected_position
            ):
                raise DocumentMetadataError(
                    "Chunk metadata vector_position must be the expected "
                    "non-negative contiguous integer."
                )

            chunk_index = record["chunk_index"]
            if (
                isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index != expected_position
            ):
                raise DocumentMetadataError(
                    "Chunk metadata chunk_index must match vector position."
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

            record = metadata.by_chunk_id.get(chunk_id)
            if record is None:
                # Prefer stable IDs for source traceability, but if the FAISS
                # result has a position that maps to the exact same chunk ID,
                # accept it. This guards against stale/partial ID mappings.
                record = metadata.by_position.get(index_position)

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
    def _prioritize_answer_bearing_candidates(
        query: str,
        candidates: Sequence[ChunkRetrievalResult],
    ) -> list[ChunkRetrievalResult]:
        """
        Deterministically prioritize answer-bearing uploaded-paper chunks.

        FAISS remains the primary retriever. This method only reranks the
        candidates that FAISS has already returned.

        Invariants:
        - raw ChunkRetrievalResult.score is never changed;
        - document_id/chunk_id/index_position/text/metadata are preserved;
        - the returned rank values are always contiguous starting at 1;
        - ordering is deterministic for equal scores;
        - empty candidate sets are returned unchanged as an empty list.

        The answer-priority bucket is intentionally conservative: it is only
        activated by the existing count/list intent detector and a strong
        answer-shape signal. Normal semantic retrieval therefore remains
        primarily dense, while explicit "what are the two forms/types..."
        questions can surface the passage that actually states the answer.
        """
        normalized_query = DenseRetriever._validate_query(query)

        if candidates is None:
            raise RetrievalResultError(
                "Uploaded-paper candidate collection cannot be None."
            )

        try:
            items = list(candidates)
        except TypeError as exc:
            raise RetrievalResultError(
                "Uploaded-paper candidates must be an iterable sequence."
            ) from exc

        if not items:
            return []

        dense_scores: list[float] = []
        for position, candidate in enumerate(items):
            if not isinstance(candidate, ChunkRetrievalResult):
                raise RetrievalResultError(
                    "Uploaded-paper candidate at position "
                    f"{position} is not a ChunkRetrievalResult."
                )

            score = candidate.score
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RetrievalResultError(
                    f"Uploaded-paper candidate {candidate.chunk_id!r} has "
                    f"an invalid raw FAISS score: {score!r}."
                )

            score_float = float(score)
            if not math.isfinite(score_float):
                raise RetrievalResultError(
                    f"Uploaded-paper candidate {candidate.chunk_id!r} has "
                    f"a non-finite raw FAISS score: {score_float!r}."
                )

            dense_scores.append(score_float)

        normalized_dense = _normalize_dense_scores(dense_scores)

        ranked: list[
            tuple[
                int,
                float,
                float,
                float,
                float,
                int,
                ChunkRetrievalResult,
            ]
        ] = []

        for original_position, (
            candidate,
            dense_score,
        ) in enumerate(zip(items, normalized_dense)):
            text = candidate.text or ""

            lexical_score = _lexical_relevance(
                normalized_query,
                text,
            )
            intent_score = _answer_intent_score(
                normalized_query,
                text,
            )
            hybrid_score = _hybrid_candidate_score(
                dense_score=dense_score,
                lexical_score=lexical_score,
                intent_score=intent_score,
            )

            answer_priority = int(
                intent_score >= _RETRIEVAL_ANSWER_PRIORITY_THRESHOLD
            )

            # The boost is secondary to the explicit priority bucket. It helps
            # order multiple answer-bearing candidates without replacing the
            # dense/lexical signals entirely.
            final_score = (
                hybrid_score
                + _RETRIEVAL_ANSWER_BOOST * intent_score
            )

            ranked.append(
                (
                    answer_priority,
                    final_score,
                    intent_score,
                    lexical_score,
                    dense_score,
                    original_position,
                    candidate,
                )
            )

        # Deterministic ordering:
        #   1. explicit answer-bearing candidates first;
        #   2. hybrid score descending;
        #   3. answer-intent descending;
        #   4. lexical relevance descending;
        #   5. normalized dense score descending;
        #   6. original FAISS response position.
        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                -item[3],
                -item[4],
                item[5],
            )
        )

        return [
            replace(candidate, rank=rank)
            for rank, (
                _answer_priority,
                _final_score,
                _intent_score,
                _lexical_score,
                _dense_score,
                _original_position,
                candidate,
            ) in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _parse_chunk_result(
        item: Any,
        *,
        response_position: int,
    ) -> tuple[str, float, int]:
        if isinstance(item, Mapping):
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                # Compatibility with FAISSVectorIndex results that historically
                # exposed the logical chunk ID through document_id.
                chunk_id = item.get("document_id")
            score = item.get("score")
            index_position = item.get("index_position")
        else:
            chunk_id = getattr(item, "chunk_id", None)
            if chunk_id is None:
                chunk_id = getattr(item, "document_id", None)
            score = getattr(item, "score", None)
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

    @staticmethod
    def _validated_nonnegative_int(
        value: Any,
        *,
        field_name: str,
        chunk_id: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DocumentMetadataError(
                f"{field_name} for chunk {chunk_id!r} must be a "
                "non-negative integer."
            )
        return value

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
        def optional_int(key: str) -> Optional[int]:
            value = record.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise DocumentMetadataError(
                    f"{key} for chunk {chunk_id!r} must be an integer or null."
                )
            return value

        source_pages_raw = record.get("source_pages") or []
        if not isinstance(source_pages_raw, list):
            raise DocumentMetadataError(
                f"source_pages for chunk {chunk_id!r} is not an array."
            )

        source_pages_values: list[int] = []
        for page in source_pages_raw:
            if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
                raise DocumentMetadataError(
                    f"source_pages for chunk {chunk_id!r} contains "
                    f"an invalid page number: {page!r}."
                )
            source_pages_values.append(page)
        source_pages = tuple(source_pages_values)

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
            section_id=str(record["section_id"]),
            section_type=str(record["section_type"]),
            section_heading=(
                str(record["section_heading"])
                if record.get("section_heading") is not None
                else None
            ),
            section_level=(
                self._validated_nonnegative_int(
                    record.get("section_level", 1),
                    field_name="section_level",
                    chunk_id=chunk_id,
                )
            ),
            start_page=optional_int("start_page"),
            end_page=optional_int("end_page"),
            source_pages=source_pages,
            parent_section_id=(
                str(record["parent_section_id"])
                if record.get("parent_section_id") is not None
                else None
            ),
            chunk_index=self._validated_nonnegative_int(
                record["chunk_index"],
                field_name="chunk_index",
                chunk_id=chunk_id,
            ),
            token_count=self._validated_nonnegative_int(
                record.get("token_count", 0),
                field_name="token_count",
                chunk_id=chunk_id,
            ),
            char_count=self._validated_nonnegative_int(
                record.get("char_count", 0),
                field_name="char_count",
                chunk_id=chunk_id,
            ),
            overlap_with_previous=self._validated_nonnegative_int(
                record.get("overlap_with_previous", 0),
                field_name="overlap_with_previous",
                chunk_id=chunk_id,
            ),
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
    ) -> list[ChunkRetrievalResult]:
        return self.uploaded.retrieve_for_document(
            query,
            document_id,
            top_k,
            candidate_k=candidate_k,
        )


class RAGRetriever:
    """
    Unified retrieval adapter for the RAG pipeline.

    Routes research, uploaded-paper, and comparison retrieval through the
    existing authoritative retrievers. It does not implement embeddings,
    FAISS loading, reranking, final ranking, context construction, or LLM calls.
    """

    SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})

    def __init__(
        self,
        *,
        paper_retriever: Optional[PaperRetriever] = None,
        research_retriever: Optional[DenseRetriever] = None,
        uploaded_retriever: Optional[UploadedPaperRetriever] = None,
        research_search_service: Optional[Any] = None,
    ) -> None:
        if paper_retriever is not None:
            if research_retriever is not None or uploaded_retriever is not None:
                raise TypeError(
                    "Pass either paper_retriever or explicit "
                    "research_retriever/uploaded_retriever, not both."
                )
            self._paper_retriever = paper_retriever
            self._research_retriever = paper_retriever.research
            self._uploaded_retriever = paper_retriever.uploaded
        else:
            if research_retriever is None:
                raise TypeError(
                    "research_retriever is required when paper_retriever "
                    "is not supplied."
                )
            if uploaded_retriever is None:
                raise TypeError(
                    "uploaded_retriever is required when paper_retriever "
                    "is not supplied."
                )
            self._research_retriever = research_retriever
            self._uploaded_retriever = uploaded_retriever
            self._paper_retriever = PaperRetriever(
                research_retriever=research_retriever,
                uploaded_retriever=uploaded_retriever,
            )

        if not callable(getattr(self._research_retriever, "retrieve", None)):
            raise TypeError("research_retriever must expose retrieve().")
        if not callable(
            getattr(self._uploaded_retriever, "retrieve_for_document", None)
        ):
            raise TypeError(
                "uploaded_retriever must expose retrieve_for_document()."
            )
        if research_search_service is not None and not callable(
            getattr(research_search_service, "search", None)
        ):
            raise TypeError(
                "research_search_service must expose search()."
            )

        self._research_search_service = research_search_service

    @property
    def paper_retriever(self) -> PaperRetriever:
        return self._paper_retriever

    @property
    def research_retriever(self) -> DenseRetriever:
        return self._research_retriever

    @property
    def uploaded_retriever(self) -> UploadedPaperRetriever:
        return self._uploaded_retriever

    @property
    def research_search_service(self) -> Optional[Any]:
        return self._research_search_service

    def document_exists(self, document_id: str) -> bool:
        """Authoritative uploaded-paper availability check for API layers."""
        return self._uploaded_retriever.document_exists(document_id)

    def validate_document_available(self, document_id: str) -> Path:
        """Validate an uploaded paper using the same path used by retrieval."""
        return self._uploaded_retriever.validate_document_available(document_id)

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int,
        final_k: int,
        task_type: str,
        scope: Any,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        exclude_ids: Optional[set[str]] = None,
        filters: Any = None,
    ) -> Any:
        """Fulfil the RAGPipeline retrieval contract."""
        normalized_query = DenseRetriever._validate_query(query)
        candidate_k = DenseRetriever._validate_candidate_k(
            candidate_k, field_name="candidate_k"
        )
        final_k = DenseRetriever._validate_candidate_k(
            final_k, field_name="final_k"
        )

        if final_k > candidate_k:
            candidate_k = final_k

        normalized_scope = self._normalize_scope(scope)

        if normalized_scope == "research":
            candidates = self._retrieve_research(
                normalized_query,
                candidate_k=candidate_k,
                final_k=final_k,
                exclude_ids=exclude_ids,
                filters=filters,
            )
            return self._make_result(
                candidates=candidates,
                scope=normalized_scope,
                task_type=task_type,
                query=normalized_query,
            )

        if normalized_scope == "uploaded":
            if not document_id:
                raise InvalidDocumentIDError(
                    "document_id is required for uploaded-paper retrieval."
                )

            safe_document_id = UploadedPaperRetriever._validate_document_id(
                document_id
            )
            candidates = self._uploaded_retriever.retrieve_for_document(
                normalized_query,
                safe_document_id,
                top_k=final_k,
                candidate_k=candidate_k,
            )
            return self._make_result(
                candidates=candidates,
                scope=normalized_scope,
                task_type=task_type,
                query=normalized_query,
                document_id=safe_document_id,
            )

        if normalized_scope == "comparison":
            ids = self._validate_document_ids(document_ids)
            all_results: list[ChunkRetrievalResult] = []

            for current_id in ids:
                results = self._uploaded_retriever.retrieve_for_document(
                    normalized_query,
                    current_id,
                    top_k=candidate_k,
                    candidate_k=candidate_k,
                )
                all_results.extend(results)

            # Each document contributes up to candidate_k candidates. Perform
            # the same deterministic query-aware prioritization globally so an
            # explicit answer-bearing chunk is not buried by a higher raw FAISS
            # score from another document.
            all_results = UploadedPaperRetriever._prioritize_answer_bearing_candidates(
                normalized_query,
                all_results,
            )

            return self._make_result(
                candidates=all_results[:final_k],
                scope=normalized_scope,
                task_type=task_type,
                query=normalized_query,
                document_ids=ids,
            )

        raise RetrievalError(
            f"Unsupported retrieval scope {normalized_scope!r}; "
            f"expected one of {sorted(self.SUPPORTED_SCOPES)}."
        )

    @staticmethod
    def _make_result(
        *,
        candidates: Any,
        scope: str,
        task_type: str,
        query: str,
        document_id: Optional[str] = None,
        document_ids: Sequence[str] = (),
    ) -> RAGRetrievalResult:
        if candidates is None:
            normalized_candidates: tuple[Any, ...] = ()
        elif isinstance(candidates, (str, bytes)):
            raise RetrievalResultError(
                "RAG retrieval candidates cannot be a raw string/bytes value."
            )
        else:
            try:
                normalized_candidates = tuple(candidates)
            except TypeError as exc:
                raise RetrievalResultError(
                    "RAG retrieval candidates must be iterable."
                ) from exc

        return RAGRetrievalResult(
            candidates=normalized_candidates,
            scope=scope,
            task_type=str(task_type),
            query=query,
            document_id=document_id,
            document_ids=tuple(document_ids),
        )

    def _retrieve_research(
        self,
        query: str,
        *,
        candidate_k: int,
        final_k: int,
        exclude_ids: Optional[set[str]],
        filters: Any,
    ) -> Any:
        # If the complete research search service is supplied, it remains the
        # authority for metadata-aware research retrieval/reranking.
        if self._research_search_service is not None:
            try:
                response = self._research_search_service.search(query)
            except TypeError:
                response = self._research_search_service.search(
                    query,
                    candidate_k=candidate_k,
                    rerank_k=final_k,
                    final_k=final_k,
                )
            return self._limit_service_response(response, final_k)

        results = self._research_retriever.retrieve(
            query,
            candidate_k=candidate_k,
            exclude_ids=exclude_ids,
            filters=filters,
        )
        if results is None:
            return []
        if isinstance(results, (str, bytes)):
            raise RetrievalResultError(
                "Research retriever returned an invalid string/bytes result."
            )
        try:
            return list(results)[:final_k]
        except TypeError as exc:
            raise RetrievalResultError(
                "Research retriever returned a non-iterable result."
            ) from exc

    @classmethod
    def _normalize_scope(cls, scope: Any) -> str:
        if scope is None:
            return "research"

        value = scope.value if hasattr(scope, "value") else scope
        value = str(value).strip().lower()

        aliases = {
            "mode1": "research",
            "mode_1": "research",
            "research_discovery": "research",
            "corpus": "research",
            "my_paper": "uploaded",
            "paper": "uploaded",
            "single": "uploaded",
            "mode2": "uploaded",
            "mode_2": "uploaded",
            "compare": "comparison",
            "comparisons": "comparison",
        }
        return aliases.get(value, value)

    @staticmethod
    def _validate_document_ids(
        document_ids: Optional[Sequence[str]],
    ) -> tuple[str, ...]:
        if document_ids is None:
            raise InvalidDocumentIDError(
                "document_ids are required for comparison retrieval."
            )
        if isinstance(document_ids, (str, bytes)):
            raise InvalidDocumentIDError(
                "document_ids must be a sequence, not a single string."
            )

        values = list(document_ids)
        if len(values) < 2:
            raise InvalidDocumentIDError(
                "Comparison retrieval requires at least two documents."
            )

        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise InvalidDocumentIDError(
                    "Every document ID must be a string."
                )
            cleaned = UploadedPaperRetriever._validate_document_id(value)
            normalized.append(cleaned)

        if len(set(normalized)) != len(normalized):
            raise InvalidDocumentIDError(
                "Comparison document IDs must be unique."
            )

        return tuple(normalized)

    @staticmethod
    def _limit_service_response(
        response: Any,
        final_k: int,
    ) -> Any:
        if isinstance(response, list):
            return response[:final_k]
        if isinstance(response, tuple):
            return response[:final_k]

        if isinstance(response, Mapping):
            limited = dict(response)
            for key in ("results", "candidates", "items"):
                value = limited.get(key)
                if isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes)
                ):
                    limited[key] = list(value[:final_k])
                    break
            return limited

        for key in ("results", "candidates", "items"):
            value = getattr(response, key, None)
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes)
            ):
                try:
                    setattr(response, key, list(value[:final_k]))
                except Exception:
                    pass
                break

        return response


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
    def validate(self) -> None:
        return None


def _run_query_aware_ranking_self_test() -> None:
    """Regression test for the production failure mode."""
    candidates = [
        ChunkRetrievalResult(
            document_id="paper",
            chunk_id="neighbor-1",
            score=0.7588,
            rank=1,
            index_position=22,
            section_id="s",
            section_type="other",
            section_heading=None,
            section_level=1,
            start_page=6,
            end_page=6,
            source_pages=(6,),
            parent_section_id=None,
            chunk_index=22,
            token_count=20,
            char_count=100,
            overlap_with_previous=0,
            chunking_method="test",
            text="Preference bias establishes an ordering over hypotheses.",
            metadata={},
        ),
        ChunkRetrievalResult(
            document_id="paper",
            chunk_id="neighbor-2",
            score=0.7496,
            rank=2,
            index_position=9,
            section_id="s",
            section_type="other",
            section_heading=None,
            section_level=1,
            start_page=6,
            end_page=6,
            source_pages=(6,),
            parent_section_id=None,
            chunk_index=9,
            token_count=20,
            char_count=100,
            overlap_with_previous=0,
            chunking_method="test",
            text="Restricted hypothesis space bias limits the hypothesis set.",
            metadata={},
        ),
        ChunkRetrievalResult(
            document_id="paper",
            chunk_id="exact-answer",
            score=0.7390,
            rank=3,
            index_position=8,
            section_id="s",
            section_type="other",
            section_heading=None,
            section_level=1,
            start_page=6,
            end_page=6,
            source_pages=(6,),
            parent_section_id=None,
            chunk_index=8,
            token_count=30,
            char_count=160,
            overlap_with_previous=0,
            chunking_method="test",
            text=(
                "There are two general forms of bias in learning from examples: "
                "restricted hypothesis space bias and preference bias."
            ),
            metadata={},
        ),
    ]

    ranked = UploadedPaperRetriever._prioritize_answer_bearing_candidates(
        "What are the two general forms of bias in learning from examples?",
        candidates,
    )

    assert ranked[0].chunk_id == "exact-answer", (
        "Exact answer-bearing evidence was not prioritized."
    )
    assert [item.rank for item in ranked] == [1, 2, 3]
    assert [item.score for item in ranked] == [0.7390, 0.7588, 0.7496]
    assert ranked[0].text and "two general forms" in ranked[0].text.lower()


def run_self_test() -> None:
    """Run unit-style tests without SPECTER2, FAISS, or a real corpus."""
    assert callable(
        getattr(
            UploadedPaperRetriever,
            "_prioritize_answer_bearing_candidates",
            None,
        )
    ), "Uploaded-paper answer-aware ranking method is missing."
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

    rag = RAGRetriever(
        research_retriever=retriever,
        uploaded_retriever=UploadedPaperRetriever(
            encoder,
            uploaded_root=Path("indexes/uploaded"),
        ),
    )
    rag_results = rag.retrieve(
        "valid",
        candidate_k=2,
        final_k=1,
        task_type="search",
        scope="research",
    )
    assert len(rag_results) == 1
    assert rag_results[0].document_id == "paper-a"
    assert isinstance(rag_results, RAGRetrievalResult)
    assert len(rag_results.candidates) == 1
    assert rag_results.results is rag_results.candidates
    assert rag_results.is_empty is False
    assert list(rag_results) == list(rag_results.candidates)

    default_scope_results = rag.retrieve(
        "valid",
        candidate_k=2,
        final_k=1,
        task_type="search",
        scope=None,
    )
    assert default_scope_results[0].document_id == "paper-a"

    try:
        rag.retrieve(
            "valid",
            candidate_k=2,
            final_k=1,
            task_type="comparison",
            scope="comparison",
            document_ids=["paper-a"],
        )
    except InvalidDocumentIDError:
        pass
    else:
        raise AssertionError(
            "Comparison retrieval accepted fewer than two document IDs."
        )

    print("Retriever self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()
    