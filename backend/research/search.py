"""
High-level research-discovery orchestration for the AI Research Paper Assistant.

Pipeline
--------
User query
    -> retriever.py
    -> reranker.py
    -> ranker.py
    -> final Top-K research results

This module is intentionally an orchestration layer only.

It does NOT:
- generate embeddings
- call SPECTER2
- call FAISS
- implement retrieval
- implement cross-encoder inference
- manually sort/rank candidates
- calculate ranking scores
- call an LLM
- summarize papers
- analyze methodology/models/datasets/findings
- perform PDF/RAG processing
- load the corpus or model weights

Dependencies are injected so the service can be unit-tested without the
real ML stack.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

try:
    from backend.retrieval.metadata_store import ResearchMetadataStore
except ImportError:  # pragma: no cover - isolated module tests
    ResearchMetadataStore = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_K = 50
DEFAULT_FINAL_K = 10

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_FINAL_K",
    "ResearchSearchError",
    "InvalidSearchRequestError",
    "RetrievalStageError",
    "RerankingStageError",
    "FinalRankingStageError",
    "SearchResultValidationError",
    "ResearchSearchConfig",
    "ResearchSearchRequest",
    "SearchTrace",
    "ResearchSearchResponse",
    "RetrieverProtocol",
    "RerankerProtocol",
    "RankerProtocol",
    "ResearchSearchService",
    "search_research",
    "run_self_test",
]


# ============================================================================
# Exceptions
# ============================================================================


class ResearchSearchError(RuntimeError):
    """Base exception for high-level research-search failures."""


class InvalidSearchRequestError(ResearchSearchError, ValueError):
    """Raised when a high-level search request is invalid."""


class RetrievalStageError(ResearchSearchError):
    """Raised when candidate retrieval fails."""


class RerankingStageError(ResearchSearchError):
    """Raised when candidate reranking fails."""


class FinalRankingStageError(ResearchSearchError):
    """Raised when final ranking fails."""


class SearchResultValidationError(ResearchSearchError, ValueError):
    """Raised when an upstream component returns malformed results."""


# ============================================================================
# Request / response types
# ============================================================================


@dataclass(frozen=True)
class ResearchSearchConfig:
    """Validated policy for the complete Mode-1 retrieval pipeline."""

    candidate_k: int = DEFAULT_CANDIDATE_K
    rerank_k: int = DEFAULT_FINAL_K
    final_k: int = DEFAULT_FINAL_K

    def validate(self) -> None:
        for name, value in (
            ("candidate_k", self.candidate_k),
            ("rerank_k", self.rerank_k),
            ("final_k", self.final_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidSearchRequestError(
                    f"{name} must be a positive integer; got {value!r}."
                )

        if self.rerank_k > self.candidate_k:
            raise InvalidSearchRequestError(
                "rerank_k must be <= candidate_k."
            )
        if self.final_k > self.rerank_k:
            raise InvalidSearchRequestError(
                "final_k must be <= rerank_k."
            )


@dataclass(frozen=True)
class ResearchSearchRequest:
    """Validated high-level Mode-1 search request."""

    query: str
    candidate_k: int = DEFAULT_CANDIDATE_K
    final_k: int = DEFAULT_FINAL_K

    # Defensive service-level limits. The retriever/reranker may impose
    # tighter model-specific limits; these prevent accidental huge requests
    # from reaching expensive downstream stages.
    MAX_CANDIDATE_K: int = 1000
    MAX_FINAL_K: int = 200

    def validate(self) -> None:
        if self.query is None:
            raise InvalidSearchRequestError(
                "query cannot be None."
            )

        if not isinstance(self.query, str):
            raise InvalidSearchRequestError(
                "query must be a string; "
                f"got {type(self.query).__name__}."
            )

        if not self.query.strip():
            raise InvalidSearchRequestError(
                "query cannot be empty or whitespace-only."
            )

        if (
            not isinstance(self.candidate_k, int)
            or isinstance(self.candidate_k, bool)
            or self.candidate_k <= 0
        ):
            raise InvalidSearchRequestError(
                "candidate_k must be a positive integer; "
                f"got {self.candidate_k!r}."
            )

        if (
            not isinstance(self.final_k, int)
            or isinstance(self.final_k, bool)
            or self.final_k <= 0
        ):
            raise InvalidSearchRequestError(
                "final_k must be a positive integer; "
                f"got {self.final_k!r}."
            )

        if self.candidate_k > self.MAX_CANDIDATE_K:
            raise InvalidSearchRequestError(
                "candidate_k exceeds the service safety limit: "
                f"{self.candidate_k} > {self.MAX_CANDIDATE_K}."
            )

        if self.final_k > self.MAX_FINAL_K:
            raise InvalidSearchRequestError(
                "final_k exceeds the service safety limit: "
                f"{self.final_k} > {self.MAX_FINAL_K}."
            )

        if self.final_k > self.candidate_k:
            raise InvalidSearchRequestError(
                "final_k must be less than or equal to candidate_k. "
                f"Received candidate_k={self.candidate_k}, "
                f"final_k={self.final_k}."
            )


@dataclass(frozen=True)
class SearchTrace:
    """
    Lightweight pipeline telemetry.

    Contains counts/timings only; no paper text, embeddings, or private
    content.
    """

    retrieval_latency_seconds: float
    reranking_latency_seconds: float
    ranking_latency_seconds: float
    total_latency_seconds: float
    candidate_count: int
    reranked_count: int
    result_count: int


@dataclass(frozen=True)
class ResearchSearchResponse:
    """
    Final Mode-1 search response.

    results are intentionally kept as the result objects produced by the
    final ranker. This avoids rebuilding or duplicating ranking logic here.
    """

    query: str
    candidate_count: int
    result_count: int
    results: list[Any]
    trace: Optional[SearchTrace] = None

    @property
    def candidates_retrieved(self) -> int:
        """Backward-compatible alias for candidate_count."""
        return self.candidate_count

    @property
    def candidates_reranked(self) -> int:
        """Number of candidates returned by the reranker."""
        return self._candidates_reranked

    # Private compatibility payload is initialized by the service constructor.
    _candidates_reranked: int = 0


# ============================================================================
# Minimal protocols
# ============================================================================


class RetrieverProtocol(Protocol):
    """Minimal interface required from retrieval/retriever.py."""

    def retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> Sequence[Any]:
        ...


class RerankerProtocol(Protocol):
    """Minimal interface required from retrieval/reranker.py."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: Optional[int] = None,
    ) -> Sequence[Any]:
        ...


class RankerProtocol(Protocol):
    """Minimal interface required from retrieval/ranker.py."""

    def rank(
        self,
        candidates: Sequence[Any],
        *,
        top_k: Optional[int] = None,
    ) -> Sequence[Any]:
        ...


# ============================================================================
# Internal helpers
# ============================================================================


def _document_id(candidate: Any) -> Optional[str]:
    """Read document_id without making search.py depend on a concrete class."""
    if isinstance(candidate, Mapping):
        value = candidate.get("document_id")
    else:
        value = getattr(candidate, "document_id", None)

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None


def _finite_score(
    candidate: Any,
    field_name: str,
) -> Optional[float]:
    """Validate a supplied score when present."""
    if isinstance(candidate, Mapping):
        value = candidate.get(field_name)
    else:
        value = getattr(candidate, field_name, None)

    if value is None:
        return None

    if isinstance(value, bool):
        raise SearchResultValidationError(
            f"Candidate {_document_id(candidate)!r} has invalid "
            f"{field_name}={value!r}."
        )

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SearchResultValidationError(
            f"Candidate {_document_id(candidate)!r} has invalid "
            f"{field_name}={value!r}."
        ) from exc

    if not math.isfinite(value):
        raise SearchResultValidationError(
            f"Candidate {_document_id(candidate)!r} has non-finite "
            f"{field_name}={value!r}."
        )

    return value


def _validate_candidate_collection(
    candidates: Sequence[Any],
    *,
    stage_name: str,
    allow_empty: bool,
) -> list[Any]:
    """Validate structural candidate output without duplicating ranking logic."""
    if candidates is None:
        raise SearchResultValidationError(
            f"{stage_name} returned None; expected a candidate sequence."
        )

    if isinstance(candidates, (str, bytes)):
        raise SearchResultValidationError(
            f"{stage_name} returned text instead of candidates."
        )

    try:
        items = list(candidates)
    except TypeError as exc:
        raise SearchResultValidationError(
            f"{stage_name} returned a non-iterable candidate collection."
        ) from exc

    if not items and not allow_empty:
        raise SearchResultValidationError(
            f"{stage_name} returned an unexpectedly empty result."
        )

    seen_ids: set[str] = set()

    for position, candidate in enumerate(items):
        if candidate is None:
            raise SearchResultValidationError(
                f"{stage_name} returned None at position {position}."
            )

        document_id = _document_id(candidate)

        if document_id is None:
            raise SearchResultValidationError(
                f"{stage_name} returned a candidate without a valid "
                f"document_id at position {position}."
            )

        # At the paper-level Mode-1 boundary, duplicate paper IDs are a
        # pipeline anomaly. Chunk-level identity policy belongs to ranker.py,
        # so we do not globally deduplicate here.
        if document_id in seen_ids:
            logger.warning(
                "%s returned duplicate document_id=%s. "
                "Passing through to the configured ranker.",
                stage_name,
                document_id,
            )

        seen_ids.add(document_id)

        # Validate scores only when fields are supplied. The ranker remains
        # authoritative for which score is required and how it is selected.
        _finite_score(candidate, "retrieval_score")
        _finite_score(candidate, "reranker_score")

    return items


def _rank_value(candidate: Any) -> Optional[int]:
    """Read a final rank if one is exposed by the upstream ranker."""
    if isinstance(candidate, Mapping):
        value = candidate.get("rank")
    else:
        value = getattr(candidate, "rank", None)

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _validate_final_results(
    results: Sequence[Any],
    *,
    final_k: int,
) -> list[Any]:
    """
    Validate the final ranker's output.

    Search.py does not sort or recalculate final scores. It only verifies
    the ranker's contract before exposing results.
    """
    if results is None:
        raise SearchResultValidationError(
            "ranker returned None."
        )

    if isinstance(results, (str, bytes)):
        raise SearchResultValidationError(
            "ranker returned text instead of ranked results."
        )

    try:
        items = list(results)
    except TypeError as exc:
        raise SearchResultValidationError(
            "ranker returned a non-iterable result collection."
        ) from exc

    if len(items) > final_k:
        raise SearchResultValidationError(
            "ranker returned more results than requested: "
            f"{len(items)} > final_k={final_k}."
        )

    seen_ids: set[str] = set()

    for expected_rank, result in enumerate(items, start=1):
        document_id = _document_id(result)

        if document_id is None:
            raise SearchResultValidationError(
                f"Final result at position {expected_rank - 1} has no "
                "valid document_id."
            )

        if document_id in seen_ids:
            raise SearchResultValidationError(
                "Final search response contains duplicate document_id="
                f"{document_id!r}."
            )

        seen_ids.add(document_id)

        rank = _rank_value(result)

        if rank is None:
            raise SearchResultValidationError(
                f"Final result {document_id!r} has no valid rank."
            )

        if rank != expected_rank:
            raise SearchResultValidationError(
                "Final result ranks are not sequential: "
                f"expected {expected_rank}, got {rank}."
            )

        final_score = (
            result.get("final_score")
            if isinstance(result, Mapping)
            else getattr(result, "final_score", None)
        )

        if final_score is None:
            raise SearchResultValidationError(
                f"Final result {document_id!r} has no final_score."
            )

        if isinstance(final_score, bool):
            raise SearchResultValidationError(
                f"Final result {document_id!r} has invalid final_score="
                f"{final_score!r}."
            )

        try:
            final_score_float = float(final_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SearchResultValidationError(
                f"Final result {document_id!r} has invalid final_score="
                f"{final_score!r}."
            ) from exc

        if not math.isfinite(final_score_float):
            raise SearchResultValidationError(
                f"Final result {document_id!r} has non-finite final_score="
                f"{final_score!r}."
            )

    return items


# ============================================================================
# Research Search Service
# ============================================================================


class ResearchSearchService:
    """
    High-level Mode-1 research-discovery orchestrator.

    The service owns no ML models and no data files. All expensive components
    are injected and should be initialized once outside the request path.

    Example
    -------
    service = ResearchSearchService(
        retriever=dense_retriever,
        reranker=cross_encoder_reranker,
        ranker=final_ranker,
    )

    response = service.search(
        "transformer based medical image segmentation"
    )
    """

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol,
        ranker: RankerProtocol,
        metadata_store: Optional[Any] = None,
        config: Optional[ResearchSearchConfig] = None,
        trace_factory: Optional[
            Callable[[SearchTrace], SearchTrace]
        ] = None,
    ) -> None:
        if retriever is None:
            raise InvalidSearchRequestError(
                "retriever dependency cannot be None."
            )

        if reranker is None:
            raise InvalidSearchRequestError(
                "reranker dependency cannot be None."
            )

        if ranker is None:
            raise InvalidSearchRequestError(
                "ranker dependency cannot be None."
            )

        self._retriever = retriever
        self._reranker = reranker
        self._ranker = ranker
        self._metadata_store = metadata_store
        self._config = config or ResearchSearchConfig()
        self._config.validate()
        self._trace_factory = trace_factory

    @property
    def retriever(self) -> RetrieverProtocol:
        return self._retriever

    @property
    def reranker(self) -> RerankerProtocol:
        return self._reranker

    @property
    def ranker(self) -> RankerProtocol:
        return self._ranker

    @property
    def metadata_store(self) -> Any:
        return self._metadata_store

    @property
    def config(self) -> ResearchSearchConfig:
        return self._config

    def search(
        self,
        query: str,
        *,
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
    ) -> ResearchSearchResponse:
        """
        Execute exactly one Mode-1 retrieval -> reranking -> final-ranking flow.

        Execution order:
            1. validate request
            2. retrieve candidates
            3. validate retrieval output
            4. rerank candidates
            5. validate reranking output
            6. final-rank candidates
            7. validate final output
            8. return response

        Empty retrieval is treated as a valid no-results search.
        Empty reranking after non-empty retrieval is treated as a pipeline
        failure because a non-empty candidate pool unexpectedly disappeared.
        """
        resolved_candidate_k = (
            self._config.candidate_k if candidate_k is None else candidate_k
        )
        resolved_rerank_k = (
            self._config.rerank_k if rerank_k is None else rerank_k
        )
        resolved_final_k = (
            self._config.final_k if final_k is None else final_k
        )

        pipeline_config = ResearchSearchConfig(
            candidate_k=resolved_candidate_k,
            rerank_k=resolved_rerank_k,
            final_k=resolved_final_k,
        )
        pipeline_config.validate()

        request = ResearchSearchRequest(
            query=query,
            candidate_k=pipeline_config.candidate_k,
            final_k=pipeline_config.final_k,
        )
        request.validate()

        clean_query = request.query.strip()

        logger.info(
            "Research search started: candidate_k=%d, final_k=%d.",
            request.candidate_k,
            request.final_k,
        )

        total_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1: Retrieval
        # ------------------------------------------------------------------
        logger.debug(
            "Query validated. Retrieving %d candidates.",
            request.candidate_k,
        )

        retrieval_start = time.perf_counter()

        try:
            retrieved_raw = self._retriever.retrieve(
                query=clean_query,
                candidate_k=request.candidate_k,
            )
        except Exception as exc:
            logger.exception("Research candidate retrieval failed.")
            raise RetrievalStageError(
                "Research candidate retrieval failed."
            ) from exc

        retrieval_latency = time.perf_counter() - retrieval_start

        retrieved = _validate_candidate_collection(
            retrieved_raw,
            stage_name="retriever",
            allow_empty=True,
        )

        logger.debug(
            "Retrieved %d candidates.",
            len(retrieved),
        )

        # Empty retrieval is a normal no-results condition.
        if not retrieved:
            total_latency = time.perf_counter() - total_start

            trace = self._make_trace(
                retrieval_latency_seconds=retrieval_latency,
                reranking_latency_seconds=0.0,
                ranking_latency_seconds=0.0,
                total_latency_seconds=total_latency,
                candidate_count=0,
                reranked_count=0,
                result_count=0,
            )

            logger.info(
                "Research search completed with no candidates. "
                "total_latency=%.4fs.",
                total_latency,
            )

            return ResearchSearchResponse(
                query=clean_query,
                candidate_count=0,
                result_count=0,
                results=[],
                trace=trace,
                _candidates_reranked=0,
            )

        # ------------------------------------------------------------------
        # Stage 2: Reranking
        # ------------------------------------------------------------------
        logger.debug(
            "Preparing %d candidates for reranking.",
            len(retrieved),
        )

        reranker_candidates: Sequence[Any] = retrieved
        if self._metadata_store is not None:
            try:
                hydrate = getattr(
                    self._metadata_store,
                    "hydrate_candidates",
                    None,
                )
                if not callable(hydrate):
                    raise SearchResultValidationError(
                        "metadata_store must expose hydrate_candidates()."
                    )
                reranker_candidates = hydrate(retrieved)
            except SearchResultValidationError:
                raise
            except Exception as exc:
                logger.exception("Research metadata hydration failed.")
                raise RerankingStageError(
                    "Research metadata hydration failed before reranking."
                ) from exc

        reranking_start = time.perf_counter()

        try:
            reranked_raw = self._reranker.rerank(
                clean_query,
                reranker_candidates,
                top_k=pipeline_config.rerank_k,
            )
        except Exception as exc:
            logger.exception("Research candidate reranking failed.")
            raise RerankingStageError(
                "Research candidate reranking failed."
            ) from exc

        reranking_latency = time.perf_counter() - reranking_start

        reranked = _validate_candidate_collection(
            reranked_raw,
            stage_name="reranker",
            allow_empty=False,
        )

        if len(reranked) > len(reranker_candidates):
            raise SearchResultValidationError(
                "Reranker returned more candidates than it received: "
                f"{len(reranked)} > {len(reranker_candidates)}."
            )

        logger.debug(
            "Reranked %d candidates.",
            len(reranked),
        )

        # ------------------------------------------------------------------
        # Stage 3: Final ranking
        # ------------------------------------------------------------------
        logger.debug(
            "Selecting final Top %d.",
            request.final_k,
        )

        ranking_start = time.perf_counter()

        try:
            ranked_raw = self._ranker.rank(
                candidates=reranked,
                top_k=request.final_k,
            )
        except Exception as exc:
            logger.exception("Final research ranking failed.")
            raise FinalRankingStageError(
                "Final research ranking failed."
            ) from exc

        ranking_latency = time.perf_counter() - ranking_start

        ranked = _validate_final_results(
            ranked_raw,
            final_k=request.final_k,
        )

        # A ranker may legitimately return fewer results than final_k when
        # fewer valid candidates exist.
        if len(ranked) > len(reranked):
            raise SearchResultValidationError(
                "Ranker returned more results than reranker supplied: "
                f"{len(ranked)} > {len(reranked)}."
            )

        total_latency = time.perf_counter() - total_start

        trace = self._make_trace(
            retrieval_latency_seconds=retrieval_latency,
            reranking_latency_seconds=reranking_latency,
            ranking_latency_seconds=ranking_latency,
            total_latency_seconds=total_latency,
            candidate_count=len(retrieved),
            reranked_count=len(reranked),
            result_count=len(ranked),
        )

        logger.info(
            "Research search completed: retrieved=%d, reranked=%d, "
            "final=%d, total_latency=%.4fs.",
            len(retrieved),
            len(reranked),
            len(ranked),
            total_latency,
        )

        return ResearchSearchResponse(
            query=clean_query,
            candidate_count=len(retrieved),
            result_count=len(ranked),
            results=ranked,
            trace=trace,
            _candidates_reranked=len(reranked),
        )

    def search_request(
        self,
        request: ResearchSearchRequest,
    ) -> ResearchSearchResponse:
        """Request-object convenience API."""
        if not isinstance(request, ResearchSearchRequest):
            raise InvalidSearchRequestError(
                "request must be a ResearchSearchRequest."
            )

        request.validate()

        return self.search(
            request.query,
            candidate_k=request.candidate_k,
            final_k=request.final_k,
        )

    def _make_trace(
        self,
        *,
        retrieval_latency_seconds: float,
        reranking_latency_seconds: float,
        ranking_latency_seconds: float,
        total_latency_seconds: float,
        candidate_count: int,
        reranked_count: int,
        result_count: int,
    ) -> SearchTrace:
        trace = SearchTrace(
            retrieval_latency_seconds=float(retrieval_latency_seconds),
            reranking_latency_seconds=float(reranking_latency_seconds),
            ranking_latency_seconds=float(ranking_latency_seconds),
            total_latency_seconds=float(total_latency_seconds),
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            result_count=result_count,
        )

        if self._trace_factory is None:
            return trace

        try:
            return self._trace_factory(trace)
        except Exception as exc:
            raise ResearchSearchError(
                "Trace factory failed."
            ) from exc


# ============================================================================
# Functional API
# ============================================================================


def search_research(
    *,
    query: str,
    retriever: RetrieverProtocol,
    reranker: RerankerProtocol,
    ranker: RankerProtocol,
    metadata_store: Optional[Any] = None,
    config: Optional[ResearchSearchConfig] = None,
    candidate_k: Optional[int] = None,
    rerank_k: Optional[int] = None,
    final_k: Optional[int] = None,
) -> ResearchSearchResponse:
    """
    Thin functional wrapper around ResearchSearchService.

    No orchestration logic is duplicated here.
    """
    service = ResearchSearchService(
        retriever=retriever,
        reranker=reranker,
        ranker=ranker,
        metadata_store=metadata_store,
        config=config,
    )

    return service.search(
        query,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
        final_k=final_k,
    )


# ============================================================================
# Model-free self-test
# ============================================================================


def run_self_test() -> None:
    """
    Comprehensive service-level test without FAISS, SPECTER2, Transformers,
    GPU, dataset files, or network access.
    """

    @dataclass(frozen=True)
    class Candidate:
        document_id: str
        retrieval_score: float
        reranker_score: float
        title: str = "Test paper"

    @dataclass(frozen=True)
    class Ranked:
        document_id: str
        rank: int
        final_score: float
        reranker_score: float
        retrieval_score: float

    class MockRetriever:
        def __init__(
            self,
            candidates: Sequence[Candidate],
        ) -> None:
            self.candidates = list(candidates)
            self.calls = 0
            self.last_query: Optional[str] = None
            self.last_candidate_k: Optional[int] = None

        def retrieve(
            self,
            query: str,
            candidate_k: int,
        ) -> list[Candidate]:
            self.calls += 1
            self.last_query = query
            self.last_candidate_k = candidate_k
            return list(self.candidates[:candidate_k])

    class MockReranker:
        def __init__(
            self,
            score_order: Optional[Sequence[float]] = None,
        ) -> None:
            self.calls = 0
            self.last_query: Optional[str] = None
            self.last_candidates: list[Candidate] = []
            self.score_order = (
                list(score_order)
                if score_order is not None
                else None
            )

        def rerank(
            self,
            query: str,
            candidates: Sequence[Candidate],
            *,
            top_k: Optional[int] = None,
        ) -> list[Candidate]:
            self.calls += 1
            self.last_query = query
            self.last_candidates = list(candidates)

            if self.score_order is None:
                return list(candidates)

            if len(self.score_order) != len(candidates):
                raise AssertionError(
                    "Mock score_order length mismatch."
                )

            return [
                Candidate(
                    document_id=candidate.document_id,
                    retrieval_score=candidate.retrieval_score,
                    reranker_score=score,
                    title=candidate.title,
                )
                for candidate, score in zip(
                    candidates,
                    self.score_order,
                )
            ]

    class MockRanker:
        def __init__(self) -> None:
            self.calls = 0
            self.last_candidates: list[Candidate] = []
            self.last_top_k: Optional[int] = None

        def rank(
            self,
            candidates: Sequence[Candidate],
            *,
            top_k: Optional[int] = None,
        ) -> list[Ranked]:
            self.calls += 1
            self.last_candidates = list(candidates)
            self.last_top_k = top_k

            ordered = sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.reranker_score,
                    candidate.document_id,
                ),
            )

            limit = top_k if top_k is not None else len(ordered)

            return [
                Ranked(
                    document_id=candidate.document_id,
                    rank=index,
                    final_score=candidate.reranker_score,
                    reranker_score=candidate.reranker_score,
                    retrieval_score=candidate.retrieval_score,
                )
                for index, candidate in enumerate(
                    ordered[:limit],
                    start=1,
                )
            ]

    base_candidates = [
        Candidate("paper-a", 0.81, 0.70),
        Candidate("paper-b", 0.88, 0.95),
        Candidate("paper-c", 0.92, 0.82),
    ]

    retriever = MockRetriever(base_candidates)
    reranker = MockReranker([0.70, 0.95, 0.82])
    ranker = MockRanker()

    service = ResearchSearchService(
        retriever=retriever,
        reranker=reranker,
        ranker=ranker,
    )

    # ------------------------------------------------------------------
    # 1. Valid search
    # ------------------------------------------------------------------
    response = service.search(
        "  transformer based medical image segmentation  ",
        candidate_k=50,
        final_k=10,
    )

    assert response.query == (
        "transformer based medical image segmentation"
    )
    assert response.candidate_count == 3
    assert response.result_count == 3
    assert [result.document_id for result in response.results] == [
        "paper-b",
        "paper-c",
        "paper-a",
    ]

    assert retriever.calls == 1
    assert retriever.last_query == response.query
    assert retriever.last_candidate_k == 50

    assert reranker.calls == 1
    assert reranker.last_query == response.query
    assert len(reranker.last_candidates) == 3

    assert ranker.calls == 1
    assert ranker.last_top_k == 10
    assert len(ranker.last_candidates) == 3

    assert response.trace is not None
    assert response.trace.candidate_count == 3
    assert response.trace.reranked_count == 3
    assert response.trace.result_count == 3

    # ------------------------------------------------------------------
    # 2. Empty retrieval
    # ------------------------------------------------------------------
    empty_retriever = MockRetriever([])
    empty_reranker = MockReranker()
    empty_ranker = MockRanker()

    empty_service = ResearchSearchService(
        retriever=empty_retriever,
        reranker=empty_reranker,
        ranker=empty_ranker,
    )

    empty_response = empty_service.search(
        "valid research query",
        candidate_k=50,
        final_k=10,
    )

    assert empty_response.results == []
    assert empty_response.result_count == 0
    assert empty_reranker.calls == 0
    assert empty_ranker.calls == 0

    # ------------------------------------------------------------------
    # 3. Invalid query
    # ------------------------------------------------------------------
    for invalid_query in (None, "", "   ", 123):
        try:
            service.search(
                invalid_query,  # type: ignore[arg-type]
                candidate_k=50,
                final_k=10,
            )
        except InvalidSearchRequestError:
            pass
        else:
            raise AssertionError(
                f"Invalid query {invalid_query!r} was accepted."
            )

    # ------------------------------------------------------------------
    # 4. Invalid K values
    # ------------------------------------------------------------------
    invalid_k_cases = [
        {"candidate_k": 0, "final_k": 1},
        {"candidate_k": -1, "final_k": 1},
        {"candidate_k": 10, "final_k": 0},
        {"candidate_k": 10, "final_k": -1},
        {"candidate_k": 10, "final_k": 11},
        {"candidate_k": False, "final_k": 1},
        {"candidate_k": 10, "final_k": False},
    ]

    for case in invalid_k_cases:
        try:
            service.search(
                "valid query",
                **case,  # type: ignore[arg-type]
            )
        except InvalidSearchRequestError:
            pass
        else:
            raise AssertionError(
                f"Invalid K case was accepted: {case!r}"
            )

    # ------------------------------------------------------------------
    # 5. Reranking failure + exception chaining
    # ------------------------------------------------------------------
    class FailingReranker:
        def rerank(
            self,
            query: str,
            candidates: Sequence[Any],
            *,
            top_k: Optional[int] = None,
        ) -> Sequence[Any]:
            raise RuntimeError("simulated reranker failure")

    failing_service = ResearchSearchService(
        retriever=MockRetriever(base_candidates),
        reranker=FailingReranker(),
        ranker=MockRanker(),
    )

    try:
        failing_service.search(
            "valid query",
            candidate_k=50,
            final_k=10,
        )
    except RerankingStageError as exc:
        assert exc.__cause__ is not None
        assert "simulated reranker failure" in str(exc.__cause__)
    else:
        raise AssertionError(
            "Reranking failure was not wrapped."
        )

    # ------------------------------------------------------------------
    # 6. Retrieval failure + exception chaining
    # ------------------------------------------------------------------
    class FailingRetriever:
        def retrieve(
            self,
            query: str,
            candidate_k: int,
        ) -> Sequence[Any]:
            raise RuntimeError("simulated retrieval failure")

    retrieval_failure_service = ResearchSearchService(
        retriever=FailingRetriever(),
        reranker=MockReranker(),
        ranker=MockRanker(),
    )

    try:
        retrieval_failure_service.search(
            "valid query",
            candidate_k=50,
            final_k=10,
        )
    except RetrievalStageError as exc:
        assert exc.__cause__ is not None
        assert "simulated retrieval failure" in str(exc.__cause__)
    else:
        raise AssertionError(
            "Retrieval failure was not wrapped."
        )

    # ------------------------------------------------------------------
    # 7. Ranking failure + exception chaining
    # ------------------------------------------------------------------
    class FailingRanker:
        def rank(
            self,
            candidates: Sequence[Any],
            *,
            top_k: Optional[int] = None,
        ) -> Sequence[Any]:
            raise RuntimeError("simulated ranking failure")

    ranking_failure_service = ResearchSearchService(
        retriever=MockRetriever(base_candidates),
        reranker=MockReranker(
            [0.70, 0.95, 0.82]
        ),
        ranker=FailingRanker(),
    )

    try:
        ranking_failure_service.search(
            "valid query",
            candidate_k=50,
            final_k=10,
        )
    except FinalRankingStageError as exc:
        assert exc.__cause__ is not None
        assert "simulated ranking failure" in str(exc.__cause__)
    else:
        raise AssertionError(
            "Ranking failure was not wrapped."
        )

    # ------------------------------------------------------------------
    # 8. Invalid final result
    # ------------------------------------------------------------------
    class InvalidRanker:
        def rank(
            self,
            candidates: Sequence[Any],
            *,
            top_k: Optional[int] = None,
        ) -> Sequence[Any]:
            return [
                {
                    "document_id": "paper-a",
                    "rank": 2,
                    "final_score": 0.9,
                }
            ]

    invalid_result_service = ResearchSearchService(
        retriever=MockRetriever(base_candidates),
        reranker=MockReranker(
            [0.70, 0.95, 0.82]
        ),
        ranker=InvalidRanker(),
    )

    try:
        invalid_result_service.search(
            "valid query",
            candidate_k=50,
            final_k=10,
        )
    except SearchResultValidationError:
        pass
    else:
        raise AssertionError(
            "Malformed final ranking result was accepted."
        )

    # ------------------------------------------------------------------
    # 9. Candidate score validation
    # ------------------------------------------------------------------
    invalid_candidate_retriever = MockRetriever(
        [
            Candidate(
                "bad-paper",
                float("nan"),
                0.5,
            )
        ]
    )

    invalid_candidate_service = ResearchSearchService(
        retriever=invalid_candidate_retriever,
        reranker=MockReranker(),
        ranker=MockRanker(),
    )

    try:
        invalid_candidate_service.search(
            "valid query",
            candidate_k=50,
            final_k=10,
        )
    except SearchResultValidationError:
        pass
    else:
        raise AssertionError(
            "NaN retrieval score was accepted."
        )

    # ------------------------------------------------------------------
    # 10. Candidate_k vs final_k
    # ------------------------------------------------------------------
    response_small = service.search(
        "valid query",
        candidate_k=3,
        final_k=2,
    )

    assert response_small.candidate_count == 3
    assert response_small.result_count == 2
    assert ranker.last_top_k == 2

    print("ResearchSearchService self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()