"""
Final deterministic ranking layer for the AI Research Paper Assistant.

Pipeline
--------
FAISS -> reranker.py -> ranker.py -> Top-K

This module is intentionally inference-only and does NOT:
- generate embeddings
- call SPECTER2
- query FAISS
- run cross-encoder inference
- call an LLM
- summarize/analyze papers
- extract datasets/models/methodology
- score scientific quality

The ranker receives already retrieved/relevance-scored candidates and decides:
1. which score is authoritative,
2. the deterministic final ordering,
3. duplicate policy,
4. final Top-K selection.

Default strategy:
    reranker_score descending

Optional explicit fallback:
    if reranker_score is unavailable, use retrieval_score.

No score combination is enabled by default because FAISS and reranker
scores generally do not share a guaranteed common scale.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from numbers import Real

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


# ============================================================================
# Exceptions
# ============================================================================


class RankingError(RuntimeError):
    """Base exception for final-ranking errors."""


class InvalidRankingInputError(RankingError, ValueError):
    """Raised when ranking input is malformed."""


class InvalidTopKError(RankingError, ValueError):
    """Raised when top_k is invalid."""


class InvalidScoreError(RankingError, ValueError):
    """Raised when a supplied score is missing, non-numeric, NaN, or Inf."""


class DuplicateDocumentError(RankingError, ValueError):
    """Raised when duplicate ranking identities are forbidden."""


class RankingStrategyError(RankingError, ValueError):
    """Raised when an unsupported ranking strategy is requested."""


# ============================================================================
# Public enums
# ============================================================================


class RankingStrategy(str, Enum):
    """
    Authoritative score used for final ordering.

    RERANKER_SCORE:
        Preferred for Mode 1 after reranker.py.

    RETRIEVAL_SCORE:
        Explicit retrieval-score-only fallback/diagnostic strategy.
    """

    RERANKER_SCORE = "reranker_score"
    RETRIEVAL_SCORE = "retrieval_score"


class DuplicatePolicy(str, Enum):
    """
    Identity handling.

    REJECT:
        Raise on duplicate identity. Safest default for paper-level results.

    KEEP:
        Keep all candidates. Appropriate when each identity is already a
        legitimate unique unit, e.g. chunk_id.

    KEEP_BEST:
        Keep only the highest-scoring candidate for each identity. This is
        deterministic and useful when duplicate records are expected, but
        should not be used when distinct chunks share a paper ID.
    """

    REJECT = "reject"
    KEEP = "keep"
    KEEP_BEST = "keep_best"


# ============================================================================
# Public result/config types
# ============================================================================


@dataclass(frozen=True)
class RankedResult:
    """
    Final ranked candidate.

    final_score:
        The score actually used for ordering. It is either the raw
        reranker_score or retrieval_score according to the configured policy.

    reranker_score / retrieval_score:
        Preserved separately. They are NOT blended.

    identity:
        The configured ranking identity, normally document_id, paper_id,
        or chunk_id.

    rank:
        One-based final rank after sorting and Top-K selection.
    """

    document_id: str
    rank: int
    final_score: float
    reranker_score: Optional[float]
    retrieval_score: Optional[float]
    index_position: Optional[int]
    title: Optional[str] = None
    summary: Optional[str] = None
    identity: Optional[str] = None
    paper_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class RankingConfig:
    """Immutable, validated ranking configuration."""

    strategy: RankingStrategy = RankingStrategy.RERANKER_SCORE
    top_k: int = DEFAULT_TOP_K
    allow_score_fallback: bool = False
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.REJECT
    identity_field: str = "document_id"

    def validate(self) -> None:
        if not isinstance(self.strategy, RankingStrategy):
            try:
                RankingStrategy(self.strategy)
            except (ValueError, TypeError) as exc:
                raise RankingStrategyError(
                    f"Unsupported ranking strategy: {self.strategy!r}."
                ) from exc

        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise InvalidTopKError(
                f"top_k must be a positive integer; got {self.top_k!r}."
            )

        if not isinstance(self.allow_score_fallback, bool):
            raise InvalidRankingInputError(
                "allow_score_fallback must be a boolean."
            )

        if not isinstance(self.duplicate_policy, DuplicatePolicy):
            try:
                DuplicatePolicy(self.duplicate_policy)
            except (ValueError, TypeError) as exc:
                raise InvalidRankingInputError(
                    "Unsupported duplicate policy: "
                    f"{self.duplicate_policy!r}."
                ) from exc

        if (
            not isinstance(self.identity_field, str)
            or not self.identity_field.strip()
        ):
            raise InvalidRankingInputError(
                "identity_field must be a non-empty string."
            )

        if self.identity_field.strip() != self.identity_field:
            raise InvalidRankingInputError(
                "identity_field must not contain leading/trailing whitespace."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the validated configuration."""
        self.validate()
        return {
            "strategy": self.strategy.value,
            "top_k": self.top_k,
            "allow_score_fallback": self.allow_score_fallback,
            "duplicate_policy": self.duplicate_policy.value,
            "identity_field": self.identity_field,
        }


# ============================================================================
# Candidate normalization
# ============================================================================


@dataclass(frozen=True)
class _Candidate:
    """Small internal representation; never stores unnecessary large data."""

    document_id: str
    reranker_score: Optional[float]
    retrieval_score: Optional[float]
    index_position: Optional[int]
    title: Optional[str]
    summary: Optional[str]
    identity: str
    paper_id: Optional[str]
    chunk_id: Optional[str]
    original_position: int


def _read_field(candidate: Any, field: str) -> Any:
    """Read a field from a mapping or an attribute-based candidate object."""
    if isinstance(candidate, Mapping):
        return candidate.get(field)
    return getattr(candidate, field, None)


def _optional_text(value: Any) -> Optional[str]:
    """Return a clean optional string without stringifying arbitrary objects."""
    if value is None:
        return None

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    return cleaned or None


def _required_document_id(candidate: Any, position: int) -> str:
    value = _optional_text(_read_field(candidate, "document_id"))

    if value is None:
        # Common compatibility aliases are accepted only for identity.
        value = _optional_text(_read_field(candidate, "id"))

    if value is None:
        raise InvalidRankingInputError(
            f"Candidate at position {position} has no valid document_id."
        )

    return value


def _finite_optional_score(
    value: Any,
    *,
    field_name: str,
    document_id: str,
) -> Optional[float]:
    """
    Validate an optional numeric score.

    None means genuinely unavailable.
    Any supplied invalid value is rejected; it is never converted to zero.
    """
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidScoreError(
            f"Candidate {document_id!r} has non-numeric "
            f"{field_name}={value!r}."
        )

    score = float(value)

    if not math.isfinite(score):
        raise InvalidScoreError(
            f"Candidate {document_id!r} has non-finite "
            f"{field_name}={value!r}."
        )

    return score


def _optional_index_position(
    value: Any,
    *,
    document_id: str,
) -> Optional[int]:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not float(value).is_integer()
        or int(value) < 0
    ):
        raise InvalidRankingInputError(
            f"Candidate {document_id!r} has invalid index_position="
            f"{value!r}."
        )

    return int(value)


def _resolve_score(
    candidate: Any,
    *,
    strategy: RankingStrategy,
    allow_score_fallback: bool,
    document_id: str,
) -> tuple[float, Optional[float], Optional[float]]:
    """
    Select the authoritative score without blending scales.

    Returns:
        final_score, reranker_score, retrieval_score
    """
    reranker_raw = _read_field(candidate, "reranker_score")

    retrieval_raw = _read_field(candidate, "retrieval_score")
    if retrieval_raw is None:
        # Compatibility with retriever outputs commonly exposing `score`.
        retrieval_raw = _read_field(candidate, "score")

    reranker_score = _finite_optional_score(
        reranker_raw,
        field_name="reranker_score",
        document_id=document_id,
    )

    retrieval_score = _finite_optional_score(
        retrieval_raw,
        field_name="retrieval_score",
        document_id=document_id,
    )

    if strategy == RankingStrategy.RERANKER_SCORE:
        if reranker_score is not None:
            return reranker_score, reranker_score, retrieval_score

        if allow_score_fallback and retrieval_score is not None:
            logger.warning(
                "Using retrieval-score fallback for document_id=%s.",
                document_id,
            )
            return retrieval_score, reranker_score, retrieval_score

        raise InvalidScoreError(
            f"Candidate {document_id!r} has no valid reranker_score. "
            "Set allow_score_fallback=True only when retrieval-score "
            "fallback is intentionally desired."
        )

    if strategy == RankingStrategy.RETRIEVAL_SCORE:
        if retrieval_score is not None:
            return retrieval_score, reranker_score, retrieval_score

        raise InvalidScoreError(
            f"Candidate {document_id!r} has no valid retrieval_score."
        )

    # Defensive branch in case enum handling changes later.
    raise RankingStrategyError(
        f"Unsupported ranking strategy: {strategy!r}."
    )


def _resolve_identity(candidate: Any, identity_field: str, document_id: str) -> str:
    """
    Resolve the configured identity.

    Examples:
        identity_field="document_id" -> paper-level result
        identity_field="paper_id"    -> paper grouping
        identity_field="chunk_id"    -> chunk-level result
    """
    value = _optional_text(_read_field(candidate, identity_field))

    if value is None and identity_field == "document_id":
        value = document_id

    if value is None:
        raise InvalidRankingInputError(
            f"Candidate {document_id!r} does not provide the configured "
            f"identity_field={identity_field!r}."
        )

    return value


def _normalize_candidates(
    candidates: Sequence[Any],
    config: RankingConfig,
) -> list[_Candidate]:
    if candidates is None:
        raise InvalidRankingInputError(
            "candidates cannot be None."
        )

    if isinstance(candidates, (str, bytes)):
        raise InvalidRankingInputError(
            "candidates must be a sequence of candidate objects."
        )

    try:
        items = list(candidates)
    except TypeError as exc:
        raise InvalidRankingInputError(
            "candidates must be an iterable collection."
        ) from exc

    normalized: list[_Candidate] = []

    for position, candidate in enumerate(items):
        if candidate is None:
            raise InvalidRankingInputError(
                f"Candidate at position {position} is None."
            )

        document_id = _required_document_id(candidate, position)

        final_score, reranker_score, retrieval_score = _resolve_score(
            candidate,
            strategy=config.strategy,
            allow_score_fallback=config.allow_score_fallback,
            document_id=document_id,
        )

        index_position = _optional_index_position(
            _read_field(candidate, "index_position"),
            document_id=document_id,
        )

        identity = _resolve_identity(
            candidate,
            config.identity_field,
            document_id,
        )

        paper_id = _optional_text(_read_field(candidate, "paper_id"))
        chunk_id = _optional_text(_read_field(candidate, "chunk_id"))

        normalized.append(
            _Candidate(
                document_id=document_id,
                reranker_score=reranker_score,
                retrieval_score=retrieval_score,
                index_position=index_position,
                title=_optional_text(
                    _read_field(candidate, "title")
                ),
                summary=(
                    _optional_text(_read_field(candidate, "summary"))
                    or _optional_text(_read_field(candidate, "abstract"))
                ),
                identity=identity,
                paper_id=paper_id,
                chunk_id=chunk_id,
                original_position=position,
            )
        )

    return normalized


# ============================================================================
# Ranker
# ============================================================================


class Ranker:
    """
    Deterministic final ranking/selection component.

    It accepts already scored candidates and performs only final ordering.

    Default:
        strategy=RERANKER_SCORE
        top_k=10
        fallback disabled
        duplicate policy=REJECT
        identity=document_id

    Example
    -------
    ranker = Ranker()

    results = ranker.rank(
        candidates=[
            {
                "document_id": "paper-a",
                "retrieval_score": 0.81,
                "reranker_score": 0.71,
            },
            ...
        ]
    )
    """

    def __init__(
        self,
        *,
        strategy: RankingStrategy = RankingStrategy.RERANKER_SCORE,
        top_k: int = DEFAULT_TOP_K,
        allow_score_fallback: bool = False,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.REJECT,
        identity_field: str = "document_id",
    ) -> None:
        # Convert string enum values into proper enum members while keeping
        # the public API convenient.
        try:
            resolved_strategy = (
                strategy
                if isinstance(strategy, RankingStrategy)
                else RankingStrategy(strategy)
            )
        except (ValueError, TypeError) as exc:
            raise RankingStrategyError(
                f"Unsupported ranking strategy: {strategy!r}."
            ) from exc

        try:
            resolved_duplicate_policy = (
                duplicate_policy
                if isinstance(duplicate_policy, DuplicatePolicy)
                else DuplicatePolicy(duplicate_policy)
            )
        except (ValueError, TypeError) as exc:
            raise InvalidRankingInputError(
                f"Unsupported duplicate policy: {duplicate_policy!r}."
            ) from exc

        self.config = RankingConfig(
            strategy=resolved_strategy,
            top_k=top_k,
            allow_score_fallback=allow_score_fallback,
            duplicate_policy=resolved_duplicate_policy,
            identity_field=identity_field,
        )
        self.config.validate()

    @property
    def strategy(self) -> RankingStrategy:
        return self.config.strategy

    @property
    def top_k(self) -> int:
        return self.config.top_k

    @property
    def allow_score_fallback(self) -> bool:
        return self.config.allow_score_fallback

    @property
    def duplicate_policy(self) -> DuplicatePolicy:
        return self.config.duplicate_policy

    @property
    def identity_field(self) -> str:
        return self.config.identity_field

    @property
    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the active ranking policy."""
        return self.config.to_dict()

    def rank(
        self,
        candidates: Sequence[Any],
        *,
        top_k: Optional[int] = None,
    ) -> list[RankedResult]:
        """
        Validate, rank, assign ranks, then select Top-K.

        Complexity:
            O(N log N) for N candidates. Retrieval/reranking candidate sets
            should normally be small; this layer intentionally prioritizes
            deterministic, auditable behavior over approximate top-k selection.

        Important:
            Top-K selection happens AFTER the full candidate set is sorted.
            This guarantees that duplicate handling and deterministic tie
            breaking are applied consistently before truncation.
        """
        requested_k = self._resolve_top_k(top_k)

        normalized = _normalize_candidates(
            candidates,
            self.config,
        )

        if not normalized:
            logger.info("Ranking 0 candidates; returning empty result.")
            return []

        logger.debug(
            "Ranking %d candidates using strategy=%s.",
            len(normalized),
            self.strategy.value,
        )

        normalized = self._apply_duplicate_policy(normalized)

        # Full deterministic ordering BEFORE Top-K.
        ordered = sorted(
            normalized,
            key=self._sort_key,
        )

        # Assign ranks before returning only the selected prefix.
        selected = ordered[:requested_k]

        results: list[RankedResult] = []

        for rank, candidate in enumerate(selected, start=1):
            results.append(
                RankedResult(
                    document_id=candidate.document_id,
                    rank=rank,
                    final_score=candidate.reranker_score
                    if self.strategy == RankingStrategy.RERANKER_SCORE
                    and candidate.reranker_score is not None
                    else (
                        candidate.retrieval_score
                        if self.strategy == RankingStrategy.RETRIEVAL_SCORE
                        else (
                            candidate.reranker_score
                            if candidate.reranker_score is not None
                            else candidate.retrieval_score
                        )
                    ),
                    reranker_score=candidate.reranker_score,
                    retrieval_score=candidate.retrieval_score,
                    index_position=candidate.index_position,
                    title=candidate.title,
                    summary=candidate.summary,
                    identity=candidate.identity,
                    paper_id=candidate.paper_id,
                    chunk_id=candidate.chunk_id,
                )
            )

        self._validate_final_results(results)

        logger.info(
            "Returning Top %d candidates from %d ranked candidates.",
            len(results),
            len(ordered),
        )

        return results

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        value = self.top_k if top_k is None else top_k

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise InvalidTopKError(
                f"top_k must be a positive integer; got {value!r}."
            )

        return value

    def _sort_key(self, candidate: _Candidate) -> tuple[float, float, str, int]:
        """
        Deterministic descending score ordering.

        Primary:
            final score

        Secondary:
            other available relevance score, useful for deterministic
            ordering without treating the two scales as mathematically
            additive.

        Tertiary:
            original candidate position

        Final:
            identity, which makes the ordering deterministic even if all
            numerical fields and positions happen to be identical.
        """
        final_score = (
            candidate.reranker_score
            if self.strategy == RankingStrategy.RERANKER_SCORE
            and candidate.reranker_score is not None
            else candidate.retrieval_score
        )

        if final_score is None:
            # This should be unreachable because _resolve_score validates it.
            raise InvalidScoreError(
                f"Candidate {candidate.document_id!r} has no final score."
            )

        if self.strategy == RankingStrategy.RERANKER_SCORE:
            secondary = (
                candidate.retrieval_score
                if candidate.retrieval_score is not None
                else float("-inf")
            )
        else:
            secondary = (
                candidate.reranker_score
                if candidate.reranker_score is not None
                else float("-inf")
            )

        # Negative values produce descending numeric ordering.
        #
        # Identity is intentionally before original_position so equal-scoring
        # candidates have an ordering independent of input sequence. The
        # original position remains the final defensive tie-breaker.
        return (
            -final_score,
            -secondary,
            candidate.identity,
            candidate.original_position,
        )

    def _apply_duplicate_policy(
        self,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        seen: dict[str, _Candidate] = {}
        duplicates: list[str] = []

        for candidate in candidates:
            if candidate.identity in seen:
                duplicates.append(candidate.identity)
            else:
                seen[candidate.identity] = candidate

        if not duplicates:
            return candidates

        unique_duplicates = list(dict.fromkeys(duplicates))

        if self.duplicate_policy == DuplicatePolicy.REJECT:
            raise DuplicateDocumentError(
                "Duplicate ranking identities detected: "
                f"{unique_duplicates!r}. "
                f"identity_field={self.identity_field!r}. "
                "For chunk-level retrieval, configure identity_field="
                "'chunk_id' or use duplicate_policy='keep'."
            )

        if self.duplicate_policy == DuplicatePolicy.KEEP:
            logger.warning(
                "Keeping %d duplicate ranking identities because "
                "duplicate_policy='keep'.",
                len(duplicates),
            )
            return candidates

        if self.duplicate_policy == DuplicatePolicy.KEEP_BEST:
            logger.warning(
                "Deduplicating %d duplicate identities using "
                "duplicate_policy='keep_best'.",
                len(duplicates),
            )

            # Choose the candidate with the best primary score. Ties use the
            # same deterministic policy as final sorting.
            grouped: dict[str, list[_Candidate]] = {}

            for candidate in candidates:
                grouped.setdefault(candidate.identity, []).append(candidate)

            result: list[_Candidate] = []

            for group in grouped.values():
                best = sorted(group, key=self._sort_key)[0]
                result.append(best)

            # Preserve first identity appearance here; final sorting happens
            # afterward. This makes deduplication independent of input order
            # for the actual winner.
            return result

        raise InvalidRankingInputError(
            f"Unsupported duplicate policy: {self.duplicate_policy!r}."
        )

    @staticmethod
    def _validate_final_results(
        results: Sequence[RankedResult],
    ) -> None:
        previous_score: Optional[float] = None
        seen_identities: set[str] = set()

        for expected_rank, result in enumerate(results, start=1):
            if result.rank != expected_rank:
                raise RankingError(
                    "Internal rank-assignment error: expected rank "
                    f"{expected_rank}, got {result.rank}."
                )

            if not result.document_id.strip():
                raise InvalidRankingInputError(
                    "Final result contains an empty document_id."
                )

            if not result.identity.strip():
                raise InvalidRankingInputError(
                    f"Final result {result.document_id!r} contains an empty identity."
                )

            if result.identity in seen_identities:
                raise DuplicateDocumentError(
                    f"Duplicate identity survived final ranking: {result.identity!r}."
                )
            seen_identities.add(result.identity)

            if not math.isfinite(result.final_score):
                raise InvalidScoreError(
                    f"Final score for {result.document_id!r} is not finite."
                )

            for field_name, score in (
                ("reranker_score", result.reranker_score),
                ("retrieval_score", result.retrieval_score),
            ):
                if score is not None and not math.isfinite(score):
                    raise InvalidScoreError(
                        f"{field_name} for {result.document_id!r} is not finite."
                    )

            if previous_score is not None and (
                result.final_score > previous_score
            ):
                raise RankingError(
                    "Internal ordering error: final scores are not "
                    "monotonically descending."
                )

            previous_score = result.final_score


# ============================================================================
# Functional API
# ============================================================================


def rank_candidates(
    candidates: Sequence[Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    strategy: RankingStrategy = RankingStrategy.RERANKER_SCORE,
    allow_score_fallback: bool = False,
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.REJECT,
    identity_field: str = "document_id",
) -> list[RankedResult]:
    """
    Thin functional wrapper around Ranker.

    This function contains no separate ranking logic.
    """
    return Ranker(
        strategy=strategy,
        top_k=top_k,
        allow_score_fallback=allow_score_fallback,
        duplicate_policy=duplicate_policy,
        identity_field=identity_field,
    ).rank(candidates)


# Backward-friendly aliases.
FinalRanker = Ranker


# ============================================================================
# Model-free self-test
# ============================================================================


def run_self_test() -> None:
    """Run comprehensive tests without FAISS, Transformers, GPU, or data."""
    candidates = [
        {
            "document_id": "paper-a",
            "title": "Paper A",
            "summary": "A",
            "retrieval_score": 0.81,
            "reranker_score": 0.71,
            "index_position": 0,
        },
        {
            "document_id": "paper-b",
            "title": "Paper B",
            "summary": "B",
            "retrieval_score": 0.88,
            "reranker_score": 0.94,
            "index_position": 1,
        },
        {
            "document_id": "paper-c",
            "title": "Paper C",
            "summary": "C",
            "retrieval_score": 0.92,
            "reranker_score": 0.82,
            "index_position": 2,
        },
    ]

    # ------------------------------------------------------------------
    # 1. Core reranker-score ranking
    # ------------------------------------------------------------------
    ranker = Ranker()
    results = ranker.rank(candidates)

    assert [r.document_id for r in results] == [
        "paper-b",
        "paper-c",
        "paper-a",
    ]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.final_score for r in results] == [0.94, 0.82, 0.71]

    # ------------------------------------------------------------------
    # 2. Explicit Top-K
    # ------------------------------------------------------------------
    top_two = ranker.rank(candidates, top_k=2)

    assert len(top_two) == 2
    assert [r.document_id for r in top_two] == [
        "paper-b",
        "paper-c",
    ]
    assert [r.rank for r in top_two] == [1, 2]

    # ------------------------------------------------------------------
    # 3. top_k > candidate count
    # ------------------------------------------------------------------
    all_results = ranker.rank(candidates, top_k=100)

    assert len(all_results) == 3

    # ------------------------------------------------------------------
    # 4. Empty input
    # ------------------------------------------------------------------
    assert ranker.rank([]) == []

    # ------------------------------------------------------------------
    # 5. Invalid top_k
    # ------------------------------------------------------------------
    for invalid_k in (0, -1, False, "10"):
        try:
            ranker.rank(candidates, top_k=invalid_k)  # type: ignore[arg-type]
        except InvalidTopKError:
            pass
        else:
            raise AssertionError(
                f"Invalid top_k={invalid_k!r} was accepted."
            )

    # ------------------------------------------------------------------
    # 6. Missing reranker score: strict default
    # ------------------------------------------------------------------
    missing_reranker = [
        {
            "document_id": "paper-x",
            "retrieval_score": 0.91,
        }
    ]

    try:
        ranker.rank(missing_reranker)
    except InvalidScoreError:
        pass
    else:
        raise AssertionError(
            "Missing reranker score was not rejected by strict default."
        )

    # ------------------------------------------------------------------
    # 7. Explicit retrieval-score fallback
    # ------------------------------------------------------------------
    fallback_ranker = Ranker(
        allow_score_fallback=True,
    )

    fallback = fallback_ranker.rank(missing_reranker)

    assert len(fallback) == 1
    assert fallback[0].final_score == 0.91
    assert fallback[0].reranker_score is None
    assert fallback[0].retrieval_score == 0.91

    # ------------------------------------------------------------------
    # 8. Retrieval-score strategy
    # ------------------------------------------------------------------
    retrieval_ranker = Ranker(
        strategy=RankingStrategy.RETRIEVAL_SCORE,
    )

    retrieval_results = retrieval_ranker.rank(candidates)

    assert [r.document_id for r in retrieval_results] == [
        "paper-c",
        "paper-b",
        "paper-a",
    ]
    assert [r.final_score for r in retrieval_results] == [
        0.92,
        0.88,
        0.81,
    ]

    # ------------------------------------------------------------------
    # 9. NaN
    # ------------------------------------------------------------------
    nan_candidate = [
        {
            "document_id": "nan-paper",
            "reranker_score": float("nan"),
            "retrieval_score": 0.5,
        }
    ]

    try:
        ranker.rank(nan_candidate)
    except InvalidScoreError:
        pass
    else:
        raise AssertionError("NaN reranker score was accepted.")

    # ------------------------------------------------------------------
    # 10. Inf
    # ------------------------------------------------------------------
    inf_candidate = [
        {
            "document_id": "inf-paper",
            "reranker_score": float("inf"),
            "retrieval_score": 0.5,
        }
    ]

    try:
        ranker.rank(inf_candidate)
    except InvalidScoreError:
        pass
    else:
        raise AssertionError("Inf reranker score was accepted.")

    # ------------------------------------------------------------------
    # 11. Non-numeric score
    # ------------------------------------------------------------------
    string_score = [
        {
            "document_id": "string-paper",
            "reranker_score": "0.9",
            "retrieval_score": 0.5,
        }
    ]

    try:
        ranker.rank(string_score)
    except InvalidScoreError:
        pass
    else:
        raise AssertionError("String score was accepted.")

    # ------------------------------------------------------------------
    # 12. Duplicate paper IDs: strict rejection
    # ------------------------------------------------------------------
    duplicate_papers = [
        candidates[0],
        {**candidates[1], "document_id": "paper-a"},
    ]

    try:
        ranker.rank(duplicate_papers)
    except DuplicateDocumentError:
        pass
    else:
        raise AssertionError("Duplicate document IDs were accepted.")

    # ------------------------------------------------------------------
    # 13. Chunk-level compatibility
    # ------------------------------------------------------------------
    chunks = [
        {
            "document_id": "chunk-1",
            "paper_id": "paper-1",
            "chunk_id": "chunk-1",
            "reranker_score": 0.80,
            "retrieval_score": 0.70,
        },
        {
            "document_id": "chunk-2",
            "paper_id": "paper-1",
            "chunk_id": "chunk-2",
            "reranker_score": 0.95,
            "retrieval_score": 0.60,
        },
        {
            "document_id": "chunk-3",
            "paper_id": "paper-2",
            "chunk_id": "chunk-3",
            "reranker_score": 0.90,
            "retrieval_score": 0.65,
        },
    ]

    chunk_ranker = Ranker(
        identity_field="chunk_id",
    )

    chunk_results = chunk_ranker.rank(chunks, top_k=3)

    assert [r.document_id for r in chunk_results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]

    # Same paper is allowed because identity is chunk_id.
    assert [r.identity for r in chunk_results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]

    # ------------------------------------------------------------------
    # 14. Explicit paper-level keep-best deduplication
    # ------------------------------------------------------------------
    paper_ranker = Ranker(
        identity_field="paper_id",
        duplicate_policy=DuplicatePolicy.KEEP_BEST,
    )

    paper_results = paper_ranker.rank(chunks, top_k=10)

    assert [r.identity for r in paper_results] == [
        "paper-1",
        "paper-2",
    ]
    assert [r.document_id for r in paper_results] == [
        "chunk-2",
        "chunk-3",
    ]

    # ------------------------------------------------------------------
    # 15. Deterministic tie-breaking
    # ------------------------------------------------------------------
    tied = [
        {
            "document_id": "b",
            "reranker_score": 0.9,
            "retrieval_score": 0.5,
        },
        {
            "document_id": "a",
            "reranker_score": 0.9,
            "retrieval_score": 0.5,
        },
    ]

    tie_results_1 = ranker.rank(tied)
    tie_results_2 = ranker.rank(tied)

    assert [
        r.document_id for r in tie_results_1
    ] == [
        r.document_id for r in tie_results_2
    ]

    # ------------------------------------------------------------------
    # 16. Deterministic tie order is input-order independent
    # ------------------------------------------------------------------
    reversed_tied = list(reversed(tied))
    assert [
        r.document_id for r in ranker.rank(reversed_tied)
    ] == ["a", "b"]

    # ------------------------------------------------------------------
    # 17. Configuration snapshot
    # ------------------------------------------------------------------
    config_snapshot = ranker.config_dict
    assert config_snapshot["strategy"] == "reranker_score"
    assert config_snapshot["top_k"] == DEFAULT_TOP_K
    assert config_snapshot["allow_score_fallback"] is False
    assert config_snapshot["duplicate_policy"] == "reject"
    assert config_snapshot["identity_field"] == "document_id"

    # ------------------------------------------------------------------
    # 18. Metadata preservation
    # ------------------------------------------------------------------
    assert results[0].title == "Paper B"
    assert results[0].summary == "B"
    assert results[0].index_position == 1
    assert results[0].reranker_score == 0.94
    assert results[0].retrieval_score == 0.88

    # ------------------------------------------------------------------
    # 19. Functional wrapper
    # ------------------------------------------------------------------
    functional_results = rank_candidates(
        candidates,
        top_k=1,
    )

    assert len(functional_results) == 1
    assert functional_results[0].document_id == "paper-b"

    # ------------------------------------------------------------------
    # 20. Mapping/object flexibility
    # ------------------------------------------------------------------
    @dataclass(frozen=True)
    class CandidateObject:
        document_id: str
        reranker_score: float
        retrieval_score: float

    object_result = ranker.rank(
        [
            CandidateObject(
                document_id="object-paper",
                reranker_score=0.99,
                retrieval_score=0.50,
            )
        ]
    )

    assert object_result[0].document_id == "object-paper"

    print("Ranker self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()