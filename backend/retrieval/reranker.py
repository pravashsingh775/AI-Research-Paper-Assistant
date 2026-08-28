"""
Production-grade cross-encoder reranking for the AI Research Paper Assistant.

Responsibilities
----------------
    Query + retrieved candidate documents
                |
                v
        Cross-encoder relevance scores
                |
                v
        deterministic re-ranking
                |
                v
             Top-K

This module deliberately does NOT:
- perform FAISS search
- create/load SPECTER2 embeddings
- implement dense retrieval
- call an LLM
- summarize papers
- extract models/datasets/methodology
- score scientific quality
- perform popularity/date/author ranking
- load the full corpus
- train/fine-tune a model

Default production model:
    BAAI/bge-reranker-v2-m3

The model is configurable and can be replaced with another compatible
cross-encoder/relevance model.

The implementation uses Hugging Face Transformers directly so that:
- query/document pairs are jointly encoded
- tokenizer truncation is explicit
- max_length is configurable
- inference_mode is used
- CPU/CUDA are supported
- model loading happens once per Reranker instance
- a deterministic injected fake scorer can be used in tests
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# Default is intentionally configurable and not a magic value in the class.
# BAAI's model card describes bge-reranker-v2-m3 as a query/passage reranker
# and documents direct Transformers inference with truncation/max_length.
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
DEFAULT_FINAL_K = 10

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RerankerError(RuntimeError):
    """Base exception for reranking failures."""


class InvalidRerankerInputError(RerankerError, ValueError):
    """Raised when query/candidate input is invalid."""


class InvalidRerankerConfigurationError(RerankerError, ValueError):
    """Raised when reranker configuration is invalid."""


class RerankerModelLoadError(RerankerError):
    """Raised when the tokenizer/model cannot be loaded."""


class RerankerInferenceError(RerankerError):
    """Raised when model inference fails."""


class RerankerScoreError(RerankerError, ValueError):
    """Raised when model scores are malformed or non-finite."""


class CandidateAlignmentError(RerankerError, ValueError):
    """Raised when candidate/score alignment is unsafe."""


# ---------------------------------------------------------------------------
# Public result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankedResult:
    """
    One candidate after cross-encoder reranking.

    retrieval_score:
        Original dense-retrieval score, preserved separately.

    reranker_score:
        Raw score produced by the reranker model. It is NOT automatically a
        probability or percentage.

    rank:
        One-based final reranker rank.

    title / summary:
        Preserved only when supplied by the input candidate. They are not
        semantically generated or modified by this module.
    """

    document_id: str
    reranker_score: float
    retrieval_score: Optional[float]
    rank: int
    index_position: Optional[int]
    title: Optional[str] = None
    summary: Optional[str] = None
    # Optional upstream identities preserved for ranking.py integration.
    paper_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class CandidateDocument:
    """
    Normalized candidate representation consumed by the reranker.

    document_id is the only mandatory identity field.
    title and summary are optional because Mode 2 may provide chunks/passages.
    """

    document_id: str
    title: Optional[str]
    summary: Optional[str]
    body_text: Optional[str]
    retrieval_score: Optional[float]
    index_position: Optional[int]
    paper_id: Optional[str]
    chunk_id: Optional[str]
    original_position: int


@dataclass(frozen=True)
class RerankerConfig:
    """Immutable reranker inference configuration."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    final_k: int = DEFAULT_FINAL_K
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    normalize_score: bool = False
    # Preserve ranking identities needed by ranking.py (paper/chunk mode).
    identity_fields: tuple[str, ...] = ("paper_id", "chunk_id")

    def validate(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise InvalidRerankerConfigurationError(
                "model_name must be a non-empty string."
            )

        if self.device not in {"auto", "cpu", "cuda"}:
            raise InvalidRerankerConfigurationError(
                "device must be one of: 'auto', 'cpu', 'cuda'."
            )

        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise InvalidRerankerConfigurationError(
                "batch_size must be a positive integer."
            )

        if (
            not isinstance(self.max_length, int)
            or isinstance(self.max_length, bool)
            or self.max_length <= 0
        ):
            raise InvalidRerankerConfigurationError(
                "max_length must be a positive integer."
            )

        if (
            not isinstance(self.final_k, int)
            or isinstance(self.final_k, bool)
            or self.final_k <= 0
        ):
            raise InvalidRerankerConfigurationError(
                "final_k must be a positive integer."
            )

        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, str) or not self.cache_dir.strip():
                raise InvalidRerankerConfigurationError(
                    "cache_dir must be None or a non-empty string."
                )

        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise InvalidRerankerConfigurationError(
                    "revision must be None or a non-empty string."
                )

        if not isinstance(self.identity_fields, tuple):
            raise InvalidRerankerConfigurationError(
                "identity_fields must be a tuple of field names."
            )
        for field in self.identity_fields:
            if not isinstance(field, str) or not field.strip():
                raise InvalidRerankerConfigurationError(
                    "identity_fields must contain only non-empty strings."
                )


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------


class PairScorer(Protocol):
    """
    Minimal injectable scoring interface.

    The production implementation is _TransformersCrossEncoderScorer.
    Tests can inject a deterministic fake without downloading a model.
    """

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
    ) -> Sequence[float]:
        """Return exactly one score for every input pair, in the same order."""

    def close(self) -> None:
        """Release optional model resources."""


# ---------------------------------------------------------------------------
# Candidate text construction
# ---------------------------------------------------------------------------


def _safe_optional_text(value: Any) -> Optional[str]:
    """Convert supported text-like values to stripped strings."""
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None

    # Do not stringify arbitrary objects: doing so can inject meaningless
    # representation text into a semantic model.
    return None


def _get_candidate_value(candidate: Any, field: str) -> Any:
    """Read a candidate field from a mapping or object."""
    if isinstance(candidate, Mapping):
        return candidate.get(field)

    return getattr(candidate, field, None)


def build_document_text(candidate: Any) -> str:
    """
    Build semantic reranker text from title + abstract/summary.

    For Mode 1:
        Title:
        <title>

        Abstract:
        <summary>

    For Mode 2:
        if title/summary are absent, an available `text`, `content`,
        `passage`, or `chunk_text` field may be used as the document body.

    Deliberately excluded:
        document ID, FAISS position, filesystem paths, row numbers,
        author metadata, publication date, popularity/citation information.

    Raises
    ------
    InvalidRerankerInputError
        If no usable semantic text is available.
    """
    title = _safe_optional_text(_get_candidate_value(candidate, "title"))

    summary = _safe_optional_text(
        _get_candidate_value(candidate, "summary")
    )
    if summary is None:
        summary = _safe_optional_text(
            _get_candidate_value(candidate, "abstract")
        )

    # Generic Mode-2 support without assuming that every candidate is a paper.
    body = None
    if title is None and summary is None:
        for field in ("text", "content", "passage", "chunk_text", "document"):
            body = _safe_optional_text(
                _get_candidate_value(candidate, field)
            )
            if body is not None:
                break

    parts: list[str] = []

    if title is not None:
        parts.append(f"Title:\n{title}")

    if summary is not None:
        parts.append(f"Abstract:\n{summary}")

    if not parts and body is not None:
        parts.append(body)

    if not parts:
        raise InvalidRerankerInputError(
            "Candidate has no usable semantic text. Expected title/summary "
            "for papers or text/content/passage/chunk_text for generic "
            "documents."
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Production Transformers scorer
# ---------------------------------------------------------------------------


class _TransformersCrossEncoderScorer:
    """
    Lazy-loading Hugging Face cross-encoder scorer.

    The model is loaded once on first use and reused across calls.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        cache_dir: Optional[str],
        revision: Optional[str],
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.cache_dir = cache_dir
        self.revision = revision

        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def resolved_device(self) -> Optional[str]:
        if self._device is None:
            return None
        return str(self._device)

    def _resolve_device(self, torch_module: Any) -> Any:
        if self.requested_device == "cpu":
            return torch_module.device("cpu")

        if self.requested_device == "cuda":
            if not torch_module.cuda.is_available():
                raise RerankerModelLoadError(
                    "device='cuda' was requested, but CUDA is unavailable."
                )
            return torch_module.device("cuda")

        # auto
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")

        return torch_module.device("cpu")

    def _load(self) -> None:
        if self._loaded:
            return

        logger.info(
            "Loading cross-encoder reranker model: %s",
            self.model_name,
        )

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RerankerModelLoadError(
                "Reranking requires PyTorch and Transformers. "
                "Install them with: pip install torch transformers"
            ) from exc

        try:
            device = self._resolve_device(torch)

            load_kwargs: dict[str, Any] = {}

            if self.cache_dir is not None:
                load_kwargs["cache_dir"] = self.cache_dir

            if self.revision is not None:
                load_kwargs["revision"] = self.revision

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=False,
                **load_kwargs,
            )

            model.to(device)
            model.eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._loaded = True

            logger.info(
                "Reranker model loaded successfully on device=%s.",
                device,
            )
            if device.type == "cuda":
                try:
                    logger.info(
                        "CUDA device: %s | allocated=%.1f MiB | reserved=%.1f MiB.",
                        torch.cuda.get_device_name(device),
                        torch.cuda.memory_allocated(device) / (1024 ** 2),
                        torch.cuda.memory_reserved(device) / (1024 ** 2),
                    )
                except Exception:
                    logger.debug("Unable to read CUDA memory diagnostics.", exc_info=True)

        except RerankerModelLoadError:
            raise
        except Exception as exc:
            logger.exception("Reranker model loading failed.")
            raise RerankerModelLoadError(
                f"Could not load reranker model {self.model_name!r}."
            ) from exc

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        if not pairs:
            return []

        self._load()

        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        scores: list[float] = []

        try:
            with self._torch.inference_mode():
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start : start + batch_size]

                    logger.debug(
                        "Reranker batch %d/%d.",
                        start // batch_size + 1,
                        math.ceil(len(pairs) / batch_size),
                    )

                    queries = [pair[0] for pair in batch]
                    documents = [pair[1] for pair in batch]

                    encoded = self._tokenizer(
                        queries,
                        documents,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )

                    encoded = {
                        key: value.to(self._device)
                        for key, value in encoded.items()
                    }

                    outputs = self._model(**encoded)

                    logits = outputs.logits

                    if logits.ndim == 1:
                        batch_scores = logits
                    elif logits.ndim == 2 and logits.shape[1] == 1:
                        batch_scores = logits[:, 0]
                    elif logits.ndim == 2 and logits.shape[1] == 2:
                        # A 2-label sequence classifier is commonly
                        # interpreted as class logits. Use the positive-class
                        # logit, not an invented probability.
                        batch_scores = logits[:, 1]
                    else:
                        raise RerankerScoreError(
                            "Unsupported reranker output shape: "
                            f"{tuple(logits.shape)}. Expected [N], [N,1], "
                            "or [N,2]."
                        )

                    batch_scores = batch_scores.detach().float().cpu()

                    for value in batch_scores.tolist():
                        score = float(value)

                        if not math.isfinite(score):
                            raise RerankerScoreError(
                                "Reranker returned NaN or Inf."
                            )

                        scores.append(score)

        except RerankerScoreError:
            raise
        except Exception as exc:
            logger.exception("Cross-encoder inference failed.")
            raise RerankerInferenceError(
                "Reranker model inference failed."
            ) from exc

        if len(scores) != len(pairs):
            raise CandidateAlignmentError(
                "Reranker score count does not match candidate-pair count: "
                f"scores={len(scores)}, pairs={len(pairs)}."
            )

        return scores

    def close(self) -> None:
        """Release references to model resources."""
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._loaded = False


# ---------------------------------------------------------------------------
# Main reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Query-document cross-encoder reranker.

    Construction is cheap. The actual Hugging Face model is loaded lazily on
    first rerank() call and then reused.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or local model path.
    device:
        "auto", "cuda", or "cpu".
    batch_size:
        Number of query-document pairs per inference batch.
    max_length:
        Tokenizer-level maximum sequence length. Longer inputs are truncated
        by the tokenizer. This does not claim that truncation preserves all
        information.
    final_k:
        Default number of reranked results to return.
    cache_dir:
        Optional Hugging Face cache directory.
    revision:
        Optional pinned model revision.
    normalize_score:
        If True, apply sigmoid to raw scores. This creates a [0,1] mapping but
        does not make the result a calibrated probability. False by default
        because raw model scores are the safest research/evaluation signal.
    scorer:
        Optional injected scorer for tests or alternative model backends.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        final_k: int = DEFAULT_FINAL_K,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        normalize_score: bool = False,
        scorer: Optional[PairScorer] = None,
    ) -> None:
        self.config = RerankerConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            final_k=final_k,
            cache_dir=cache_dir,
            revision=revision,
            normalize_score=normalize_score,
            identity_fields=("paper_id", "chunk_id"),
        )
        self.config.validate()

        self._scorer: PairScorer = (
            scorer
            if scorer is not None
            else _TransformersCrossEncoderScorer(
                model_name=self.config.model_name,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                revision=self.config.revision,
            )
        )

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def max_length(self) -> int:
        return self.config.max_length

    @property
    def final_k(self) -> int:
        return self.config.final_k

    @property
    def resolved_device(self) -> Optional[str]:
        """Return the actual scorer device after lazy model loading."""
        return getattr(self._scorer, "resolved_device", None)

    @property
    def model_loaded(self) -> bool:
        """Whether the production scorer has loaded its model."""
        return bool(getattr(self._scorer, "loaded", False))

    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration snapshot for diagnostics."""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "resolved_device": self.resolved_device,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "final_k": self.config.final_k,
            "cache_dir": self.config.cache_dir,
            "revision": self.config.revision,
            "normalize_score": self.config.normalize_score,
            "identity_fields": list(self.config.identity_fields),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: Optional[int] = None,
    ) -> list[RerankedResult]:
        """
        Cross-encode all supplied candidates and return the best final_k.

        Candidate order is never changed before scoring. Each score remains
        aligned with its original candidate by list position. Sorting happens
        only after all scores have been validated.

        Tie-breaking:
            1. reranker_score descending
            2. retrieval_score descending, when available
            3. original candidate order ascending

        No random ordering is introduced.
        """
        validated_query = self._validate_query(query)
        normalized_candidates = self._validate_candidates(candidates)

        if not normalized_candidates:
            return []

        requested_k = self._resolve_top_k(top_k, len(normalized_candidates))

        logger.info(
            "Reranking %d candidates with model=%s, batch_size=%d, "
            "max_length=%d, final_k=%d.",
            len(normalized_candidates),
            self.model_name,
            self.batch_size,
            self.max_length,
            requested_k,
        )

        document_texts = [
            self._build_normalized_document_text(candidate)
            for candidate in normalized_candidates
        ]

        pairs = [
            (validated_query, document_text)
            for document_text in document_texts
        ]

        start_time = time.perf_counter()

        try:
            raw_scores = self._scorer.score_pairs(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
        except RerankerError:
            raise
        except Exception as exc:
            logger.exception("Reranker scoring failed.")
            raise RerankerInferenceError(
                "Reranker scoring failed."
            ) from exc

        elapsed = time.perf_counter() - start_time

        scores = self._validate_scores(
            raw_scores,
            expected_count=len(normalized_candidates),
        )

        results: list[RerankedResult] = []

        for candidate, score in zip(normalized_candidates, scores):
            final_score = self._score_transform(score)

            results.append(
                RerankedResult(
                    document_id=candidate.document_id,
                    reranker_score=final_score,
                    retrieval_score=candidate.retrieval_score,
                    rank=0,
                    index_position=candidate.index_position,
                    title=candidate.title,
                    summary=candidate.summary,
                    paper_id=candidate.paper_id,
                    chunk_id=candidate.chunk_id,
                )
            )

        # Deterministic and research-friendly tie handling.
        results_with_position = list(
            enumerate(results)
        )

        def sort_key(
            item: tuple[int, RerankedResult],
        ) -> tuple[float, float, str, int]:
            original_position, result = item

            retrieval_score = (
                result.retrieval_score
                if result.retrieval_score is not None
                else float("-inf")
            )

            return (
                -result.reranker_score,
                -retrieval_score,
                result.document_id,
                original_position,
            )

        results_with_position.sort(key=sort_key)

        final_results = [
            RerankedResult(
                document_id=result.document_id,
                reranker_score=result.reranker_score,
                retrieval_score=result.retrieval_score,
                rank=rank,
                index_position=result.index_position,
                title=result.title,
                summary=result.summary,
                paper_id=result.paper_id,
                chunk_id=result.chunk_id,
            )
            for rank, (_, result) in enumerate(
                results_with_position[:requested_k],
                start=1,
            )
        ]

        self._validate_final_results(final_results)

        logger.info(
            "Reranking complete: returned=%d/%d, elapsed=%.4fs.",
            len(final_results),
            len(normalized_candidates),
            elapsed,
        )

        return final_results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:
        if query is None:
            raise InvalidRerankerInputError(
                "Query cannot be None."
            )

        if not isinstance(query, str):
            raise InvalidRerankerInputError(
                "Query must be a string; "
                f"got {type(query).__name__}."
            )

        query = query.strip()

        if not query:
            raise InvalidRerankerInputError(
                "Query cannot be empty or whitespace-only."
            )

        return query

    @staticmethod
    def _build_normalized_document_text(
        candidate: CandidateDocument,
    ) -> str:
        """Build semantic text from the already-validated candidate."""
        parts: list[str] = []

        if candidate.title is not None:
            parts.append(f"Title:\n{candidate.title}")

        if candidate.summary is not None:
            parts.append(f"Summary:\n{candidate.summary}")

        if not parts and candidate.body_text is not None:
            parts.append(candidate.body_text)

        if not parts:
            raise InvalidRerankerInputError(
                f"Candidate {candidate.document_id!r} has no usable "
                "semantic text."
            )

        return "\n\n".join(parts)

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[Any],
    ) -> list[CandidateDocument]:
        if candidates is None:
            raise InvalidRerankerInputError(
                "candidates cannot be None."
            )

        if isinstance(candidates, (str, bytes)):
            raise InvalidRerankerInputError(
                "candidates must be a sequence of candidate documents."
            )

        try:
            items = list(candidates)
        except TypeError as exc:
            raise InvalidRerankerInputError(
                "candidates must be an iterable sequence."
            ) from exc

        normalized: list[CandidateDocument] = []
        # Mode 1: one candidate per paper/document.
        # Mode 2: multiple chunks may legitimately share the same document_id,
        # so identity must become (document_id, chunk_id) when chunk_id exists.
        seen_identities: set[tuple[str, Optional[str]]] = set()

        for position, candidate in enumerate(items):
            if candidate is None:
                raise InvalidRerankerInputError(
                    f"Candidate at position {position} is None."
                )

            document_id = _safe_optional_text(
                _get_candidate_value(candidate, "document_id")
            )

            if document_id is None:
                raise CandidateAlignmentError(
                    "Candidate at position "
                    f"{position} has no valid document_id."
                )

            title = _safe_optional_text(
                _get_candidate_value(candidate, "title")
            )

            summary = _safe_optional_text(
                _get_candidate_value(candidate, "summary")
            )
            if summary is None:
                summary = _safe_optional_text(
                    _get_candidate_value(candidate, "abstract")
                )

            paper_id = _safe_optional_text(
                _get_candidate_value(candidate, "paper_id")
            )
            chunk_id = _safe_optional_text(
                _get_candidate_value(candidate, "chunk_id")
            )

            identity = (document_id, chunk_id)
            if identity in seen_identities:
                raise CandidateAlignmentError(
                    "Duplicate reranker candidate identity detected: "
                    f"document_id={document_id!r}, chunk_id={chunk_id!r}."
                )
            seen_identities.add(identity)

            body_text: Optional[str] = None
            if title is None and summary is None:
                for field in (
                    "text",
                    "content",
                    "passage",
                    "chunk_text",
                    "document",
                ):
                    body_text = _safe_optional_text(
                        _get_candidate_value(candidate, field)
                    )
                    if body_text is not None:
                        break

            retrieval_score_raw = _get_candidate_value(
                candidate,
                "retrieval_score",
            )
            if retrieval_score_raw is None:
                # DenseRetriever uses "score". Preserve compatibility without
                # confusing it with the new reranker score.
                retrieval_score_raw = _get_candidate_value(
                    candidate,
                    "score",
                )

            retrieval_score: Optional[float]
            if retrieval_score_raw is None:
                retrieval_score = None
            else:
                if (
                    isinstance(retrieval_score_raw, bool)
                    or not isinstance(
                        retrieval_score_raw,
                        (int, float),
                    )
                ):
                    raise CandidateAlignmentError(
                        f"Candidate {document_id!r} has an invalid "
                        f"retrieval score: {retrieval_score_raw!r}."
                    )

                retrieval_score = float(retrieval_score_raw)

                if not math.isfinite(retrieval_score):
                    raise CandidateAlignmentError(
                        f"Candidate {document_id!r} has a non-finite "
                        "retrieval score."
                    )

            index_position_raw = _get_candidate_value(
                candidate,
                "index_position",
            )

            index_position: Optional[int]
            if index_position_raw is None:
                index_position = None
            else:
                if (
                    isinstance(index_position_raw, bool)
                    or not isinstance(index_position_raw, int)
                    or index_position_raw < 0
                ):
                    raise CandidateAlignmentError(
                        f"Candidate {document_id!r} has invalid "
                        f"index_position={index_position_raw!r}."
                    )

                index_position = int(index_position_raw)

            normalized.append(
                CandidateDocument(
                    document_id=document_id,
                    title=title,
                    summary=summary,
                    body_text=body_text,
                    retrieval_score=retrieval_score,
                    index_position=index_position,
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    original_position=position,
                )
            )

        return normalized

    @staticmethod
    def _validate_scores(
        scores: Sequence[float],
        *,
        expected_count: int,
    ) -> list[float]:
        if scores is None:
            raise CandidateAlignmentError(
                "Reranker returned None instead of scores."
            )

        try:
            values = list(scores)
        except TypeError as exc:
            raise CandidateAlignmentError(
                "Reranker returned a non-iterable score collection."
            ) from exc

        if len(values) != expected_count:
            raise CandidateAlignmentError(
                "Reranker score count does not match candidate count: "
                f"scores={len(values)}, candidates={expected_count}."
            )

        validated: list[float] = []

        for position, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RerankerScoreError(
                    f"Invalid reranker score at position {position}: "
                    f"{value!r}."
                )

            score = float(value)

            if not math.isfinite(score):
                raise RerankerScoreError(
                    f"Reranker score at position {position} is "
                    f"NaN/Inf: {score!r}."
                )

            validated.append(score)

        return validated

    @staticmethod
    def _validate_final_results(results: Sequence[RerankedResult]) -> None:
        """Defensive postcondition checks before results leave the module."""
        seen: set[tuple[str, Optional[str]]] = set()
        previous_score: Optional[float] = None

        for expected_rank, result in enumerate(results, start=1):
            if result.rank != expected_rank:
                raise RerankerError(
                    f"Invalid final rank: expected {expected_rank}, got {result.rank}."
                )
            if not result.document_id:
                raise CandidateAlignmentError("Final result has empty document_id.")
            identity = (result.document_id, result.chunk_id)
            if identity in seen:
                raise CandidateAlignmentError(
                    "Duplicate final candidate identity: "
                    f"document_id={result.document_id!r}, "
                    f"chunk_id={result.chunk_id!r}."
                )
            seen.add(identity)

            if not math.isfinite(result.reranker_score):
                raise RerankerScoreError(
                    f"Non-finite final score for {result.document_id!r}."
                )
            if (
                previous_score is not None
                and result.reranker_score > previous_score
            ):
                raise RerankerError("Final reranker scores are not descending.")
            previous_score = result.reranker_score

    def _resolve_top_k(
        self,
        top_k: Optional[int],
        candidate_count: int,
    ) -> int:
        if top_k is None:
            value = self.final_k
        else:
            if (
                not isinstance(top_k, int)
                or isinstance(top_k, bool)
                or top_k <= 0
            ):
                raise InvalidRerankerConfigurationError(
                    f"top_k must be a positive integer; got {top_k!r}."
                )
            value = top_k

        return min(value, candidate_count)

    def _score_transform(self, score: float) -> float:
        if not self.config.normalize_score:
            return score

        # Sigmoid is a monotonic mapping only; it does not make a calibrated
        # probability. This is explicitly documented in the API semantics.
        if score >= 0:
            z = math.exp(-score)
            normalized = 1.0 / (1.0 + z)
        else:
            z = math.exp(score)
            normalized = z / (1.0 + z)

        if not math.isfinite(normalized):
            raise RerankerScoreError(
                "Score normalization produced a non-finite value."
            )

        return normalized

    def close(self) -> None:
        """Release the injected/production scorer's resources."""
        try:
            self._scorer.close()
        except Exception as exc:
            raise RerankerError(
                "Failed to close reranker scorer."
            ) from exc

    def __enter__(self) -> "CrossEncoderReranker":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


# Backward-friendly generic alias.
Reranker = CrossEncoderReranker


# ---------------------------------------------------------------------------
# Deterministic fake scorer for unit tests
# ---------------------------------------------------------------------------


class FakeRerankerScorer:
    """
    Deterministic scorer for unit tests.

    The caller supplies one score per pair. Calls must receive exactly the
    expected number of pairs. No transformer model is downloaded.
    """

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = [float(score) for score in scores]
        self.calls = 0
        self.last_pairs: list[tuple[str, str]] = []
        self.last_batch_size: Optional[int] = None
        self.last_max_length: Optional[int] = None

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        self.calls += 1
        self.last_pairs = list(pairs)
        self.last_batch_size = batch_size
        self.last_max_length = max_length

        if len(self._scores) != len(pairs):
            raise CandidateAlignmentError(
                "Fake scorer score count does not match pair count."
            )

        return list(self._scores)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """
    Complete model-free test suite.

    Tests the orchestration contract without downloading BGE or requiring
    CUDA.
    """
    candidates = [
        {
            "document_id": "paper-a",
            "title": "Transformer medical image segmentation",
            "summary": "A transformer architecture for medical image segmentation.",
            "score": 0.91,
            "index_position": 0,
        },
        {
            "document_id": "paper-b",
            "title": "CNN image classification",
            "summary": "Image classification using convolutional neural networks.",
            "score": 0.89,
            "index_position": 1,
        },
        {
            "document_id": "paper-c",
            "title": "Medical image segmentation with U-Net",
            "summary": "A U-Net based medical image segmentation method.",
            "score": 0.88,
            "index_position": 2,
        },
    ]

    fake = FakeRerankerScorer([0.20, 0.95, 0.95])

    reranker = CrossEncoderReranker(
        model_name="test-model",
        device="cpu",
        batch_size=2,
        max_length=128,
        final_k=2,
        scorer=fake,
    )

    results = reranker.rerank(
        "deep learning based medical image segmentation",
        candidates,
    )

    assert fake.calls == 1
    assert len(fake.last_pairs) == 3
    assert fake.last_batch_size == 2
    assert fake.last_max_length == 128

    # Equal reranker score: retrieval score is the deterministic tie-breaker.
    assert [result.document_id for result in results] == [
        "paper-b",
        "paper-c",
    ]

    assert [result.rank for result in results] == [1, 2]
    assert results[0].reranker_score == 0.95
    assert results[0].retrieval_score == 0.89
    assert results[1].retrieval_score == 0.88

    assert results[0].title == candidates[1]["title"]
    assert results[0].summary == candidates[1]["summary"]

    # Identity metadata is preserved for ranking.py paper/chunk modes.
    identity_candidate = {
        "document_id": "identity-paper",
        "paper_id": "paper-42",
        "chunk_id": "chunk-7",
        "title": "Identity test",
        "summary": "Identity preservation.",
        "score": 0.5,
    }
    identity_fake = FakeRerankerScorer([0.7])
    identity_reranker = CrossEncoderReranker(
        model_name="test-model",
        device="cpu",
        scorer=identity_fake,
    )
    identity_result = identity_reranker.rerank("identity", [identity_candidate])[0]
    assert identity_result.paper_id == "paper-42"
    assert identity_result.chunk_id == "chunk-7"

    # Exact candidate-score alignment.
    assert fake.last_pairs[0][0] == (
        "deep learning based medical image segmentation"
    )
    assert fake.last_pairs[0][1].startswith("Title:\n")

    # Query validation.
    for invalid_query in (None, "", "   ", 123):
        try:
            reranker.rerank(
                invalid_query,  # type: ignore[arg-type]
                candidates,
            )
        except InvalidRerankerInputError:
            pass
        else:
            raise AssertionError(
                f"Invalid query {invalid_query!r} was not rejected."
            )

    # Empty candidates are valid and return no results.
    assert reranker.rerank("valid query", []) == []

    # Explicit top_k.
    top_one = reranker.rerank(
        "valid query",
        candidates,
        top_k=1,
    )
    assert len(top_one) == 1
    assert top_one[0].rank == 1

    # Invalid top_k.
    for invalid_k in (0, -1, False):
        try:
            reranker.rerank(
                "valid query",
                candidates,
                top_k=invalid_k,  # type: ignore[arg-type]
            )
        except InvalidRerankerConfigurationError:
            pass
        else:
            raise AssertionError(
                f"Invalid top_k={invalid_k!r} was not rejected."
            )

    # Duplicate IDs.
    duplicate_candidates = [
        candidates[0],
        {**candidates[1], "document_id": "paper-a"},
    ]

    try:
        reranker.rerank("valid query", duplicate_candidates)
    except CandidateAlignmentError:
        pass
    else:
        raise AssertionError("Duplicate IDs were not rejected.")

    # Invalid retrieval score.
    invalid_score = [
        {**candidates[0], "score": float("nan")},
    ]

    try:
        reranker.rerank("valid query", invalid_score)
    except CandidateAlignmentError:
        pass
    else:
        raise AssertionError("NaN retrieval score was not rejected.")

    # Invalid reranker score.
    bad_scorer = FakeRerankerScorer([float("nan"), 0.1, 0.2])
    bad_reranker = CrossEncoderReranker(
        model_name="test-model",
        device="cpu",
        scorer=bad_scorer,
    )

    try:
        bad_reranker.rerank("valid query", candidates)
    except RerankerScoreError:
        pass
    else:
        raise AssertionError("NaN reranker score was not rejected.")

    # Score-count mismatch.
    mismatch_scorer = FakeRerankerScorer([0.1])
    mismatch_reranker = CrossEncoderReranker(
        model_name="test-model",
        device="cpu",
        scorer=mismatch_scorer,
    )

    try:
        mismatch_reranker.rerank("valid query", candidates)
    except CandidateAlignmentError:
        pass
    else:
        raise AssertionError("Score-count mismatch was not rejected.")

    # Missing semantic text.
    no_text = [
        {
            "document_id": "empty-paper",
            "score": 0.5,
            "index_position": 0,
        }
    ]

    try:
        reranker.rerank("valid query", no_text)
    except InvalidRerankerInputError:
        pass
    else:
        raise AssertionError("Missing candidate text was not rejected.")

    # Mode-2 generic chunk compatibility.
    chunk = [
        {
            "document_id": "chunk-1",
            "text": "The dataset contains 10,000 annotated images.",
            "score": 0.7,
            "index_position": 10,
        }
    ]

    chunk_scorer = FakeRerankerScorer([0.8])
    chunk_reranker = CrossEncoderReranker(
        model_name="test-model",
        scorer=chunk_scorer,
    )

    chunk_result = chunk_reranker.rerank(
        "What dataset was used?",
        chunk,
    )

    assert len(chunk_result) == 1

    # Mode 2 must allow multiple chunks from the same paper/document.
    multi_chunk_candidates = [
        {
            "document_id": "paper-1",
            "chunk_id": "chunk-1",
            "text": "The dataset contains annotated images.",
            "score": 0.70,
        },
        {
            "document_id": "paper-1",
            "chunk_id": "chunk-2",
            "text": "The model uses a transformer encoder.",
            "score": 0.69,
        },
    ]
    multi_chunk_reranker = CrossEncoderReranker(
        model_name="test-model",
        scorer=FakeRerankerScorer([0.9, 0.8]),
    )
    multi_chunk_result = multi_chunk_reranker.rerank(
        "What model was used?",
        multi_chunk_candidates,
        top_k=2,
    )
    assert [item.chunk_id for item in multi_chunk_result] == [
        "chunk-1",
        "chunk-2",
    ]
    assert chunk_scorer.last_pairs[0][1].startswith(
        "The dataset contains"
    )

    # Normalized score option: monotonic sigmoid, still not a probability
    # claim.
    normalized_scorer = FakeRerankerScorer([0.0])
    normalized_reranker = CrossEncoderReranker(
        model_name="test-model",
        scorer=normalized_scorer,
        normalize_score=True,
    )

    normalized_result = normalized_reranker.rerank(
        "valid query",
        [candidates[0]],
    )

    assert math.isclose(
        normalized_result[0].reranker_score,
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    print("CrossEncoderReranker self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()