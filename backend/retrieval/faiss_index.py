"""
FAISS vector-index abstraction for AI Research Paper Assistant.

Responsibilities
----------------
- Maintain a FAISS vector index.
- Validate embedding matrices and document IDs.
- Preserve vector-position <-> document-ID alignment.
- Perform similarity search.
- Persist/restore the FAISS index plus a validated JSON manifest.

This module deliberately does NOT:
- load SPECTER2 or any embedding model
- tokenize or preprocess text
- perform PDF parsing/chunking
- rerank candidates
- run BM25/hybrid retrieval
- call an LLM
- summarize or analyze papers

Default retrieval metric:
    L2-normalized vectors + FAISS IndexFlatIP

For normalized vectors:
    cosine(a, b) == dot(a, b)

Therefore IndexFlatIP gives exact cosine-similarity nearest-neighbor
search when both corpus and query vectors are L2-normalized.

Persistence layout:
    <path>.index
    <path>.manifest.json

The mapping is stored in the manifest, preserving:
    FAISS position i <-> document_ids[i]

Security:
    Loading uses FAISS's native index reader, not Python pickle.
    The manifest is strict JSON and is schema-validated.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "FAISS is required for backend/retrieval/faiss_index.py. "
        "Install CPU FAISS with: pip install faiss-cpu"
    ) from exc


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FAISSIndexError(RuntimeError):
    """Base exception for FAISS index failures."""


class InvalidEmbeddingError(FAISSIndexError, ValueError):
    """Raised when embeddings are invalid."""


class DimensionMismatchError(InvalidEmbeddingError):
    """Raised when embedding dimensions do not match the index."""


class DocumentMappingError(FAISSIndexError, ValueError):
    """Raised when document IDs cannot be safely aligned with vectors."""


class IndexPersistenceError(FAISSIndexError):
    """Raised when saving/loading an index fails."""


class IndexNotLoadedError(FAISSIndexError):
    """Raised when an operation requires an initialized FAISS index."""


# ---------------------------------------------------------------------------
# Public result / manifest objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """
    One raw FAISS retrieval result.

    score:
        For normalized IndexFlatIP search, this is cosine similarity.
        It is NOT a probability and must not be displayed as a percentage
        unless a separate calibration layer is explicitly implemented.
    """

    document_id: str
    score: float
    index_position: int


@dataclass(frozen=True)
class IndexManifest:
    """Serializable metadata describing an on-disk index."""

    schema_version: str
    index_type: str
    metric: str
    similarity: str
    normalized: bool
    embedding_dimension: int
    document_count: int
    document_ids: tuple[str, ...]
    model_name: Optional[str]
    model_version: Optional[str]
    created_at: str
    index_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "index_type": self.index_type,
            "metric": self.metric,
            "similarity": self.similarity,
            "normalized": self.normalized,
            "embedding_dimension": self.embedding_dimension,
            "document_count": self.document_count,
            "document_ids": list(self.document_ids),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "index_file": self.index_file,
        }


# ---------------------------------------------------------------------------
# FAISS vector index
# ---------------------------------------------------------------------------


class FAISSVectorIndex:
    """
    Exact dense vector index using FAISS IndexFlatIP by default.

    The class is generic: it knows only vectors and opaque document IDs.
    It can therefore serve:
        - research-paper retrieval
        - uploaded-paper chunks
        - other future text collections

    Parameters
    ----------
    dimension:
        Positive embedding dimension. Required for an empty index.
    normalize:
        If True, every added/search vector is L2-normalized. This gives cosine
        similarity semantics with IndexFlatIP.
    model_name / model_version:
        Optional metadata supplied by the caller. Never inferred/fabricated.
    expected_document_count:
        Optional corpus completeness check. If supplied, the index must contain
        exactly this many vectors when save() is called.
    """

    SCHEMA_VERSION = "1"
    INDEX_TYPE = "IndexFlatIP"
    METRIC = "inner_product"
    SIMILARITY = "cosine"

    def __init__(
        self,
        dimension: int,
        *,
        normalize: bool = True,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        expected_document_count: Optional[int] = None,
    ) -> None:
        self._dimension = self._validate_dimension(dimension)
        self._normalize = bool(normalize)

        self._model_name = self._validate_optional_metadata(
            model_name, "model_name"
        )
        self._model_version = self._validate_optional_metadata(
            model_version, "model_version"
        )

        if expected_document_count is not None:
            self._validate_non_negative_int(
                expected_document_count,
                "expected_document_count",
            )

        self._expected_document_count = expected_document_count

        # Explicitly use exact inner-product search. For normalized vectors
        # this is exact cosine nearest-neighbor search.
        self._index = faiss.IndexFlatIP(self._dimension)

        # Position i in FAISS always maps to _document_ids[i].
        self._document_ids: list[str] = []
        # Keep a parallel O(1) membership structure. Rebuilding a set from
        # all IDs on every batch would make large corpus indexing needlessly
        # expensive (especially for 287K+ papers).
        self._document_id_set: set[str] = set()

        # FAISS mutation/search and mapping updates must remain consistent.
        # Adapter/model inference is handled by SPECTER2; this lock only
        # protects the index object's own mutable state.
        self._lock = threading.RLock()

        logger.info(
            "Created FAISS %s: dimension=%d, normalize=%s",
            self.INDEX_TYPE,
            self._dimension,
            self._normalize,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def normalize(self) -> bool:
        """Whether vectors are normalized before insertion/search."""
        return self._normalize

    @property
    def ntotal(self) -> int:
        """Number of vectors currently indexed."""
        return int(self._index.ntotal)

    @property
    def document_count(self) -> int:
        """Number of mapped document IDs."""
        return len(self._document_ids)

    @property
    def index_type(self) -> str:
        """FAISS index type."""
        return self.INDEX_TYPE

    @property
    def metric(self) -> str:
        """Raw FAISS metric."""
        return self.METRIC

    @property
    def similarity(self) -> str:
        """Human-readable similarity semantics."""
        return self.SIMILARITY if self._normalize else "inner_product"

    @property
    def document_ids(self) -> tuple[str, ...]:
        """
        Immutable snapshot of the current FAISS-position mapping.

        Position i maps to document_ids[i].
        """
        with self._lock:
            return tuple(self._document_ids)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dimension(dimension: int) -> int:
        if (
            not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, bool)
            or int(dimension) <= 0
        ):
            raise DimensionMismatchError(
                f"dimension must be a positive integer, got {dimension!r}."
            )
        return int(dimension)

    @staticmethod
    def _validate_non_negative_int(value: int, name: str) -> int:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or int(value) < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer, got {value!r}."
            )
        return int(value)

    @staticmethod
    def _validate_optional_metadata(
        value: Optional[str],
        name: str,
    ) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be None or a non-empty string.")
        return value.strip()

    @staticmethod
    def _validate_document_ids(
        document_ids: Sequence[str],
        *,
        expected_count: int,
    ) -> list[str]:
        if document_ids is None:
            raise DocumentMappingError("document_ids cannot be None.")

        if isinstance(document_ids, (str, bytes)):
            raise DocumentMappingError(
                "document_ids must be a sequence of IDs, not one raw string."
            )

        try:
            ids = list(document_ids)
        except TypeError as exc:
            raise DocumentMappingError(
                "document_ids must be an iterable sequence of IDs."
            ) from exc

        if len(ids) != expected_count:
            raise DocumentMappingError(
                "Embedding/document-ID count mismatch: "
                f"{expected_count} vectors vs {len(ids)} IDs."
            )

        cleaned: list[str] = []
        seen: set[str] = set()

        for position, document_id in enumerate(ids):
            if not isinstance(document_id, str):
                raise DocumentMappingError(
                    f"document_ids[{position}] must be a string, "
                    f"got {type(document_id).__name__}."
                )

            value = document_id.strip()
            if not value:
                raise DocumentMappingError(
                    f"document_ids[{position}] cannot be empty."
                )

            if value in seen:
                raise DocumentMappingError(
                    f"Duplicate document ID detected: {value!r}. "
                    "IDs must map one-to-one to FAISS positions."
                )

            seen.add(value)
            cleaned.append(value)

        return cleaned

    def _validate_embedding_matrix(
        self,
        embeddings: np.ndarray,
        *,
        allow_empty: bool = False,
    ) -> np.ndarray:
        """
        Validate and, only when required, convert a matrix to contiguous
        float32. FAISS's CPU IndexFlatIP expects float32 vectors.

        Conversion is explicit and occurs once at the FAISS boundary.
        """
        if not isinstance(embeddings, np.ndarray):
            raise InvalidEmbeddingError(
                "embeddings must be a numpy.ndarray."
            )

        if embeddings.ndim != 2:
            raise InvalidEmbeddingError(
                "embeddings must be 2-D with shape "
                f"(n, {self._dimension}); got {embeddings.shape}."
            )

        if embeddings.shape[1] != self._dimension:
            raise DimensionMismatchError(
                f"Expected embedding dimension {self._dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if embeddings.shape[0] == 0 and not allow_empty:
            raise InvalidEmbeddingError(
                "Cannot add an empty embedding batch."
            )

        if embeddings.dtype.kind not in {"f"}:
            raise InvalidEmbeddingError(
                "Embeddings must have a floating-point dtype; "
                f"got {embeddings.dtype}."
            )

        if not np.isfinite(embeddings).all():
            raise InvalidEmbeddingError(
                "Embeddings contain NaN or infinite values."
            )

        # The FAISS boundary is explicitly float32. When normalization is
        # enabled, copy the caller's data before in-place normalization so
        # ``add()`` never mutates the caller-owned embedding matrix.
        matrix = embeddings.astype(
            np.float32,
            copy=self._normalize,
        )

        if not np.isfinite(matrix).all():
            raise InvalidEmbeddingError(
                "Embeddings became non-finite during float32 conversion."
            )

        if self._normalize and matrix.shape[0] > 0:
            self._normalize_matrix_in_place(matrix)

        # Ensure FAISS receives C-contiguous float32 memory.
        if not matrix.flags.c_contiguous:
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)

        return matrix

    def _validate_query(self, query_embedding: np.ndarray) -> np.ndarray:
        if not isinstance(query_embedding, np.ndarray):
            raise InvalidEmbeddingError(
                "query_embedding must be a numpy.ndarray."
            )

        if query_embedding.ndim == 1:
            if query_embedding.shape[0] != self._dimension:
                raise DimensionMismatchError(
                    f"Expected query dimension {self._dimension}, "
                    f"got {query_embedding.shape[0]}."
                )
            query = query_embedding.reshape(1, -1)
        elif query_embedding.ndim == 2:
            if query_embedding.shape != (1, self._dimension):
                raise InvalidEmbeddingError(
                    "query_embedding must have shape "
                    f"({self._dimension},) or (1, {self._dimension}); "
                    f"got {query_embedding.shape}."
                )
            query = query_embedding
        else:
            raise InvalidEmbeddingError(
                "query_embedding must be 1-D or a single-row 2-D array."
            )

        if query.dtype.kind not in {"f"}:
            raise InvalidEmbeddingError(
                f"Query embedding must be floating-point; got {query.dtype}."
            )

        if not np.isfinite(query).all():
            raise InvalidEmbeddingError(
                "Query embedding contains NaN or infinite values."
            )

        # Never mutate the caller's query array during normalization.
        query = query.astype(
            np.float32,
            copy=self._normalize,
        )

        if self._normalize:
            self._normalize_matrix_in_place(query)

        if not query.flags.c_contiguous:
            query = np.ascontiguousarray(query, dtype=np.float32)

        return query

    def _normalize_matrix_in_place(self, matrix: np.ndarray) -> None:
        """
        L2-normalize float32 vectors safely.

        Norms are calculated in float64 for numerical stability. No zero vector
        is ever divided by zero.
        """
        if matrix.shape[0] == 0:
            return

        norms = np.linalg.norm(
            matrix.astype(np.float64, copy=False),
            axis=1,
        )

        if not np.isfinite(norms).all():
            raise InvalidEmbeddingError(
                "Vector norms contain NaN or infinite values."
            )

        zero_mask = norms <= np.finfo(np.float64).eps
        if np.any(zero_mask):
            first_zero = int(np.flatnonzero(zero_mask)[0])
            raise InvalidEmbeddingError(
                "Zero/near-zero embedding vector cannot be normalized "
                f"(batch row {first_zero})."
            )

        matrix /= norms.astype(np.float32)[:, None]

        if not np.isfinite(matrix).all():
            raise InvalidEmbeddingError(
                "Normalization produced NaN or infinite values."
            )

    def _validate_alignment(self) -> None:
        if self.ntotal != len(self._document_ids):
            raise DocumentMappingError(
                "FAISS/document-ID alignment is broken: "
                f"FAISS ntotal={self.ntotal}, "
                f"mapping length={len(self._document_ids)}."
            )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        document_ids: Sequence[str],
    ) -> None:
        """
        Append a batch of vectors and IDs.

        This operation is atomic with respect to validation: all inputs are
        validated before FAISS is mutated.

        Existing IDs are rejected. Existing vectors remain untouched if any
        validation fails.
        """
        matrix = self._validate_embedding_matrix(embeddings)

        ids = self._validate_document_ids(
            document_ids,
            expected_count=matrix.shape[0],
        )

        # Validate the complete batch before touching FAISS.
        if len(set(ids)) != len(ids):
            # _validate_document_ids already checks duplicates, but retaining
            # this invariant here makes the mutation boundary explicit.
            raise DocumentMappingError("Duplicate document IDs in add batch.")

        with self._lock:
            duplicate_existing = [
                doc_id for doc_id in ids
                if doc_id in self._document_id_set
            ]

            if duplicate_existing:
                raise DocumentMappingError(
                    "Document ID already exists in index: "
                    f"{duplicate_existing[0]!r}. "
                    "Use an explicit rebuild workflow instead of silently "
                    "creating multiple positions for one ID."
                )

            # All validation completed. Only now mutate FAISS + mapping.
            old_count = self.ntotal

            try:
                self._index.add(matrix)
                self._document_ids.extend(ids)
                self._document_id_set.update(ids)
                self._validate_alignment()
            except Exception as exc:
                # FAISS IndexFlatIP does not expose a transactional rollback API.
                # If a post-add invariant unexpectedly fails, surface the failure
                # loudly rather than silently hiding possible state corruption.
                logger.exception(
                    "FAISS mutation failed after starting at position %d.",
                    old_count,
                )
                raise FAISSIndexError(
                    "FAISS index mutation failed; do not continue using this "
                    "instance until it has been reloaded/rebuilt."
                ) from exc

        logger.info(
            "Added %d embeddings. Index now contains %d vectors.",
            len(ids),
            self.ntotal,
        )

    # Alias requested by some callers.
    add_documents = add

    def clear(self) -> None:
        """Explicitly rebuild the index from an empty state."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._document_ids.clear()
            self._document_id_set.clear()
        logger.info("FAISS index cleared and rebuilt as an empty index.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[SearchResult]:
        """
        Search for nearest vectors.

        Parameters
        ----------
        query_embedding:
            Shape (dimension,) or (1, dimension).
        top_k:
            Positive number of desired results.
        exclude_ids:
            Optional document IDs to exclude. Filtering is performed after
            FAISS candidate retrieval and does not alter the raw FAISS index.

        Returns
        -------
        list[SearchResult]
            Ordered from highest similarity score to lowest.

        Notes
        -----
        If the index contains fewer vectors than top_k, the effective FAISS K
        is clamped to ntotal. The result list therefore contains at most
        ntotal entries.
        """
        top_k = self._validate_top_k(top_k)

        if self.ntotal == 0:
            return []

        query = self._validate_query(query_embedding)

        if exclude_ids is None:
            exclusions: set[str] = set()
        else:
            if not isinstance(exclude_ids, set):
                exclusions = set(exclude_ids)
            else:
                exclusions = set(exclude_ids)

            for document_id in exclusions:
                if not isinstance(document_id, str) or not document_id.strip():
                    raise DocumentMappingError(
                        "exclude_ids must contain non-empty string IDs."
                    )

        with self._lock:
            self._validate_alignment()

            # Without exclusions, FAISS only needs the requested K.
            if not exclusions:
                search_k = min(top_k, self.ntotal)
                scores, positions = self._index.search(query, search_k)
            else:
                # Avoid scanning all 287K+ vectors merely because a caller
                # excluded a few IDs. Start with a modest over-fetch and grow
                # only if excluded candidates consume the result set.
                search_k = min(
                    self.ntotal,
                    max(top_k, top_k + len(exclusions)),
                )
                scores, positions = self._index.search(query, search_k)

            results: list[SearchResult] = []

            while True:
                results.clear()

                for score, position in zip(scores[0], positions[0]):
                    position_int = int(position)

                    # FAISS can use -1 for missing neighbors in some index
                    # types. IndexFlatIP should not produce it here, but
                    # guard defensively.
                    if position_int < 0:
                        continue

                    if position_int >= len(self._document_ids):
                        raise DocumentMappingError(
                            "FAISS returned an out-of-range document position: "
                            f"{position_int}."
                        )

                    document_id = self._document_ids[position_int]

                    if document_id in exclusions:
                        continue

                    results.append(
                        SearchResult(
                            document_id=document_id,
                            score=float(score),
                            index_position=position_int,
                        )
                    )

                    if len(results) >= top_k:
                        break

                if len(results) >= top_k or search_k >= self.ntotal:
                    break

                # Not enough unexcluded candidates. Double the candidate pool
                # until we either have enough results or have exhausted the
                # index.
                next_k = min(
                    self.ntotal,
                    max(search_k + 1, search_k * 2),
                )
                if next_k == search_k:
                    break

                search_k = next_k
                scores, positions = self._index.search(query, search_k)

            # IndexFlatIP returns descending inner-product scores. Sorting
            # again makes the public result contract explicit and deterministic.
            results.sort(
                key=lambda result: (
                    -result.score,
                    result.index_position,
                )
            )

            return results

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if (
            not isinstance(top_k, (int, np.integer))
            or isinstance(top_k, bool)
            or int(top_k) <= 0
        ):
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")
        return int(top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _build_manifest(self, index_filename: str) -> IndexManifest:
        self._validate_alignment()

        if (
            self._expected_document_count is not None
            and self.ntotal != self._expected_document_count
        ):
            raise IndexPersistenceError(
                "Corpus completeness check failed: "
                f"expected {self._expected_document_count} vectors, "
                f"but index contains {self.ntotal}."
            )

        return IndexManifest(
            schema_version=self.SCHEMA_VERSION,
            index_type=self.INDEX_TYPE,
            metric=self.METRIC,
            similarity=self.similarity,
            normalized=self._normalize,
            embedding_dimension=self._dimension,
            document_count=self.ntotal,
            document_ids=tuple(self._document_ids),
            model_name=self._model_name,
            model_version=self._model_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            index_file=index_filename,
        )

    @staticmethod
    def _paths(path: str | Path) -> tuple[Path, Path]:
        """
        Resolve a user-facing base path.

        If path ends in .index:
            path itself is the FAISS file.
        Otherwise:
            <path>.index is used.

        Manifest is always:
            <index>.manifest.json
        """
        raw = Path(path)

        if raw.suffix.lower() == ".index":
            index_path = raw
        else:
            index_path = raw.with_suffix(raw.suffix + ".index") if raw.suffix else raw.with_name(
                raw.name + ".index"
            )

        manifest_path = index_path.with_name(
            index_path.name + ".manifest.json"
        )

        return index_path, manifest_path

    @staticmethod
    def _atomic_replace(source: Path, destination: Path) -> None:
        """
        Atomically replace ``destination`` with ``source``.

        The temporary file must be on the same filesystem as the destination.
        ``os.replace`` provides the required atomic replacement semantics on
        Windows and POSIX filesystems supported by Python.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(source), str(destination))

    @staticmethod
    def _temporary_path(
        directory: Path,
        *,
        prefix: str,
        suffix: str,
    ) -> Path:
        """
        Return a unique temporary pathname without leaving a file at that path.

        Important Windows/FAISS detail:
        ``tempfile.mkstemp()`` creates the file immediately.  FAISS should
        instead receive a path that does not already exist.  We therefore use
        ``mkstemp`` only to obtain a collision-resistant name, close its file
        descriptor, and remove the created zero-byte file before FAISS writes.
        """
        directory.mkdir(parents=True, exist_ok=True)

        fd, name = tempfile.mkstemp(
            prefix=prefix,
            suffix=suffix,
            dir=str(directory),
        )
        os.close(fd)

        temporary = Path(name)

        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

        return temporary

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """
        Atomically save the FAISS index and its validated JSON manifest.

        The persistence sequence is deliberately:

        1. Validate the in-memory index and build the manifest.
        2. Ask FAISS to write to a *non-existing* temporary pathname.
        3. Verify that FAISS produced a readable, non-empty index.
        4. Atomically replace the final ``.index`` file.
        5. Write and validate the manifest to a temporary pathname.
        6. Atomically replace the final ``.manifest.json`` file.

        This avoids the Windows failure mode caused by passing a pre-created
        zero-byte ``mkstemp`` file directly to ``faiss.write_index``.

        Returns:
            (index_path, manifest_path)

        Raises:
            IndexPersistenceError:
                If validation, FAISS persistence, manifest persistence, or
                read-back verification fails.
        """
        index_path, manifest_path = self._paths(path)

        if index_path.exists() or manifest_path.exists():
            logger.warning(
                "Overwriting existing FAISS persistence files: %s",
                index_path,
            )

        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Build/validate before mutating anything on disk.
        manifest = self._build_manifest(index_path.name)

        temp_index: Optional[Path] = None
        temp_manifest: Optional[Path] = None

        try:
            # ----------------------------------------------------------
            # 1. Create a unique pathname, not an existing file.
            # ----------------------------------------------------------
            temp_index = self._temporary_path(
                index_path.parent,
                prefix=f".{index_path.name}.",
                suffix=".tmp",
            )

            # ----------------------------------------------------------
            # 2. Let FAISS create the file itself.
            # ----------------------------------------------------------
            try:
                result = faiss.write_index(
                    self._index,
                    str(temp_index),
                )
            except Exception as exc:
                raise IndexPersistenceError(
                    f"FAISS failed to write temporary index: {temp_index}"
                ) from exc

            # Some FAISS builds return None while others may return a truthy
            # status. Treat an explicit False as failure, but rely primarily
            # on filesystem validation because that is the portable contract.
            if result is False:
                raise IndexPersistenceError(
                    f"FAISS reported failure while writing: {temp_index}"
                )

            # ----------------------------------------------------------
            # 3. Validate the generated index file.
            # ----------------------------------------------------------
            if (
                not temp_index.is_file()
                or temp_index.stat().st_size <= 0
            ):
                raise IndexPersistenceError(
                    "FAISS index write produced an empty or missing file: "
                    f"{temp_index}"
                )

            # Read the temporary file back before replacing the production
            # index. This catches truncated/corrupt persistence early.
            try:
                verification_index = faiss.read_index(
                    str(temp_index)
                )
            except Exception as exc:
                raise IndexPersistenceError(
                    "FAISS produced an index file that could not be "
                    f"read back for verification: {temp_index}"
                ) from exc

            if not isinstance(verification_index, faiss.IndexFlatIP):
                raise IndexPersistenceError(
                    "Persisted FAISS index has an unexpected type: "
                    f"{type(verification_index).__name__}; "
                    f"expected {self.INDEX_TYPE}."
                )

            if int(verification_index.d) != self.dimension:
                raise DimensionMismatchError(
                    "Persisted FAISS index dimension mismatch: "
                    f"expected={self.dimension}, "
                    f"actual={verification_index.d}."
                )

            if int(verification_index.ntotal) != self.ntotal:
                raise DocumentMappingError(
                    "Persisted FAISS vector-count mismatch: "
                    f"expected={self.ntotal}, "
                    f"actual={verification_index.ntotal}."
                )

            del verification_index

            # ----------------------------------------------------------
            # 4. Atomically publish the verified FAISS index.
            # ----------------------------------------------------------
            self._atomic_replace(
                temp_index,
                index_path,
            )
            temp_index = None

            # ----------------------------------------------------------
            # 5. Create a unique manifest pathname.
            # ----------------------------------------------------------
            temp_manifest = self._temporary_path(
                manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
            )

            # ----------------------------------------------------------
            # 6. Write + flush + validate the manifest.
            # ----------------------------------------------------------
            with temp_manifest.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    manifest.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            if (
                not temp_manifest.is_file()
                or temp_manifest.stat().st_size <= 0
            ):
                raise IndexPersistenceError(
                    "Manifest write produced an empty or missing file."
                )

            try:
                with temp_manifest.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    manifest_raw = json.load(handle)

                parsed_manifest = self._parse_manifest(manifest_raw)

            except Exception as exc:
                raise IndexPersistenceError(
                    "Generated FAISS manifest failed schema validation."
                ) from exc

            if parsed_manifest.index_file != index_path.name:
                raise IndexPersistenceError(
                    "Generated manifest/index filename mismatch: "
                    f"manifest={parsed_manifest.index_file!r}, "
                    f"actual={index_path.name!r}."
                )

            # ----------------------------------------------------------
            # 7. Atomically publish the manifest.
            # ----------------------------------------------------------
            self._atomic_replace(
                temp_manifest,
                manifest_path,
            )
            temp_manifest = None

            logger.info(
                "Saved FAISS index (%d vectors, %d dimensions) to %s",
                self.ntotal,
                self.dimension,
                index_path,
            )

            return index_path, manifest_path

        except IndexPersistenceError:
            raise

        except (FAISSIndexError, ValueError, TypeError):
            raise

        except Exception as exc:
            raise IndexPersistenceError(
                f"Failed to save FAISS index to {index_path}: {exc}"
            ) from exc

        finally:
            # Best-effort cleanup of temporary files if an operation failed
            # before atomic publication.
            for temporary in (
                temp_index,
                temp_manifest,
            ):
                if temporary is None:
                    continue

                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove temporary FAISS persistence file: %s",
                        temporary,
                    )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model_name: Optional[str] = None,
        expected_model_version: Optional[str] = None,
        expected_document_count: Optional[int] = None,
    ) -> "FAISSVectorIndex":
        """
        Safely restore an index from its FAISS file + JSON manifest.

        No pickle/deserialization of arbitrary Python objects is used.
        """
        index_path, manifest_path = cls._paths(path)

        if not index_path.is_file():
            raise IndexPersistenceError(
                f"FAISS index file does not exist: {index_path}"
            )

        if not manifest_path.is_file():
            raise IndexPersistenceError(
                f"FAISS manifest file does not exist: {manifest_path}"
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)

            manifest = cls._parse_manifest(raw)

            if manifest.index_file != index_path.name:
                raise IndexPersistenceError(
                    "Manifest/index filename mismatch: "
                    f"manifest={manifest.index_file!r}, "
                    f"actual={index_path.name!r}."
                )

            if expected_model_name is not None and (
                manifest.model_name != expected_model_name
            ):
                raise IndexPersistenceError(
                    "Model-name mismatch: "
                    f"expected={expected_model_name!r}, "
                    f"stored={manifest.model_name!r}."
                )

            if expected_model_version is not None and (
                manifest.model_version != expected_model_version
            ):
                raise IndexPersistenceError(
                    "Model-version mismatch: "
                    f"expected={expected_model_version!r}, "
                    f"stored={manifest.model_version!r}."
                )

            if (
                expected_document_count is not None
                and manifest.document_count != expected_document_count
            ):
                raise IndexPersistenceError(
                    "Document-count mismatch: "
                    f"expected={expected_document_count}, "
                    f"stored={manifest.document_count}."
                )

            loaded = faiss.read_index(str(index_path))

            if not isinstance(loaded, faiss.IndexFlatIP):
                raise IndexPersistenceError(
                    "Unsupported FAISS index type. This implementation expects "
                    f"{cls.INDEX_TYPE}; loaded {type(loaded).__name__}."
                )

            if loaded.d != manifest.embedding_dimension:
                raise DimensionMismatchError(
                    "Loaded FAISS dimension does not match manifest: "
                    f"FAISS={loaded.d}, manifest={manifest.embedding_dimension}."
                )

            if int(loaded.ntotal) != manifest.document_count:
                raise DocumentMappingError(
                    "Loaded FAISS vector count does not match manifest: "
                    f"FAISS={loaded.ntotal}, "
                    f"manifest={manifest.document_count}."
                )

            if len(manifest.document_ids) != int(loaded.ntotal):
                raise DocumentMappingError(
                    "Manifest document-ID mapping length does not match "
                    f"FAISS ntotal: {len(manifest.document_ids)} vs "
                    f"{loaded.ntotal}."
                )

            if manifest.metric != cls.METRIC:
                raise IndexPersistenceError(
                    f"Unsupported metric in manifest: {manifest.metric!r}."
                )

            expected_similarity = (
                cls.SIMILARITY if manifest.normalized else "inner_product"
            )
            if manifest.similarity != expected_similarity:
                raise IndexPersistenceError(
                    "Manifest similarity semantics are inconsistent with "
                    f"normalized={manifest.normalized!r}."
                )

            instance = cls(
                manifest.embedding_dimension,
                normalize=manifest.normalized,
                model_name=manifest.model_name,
                model_version=manifest.model_version,
                expected_document_count=expected_document_count,
            )

            instance._index = loaded
            instance._document_ids = list(manifest.document_ids)
            instance._document_id_set = set(manifest.document_ids)
            if len(instance._document_id_set) != len(instance._document_ids):
                raise DocumentMappingError(
                    "Loaded manifest contains duplicate document IDs."
                )
            instance._validate_alignment()

            logger.info(
                "Loaded FAISS index: %s | vectors=%d | dimension=%d",
                index_path,
                instance.ntotal,
                instance.dimension,
            )

            return instance

        except (FAISSIndexError, ValueError, TypeError):
            raise
        except Exception as exc:
            raise IndexPersistenceError(
                f"Failed to load FAISS index from {index_path}: {exc}"
            ) from exc

    @classmethod
    def _parse_manifest(cls, raw: Any) -> IndexManifest:
        if not isinstance(raw, dict):
            raise IndexPersistenceError(
                "Manifest root must be a JSON object."
            )

        required = {
            "schema_version",
            "index_type",
            "metric",
            "similarity",
            "normalized",
            "embedding_dimension",
            "document_count",
            "document_ids",
            "model_name",
            "model_version",
            "created_at",
            "index_file",
        }

        missing = sorted(required.difference(raw))
        if missing:
            raise IndexPersistenceError(
                f"Manifest is missing required fields: {missing}"
            )

        if raw["schema_version"] != cls.SCHEMA_VERSION:
            raise IndexPersistenceError(
                "Unsupported manifest schema version: "
                f"{raw['schema_version']!r}."
            )

        if raw["index_type"] != cls.INDEX_TYPE:
            raise IndexPersistenceError(
                f"Unsupported index_type: {raw['index_type']!r}."
            )

        if raw["metric"] != cls.METRIC:
            raise IndexPersistenceError(
                f"Unsupported metric: {raw['metric']!r}."
            )

        if not isinstance(raw["normalized"], bool):
            raise IndexPersistenceError(
                "Manifest 'normalized' must be boolean."
            )

        dimension = cls._validate_dimension(raw["embedding_dimension"])
        document_count = cls._validate_non_negative_int(
            raw["document_count"],
            "document_count",
        )

        raw_ids = raw["document_ids"]
        if not isinstance(raw_ids, list):
            raise DocumentMappingError(
                "Manifest 'document_ids' must be a JSON array."
            )

        ids = cls._validate_document_ids(
            raw_ids,
            expected_count=document_count,
        )

        model_name = raw["model_name"]
        model_version = raw["model_version"]

        if model_name is not None and (
            not isinstance(model_name, str) or not model_name.strip()
        ):
            raise IndexPersistenceError(
                "Manifest 'model_name' must be null or a non-empty string."
            )

        if model_version is not None and (
            not isinstance(model_version, str) or not model_version.strip()
        ):
            raise IndexPersistenceError(
                "Manifest 'model_version' must be null or a non-empty string."
            )

        if not isinstance(raw["created_at"], str) or not raw["created_at"]:
            raise IndexPersistenceError(
                "Manifest 'created_at' must be a non-empty string."
            )

        if not isinstance(raw["index_file"], str) or not raw["index_file"]:
            raise IndexPersistenceError(
                "Manifest 'index_file' must be a non-empty string."
            )

        expected_similarity = (
            cls.SIMILARITY if raw["normalized"] else "inner_product"
        )

        if raw["similarity"] != expected_similarity:
            raise IndexPersistenceError(
                "Manifest similarity is inconsistent with normalization."
            )

        return IndexManifest(
            schema_version=raw["schema_version"],
            index_type=raw["index_type"],
            metric=raw["metric"],
            similarity=raw["similarity"],
            normalized=raw["normalized"],
            embedding_dimension=dimension,
            document_count=document_count,
            document_ids=tuple(ids),
            model_name=model_name.strip() if model_name else None,
            model_version=model_version.strip() if model_version else None,
            created_at=raw["created_at"],
            index_file=raw["index_file"],
        )

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def manifest(self) -> IndexManifest:
        """Return the current in-memory manifest without writing files."""
        return self._build_manifest("")

    def validate(self) -> None:
        """
        Validate internal FAISS/mapping consistency.

        Raises instead of returning False so callers cannot accidentally
        ignore a broken index.
        """
        self._validate_alignment()

        if self._index.d != self._dimension:
            raise DimensionMismatchError(
                f"FAISS dimension={self._index.d} does not equal "
                f"configured dimension={self._dimension}."
            )

        if not isinstance(self._index, faiss.IndexFlatIP):
            raise FAISSIndexError(
                f"Expected {self.INDEX_TYPE}, "
                f"got {type(self._index).__name__}."
            )

        if (
            self._expected_document_count is not None
            and self.ntotal != self._expected_document_count
        ):
            raise FAISSIndexError(
                "Expected document count mismatch: "
                f"expected={self._expected_document_count}, "
                f"actual={self.ntotal}."
            )

    def __len__(self) -> int:
        return self.ntotal


# ---------------------------------------------------------------------------
# Model-free helper tests
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """
    Small deterministic test suite.

    Requires FAISS but no SPECTER2 model and no external dataset.
    """
    rng = np.random.default_rng(42)
    dimension = 16

    index = FAISSVectorIndex(
        dimension=dimension,
        normalize=True,
        model_name="test-model",
        model_version="test-1",
    )

    assert index.ntotal == 0

    vectors = rng.normal(size=(5, dimension)).astype(np.float32)
    ids = [f"paper-{i}" for i in range(5)]

    original_vectors = vectors.copy()
    index.add(vectors, ids)

    assert index.ntotal == 5
    assert index.document_count == 5
    assert index.document_ids == tuple(ids)
    # add(normalize=True) must not mutate caller-owned embeddings.
    assert np.array_equal(vectors, original_vectors)
    assert len(index._document_id_set) == index.document_count

    # Search with an exact indexed vector: it must rank itself first.
    results = index.search(vectors[2], top_k=3)

    assert len(results) == 3
    assert results[0].document_id == "paper-2"
    assert results[0].index_position == 2
    assert results[0].score > 0.9999

    # Scores must be descending.
    assert all(
        results[i].score >= results[i + 1].score
        for i in range(len(results) - 1)
    )

    # Self exclusion.
    excluded = index.search(
        vectors[2],
        top_k=3,
        exclude_ids={"paper-2"},
    )
    assert excluded
    assert all(result.document_id != "paper-2" for result in excluded)

    # Adaptive exclusion path: exclude the top three candidates and still
    # return the requested number of results without scanning all vectors.
    excluded_many = index.search(
        vectors[2],
        top_k=2,
        exclude_ids={"paper-2", "paper-0", "paper-1"},
    )
    assert len(excluded_many) == 2
    assert all(
        result.document_id not in {"paper-2", "paper-0", "paper-1"}
        for result in excluded_many
    )

    # Invalid duplicate ID.
    try:
        index.add(
            np.ones((1, dimension), dtype=np.float32),
            ["paper-2"],
        )
    except DocumentMappingError:
        pass
    else:
        raise AssertionError("Duplicate document ID was not rejected.")

    # Wrong dimension.
    try:
        index.add(
            np.ones((1, dimension + 1), dtype=np.float32),
            ["paper-new"],
        )
    except DimensionMismatchError:
        pass
    else:
        raise AssertionError("Wrong embedding dimension was not rejected.")

    # NaN.
    bad_nan = np.ones((1, dimension), dtype=np.float32)
    bad_nan[0, 0] = np.nan

    try:
        index.add(bad_nan, ["paper-new"])
    except InvalidEmbeddingError:
        pass
    else:
        raise AssertionError("NaN embedding was not rejected.")

    # Inf.
    bad_inf = np.ones((1, dimension), dtype=np.float32)
    bad_inf[0, 0] = np.inf

    try:
        index.add(bad_inf, ["paper-new"])
    except InvalidEmbeddingError:
        pass
    else:
        raise AssertionError("Inf embedding was not rejected.")

    # Zero vector under normalization.
    try:
        index.add(
            np.zeros((1, dimension), dtype=np.float32),
            ["paper-new"],
        )
    except InvalidEmbeddingError:
        pass
    else:
        raise AssertionError("Zero vector was not rejected.")

    # Search empty index.
    empty = FAISSVectorIndex(dimension=dimension)
    assert empty.search(vectors[0], top_k=10) == []

    # top_k > ntotal clamps to ntotal.
    assert len(index.search(vectors[0], top_k=100)) == 5

    # Persistence.
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "research"

        index_path, manifest_path = index.save(base)

        assert index_path.is_file()
        assert manifest_path.is_file()
        assert index_path.stat().st_size > 0
        assert manifest_path.stat().st_size > 0

        # Regression check: the published index must be readable immediately
        # after save(). This specifically protects the Windows tempfile path
        # used by save().
        persisted = faiss.read_index(str(index_path))
        assert persisted.ntotal == index.ntotal
        assert persisted.d == index.dimension
        assert isinstance(persisted, faiss.IndexFlatIP)
        del persisted

        loaded = FAISSVectorIndex.load(
            base,
            expected_model_name="test-model",
            expected_model_version="test-1",
        )

        loaded.validate()

        assert loaded.ntotal == index.ntotal
        assert loaded.dimension == index.dimension
        assert loaded.document_ids == index.document_ids
        assert loaded.normalize is True

        loaded_results = loaded.search(vectors[2], top_k=3)
        assert loaded_results[0].document_id == "paper-2"

    print("FAISSVectorIndex self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()