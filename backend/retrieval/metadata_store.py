"""
Production metadata repository for the AI Research Paper Assistant.

Responsibilities
----------------
This module is the authoritative bridge between opaque research-paper IDs
returned by retrieval and the semantic metadata required by reranking.

Source of truth:
    indexes/research/metadata.json

Expected schema:
    {
        "schema_version": "1",
        "document_count": N,
        "documents": [
            {
                "document_id": "...",
                "title": "...",
                "summary": "..."
            }
        ]
    }

Design boundaries
-----------------
This module intentionally does NOT:
- create embeddings;
- query FAISS;
- perform reranking;
- perform final ranking;
- read the 287k-row source CSV;
- make network requests;
- mutate the metadata file.

The store is:
- lazy-loaded;
- thread-safe;
- fail-closed on schema corruption;
- deterministic;
- cached in-process;
- able to detect atomic sidecar replacement via mtime/size;
- compatible with the retriever -> reranker metadata bridge.

Important memory note
---------------------
The current metadata sidecar contains one JSON object containing all research
metadata. Once loaded, records are kept in an in-memory dictionary for fast
ID lookup. This is intentional for the current architecture because repeated
retrieval requests must not parse the ~400 MB JSON file on every query.

For a future much larger corpus, an SQLite/LMDB/Parquet-backed store can replace
this implementation behind the same public interface.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RESEARCH_METADATA_PATH",
    "EXPECTED_SCHEMA_VERSION",
    "MetadataStoreError",
    "MetadataStoreNotFoundError",
    "MetadataStoreSchemaError",
    "MetadataNotFoundError",
    "PaperMetadata",
    "ResearchMetadataStore",
    "run_self_test",
]

DEFAULT_RESEARCH_METADATA_PATH = (
    Path("indexes") / "research" / "metadata.json"
)
EXPECTED_SCHEMA_VERSION = "1"

# Defensive upper bound for the JSON sidecar itself. The current corpus is
# intentionally large (~400 MB), so the default is deliberately generous.
# This is a safety rail against accidentally pointing the store at an
# unrelated huge file.
DEFAULT_MAX_METADATA_FILE_MB = 1024

# A replaced file can briefly be observed while another process is writing it.
# We retry only when the filesystem signature changes during parsing.
DEFAULT_LOAD_RETRIES = 2
DEFAULT_LOAD_RETRY_DELAY_SECONDS = 0.05


class MetadataStoreError(RuntimeError):
    """Base metadata repository failure."""


class MetadataStoreNotFoundError(MetadataStoreError, FileNotFoundError):
    """Metadata sidecar does not exist."""


class MetadataStoreSchemaError(MetadataStoreError, ValueError):
    """Metadata sidecar violates the expected schema."""


class MetadataNotFoundError(MetadataStoreError, KeyError):
    """Requested document ID has no metadata record."""


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Validated research-paper metadata required by retrieval/reranking."""

    document_id: str
    title: str
    summary: str

    def as_reranker_candidate(
        self,
        *,
        retrieval_score: float,
        index_position: int,
    ) -> dict[str, Any]:
        """
        Build the canonical metadata-enriched candidate contract.

        The values are copied into a plain dictionary deliberately: downstream
        retrieval/reranking components should not depend on this repository's
        dataclass implementation.
        """
        score = _validate_finite_float(
            retrieval_score,
            field="retrieval_score",
        )
        position = _validate_non_negative_int(
            index_position,
            field="index_position",
        )

        return {
            "document_id": self.document_id,
            "title": self.title,
            "summary": self.summary,
            "retrieval_score": score,
            "index_position": position,
        }


class ResearchMetadataStore:
    """
    Thread-safe, process-local repository for research-paper metadata.

    The metadata file is loaded lazily. A cached load is reused until the
    sidecar's filesystem signature changes. This prevents repeated JSON parsing
    during high-frequency search requests while still allowing an atomically
    replaced metadata file to be picked up without restarting the process.

    Parameters
    ----------
    path:
        Metadata sidecar path.
    expected_document_count:
        Optional expected count, normally obtained from the validated FAISS
        index. When strict_count=True, mismatch is a hard failure.
    strict_count:
        Whether expected_document_count is authoritative.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_RESEARCH_METADATA_PATH,
        *,
        expected_document_count: int | None = None,
        strict_count: bool = True,
        max_file_size_mb: int = DEFAULT_MAX_METADATA_FILE_MB,
        load_retries: int = DEFAULT_LOAD_RETRIES,
        load_retry_delay_seconds: float = DEFAULT_LOAD_RETRY_DELAY_SECONDS,
    ) -> None:
        self._path = Path(path)
        self._max_file_size_bytes = _validate_positive_int(
            max_file_size_mb,
            field="max_file_size_mb",
        ) * 1024 * 1024
        self._load_retries = _validate_non_negative_int(
            load_retries,
            field="load_retries",
        )
        self._load_retry_delay_seconds = _validate_finite_non_negative_float(
            load_retry_delay_seconds,
            field="load_retry_delay_seconds",
        )
        self._expected_document_count = (
            _validate_optional_non_negative_int(
                expected_document_count,
                field="expected_document_count",
            )
        )
        self._strict_count = bool(strict_count)

        self._lock = threading.RLock()
        self._records: dict[str, PaperMetadata] | None = None
        self._document_count: int | None = None
        self._file_signature: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Configured metadata sidecar path."""
        return self._path

    @property
    def document_count(self) -> int:
        """Number of validated metadata records."""
        self._ensure_loaded()
        assert self._document_count is not None
        return self._document_count

    @property
    def loaded(self) -> bool:
        """Whether metadata is currently resident in the process cache."""
        with self._lock:
            return self._records is not None

    @property
    def expected_document_count(self) -> int | None:
        """Configured expected document count, if any."""
        return self._expected_document_count

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """
        Invalidate the local cache.

        The next operation reloads and validates the sidecar.
        """
        with self._lock:
            self._records = None
            self._document_count = None
            self._file_signature = None

    def clear_cache(self) -> None:
        """Alias for reload(), useful for service lifecycle management."""
        self.reload()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> PaperMetadata:
        """Resolve one stable research-paper ID to validated metadata."""
        normalized_id = self._validate_document_id(document_id)
        self._ensure_loaded()

        assert self._records is not None

        record = self._records.get(normalized_id)
        if record is None:
            raise MetadataNotFoundError(
                "Research metadata not found for "
                f"document_id={normalized_id!r}."
            )

        return record

    def get_many(
        self,
        document_ids: Iterable[str],
    ) -> dict[str, PaperMetadata]:
        """
        Resolve multiple IDs.

        The returned dictionary preserves the first occurrence order supplied
        by the caller. Duplicate requested IDs are collapsed naturally.
        Missing IDs cause a fail-closed MetadataNotFoundError.
        """
        ids = self._normalize_ids(document_ids)
        if not ids:
            return {}

        self._ensure_loaded()
        assert self._records is not None

        missing = [
            document_id
            for document_id in ids
            if document_id not in self._records
        ]

        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            raise MetadataNotFoundError(
                "Research metadata missing for retrieved document IDs: "
                f"{preview!r}{suffix}"
            )

        return {
            document_id: self._records[document_id]
            for document_id in ids
        }

    def resolve(self, document_id: str) -> Mapping[str, Any]:
        """
        Resolve one document ID to the Mapping contract expected by
        DenseRetriever.metadata_resolver.
        """
        record = self.get(document_id)
        return {
            "document_id": record.document_id,
            "title": record.title,
            "summary": record.summary,
        }

    def contains(self, document_id: str) -> bool:
        """Return whether a validated document ID exists."""
        normalized_id = self._validate_document_id(document_id)
        self._ensure_loaded()

        assert self._records is not None
        return normalized_id in self._records

    def validate_against_ids(
        self,
        document_ids: Iterable[str],
    ) -> None:
        """
        Fail closed if any supplied retrieval IDs cannot be hydrated.

        This should be called by the retrieval integration boundary when
        strict metadata integrity is required.
        """
        ids = self._normalize_ids(document_ids)
        if not ids:
            return

        self._ensure_loaded()
        assert self._records is not None

        missing = [
            document_id
            for document_id in ids
            if document_id not in self._records
        ]

        if missing:
            raise MetadataNotFoundError(
                "FAISS returned IDs that cannot be hydrated from research "
                f"metadata: {missing[:10]!r}"
            )

    def hydrate_candidate(
        self,
        *,
        document_id: str,
        retrieval_score: float,
        index_position: int,
    ) -> dict[str, Any]:
        """
        Resolve one retrieval hit into the canonical reranker candidate.
        """
        metadata = self.get(document_id)
        return metadata.as_reranker_candidate(
            retrieval_score=retrieval_score,
            index_position=index_position,
        )

    def hydrate_candidates(
        self,
        candidates: Iterable[Any],
    ) -> list[dict[str, Any]]:
        """
        Hydrate retrieval result objects into reranker-ready dictionaries.

        Supported candidate attributes/keys:
            document_id
            score OR retrieval_score
            index_position

        This deliberately supports both object-style and mapping-style
        retrieval results so the metadata layer remains decoupled from the
        retriever implementation.
        """
        candidate_list = list(candidates)
        if not candidate_list:
            return []

        ids: list[str] = []
        for position, candidate in enumerate(candidate_list):
            document_id = _candidate_value(
                candidate,
                "document_id",
                position=position,
            )
            ids.append(self._validate_document_id(document_id))

        metadata_map = self.get_many(ids)

        hydrated: list[dict[str, Any]] = []

        for position, candidate in enumerate(candidate_list):
            document_id = ids[position]

            raw_score = _candidate_value_or_missing(
                candidate,
                "retrieval_score",
            )
            if raw_score is _MISSING:
                raw_score = _candidate_value(
                    candidate,
                    "score",
                    position=position,
                )

            raw_index_position = _candidate_value(
                candidate,
                "index_position",
                default=position,
                position=position,
            )

            metadata = metadata_map[document_id]

            hydrated.append(
                metadata.as_reranker_candidate(
                    retrieval_score=_validate_finite_float(
                        raw_score,
                        field=f"candidate[{position}].retrieval_score",
                    ),
                    index_position=_validate_non_negative_int(
                        raw_index_position,
                        field=f"candidate[{position}].index_position",
                    ),
                )
            )

        return hydrated

    # ------------------------------------------------------------------
    # Validation / loading
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Force a complete metadata validation.

        Useful during application startup or index deployment checks.
        """
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._records is not None:
                current_signature = self._stat_signature(
                    allow_missing=True,
                )

                if current_signature == self._file_signature:
                    return

                logger.info(
                    "Research metadata sidecar changed; reloading: %s",
                    self._path,
                )

            self._load_locked()

    def _load_locked(self) -> None:
        """
        Load, validate, and atomically publish the metadata cache.

        The important invariant is that a file is never cached merely because
        JSON parsing succeeded: its filesystem identity is checked before and
        after parsing. This protects the service from observing a partially
        replaced sidecar.
        """
        attempts = self._load_retries + 1

        last_error: Exception | None = None

        for attempt in range(attempts):
            before = self._stat_signature()
            self._validate_file_size(before)

            try:
                with self._path.open("r", encoding="utf-8-sig", newline="") as handle:
                    payload = json.load(
                        handle,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
            except json.JSONDecodeError as exc:
                last_error = MetadataStoreSchemaError(
                    f"Invalid JSON in research metadata sidecar "
                    f"{self._path}: {exc}"
                )
            except UnicodeDecodeError as exc:
                last_error = MetadataStoreSchemaError(
                    f"Research metadata sidecar is not valid UTF-8: {self._path}"
                )
            except OSError as exc:
                last_error = MetadataStoreNotFoundError(
                    f"Could not read research metadata sidecar "
                    f"{self._path}: {exc}"
                )

            if last_error is not None:
                # A JSON error during an active atomic replacement may be
                # transient. Retry only if the file identity actually changed.
                if attempt + 1 < attempts:
                    after = self._stat_signature(allow_missing=True)
                    if after != before:
                        time.sleep(self._load_retry_delay_seconds)
                        last_error = None
                        continue
                raise last_error

            records, declared_count = self._parse_payload(payload)

            after = self._stat_signature()
            if after != before:
                if attempt + 1 < attempts:
                    logger.warning(
                        "Research metadata changed while loading; retrying: %s",
                        self._path,
                    )
                    time.sleep(self._load_retry_delay_seconds)
                    continue

                raise MetadataStoreSchemaError(
                    "Research metadata sidecar changed during validation; "
                    "refusing to cache a potentially inconsistent snapshot: "
                    f"{self._path}"
                )

            # Publish the complete validated snapshot only after every check
            # succeeds. Existing cache remains untouched if anything fails.
            self._records = records
            self._document_count = declared_count
            self._file_signature = after

            logger.info(
                "Loaded research metadata: path=%s documents=%d",
                self._path,
                declared_count,
            )
            return

        raise MetadataStoreSchemaError(
            f"Could not obtain a stable metadata snapshot: {self._path}"
        )

    def _parse_payload(
        self,
        payload: Any,
    ) -> tuple[dict[str, PaperMetadata], int]:
        if not isinstance(payload, Mapping):
            raise MetadataStoreSchemaError(
                "Research metadata root must be a JSON object."
            )

        schema_version = payload.get("schema_version")
        if schema_version != EXPECTED_SCHEMA_VERSION:
            raise MetadataStoreSchemaError(
                "Unsupported research metadata schema version: "
                f"{schema_version!r}; expected "
                f"{EXPECTED_SCHEMA_VERSION!r}."
            )

        raw_count = payload.get("document_count")

        declared_count = _validate_non_negative_int(
            raw_count,
            field="document_count",
        )

        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise MetadataStoreSchemaError(
                "Research metadata must contain a 'documents' array."
            )

        if len(raw_documents) != declared_count:
            raise MetadataStoreSchemaError(
                "Research metadata document_count does not match "
                "documents length: "
                f"declared={declared_count}, "
                f"actual={len(raw_documents)}."
            )

        if (
            self._expected_document_count is not None
            and self._strict_count
            and declared_count != self._expected_document_count
        ):
            raise MetadataStoreSchemaError(
                "Research metadata count does not match the expected "
                "FAISS document count: "
                f"metadata={declared_count}, "
                f"expected={self._expected_document_count}."
            )

        records: dict[str, PaperMetadata] = {}

        for position, raw in enumerate(raw_documents):
            if not isinstance(raw, Mapping):
                raise MetadataStoreSchemaError(
                    f"Research metadata record {position} "
                    "must be a JSON object."
                )

            document_id = self._clean_required_text(
                raw.get("document_id"),
                "document_id",
                position,
            )
            title = self._clean_required_text(
                raw.get("title"),
                "title",
                position,
            )
            summary = self._clean_required_text(
                raw.get("summary"),
                "summary",
                position,
            )

            if document_id in records:
                raise MetadataStoreSchemaError(
                    "Duplicate research metadata document_id: "
                    f"{document_id!r}."
                )

            records[document_id] = PaperMetadata(
                document_id=document_id,
                title=title,
                summary=summary,
            )

        if len(records) != declared_count:
            # Defensive postcondition; normally guaranteed by duplicate
            # detection above.
            raise MetadataStoreSchemaError(
                "Validated metadata record count mismatch: "
                f"records={len(records)}, declared={declared_count}."
            )

        return records, declared_count

    def _stat_signature(
        self,
        *,
        allow_missing: bool = False,
    ) -> tuple[int, int, int] | None:
        try:
            stat = self._path.stat()
        except OSError:
            if allow_missing:
                return None
            raise MetadataStoreNotFoundError(
                f"Research metadata sidecar not found: {self._path}"
            )

        return (
            int(stat.st_mtime_ns),
            int(stat.st_size),
            int(getattr(stat, "st_ino", 0)),
        )

    def _validate_file_size(
        self,
        signature: tuple[int, int, int] | None,
    ) -> None:
        if signature is None:
            raise MetadataStoreNotFoundError(
                f"Research metadata sidecar not found: {self._path}"
            )

        size_bytes = signature[1]
        if size_bytes > self._max_file_size_bytes:
            max_mb = self._max_file_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise MetadataStoreSchemaError(
                "Research metadata sidecar exceeds the configured size limit: "
                f"size={actual_mb:.2f} MB, limit={max_mb:.2f} MB, "
                f"path={self._path}"
            )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_required_text(
        value: Any,
        field: str,
        position: int,
    ) -> str:
        if not isinstance(value, str):
            raise MetadataStoreSchemaError(
                f"Metadata record {position} field "
                f"{field!r} must be a string."
            )

        cleaned = value.strip()

        if not cleaned:
            raise MetadataStoreSchemaError(
                f"Metadata record {position} field "
                f"{field!r} cannot be empty."
            )

        return cleaned

    @staticmethod
    def _validate_document_id(
        document_id: str,
    ) -> str:
        if not isinstance(document_id, str):
            raise MetadataStoreSchemaError(
                "document_id must be a string."
            )

        value = document_id.strip()

        if not value:
            raise MetadataStoreSchemaError(
                "document_id cannot be empty."
            )

        return value

    @classmethod
    def _normalize_ids(
        cls,
        document_ids: Iterable[str],
    ) -> list[str]:
        try:
            values = list(document_ids)
        except TypeError as exc:
            raise MetadataStoreSchemaError(
                "document_ids must be iterable."
            ) from exc

        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            document_id = cls._validate_document_id(value)

            # Preserve first occurrence order but avoid duplicate lookups.
            if document_id not in seen:
                normalized.append(document_id)
                seen.add(document_id)

        return normalized


def _validate_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MetadataStoreSchemaError(
            f"{field} must be a positive integer."
        )
    return value


def _validate_finite_non_negative_float(
    value: Any,
    *,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise MetadataStoreSchemaError(
            f"{field} must be a finite non-negative number, not bool."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetadataStoreSchemaError(
            f"{field} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise MetadataStoreSchemaError(
            f"{field} must be a finite non-negative number."
        )
    return result


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently taking the last."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataStoreSchemaError(
                f"Duplicate JSON object key encountered: {key!r}."
            )
        result[key] = value
    return result


# ----------------------------------------------------------------------
# Generic candidate helpers
# ----------------------------------------------------------------------


_MISSING = object()


def _candidate_value_or_missing(
    candidate: Any,
    field: str,
) -> Any:
    """Read a candidate field and return _MISSING when it is absent."""
    if isinstance(candidate, Mapping):
        return candidate.get(field, _MISSING)
    return getattr(candidate, field, _MISSING)


def _candidate_value(
    candidate: Any,
    field: str,
    *,
    default: Any = _MISSING,
    position: int | None = None,
) -> Any:
    """Read a field from either a mapping or an object."""
    if isinstance(candidate, Mapping):
        value = candidate.get(field, _MISSING)
    else:
        value = getattr(candidate, field, _MISSING)

    if value is _MISSING:
        if default is not _MISSING:
            return default

        location = (
            f"candidate[{position}]"
            if position is not None
            else "candidate"
        )

        raise MetadataStoreSchemaError(
            f"{location} is missing required field {field!r}."
        )

    return value


def _validate_finite_float(
    value: Any,
    *,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise MetadataStoreSchemaError(
            f"{field} must be a finite real number, not bool."
        )

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetadataStoreSchemaError(
            f"{field} must be a finite real number."
        ) from exc

    if not math.isfinite(result):
        raise MetadataStoreSchemaError(
            f"{field} must be finite; got {value!r}."
        )

    return result


def _validate_non_negative_int(
    value: Any,
    *,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetadataStoreSchemaError(
            f"{field} must be a non-negative integer."
        )

    if value < 0:
        raise MetadataStoreSchemaError(
            f"{field} must be a non-negative integer."
        )

    return value


def _validate_optional_non_negative_int(
    value: Any,
    *,
    field: str,
) -> int | None:
    if value is None:
        return None

    return _validate_non_negative_int(
        value,
        field=field,
    )


# ----------------------------------------------------------------------
# Model-free self-test
# ----------------------------------------------------------------------


def run_self_test() -> None:
    """Run deterministic metadata-store tests without external dependencies."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "metadata.json"

        payload = {
            "schema_version": "1",
            "document_count": 3,
            "documents": [
                {
                    "document_id": "p1",
                    "title": "Paper One",
                    "summary": "Alpha",
                },
                {
                    "document_id": "p2",
                    "title": "Paper Two",
                    "summary": "Beta",
                },
                {
                    "document_id": "p3",
                    "title": "Paper Three",
                    "summary": "Gamma",
                },
            ],
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        store = ResearchMetadataStore(
            path,
            expected_document_count=3,
        )

        assert not store.loaded

        paper = store.get("p1")
        assert store.loaded
        assert store.document_count == 3
        assert paper.document_id == "p1"
        assert paper.title == "Paper One"
        assert paper.summary == "Alpha"

        resolved = store.resolve("p1")
        assert resolved == {
            "document_id": "p1",
            "title": "Paper One",
            "summary": "Alpha",
        }

        many = store.get_many(["p2", "p3", "p2"])
        assert list(many) == ["p2", "p3"]
        assert many["p2"].title == "Paper Two"

        assert store.contains("p1")
        assert not store.contains("does-not-exist")

        candidate = store.hydrate_candidate(
            document_id="p2",
            retrieval_score=0.91,
            index_position=7,
        )

        assert candidate == {
            "document_id": "p2",
            "title": "Paper Two",
            "summary": "Beta",
            "retrieval_score": 0.91,
            "index_position": 7,
        }

        hydrated = store.hydrate_candidates(
            [
                {
                    "document_id": "p1",
                    "score": 0.95,
                    "index_position": 0,
                },
                {
                    "document_id": "p3",
                    "retrieval_score": 0.82,
                    "index_position": 2,
                },
            ]
        )

        assert [item["document_id"] for item in hydrated] == [
            "p1",
            "p3",
        ]
        assert hydrated[0]["title"] == "Paper One"
        assert hydrated[1]["summary"] == "Gamma"

        store.validate_against_ids(["p1", "p2", "p3"])

        try:
            store.get("missing")
        except MetadataNotFoundError:
            pass
        else:
            raise AssertionError(
                "Missing document ID must raise MetadataNotFoundError."
            )

        # Schema mismatch must fail closed.
        bad_path = root / "bad.json"
        bad_path.write_text(
            json.dumps(
                {
                    "schema_version": "999",
                    "document_count": 0,
                    "documents": [],
                }
            ),
            encoding="utf-8",
        )

        try:
            ResearchMetadataStore(bad_path).validate()
        except MetadataStoreSchemaError:
            pass
        else:
            raise AssertionError(
                "Unsupported schema version must fail validation."
            )

        # Duplicate IDs must fail closed.
        duplicate_path = root / "duplicate.json"
        duplicate_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "document_count": 2,
                    "documents": [
                        {
                            "document_id": "dup",
                            "title": "A",
                            "summary": "A",
                        },
                        {
                            "document_id": "dup",
                            "title": "B",
                            "summary": "B",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        try:
            ResearchMetadataStore(duplicate_path).validate()
        except MetadataStoreSchemaError:
            pass
        else:
            raise AssertionError(
                "Duplicate document IDs must fail validation."
            )

        # Duplicate JSON keys must fail closed (json.load normally keeps
        # only the final value, which is unsafe for a source-of-truth file).
        duplicate_key_path = root / "duplicate-key.json"
        duplicate_key_path.write_text(
            '{"schema_version":"1","schema_version":"999",'
            '"document_count":0,"documents":[]}',
            encoding="utf-8",
        )
        try:
            ResearchMetadataStore(duplicate_key_path).validate()
        except MetadataStoreSchemaError:
            pass
        else:
            raise AssertionError(
                "Duplicate JSON keys must fail validation."
            )

        # Empty batch operations should be cheap and must not require the
        # metadata file to exist.
        empty_store = ResearchMetadataStore(root / "does-not-exist.json")
        assert empty_store.get_many([]) == {}
        assert empty_store.hydrate_candidates([]) == []
        empty_store.validate_against_ids([])

        # Atomic replacement detection: replacing a valid sidecar must be
        # visible without explicitly calling reload().
        replacement = {
            "schema_version": "1",
            "document_count": 3,
            "documents": [
                {
                    "document_id": "replacement",
                    "title": "Replacement",
                    "summary": "New snapshot",
                },
                {
                    "document_id": "p2",
                    "title": "Paper Two Updated",
                    "summary": "Beta Updated",
                },
                {
                    "document_id": "p3",
                    "title": "Paper Three Updated",
                    "summary": "Gamma Updated",
                },
            ],
        }
        path.write_text(
            json.dumps(replacement, ensure_ascii=False),
            encoding="utf-8",
        )
        assert store.get("replacement").title == "Replacement"
        assert store.document_count == 3

        # File-size guard.
        tiny_limit_store = ResearchMetadataStore(
            path,
            max_file_size_mb=1,
        )
        tiny_limit_store.validate()

    print("ResearchMetadataStore self-test: PASSED")


if __name__ == "__main__":
    run_self_test()