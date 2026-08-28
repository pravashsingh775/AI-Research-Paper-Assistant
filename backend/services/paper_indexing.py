"""
Offline research-corpus FAISS index builder.

AI Research Paper Assistant
============================

Mode 1:
    cleaned research corpus
        ↓
    canonical title + abstract text
        ↓
    SPECTER2 document embeddings
        ↓
    FAISS IndexFlatIP / cosine similarity
        ↓
    indexes/research/index.index
    indexes/research/index.index.manifest.json

Responsibilities
----------------
This module ONLY builds the persistent Mode-1 research index.

It does NOT:
    - serve HTTP requests
    - perform query retrieval
    - rerank papers
    - perform final ranking
    - call an LLM
    - analyze papers
    - process PDFs
    - implement SPECTER2 internals
    - implement FAISS internals

The existing embedding and FAISS implementations remain the source of
truth for those responsibilities.

Python compatibility
--------------------
The project currently runs on Python 3.10, therefore this module avoids
Python-3.11-only typing features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import re
import shutil
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from backend.retrieval.faiss_index import FAISSVectorIndex


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResearchIndexingError(RuntimeError):
    """Base exception for research-index building failures."""


class ResearchDatasetError(ResearchIndexingError):
    """Raised when the research corpus is invalid or unusable."""


class ResearchDatasetSchemaError(ResearchDatasetError):
    """Raised when required dataset columns cannot be identified."""


class ResearchIndexValidationError(ResearchIndexingError):
    """Raised when index-building invariants are violated."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchIndexConfig:
    """
    Configuration for the offline research index builder.

    The defaults match the current project layout.
    """

    dataset_path: Path = (
        Path("data")
        / "processed"
        / "cleaned_arxiv_dataset.csv"
    )

    index_base: Path = (
        Path("indexes")
        / "research"
        / "index"
    )

    # Number of CSV rows processed per streaming batch.
    batch_size: int = 256

    # Metadata sidecar written alongside the FAISS index.
    metadata_path: Optional[Path] = (
        Path("indexes")
        / "research"
        / "metadata.json"
    )

    # Small smoke-test limit.
    # None means full corpus.
    limit: Optional[int] = None

    # Column overrides. If None, safe aliases are auto-detected.
    id_column: Optional[str] = None
    title_column: Optional[str] = None
    abstract_column: Optional[str] = None

    # Refuse to build a final production index when any paper has no
    # usable semantic text.
    strict_text_validation: bool = True

    # Existing final artifacts are not overwritten unless explicitly
    # requested by the caller.
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchColumnMapping:
    """Resolved dataset-column mapping."""

    document_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class ResearchRecord:
    """One validated research-paper record."""

    document_id: str
    title: str
    abstract: str

    @property
    def canonical_text(self) -> str:
        """
        Build the semantic document representation.

        SPECTER2 receives the scientific paper title and abstract.
        """
        return (
            f"Title:\n{self.title}\n\n"
            f"Abstract:\n{self.abstract}"
        )


# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------


_ID_ALIASES = (
    "id",
    "paper_id",
    "document_id",
    "arxiv_id",
    "identifier",
    "paperid",
)

_TITLE_ALIASES = (
    "title",
    "paper_title",
    "document_title",
)

_ABSTRACT_ALIASES = (
    "abstract",
    "summary",
    "paper_abstract",
    "description",
)


def _normalise_column_name(value: str) -> str:
    """Normalize a column name for alias matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _resolve_column(
    columns: Sequence[str],
    explicit: Optional[str],
    aliases: Sequence[str],
    field_name: str,
) -> str:
    """Resolve one required dataset column safely."""

    if explicit is not None:
        if explicit not in columns:
            raise ResearchDatasetSchemaError(
                f"Configured {field_name} column {explicit!r} "
                f"does not exist. Available columns: {list(columns)!r}"
            )
        return explicit

    normalized = {
        _normalise_column_name(column): column
        for column in columns
    }

    for alias in aliases:
        resolved = normalized.get(_normalise_column_name(alias))
        if resolved is not None:
            return resolved

    raise ResearchDatasetSchemaError(
        f"Could not identify the {field_name} column. "
        f"Expected one of {list(aliases)!r}. "
        f"Available columns: {list(columns)!r}"
    )


def resolve_column_mapping(
    columns: Sequence[str],
    *,
    id_column: Optional[str] = None,
    title_column: Optional[str] = None,
    abstract_column: Optional[str] = None,
) -> ResearchColumnMapping:
    """
    Resolve the three semantic-index columns.

    No column is guessed from arbitrary values. Only explicit configuration
    or known aliases is accepted.
    """

    if not columns:
        raise ResearchDatasetSchemaError(
            "Research dataset contains no columns."
        )

    document_id = _resolve_column(
        columns,
        id_column,
        _ID_ALIASES,
        "document ID",
    )

    title = _resolve_column(
        columns,
        title_column,
        _TITLE_ALIASES,
        "title",
    )

    abstract = _resolve_column(
        columns,
        abstract_column,
        _ABSTRACT_ALIASES,
        "abstract",
    )

    if len({document_id, title, abstract}) != 3:
        raise ResearchDatasetSchemaError(
            "Document ID, title, and abstract columns must be distinct."
        )

    return ResearchColumnMapping(
        document_id=document_id,
        title=title,
        abstract=abstract,
    )


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------


class ResearchDatasetReader:
    """
    Memory-conscious CSV reader.

    The 433 MB research CSV is streamed rather than loaded into a giant
    DataFrame/list.
    """

    def __init__(
        self,
        path: Path,
        *,
        mapping: Optional[ResearchColumnMapping] = None,
        id_column: Optional[str] = None,
        title_column: Optional[str] = None,
        abstract_column: Optional[str] = None,
        encoding: str = "utf-8-sig",
    ) -> None:
        self.path = Path(path)
        self.mapping = mapping
        self.id_column = id_column
        self.title_column = title_column
        self.abstract_column = abstract_column
        self.encoding = encoding

    def _open(self):
        if not self.path.is_file():
            raise ResearchDatasetError(
                f"Research dataset does not exist: {self.path}"
            )

        if self.path.stat().st_size <= 0:
            raise ResearchDatasetError(
                f"Research dataset is empty: {self.path}"
            )

        return self.path.open(
            "r",
            encoding=self.encoding,
            newline="",
        )

    def resolve_mapping(self) -> ResearchColumnMapping:
        """Read only the CSV header and resolve its schema."""

        with self._open() as handle:
            reader = csv.reader(handle)

            try:
                header = next(reader)
            except StopIteration as exc:
                raise ResearchDatasetSchemaError(
                    "Research CSV contains no header."
                ) from exc

        columns = [
            str(column).strip()
            for column in header
        ]

        mapping = resolve_column_mapping(
            columns,
            id_column=self.id_column,
            title_column=self.title_column,
            abstract_column=self.abstract_column,
        )

        self.mapping = mapping
        return mapping

    def iter_records(
        self,
        *,
        limit: Optional[int] = None,
    ) -> Iterator[ResearchRecord]:
        """
        Stream validated records from the corpus.

        Input order is preserved exactly.
        """

        mapping = self.mapping or self.resolve_mapping()

        processed = 0

        with self._open() as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ResearchDatasetSchemaError(
                    "Research CSV has no readable field names."
                )

            actual_fields = [
                str(field).strip()
                for field in reader.fieldnames
                if field is not None
            ]

            # The header must still contain the resolved columns.
            for required in (
                mapping.document_id,
                mapping.title,
                mapping.abstract,
            ):
                if required not in actual_fields:
                    raise ResearchDatasetSchemaError(
                        f"Resolved column {required!r} is absent from "
                        f"the CSV header."
                    )

            for row_number, row in enumerate(reader, start=2):
                if limit is not None and processed >= limit:
                    break

                raw_id = row.get(mapping.document_id)
                raw_title = row.get(mapping.title)
                raw_abstract = row.get(mapping.abstract)

                document_id = _clean_text(raw_id)
                title = _clean_text(raw_title)
                abstract = _clean_text(raw_abstract)

                if not document_id:
                    raise ResearchDatasetError(
                        f"Missing document ID at CSV row {row_number}."
                    )

                if not title and not abstract:
                    raise ResearchDatasetError(
                        "Paper has no usable title or abstract at "
                        f"CSV row {row_number}; document_id={document_id!r}."
                    )

                # For retrieval, an abstract-only paper is still valid.
                # The title/abstract combination is preferred whenever
                # both are available.
                yield ResearchRecord(
                    document_id=document_id,
                    title=title,
                    abstract=abstract,
                )

                processed += 1


def _clean_text(value: Any) -> str:
    """Convert supported CSV text values into stripped strings."""

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return value.strip()


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------


class ResearchMetadataWriter:
    """
    Persist retrieval metadata separately from FAISS.

    FAISS stores only document IDs and vectors. This sidecar preserves the
    information needed later to hydrate search results without forcing
    search.py to load the entire research corpus.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(
        self,
        records: Sequence[ResearchRecord],
    ) -> Path:
        """Write metadata atomically."""

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": "1",
            "document_count": len(records),
            "documents": [
                {
                    "document_id": record.document_id,
                    "title": record.title,
                    "summary": record.abstract,
                }
                for record in records
            ],
        }

        temporary: Optional[Path] = None

        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                text=True,
            )
            os.close(fd)
            temporary = Path(temporary_name)

            with temporary.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")

                handle.flush()
                os.fsync(handle.fileno())

            if temporary.stat().st_size <= 0:
                raise ResearchIndexValidationError(
                    "Research metadata writer produced an empty file."
                )

            os.replace(
                temporary,
                self.path,
            )
            temporary = None

            return self.path

        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning(
                        "Unable to remove temporary metadata file: %s",
                        temporary,
                        exc_info=True,
                    )


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


class ResearchIndexBuilder:
    """
    Build the persistent Mode-1 research FAISS index.

    Dependencies are injected so this class does not implement SPECTER2
    or FAISS itself.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        faiss_index_cls: Any,
        config: Optional[ResearchIndexConfig] = None,
    ) -> None:
        if encoder is None:
            raise ValueError("encoder cannot be None.")

        if faiss_index_cls is None:
            raise ValueError(
                "faiss_index_cls cannot be None."
            )

        self.encoder = encoder
        self.faiss_index_cls = faiss_index_cls
        self.config = config or ResearchIndexConfig()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Validate builder configuration."""

        if self.config.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.config.limit is not None and self.config.limit <= 0:
            raise ValueError(
                "limit must be greater than zero when provided."
            )

        if self.config.dataset_path.suffix.lower() != ".csv":
            raise ResearchDatasetError(
                "The current research builder expects a CSV corpus. "
                f"Received: {self.config.dataset_path}"
            )

    def _validate_embedding_batch(
        self,
        embeddings: Any,
        expected_count: int,
    ) -> np.ndarray:
        """Validate one embedding batch before FAISS mutation."""

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ResearchIndexValidationError(
                "Embedding encoder returned an invalid matrix shape: "
                f"{matrix.shape!r}; expected 2 dimensions."
            )

        if matrix.shape[0] != expected_count:
            raise ResearchIndexValidationError(
                "Embedding/document alignment failure: "
                f"received {matrix.shape[0]} embeddings for "
                f"{expected_count} documents."
            )

        expected_dimension = int(
            self.encoder.embedding_dimension
        )

        if matrix.shape[1] != expected_dimension:
            raise ResearchIndexValidationError(
                "Embedding dimension mismatch: "
                f"encoder reports {expected_dimension}, "
                f"but returned matrix has dimension "
                f"{matrix.shape[1]}."
            )

        if not np.isfinite(matrix).all():
            raise ResearchIndexValidationError(
                "Embedding batch contains NaN or infinite values."
            )

        if matrix.shape[0] > 0:
            norms = np.linalg.norm(
                matrix,
                axis=1,
            )

            if not np.isfinite(norms).all():
                raise ResearchIndexValidationError(
                    "Embedding batch contains invalid vector norms."
                )

            if np.any(norms <= 1e-12):
                raise ResearchIndexValidationError(
                    "Embedding batch contains zero/near-zero vectors."
                )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """
        Build the complete research index.

        The final FAISS files are created only after the complete corpus
        has been successfully embedded and added.

        Returns
        -------
        dict
            Build statistics and output artifact paths.
        """

        self.validate_config()

        config = self.config

        final_index_path = Path(
            f"{config.index_base}.index"
        )

        final_manifest_path = Path(
            f"{config.index_base}.index.manifest.json"
        )

        if not config.overwrite:
            if final_index_path.exists():
                raise ResearchIndexingError(
                    f"Research index already exists: {final_index_path}. "
                    "Use overwrite=True to rebuild it."
                )

            if final_manifest_path.exists():
                raise ResearchIndexingError(
                    f"Research index manifest already exists: "
                    f"{final_manifest_path}. "
                    "Use overwrite=True to rebuild it."
                )

        started_at = time.perf_counter()

        reader = ResearchDatasetReader(
            config.dataset_path,
            id_column=config.id_column,
            title_column=config.title_column,
            abstract_column=config.abstract_column,
        )

        mapping = reader.resolve_mapping()

        LOGGER.info(
            "Research dataset: %s",
            config.dataset_path,
        )
        LOGGER.info(
            "Resolved columns: id=%s, title=%s, abstract=%s",
            mapping.document_id,
            mapping.title,
            mapping.abstract,
        )

        # --------------------------------------------------------------
        # Read + embed in bounded batches
        # --------------------------------------------------------------

        index = None

        all_metadata: list[ResearchRecord] = []
        seen_ids: set[str] = set()

        total_documents = 0
        total_batches = 0

        batch_records: list[ResearchRecord] = []

        def flush_batch() -> None:
            nonlocal index
            nonlocal total_documents
            nonlocal total_batches

            if not batch_records:
                return

            records = list(batch_records)
            texts = [
                record.canonical_text
                for record in records
            ]
            document_ids = [
                record.document_id
                for record in records
            ]

            LOGGER.info(
                "Embedding research batch %d: %d papers",
                total_batches + 1,
                len(records),
            )

            try:
                embeddings = self.encoder.embed_documents(
                    texts
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "SPECTER2 document embedding failed for "
                    f"batch {total_batches + 1}."
                ) from exc

            matrix = self._validate_embedding_batch(
                embeddings,
                len(records),
            )

            if index is None:
                model_name = getattr(
                    self.encoder,
                    "model_name",
                    None,
                )

                model_version = getattr(
                    self.encoder,
                    "model_version",
                    None,
                )

                normalize = getattr(
                    self.encoder,
                    "normalized",
                    None,
                )

                # The FAISS implementation defaults to normalized vectors.
                # If the encoder explicitly reports normalization, respect it.
                if normalize is None:
                    normalize = True

                index = self.faiss_index_cls(
                    matrix.shape[1],
                    normalize=bool(normalize),
                    model_name=model_name,
                    model_version=model_version,
                )

            try:
                index.add(
                    matrix,
                    document_ids,
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "Failed to add research embedding batch "
                    f"{total_batches + 1} to FAISS."
                ) from exc

            all_metadata.extend(records)

            total_documents += len(records)
            total_batches += 1

            LOGGER.info(
                "Research index progress: %d papers",
                total_documents,
            )

            batch_records.clear()

        for record in reader.iter_records(
            limit=config.limit,
        ):
            if record.document_id in seen_ids:
                raise ResearchDatasetError(
                    "Duplicate document ID encountered: "
                    f"{record.document_id!r}"
                )

            seen_ids.add(record.document_id)
            batch_records.append(record)

            if len(batch_records) >= config.batch_size:
                flush_batch()

        # Final partial batch.
        flush_batch()

        if index is None or total_documents == 0:
            raise ResearchIndexingError(
                "No valid research papers were indexed."
            )

        # --------------------------------------------------------------
        # Final integrity checks
        # --------------------------------------------------------------

        if index.ntotal != total_documents:
            raise ResearchIndexValidationError(
                "Final FAISS/document count mismatch: "
                f"FAISS={index.ntotal}, "
                f"records={total_documents}."
            )

        if len(index.document_ids) != total_documents:
            raise ResearchIndexValidationError(
                "Final FAISS document-ID mapping mismatch: "
                f"IDs={len(index.document_ids)}, "
                f"records={total_documents}."
            )

        if len(all_metadata) != total_documents:
            raise ResearchIndexValidationError(
                "Final metadata/document count mismatch: "
                f"metadata={len(all_metadata)}, "
                f"records={total_documents}."
            )

        if tuple(index.document_ids) != tuple(
            record.document_id
            for record in all_metadata
        ):
            raise ResearchIndexValidationError(
                "FAISS document-ID order does not match metadata order."
            )

        try:
            index.validate()
        except Exception as exc:
            raise ResearchIndexValidationError(
                "Final FAISS integrity validation failed."
            ) from exc

        # --------------------------------------------------------------
        # Persist FAISS
        # --------------------------------------------------------------

        config.index_base.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            index_path, manifest_path = index.save(
                config.index_base,
            )
        except Exception as exc:
            raise ResearchIndexingError(
                "Failed to persist the final research FAISS index."
            ) from exc

        if not index_path.is_file() or index_path.stat().st_size <= 0:
            raise ResearchIndexValidationError(
                f"Final FAISS index is missing or empty: {index_path}"
            )

        if (
            not manifest_path.is_file()
            or manifest_path.stat().st_size <= 0
        ):
            raise ResearchIndexValidationError(
                "Final FAISS manifest is missing or empty: "
                f"{manifest_path}"
            )

        # --------------------------------------------------------------
        # Persist metadata sidecar
        # --------------------------------------------------------------

        metadata_path = None

        if config.metadata_path is not None:
            writer = ResearchMetadataWriter(
                config.metadata_path
            )

            try:
                metadata_path = writer.write(
                    all_metadata
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "FAISS index was created, but research metadata "
                    "could not be persisted."
                ) from exc

        elapsed = time.perf_counter() - started_at

        statistics = {
            "status": "completed",
            "dataset_path": str(
                config.dataset_path
            ),
            "index_base": str(
                config.index_base
            ),
            "index_path": str(
                index_path
            ),
            "manifest_path": str(
                manifest_path
            ),
            "metadata_path": (
                str(metadata_path)
                if metadata_path is not None
                else None
            ),
            "document_count": total_documents,
            "embedding_dimension": int(
                index.dimension
            ),
            "index_type": index.index_type,
            "metric": index.metric,
            "similarity": index.similarity,
            "model_name": getattr(
                self.encoder,
                "model_name",
                None,
            ),
            "model_version": getattr(
                self.encoder,
                "model_version",
                None,
            ),
            "batches": total_batches,
            "elapsed_seconds": round(
                elapsed,
                3,
            ),
        }

        LOGGER.info(
            "Research FAISS index build completed: "
            "%d papers, dimension=%d, elapsed=%.2fs",
            total_documents,
            index.dimension,
            elapsed,
        )

        return statistics



# ---------------------------------------------------------------------------
# Mode-2 uploaded-paper indexing
# ---------------------------------------------------------------------------

INDEX_ROOT = Path("indexes") / "uploaded"
INDEX_BASE_NAME = "index"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = "1"
METADATA_SCHEMA_VERSION = "1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PaperIndexingError(RuntimeError):
    """Base uploaded-paper indexing error."""


class InvalidPaperIndexingInputError(PaperIndexingError, ValueError):
    """Invalid indexing input."""


class PaperIndexingAlignmentError(PaperIndexingError):
    """Chunk/vector/metadata alignment failure."""


class PaperIndexPersistenceError(PaperIndexingError):
    """Persistence or filesystem failure."""


class PaperIndexValidationError(PaperIndexingError):
    """Persisted index integrity failure."""


@dataclass(frozen=True)
class UploadedPaperIndexConfig:
    root_dir: Path = INDEX_ROOT
    overwrite: bool = False
    include_chunk_text: bool = True
    verify_reload: bool = True
    max_chunk_text_chars: int = 2_000_000

    # Retrieval/index quality controls.  These do not silently invent chunks;
    # they only reject an obviously incomplete indexing result when the caller
    # supplies the expected page count or asks for a minimum chunk count.
    min_chunks: int = 1
    expected_page_count: Optional[int] = None
    require_page_coverage: bool = False

    def __post_init__(self) -> None:
        if self.max_chunk_text_chars <= 0:
            raise ValueError("max_chunk_text_chars must be > 0.")
        if self.min_chunks <= 0:
            raise ValueError("min_chunks must be > 0.")
        if self.expected_page_count is not None and self.expected_page_count <= 0:
            raise ValueError("expected_page_count must be > 0 when provided.")
        if self.require_page_coverage and self.expected_page_count is None:
            raise ValueError(
                "require_page_coverage=True requires expected_page_count."
            )


@dataclass(frozen=True)
class IndexedPaperResult:
    document_id: str
    index_directory: Path
    index_path: Path
    faiss_manifest_path: Path
    metadata_path: Path
    vector_count: int
    metadata_count: int
    embedding_dimension: int
    embedding_model: Optional[str]
    embedding_model_version: Optional[str]
    normalized: bool
    index_type: str
    similarity: str
    created_at: str
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "index_directory": str(self.index_directory),
            "index_path": str(self.index_path),
            "faiss_manifest_path": str(self.faiss_manifest_path),
            "metadata_path": str(self.metadata_path),
            "vector_count": self.vector_count,
            "metadata_count": self.metadata_count,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "embedding_model_version": self.embedding_model_version,
            "normalized": self.normalized,
            "index_type": self.index_type,
            "similarity": self.similarity,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    document_id: str
    text: str
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
    metadata: Mapping[str, Any]


class UploadedPaperIndexer:
    """Build one isolated FAISS index for one uploaded document."""

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        config: Optional[UploadedPaperIndexConfig] = None,
        faiss_index_cls: type[FAISSVectorIndex] = FAISSVectorIndex,
    ) -> None:
        if encoder is None:
            raise InvalidPaperIndexingInputError("encoder cannot be None.")
        self.encoder = encoder
        self.config = config or UploadedPaperIndexConfig()
        self.faiss_index_cls = faiss_index_cls

    def build(
        self,
        *,
        document_id: str,
        chunks: Sequence[Any],
        source_document_id: Optional[str] = None,
        overwrite: Optional[bool] = None,
    ) -> IndexedPaperResult:
        document_id = self._validate_document_id(document_id)
        normalized = self._validate_chunks(document_id, chunks)

        if len(normalized) < self.config.min_chunks:
            raise InvalidPaperIndexingInputError(
                f"Indexed chunk count {len(normalized)} is below the configured "
                f"minimum {self.config.min_chunks}. This usually indicates "
                "incomplete parsing/chunking upstream."
            )

        if self.config.expected_page_count is not None:
            self._validate_page_coverage(
                normalized,
                expected_page_count=self.config.expected_page_count,
                require_full_coverage=self.config.require_page_coverage,
            )

        if not normalized:
            raise InvalidPaperIndexingInputError(
                "Cannot index an empty chunk collection."
            )

        target = self._target_dir(document_id)
        allow_overwrite = (
            self.config.overwrite
            if overwrite is None
            else bool(overwrite)
        )
        if target.exists():
            if not allow_overwrite:
                raise PaperIndexPersistenceError(
                    f"Index already exists for {document_id!r}. "
                    "Use overwrite=True for an explicit rebuild."
                )
            if not target.is_dir():
                raise PaperIndexPersistenceError(
                    f"Index target is not a directory: {target}"
                )

        LOGGER.info(
            "Starting uploaded-paper indexing: document_id=%s chunks=%d",
            document_id, len(normalized),
        )

        embeddings = self._embed(normalized)
        self._validate_alignment(normalized, embeddings)

        created_at = datetime.now(timezone.utc).isoformat()
        staging = self._make_staging_dir(target)

        try:
            result = self._build_staged(
                document_id=document_id,
                source_document_id=source_document_id or document_id,
                chunks=normalized,
                embeddings=embeddings,
                staging=staging,
                created_at=created_at,
            )
            self._publish(staging, target)
            staging = None  # type: ignore[assignment]

            return IndexedPaperResult(
                document_id=result.document_id,
                index_directory=target,
                index_path=target / result.index_path.name,
                faiss_manifest_path=target / result.faiss_manifest_path.name,
                metadata_path=target / METADATA_FILENAME,
                vector_count=result.vector_count,
                metadata_count=result.metadata_count,
                embedding_dimension=result.embedding_dimension,
                embedding_model=result.embedding_model,
                embedding_model_version=result.embedding_model_version,
                normalized=result.normalized,
                index_type=result.index_type,
                similarity=result.similarity,
                created_at=result.created_at,
            )
        finally:
            if staging is not None:
                self._safe_remove(staging)

    build_uploaded_paper_index = build

    def load(self, *, document_id: str) -> FAISSVectorIndex:
        document_id = self._validate_document_id(document_id)
        directory = self._target_dir(document_id)

        if not directory.is_dir():
            raise PaperIndexPersistenceError(
                f"Uploaded-paper index not found: {directory}"
            )

        try:
            index = self.faiss_index_cls.load(
                directory / INDEX_BASE_NAME,
                expected_model_name=self._model_name(),
                expected_model_version=self._model_version(),
            )
            index.validate()
        except Exception as exc:
            raise PaperIndexPersistenceError(
                f"Failed to load index for {document_id!r}: {exc}"
            ) from exc

        self._validate_metadata(
            directory / METADATA_FILENAME,
            document_id=document_id,
            expected_count=index.ntotal,
            expected_dimension=index.dimension,
        )
        self._validate_faiss_manifest(
            directory / f"{INDEX_BASE_NAME}.index.manifest.json",
            document_id=document_id,
            expected_chunk_ids=list(index.document_ids),
            expected_dimension=index.dimension,
        )
        return index

    def validate_persisted(
        self,
        *,
        document_id: str,
        expected_chunk_ids: Optional[Sequence[str]] = None,
        expected_source_document_id: Optional[str] = None,
    ) -> IndexedPaperResult:
        """
        Validate a persisted uploaded-paper index.

        When current chunk IDs/source identity are supplied, validation also
        proves that the persisted index represents the CURRENT ingestion graph.
        This prevents an older index for the same content-addressed PDF from
        being incorrectly returned as a duplicate after parser/chunker changes.
        """
        document_id = self._validate_document_id(document_id)
        index = self.load(document_id=document_id)
        directory = self._target_dir(document_id)
        payload = self._read_json(directory / METADATA_FILENAME)
        manifest = index.manifest()

        if expected_source_document_id is not None:
            persisted_source_id = payload.get("source_document_id")
            if persisted_source_id != expected_source_document_id:
                raise PaperIndexValidationError(
                    "Persisted source_document_id does not match the current "
                    f"source: expected={expected_source_document_id!r}, "
                    f"actual={persisted_source_id!r}."
                )

        if expected_chunk_ids is not None:
            current_ids = list(expected_chunk_ids)
            persisted_ids = list(index.document_ids)
            if persisted_ids != current_ids:
                raise PaperIndexingAlignmentError(
                    "Persisted chunk-ID mapping does not match the current "
                    f"chunk graph: persisted={len(persisted_ids)} "
                    f"expected={len(current_ids)}."
                )

        return IndexedPaperResult(
            document_id=document_id,
            index_directory=directory,
            index_path=directory / f"{INDEX_BASE_NAME}.index",
            faiss_manifest_path=directory / f"{INDEX_BASE_NAME}.index.manifest.json",
            metadata_path=directory / METADATA_FILENAME,
            vector_count=index.ntotal,
            metadata_count=len(payload["chunks"]),
            embedding_dimension=index.dimension,
            embedding_model=manifest.model_name,
            embedding_model_version=manifest.model_version,
            normalized=manifest.normalized,
            index_type=manifest.index_type,
            similarity=manifest.similarity,
            created_at=str(payload.get("created_at", manifest.created_at)),
            status="validated",
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_document_id(document_id: str) -> str:
        if not isinstance(document_id, str):
            raise InvalidPaperIndexingInputError(
                "document_id must be a string."
            )
        value = document_id.strip()
        if (
            not value
            or not _SAFE_ID.fullmatch(value)
            or "/" in value
            or "\\" in value
            or ":" in value
            or value in {".", ".."}
            or ".." in Path(value).parts
            or Path(value).is_absolute()
        ):
            raise InvalidPaperIndexingInputError(
                "document_id is invalid or unsafe for filesystem use."
            )
        return value

    def _validate_chunks(
        self,
        document_id: str,
        chunks: Sequence[Any],
    ) -> tuple[_Chunk, ...]:
        if chunks is None or isinstance(chunks, (str, bytes)):
            raise InvalidPaperIndexingInputError(
                "chunks must be a sequence of SemanticChunk objects."
            )

        try:
            raw = list(chunks)
        except TypeError as exc:
            raise InvalidPaperIndexingInputError(
                "chunks must be iterable."
            ) from exc

        result: list[_Chunk] = []
        seen: set[str] = set()

        required = (
            "chunk_id", "document_id", "text", "section_id",
            "section_type", "section_level", "start_page", "end_page",
            "source_pages", "parent_section_id", "chunk_index",
            "token_count", "char_count", "overlap_with_previous",
            "chunking_method",
        )

        for pos, item in enumerate(raw):
            if item is None:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {pos} is None."
                )

            missing = [
                name for name in required
                if not hasattr(item, name)
            ]
            if missing:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {pos} is missing fields: {missing}."
                )

            values = {name: getattr(item, name) for name in required}
            metadata = getattr(item, "metadata", {})

            cid = values["chunk_id"]
            if not isinstance(cid, str) or not cid.strip():
                raise InvalidPaperIndexingInputError(
                    f"Chunk {pos}: invalid chunk_id."
                )
            cid = cid.strip()

            if cid in seen:
                raise PaperIndexingAlignmentError(
                    f"Duplicate chunk_id: {cid!r}."
                )
            seen.add(cid)

            if values["document_id"] != document_id:
                raise PaperIndexingAlignmentError(
                    f"Chunk {cid!r} belongs to document "
                    f"{values['document_id']!r}, expected {document_id!r}."
                )

            text = values["text"]
            if not isinstance(text, str) or not text.strip():
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: text must be non-empty."
                )
            text = text.strip()

            if len(text) > self.config.max_chunk_text_chars:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: text exceeds configured metadata limit."
                )

            if not isinstance(values["section_id"], str) or not values["section_id"].strip():
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: invalid section_id."
                )
            if not isinstance(values["section_type"], str) or not values["section_type"].strip():
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: invalid section_type."
                )

            for field in (
                "section_level", "chunk_index", "token_count",
                "char_count", "overlap_with_previous",
            ):
                value = values[field]
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise InvalidPaperIndexingInputError(
                        f"Chunk {cid!r}: {field} must be a non-negative integer."
                    )

            if values["section_level"] < 1:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: section_level must be >= 1."
                )
            if values["token_count"] < 1:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: token_count must be >= 1."
                )
            if values["char_count"] != len(text):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: char_count does not match text."
                )

            pages = tuple(values["source_pages"] or ())
            if pages != tuple(sorted(set(pages))):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: source_pages must be sorted and unique."
                )
            if any(
                not isinstance(p, int) or isinstance(p, bool) or p < 1
                for p in pages
            ):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: invalid source page."
                )

            start_page = values["start_page"]
            end_page = values["end_page"]
            if start_page is not None and (
                not isinstance(start_page, int) or start_page < 1
            ):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: invalid start_page."
                )
            if end_page is not None and (
                not isinstance(end_page, int) or end_page < 1
            ):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: invalid end_page."
                )
            if (
                start_page is not None
                and end_page is not None
                and start_page > end_page
            ):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: start_page > end_page."
                )
            if pages and start_page is not None and pages[0] != start_page:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: source_pages/start_page mismatch."
                )
            if pages and end_page is not None and pages[-1] != end_page:
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: source_pages/end_page mismatch."
                )

            if not isinstance(metadata, Mapping):
                raise InvalidPaperIndexingInputError(
                    f"Chunk {cid!r}: metadata must be a mapping."
                )
            self._ensure_jsonable(
                metadata,
                f"metadata for {cid!r}",
            )

            result.append(
                _Chunk(
                    chunk_id=cid,
                    document_id=document_id,
                    text=text,
                    section_id=values["section_id"].strip(),
                    section_type=values["section_type"].strip(),
                    section_heading=(
                        values.get("section_heading")
                        if isinstance(values.get("section_heading"), str)
                        else None
                    ),
                    section_level=values["section_level"],
                    start_page=start_page,
                    end_page=end_page,
                    source_pages=pages,
                    parent_section_id=(
                        values["parent_section_id"].strip()
                        if isinstance(values["parent_section_id"], str)
                        else None
                    ),
                    chunk_index=values["chunk_index"],
                    token_count=values["token_count"],
                    char_count=values["char_count"],
                    overlap_with_previous=values["overlap_with_previous"],
                    chunking_method=str(values["chunking_method"]),
                    metadata=dict(metadata),
                )
            )

        expected = list(range(len(result)))
        actual = [item.chunk_index for item in result]
        if actual != expected:
            raise PaperIndexingAlignmentError(
                f"Chunk order is invalid. Expected {expected}, got {actual}."
            )

        return tuple(result)

    @staticmethod
    def _validate_page_coverage(
        chunks: Sequence[_Chunk],
        *,
        expected_page_count: int,
        require_full_coverage: bool,
    ) -> None:
        """Validate page references when the upload pipeline provides a page count."""
        referenced: set[int] = set()
        for chunk in chunks:
            referenced.update(chunk.source_pages)
            if chunk.start_page is not None:
                referenced.add(chunk.start_page)
            if chunk.end_page is not None:
                referenced.add(chunk.end_page)

        invalid = sorted(
            page for page in referenced
            if page < 1 or page > expected_page_count
        )
        if invalid:
            raise PaperIndexingAlignmentError(
                "Chunk page metadata exceeds the source document bounds: "
                f"invalid_pages={invalid[:10]!r}, "
                f"expected_page_count={expected_page_count}."
            )

        if require_full_coverage:
            missing = sorted(set(range(1, expected_page_count + 1)) - referenced)
            if missing:
                raise PaperIndexingAlignmentError(
                    "Indexed chunks do not cover the complete uploaded document: "
                    f"missing_pages={missing[:20]!r}, "
                    f"missing_count={len(missing)}."
                )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, chunks: Sequence[_Chunk]) -> np.ndarray:
        texts = tuple(chunk.text for chunk in chunks)

        LOGGER.info(
            "Generating embeddings: chunks=%d model=%s",
            len(texts), self._model_name() or "unknown",
        )

        try:
            embeddings = self.encoder.embed_documents(texts)
        except Exception as exc:
            raise PaperIndexingError(
                f"Embedding generation failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise PaperIndexingAlignmentError(
                f"Encoder must return a 2-D numpy array; got "
                f"{type(embeddings).__name__} {getattr(embeddings, 'shape', None)}."
            )

        expected_dim = int(self.encoder.embedding_dimension)

        if embeddings.shape != (len(chunks), expected_dim):
            raise PaperIndexingAlignmentError(
                "Embedding matrix shape mismatch: "
                f"expected={(len(chunks), expected_dim)}, "
                f"got={embeddings.shape}."
            )

        if embeddings.dtype.kind != "f":
            raise PaperIndexingAlignmentError(
                f"Embedding dtype must be floating-point, got {embeddings.dtype}."
            )

        if not np.isfinite(embeddings).all():
            raise PaperIndexingAlignmentError(
                "Embedding matrix contains NaN or infinity."
            )

        # Reuse the authoritative encoder validation contract.
        try:
            embeddings = self.encoder.validate_embeddings(
                embeddings,
                expected_count=len(chunks),
                expected_dimension=expected_dim,
            )
        except Exception as exc:
            raise PaperIndexingAlignmentError(
                f"Encoder validation failed: {exc}"
            ) from exc

        LOGGER.info(
            "Embeddings ready: vectors=%d dimension=%d",
            embeddings.shape[0], embeddings.shape[1],
        )
        return embeddings

    @staticmethod
    def _validate_alignment(
        chunks: Sequence[_Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != int(embeddings.shape[0]):
            raise PaperIndexingAlignmentError(
                f"Alignment failure: chunks={len(chunks)}, "
                f"embeddings={embeddings.shape[0]}."
            )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_staged(
        self,
        *,
        document_id: str,
        source_document_id: str,
        chunks: Sequence[_Chunk],
        embeddings: np.ndarray,
        staging: Path,
        created_at: str,
    ) -> IndexedPaperResult:
        staging.mkdir(parents=True, exist_ok=True)

        dimension = int(embeddings.shape[1])
        normalized = self._encoder_normalized()
        if normalized is None:
            # The existing FAISS implementation's default is True and its
            # IndexFlatIP semantics are cosine when vectors are normalized.
            normalized = True

        LOGGER.info(
            "Building FAISS index: dimension=%d normalize=%s",
            dimension, normalized,
        )

        index = self.faiss_index_cls(
            dimension=dimension,
            normalize=bool(normalized),
            model_name=self._model_name(),
            model_version=self._model_version(),
            expected_document_count=len(chunks),
        )

        index.add(
            embeddings,
            [chunk.chunk_id for chunk in chunks],
        )
        index.validate()

        if index.ntotal != len(chunks):
            raise PaperIndexingAlignmentError(
                f"FAISS/vector count mismatch: {index.ntotal} != {len(chunks)}."
            )

        index_path, faiss_manifest_path = index.save(
            staging / INDEX_BASE_NAME
        )

        # IMPORTANT:
        # FAISSVectorIndex.document_ids are vector-level identifiers. For
        # uploaded-paper indexes those identifiers MUST remain chunk IDs,
        # because retrieval needs vector_position -> chunk_id resolution.
        # The previous failure happened because the consumer treated that
        # field as the paper/document ID.
        #
        # Keep FAISS's native field intact and add an explicit owner-level
        # contract to the manifest. This makes the artifact self-describing
        # without breaking FAISS result mapping.
        self._augment_faiss_manifest(
            faiss_manifest_path,
            document_id=document_id,
            chunks=chunks,
        )

        metadata_path = staging / METADATA_FILENAME
        payload = self._metadata_payload(
            document_id=document_id,
            source_document_id=source_document_id,
            chunks=chunks,
            index=index,
            created_at=created_at,
        )
        self._write_json_atomic(metadata_path, payload)

        self._validate_metadata(
            metadata_path,
            document_id=document_id,
            expected_count=len(chunks),
            expected_dimension=dimension,
        )
        self._validate_faiss_manifest(
            faiss_manifest_path,
            document_id=document_id,
            expected_chunk_ids=[chunk.chunk_id for chunk in chunks],
            expected_dimension=dimension,
        )

        if self.config.verify_reload:
            try:
                reloaded = self.faiss_index_cls.load(
                    staging / INDEX_BASE_NAME,
                    expected_model_name=self._model_name(),
                    expected_model_version=self._model_version(),
                    expected_document_count=len(chunks),
                )
                reloaded.validate()
                if reloaded.ntotal != len(chunks):
                    raise PaperIndexingAlignmentError(
                        "Reloaded FAISS vector count mismatch."
                    )
            except Exception as exc:
                raise PaperIndexValidationError(
                    f"Index save/reload validation failed: {exc}"
                ) from exc

        return IndexedPaperResult(
            document_id=document_id,
            index_directory=staging,
            index_path=index_path,
            faiss_manifest_path=faiss_manifest_path,
            metadata_path=metadata_path,
            vector_count=index.ntotal,
            metadata_count=len(chunks),
            embedding_dimension=index.dimension,
            embedding_model=self._model_name(),
            embedding_model_version=self._model_version(),
            normalized=index.normalize,
            index_type=index.index_type,
            similarity=index.similarity,
            created_at=created_at,
        )

    def _metadata_payload(
        self,
        *,
        document_id: str,
        source_document_id: str,
        chunks: Sequence[_Chunk],
        index: FAISSVectorIndex,
        created_at: str,
    ) -> dict[str, Any]:
        model_info = self._model_info()

        records: list[dict[str, Any]] = []
        for position, chunk in enumerate(chunks):
            record: dict[str, Any] = {
                "vector_position": position,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "section_type": chunk.section_type,
                "section_heading": chunk.section_heading,
                "section_level": chunk.section_level,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "source_pages": list(chunk.source_pages),
                "parent_section_id": chunk.parent_section_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "overlap_with_previous": chunk.overlap_with_previous,
                "chunking_method": chunk.chunking_method,
                "metadata": self._json_safe(chunk.metadata),
            }
            if self.config.include_chunk_text:
                record["text"] = chunk.text
            records.append(record)

        payload = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "document_id": document_id,
            "source_document_id": source_document_id,
            "created_at": created_at,
            "vector_count": index.ntotal,
            "embedding_dimension": index.dimension,
            "index_type": index.index_type,
            "metric": index.metric,
            "similarity": index.similarity,
            "normalized": index.normalize,
            "embedding": model_info,

            # Explicit retrieval contract:
            # vector_position -> chunk_id -> document_id.
            # `document_id` above is the owning uploaded paper ID.
            "id_contract": {
                "document_id": "paper/document owner ID",
                "vector_id": "chunk_id",
                "vector_id_field": "faiss_manifest.document_ids",
                "mapping_field": "chunks[].vector_position",
                "mapping_key": "chunks[].chunk_id",
            },
            "vector_ids": [chunk.chunk_id for chunk in chunks],
            "chunks": records,
        }

        self._ensure_jsonable(payload, "metadata payload")
        return payload

    # ------------------------------------------------------------------
    # Persistence and validation
    # ------------------------------------------------------------------

    def _target_dir(self, document_id: str) -> Path:
        root = Path(self.config.root_dir)
        target = root / document_id

        try:
            root_resolved = root.resolve()
            target_resolved = target.resolve()
        except OSError as exc:
            raise PaperIndexPersistenceError(
                "Could not resolve index path safely."
            ) from exc

        if target_resolved.parent != root_resolved:
            raise PaperIndexPersistenceError(
                "Resolved document path escapes uploaded-index root."
            )
        return target

    @staticmethod
    def _make_staging_dir(target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.building-",
                dir=str(target.parent),
            )
        )

    @staticmethod
    def _publish(staging: Path, target: Path) -> None:
        """
        Publish only a completely built directory.

        If a target exists, rename it to a same-parent backup first, publish the
        new directory, then remove the backup. Rollback is attempted on failure.
        """
        backup: Optional[Path] = None
        parent = target.parent

        try:
            if target.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{target.name}.backup-",
                        dir=str(parent),
                    )
                )
                backup.rmdir()
                os.replace(target, backup)

            os.replace(staging, target)
            staging = None  # type: ignore[assignment]

            if backup is not None:
                shutil.rmtree(backup, ignore_errors=False)
                backup = None
        except Exception as exc:
            try:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                    backup = None
            except Exception:
                LOGGER.exception("Index publication rollback failed.")
            raise PaperIndexPersistenceError(
                f"Could not publish index directory: {exc}"
            ) from exc
        finally:
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)

    @staticmethod
    def _safe_remove(path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            LOGGER.warning("Could not remove temporary directory: %s", path)

    def _validate_metadata(
        self,
        path: Path,
        *,
        document_id: str,
        expected_count: int,
        expected_dimension: Optional[int] = None,
    ) -> None:
        payload = self._read_json(path)

        required = {
            "schema_version", "document_id", "vector_count",
            "embedding_dimension", "index_type", "metric",
            "similarity", "normalized", "embedding", "id_contract",
            "vector_ids", "chunks",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise PaperIndexValidationError(
                f"Metadata missing required fields: {missing}."
            )

        if payload["schema_version"] != METADATA_SCHEMA_VERSION:
            raise PaperIndexValidationError(
                f"Unsupported metadata schema: {payload['schema_version']!r}."
            )

        if payload["document_id"] != document_id:
            raise PaperIndexValidationError(
                "Metadata document_id does not match requested document."
            )

        if payload["vector_count"] != expected_count:
            raise PaperIndexingAlignmentError(
                f"Metadata vector_count={payload['vector_count']} "
                f"but expected {expected_count}."
            )

        dimension = payload["embedding_dimension"]
        if not isinstance(dimension, int) or dimension <= 0:
            raise PaperIndexValidationError(
                "Invalid metadata embedding_dimension."
            )
        if expected_dimension is not None and dimension != expected_dimension:
            raise PaperIndexingAlignmentError(
                f"Metadata dimension={dimension}, "
                f"expected={expected_dimension}."
            )

        records = payload["chunks"]
        if not isinstance(records, list) or len(records) != expected_count:
            raise PaperIndexingAlignmentError(
                "Metadata chunk count does not equal vector count."
            )

        vector_ids = payload["vector_ids"]
        if not isinstance(vector_ids, list) or len(vector_ids) != expected_count:
            raise PaperIndexingAlignmentError(
                "Metadata vector_ids count does not equal vector count."
            )

        contract = payload["id_contract"]
        if not isinstance(contract, dict):
            raise PaperIndexValidationError(
                "Metadata id_contract must be an object."
            )
        if contract.get("vector_id") != "chunk_id":
            raise PaperIndexValidationError(
                "Metadata ID contract must define vector_id='chunk_id'."
            )
        if contract.get("mapping_key") != "chunks[].chunk_id":
            raise PaperIndexValidationError(
                "Metadata ID contract has an invalid mapping key."
            )

        seen: set[str] = set()
        for position, record in enumerate(records):
            if not isinstance(record, dict):
                raise PaperIndexValidationError(
                    f"Metadata record {position} is not an object."
                )

            for key in (
                "vector_position", "chunk_id", "document_id",
                "section_id", "section_type", "chunk_index",
            ):
                if key not in record:
                    raise PaperIndexValidationError(
                        f"Metadata record {position} missing {key!r}."
                    )

            if record["vector_position"] != position:
                raise PaperIndexingAlignmentError(
                    "Metadata vector positions are not contiguous."
                )
            if record["chunk_index"] != position:
                raise PaperIndexingAlignmentError(
                    "Metadata chunk indices do not match vector positions."
                )
            if record["document_id"] != document_id:
                raise PaperIndexingAlignmentError(
                    "Metadata contains a chunk from another document."
                )

            cid = record["chunk_id"]
            if not isinstance(cid, str) or not cid or cid in seen:
                raise PaperIndexingAlignmentError(
                    f"Invalid/duplicate metadata chunk_id: {cid!r}."
                )
            seen.add(cid)

            if vector_ids[position] != cid:
                raise PaperIndexingAlignmentError(
                    "Metadata vector_ids are not aligned with chunk records: "
                    f"position={position}, vector_id={vector_ids[position]!r}, "
                    f"chunk_id={cid!r}."
                )

    @classmethod
    def _augment_faiss_manifest(
        cls,
        path: Path,
        *,
        document_id: str,
        chunks: Sequence[_Chunk],
    ) -> None:
        """Add owner-level metadata while preserving FAISS vector IDs."""
        payload = cls._read_json(path)

        faiss_ids = payload.get("document_ids")
        expected_ids = [chunk.chunk_id for chunk in chunks]
        if faiss_ids != expected_ids:
            raise PaperIndexingAlignmentError(
                "FAISS manifest vector IDs do not match chunk order."
            )

        payload["schema_version"] = str(payload.get("schema_version", SCHEMA_VERSION))
        payload["paper_document_id"] = document_id
        payload["document_id"] = document_id
        payload["id_contract"] = {
            "document_id": "paper/document owner ID",
            "vector_id": "chunk_id",
            "vector_ids_field": "document_ids",
        }
        payload["vector_ids"] = expected_ids
        payload["vector_count"] = len(expected_ids)

        cls._write_json_atomic(path, payload)

    @classmethod
    def _validate_faiss_manifest(
        cls,
        path: Path,
        *,
        document_id: str,
        expected_chunk_ids: Sequence[str],
        expected_dimension: int,
    ) -> None:
        """Validate owner ID and vector-level chunk-ID mapping in the manifest."""
        payload = cls._read_json(path)

        owner = payload.get("paper_document_id", payload.get("document_id"))
        if owner != document_id:
            raise PaperIndexValidationError(
                "FAISS manifest paper/document ID does not match the "
                f"requested uploaded paper: expected={document_id!r}, "
                f"actual={owner!r}."
            )

        ids = payload.get("document_ids")
        if ids != list(expected_chunk_ids):
            raise PaperIndexingAlignmentError(
                "FAISS manifest document_ids must contain chunk IDs in "
                "exact vector order."
            )

        vector_ids = payload.get("vector_ids", ids)
        if vector_ids != list(expected_chunk_ids):
            raise PaperIndexingAlignmentError(
                "FAISS manifest vector_ids do not match chunk IDs."
            )

        if payload.get("vector_count", payload.get("document_count")) != len(
            expected_chunk_ids
        ):
            raise PaperIndexingAlignmentError(
                "FAISS manifest vector count does not match chunk count."
            )

        manifest_dimension = payload.get("embedding_dimension")
        if manifest_dimension is not None and int(manifest_dimension) != expected_dimension:
            raise PaperIndexingAlignmentError(
                "FAISS manifest embedding dimension does not match the index."
            )

        contract = payload.get("id_contract")
        if not isinstance(contract, dict):
            raise PaperIndexValidationError(
                "FAISS manifest is missing id_contract."
            )
        if contract.get("vector_id") != "chunk_id":
            raise PaperIndexValidationError(
                "FAISS manifest id_contract must define vector_id='chunk_id'."
            )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise PaperIndexPersistenceError(
                f"Metadata file does not exist: {path}"
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PaperIndexPersistenceError(
                f"Could not read JSON metadata: {path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise PaperIndexValidationError(
                "Metadata root must be a JSON object."
            )
        return payload

    @classmethod
    def _write_json_atomic(
        cls,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            safe = cls._json_safe(payload)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(
                    safe,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise PaperIndexPersistenceError(
                f"Failed to write metadata {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Encoder metadata / JSON
    # ------------------------------------------------------------------

    def _model_name(self) -> Optional[str]:
        value = getattr(self.encoder, "model_name", None)
        return value if isinstance(value, str) else None

    def _model_version(self) -> Optional[str]:
        value = getattr(self.encoder, "model_version", None)
        return value if isinstance(value, str) else None

    def _encoder_normalized(self) -> Optional[bool]:
        value = getattr(self.encoder, "normalized", None)
        return None if value is None else bool(value)

    def _model_info(self) -> dict[str, Any]:
        try:
            info = self.encoder.get_model_info()
            if is_dataclass(info):
                raw = asdict(info)
            elif isinstance(info, Mapping):
                raw = dict(info)
            else:
                raw = {
                    key: getattr(info, key)
                    for key in (
                        "model_name", "embedding_dimension",
                        "model_version", "device",
                        "dtype", "normalized",
                    )
                    if hasattr(info, key)
                }
        except Exception:
            raw = {}

        raw.setdefault("model_name", self._model_name())
        raw.setdefault("model_version", self._model_version())
        raw.setdefault(
            "embedding_dimension",
            int(self.encoder.embedding_dimension),
        )
        raw.setdefault("normalized", self._encoder_normalized())
        self._ensure_jsonable(raw, "embedding model metadata")
        return raw

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return cls._json_safe(asdict(value))
        if isinstance(value, Mapping):
            return {
                str(k): cls._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(v) for v in value]
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable."
        )

    @classmethod
    def _ensure_jsonable(cls, value: Any, context: str) -> None:
        try:
            json.dumps(
                cls._json_safe(value),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidPaperIndexingInputError(
                f"{context} is not JSON serializable: {exc}"
            ) from exc


def build_uploaded_paper_index(
    *,
    document_id: str,
    chunks: Sequence[Any],
    encoder: EmbeddingEncoder,
    root_dir: Path = INDEX_ROOT,
    overwrite: bool = False,
    source_document_id: Optional[str] = None,
    include_chunk_text: bool = True,
    min_chunks: int = 1,
    expected_page_count: Optional[int] = None,
    require_page_coverage: bool = False,
) -> IndexedPaperResult:
    """Functional entry point for service/API callers."""
    return UploadedPaperIndexer(
        encoder=encoder,
        config=UploadedPaperIndexConfig(
            root_dir=Path(root_dir),
            overwrite=overwrite,
            include_chunk_text=include_chunk_text,
            min_chunks=min_chunks,
            expected_page_count=expected_page_count,
            require_page_coverage=require_page_coverage,
        ),
    ).build(
        document_id=document_id,
        chunks=chunks,
        source_document_id=source_document_id,
    )


# ---------------------------------------------------------------------------
# Production dependency construction
# ---------------------------------------------------------------------------


def build_research_index_from_settings(
    settings: Any,
    *,
    limit: Optional[int] = None,
    overwrite: bool = False,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build the research index using the existing project configuration.

    This function intentionally imports SPECTER2 and FAISS lazily so simply
    importing this module does not download models or initialize FAISS.
    """

    try:
        from backend.embeddings.specter2 import (
            SPECTER2Embedder,
        )
        from backend.retrieval.faiss_index import (
            FAISSVectorIndex,
        )
    except ImportError as exc:
        raise ResearchIndexingError(
            "Unable to import the project's embedding/FAISS "
            "implementations."
        ) from exc

    def setting(
        name: str,
        default: Any,
    ) -> Any:
        return getattr(
            settings,
            name,
            default,
        )

    dataset_path = Path(
        setting(
            "cleaned_dataset_path",
            Path("data")
            / "processed"
            / "cleaned_arxiv_dataset.csv",
        )
    )

    index_base = Path(
        setting(
            "research_index_base",
            Path("indexes")
            / "research"
            / "index",
        )
    )

    metadata_path = Path(
        index_base.parent
        / "metadata.json"
    )

    config = ResearchIndexConfig(
        dataset_path=dataset_path,
        index_base=index_base,
        batch_size=(
            batch_size
            if batch_size is not None
            else int(
                setting(
                    "specter2_batch_size",
                    16,
                )
            )
        ),
        metadata_path=metadata_path,
        limit=limit,
        overwrite=overwrite,
    )

    # Mirror the SPECTER2 configuration already used by main.py.
    encoder = SPECTER2Embedder(
        base_model=setting(
            "specter2_base_model",
            "allenai/specter2_base",
        ),
        document_adapter=setting(
            "specter2_document_adapter",
            "allenai/specter2",
        ),
        query_adapter=setting(
            "specter2_query_adapter",
            "allenai/specter2_adhoc_query",
        ),
        device=(
            device
            if device is not None
            else setting(
                "specter2_device",
                "auto",
            )
        ),
        batch_size=setting(
            "specter2_batch_size",
            16,
        ),
        max_length=setting(
            "specter2_max_length",
            512,
        ),
        normalize=setting(
            "specter2_normalize",
            True,
        ),
        output_dtype=setting(
            "specter2_output_dtype",
            "float32",
        ),
        cache_dir=setting(
            "specter2_cache_dir",
            None,
        ),
        local_files_only=setting(
            "specter2_local_files_only",
            False,
        ),
        trust_remote_code=setting(
            "specter2_trust_remote_code",
            False,
        ),
        show_progress=setting(
            "specter2_show_progress",
            True,
        ),
        deterministic=setting(
            "specter2_deterministic",
            True,
        ),
        use_autocast=setting(
            "specter2_use_autocast",
            False,
        ),
    )

    LOGGER.info(
        "Research embedding device: %s",
        getattr(encoder, "device", "unknown"),
    )
    LOGGER.info(
        "Research embedding batch size: %s",
        getattr(encoder, "batch_size", config.batch_size),
    )

    builder = ResearchIndexBuilder(
        encoder=encoder,
        faiss_index_cls=FAISSVectorIndex,
        config=config,
    )

    return builder.build()



# ---------------------------------------------------------------------------
# Mode-2 model-free self-test
# ---------------------------------------------------------------------------

def run_uploaded_paper_index_self_test() -> None:
    """
    Validate orchestration with a fake FAISS backend and the project's
    MockEmbeddingEncoder. No SPECTER2 model and no 287K-paper corpus.
    """
    from tempfile import TemporaryDirectory

    class FakeIndex:
        def __init__(
            self,
            dimension: int,
            *,
            normalize: bool = True,
            model_name: Optional[str] = None,
            model_version: Optional[str] = None,
            expected_document_count: Optional[int] = None,
        ) -> None:
            self.dimension = dimension
            self.normalize = normalize
            self.model_name = model_name
            self.model_version = model_version
            self.expected_document_count = expected_document_count
            self.document_ids: list[str] = []

        @property
        def ntotal(self) -> int:
            return len(self.document_ids)

        @property
        def index_type(self) -> str:
            return "IndexFlatIP"

        @property
        def metric(self) -> str:
            return "inner_product"

        @property
        def similarity(self) -> str:
            return "cosine"

        def add(self, embeddings: np.ndarray, document_ids: Sequence[str]) -> None:
            assert embeddings.shape == (
                len(document_ids),
                self.dimension,
            )
            self.document_ids = list(document_ids)

        def validate(self) -> None:
            assert self.ntotal == len(self.document_ids)

        @dataclass(frozen=True)
        class Manifest:
            index_type: str
            similarity: str
            normalized: bool
            embedding_dimension: int
            model_name: Optional[str]
            model_version: Optional[str]
            created_at: str

        def manifest(self) -> "FakeIndex.Manifest":
            return self.Manifest(
                index_type=self.index_type,
                similarity=self.similarity,
                normalized=self.normalize,
                embedding_dimension=self.dimension,
                model_name=self.model_name,
                model_version=self.model_version,
                created_at="test",
            )

        def save(self, path: Path) -> tuple[Path, Path]:
            index_path = Path(f"{path}.index")
            manifest_path = Path(f"{index_path}.manifest.json")
            index_path.write_bytes(b"FAKE")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "index_type": self.index_type,
                        "metric": self.metric,
                        "similarity": self.similarity,
                        "normalized": self.normalize,
                        "embedding_dimension": self.dimension,
                        "document_count": self.ntotal,
                        "document_ids": self.document_ids,
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "created_at": "test",
                        "index_file": index_path.name,
                    }
                ),
                encoding="utf-8",
            )
            return index_path, manifest_path

        @classmethod
        def load(cls, path: Path, **_: Any) -> "FakeIndex":
            if not Path(f"{path}.index").is_file():
                raise FileNotFoundError(path)

            manifest_path = Path(f"{path}.index.manifest.json")
            raw = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            restored = cls(
                dimension=int(raw["embedding_dimension"]),
                normalize=bool(raw["normalized"]),
                model_name=raw.get("model_name"),
                model_version=raw.get("model_version"),
            )
            restored.document_ids = list(
                raw["document_ids"]
            )
            return restored

    from embeddings.encoder import MockEmbeddingEncoder

    @dataclass(frozen=True)
    class Chunk:
        chunk_id: str
        document_id: str
        section_id: str
        section_type: str
        section_heading: str
        section_level: int
        text: str
        start_page: int
        end_page: int
        source_pages: tuple[int, ...]
        parent_section_id: None
        chunk_index: int
        token_count: int
        char_count: int
        overlap_with_previous: int
        chunking_method: str
        metadata: Mapping[str, Any]

    document_id = "a" * 64
    texts = (
        "Introduction explains the research problem.",
        "Related work compares previous approaches.",
        "Methodology describes the experiment.",
        "Dataset contains scientific examples.",
        "Results report retrieval performance.",
    )
    chunks = tuple(
        Chunk(
            chunk_id=f"chk_{i:03d}",
            document_id=document_id,
            section_id=f"sec_{i}",
            section_type=(
                "introduction", "related_work", "methodology",
                "dataset", "results"
            )[i],
            section_heading=None,
            section_level=1,
            text=text,
            start_page=i + 1,
            end_page=i + 1,
            source_pages=(i + 1,),
            parent_section_id=None,
            chunk_index=i,
            token_count=len(text.split()),
            char_count=len(text),
            overlap_with_previous=0,
            chunking_method="paragraph",
            metadata={"test": True},
        )
        for i, text in enumerate(texts)
    )

    with TemporaryDirectory() as tmp:
        encoder = MockEmbeddingEncoder(embedding_dimension=8)
        service = UploadedPaperIndexer(
            encoder=encoder,
            config=UploadedPaperIndexConfig(
                root_dir=Path(tmp),
            ),
            faiss_index_cls=FakeIndex,
        )

        result = service.build(
            document_id=document_id,
            chunks=chunks,
        )

        assert result.vector_count == 5
        assert result.metadata_count == 5
        assert result.embedding_dimension == 8

        target = Path(tmp) / document_id
        metadata = json.loads(
            (target / METADATA_FILENAME).read_text(
                encoding="utf-8"
            )
        )

        assert metadata["vector_count"] == 5
        assert len(metadata["chunks"]) == 5
        assert [
            record["vector_position"]
            for record in metadata["chunks"]
        ] == list(range(5))
        assert [
            record["chunk_id"]
            for record in metadata["chunks"]
        ] == [chunk.chunk_id for chunk in chunks]

        try:
            service.build(
                document_id=document_id,
                chunks=chunks,
            )
        except PaperIndexPersistenceError:
            pass
        else:
            raise AssertionError(
                "Existing index was silently overwritten."
            )

        rebuild = UploadedPaperIndexer(
            encoder=encoder,
            config=UploadedPaperIndexConfig(
                root_dir=Path(tmp),
                overwrite=True,
            ),
            faiss_index_cls=FakeIndex,
        )
        rebuilt = rebuild.build(
            document_id=document_id,
            chunks=chunks,
        )
        assert rebuilt.vector_count == 5

        try:
            service.build(
                document_id="../escape",
                chunks=chunks,
            )
        except InvalidPaperIndexingInputError:
            pass
        else:
            raise AssertionError(
                "Unsafe document ID was accepted."
            )

    print("UploadedPaperIndexer self-test: PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the AI Research Paper Assistant "
            "Mode-1 SPECTER2 + FAISS research index."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Research CSV path. Defaults to the configured "
            "processed research dataset."
        ),
    )

    parser.add_argument(
        "--index-base",
        type=Path,
        default=None,
        help=(
            "FAISS index base. Example: "
            "indexes/research/index"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of papers embedded per batch.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help=(
            "Embedding device override. Use 'cuda' for NVIDIA GPU, "
            "'cpu' for CPU, or 'auto' to let SPECTER2 choose."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N papers. "
            "Useful for smoke testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing final artifacts.",
    )

    parser.add_argument(
        "--id-column",
        default=None,
        help="Explicit document-ID CSV column.",
    )

    parser.add_argument(
        "--title-column",
        default=None,
        help="Explicit title CSV column.",
    )

    parser.add_argument(
        "--abstract-column",
        default=None,
        help="Explicit abstract CSV column.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        help="Logging level.",
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    try:
        from backend.config import get_settings

        settings = get_settings()

        dataset_path = (
            args.dataset
            if args.dataset is not None
            else Path(
                getattr(
                    settings,
                    "cleaned_dataset_path",
                    Path("data")
                    / "processed"
                    / "cleaned_arxiv_dataset.csv",
                )
            )
        )

        index_base = (
            args.index_base
            if args.index_base is not None
            else Path(
                getattr(
                    settings,
                    "research_index_base",
                    Path("indexes")
                    / "research"
                    / "index",
                )
            )
        )

        metadata_path = (
            index_base.parent
            / "metadata.json"
        )

        from backend.embeddings.specter2 import (
            SPECTER2Embedder,
        )
        from backend.retrieval.faiss_index import (
            FAISSVectorIndex,
        )

        encoder = SPECTER2Embedder(
            base_model=getattr(
                settings,
                "specter2_base_model",
                "allenai/specter2_base",
            ),
            document_adapter=getattr(
                settings,
                "specter2_document_adapter",
                "allenai/specter2",
            ),
            query_adapter=getattr(
                settings,
                "specter2_query_adapter",
                "allenai/specter2_adhoc_query",
            ),
            device=(
                args.device
                if args.device is not None
                else getattr(
                    settings,
                    "specter2_device",
                    "auto",
                )
            ),
            batch_size=(
                args.batch_size
                if args.batch_size is not None
                else getattr(
                    settings,
                    "specter2_batch_size",
                    16,
                )
            ),
            max_length=getattr(
                settings,
                "specter2_max_length",
                512,
            ),
            normalize=getattr(
                settings,
                "specter2_normalize",
                True,
            ),
            output_dtype=getattr(
                settings,
                "specter2_output_dtype",
                "float32",
            ),
            cache_dir=getattr(
                settings,
                "specter2_cache_dir",
                None,
            ),
            local_files_only=getattr(
                settings,
                "specter2_local_files_only",
                False,
            ),
            trust_remote_code=getattr(
                settings,
                "specter2_trust_remote_code",
                False,
            ),
            show_progress=getattr(
                settings,
                "specter2_show_progress",
                True,
            ),
            deterministic=getattr(
                settings,
                "specter2_deterministic",
                True,
            ),
            use_autocast=getattr(
                settings,
                "specter2_use_autocast",
                False,
            ),
        )

        config = ResearchIndexConfig(
            dataset_path=dataset_path,
            index_base=index_base,
            batch_size=(
                args.batch_size
                if args.batch_size is not None
                else getattr(
                    settings,
                    "specter2_batch_size",
                    16,
                )
            ),
            metadata_path=metadata_path,
            limit=args.limit,
            id_column=args.id_column,
            title_column=args.title_column,
            abstract_column=args.abstract_column,
            overwrite=args.overwrite,
        )

        LOGGER.info(
            "Research embedding device: %s",
            getattr(encoder, "device", "unknown"),
        )
        LOGGER.info(
            "Research embedding batch size: %s",
            getattr(encoder, "batch_size", config.batch_size),
        )

        builder = ResearchIndexBuilder(
            encoder=encoder,
            faiss_index_cls=FAISSVectorIndex,
            config=config,
        )

        statistics = builder.build()

        print(
            json.dumps(
                statistics,
                indent=2,
                ensure_ascii=False,
            )
        )

        return 0

    except Exception:
        LOGGER.exception(
            "Research index build failed."
        )
        return 1


# ---------------------------------------------------------------------------
# Lightweight self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """
    Model-free structural self-test.

    This deliberately does NOT:
        - download SPECTER2
        - load the 433 MB corpus
        - initialize real FAISS

    It verifies the dataset/text contract.
    """

    columns = [
        "id",
        "title",
        "abstract",
        "categories",
    ]

    mapping = resolve_column_mapping(columns)

    assert mapping.document_id == "id"
    assert mapping.title == "title"
    assert mapping.abstract == "abstract"

    record = ResearchRecord(
        document_id="paper-1",
        title="Deep Learning",
        abstract="A scientific abstract.",
    )

    assert record.canonical_text == (
        "Title:\nDeep Learning\n\n"
        "Abstract:\nA scientific abstract."
    )

    try:
        resolve_column_mapping(
            ["foo", "bar"],
        )
    except ResearchDatasetSchemaError:
        pass
    else:
        raise AssertionError(
            "Invalid dataset schema was accepted."
        )

    # Configuration invariants.
    config = ResearchIndexConfig(
        dataset_path=Path("data/processed/cleaned_arxiv_dataset.csv"),
        index_base=Path("indexes/research/index"),
        batch_size=16,
    )
    assert config.batch_size > 0
    assert config.dataset_path.suffix.lower() == ".csv"

    # Empty semantic text must never be accepted.
    try:
        ResearchRecord(
            document_id="paper-empty",
            title="",
            abstract="",
        )
        # Construction itself is intentionally lightweight; the reader/build
        # layer is responsible for rejecting empty semantic text.
    except Exception as exc:
        raise AssertionError(
            f"Unexpected ResearchRecord construction failure: {exc}"
        ) from exc

    print(
        "ResearchIndexBuilder self-test: PASSED"
    )
if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.argv.remove("--self-test")
        run_self_test()
        run_uploaded_paper_index_self_test()
        raise SystemExit(0)

    raise SystemExit(main())