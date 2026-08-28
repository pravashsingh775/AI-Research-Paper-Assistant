"""
Offline research-corpus FAISS index builder.

AI Research Paper Assistant
===========================

Pipeline
--------
cleaned_arxiv_dataset.csv
        |
        v
id + title + summary
        |
        v
canonical scientific-paper text
        |
        v
SPECTER2 document embeddings
        |
        v
FAISSVectorIndex
        |
        +--> indexes/research/index.index
        +--> indexes/research/index.index.manifest.json
        +--> indexes/research/metadata.json

Responsibilities
----------------
This module ONLY builds the persistent Mode-1 research index.

It does NOT:
    - serve HTTP requests
    - perform query retrieval
    - rerank papers
    - call an LLM
    - analyze papers
    - process PDFs
    - implement SPECTER2 internals
    - implement FAISS internals

Existing project implementations remain the source of truth for:
    - SPECTER2 embeddings
    - FAISS persistence/search

Dataset contract
----------------
The current project dataset is:

    data/processed/cleaned_arxiv_dataset.csv

Required semantic fields:

    id
    title
    summary

The builder also supports safe aliases for backwards compatibility.

Python compatibility
--------------------
The project currently uses Python 3.10.
This module therefore avoids Python 3.11-only features.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence


import numpy as np


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ResearchIndexingError(RuntimeError):
    """Base exception for research-index building failures."""


class ResearchDatasetError(ResearchIndexingError):
    """Raised when the research corpus is invalid or unusable."""


class ResearchDatasetSchemaError(ResearchDatasetError):
    """Raised when required dataset columns cannot be identified."""


class ResearchIndexValidationError(ResearchIndexingError):
    """Raised when index-building invariants are violated."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ResearchIndexConfig:
    """
    Configuration for the offline research index builder.

    Defaults intentionally match the current project structure.
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

    metadata_path: Optional[Path] = (
        Path("indexes")
        / "research"
        / "metadata.json"
    )

    # Number of papers embedded per batch.
    batch_size: int = 16

    # None = complete corpus.
    limit: Optional[int] = None

    # Explicit real-project dataset schema.
    id_column: Optional[str] = "id"
    title_column: Optional[str] = "title"
    abstract_column: Optional[str] = "summary"

    # Do not overwrite a valid final index unless explicitly requested.
    overwrite: bool = False


# ============================================================================
# Dataset schema
# ============================================================================


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
        Canonical semantic representation supplied to SPECTER2.

        SPECTER2 is designed around scientific-paper title + abstract text.
        """

        title = self.title.strip()
        abstract = self.abstract.strip()

        if title and abstract:
            return (
                f"Title:\n{title}\n\n"
                f"Abstract:\n{abstract}"
            )

        if title:
            return f"Title:\n{title}"

        return f"Abstract:\n{abstract}"


# ============================================================================
# Column aliases
# ============================================================================


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
    # Current project field comes first.
    "summary",
    "abstract",
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
        resolved = normalized.get(
            _normalise_column_name(alias)
        )

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
    id_column: Optional[str] = "id",
    title_column: Optional[str] = "title",
    abstract_column: Optional[str] = "summary",
) -> ResearchColumnMapping:
    """
    Resolve the three semantic index columns.

    Explicit configuration is preferred. Safe aliases are used only when
    an explicit value is not supplied.
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
        "summary/abstract",
    )

    if len({document_id, title, abstract}) != 3:
        raise ResearchDatasetSchemaError(
            "Document ID, title, and summary/abstract columns "
            "must be distinct."
        )

    return ResearchColumnMapping(
        document_id=document_id,
        title=title,
        abstract=abstract,
    )


# ============================================================================
# Dataset reader
# ============================================================================


class ResearchDatasetReader:
    """
    Memory-conscious CSV reader.

    The project dataset is hundreds of MB, so the entire CSV is never loaded
    into a pandas DataFrame or Python list.
    """

    def __init__(
        self,
        path: Path,
        *,
        mapping: Optional[ResearchColumnMapping] = None,
        id_column: Optional[str] = "id",
        title_column: Optional[str] = "title",
        abstract_column: Optional[str] = "summary",
        encoding: str = "utf-8-sig",
    ) -> None:

        self.path = Path(path)
        self.mapping = mapping
        self.id_column = id_column
        self.title_column = title_column
        self.abstract_column = abstract_column
        self.encoding = encoding

    def _open(self):
        """Open the dataset safely."""

        if not self.path.is_file():
            raise ResearchDatasetError(
                f"Research dataset does not exist: {self.path}"
            )

        size = self.path.stat().st_size

        if size <= 0:
            raise ResearchDatasetError(
                f"Research dataset is empty: {self.path}"
            )

        LOGGER.info(
            "Opening research dataset: %s (%.2f MB)",
            self.path,
            size / (1024 * 1024),
        )

        return self.path.open(
            "r",
            encoding=self.encoding,
            newline="",
        )

    def resolve_mapping(self) -> ResearchColumnMapping:
        """Read only the CSV header and resolve the schema."""

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
            if column is not None
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

        if limit is not None and limit <= 0:
            raise ValueError(
                "limit must be greater than zero when provided."
            )

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

            for required in (
                mapping.document_id,
                mapping.title,
                mapping.abstract,
            ):
                if required not in actual_fields:
                    raise ResearchDatasetSchemaError(
                        f"Resolved column {required!r} is absent "
                        f"from the CSV header."
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
                        "Paper has no usable title or summary/abstract "
                        f"at CSV row {row_number}; "
                        f"document_id={document_id!r}."
                    )

                yield ResearchRecord(
                    document_id=document_id,
                    title=title,
                    abstract=abstract,
                )

                processed += 1


def _clean_text(value: Any) -> str:
    """Convert CSV values into normalized strings."""

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Remove accidental surrounding whitespace.
    return value.strip()


# ============================================================================
# Metadata writer
# ============================================================================


class ResearchMetadataWriter:
    """
    Atomically write retrieval metadata.

    The final JSON contract remains:

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

    Metadata is written incrementally so the builder does not have to keep
    every ResearchRecord in RAM.
    """

    SCHEMA_VERSION = "1"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

        self._temporary: Optional[Path] = None
        self._handle = None
        self._count = 0
        self._first = True

    def begin(self) -> None:
        """Start an atomic metadata transaction."""

        if self._handle is not None:
            raise ResearchIndexingError(
                "Metadata writer transaction already started."
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )

        os.close(fd)

        self._temporary = Path(temporary_name)

        try:
            self._handle = self._temporary.open(
                "w",
                encoding="utf-8",
                newline="\n",
            )

            self._handle.write(
                '{\n'
                f'  "schema_version": {json.dumps(self.SCHEMA_VERSION)},\n'
                '  "document_count": '
            )

            # Placeholder is replaced during finalize.
            self._handle.write("0")
            self._handle.write(",\n")
            self._handle.write('  "documents": [\n')

        except Exception:
            self.abort()
            raise

    def append_batch(
        self,
        records: Sequence[ResearchRecord],
    ) -> None:
        """Append a validated batch of metadata."""

        if self._handle is None:
            raise ResearchIndexingError(
                "Metadata writer has not been started."
            )

        for record in records:

            payload = {
                "document_id": record.document_id,
                "title": record.title,
                "summary": record.abstract,
            }

            if not self._first:
                self._handle.write(",\n")

            self._handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )

            self._first = False
            self._count += 1

    def finalize(self) -> Path:
        """Finish and atomically replace the final metadata file."""

        if self._handle is None or self._temporary is None:
            raise ResearchIndexingError(
                "Metadata writer has not been started."
            )

        temporary = self._temporary

        try:
            self._handle.write("\n")
            self._handle.write("  ]\n")
            self._handle.write("}\n")

            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

            # The initial implementation writes document_count as zero.
            # Replace only that exact field in the temporary file before
            # final atomic rename.
            text = temporary.read_text(
                encoding="utf-8"
            )

            marker = (
                f'"document_count": 0'
            )

            replacement = (
                f'"document_count": {self._count}'
            )

            if marker not in text:
                raise ResearchIndexValidationError(
                    "Metadata document-count placeholder was not found."
                )

            text = text.replace(
                marker,
                replacement,
                1,
            )

            temporary.write_text(
                text,
                encoding="utf-8",
                newline="\n",
            )

            if temporary.stat().st_size <= 0:
                raise ResearchIndexValidationError(
                    "Metadata writer produced an empty file."
                )

            os.replace(
                temporary,
                self.path,
            )

            self._temporary = None

            return self.path

        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Abort an unfinished metadata transaction."""

        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass

            self._handle = None

        if self._temporary is not None:
            try:
                self._temporary.unlink(
                    missing_ok=True
                )
            except OSError:
                LOGGER.warning(
                    "Unable to remove temporary metadata file: %s",
                    self._temporary,
                    exc_info=True,
                )

            self._temporary = None


# ============================================================================
# Research index builder
# ============================================================================


class ResearchIndexBuilder:
    """
    Build the persistent Mode-1 research FAISS index.

    Dependencies are injected so this class does not implement SPECTER2 or
    FAISS itself.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        faiss_index_cls: Any,
        config: Optional[ResearchIndexConfig] = None,
    ) -> None:

        if encoder is None:
            raise ValueError(
                "encoder cannot be None."
            )

        if faiss_index_cls is None:
            raise ValueError(
                "faiss_index_cls cannot be None."
            )

        self.encoder = encoder
        self.faiss_index_cls = faiss_index_cls
        self.config = (
            config
            if config is not None
            else ResearchIndexConfig()
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Validate builder configuration."""

        config = self.config

        if config.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if config.limit is not None and config.limit <= 0:
            raise ValueError(
                "limit must be greater than zero when provided."
            )

        if config.dataset_path.suffix.lower() != ".csv":
            raise ResearchDatasetError(
                "The current research index builder expects a CSV corpus. "
                f"Received: {config.dataset_path}"
            )

        if not config.dataset_path.is_file():
            raise ResearchDatasetError(
                "Configured research dataset does not exist: "
                f"{config.dataset_path}"
            )

        if config.index_base.name.strip() == "":
            raise ValueError(
                "index_base must contain a valid filename."
            )

        if config.metadata_path is not None:
            if config.metadata_path.suffix.lower() != ".json":
                raise ValueError(
                    "metadata_path must point to a .json file."
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

        if not hasattr(
            self.encoder,
            "embedding_dimension",
        ):
            raise ResearchIndexValidationError(
                "SPECTER2 encoder does not expose "
                "embedding_dimension."
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
    # Artifact paths
    # ------------------------------------------------------------------

    def _artifact_paths(self) -> tuple[Path, Path]:
        """
        Return the exact artifacts expected by backend.main.
        """

        base = Path(self.config.index_base)

        index_path = Path(
            f"{base}.index"
        )

        manifest_path = Path(
            f"{base}.index.manifest.json"
        )

        return index_path, manifest_path

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """
        Build the complete research index.

        Final artifacts are accepted only after all integrity checks pass.
        """

        self.validate_config()

        config = self.config

        final_index_path, final_manifest_path = (
            self._artifact_paths()
        )

        # --------------------------------------------------------------
        # Existing artifact protection
        # --------------------------------------------------------------

        if not config.overwrite:

            if final_index_path.exists():
                raise ResearchIndexingError(
                    "Research FAISS index already exists: "
                    f"{final_index_path}. "
                    "Use --overwrite to rebuild it."
                )

            if final_manifest_path.exists():
                raise ResearchIndexingError(
                    "Research FAISS manifest already exists: "
                    f"{final_manifest_path}. "
                    "Use --overwrite to rebuild it."
                )

            if (
                config.metadata_path is not None
                and config.metadata_path.exists()
            ):
                raise ResearchIndexingError(
                    "Research metadata already exists: "
                    f"{config.metadata_path}. "
                    "Use --overwrite to rebuild it."
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
            "Resolved columns: id=%s, title=%s, summary=%s",
            mapping.document_id,
            mapping.title,
            mapping.abstract,
        )

        LOGGER.info(
            "Index base: %s",
            config.index_base,
        )

        # --------------------------------------------------------------
        # Build state
        # --------------------------------------------------------------

        index = None

        seen_ids: set[str] = set()

        total_documents = 0
        total_batches = 0

        batch_records: list[ResearchRecord] = []

        metadata_writer: Optional[
            ResearchMetadataWriter
        ] = None

        if config.metadata_path is not None:
            metadata_writer = ResearchMetadataWriter(
                config.metadata_path
            )

            metadata_writer.begin()

        # --------------------------------------------------------------
        # Batch flush
        # --------------------------------------------------------------

        def flush_batch() -> None:
            nonlocal index
            nonlocal total_documents
            nonlocal total_batches

            if not batch_records:
                return

            records = list(
                batch_records
            )

            texts = [
                record.canonical_text
                for record in records
            ]

            document_ids = [
                record.document_id
                for record in records
            ]

            batch_number = (
                total_batches + 1
            )

            LOGGER.info(
                "Embedding research batch %d: %d papers",
                batch_number,
                len(records),
            )

            try:
                embeddings = (
                    self.encoder.embed_documents(
                        texts
                    )
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "SPECTER2 document embedding failed for "
                    f"batch {batch_number}."
                ) from exc

            matrix = (
                self._validate_embedding_batch(
                    embeddings,
                    len(records),
                )
            )

            # ----------------------------------------------------------
            # Create FAISS index from first embedding batch
            # ----------------------------------------------------------

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

                normalized = getattr(
                    self.encoder,
                    "normalized",
                    None,
                )

                if normalized is None:
                    normalized = True

                try:
                    index = self.faiss_index_cls(
                        matrix.shape[1],
                        normalize=bool(
                            normalized
                        ),
                        model_name=model_name,
                        model_version=model_version,
                    )
                except Exception as exc:
                    raise ResearchIndexingError(
                        "Failed to initialize the project's "
                        "FAISSVectorIndex."
                    ) from exc

            # ----------------------------------------------------------
            # Add vectors
            # ----------------------------------------------------------

            try:
                index.add(
                    matrix,
                    document_ids,
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "Failed to add research embedding batch "
                    f"{batch_number} to FAISS."
                ) from exc

            # ----------------------------------------------------------
            # Metadata
            # ----------------------------------------------------------

            if metadata_writer is not None:
                try:
                    metadata_writer.append_batch(
                        records
                    )
                except Exception as exc:
                    raise ResearchIndexingError(
                        "Failed to append research metadata "
                        f"for batch {batch_number}."
                    ) from exc

            total_documents += len(
                records
            )

            total_batches += 1

            LOGGER.info(
                "Research index progress: "
                "%d papers indexed",
                total_documents,
            )

            batch_records.clear()

        # --------------------------------------------------------------
        # Stream dataset
        # --------------------------------------------------------------

        try:

            for record in reader.iter_records(
                limit=config.limit,
            ):

                if record.document_id in seen_ids:
                    raise ResearchDatasetError(
                        "Duplicate document ID encountered: "
                        f"{record.document_id!r}"
                    )

                seen_ids.add(
                    record.document_id
                )

                batch_records.append(
                    record
                )

                if (
                    len(batch_records)
                    >= config.batch_size
                ):
                    flush_batch()

            # Final partial batch.
            flush_batch()

            if index is None or total_documents == 0:
                raise ResearchIndexingError(
                    "No valid research papers were indexed."
                )

            # ----------------------------------------------------------
            # Final FAISS integrity checks
            # ----------------------------------------------------------

            if index.ntotal != total_documents:
                raise ResearchIndexValidationError(
                    "Final FAISS/document count mismatch: "
                    f"FAISS={index.ntotal}, "
                    f"records={total_documents}."
                )

            if not hasattr(
                index,
                "document_ids",
            ):
                raise ResearchIndexValidationError(
                    "FAISS index does not expose document_ids."
                )

            if len(index.document_ids) != total_documents:
                raise ResearchIndexValidationError(
                    "Final FAISS document-ID mapping mismatch: "
                    f"IDs={len(index.document_ids)}, "
                    f"records={total_documents}."
                )

            try:
                index.validate()
            except Exception as exc:
                raise ResearchIndexValidationError(
                    "Final FAISS integrity validation failed."
                ) from exc

            # ----------------------------------------------------------
            # Ensure output directory exists
            # ----------------------------------------------------------

            config.index_base.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------
            # Persist FAISS
            # ----------------------------------------------------------

            try:
                index_path, manifest_path = (
                    index.save(
                        config.index_base
                    )
                )
            except Exception as exc:
                raise ResearchIndexingError(
                    "Failed to persist the final research "
                    "FAISS index."
                ) from exc

            index_path = Path(
                index_path
            )

            manifest_path = Path(
                manifest_path
            )

            # ----------------------------------------------------------
            # Verify FAISS artifacts
            # ----------------------------------------------------------

            if (
                not index_path.is_file()
                or index_path.stat().st_size <= 0
            ):
                raise ResearchIndexValidationError(
                    "Final FAISS index is missing or empty: "
                    f"{index_path}"
                )

            if (
                not manifest_path.is_file()
                or manifest_path.stat().st_size <= 0
            ):
                raise ResearchIndexValidationError(
                    "Final FAISS manifest is missing or empty: "
                    f"{manifest_path}"
                )

            # ----------------------------------------------------------
            # Finalize metadata
            # ----------------------------------------------------------

            metadata_path = None

            if metadata_writer is not None:

                try:
                    metadata_path = (
                        metadata_writer.finalize()
                    )
                except Exception as exc:
                    raise ResearchIndexingError(
                        "FAISS index was created, but research "
                        "metadata could not be finalized."
                    ) from exc

                if (
                    not metadata_path.is_file()
                    or metadata_path.stat().st_size <= 0
                ):
                    raise ResearchIndexValidationError(
                        "Final research metadata is missing "
                        "or empty: "
                        f"{metadata_path}"
                    )

                # Verify the metadata count without loading the
                # entire file into RAM.
                _validate_metadata_file(
                    metadata_path,
                    expected_count=total_documents,
                )

            elapsed = (
                time.perf_counter()
                - started_at
            )

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
                "index_type": getattr(
                    index,
                    "index_type",
                    "IndexFlatIP",
                ),
                "metric": getattr(
                    index,
                    "metric",
                    "inner_product",
                ),
                "similarity": getattr(
                    index,
                    "similarity",
                    "cosine",
                ),
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
                "batch_size": config.batch_size,
                "limit": config.limit,
                "elapsed_seconds": round(
                    elapsed,
                    3,
                ),
            }

            LOGGER.info(
                "Research FAISS index build completed: "
                "%d papers | dimension=%d | batches=%d | "
                "elapsed=%.2fs",
                total_documents,
                index.dimension,
                total_batches,
                elapsed,
            )

            LOGGER.info(
                "FAISS index: %s",
                index_path,
            )

            LOGGER.info(
                "FAISS manifest: %s",
                manifest_path,
            )

            if metadata_path is not None:
                LOGGER.info(
                    "Metadata: %s",
                    metadata_path,
                )

            return statistics

        except Exception:

            if metadata_writer is not None:
                metadata_writer.abort()

            raise


# ============================================================================
# Metadata validation
# ============================================================================


def _validate_metadata_file(
    path: Path,
    *,
    expected_count: int,
) -> None:
    """
    Validate the final metadata JSON.

    The metadata file is expected to contain the complete document list.
    This validator uses the standard JSON parser for correctness.
    """

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

    except Exception as exc:
        raise ResearchIndexValidationError(
            "Research metadata JSON is invalid: "
            f"{path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ResearchIndexValidationError(
            "Research metadata root must be a JSON object."
        )

    if payload.get(
        "schema_version"
    ) != ResearchMetadataWriter.SCHEMA_VERSION:
        raise ResearchIndexValidationError(
            "Unsupported research metadata schema version: "
            f"{payload.get('schema_version')!r}"
        )

    document_count = payload.get(
        "document_count"
    )

    documents = payload.get(
        "documents"
    )

    if not isinstance(
        document_count,
        int,
    ):
        raise ResearchIndexValidationError(
            "Research metadata document_count must be an integer."
        )

    if document_count != expected_count:
        raise ResearchIndexValidationError(
            "Research metadata/document count mismatch: "
            f"metadata={document_count}, "
            f"expected={expected_count}."
        )

    if not isinstance(
        documents,
        list,
    ):
        raise ResearchIndexValidationError(
            "Research metadata documents must be a list."
        )

    if len(documents) != expected_count:
        raise ResearchIndexValidationError(
            "Research metadata list length mismatch: "
            f"documents={len(documents)}, "
            f"expected={expected_count}."
        )

    previous_id: Optional[str] = None

    for position, document in enumerate(
        documents
    ):

        if not isinstance(
            document,
            dict,
        ):
            raise ResearchIndexValidationError(
                "Research metadata document at position "
                f"{position} is not an object."
            )

        document_id = document.get(
            "document_id"
        )

        if not isinstance(
            document_id,
            str,
        ) or not document_id.strip():
            raise ResearchIndexValidationError(
                "Research metadata contains an invalid "
                f"document_id at position {position}."
            )

        if (
            previous_id is not None
            and document_id == previous_id
        ):
            raise ResearchIndexValidationError(
                "Research metadata contains consecutive "
                f"duplicate document ID: {document_id!r}"
            )

        previous_id = document_id


# ============================================================================
# Production dependency construction
# ============================================================================


def _setting(
    settings: Any,
    name: str,
    default: Any,
) -> Any:
    """
    Safely read a setting.

    Pydantic Settings objects support normal attribute access.
    """

    value = getattr(
        settings,
        name,
        default,
    )

    # Empty strings should not silently override valid defaults for
    # model/path configuration.
    if isinstance(
        value,
        str,
    ) and not value.strip():
        return default

    return value


def build_research_index_from_settings(
    settings: Any,
    *,
    limit: Optional[int] = None,
    overwrite: bool = False,
    batch_size: Optional[int] = None,
) -> dict[str, Any]:
    """
    Build the research index using the existing project configuration.

    Imports SPECTER2 and FAISS lazily so importing this module itself does not
    download models or initialize FAISS.
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
            "Unable to import the project's SPECTER2/FAISS "
            "implementations."
        ) from exc

    # ------------------------------------------------------------------
    # Dataset path
    # ------------------------------------------------------------------

    dataset_path = Path(
        _setting(
            settings,
            "cleaned_dataset_path",
            Path("data")
            / "processed"
            / "cleaned_arxiv_dataset.csv",
        )
    )

    # ------------------------------------------------------------------
    # Index path
    # ------------------------------------------------------------------

    index_base = Path(
        _setting(
            settings,
            "research_index_base",
            Path("indexes")
            / "research"
            / "index",
        )
    )

    # ------------------------------------------------------------------
    # Metadata path
    # ------------------------------------------------------------------

    metadata_path = Path(
        _setting(
            settings,
            "research_metadata_path",
            index_base.parent
            / "metadata.json",
        )
    )

    # ------------------------------------------------------------------
    # Batch size
    # ------------------------------------------------------------------

    effective_batch_size = (
        batch_size
        if batch_size is not None
        else int(
            _setting(
                settings,
                "specter2_batch_size",
                16,
            )
        )
    )

    # ------------------------------------------------------------------
    # Builder configuration
    # ------------------------------------------------------------------

    config = ResearchIndexConfig(
        dataset_path=dataset_path,
        index_base=index_base,
        metadata_path=metadata_path,
        batch_size=effective_batch_size,
        limit=limit,
        id_column="id",
        title_column="title",
        abstract_column="summary",
        overwrite=overwrite,
    )

    # ------------------------------------------------------------------
    # SPECTER2
    # ------------------------------------------------------------------

    encoder = SPECTER2Embedder(
        base_model=_setting(
            settings,
            "specter2_base_model",
            "allenai/specter2_base",
        ),
        document_adapter=_setting(
            settings,
            "specter2_document_adapter",
            "allenai/specter2",
        ),
        query_adapter=_setting(
            settings,
            "specter2_query_adapter",
            "allenai/specter2_adhoc_query",
        ),
        device=_setting(
            settings,
            "specter2_device",
            "auto",
        ),
        batch_size=effective_batch_size,
        max_length=int(
            _setting(
                settings,
                "specter2_max_length",
                512,
            )
        ),
        normalize=bool(
            _setting(
                settings,
                "specter2_normalize",
                True,
            )
        ),
        output_dtype=_setting(
            settings,
            "specter2_output_dtype",
            "float32",
        ),
        cache_dir=_setting(
            settings,
            "specter2_cache_dir",
            None,
        ),
        local_files_only=bool(
            _setting(
                settings,
                "specter2_local_files_only",
                False,
            )
        ),
        trust_remote_code=bool(
            _setting(
                settings,
                "specter2_trust_remote_code",
                False,
            )
        ),
        show_progress=bool(
            _setting(
                settings,
                "specter2_show_progress",
                True,
            )
        ),
        deterministic=bool(
            _setting(
                settings,
                "specter2_deterministic",
                True,
            )
        ),
        use_autocast=bool(
            _setting(
                settings,
                "specter2_use_autocast",
                False,
            )
        ),
    )

    builder = ResearchIndexBuilder(
        encoder=encoder,
        faiss_index_cls=FAISSVectorIndex,
        config=config,
    )

    return builder.build()


# ============================================================================
# CLI
# ============================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the AI Research Paper Assistant "
            "Mode-1 SPECTER2 + FAISS research index."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run a model-free structural self-test and exit."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Research CSV path. Defaults to "
            "settings.cleaned_dataset_path."
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
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Metadata JSON path. Defaults to "
            "<index directory>/metadata.json."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Number of papers embedded per batch."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N papers. "
            "Recommended for smoke testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of existing final artifacts."
        ),
    )

    parser.add_argument(
        "--id-column",
        default="id",
        help="Document-ID CSV column.",
    )

    parser.add_argument(
        "--title-column",
        default="title",
        help="Title CSV column.",
    )

    parser.add_argument(
        "--summary-column",
        "--abstract-column",
        dest="summary_column",
        default="summary",
        help=(
            "Summary/abstract CSV column. "
            "The current dataset uses 'summary'."
        ),
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


# ============================================================================
# Self-test
# ============================================================================


def run_self_test() -> None:
    """
    Model-free structural self-test.

    This test does NOT:
        - download SPECTER2
        - load the 433 MB corpus
        - initialize FAISS

    It validates the actual project dataset contract.
    """

    # --------------------------------------------------------------
    # Real project schema
    # --------------------------------------------------------------

    columns = [
        "id",
        "title",
        "category",
        "category_code",
        "published_date",
        "updated_date",
        "authors",
        "first_author",
        "summary",
        "summary_word_count",
        "standardized_date",
    ]

    mapping = resolve_column_mapping(
        columns,
        id_column="id",
        title_column="title",
        abstract_column="summary",
    )

    assert mapping.document_id == "id"
    assert mapping.title == "title"
    assert mapping.abstract == "summary"

    # --------------------------------------------------------------
    # Canonical text
    # --------------------------------------------------------------

    record = ResearchRecord(
        document_id="arxiv-1",
        title="Deep Learning",
        abstract="A scientific abstract.",
    )

    assert record.canonical_text == (
        "Title:\nDeep Learning\n\n"
        "Abstract:\nA scientific abstract."
    )

    # --------------------------------------------------------------
    # Abstract-only paper
    # --------------------------------------------------------------

    abstract_only = ResearchRecord(
        document_id="arxiv-2",
        title="",
        abstract="Only a summary.",
    )

    assert abstract_only.canonical_text == (
        "Abstract:\nOnly a summary."
    )

    # --------------------------------------------------------------
    # Title-only paper
    # --------------------------------------------------------------

    title_only = ResearchRecord(
        document_id="arxiv-3",
        title="Only a title",
        abstract="",
    )

    assert title_only.canonical_text == (
        "Title:\nOnly a title"
    )

    # --------------------------------------------------------------
    # Invalid schema
    # --------------------------------------------------------------

    try:
        resolve_column_mapping(
            ["foo", "bar"],
            id_column=None,
            title_column=None,
            abstract_column=None,
        )

    except ResearchDatasetSchemaError:
        pass

    else:
        raise AssertionError(
            "Invalid dataset schema was accepted."
        )

    # --------------------------------------------------------------
    # Duplicate semantic fields
    # --------------------------------------------------------------

    try:
        resolve_column_mapping(
            ["id", "title"],
            id_column="id",
            title_column="id",
            abstract_column="title",
        )

    except ResearchDatasetSchemaError:
        pass

    else:
        raise AssertionError(
            "Duplicate semantic columns were accepted."
        )

    print(
        "ResearchIndexBuilder self-test: PASSED"
    )


# ============================================================================
# CLI entry point
# ============================================================================


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

    if args.self_test:
        run_self_test()
        return 0

    try:

        from backend.config import (
            get_settings,
        )

        settings = get_settings()

        # --------------------------------------------------------------
        # Resolve dataset
        # --------------------------------------------------------------

        dataset_path = (
            args.dataset
            if args.dataset is not None
            else Path(
                _setting(
                    settings,
                    "cleaned_dataset_path",
                    Path("data")
                    / "processed"
                    / "cleaned_arxiv_dataset.csv",
                )
            )
        )

        # --------------------------------------------------------------
        # Resolve index
        # --------------------------------------------------------------

        index_base = (
            args.index_base
            if args.index_base is not None
            else Path(
                _setting(
                    settings,
                    "research_index_base",
                    Path("indexes")
                    / "research"
                    / "index",
                )
            )
        )

        # --------------------------------------------------------------
        # Resolve metadata
        # --------------------------------------------------------------

        metadata_path = (
            args.metadata
            if args.metadata is not None
            else Path(
                _setting(
                    settings,
                    "research_metadata_path",
                    index_base.parent
                    / "metadata.json",
                )
            )
        )

        # --------------------------------------------------------------
        # Resolve batch size
        # --------------------------------------------------------------

        effective_batch_size = (
            args.batch_size
            if args.batch_size is not None
            else int(
                _setting(
                    settings,
                    "specter2_batch_size",
                    16,
                )
            )
        )

        # --------------------------------------------------------------
        # Lazy imports
        # --------------------------------------------------------------

        from backend.embeddings.specter2 import (
            SPECTER2Embedder,
        )

        from backend.retrieval.faiss_index import (
            FAISSVectorIndex,
        )

        # --------------------------------------------------------------
        # Encoder
        # --------------------------------------------------------------

        encoder = SPECTER2Embedder(
            base_model=_setting(
                settings,
                "specter2_base_model",
                "allenai/specter2_base",
            ),
            document_adapter=_setting(
                settings,
                "specter2_document_adapter",
                "allenai/specter2",
            ),
            query_adapter=_setting(
                settings,
                "specter2_query_adapter",
                "allenai/specter2_adhoc_query",
            ),
            device=_setting(
                settings,
                "specter2_device",
                "auto",
            ),
            batch_size=effective_batch_size,
            max_length=int(
                _setting(
                    settings,
                    "specter2_max_length",
                    512,
                )
            ),
            normalize=bool(
                _setting(
                    settings,
                    "specter2_normalize",
                    True,
                )
            ),
            output_dtype=_setting(
                settings,
                "specter2_output_dtype",
                "float32",
            ),
            cache_dir=_setting(
                settings,
                "specter2_cache_dir",
                None,
            ),
            local_files_only=bool(
                _setting(
                    settings,
                    "specter2_local_files_only",
                    False,
                )
            ),
            trust_remote_code=bool(
                _setting(
                    settings,
                    "specter2_trust_remote_code",
                    False,
                )
            ),
            show_progress=bool(
                _setting(
                    settings,
                    "specter2_show_progress",
                    True,
                )
            ),
            deterministic=bool(
                _setting(
                    settings,
                    "specter2_deterministic",
                    True,
                )
            ),
            use_autocast=bool(
                _setting(
                    settings,
                    "specter2_use_autocast",
                    False,
                )
            ),
        )

        # --------------------------------------------------------------
        # Builder
        # --------------------------------------------------------------

        config = ResearchIndexConfig(
            dataset_path=dataset_path,
            index_base=index_base,
            metadata_path=metadata_path,
            batch_size=effective_batch_size,
            limit=args.limit,
            id_column=args.id_column,
            title_column=args.title_column,
            abstract_column=args.summary_column,
            overwrite=args.overwrite,
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

    except KeyboardInterrupt:

        LOGGER.error(
            "Research index build interrupted by user."
        )

        return 130

    except Exception:

        LOGGER.exception(
            "Research index build failed."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )