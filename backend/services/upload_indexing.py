"""
Mode-2 uploaded-paper ingestion and indexing pipeline.

Responsibilities
----------------
This service is the application-level bridge between the existing PDF/RAG
components:

    PDF file
      -> PDFExtractor
      -> SectionParser
      -> semantic_chunk()
      -> UploadedPaperIndexer
      -> persistent per-paper FAISS + chunk metadata

It deliberately does NOT:
    - implement FastAPI routes
    - implement RAG retrieval
    - call the LLM
    - generate summaries/analysis
    - perform reranking
    - modify the research-corpus index

The existing components remain authoritative for extraction, parsing,
chunking, embeddings, FAISS persistence, and retrieval contracts.

Public integration contract
----------------------------
    pipeline = PaperUploadPipeline(...)
    result = pipeline.process_uploaded_file("/path/to/upload.pdf")

The method is synchronous because the underlying PDF extraction and embedding
pipeline is synchronous. The API layer can execute it in a worker thread, as
the current papers.py gateway already does.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from backend.pdf.extractor import PDFExtractor
from backend.pdf.section_parser import SectionParser
from backend.pdf.chunker import (
    ChunkingConfig,
    semantic_chunk,
)

# These imports are needed only for static type checking.  The concrete
# uploaded-paper indexer is imported lazily by PaperUploadPipeline.__init__.
# This prevents upload_indexing.py from requiring legacy/removed result-model
# symbols such as IndexedPaperResult at module import time.
if TYPE_CHECKING:
    from backend.services.paper_indexing import (
        IndexedPaperResult,
        UploadedPaperIndexer,
    )

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "outputs" / "uploaded_papers"
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "indexes" / "uploaded"

# A stable, filesystem-safe public ID. The extractor's SHA-256 document ID is
# preserved separately as source_document_id.
PAPER_ID_PREFIX = "PAPER-"

# Avoid copying arbitrarily large files into persistent storage even when the
# API layer is configured incorrectly. The API remains the authoritative
# upload-size gate.
DEFAULT_MAX_SOURCE_BYTES = 100 * 1024 * 1024


class UploadIndexingError(RuntimeError):
    """Base exception for Mode-2 upload/indexing failures."""


class UploadInputError(UploadIndexingError, ValueError):
    """Uploaded file/input is invalid."""


class UploadExtractionError(UploadIndexingError):
    """PDF extraction or structural parsing failed."""


class UploadChunkingError(UploadIndexingError):
    """Semantic chunking failed or produced no usable chunks."""


class UploadPersistenceError(UploadIndexingError):
    """Uploaded-paper index persistence failed."""


@dataclass(frozen=True)
class UploadIndexingConfig:
    """Runtime configuration for the Mode-2 ingestion service."""

    source_root: Path = DEFAULT_SOURCE_ROOT
    index_root: Path = DEFAULT_INDEX_ROOT

    # Keep source PDFs so the "My Paper" workflow can later reopen the
    # authoritative original document if needed.
    persist_source_pdf: bool = True

    # Refuse unexpectedly large source files at the service boundary too.
    # The HTTP gateway normally enforces the smaller application limit.
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES

    # Explicitly use the existing chunking implementation/configuration.
    chunking_config: Optional[ChunkingConfig] = None

    # Existing indexer owns FAISS/index metadata persistence.
    # Valid indexes are reused; stale/corrupt indexes are rebuilt.
    overwrite_existing: bool = True
    verify_reload: bool = True
    include_chunk_text: bool = True

    # Stale-index lifecycle controls.
    validate_existing_index: bool = True
    compare_existing_chunks: bool = True
    quarantine_stale_index: bool = True

    def __post_init__(self) -> None:
        if self.max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be > 0.")

        object.__setattr__(
            self,
            "source_root",
            Path(self.source_root),
        )
        object.__setattr__(
            self,
            "index_root",
            Path(self.index_root),
        )


@dataclass(frozen=True)
class UploadIndexingResult:
    """
    Typed internal result.

    The API gateway can serialize/adapt this result without needing to know
    how PDF extraction, chunking, or FAISS persistence works.
    """

    document_id: str
    paper_id: str
    filename: str
    source_document_id: str
    status: str
    duplicate: bool
    page_count: int
    section_count: int
    chunk_count: int
    extraction_status: str
    parsing_status: str
    chunking_status: str
    index: IndexedPaperResult
    title: Optional[str]
    warnings: tuple[str, ...]
    processing_time_ms: float
    source_pdf_path: Optional[Path] = None

    def to_dict(self, *, expose_paths: bool = False) -> dict[str, Any]:
        """
        Convert to the existing API pipeline contract.

        By default filesystem paths are intentionally omitted from the public
        response. They are internal implementation details.
        """
        paper: dict[str, Any] = {
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "original_filename": self.filename,
            "pages": self.page_count,
            "title": self.title,
            "source_document_id": self.source_document_id,
        }

        payload: dict[str, Any] = {
            "success": True,
            "duplicate": self.duplicate,
            "paper": paper,
            "document_id": self.document_id,
            "paper_id": self.paper_id,
            "status": self.status,
            "page_count": self.page_count,
            "section_count": self.section_count,
            "chunk_count": self.chunk_count,
            "processing_time_ms": self.processing_time_ms,
            "extraction_status": self.extraction_status,
            "parsing_status": self.parsing_status,
            "chunking_status": self.chunking_status,
            "title": self.title,
            "warnings": list(self.warnings),
            "index": self.index.to_dict(),
        }

        if expose_paths:
            payload["source_pdf_path"] = (
                str(self.source_pdf_path)
                if self.source_pdf_path is not None
                else None
            )

        return payload


class PaperUploadPipeline:
    """
    Production Mode-2 upload/indexing orchestrator.

    Dependencies are injected so the service is easy to test and so expensive
    model resources are created once by application startup rather than once
    per request.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        indexer: Optional[UploadedPaperIndexer] = None,
        extractor: Optional[PDFExtractor] = None,
        section_parser: Optional[SectionParser] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        config: Optional[UploadIndexingConfig] = None,
    ) -> None:
        if encoder is None:
            raise UploadInputError("encoder cannot be None.")

        self.encoder = encoder
        self.config = config or UploadIndexingConfig()

        self.extractor = extractor or PDFExtractor()
        self.section_parser = section_parser or SectionParser()
        self.chunking_config = (
            chunking_config
            if chunking_config is not None
            else self.config.chunking_config
        )

        if indexer is None:
            # Import locally to keep this module's dependency surface small and
            # preserve the existing FAISS implementation as the authority.
            from backend.retrieval.faiss_index import FAISSVectorIndex
            from backend.services.paper_indexing import UploadedPaperIndexConfig

            indexer = UploadedPaperIndexer(
                encoder=encoder,
                config=UploadedPaperIndexConfig(
                    root_dir=self.config.index_root,
                    overwrite=self.config.overwrite_existing,
                    include_chunk_text=self.config.include_chunk_text,
                    verify_reload=self.config.verify_reload,
                ),
                faiss_index_cls=FAISSVectorIndex,
            )

        self.indexer = indexer

        self.config.source_root.mkdir(parents=True, exist_ok=True)
        self.config.index_root.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "PaperUploadPipeline initialized: index_root=%s",
            self.config.index_root,
        )

    # ------------------------------------------------------------------
    # Public API expected by backend/api/papers.py
    # ------------------------------------------------------------------

    def process_uploaded_file(self, file_path: str | os.PathLike[str]) -> dict[str, Any]:
        """
        Process one staged PDF synchronously and return the API pipeline dict.

        The staged file remains owned by the API gateway; this service only
        reads it and optionally copies the source PDF into persistent storage.
        """
        started = time.perf_counter()
        path = self._validate_pdf_path(file_path)

        filename = path.name
        file_size = path.stat().st_size

        LOGGER.info(
            "Starting Mode-2 upload processing: filename=%s size_bytes=%d",
            filename,
            file_size,
        )

        try:
            extracted = self.extractor.extract(path)
        except Exception as exc:
            LOGGER.exception(
                "PDF extraction failed: filename=%s",
                filename,
            )
            raise UploadExtractionError(
                "PDF extraction failed."
            ) from exc

        return self._process_extracted_document(
            extracted=extracted,
            filename=filename,
            source_path=path,
            started=started,
        )

    def process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        filename: str = "uploaded.pdf",
    ) -> dict[str, Any]:
        """
        Direct byte-oriented API for tests and non-FastAPI integrations.
        """
        if not isinstance(pdf_bytes, bytes):
            raise UploadInputError("pdf_bytes must be bytes.")
        if not pdf_bytes:
            raise UploadInputError("pdf_bytes cannot be empty.")
        if len(pdf_bytes) > self.config.max_source_bytes:
            raise UploadInputError(
                "PDF exceeds the configured service size limit."
            )

        if b"%PDF-" not in pdf_bytes[:1024]:
            raise UploadInputError(
                "Uploaded bytes do not appear to be a valid PDF."
            )

        started = time.perf_counter()
        safe_filename = Path(filename).name

        try:
            extracted = self.extractor.extract_bytes(
                pdf_bytes,
                filename=safe_filename,
            )
        except Exception as exc:
            LOGGER.exception(
                "PDF byte extraction failed: filename=%s",
                safe_filename,
            )
            raise UploadExtractionError(
                "PDF extraction failed."
            ) from exc

        return self._process_extracted_document(
            extracted=extracted,
            filename=safe_filename,
            source_path=None,
            source_bytes=pdf_bytes,
            started=started,
        )

    # Backwards/compatibility aliases useful to callers that already use
    # service-oriented naming.
    index_uploaded_file = process_uploaded_file
    build_uploaded_paper_index = process_uploaded_file

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _process_extracted_document(
        self,
        *,
        extracted: Any,
        filename: str,
        source_path: Optional[Path],
        source_bytes: Optional[bytes] = None,
        started: float,
    ) -> dict[str, Any]:
        source_document_id = self._validate_source_document_id(
            getattr(extracted, "document_id", None)
        )

        page_count = self._positive_int(
            getattr(extracted, "page_count", None),
            "page_count",
        )
        extraction_status = str(
            getattr(extracted, "extraction_status", "unknown")
        )

        if extraction_status not in {"success", "partial"}:
            raise UploadExtractionError(
                "PDF extraction did not produce a usable document."
            )

        paper_id = self._make_paper_id(source_document_id)

        # IMPORTANT: never declare a duplicate before parsing/chunking.
        # The canonical PDF ID proves content identity, but it does not prove
        # that the persisted index was produced by the current parser/chunker.

        # Stage 1: deterministic section parsing.
        try:
            structured = self.section_parser.parse(extracted)
        except Exception as exc:
            LOGGER.exception(
                "Section parsing failed: document_id=%s",
                source_document_id,
            )
            raise UploadExtractionError(
                "Research-paper section parsing failed."
            ) from exc

        # Stage 2: source-faithful semantic chunking.
        try:
            chunked = semantic_chunk(
                structured,
                config=self.chunking_config,
            )
        except Exception as exc:
            LOGGER.exception(
                "Semantic chunking failed: document_id=%s",
                source_document_id,
            )
            raise UploadChunkingError(
                "Semantic chunking failed."
            ) from exc

        if not chunked.chunks:
            raise UploadChunkingError(
                "PDF produced no usable retrieval chunks."
            )

        # The extractor's SHA-256 ID remains source_document_id. The public
        # application/retrieval identity is PAPER-<sha256>, which is what
        # /api/papers/upload returns and what UploadedPaperRetriever receives.
        #
        # SemanticChunk is frozen, so create exact copies with only the
        # document_id changed. No text, page, section, or chunk ordering is
        # modified.
        chunks = tuple(
            self._rebind_chunk_document_id(
                chunk,
                document_id=paper_id,
                source_document_id=source_document_id,
            )
            for chunk in chunked.chunks
        )

        # Stage 3: validate the persisted artifact against the CURRENT
        # extraction/parser/chunker output. Only an exact match is a duplicate.
        existing = None
        existing_state = "missing"
        if self.config.validate_existing_index:
            existing, existing_state = self._inspect_existing_index(
                paper_id=paper_id,
                source_document_id=source_document_id,
                chunks=chunks,
            )

        if existing is not None and existing_state == "valid":
            warnings = self._dedupe(
                (
                    "duplicate_upload",
                    "An identical PDF is already indexed with the current "
                    "chunking contract.",
                    *tuple(
                        str(value).strip()
                        for value in getattr(extracted, "warnings", ())
                        if str(value).strip()
                    ),
                    *tuple(
                        str(value).strip()
                        for value in getattr(structured, "warnings", ())
                        if str(value).strip()
                    ),
                    *tuple(
                        str(value).strip()
                        for value in getattr(chunked, "warnings", ())
                        if str(value).strip()
                    ),
                )
            )
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            return UploadIndexingResult(
                document_id=paper_id,
                paper_id=paper_id,
                filename=filename,
                source_document_id=source_document_id,
                status="completed",
                duplicate=True,
                page_count=page_count,
                section_count=int(getattr(structured, "section_count", 0)),
                chunk_count=existing.vector_count,
                extraction_status=extraction_status,
                parsing_status=str(
                    getattr(structured, "parsing_status", "success")
                ),
                chunking_status=str(
                    getattr(chunked, "chunking_status", "success")
                ),
                index=existing,
                title=getattr(structured, "title", None),
                warnings=warnings,
                processing_time_ms=elapsed,
                source_pdf_path=self._existing_source_path(paper_id),
            ).to_dict()

        rebuild_required = existing_state in {
            "stale",
            "invalid",
            "corrupt",
        }
        if rebuild_required:
            LOGGER.warning(
                "Existing uploaded-paper index is %s; rebuilding: "
                "document_id=%s",
                existing_state,
                paper_id,
            )

        try:
            indexed = self._build_index(
                paper_id=paper_id,
                chunks=chunks,
                source_document_id=source_document_id,
                force_rebuild=rebuild_required,
            )
        except Exception as exc:
            LOGGER.exception(
                "Uploaded-paper FAISS indexing failed: document_id=%s",
                paper_id,
            )
            raise UploadPersistenceError(
                "Uploaded-paper index creation failed."
            ) from exc

        # Optional persistent original PDF. It is copied only after successful
        # index publication, preventing a source file from looking indexed when
        # vector persistence actually failed.
        source_pdf_path: Optional[Path] = None
        if self.config.persist_source_pdf:
            try:
                source_pdf_path = self._persist_source_pdf(
                    paper_id=paper_id,
                    filename=filename,
                    source_path=source_path,
                    source_bytes=source_bytes,
                )
            except Exception as exc:
                # Do not silently report a fully healthy source archive when
                # source persistence fails. The vector index is still valid, so
                # expose the issue as a warning rather than deleting a good
                # index.
                LOGGER.exception(
                    "Source PDF persistence failed: document_id=%s",
                    paper_id,
                )
                source_pdf_path = None
                persistence_warning = (
                    "Index created, but original PDF persistence failed."
                )
            else:
                persistence_warning = None
        else:
            persistence_warning = None

        warnings = self._dedupe(
            (
                *tuple(
                    str(value).strip()
                    for value in getattr(extracted, "warnings", ())
                    if str(value).strip()
                ),
                *tuple(
                    str(value).strip()
                    for value in getattr(structured, "warnings", ())
                    if str(value).strip()
                ),
                *tuple(
                    str(value).strip()
                    for value in getattr(chunked, "warnings", ())
                    if str(value).strip()
                ),
                *(
                    (persistence_warning,)
                    if persistence_warning
                    else ()
                ),
            )
        )

        elapsed = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        result = UploadIndexingResult(
            document_id=paper_id,
            paper_id=paper_id,
            filename=filename,
            source_document_id=source_document_id,
            status="completed",
            duplicate=False,
            page_count=page_count,
            section_count=int(
                getattr(structured, "section_count", 0)
            ),
            chunk_count=indexed.vector_count,
            extraction_status=extraction_status,
            parsing_status=str(
                getattr(structured, "parsing_status", "unknown")
            ),
            chunking_status=str(
                getattr(chunked, "chunking_status", "unknown")
            ),
            index=indexed,
            title=getattr(structured, "title", None),
            warnings=warnings,
            processing_time_ms=elapsed,
            source_pdf_path=source_pdf_path,
        )

        LOGGER.info(
            "Mode-2 upload indexed successfully: document_id=%s "
            "pages=%d chunks=%d latency_ms=%.2f",
            paper_id,
            result.page_count,
            result.chunk_count,
            result.processing_time_ms,
        )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Validation / identity
    # ------------------------------------------------------------------

    def _validate_pdf_path(self, file_path: str | os.PathLike[str]) -> Path:
        if file_path is None:
            raise UploadInputError("file_path cannot be None.")

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise UploadInputError("Uploaded PDF does not exist.")

        if path.suffix.lower() != ".pdf":
            raise UploadInputError("Only PDF files are supported.")

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise UploadInputError(
                "Unable to inspect uploaded PDF."
            ) from exc

        if size <= 0:
            raise UploadInputError("Uploaded PDF is empty.")

        if size > self.config.max_source_bytes:
            raise UploadInputError(
                "PDF exceeds the configured service size limit."
            )

        # Lightweight container validation. The PDF extractor remains
        # authoritative for full PDF validity, but rejecting obviously
        # non-PDF files here avoids unnecessary parser/model work.
        try:
            with path.open("rb") as handle:
                header = handle.read(1024)
        except OSError as exc:
            raise UploadInputError(
                "Unable to read uploaded PDF."
            ) from exc

        if b"%PDF-" not in header:
            raise UploadInputError(
                "Uploaded file does not appear to be a valid PDF."
            )

        return path

    @staticmethod
    def _validate_source_document_id(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise UploadExtractionError(
                "PDF extractor did not produce a canonical document ID."
            )

        normalized = value.strip().lower()

        # PDFExtractor currently uses SHA-256. Keep the validation strict so a
        # future accidental identity change cannot silently alter persistence.
        if len(normalized) != 64:
            raise UploadExtractionError(
                "PDF extractor produced an invalid content identity."
            )

        try:
            int(normalized, 16)
        except ValueError as exc:
            raise UploadExtractionError(
                "PDF extractor produced an invalid content identity."
            ) from exc

        return normalized

    @staticmethod
    def _make_paper_id(source_document_id: str) -> str:
        return f"{PAPER_ID_PREFIX}{source_document_id}"

    # ------------------------------------------------------------------
    # Chunk identity adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _rebind_chunk_document_id(
        chunk: Any,
        *,
        document_id: str,
        source_document_id: str,
    ) -> Any:
        """
        Preserve the existing SemanticChunk object except for document_id.

        The existing chunking implementation is authoritative. This adapter
        does not change chunk text, page mapping, section mapping, indices,
        token counts, or overlap.
        """
        try:
            from dataclasses import replace

            metadata = dict(getattr(chunk, "metadata", {}) or {})
            # These identities are owned by the upload pipeline. Overwrite
            # conflicting values instead of preserving stale metadata.
            metadata["source_document_id"] = source_document_id
            metadata["paper_id"] = document_id

            return replace(
                chunk,
                document_id=document_id,
                metadata=metadata,
            )
        except Exception as exc:
            raise UploadChunkingError(
                "Semantic chunk could not be adapted to the "
                "uploaded-paper document identity."
            ) from exc

    # ------------------------------------------------------------------
    # Duplicate / persistence lifecycle
    # ------------------------------------------------------------------

    def _inspect_existing_index(
        self,
        *,
        paper_id: str,
        source_document_id: str,
        chunks: Sequence[Any],
    ) -> tuple[Optional[IndexedPaperResult], str]:
        """Classify persisted data as missing, valid, stale, or corrupt."""
        try:
            result = self.indexer.validate_persisted(document_id=paper_id)
        except FileNotFoundError:
            return None, "missing"
        except Exception as exc:
            LOGGER.warning(
                "Existing uploaded-paper index failed integrity validation: "
                "document_id=%s reason=%r",
                paper_id,
                exc,
            )
            return None, "corrupt"

        if result is None:
            return None, "missing"

        if not self.config.compare_existing_chunks:
            return result, "valid"

        try:
            metadata = self._read_index_metadata(result)
        except Exception as exc:
            LOGGER.warning(
                "Existing uploaded-paper metadata could not be read: "
                "document_id=%s reason=%r",
                paper_id,
                exc,
            )
            return None, "corrupt"

        if metadata.get("document_id") != paper_id:
            return None, "stale"
        if metadata.get("source_document_id") != source_document_id:
            return None, "stale"

        stored_chunks = metadata.get("chunks")
        if not isinstance(stored_chunks, list):
            return None, "corrupt"

        try:
            stored_count = int(metadata.get("vector_count", -1))
        except (TypeError, ValueError):
            return None, "corrupt"

        if stored_count != len(chunks) or len(stored_chunks) != len(chunks):
            return None, "stale"

        stored_ids = metadata.get("vector_ids")
        current_ids = [
            str(getattr(chunk, "chunk_id", "")) for chunk in chunks
        ]
        if not isinstance(stored_ids, list) or stored_ids != current_ids:
            return None, "stale"

        for current, stored in zip(chunks, stored_chunks):
            if not self._chunk_records_match(current, stored):
                return None, "stale"

        return result, "valid"

    @staticmethod
    def _read_index_metadata(result: Any) -> dict[str, Any]:
        metadata_path = Path(getattr(result, "metadata_path"))
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        import json

        payload = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("Persisted metadata root is not an object.")
        return payload

    @staticmethod
    def _chunk_records_match(chunk: Any, record: Any) -> bool:
        """Compare persisted chunk metadata with the current chunk."""
        if not isinstance(record, Mapping):
            return False

        fields = (
            "chunk_id",
            "document_id",
            "section_id",
            "section_type",
            "section_heading",
            "section_level",
            "start_page",
            "end_page",
            "parent_section_id",
            "chunk_index",
            "token_count",
            "char_count",
            "overlap_with_previous",
            "chunking_method",
        )
        for field in fields:
            if record.get(field) != getattr(chunk, field, None):
                return False

        if record.get("source_pages", []) != list(
            getattr(chunk, "source_pages", ()) or ()
        ):
            return False

        if "text" in record and record.get("text") != getattr(
            chunk, "text", None
        ):
            return False

        if record.get("metadata", {}) != (
            getattr(chunk, "metadata", {}) or {}
        ):
            return False

        return True

    def _build_index(
        self,
        *,
        paper_id: str,
        chunks: Sequence[Any],
        source_document_id: str,
        force_rebuild: bool,
    ) -> IndexedPaperResult:
        """Build the index and enable overwrite only for stale artifacts."""
        if force_rebuild:
            self._enable_indexer_overwrite()

        return self.indexer.build(
            document_id=paper_id,
            chunks=chunks,
            source_document_id=source_document_id,
        )

    def _enable_indexer_overwrite(self) -> None:
        """Enable overwrite for both current and compatibility indexers."""
        from dataclasses import replace

        config = getattr(self.indexer, "config", None)
        if config is not None and hasattr(config, "overwrite"):
            try:
                self.indexer.config = replace(config, overwrite=True)
                return
            except Exception as exc:
                LOGGER.debug(
                    "Could not replace indexer config: %r",
                    exc,
                )

        if hasattr(self.indexer, "overwrite"):
            try:
                setattr(self.indexer, "overwrite", True)
                return
            except Exception as exc:
                LOGGER.debug(
                    "Could not enable indexer overwrite: %r",
                    exc,
                )

        raise UploadPersistenceError(
            "The installed uploaded-paper indexer cannot rebuild an "
            "existing stale index. Enable overwrite support in "
            "paper_indexing.py."
        )

    def _existing_source_path(
        self,
        paper_id: str,
    ) -> Optional[Path]:
        directory = self.config.source_root / paper_id
        if not directory.is_dir():
            return None

        candidates = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.name.lower(),
        )
        return candidates[0] if candidates else None

    def _persist_source_pdf(
        self,
        *,
        paper_id: str,
        filename: str,
        source_path: Optional[Path],
        source_bytes: Optional[bytes],
    ) -> Path:
        target_dir = self.config.source_root / paper_id
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            safe_name = "paper.pdf"

        target = target_dir / safe_name
        if target.exists():
            return target

        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{safe_name}.",
            suffix=".tmp",
            dir=str(target_dir),
        )
        os.close(temp_fd)
        temp_path = Path(temp_name)

        try:
            if source_bytes is not None:
                if not source_bytes:
                    raise UploadPersistenceError(
                        "No source PDF bytes were available for persistence."
                    )
                temp_path.write_bytes(source_bytes)
            elif source_path is not None:
                with source_path.open("rb") as source, temp_path.open(
                    "wb"
                ) as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            else:
                raise UploadPersistenceError(
                    "No source PDF was available for persistence."
                )

            os.replace(temp_path, target)
            return target
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)

        return tuple(result)

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise UploadExtractionError(
                f"Extractor returned an invalid {name}."
            )
        return value


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def _parse_bool_setting(value: Any, *, name: str) -> bool:
    """Parse a configuration boolean without silently accepting typos."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value "
        "(true/false, 1/0, yes/no, on/off)."
    )


def create_paper_upload_pipeline(
    *,
    encoder: Any,
    indexer: Optional[UploadedPaperIndexer] = None,
    extractor: Optional[PDFExtractor] = None,
    section_parser: Optional[SectionParser] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> PaperUploadPipeline:
    """
    Application-level factory used by backend.main.

    The factory intentionally keeps settings translation here minimal. The
    existing Settings object remains the owner of application configuration.
    """
    settings = settings or {}

    source_root = Path(
        str(
            settings.get(
                "UPLOADED_PAPER_SOURCE_ROOT",
                DEFAULT_SOURCE_ROOT,
            )
        )
    )
    index_root = Path(
        str(
            settings.get(
                "UPLOADED_INDEX_ROOT",
                DEFAULT_INDEX_ROOT,
            )
        )
    )

    try:
        max_source_mb = int(
            settings.get(
                "MAX_UPLOAD_SIZE_MB",
                "50",
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MAX_UPLOAD_SIZE_MB must be a positive integer."
        ) from exc

    if max_source_mb <= 0:
        raise ValueError(
            "MAX_UPLOAD_SIZE_MB must be greater than zero."
        )

    config = UploadIndexingConfig(
        source_root=source_root,
        index_root=index_root,
        max_source_bytes=max_source_mb * 1024 * 1024,
        persist_source_pdf=_parse_bool_setting(
            settings.get(
                "PERSIST_UPLOADED_SOURCE_PDF",
                "true",
            ),
            name="PERSIST_UPLOADED_SOURCE_PDF",
        ),
        overwrite_existing=True,
        verify_reload=True,
        include_chunk_text=True,
        validate_existing_index=_parse_bool_setting(
            settings.get("VALIDATE_EXISTING_UPLOADED_INDEX", "true"),
            name="VALIDATE_EXISTING_UPLOADED_INDEX",
        ),
        compare_existing_chunks=_parse_bool_setting(
            settings.get("COMPARE_EXISTING_UPLOADED_CHUNKS", "true"),
            name="COMPARE_EXISTING_UPLOADED_CHUNKS",
        ),
        quarantine_stale_index=_parse_bool_setting(
            settings.get("QUARANTINE_STALE_UPLOADED_INDEX", "true"),
            name="QUARANTINE_STALE_UPLOADED_INDEX",
        ),
    )

    return PaperUploadPipeline(
        encoder=encoder,
        indexer=indexer,
        extractor=extractor,
        section_parser=section_parser,
        config=config,
    )


# ---------------------------------------------------------------------------
# Model-free self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """
    Fast contract test for the orchestration layer.

    This test does not require PyMuPDF, SPECTER2, FAISS, or Ollama because all
    expensive dependencies are injected.
    """
    from dataclasses import dataclass
    from tempfile import TemporaryDirectory

    @dataclass(frozen=True)
    class FakeExtracted:
        document_id: str
        page_count: int = 1
        extraction_status: str = "success"
        warnings: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class FakeStructured:
        document_id: str
        title: str = "Test Paper"
        section_count: int = 1
        parsing_status: str = "success"
        warnings: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class FakeChunk:
        chunk_id: str
        document_id: str
        section_id: str
        section_type: str
        section_heading: Optional[str]
        section_level: int
        text: str
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

    @dataclass(frozen=True)
    class FakeChunked:
        document_id: str
        chunks: tuple[FakeChunk, ...]
        chunk_count: int
        chunking_status: str = "success"
        warnings: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class FakeIndexResult:
        document_id: str
        index_directory: Path
        index_path: Path
        faiss_manifest_path: Path
        metadata_path: Path
        vector_count: int
        metadata_count: int
        embedding_dimension: int
        embedding_model: Optional[str] = "fake"
        embedding_model_version: Optional[str] = "test"
        normalized: bool = True
        index_type: str = "IndexFlatIP"
        similarity: str = "inner_product"
        created_at: str = "test"
        status: str = "ready"

        def to_dict(self) -> dict[str, Any]:
            return {
                "document_id": self.document_id,
                "vector_count": self.vector_count,
                "metadata_count": self.metadata_count,
                "embedding_dimension": self.embedding_dimension,
                "status": self.status,
            }

    class FakeExtractor:
        def extract(self, path: Path) -> FakeExtracted:
            return FakeExtracted(
                document_id=hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            )

    class FakeParser:
        def parse(self, document: FakeExtracted) -> FakeStructured:
            return FakeStructured(document_id=document.document_id)

    class FakeIndexer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def validate_persisted(
            self,
            *,
            document_id: str,
        ) -> Optional[FakeIndexResult]:
            return None

        def build(
            self,
            *,
            document_id: str,
            chunks: Sequence[Any],
            source_document_id: Optional[str] = None,
        ) -> FakeIndexResult:
            self.calls.append(
                {
                    "document_id": document_id,
                    "source_document_id": source_document_id,
                    "chunks": tuple(chunks),
                }
            )
            return FakeIndexResult(
                document_id=document_id,
                index_directory=Path("indexes/uploaded") / document_id,
                index_path=Path("index.index"),
                faiss_manifest_path=Path("index.index.manifest.json"),
                metadata_path=Path("metadata.json"),
                vector_count=len(chunks),
                metadata_count=len(chunks),
                embedding_dimension=8,
            )

    # Patch the module-level semantic_chunk function for a model-free test.
    global semantic_chunk
    original_semantic_chunk = semantic_chunk

    fake_chunk = FakeChunk(
        chunk_id="chunk-0",
        document_id="",
        section_id="sec-0",
        section_type="introduction",
        section_heading="Introduction",
        section_level=1,
        text="This is a valid scientific research chunk.",
        start_page=1,
        end_page=1,
        source_pages=(1,),
        parent_section_id=None,
        chunk_index=0,
        token_count=8,
        char_count=len("This is a valid scientific research chunk."),
        overlap_with_previous=0,
        chunking_method="paragraph",
        metadata={},
    )

    def fake_semantic_chunk(
        paper: FakeStructured,
        *,
        config: Optional[ChunkingConfig] = None,
    ) -> FakeChunked:
        return FakeChunked(
            document_id=paper.document_id,
            chunks=(
                FakeChunk(
                    **{
                        **fake_chunk.__dict__,
                        "document_id": paper.document_id,
                    }
                ),
            ),
            chunk_count=1,
        )

    semantic_chunk = fake_semantic_chunk

    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.7\n%test")

            fake_indexer = FakeIndexer()

            pipeline = PaperUploadPipeline(
                encoder=object(),
                indexer=fake_indexer,  # type: ignore[arg-type]
                extractor=FakeExtractor(),  # type: ignore[arg-type]
                section_parser=FakeParser(),  # type: ignore[arg-type]
                config=UploadIndexingConfig(
                    source_root=root / "sources",
                    index_root=root / "indexes",
                    max_source_bytes=1024 * 1024,
                ),
            )

            result = pipeline.process_uploaded_file(pdf)

            assert result["success"] is True
            assert result["duplicate"] is False
            assert result["status"] == "completed"
            assert result["page_count"] == 1
            assert result["chunk_count"] == 1
            assert result["paper_id"].startswith("PAPER-")
            assert len(fake_indexer.calls) == 1
            assert (
                fake_indexer.calls[0]["source_document_id"]
                == hashlib.sha256(b"%PDF-1.7\n%test").hexdigest()
            )
            assert (
                fake_indexer.calls[0]["chunks"][0].document_id
                == result["paper_id"]
            )
            expected_source_document_id = hashlib.sha256(
                b"%PDF-1.7\\n%test"
            ).hexdigest()
            assert (
                fake_indexer.calls[0]["chunks"][0].metadata["source_document_id"]
                == expected_source_document_id
            )
            assert (
                fake_indexer.calls[0]["chunks"][0].metadata["paper_id"]
                == result["paper_id"]
            )

            # Direct byte ingestion must use the same PDF boundary validation.
            byte_result = pipeline.process_pdf_bytes(
                b"%PDF-1.7\\n%byte-test",
                filename="byte-paper.pdf",
            )
            assert byte_result["success"] is True
            assert byte_result["duplicate"] is False

            # Obvious non-PDF input must be rejected before extraction.
            try:
                pipeline.process_pdf_bytes(
                    b"not-a-pdf",
                    filename="bad.pdf",
                )
            except UploadInputError:
                pass
            else:
                raise AssertionError(
                    "Non-PDF byte input was not rejected."
                )

        print("PaperUploadPipeline self-test: PASSED")
    finally:
        semantic_chunk = original_semantic_chunk


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()