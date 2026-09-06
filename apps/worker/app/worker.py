from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx
import pymupdf as fitz
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.config import get_settings
from apps.api.app.core.database import SessionFactory
from apps.api.app.core.evidence_state import EvidenceState
from apps.api.app.core.storage import generate_object_key, get_storage
from apps.api.app.models import (
    BackgroundJob,
    Document,
    DocumentChunk,
    Paper,
    PaperAnalysis,
)
from apps.api.app.services.analysis import (
    analyze_paper_text,
    clean_extracted_text,
    extract_uploaded_title,
)
from apps.api.app.services.queue import QUEUE_NAME
from apps.api.app.services.rag import Chunk, chunk_text
from apps.api.app.services.text_normalization import (
    normalize_extracted_text,
    validate_text_for_persistence,
)

logger = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_shutdown_event = asyncio.Event()


def handle_shutdown_signal(sig: int, frame: Any) -> None:
    logger.info(f"Received shutdown signal {sig}. Initiating graceful shutdown...")
    _shutdown_event.set()


async def consume() -> None:
    """Consume jobs from the Redis queue with automatic reconnect and graceful shutdown."""
    # Register signal handlers for graceful termination
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_event.set)

    logger.info(f"Starting background worker on queue '{QUEUE_NAME}'...")
    redis: Redis | None = None

    while not _shutdown_event.is_set():
        try:
            if redis is None:
                redis = Redis.from_url(
                    get_settings().redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            item = await redis.blpop(QUEUE_NAME, timeout=2)
            if item:
                payload = json.loads(item[1])
                logger.info(
                    f"Processing job payload: {payload.get('job_type', 'UNKNOWN')} ({payload.get('job_id')})"
                )
                await process(payload)
        except RedisError as exc:
            logger.warning(f"Redis connection error: {exc}. Reconnecting in 3 seconds...")
            if redis:
                await redis.aclose()
                redis = None
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Unexpected error in worker loop: {exc}", exc_info=True)
            await asyncio.sleep(1)

    if redis:
        await redis.aclose()
    logger.info("Background worker stopped gracefully.")


async def _mark_job_failed(job_id: UUID, paper_id: UUID, error_message: str) -> None:
    """Record job and paper failure in an isolated, clean database session.

    Guarantees no PendingRollbackError cascade from the failed processing transaction.
    """
    try:
        async with SessionFactory() as fail_session:
            job = await fail_session.get(BackgroundJob, job_id)
            if job:
                job.status = "FAILED"
                job.error = error_message[:500]
                job.completed_at = datetime.now(timezone.utc)
            paper = await fail_session.get(Paper, paper_id)
            if paper:
                paper.evidence_state = EvidenceState.FAILED.value
            await fail_session.commit()
            logger.info(f"Successfully recorded FAILED state for job {job_id} (paper: {paper_id}).")
    except Exception as record_exc:
        logger.error(
            f"Critical failure recording FAILED state for job {job_id}: {record_exc}",
            exc_info=True,
        )


async def process(payload: dict[str, Any]) -> None:
    job_id = UUID(payload["job_id"])
    paper_id = UUID(payload["paper_id"])
    job_type = payload.get("job_type", "PAPER_PROCESSING")

    try:
        async with SessionFactory() as session:
            job = await session.get(BackgroundJob, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found in database.")
                return

            if job_type == "CORPUS_FULL_TEXT_INGESTION":
                await _ingest_corpus_full_text(session, job, paper_id)
            elif job_type == "PAPER_PROCESSING":
                document_id = UUID(payload["document_id"]) if payload.get("document_id") else None
                await _process_uploaded_pdf(session, job, paper_id, document_id)
            else:
                logger.error(f"Unknown job type: {job_type}")
                job.status = "FAILED"
                job.error = f"Unsupported job type: {job_type}"
                await session.commit()
    except Exception as exc:
        logger.error(f"Failed processing job {job_id}: {exc}", exc_info=True)
        await _mark_job_failed(job_id, paper_id, str(exc))


async def _process_uploaded_pdf(
    session: AsyncSession, job: BackgroundJob, paper_id: UUID, document_id: UUID | None
) -> None:
    job.status = "PROCESSING"
    job.progress = 15
    job.started_at = datetime.now(timezone.utc)
    await session.commit()

    paper = await session.get(Paper, paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found.")

    if not document_id:
        doc_query = await session.execute(
            select(Document).where(Document.paper_id == paper_id).limit(1)
        )
        document = doc_query.scalar_one_or_none()
    else:
        document = await session.get(Document, document_id)

    if not document:
        raise ValueError(f"Document record for paper {paper_id} not found.")

    # Retrieve raw binary PDF from storage
    storage = get_storage()
    if not document.object_key:
        raise ValueError("Document has no object storage key.")

    pdf_bytes = await storage.get(document.object_key)
    if not pdf_bytes:
        raise ValueError("Empty binary content retrieved from storage.")

    job.progress = 30
    await session.commit()

    # Extract pages and text
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts: list[str] = []
    for page in pdf:
        raw_page_text = page.get_text()
        normalized_page = normalize_extracted_text(
            raw_page_text, document_id=str(document.id), job_id=str(job.id)
        )
        page_texts.append(normalized_page.strip())
    page_count = len(pdf)
    pdf.close()

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError("PDF contains no extractable text.")

    # Validation boundary before database persistence
    valid, err_msg = validate_text_for_persistence(full_text)
    if not valid:
        raise ValueError(f"Extracted document text failed database validation: {err_msg}")

    document.content = full_text
    document.page_count = page_count

    # Update paper title if default or fallback
    extracted_title = extract_uploaded_title(full_text, document.filename)
    if extracted_title:
        extracted_title = normalize_extracted_text(extracted_title)
        if not paper.title or paper.title in ("Uploaded paper", document.filename):
            paper.title = extracted_title
    paper.abstract = normalize_extracted_text(full_text[:10000])

    job.progress = 50
    await session.commit()

    # Chunking and embedding
    chunks: list[Chunk] = []
    chunk_index = 0
    for page_no, page_text in enumerate(page_texts, start=1):
        for chunk in chunk_text(page_text, page_no):
            norm_chunk_text = normalize_extracted_text(chunk.text)
            norm_section = normalize_extracted_text(chunk.section) if chunk.section else "Body"
            chunks.append(
                Chunk(
                    text=norm_chunk_text,
                    page=page_no,
                    section=norm_section,
                    index=chunk_index,
                    embedding=chunk.embedding,
                )
            )
            chunk_index += 1

    if not chunks:
        raise ValueError("Text chunking produced zero segments.")

    # Delete existing chunks if any (idempotency)
    existing_chunks = (
        (
            await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document.id)
            )
        )
        .scalars()
        .all()
    )
    for existing in existing_chunks:
        await session.delete(existing)
    await session.flush()

    for chunk in chunks:
        session.add(
            DocumentChunk(
                document_id=document.id,
                text=chunk.text,
                section=chunk.section,
                page=chunk.page,
                chunk_index=chunk.index,
                embedding=chunk.embedding,
            )
        )

    job.progress = 75
    await session.commit()

    # Run paper analysis
    analysis = await analyze_paper_text(paper.title, full_text, "extracted PDF text")

    existing_analysis = await session.get(PaperAnalysis, paper_id)
    if existing_analysis:
        existing_analysis.payload = analysis
        existing_analysis.model = str(analysis.get("method", "local"))
        existing_analysis.confidence = analysis.get("confidence")
    else:
        session.add(
            PaperAnalysis(
                paper_id=paper_id,
                payload=analysis,
                model=str(analysis.get("method", "local")),
                confidence=analysis.get("confidence"),
            )
        )

    paper.evidence_state = EvidenceState.FULL_TEXT.value
    job.status = "COMPLETED"
    job.progress = 100
    job.completed_at = datetime.now(timezone.utc)
    job.result = {
        "paper_id": str(paper.id),
        "document_id": str(document.id),
        "page_count": page_count,
        "chunk_count": len(chunks),
        "analysis": analysis,
    }
    await session.commit()
    logger.info(
        f"Successfully processed paper {paper.id} ({len(chunks)} chunks, {page_count} pages)."
    )


async def _ingest_corpus_full_text(
    session: AsyncSession, job: BackgroundJob, paper_id: UUID
) -> None:
    paper = await session.get(Paper, paper_id)
    if not paper or not paper.pdf_url:
        if paper:
            paper.evidence_state = EvidenceState.METADATA_ONLY.value
        job.status = "FAILED"
        job.error = "No accessible full-text PDF URL was supplied."
        await session.commit()
        return

    job.status = "PROCESSING"
    job.progress = 20
    job.started_at = datetime.now(timezone.utc)
    paper.evidence_state = EvidenceState.PROCESSING.value
    await session.commit()

    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        response = await client.get(paper.pdf_url, headers={"Accept": "application/pdf"})
        response.raise_for_status()

    content = response.content
    if not content.startswith(b"%PDF"):
        raise ValueError("Remote URL did not return a valid PDF.")

    storage = get_storage()
    filename = f"{paper.external_id or paper.id}.pdf"
    object_key = generate_object_key(None, paper.id, filename)
    await storage.put(object_key, content, content_type="application/pdf")

    pdf = fitz.open(stream=content, filetype="pdf")
    page_texts: list[str] = []
    for page in pdf:
        raw_page_text = page.get_text()
        normalized_page = normalize_extracted_text(
            raw_page_text, document_id=None, job_id=str(job.id)
        )
        page_texts.append(normalized_page.strip())
    page_count = len(pdf)
    pdf.close()

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError("PDF has no extractable text.")

    # Validation boundary before database persistence
    valid, err_msg = validate_text_for_persistence(full_text)
    if not valid:
        raise ValueError(f"Corpus document text failed database validation: {err_msg}")

    chunks: list[Chunk] = []
    chunk_index = 0
    for page_no, page_text in enumerate(page_texts, start=1):
        for chunk in chunk_text(page_text, page_no):
            norm_chunk_text = normalize_extracted_text(chunk.text)
            norm_section = normalize_extracted_text(chunk.section) if chunk.section else "Body"
            chunks.append(
                Chunk(
                    text=norm_chunk_text,
                    page=page_no,
                    section=norm_section,
                    index=chunk_index,
                    embedding=chunk.embedding,
                )
            )
            chunk_index += 1

    if not chunks:
        raise ValueError("PDF extraction produced no chunks.")

    document = Document(
        paper_id=paper.id,
        filename=filename,
        content=full_text,
        page_count=page_count,
        content_hash=hashlib.sha256(content).hexdigest(),
        object_key=object_key,
        size_bytes=len(content),
        mime_type="application/pdf",
    )
    session.add(document)
    await session.flush()

    for chunk in chunks:
        session.add(
            DocumentChunk(
                document_id=document.id,
                text=chunk.text,
                section=chunk.section,
                page=chunk.page,
                chunk_index=chunk.index,
                embedding=chunk.embedding,
            )
        )

    analysis = await analyze_paper_text(paper.title, full_text, "extracted PDF text")
    session.add(
        PaperAnalysis(
            paper_id=paper.id,
            payload=analysis,
            model=str(analysis.get("method", "local")),
            confidence=analysis.get("confidence"),
        )
    )

    paper.evidence_state = EvidenceState.FULL_TEXT.value
    job.status = "COMPLETED"
    job.progress = 100
    job.completed_at = datetime.now(timezone.utc)
    job.result = {
        "paper_id": str(paper.id),
        "document_id": str(document.id),
        "chunk_count": len(chunks),
    }
    await session.commit()
    logger.info(f"Successfully ingested full-text corpus paper {paper.id}.")


if __name__ == "__main__":
    asyncio.run(consume())
