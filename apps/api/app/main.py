from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pymupdf as fitz
import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.config import get_settings
from apps.api.app.core.database import SessionFactory, get_session
from apps.api.app.core.evidence_state import EvidenceState, supports_full_text_qa
from apps.api.app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from apps.api.app.core.storage import (
    MAX_PDF_SIZE_BYTES,
    generate_object_key,
    get_storage,
    sanitize_filename,
    validate_pdf_bytes,
)
from apps.api.app.models import (
    BackgroundJob,
    ChatMessage,
    ChatSession,
    CollectionPaper,
    Document,
    DocumentChunk,
    Paper,
    PaperAnalysis,
    ResearchCollection,
    User,
)
from apps.api.app.providers.scholarly import search_scholarly
from apps.api.app.services.analysis import (
    analyze_paper_text,
    clean_extracted_text,
    extract_uploaded_title,
    fallback_analysis,
    model_analysis,
)
from apps.api.app.services.queue import enqueue_job
from apps.api.app.services.rag import (
    Chunk,
    chunk_text,
    classify_question,
    embed,
    evidence_supports,
    hybrid_retrieve,
)
from apps.api.app.services.reranker import rerank

app = FastAPI(title="AI Research Paper Assistant API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in get_settings().cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "arxiv_papers.jsonl"
)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def _clean_extracted_text(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def _uploaded_title(text: str, filename: str) -> str:
    fallback = Path(filename).stem or "Uploaded paper"
    for line in text.splitlines():
        candidate = " ".join(line.split()).strip()
        if 8 <= len(candidate) <= 180 and candidate.lower() not in {
            "abstract",
            "introduction",
            "references",
        }:
            return candidate
    return fallback


class SearchRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class PaperInput(BaseModel):
    id: str
    title: str
    summary: str = ""
    authors: str = ""
    year: int | None = None
    venue: str | None = None
    citation_count: int = 0


class CollectionRequest(BaseModel):
    papers: list[PaperInput] = Field(min_length=2, max_length=20)
    topic: str = Field(min_length=2, max_length=200)


class CollectionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class CollectionPaperRequest(BaseModel):
    paper_id: UUID


class ChatRequest(BaseModel):
    paper_id: UUID
    session_id: UUID | None = None
    question: str = Field(min_length=2, max_length=1000)


class PaperQARequest(BaseModel):
    document_id: UUID
    question: str = Field(min_length=2, max_length=1000)


def _paper_terms(paper: PaperInput) -> set[str]:
    return _words(f"{paper.title} {paper.summary}")


async def _verify_paper_access(
    papers: list[PaperInput], user: User, session: AsyncSession
) -> None:
    uuid_ids: list[UUID] = []
    for p in papers:
        try:
            uuid_ids.append(UUID(p.id))
        except (ValueError, TypeError):
            continue
    if not uuid_ids:
        return

    result = await session.execute(
        select(Paper).where(
            Paper.id.in_(uuid_ids),
            Paper.owner_id.is_not(None),
            Paper.owner_id != user.id,
        )
    )
    unauthorized = result.scalars().all()
    if unauthorized:
        raise HTTPException(
            status_code=403,
            detail="Access denied: One or more selected papers belong to another user.",
        )


async def current_user(
    token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)
) -> User:
    try:
        user_id = decode_access_token(token)
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid or expired access token"
        ) from error
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _words(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", value.lower()) if len(word) > 2}


def _analysis(title: str, summary: str, source: str = "abstract") -> dict[str, Any]:
    words = summary.split()
    lead = " ".join(words[:45]) + ("..." if len(words) > 45 else "")
    return {
        "summary": lead or "No abstract text was available for this paper.",
        "strengths": ["Not verified — full text unavailable."],
        "weaknesses": [
            "Not verified — full text unavailable.",
        ],
        "advantages": "Metadata is available for discovery only.",
        "disadvantages": "Paper-specific strengths and limitations require full text.",
        "method": "Evidence-limited local extraction",
        "evidence": f"Title: {title}. Source: {source}.",
    }


ANALYSIS_SCHEMA = """Return only valid JSON with these keys: summary (string), strengths (array of 2-4 strings), weaknesses (array of 2-4 strings), advantages (string), disadvantages (string), evidence (array of 2-5 short strings), confidence (number 0 to 1). Every claim must be supported by the supplied text. If the text is insufficient, say so explicitly instead of guessing."""


async def _model_analysis(title: str, text: str, source: str) -> dict[str, Any] | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        "max_tokens": 1400,
        "system": "You are a rigorous research-paper analyst. Do not invent findings, methods, metrics, citations, or limitations.",
        "messages": [
            {
                "role": "user",
                "content": f"{ANALYSIS_SCHEMA}\n\nPaper title: {title}\nText source: {source}\nPaper text:\n{text[:24000]}",
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        raw = response.json()["content"][0]["text"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        result = json.loads(raw)
        required = {
            "summary",
            "strengths",
            "weaknesses",
            "advantages",
            "disadvantages",
            "evidence",
            "confidence",
        }
        if not required.issubset(result) or not isinstance(
            result["confidence"], (int, float)
        ):
            return None
        result["method"] = "Claude Sonnet 4.5"
        return result
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _corpus_papers(topic: str) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        return []
    query_words = _words(topic)
    matches: list[tuple[int, dict[str, Any]]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
            haystack = f"{paper.get('title', '')} {paper.get('summary', '')} {paper.get('category', '')}"
            score = sum(
                3 if word in str(paper.get("title", "")).lower() else 1
                for word in query_words
                if word in haystack.lower()
            )
            if score:
                matches.append((score, paper))
    matches.sort(
        key=lambda item: (item[0], item[1].get("published_date") or ""), reverse=True
    )
    return [
        {
            **paper,
            "score": score,
            "analysis": _analysis(
                str(paper.get("title", "")), str(paper.get("summary", ""))
            ),
        }
        for score, paper in matches[:10]
    ]


async def _model_answer(context: str, question: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        "max_tokens": 500,
        "system": "Answer only from supplied evidence. If it does not explicitly support the answer, reply exactly: Insufficient evidence in the available paper text.",
        "messages": [
            {
                "role": "user",
                "content": f"Paper:\n{context[:12000]}\n\nQuestion: {question}",
            }
        ],
    }
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
    if response.is_error:
        return None
    return response.json()["content"][0]["text"]


def _fallback_answer(paper: dict[str, Any], question: str) -> str:
    chunks: list[Chunk] = paper.get("chunks", [])
    selected = hybrid_retrieve(chunks, question) if chunks else []
    if not selected:
        return "I could not verify this information from the available paper content."
    query_terms = _words(question)
    sentences = [
        sentence.strip()
        for chunk in selected
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text)
        if sentence.strip()
    ]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(query_terms.intersection(_words(sentence))),
        reverse=True,
    )
    evidence = " ".join(ranked[:3])[:900]
    if not evidence or not query_terms.intersection(_words(evidence)):
        return "I could not verify this information from the available paper content."
    citations = ", ".join(
        f"Page {chunk.page}, {chunk.section}" if chunk.page else chunk.section
        for chunk in selected[:3]
    )
    return f"Based only on the retrieved paper evidence: {evidence}\n\nEvidence: {citations}"


INSUFFICIENT = "Insufficient evidence in the available paper text. The retrieved source does not contain enough information to verify this."


def _evidence_response(
    question: str,
    candidates: list[Chunk],
    document_id: UUID | None = None,
    full_text: bool = True,
) -> dict[str, Any]:
    question_type = classify_question(question)
    selected, reranker_mode = rerank(
        question, hybrid_retrieve(candidates, question, limit=30), limit=6
    )
    supported = full_text and evidence_supports(question, selected, question_type)
    evidence = (
        [
            {
                "document_id": str(document_id) if document_id else None,
                "page": item.page,
                "section": item.section,
                "chunk_index": item.index,
                "text": item.text,
            }
            for item in selected
        ]
        if supported
        else []
    )
    return {
        "answer": _fallback_answer({"chunks": selected}, question)
        if supported
        else INSUFFICIENT,
        "evidence": evidence,
        "citations": [
            {
                "page": item["page"],
                "section": item["section"],
                "chunk_index": item["chunk_index"],
            }
            for item in evidence
        ],
        "status": "evidence-backed" if supported else "insufficient-evidence",
        "validation": {
            "status": "evidence-backed" if supported else "insufficient-evidence",
            "question_type": question_type,
            "retrieval_count": len(candidates),
            "evidence_count": len(evidence),
            "reranker": reranker_mode,
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs/{job_id}")
async def get_job(
    job_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    job = (
        await session.execute(
            select(BackgroundJob).where(
                BackgroundJob.id == job_id, BackgroundJob.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": str(job.id),
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "result": job.result,
    }


@app.post("/api/auth/register", status_code=201)
async def register(
    request: RegisterRequest, session: AsyncSession = Depends(get_session)
) -> dict[str, str]:
    email = request.email.lower()
    if (
        await session.execute(select(User).where(User.email == email))
    ).scalar_one_or_none():
        raise HTTPException(
            status_code=409, detail="An account with this email already exists"
        )
    user = User(email=email, password_hash=hash_password(request.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return {"access_token": create_access_token(str(user.id)), "token_type": "bearer"}


@app.post("/api/auth/token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    user = (
        await session.execute(select(User).where(User.email == form.username.lower()))
    ).scalar_one_or_none()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {"access_token": create_access_token(str(user.id)), "token_type": "bearer"}


@app.get("/api/auth/me")
async def me(user: User = Depends(current_user)) -> dict[str, str]:
    return {"id": str(user.id), "email": user.email}


@app.post("/api/collections", status_code=201)
async def create_collection(
    request: CollectionCreateRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    collection = ResearchCollection(name=request.name, user_id=user.id)
    session.add(collection)
    await session.commit()
    await session.refresh(collection)
    return {"id": str(collection.id), "name": collection.name}


@app.get("/api/collections")
async def list_collections(
    user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> list[dict[str, str]]:
    result = await session.execute(
        select(ResearchCollection)
        .where(ResearchCollection.user_id == user.id)
        .order_by(ResearchCollection.created_at.desc())
    )
    return [{"id": str(item.id), "name": item.name} for item in result.scalars()]


@app.post("/api/collections/{collection_id}/papers", status_code=201)
async def add_collection_paper(
    collection_id: UUID,
    request: CollectionPaperRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    collection = (
        await session.execute(
            select(ResearchCollection).where(
                ResearchCollection.id == collection_id,
                ResearchCollection.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not collection or not await session.get(Paper, request.paper_id):
        raise HTTPException(status_code=404, detail="Collection or paper not found")
    session.add(CollectionPaper(collection_id=collection_id, paper_id=request.paper_id))
    await session.commit()
    return {"collection_id": str(collection_id), "paper_id": str(request.paper_id)}


@app.get("/api/collections/{collection_id}")
async def get_collection(
    collection_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    collection = (
        await session.execute(
            select(ResearchCollection).where(
                ResearchCollection.id == collection_id,
                ResearchCollection.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    result = await session.execute(
        select(Paper)
        .join(CollectionPaper, CollectionPaper.paper_id == Paper.id)
        .where(CollectionPaper.collection_id == collection_id)
    )
    return {
        "id": str(collection.id),
        "name": collection.name,
        "papers": [
            {"id": str(paper.id), "title": paper.title, "source": paper.source}
            for paper in result.scalars()
        ],
    }


@app.patch("/api/collections/{collection_id}")
async def rename_collection(
    collection_id: UUID,
    request: CollectionCreateRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    collection = (
        await session.execute(
            select(ResearchCollection).where(
                ResearchCollection.id == collection_id,
                ResearchCollection.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    collection.name = request.name
    await session.commit()
    return {"id": str(collection.id), "name": collection.name}


@app.delete("/api/collections/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    collection = (
        await session.execute(
            select(ResearchCollection).where(
                ResearchCollection.id == collection_id,
                ResearchCollection.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    await session.delete(collection)
    await session.commit()


@app.delete("/api/collections/{collection_id}/papers/{paper_id}", status_code=204)
async def remove_collection_paper(
    collection_id: UUID,
    paper_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    collection = (
        await session.execute(
            select(ResearchCollection).where(
                ResearchCollection.id == collection_id,
                ResearchCollection.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    membership = (
        await session.execute(
            select(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id,
                CollectionPaper.paper_id == paper_id,
            )
        )
    ).scalar_one_or_none()
    if membership:
        await session.delete(membership)
        await session.commit()


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    local_papers = _corpus_papers(request.topic)
    scholarly_papers = await search_scholarly(request.topic)
    query_words = _words(request.topic)
    merged: dict[str, dict[str, Any]] = {}
    for paper in [*local_papers, *scholarly_papers]:
        key = str(paper.get("doi") or paper.get("title", "")).lower().strip()
        if not key:
            continue
        haystack = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
        paper["score"] = sum(
            3 if word in str(paper.get("title", "")).lower() else 1
            for word in query_words
            if word in haystack
        )
        paper["analysis"] = paper.get("analysis") or _analysis(
            str(paper.get("title", "")), str(paper.get("summary", "")), "abstract"
        )
        current = merged.get(key)
        if not current or paper.get("citation_count", 0) > current.get(
            "citation_count", 0
        ):
            merged[key] = paper
    papers = sorted(
        merged.values(),
        key=lambda item: (item.get("score", 0), item.get("citation_count", 0)),
        reverse=True,
    )[:10]
    corpus_jobs: list[tuple[UUID, UUID]] = []
    async with SessionFactory() as session:
        for paper in papers:
            external_id = str(paper.get("doi") or paper.get("id") or paper.get("title"))
            stored = (
                await session.execute(
                    select(Paper).where(Paper.external_id == external_id)
                )
            ).scalar_one_or_none()
            if not stored:
                state = "processing" if paper.get("pdf_url") else "metadata-only"
                stored = Paper(
                    external_id=external_id,
                    title=str(paper.get("title", "Untitled paper")),
                    abstract=str(paper.get("summary", "")),
                    authors=str(paper.get("authors_raw", "")),
                    venue=paper.get("venue"),
                    source=str(paper.get("source", "unknown")),
                    citation_count=int(paper.get("citation_count", 0) or 0),
                    doi=paper.get("doi"),
                    url=paper.get("url"),
                    pdf_url=paper.get("pdf_url"),
                    evidence_state=state,
                )
                session.add(stored)
            else:
                stored.citation_count = int(
                    paper.get("citation_count", stored.citation_count) or 0
                )
            await session.flush()
            paper["id"] = str(stored.id)
            paper["evidence_state"] = stored.evidence_state
            if (
                stored.evidence_state == "processing"
                and not (
                    await session.execute(
                        select(BackgroundJob.id).where(
                            BackgroundJob.type == "CORPUS_FULL_TEXT_INGESTION",
                            BackgroundJob.result["paper_id"].astext == str(stored.id),
                        )
                    )
                ).scalar_one_or_none()
            ):
                job = BackgroundJob(
                    type="CORPUS_FULL_TEXT_INGESTION",
                    status="PENDING",
                    progress=0,
                    result={"paper_id": str(stored.id)},
                )
                session.add(job)
                await session.flush()
                corpus_jobs.append((job.id, stored.id))
        await session.commit()
    for job_id, paper_id in corpus_jobs:
        try:
            await enqueue_job(
                {
                    "job_id": str(job_id),
                    "paper_id": str(paper_id),
                    "job_type": "CORPUS_FULL_TEXT_INGESTION",
                }
            )
        except Exception:
            # The durable PENDING job remains observable and can be retried by an operator.
            pass
    notice = (
        "Results combine OpenAlex, optional Semantic Scholar, and the local corpus."
        if papers
        else "No scholarly matches are available. Try a broader topic or check your network connection."
    )
    return {
        "topic": request.topic,
        "count": len(papers),
        "papers": papers,
        "notice": notice,
    }


@app.post("/api/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    content = await file.read(MAX_PDF_SIZE_BYTES + 1)
    valid, err_msg = validate_pdf_bytes(content, max_bytes=MAX_PDF_SIZE_BYTES)
    if not valid:
        code = 413 if "limit" in (err_msg or "").lower() else 422
        raise HTTPException(status_code=code, detail=err_msg)

    content_hash = hashlib.sha256(content).hexdigest()
    paper_external_id = content_hash[:16]

    duplicate = (
        await session.execute(
            select(Paper.id).where(
                Paper.external_id == paper_external_id, Paper.owner_id == user.id
            )
        )
    ).scalar_one_or_none()
    if duplicate:
        raise HTTPException(
            status_code=409, detail="This PDF has already been uploaded."
        )

    safe_filename = sanitize_filename(file.filename or "uploaded.pdf")
    initial_title = (
        Path(safe_filename).stem.replace("_", " ").title() or "Uploaded paper"
    )

    paper = Paper(
        external_id=paper_external_id,
        title=initial_title,
        abstract="",
        source="upload",
        owner_id=user.id,
        evidence_state=EvidenceState.PROCESSING.value,
    )
    session.add(paper)
    await session.flush()

    object_key = generate_object_key(user.id, paper.id, safe_filename)
    storage = get_storage()
    await storage.put(object_key, content, content_type="application/pdf")

    document_record = Document(
        paper_id=paper.id,
        filename=safe_filename,
        content="",
        page_count=0,
        content_hash=content_hash,
        object_key=object_key,
        size_bytes=len(content),
        mime_type="application/pdf",
    )
    session.add(document_record)
    await session.flush()

    job = BackgroundJob(
        user_id=user.id,
        type="PAPER_PROCESSING",
        status="PENDING",
        progress=0,
        result={"paper_id": str(paper.id), "document_id": str(document_record.id)},
    )
    session.add(job)
    await session.commit()

    paper_uuid, job_uuid, doc_uuid = paper.id, job.id, document_record.id

    try:
        await enqueue_job(
            {
                "job_id": str(job_uuid),
                "paper_id": str(paper_uuid),
                "document_id": str(doc_uuid),
                "user_id": str(user.id),
                "job_type": "PAPER_PROCESSING",
            }
        )
    except Exception:
        pass

    return {
        "id": str(paper_uuid),
        "document_id": str(doc_uuid),
        "external_id": paper_external_id,
        "title": initial_title,
        "source": "upload",
        "job_id": str(job_uuid),
        "status": "PENDING",
        "evidence_state": "processing",
    }


async def _process_paper_job(
    job_id: UUID,
    paper_id: UUID,
    document_id: UUID,
    chunks: list[Chunk],
    title: str,
    text: str,
) -> None:
    async with SessionFactory() as session:
        job = await session.get(BackgroundJob, job_id)
        if not job:
            return
        try:
            job.status, job.progress, job.started_at = (
                "PROCESSING",
                35,
                datetime.now(timezone.utc),
            )
            document = (
                await session.execute(
                    select(Document).where(
                        Document.id == document_id, Document.paper_id == paper_id
                    )
                )
            ).scalar_one()
            indexed = (
                await session.execute(
                    select(DocumentChunk.id)
                    .where(DocumentChunk.document_id == document.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if not indexed:
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
            job.progress = 70
            analysis = await _model_analysis(
                title, text, "extracted PDF text"
            ) or _analysis(title, text, "extracted PDF text")
            session.add(
                PaperAnalysis(
                    paper_id=paper_id,
                    payload=analysis,
                    model=str(analysis.get("method", "local")),
                    confidence=analysis.get("confidence"),
                )
            )
            paper_rec = await session.get(Paper, paper_id)
            if paper_rec:
                paper_rec.evidence_state = "full-text"
            job.result = {"paper_id": str(paper_id), "analysis": analysis}
            job.status, job.progress, job.completed_at = (
                "COMPLETED",
                100,
                datetime.now(timezone.utc),
            )
            await session.commit()
        except Exception as error:
            job.status, job.error = (
                "FAILED",
                "Paper processing failed: " + str(error)[:300],
            )
            await session.commit()


@app.get("/api/papers/{paper_id}")
async def get_paper(
    paper_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    paper = await session.get(Paper, paper_id)
    if not paper or (paper.owner_id is not None and paper.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Paper not found")
    document_id = (
        await session.execute(
            select(Document.id).where(Document.paper_id == paper.id).limit(1)
        )
    ).scalar_one_or_none()
    analysis = (
        await session.execute(
            select(PaperAnalysis).where(PaperAnalysis.paper_id == paper.id)
        )
    ).scalar_one_or_none()
    return {
        "id": str(paper.id),
        "document_id": str(document_id) if document_id else None,
        "title": paper.title,
        "summary": paper.abstract,
        "authors_raw": paper.authors,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "source": paper.source,
        "evidence_state": paper.evidence_state,
        "analysis": analysis.payload
        if analysis
        else _analysis(paper.title, paper.abstract),
        "processing_status": "COMPLETED" if analysis else "PROCESSING",
    }


@app.get("/api/papers")
async def list_papers(
    user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> list[dict[str, Any]]:
    document_id = (
        select(Document.id)
        .where(Document.paper_id == Paper.id)
        .limit(1)
        .scalar_subquery()
    )
    result = await session.execute(
        select(Paper, document_id)
        .where((Paper.owner_id == user.id) | (Paper.owner_id.is_(None)))
        .order_by(Paper.created_at.desc())
    )
    return [
        {
            "id": str(paper.id),
            "document_id": str(doc_id) if doc_id else None,
            "title": paper.title,
            "summary": paper.abstract,
            "authors_raw": paper.authors,
            "source": paper.source,
            "evidence_state": paper.evidence_state,
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
        }
        for paper, doc_id in result.all()
    ]


@app.post("/api/qa")
async def paper_qa(
    request: PaperQARequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    document = (
        await session.execute(
            select(Document)
            .join(Paper)
            .where(Document.id == request.document_id, Paper.owner_id == user.id)
        )
    ).scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document not found")

    analysis = await session.get(PaperAnalysis, document.paper_id)
    if not analysis:
        raise HTTPException(
            status_code=409,
            detail="This document is still processing. Q&A will be available when processing is complete.",
        )

    rows = (
        (
            await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document.id)
            )
        )
        .scalars()
        .all()
    )
    candidates = [
        Chunk(
            chunk.text,
            chunk.page,
            chunk.section or "Body",
            chunk.chunk_index,
            chunk.embedding or [],
        )
        for chunk in rows
        if chunk.text.strip() and chunk.embedding
    ]
    response = _evidence_response(request.question, candidates, document.id)
    return {
        "document_id": str(document.id),
        "question": request.question,
        "confidence": None,
        **response,
    }


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    paper = await session.get(Paper, request.paper_id)
    if not paper or (paper.owner_id is not None and paper.owner_id != user.id):
        raise HTTPException(
            status_code=403, detail="You do not have access to this paper"
        )
    if request.session_id:
        chat_session = (
            await session.execute(
                select(ChatSession).where(
                    ChatSession.id == request.session_id,
                    ChatSession.user_id == user.id,
                    ChatSession.paper_id == paper.id,
                )
            )
        ).scalar_one_or_none()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        chat_session = ChatSession(
            user_id=user.id, paper_id=paper.id, title=request.question[:80]
        )
        session.add(chat_session)
        await session.flush()
    # Corpus metadata is deliberately never indexed as a DocumentChunk. Only
    # an extracted, page-addressable PDF may support detailed paper answers.
    full_text = supports_full_text_qa(paper.evidence_state)
    chunks = (
        (
            await session.execute(
                select(DocumentChunk)
                .join(Document)
                .where(Document.paper_id == paper.id)
                .limit(200)
            )
        )
        .scalars()
        .all()
        if full_text
        else []
    )
    candidates = [
        Chunk(
            chunk.text,
            chunk.page,
            chunk.section or "Body",
            chunk.chunk_index,
            chunk.embedding or [],
        )
        for chunk in chunks
    ]
    response = _evidence_response(request.question, candidates, full_text=full_text)
    if not full_text:
        response["answer"] = (
            "Full-text evidence is not available for this paper. " + INSUFFICIENT
        )
    session.add(
        ChatMessage(
            session_id=chat_session.id,
            role="user",
            content=request.question,
            citations=[],
        )
    )
    session.add(
        ChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content=str(response["answer"]),
            citations=response["citations"],
        )
    )
    await session.commit()
    return {
        "session_id": str(chat_session.id),
        "evidence_state": paper.evidence_state,
        **response,
        "retrieval": {
            "candidates": len(chunks),
            "selected": len(response["evidence"]),
            "reranker": response["validation"]["reranker"],
        },
    }


@app.get("/api/chat/{session_id}")
async def get_chat(
    session_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    chat_session = (
        await session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = (
        (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
        )
        .scalars()
        .all()
    )
    return {
        "id": str(chat_session.id),
        "paper_id": str(chat_session.paper_id),
        "messages": [
            {"role": item.role, "content": item.content, "citations": item.citations}
            for item in messages
        ],
    }


@app.get("/api/papers/{paper_id}/chats")
async def list_paper_chats(
    paper_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    paper = await session.get(Paper, paper_id)
    if not paper or (paper.owner_id is not None and paper.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Paper not found")
    result = await session.execute(
        select(ChatSession)
        .where(ChatSession.paper_id == paper_id, ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
    )
    return [{"id": str(item.id), "title": item.title} for item in result.scalars()]


@app.delete("/api/chat/{session_id}", status_code=204)
async def delete_chat(
    session_id: UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    chat_session = (
        await session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    await session.delete(chat_session)
    await session.commit()


@app.post("/api/compare")
async def compare(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    rows = [
        {
            "paper": paper.title,
            "year": paper.year,
            "venue": paper.venue,
            "citations": paper.citation_count,
            "authors": paper.authors,
            "evidence": paper.summary[:500] or "Not available in the supplied source.",
        }
        for paper in request.papers
    ]
    common = (
        set.intersection(*[_paper_terms(paper) for paper in request.papers])
        if request.papers
        else set()
    )
    return {
        "topic": request.topic,
        "papers": rows,
        "key_similarities": sorted(common)[:12]
        or ["No shared terms could be verified from the supplied abstracts."],
        "key_differences": "Compare the paper-specific methods and results in the evidence column; full-text details were not supplied for every paper.",
        "common_limitations": "Not available / not clearly stated in the supplied metadata.",
        "potential_research_opportunity": "AI-generated hypothesis: investigate the shared topic with a common evaluation protocol across these papers. This is not an established finding.",
    }


@app.post("/api/trends")
async def trends(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    years: dict[str, int] = {}
    terms: dict[str, int] = {}
    for paper in request.papers:
        if paper.year:
            years[str(paper.year)] = years.get(str(paper.year), 0) + 1
        for term in _paper_terms(paper):
            terms[term] = terms.get(term, 0) + 1
    return {
        "topic": request.topic,
        "publication_trend": years,
        "emerging_keywords": [
            term
            for term, _ in sorted(
                terms.items(), key=lambda item: item[1], reverse=True
            )[:15]
        ],
        "paper_count": len(request.papers),
        "notice": "Trend signals are computed only from the supplied paper metadata and abstracts.",
    }


@app.post("/api/research-gaps")
async def research_gaps(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    missing_abstracts = [
        paper.title for paper in request.papers if not paper.summary.strip()
    ]
    return {
        "topic": request.topic,
        "evidence_based_findings": [
            {
                "category": "Evidence coverage",
                "finding": f"{len(missing_abstracts)} of {len(request.papers)} selected papers lack an abstract in the supplied records.",
                "affected_papers": missing_abstracts,
                "confidence": 1.0,
            }
        ],
        "ai_generated_hypotheses": [
            {
                "category": "Evaluation gap",
                "finding": "A shared evaluation protocol may improve comparability across this collection.",
                "evidence": "Hypothesis generated from the collection structure; verify against full text.",
                "confidence": 0.35,
            }
        ],
    }


@app.post("/api/research-ideas")
async def research_ideas(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    return {
        "topic": request.topic,
        "label": "AI-generated research directions; novelty is not guaranteed",
        "ideas": [
            {
                "title": f"Robust evaluation for {request.topic}",
                "problem": "Results from different papers may not be directly comparable.",
                "proposed_contribution": "A reproducible benchmark with shared metrics and disclosed splits.",
                "evidence": [paper.title for paper in request.papers[:5]],
            }
        ],
    }


@app.post("/api/proposals")
async def proposal(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    references = [
        {"title": paper.title, "authors": paper.authors, "year": paper.year}
        for paper in request.papers
    ]
    return {
        "label": "AI-generated proposal draft",
        "title": f"A systematic study of {request.topic}",
        "problem_statement": f"Investigate reproducible progress in {request.topic} using evidence from the selected collection.",
        "methodology": "Define a shared dataset split, establish baselines, run ablations, and report confidence intervals.",
        "evaluation": "Use metrics appropriate to the task and report all dataset and preprocessing decisions.",
        "references": references,
    }


@app.post("/api/similarity-map")
async def similarity_map(
    request: CollectionRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _verify_paper_access(request.papers, user, session)
    nodes = [
        {
            "id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "authors": paper.authors,
        }
        for paper in request.papers
    ]
    vectors = {
        paper.id: embed(f"{paper.title} {paper.summary}") for paper in request.papers
    }
    edges = []
    for index, left in enumerate(request.papers):
        for right in request.papers[index + 1 :]:
            similarity = sum(
                a * b for a, b in zip(vectors[left.id], vectors[right.id], strict=True)
            )
            if similarity >= 0.25:
                edges.append(
                    {
                        "source": left.id,
                        "target": right.id,
                        "similarity": round(similarity, 4),
                    }
                )
    return {
        "nodes": nodes,
        "edges": edges,
        "notice": "Similarity is calculated from the supplied paper title and abstract embeddings.",
    }
