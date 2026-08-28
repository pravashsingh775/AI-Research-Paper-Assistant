"""
Mode-2 "My Paper" HTTP gateway.

This router owns only:
    multipart upload validation
    safe temporary staging
    dependency resolution
    PaperUploadPipeline invocation
    PaperResponse serialization

It deliberately does NOT implement PDF extraction, chunking, embeddings,
FAISS, RAG, LLM calls, analysis, or evidence validation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi import status as http_status
from pydantic import ValidationError

from backend.schemas.paper import PaperResponse, PaperUploadRequest, ProcessingStatus

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/papers",
    tags=["My Papers"],
)

# Keep upload policy configurable and aligned with the current application's
# existing configuration. The current main.py uses the same environment
# variable and 50 MB default.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
if MAX_UPLOAD_SIZE_MB <= 0:
    raise RuntimeError("MAX_UPLOAD_SIZE_MB must be greater than zero.")

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_PDF_EXTENSIONS = frozenset({".pdf"})
PDF_MAGIC = b"%PDF-"
READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# The existing application stages uploaded files under this directory.
# A per-request random directory prevents filename collisions and traversal.
UPLOAD_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "uploaded_papers"
)
STAGING_ROOT = UPLOAD_ROOT / "_staging"


class PaperUploadGatewayError(RuntimeError):
    """Base error raised by the API gateway's integration boundary."""


class PaperPipelineUnavailableError(PaperUploadGatewayError):
    """Raised when the application has not registered its upload pipeline."""


class PaperPipelineContractError(PaperUploadGatewayError):
    """Raised when the existing pipeline returns an invalid contract."""


def get_paper_upload_pipeline(request: Request) -> Any:
    """
    Resolve the already-initialized upload pipeline.

    Preferred application lifecycle contract:
        request.app.state.paper_upload_pipeline

    ``paper_indexer`` is accepted as a compatibility alias because some
    project revisions register the Mode-2 service under that name.

    The API never constructs the pipeline here. This is important because the
    pipeline owns expensive resources such as the embedding/index stack.
    """
    for attr_name in ("paper_upload_pipeline", "paper_indexer"):
        pipeline = getattr(request.app.state, attr_name, None)
        if pipeline is not None:
            process = getattr(pipeline, "process_uploaded_file", None)
            if callable(process):
                return pipeline

    raise PaperPipelineUnavailableError(
        "Mode-2 paper upload pipeline is not registered in app.state."
    )


def _safe_display_filename(filename: str | None) -> str:
    """
    Convert an untrusted multipart filename into a safe basename.

    The resulting value is used only as display/original-file metadata and as
    the leaf filename inside a random staging directory.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    raw = filename.replace("\x00", "")
    # Handle both POSIX and Windows separators even when the server runs on
    # the other platform.
    raw = raw.replace("\\", "/")
    safe = Path(raw).name.strip()

    if not safe or safe in {".", ".."}:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    if "/" in safe or "\\" in safe:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    return safe


def _validate_pdf_metadata(filename: str, content_type: str | None) -> None:
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_PDF_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF research papers are supported.",
        )

    normalized_content_type = (
        (content_type or "").split(";", 1)[0].strip().lower()
    )
    if normalized_content_type != "application/pdf":
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must use application/pdf content type.",
        )


async def _stage_upload(
    file: UploadFile,
    *,
    display_filename: str,
) -> tuple[Path, int]:
    """
    Stream the upload to a private staging directory.

    The complete PDF is never logged and the API never trusts the client's
    filename as a directory path.
    """
    request_dir = STAGING_ROOT / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=False)

    staged_path = request_dir / display_filename
    total = 0

    try:
        with staged_path.open("wb") as destination:
            first_chunk = True

            while True:
                chunk = await file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break

                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(PDF_MAGIC):
                        raise HTTPException(
                            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Uploaded file is not a valid PDF.",
                        )

                total += len(chunk)

                if total > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "PDF exceeds the maximum allowed size of "
                            f"{MAX_UPLOAD_SIZE_MB} MB."
                        ),
                    )

                destination.write(chunk)

        if total == 0:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )

        return staged_path, total

    except BaseException:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise


async def _call_pipeline(
    pipeline: Any,
    staged_pdf: Path,
) -> Any:
    """
    Invoke the actual existing PaperUploadPipeline interface.

    The discovered project contract is:
        process_uploaded_file(file_path) -> dict

    Both synchronous and accidentally-awaitable implementations are accepted
    without changing the pipeline itself.
    """
    method = getattr(pipeline, "process_uploaded_file", None)
    if not callable(method):
        raise PaperPipelineContractError(
            "Paper upload pipeline does not expose "
            "process_uploaded_file(file_path)."
        )

    result = await asyncio.to_thread(
        method,
        str(staged_pdf),
    )

    if inspect.isawaitable(result):
        result = await result

    if not isinstance(result, dict):
        raise PaperPipelineContractError(
            "PaperUploadPipeline.process_uploaded_file() must return a dict."
        )

    return result


def _extract_canonical_document_id(result: dict[str, Any]) -> str:
    """
    Extract the pipeline-generated canonical PAPER-* identity.

    The API never generates a competing document identifier.
    """
    paper = result.get("paper")
    if isinstance(paper, dict):
        paper_id = paper.get("paper_id")
        if isinstance(paper_id, str) and paper_id.strip():
            return paper_id.strip()

    paper_id = result.get("paper_id")
    if isinstance(paper_id, str) and paper_id.strip():
        return paper_id.strip()

    raise PaperPipelineContractError(
        "PaperUploadPipeline did not return a canonical paper_id."
    )


def _paper_response_from_pipeline(
    result: dict[str, Any],
    *,
    filename: str,
    content_type: str,
    size_bytes: int,
    processing_time_ms: float,
) -> PaperResponse:
    """
    Adapt the existing pipeline output into the authoritative PaperResponse.

    Only fields actually supplied by the pipeline are mapped. No page count,
    chunk count, index metadata, analysis claims, or evidence are fabricated.
    """
    document_id = _extract_canonical_document_id(result)

    raw_paper = result.get("paper")
    paper_data = dict(raw_paper) if isinstance(raw_paper, dict) else {}

    # ``PaperMetadata`` intentionally contains only frontend metadata. The
    # current pipeline exposes pages under ``pages`` and original_filename.
    paper_metadata = {
        "filename": paper_data.get("original_filename", filename),
        "content_type": content_type,
        "file_size_bytes": size_bytes,
    }

    if paper_data.get("pages") is not None:
        paper_metadata["page_count"] = paper_data["pages"]

    # The current pipeline's analysis dataclass is not automatically assumed
    # to be the newer structured PaperAnalysis contract. If it already matches
    # the contract, Pydantic will accept it; otherwise we leave it absent rather
    # than silently converting or inventing fields.
    analysis = None
    raw_analysis = result.get("analysis")
    if raw_analysis is not None:
        try:
            from backend.schemas.paper import PaperAnalysis

            analysis = PaperAnalysis.model_validate(
                raw_analysis,
                from_attributes=True,
            )
        except ValidationError:
            logger.warning(
                "Pipeline analysis for document_id=%s does not match "
                "PaperAnalysis transport schema; returning indexing metadata "
                "without fabricating analysis fields.",
                document_id,
            )

    duplicate = bool(result.get("duplicate", False))
    message = (
        "Research paper already exists."
        if duplicate
        else "Research paper uploaded and processed successfully."
    )

    return PaperResponse(
        document_id=document_id,
        filename=str(paper_metadata["filename"]),
        status=ProcessingStatus.COMPLETED,
        paper=paper_metadata,
        analysis=analysis,
        message=message,
        warnings=tuple(
            ["duplicate_upload"]
            if duplicate
            else []
        ),
    )


@router.post(
    "/upload",
    response_model=PaperResponse,
    response_model_exclude_none=True,
    status_code=http_status.HTTP_201_CREATED,
    summary="Upload and index one research paper",
    description=(
        "Accept one PDF research paper, validate it safely, pass it to the "
        "existing Mode-2 PaperUploadPipeline, and return the canonical "
        "document identity and typed PaperResponse. PDF extraction, chunking, "
        "embedding, vector indexing, and analysis remain inside the existing "
        "pipeline."
    ),
    responses={
        400: {"description": "Invalid or empty upload."},
        413: {"description": "PDF exceeds the configured size limit."},
        415: {"description": "Unsupported file type or invalid PDF content."},
        500: {"description": "Unexpected paper-processing failure."},
        503: {"description": "Mode-2 paper pipeline is unavailable."},
    },
)
async def upload_paper(
    request: Request,
    file: UploadFile = File(..., description="PDF research paper."),
    pipeline: Any = Depends(get_paper_upload_pipeline),
) -> PaperResponse:
    """Thin HTTP gateway for the existing Mode-2 ingestion pipeline."""
    started = time.perf_counter()

    display_filename = _safe_display_filename(file.filename)
    _validate_pdf_metadata(display_filename, file.content_type)

    # Validate the upload metadata against the project's authoritative
    # Pydantic contract as well as the HTTP-layer checks above.
    try:
        upload_metadata = PaperUploadRequest(
            filename=display_filename,
            content_type="application/pdf",
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid PDF upload metadata.",
        ) from exc

    staged_pdf: Path | None = None
    staging_dir: Path | None = None

    try:
        staged_pdf, size_bytes = await _stage_upload(
            file,
            display_filename=upload_metadata.filename,
        )
        upload_metadata.size_bytes = size_bytes
        staging_dir = staged_pdf.parent

        logger.info(
            "Mode-2 paper upload accepted: filename=%s size_bytes=%d",
            display_filename,
            size_bytes,
        )

        result = await _call_pipeline(
            pipeline,
            staged_pdf,
        )

        processing_time_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        response = _paper_response_from_pipeline(
            result,
            filename=display_filename,
            content_type="application/pdf",
            size_bytes=size_bytes,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Mode-2 paper upload completed: document_id=%s "
            "duplicate=%s latency_ms=%.2f",
            response.document_id,
            "duplicate_upload" in response.warnings,
            processing_time_ms,
        )

        return response

    except HTTPException:
        raise

    except PaperPipelineUnavailableError as exc:
        logger.error("Mode-2 pipeline unavailable: %s", exc)
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper processing service is unavailable.",
        ) from exc

    except PaperPipelineContractError as exc:
        logger.exception("Mode-2 pipeline contract failure.")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper processing service returned an invalid result.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Mode-2 paper processing failed for filename=%s.",
            display_filename,
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper upload/processing failed.",
        ) from exc

    finally:
        await file.close()

        # The existing pipeline copies/owns the persistent paper artifacts.
        # Only the API's temporary staging directory is removed here.
        if staging_dir is not None:
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


# ---------------------------------------------------------------------------
# Model-free contract tests
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """
    Fast, dependency-free API gateway tests.

    These tests intentionally do not load SPECTER2, FAISS, or an LLM.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class FakePipeline:
        calls = 0
        received_path: str | None = None

        def process_uploaded_file(self, file_path: str) -> dict[str, Any]:
            type(self).calls += 1
            type(self).received_path = file_path

            return {
                "success": True,
                "duplicate": False,
                "paper": {
                    "paper_id": "PAPER-TEST123",
                    "original_filename": "paper.pdf",
                    "pages": 3,
                },
            }

    fake_pipeline = FakePipeline()

    app = FastAPI()
    app.state.paper_upload_pipeline = fake_pipeline
    app.include_router(router)

    client = TestClient(app)

    valid_pdf = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"

    response = client.post(
        "/api/papers/upload",
        files={
            "file": (
                "../../paper.pdf",
                valid_pdf,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201, response.text

    payload = response.json()

    assert payload["document_id"] == "PAPER-TEST123"
    assert payload["filename"] == "paper.pdf"
    assert payload["status"] == "completed"
    assert payload["paper"]["page_count"] == 3
    assert payload["paper"]["content_type"] == "application/pdf"
    assert payload["paper"]["file_size_bytes"] == len(valid_pdf)

    assert FakePipeline.calls == 1
    assert fake_pipeline.received_path is not None
    assert "paper.pdf" in fake_pipeline.received_path
    assert "../../" not in fake_pipeline.received_path

    # Missing filename.
    missing_filename = client.post(
        "/api/papers/upload",
        files={"file": ("", valid_pdf, "application/pdf")},
    )
    assert missing_filename.status_code in {400, 422}

    # Unsupported extension.
    bad_extension = client.post(
        "/api/papers/upload",
        files={"file": ("paper.txt", valid_pdf, "application/pdf")},
    )
    assert bad_extension.status_code == 415

    # Wrong content type.
    bad_mime = client.post(
        "/api/papers/upload",
        files={"file": ("paper.pdf", valid_pdf, "text/plain")},
    )
    assert bad_mime.status_code == 415

    # Empty file.
    empty = client.post(
        "/api/papers/upload",
        files={"file": ("paper.pdf", b"", "application/pdf")},
    )
    assert empty.status_code == 400

    # Non-PDF bytes.
    not_pdf = client.post(
        "/api/papers/upload",
        files={"file": ("paper.pdf", b"hello", "application/pdf")},
    )
    assert not_pdf.status_code == 415

    # No pipeline registered.
    unavailable_app = FastAPI()
    unavailable_app.include_router(router)
    unavailable_client = TestClient(unavailable_app)

    unavailable = unavailable_client.post(
        "/api/papers/upload",
        files={"file": ("paper.pdf", valid_pdf, "application/pdf")},
    )
    assert unavailable.status_code == 503

    # OpenAPI contract.
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/papers/upload" in openapi.json()["paths"]

    print("backend/api/papers.py self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()