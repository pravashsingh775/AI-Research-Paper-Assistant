"""
FastAPI API gateway for Mode-2 uploaded-paper question answering.

Responsibilities
----------------
HTTP request validation
    -> shared RAG pipeline resolution
    -> document-scoped QA execution
    -> typed QAResponse serialization
    -> safe pipeline-error-to-HTTP mapping

This module intentionally does NOT implement:
    - retrieval
    - embeddings
    - FAISS loading
    - PDF processing
    - chunking
    - context construction
    - prompt construction
    - LLM calls
    - answer validation algorithms

Those responsibilities belong to the shared RAG pipeline and its injected
dependencies.

Architecture
------------
POST /api/qa
    |
    v
QARequest
    |
    v
get_rag_pipeline()
    |
    v
RAGPipeline.run(
    query,
    task_type="qa",
    scope="uploaded",
    document_id=<requested paper>
)
    |
    v
PipelineResponse
    |
    v
document/provenance validation
    |
    v
QAResponse
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from backend.rag.pipeline import (
    PipelineConfigurationError,
    PipelineDependencyError,
    PipelineRequestError,
    PipelineStageError,
)
from backend.schemas.qa import QARequest, QAResponse


LOGGER = logging.getLogger(__name__)

QA_API_VERSION = "2.5.0"


router = APIRouter(
    prefix="/api",
    tags=["Paper QA"],
)


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


class RAGPipelineProtocol(Protocol):
    """
    Minimal interface required by the API gateway.

    The concrete implementation lives in backend.rag.pipeline.
    """

    def run(
        self,
        query: str,
        task_type: str,
        *,
        scope: str,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# Generic serialization helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """
    Convert project-owned objects into ordinary Python JSON-compatible
    structures.

    Supported:
        - None
        - Enum
        - Pydantic v2 models
        - dataclasses
        - Mapping
        - list/tuple/set
        - ordinary objects exposing __dict__

    This function does not modify scientific content.
    """

    if value is None:
        return None

    if isinstance(value, Enum):
        return _to_jsonable(value.value)

    # Pydantic v2.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(model_dump())
        except Exception:
            LOGGER.debug(
                "model_dump() failed for %s",
                type(value).__name__,
                exc_info=True,
            )

    # Dataclass.
    if is_dataclass(value):
        try:
            return _to_jsonable(asdict(value))
        except Exception:
            LOGGER.debug(
                "asdict() failed for %s",
                type(value).__name__,
                exc_info=True,
            )

    # Mapping.
    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    # Sequence/set.
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_jsonable(item) for item in value]

    # Ordinary project-owned object.
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): _to_jsonable(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:
            LOGGER.debug(
                "__dict__ serialization failed for %s",
                type(value).__name__,
                exc_info=True,
            )

    return value


def _normalized_string(value: Any) -> str:
    """Return a safely normalized string representation."""

    if value is None:
        return ""

    if isinstance(value, Enum):
        value = value.value

    return str(value).strip()


def _stage_value(value: Any) -> str:
    """
    Normalize pipeline stage values.

    Supports:
        "retrieval"
        PipelineStage.RETRIEVAL
        enum-like objects exposing .value
    """

    if value is None:
        return ""

    if isinstance(value, Enum):
        value = value.value

    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# RAG pipeline dependency
# ---------------------------------------------------------------------------


def _pipeline_is_usable(pipeline: Any) -> bool:
    """
    Validate the minimum runtime contract without constructing or probing
    heavyweight dependencies.

    A pipeline is considered usable when it exposes a callable run().
    """

    return callable(getattr(pipeline, "run", None))


def get_rag_pipeline(request: Request) -> RAGPipelineProtocol:
    """
    Resolve the already-initialized shared RAG pipeline.

    IMPORTANT:
        The API layer never creates:
            - RAGPipeline
            - embedding models
            - FAISS indexes
            - retrievers
            - LLM clients

        Application startup/lifespan owns those resources.

    Primary state key:
        app.state.rag_pipeline

    Compatibility fallback:
        app.state.qa_pipeline
    """

    app_state = request.app.state

    pipeline = getattr(app_state, "rag_pipeline", None)

    # Backward compatibility with older application wiring.
    if pipeline is None:
        pipeline = getattr(app_state, "qa_pipeline", None)

        if pipeline is not None:
            LOGGER.warning(
                "Using deprecated app.state.qa_pipeline alias. "
                "Prefer app.state.rag_pipeline."
            )

    if pipeline is None:
        LOGGER.error(
            "RAG pipeline is not registered in application state."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper QA service is not available.",
        )

    if not _pipeline_is_usable(pipeline):
        LOGGER.error(
            "Invalid RAG pipeline registered in application state: "
            "type=%s; callable_run=%s",
            type(pipeline).__name__,
            callable(getattr(pipeline, "run", None)),
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper QA service is not configured correctly.",
        )

    return pipeline


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


def _extract_field(payload: Mapping[str, Any], key: str) -> Any:
    """Safely extract a response field."""

    try:
        return payload.get(key)
    except Exception:
        return None


def _extract_validation_metadata(structured_output: Any) -> Any:
    """
    Preserve validation metadata supplied by the existing pipeline.

    No synthetic validation object is created.
    """

    if structured_output is None:
        return None

    if isinstance(structured_output, Mapping):
        return structured_output.get("validation")

    return getattr(structured_output, "validation", None)


def _extract_document_id(value: Any) -> Optional[str]:
    """
    Extract document_id from evidence/citation-like objects.
    """

    normalized = _to_jsonable(value)

    if not isinstance(normalized, Mapping):
        return None

    raw = normalized.get("document_id")

    if raw is None:
        return None

    result = str(raw).strip()

    return result or None


def _validate_document_provenance(
    *,
    request_document_id: str,
    field_name: str,
    values: Any,
) -> None:
    """
    Enforce strict document isolation.

    Evidence/citations that explicitly identify another document are rejected.

    Items without a document_id are not fabricated or rewritten here because
    the authoritative QAResponse schema owns the final schema validation.
    """

    if values is None:
        return

    normalized_values = _to_jsonable(values)

    if not isinstance(normalized_values, list):
        normalized_values = [normalized_values]

    for index, item in enumerate(normalized_values):
        document_id = _extract_document_id(item)

        if document_id is None:
            continue

        if document_id != request_document_id:
            raise QAResponseIntegrityError(
                f"{field_name}[{index}] belongs to document "
                f"{document_id!r}, requested {request_document_id!r}."
            )


# ---------------------------------------------------------------------------
# Pipeline response -> QAResponse
# ---------------------------------------------------------------------------


def _extract_structured_field(structured_output: Any, key: str) -> Any:
    """Read one field from a mapping or project-owned model/object."""
    if structured_output is None:
        return None

    normalized = _to_jsonable(structured_output)
    if isinstance(normalized, Mapping):
        return normalized.get(key)

    return getattr(structured_output, key, None)


def _normalize_structured_qa_output(
    *,
    structured_output: Any,
    requested_question: str,
) -> Any:
    """
    Enforce the canonical structured QA contract.

    Canonical contract:
        question: exact API question
        answer: model answer
        grounded: bool
        evidence: positive integer evidence references

    Provenance objects NEVER belong in structured_output.evidence.
    Trusted provenance is carried separately by PipelineResponse.evidence and
    PipelineResponse.citations.
    """
    normalized = _to_jsonable(structured_output)

    if not isinstance(normalized, Mapping):
        raise QAResponseIntegrityError(
            "RAG pipeline returned an invalid structured QA object."
        )

    result = dict(normalized)

    model_question = _normalized_string(result.get("question"))
    if not model_question:
        raise QAResponseIntegrityError(
            "RAG pipeline structured QA output is missing question."
        )

    if model_question != requested_question:
        raise QAResponseIntegrityError(
            "RAG pipeline structured QA question does not match the "
            "requested question."
        )

    evidence = result.get("evidence", ())
    if evidence is None:
        evidence = ()

    if isinstance(evidence, (str, bytes, Mapping)):
        raise QAResponseIntegrityError(
            "structured_output.evidence must contain integer evidence "
            "references, not provenance objects."
        )

    try:
        references = list(evidence)
    except TypeError as exc:
        raise QAResponseIntegrityError(
            "structured_output.evidence must be an iterable of integers."
        ) from exc

    invalid = [
        item
        for item in references
        if isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
    ]
    if invalid:
        raise QAResponseIntegrityError(
            "structured_output.evidence contains invalid evidence references."
        )

    # Preserve deterministic order while removing duplicate references.
    result["evidence"] = list(dict.fromkeys(references))
    result["question"] = requested_question

    return result


def _validate_evidence_reference_bounds(
    *,
    structured_output: Any,
    evidence: Any,
) -> None:
    """Ensure every model-selected evidence number points to trusted evidence."""
    references = _extract_structured_field(structured_output, "evidence")
    if references is None:
        return

    normalized_evidence = _to_jsonable(evidence) or ()
    try:
        evidence_count = len(normalized_evidence)
    except TypeError:
        raise QAResponseIntegrityError(
            "RAG pipeline returned invalid evidence provenance."
        )

    for reference in references:
        if reference > evidence_count:
            raise QAResponseIntegrityError(
                f"structured_output.evidence references unavailable evidence "
                f"number {reference}; only {evidence_count} evidence items exist."
            )


class QAResponseIntegrityError(ValueError):
    """Internal error raised when a pipeline response violates the API contract.

    The public API intentionally exposes only a stable validation message. The
    original exception is retained through normal exception chaining so the
    endpoint can log the complete traceback without leaking internals to clients.
    """


def _build_qa_response(
    *,
    request: QARequest,
    pipeline_response: Any,
) -> QAResponse:
    """
    Convert the shared PipelineResponse into the public QAResponse schema.

    Integration invariants:
      1. API request question is authoritative.
      2. Pipeline structured question must exactly match it.
      3. structured_output.evidence contains only integer references.
      4. PipelineResponse.evidence/citations contain trusted provenance.
      5. Evidence/citations must belong to the requested document.
      6. Evidence references must be in bounds.
    """
    payload = _to_jsonable(pipeline_response)

    if not isinstance(payload, Mapping):
        raise QAResponseIntegrityError(
            "RAG pipeline returned an unsupported response object."
        )

    requested_document_id = _normalized_string(
        getattr(request, "document_id", None)
    )
    requested_question = _normalized_string(
        getattr(request, "question", None)
    )

    if not requested_document_id:
        raise QAResponseIntegrityError("QA request contains an empty document_id.")

    if not requested_question:
        raise QAResponseIntegrityError("QA request contains an empty question.")

    response_document_id = _normalized_string(
        _extract_field(payload, "document_id")
    )

    if response_document_id and response_document_id != requested_document_id:
        raise QAResponseIntegrityError(
            "RAG pipeline returned a document_id inconsistent with the "
            "requested uploaded paper."
        )

    response_document_id = response_document_id or requested_document_id

    evidence = _extract_field(payload, "evidence") or ()
    citations = _extract_field(payload, "citations") or ()
    structured_output = _extract_field(payload, "structured_output")

    _validate_document_provenance(
        request_document_id=requested_document_id,
        field_name="evidence",
        values=evidence,
    )
    _validate_document_provenance(
        request_document_id=requested_document_id,
        field_name="citations",
        values=citations,
    )

    normalized_structured = _normalize_structured_qa_output(
        structured_output=structured_output,
        requested_question=requested_question,
    )

    _validate_evidence_reference_bounds(
        structured_output=normalized_structured,
        evidence=evidence,
    )

    # Cross-check the structured answer against the top-level answer when both
    # are present. Do not silently substitute one for the other.
    top_level_answer = _extract_field(payload, "answer")
    structured_answer = _extract_structured_field(
        normalized_structured,
        "answer",
    )

    if (
        top_level_answer is not None
        and structured_answer is not None
        and _normalized_string(top_level_answer)
        != _normalized_string(structured_answer)
    ):
        raise QAResponseIntegrityError(
            "Top-level answer and structured QA answer do not match."
        )

    response_payload: dict[str, Any] = {
        "document_id": response_document_id,
        "question": requested_question,
        "answer": (
            top_level_answer
            if top_level_answer is not None
            else structured_answer
        ),
        "status": _extract_field(payload, "status"),
        "success": bool(_extract_field(payload, "success")),
        "grounded": bool(_extract_field(payload, "grounded")),
        "evidence": evidence,
        "citations": citations,
        "structured_output": normalized_structured,
        "validation": _extract_validation_metadata(normalized_structured),
        "diagnostics": _extract_field(payload, "diagnostics"),
        "error": _extract_field(payload, "error"),
    }

    try:
        return QAResponse.model_validate(response_payload)
    except PydanticValidationError as exc:
        raise QAResponseIntegrityError(
            "RAG pipeline response does not satisfy QAResponse schema."
        ) from exc
    except Exception as exc:
        raise QAResponseIntegrityError(
            "Unexpected error while validating the QAResponse schema."
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline exception -> HTTP exception
# ---------------------------------------------------------------------------


def _map_pipeline_exception(exc: Exception) -> HTTPException:
    """
    Convert known pipeline failures into stable public HTTP semantics.

    Internal exception messages are deliberately not exposed.
    """

    if isinstance(exc, PipelineRequestError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid paper QA request.",
        )

    if isinstance(
        exc,
        (
            PipelineConfigurationError,
            PipelineDependencyError,
        ),
    ):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper QA service is not available.",
        )

    if isinstance(exc, PipelineStageError):
        stage = _stage_value(getattr(exc, "stage", None))

        # Document/index/retrieval availability is a resource-level failure.
        document_stages = {
            "retrieval",
            "retrieve",
            "index",
            "document",
            "uploaded",
            "uploaded_paper",
        }

        if stage in document_stages:
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "The requested uploaded paper is not available for QA."
                ),
            )

        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Paper QA processing is temporarily unavailable."
            ),
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Paper QA request failed.",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/qa",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about an uploaded research paper",
    description=(
        "Runs the shared Mode-2 evidence-grounded RAG pipeline against "
        "exactly one uploaded research paper. The document_id is mandatory "
        "and is passed unchanged to the uploaded-paper retrieval scope."
    ),
    responses={
        404: {
            "description": (
                "The requested uploaded paper or its index is unavailable."
            )
        },
        422: {
            "description": "The QA request is invalid."
        },
        503: {
            "description": (
                "The shared RAG, retrieval, LLM, or validation pipeline "
                "is unavailable."
            )
        },
    },
)
async def ask_paper_question(
    qa_request: QARequest,
    rag_pipeline: RAGPipelineProtocol = Depends(get_rag_pipeline),
) -> QAResponse:
    """
    Execute one document-scoped Mode-2 QA request.

    Fixed API-level values:
        task_type = "qa"
        scope = "uploaded"

    These values are not user-controlled.
    """

    started = time.perf_counter()

    document_id = _normalized_string(
        getattr(qa_request, "document_id", None)
    )

    question = _normalized_string(
        getattr(qa_request, "question", None)
    )

    if not document_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document_id is required.",
        )

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question is required.",
        )

    try:
        session_id = getattr(
            qa_request,
            "session_id",
            None,
        )

        # The API intentionally does not modify the scientific question.
        result = rag_pipeline.run(
            question,
            "qa",
            scope="uploaded",
            document_id=document_id,
            session_id=session_id,
        )

        # Support both synchronous and asynchronous pipeline implementations.
        if inspect.isawaitable(result):
            result = await result

        response = _build_qa_response(
            request=qa_request,
            pipeline_response=result,
        )

        duration = time.perf_counter() - started

        LOGGER.info(
            "Paper QA completed | document_id=%s | success=%s | "
            "grounded=%s | status=%s | duration=%.4fs",
            document_id,
            response.success,
            response.grounded,
            _normalized_string(response.status),
            duration,
        )

        # Insufficient evidence is a controlled application result.
        # It is not converted into a fabricated answer or infrastructure error.
        return response

    except HTTPException:
        raise

    except (
        PipelineRequestError,
        PipelineConfigurationError,
        PipelineDependencyError,
        PipelineStageError,
    ) as exc:

        LOGGER.warning(
            "Paper QA pipeline failure | document_id=%s | "
            "stage=%s | type=%s | duration=%.4fs",
            document_id,
            _stage_value(getattr(exc, "stage", None)),
            type(exc).__name__,
            time.perf_counter() - started,
        )

        raise _map_pipeline_exception(exc) from exc

    except QAResponseIntegrityError as exc:
        # This is an application-contract failure, not an ordinary pipeline
        # failure. Log the traceback internally so the root cause (for example
        # a Pydantic field/type mismatch) is visible during debugging, while
        # keeping implementation details out of the HTTP response.
        LOGGER.exception(
            "Paper QA response integrity failure | document_id=%s | "
            "type=%s | error=%s | duration=%.4fs",
            document_id,
            type(exc).__name__,
            str(exc),
            time.perf_counter() - started,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper QA response validation failed.",
        ) from exc

    except (TypeError, KeyError, AttributeError) as exc:
        # Defensive boundary for malformed third-party/project-owned response
        # objects. These are response-contract failures, not successful QA.
        LOGGER.exception(
            "Paper QA response construction failure | document_id=%s | "
            "type=%s | error=%s | duration=%.4fs",
            document_id,
            type(exc).__name__,
            str(exc),
            time.perf_counter() - started,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper QA response validation failed.",
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Unexpected Paper QA failure | document_id=%s | "
            "type=%s | duration=%.4fs",
            document_id,
            type(exc).__name__,
            time.perf_counter() - started,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper QA request failed.",
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic API contract self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """
    Run a lightweight API contract test.

    No FAISS, embedding model, SPECTER2 model, or LLM is loaded.

    This verifies:
        1. successful QA execution
        2. uploaded scope enforcement
        3. document ID propagation
        4. evidence provenance
        5. citation provenance
        6. request validation
        7. missing-pipeline handling
        8. wrong-document response protection
        9. wrong-document evidence protection
        10. pipeline stage -> HTTP mapping
        11. OpenAPI registration
    """

    # -----------------------------------------------------------------------
    # Fake response
    # -----------------------------------------------------------------------

    class FakePipelineResponse:
        def __init__(
            self,
            *,
            document_id: str,
            evidence_document_id: str,
        ) -> None:
            self.document_id = document_id
            self.status = "success"
            self.success = True
            self.grounded = True
            self.query = "What dataset was used?"
            self.task_type = "qa"
            self.scope = "uploaded"
            self.answer = "The paper uses CIFAR-10."

            self.structured_output = {
                "question": "What dataset was used?",
                "answer": "The paper uses CIFAR-10.",
                "grounded": True,
                "evidence": [1],
            }

            self.evidence = (
                {
                    "evidence_id": "ev_1",
                    "document_id": evidence_document_id,
                    "chunk_id": "chunk_1",
                    "section": "Dataset",
                    "page": 3,
                    "text": "The authors use CIFAR-10.",
                    "task_type": "qa",
                    "section_priority": 1,
                    "selection_order": 0,
                },
            )

            self.citations = (
                {
                    "evidence_index": 1,
                    "document_id": evidence_document_id,
                    "chunk_id": "chunk_1",
                    "section": "Dataset",
                    "page": 3,
                },
            )

            self.diagnostics = {
                "candidate_k": 50,
                "final_k": 10,
                "task_type": "qa",
                "scope": "uploaded",
                "candidate_count": 1,
                "evidence_count": 1,
                "session_scoped": False,
            }

            self.error = None

    # -----------------------------------------------------------------------
    # Fake pipeline
    # -----------------------------------------------------------------------

    class FakePipeline:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:

            self.calls.append(
                {
                    "query": query,
                    "task_type": task_type,
                    **kwargs,
                }
            )

            return FakePipelineResponse(
                document_id="paper_A",
                evidence_document_id="paper_A",
            )

    # -----------------------------------------------------------------------
    # Successful application
    # -----------------------------------------------------------------------

    app = FastAPI()

    fake = FakePipeline()

    app.state.rag_pipeline = fake

    app.include_router(router)

    client = TestClient(app)

    response = client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert response.status_code == 200, response.text

    payload = response.json()

    assert payload["document_id"] == "paper_A"
    assert payload["question"] == "What dataset was used?"
    assert payload["answer"] == "The paper uses CIFAR-10."
    assert payload["structured_output"]["question"] == "What dataset was used?"
    assert payload["structured_output"]["evidence"] == [1]
    assert all(
        isinstance(reference, int) and not isinstance(reference, bool)
        for reference in payload["structured_output"]["evidence"]
    )

    assert payload["evidence"]
    assert payload["citations"]

    assert payload["evidence"][0]["document_id"] == "paper_A"
    assert payload["citations"][0]["document_id"] == "paper_A"

    assert len(fake.calls) == 1

    assert fake.calls[0]["query"] == "What dataset was used?"
    assert fake.calls[0]["task_type"] == "qa"
    assert fake.calls[0]["scope"] == "uploaded"
    assert fake.calls[0]["document_id"] == "paper_A"

    # -----------------------------------------------------------------------
    # Invalid question
    # -----------------------------------------------------------------------

    bad_question = client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "   ",
        },
    )

    assert bad_question.status_code == 422

    # -----------------------------------------------------------------------
    # Missing document
    # -----------------------------------------------------------------------

    missing_document = client.post(
        "/api/qa",
        json={
            "question": "What dataset was used?",
        },
    )

    assert missing_document.status_code == 422

    # -----------------------------------------------------------------------
    # No pipeline registered
    # -----------------------------------------------------------------------

    unavailable_app = FastAPI()

    unavailable_app.include_router(router)

    unavailable_client = TestClient(
        unavailable_app
    )

    unavailable = unavailable_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert unavailable.status_code == 503

    # -----------------------------------------------------------------------
    # Structured question identity protection
    # -----------------------------------------------------------------------

    class WrongQuestionPipeline(FakePipeline):
        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:
            response = FakePipelineResponse(
                document_id="paper_A",
                evidence_document_id="paper_A",
            )
            response.structured_output["question"] = (
                "What is the relationship between two unrelated questions?"
            )
            return response

    wrong_question_app = FastAPI()
    wrong_question_app.state.rag_pipeline = WrongQuestionPipeline()
    wrong_question_app.include_router(router)
    wrong_question_client = TestClient(wrong_question_app)

    wrong_question = wrong_question_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert wrong_question.status_code == 500

    # -----------------------------------------------------------------------
    # Wrong-document pipeline response
    # -----------------------------------------------------------------------

    class WrongDocumentPipeline(FakePipeline):
        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:

            return FakePipelineResponse(
                document_id="paper_B",
                evidence_document_id="paper_B",
            )

    wrong_document_app = FastAPI()

    wrong_document_app.state.rag_pipeline = (
        WrongDocumentPipeline()
    )

    wrong_document_app.include_router(router)

    wrong_document_client = TestClient(
        wrong_document_app
    )

    wrong_document = wrong_document_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert wrong_document.status_code == 500

    # -----------------------------------------------------------------------
    # Wrong-document evidence
    # -----------------------------------------------------------------------

    class WrongEvidencePipeline(FakePipeline):
        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:

            return FakePipelineResponse(
                document_id="paper_A",
                evidence_document_id="paper_B",
            )

    wrong_evidence_app = FastAPI()

    wrong_evidence_app.state.rag_pipeline = (
        WrongEvidencePipeline()
    )

    wrong_evidence_app.include_router(router)

    wrong_evidence_client = TestClient(
        wrong_evidence_app
    )

    wrong_evidence = wrong_evidence_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert wrong_evidence.status_code == 500

    # -----------------------------------------------------------------------
    # Legacy structured evidence contract protection
    # -----------------------------------------------------------------------

    class LegacyEvidencePipeline(FakePipeline):
        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:
            response = FakePipelineResponse(
                document_id="paper_A",
                evidence_document_id="paper_A",
            )
            response.structured_output["evidence"] = [
                {"evidence_number": 1, "document_id": "paper_A"}
            ]
            return response

    legacy_evidence_app = FastAPI()
    legacy_evidence_app.state.rag_pipeline = LegacyEvidencePipeline()
    legacy_evidence_app.include_router(router)
    legacy_evidence_client = TestClient(legacy_evidence_app)

    legacy_evidence = legacy_evidence_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert legacy_evidence.status_code == 500

    # -----------------------------------------------------------------------
    # Pipeline exception mapping
    # -----------------------------------------------------------------------

    class MissingPaperPipeline:
        def run(
            self,
            query: str,
            task_type: str,
            **kwargs: Any,
        ) -> Any:

            # Construct a real PipelineStageError according to the project's
            # existing exception contract where possible.
            raise PipelineStageError(
                "Uploaded paper/index unavailable.",
                stage="retrieval",
            )

    missing_paper_app = FastAPI()

    missing_paper_app.state.rag_pipeline = (
        MissingPaperPipeline()
    )

    missing_paper_app.include_router(router)

    missing_paper_client = TestClient(
        missing_paper_app
    )

    missing_paper = missing_paper_client.post(
        "/api/qa",
        json={
            "document_id": "paper_A",
            "question": "What dataset was used?",
        },
    )

    assert missing_paper.status_code == 404

    # -----------------------------------------------------------------------
    # OpenAPI
    # -----------------------------------------------------------------------

    openapi = client.get("/openapi.json")

    assert openapi.status_code == 200

    openapi_payload = openapi.json()

    assert "/api/qa" in openapi_payload["paths"]

    print("backend/api/qa.py self-test: PASSED")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    run_self_test()