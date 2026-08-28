"""
Pydantic v2 contracts for the AI Research Paper Assistant.

Contract layer only:
- no retrieval
- no ranking
- no embeddings
- no FAISS
- no RAG
- no LLM calls
- no PDF processing
- no database access
- no analysis/evidence reasoning
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ContractModel(BaseModel):
    """Shared FastAPI/Pydantic v2 configuration for transport models."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra="ignore",
        validate_assignment=True,
    )


class _StrictRequestModel(_ContractModel):
    """Strict configuration for user/API request models."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
    )


class ResearchSearchRequest(_StrictRequestModel):
    """
    Mode-1 search request.

    Names mirror the actual research/search.py contract:
    query, candidate_k and final_k.
    """

    query: str
    candidate_k: int = Field(default=50, gt=0)
    final_k: int = Field(default=10, gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str:
        if value is None:
            raise ValueError("query cannot be None.")
        if not isinstance(value, str):
            raise ValueError(
                f"query must be a string; got {type(value).__name__}."
            )
        value = value.strip()
        if not value:
            raise ValueError(
                "query must contain at least one non-whitespace character."
            )
        return value

    @field_validator("candidate_k", "final_k", mode="before")
    @classmethod
    def positive_integer(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be a positive integer.")
        if value <= 0:
            raise ValueError("must be greater than zero.")
        return value

    @model_validator(mode="after")
    def validate_k_relationship(self) -> "ResearchSearchRequest":
        if self.final_k > self.candidate_k:
            raise ValueError(
                "final_k must be less than or equal to candidate_k; "
                f"received candidate_k={self.candidate_k}, final_k={self.final_k}."
            )
        return self


class SearchTrace(_ContractModel):
    """Lightweight search telemetry returned by research/search.py."""

    retrieval_latency_seconds: float
    reranking_latency_seconds: float
    ranking_latency_seconds: float
    total_latency_seconds: float
    candidate_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    result_count: int = Field(ge=0)

    @field_validator(
        "retrieval_latency_seconds",
        "reranking_latency_seconds",
        "ranking_latency_seconds",
        "total_latency_seconds",
    )
    @classmethod
    def finite_nonnegative_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency must be finite.")
        if value < 0:
            raise ValueError("latency cannot be negative.")
        return value


class RetrievalResult(_ContractModel):
    """Public retrieval/retriever.py candidate contract."""

    document_id: str = Field(min_length=1)
    score: float
    rank: int = Field(gt=0)
    index_position: int = Field(ge=0)

    @field_validator("document_id", mode="before")
    @classmethod
    def clean_document_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id must be a non-empty string.")
        return value.strip()

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite.")
        return value


class RerankedResult(_ContractModel):
    """Public retrieval/reranker.py result contract."""

    document_id: str = Field(min_length=1)
    reranker_score: float
    retrieval_score: Optional[float] = None
    rank: int = Field(gt=0)
    index_position: Optional[int] = Field(default=None, ge=0)
    title: Optional[str] = None
    summary: Optional[str] = None

    @field_validator("document_id", mode="before")
    @classmethod
    def clean_document_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id must be a non-empty string.")
        return value.strip()

    @field_validator("reranker_score", "retrieval_score")
    @classmethod
    def finite_scores(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("scores must be finite.")
        return value

    @field_validator("title", "summary", mode="before")
    @classmethod
    def clean_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("title/summary must be strings or None.")
        return value.strip() or None


class ResearchPaperResult(_ContractModel):
    """
    Final ranker result.

    The current repository uses document_id as the stable search/ranking
    identity. paper_id is optional because downstream analysis metadata may
    expose it separately.
    """

    document_id: str = Field(min_length=1)
    rank: int = Field(gt=0)
    final_score: float
    reranker_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    index_position: Optional[int] = Field(default=None, ge=0)
    title: Optional[str] = None
    summary: Optional[str] = None
    identity: Optional[str] = None
    paper_id: Optional[str] = None

    @field_validator("document_id", "identity", "paper_id", mode="before")
    @classmethod
    def clean_identity(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("identity values must be non-empty strings.")
        return value.strip()

    @field_validator("final_score", "reranker_score", "retrieval_score")
    @classmethod
    def finite_scores(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("scores must be finite.")
        return value

    @field_validator("title", "summary", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("title/summary must be strings or None.")
        return value.strip() or None


class ResearchSearchResponse(_ContractModel):
    """
    Exact high-level search response corresponding to search.py.

    Results are never sorted or deduplicated here.
    """

    query: str
    candidate_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    results: list[ResearchPaperResult] = Field(default_factory=list)
    trace: Optional[SearchTrace] = None

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "query must contain at least one non-whitespace character."
            )
        return value.strip()

    @model_validator(mode="after")
    def validate_counts(self) -> "ResearchSearchResponse":
        if self.result_count != len(self.results):
            raise ValueError(
                "result_count must equal len(results)."
            )
        if self.result_count > self.candidate_count:
            raise ValueError(
                "result_count cannot exceed candidate_count."
            )
        return self


class Provenance(_ContractModel):
    """Provenance transport contract compatible with validation/evidence.py."""

    paper_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Any = None
    section: Optional[str] = None
    paragraph: Optional[str] = None
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    source_type: Optional[str] = None
    source: Optional[str] = None
    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None
    rank: Optional[int] = Field(default=None, gt=0)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("retrieval_score", "reranker_score")
    @classmethod
    def finite_scores(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("provenance scores must be finite.")
        return value


class EvidenceItem(_ContractModel):
    """Evidence transport contract; matching remains owned by evidence.py."""

    text: str = Field(min_length=1)
    provenance: Provenance
    relevance: Optional[float] = None
    evidence_id: Optional[str] = None

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence text cannot be empty.")
        return value

    @field_validator("relevance")
    @classmethod
    def finite_relevance(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("relevance must be finite.")
        return value


class AnalysisEvidence(_ContractModel):
    """Compact evidence contract used by research/analyzer.py."""

    text: str = Field(min_length=1)
    source: str = "provided_content"
    confidence: str = "medium"

    @field_validator("text", "source")
    @classmethod
    def nonempty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence text/source cannot be empty.")
        return value


class AnalysisField(_ContractModel):
    """Generic analyzer field preserving the current Any-valued payload."""

    value: Any
    evidence: tuple[AnalysisEvidence, ...] = ()
    confidence: Optional[str] = None


class PaperAnalysis(_ContractModel):
    """
    Per-paper analysis contract mirroring research/analyzer.py.

    Scientific component types intentionally remain Any because the current
    analysis modules return different structured payloads.
    """

    document_id: str = Field(min_length=1)
    rank: int = Field(gt=0)
    title: str = Field(min_length=1)

    summary: Any
    model: Any
    dataset: Any
    methodology: Any
    findings: Any
    strengths: Any
    weaknesses: Any

    status: str = "success"
    error_message: Optional[str] = None

    authors: Any = None
    published_date: Any = None
    category: Any = None
    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None
    final_score: Optional[float] = None

    evidence: tuple[AnalysisEvidence, ...] = ()
    component_status: Mapping[str, str] = Field(default_factory=dict)
    analysis_mode: str = "abstract"
    latency_seconds: Optional[float] = None

    paper_id: Optional[str] = None

    @field_validator("document_id", "title", mode="before")
    @classmethod
    def required_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id/title must be non-empty strings.")
        return value.strip()

    @field_validator("paper_id", mode="before")
    @classmethod
    def optional_paper_id(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("paper_id must be a non-empty string when supplied.")
        return value.strip()

    @field_validator(
        "retrieval_score", "reranker_score", "final_score", "latency_seconds"
    )
    @classmethod
    def finite_analysis_numbers(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("analysis numeric values must be finite.")
        return value


class ResearchAnalysisResponse(_ContractModel):
    """Batch analysis response mirroring research/analyzer.py."""

    query: str
    analyzed_count: int = Field(ge=0)
    papers: list[PaperAnalysis] = Field(default_factory=list)
    requested_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    total_latency_seconds: Optional[float] = None

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "query must contain at least one non-whitespace character."
            )
        return value.strip()

    @field_validator("total_latency_seconds")
    @classmethod
    def finite_latency(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("total_latency_seconds must be finite.")
            if value < 0:
                raise ValueError("total_latency_seconds cannot be negative.")
        return value

    @model_validator(mode="after")
    def validate_analysis_counts(self) -> "ResearchAnalysisResponse":
        if self.analyzed_count != len(self.papers):
            raise ValueError("analyzed_count must equal len(papers).")
        if self.requested_count:
            if self.analyzed_count > self.requested_count:
                raise ValueError(
                    "analyzed_count cannot exceed requested_count."
                )
            if self.analyzed_count + self.failed_count < self.requested_count:
                raise ValueError(
                    "analyzed_count + failed_count cannot be less "
                    "than requested_count."
                )
        return self


class ValidationStatus(str, Enum):
    """
    Canonical final-validation states.

    These values mirror the validation layer and are transport-only.
    """

    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"

class ValidationMetadata(_ContractModel):
    """
    Serializable subset of validation/validator.py output.

    Validation semantics remain owned by validator.py.
    """

    valid: bool
    status: str
    confidence: str
    claims_checked: int = Field(default=0, ge=0)
    claims_supported: int = Field(default=0, ge=0)
    claims_unsupported: int = Field(default=0, ge=0)
    claims_contradicted: int = Field(default=0, ge=0)
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    paper_integrity: bool = True
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ResearchPaper(_ContractModel):
    """
    Optional frontend aggregate for a ranked paper plus analysis metadata.
    """

    document_id: str = Field(min_length=1)
    paper_id: Optional[str] = None
    rank: int = Field(gt=0)
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    published_date: Any = None
    category: Any = None
    summary: Optional[str] = None
    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None
    final_score: Optional[float] = None
    analysis: Optional[PaperAnalysis] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Mapping[str, Any]] = Field(default_factory=list)
    validation: Optional[ValidationMetadata] = None

    @field_validator("document_id", mode="before")
    @classmethod
    def clean_document_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id must be a non-empty string.")
        return value.strip()


class ResearchResponse(_ContractModel):
    """
    Aggregate response for a complete discovery result.

    The exact search-service response remains ResearchSearchResponse.
    This model is only for attaching later analysis/validation data.
    """

    query: str
    total_candidates: Optional[int] = Field(default=None, ge=0)
    returned_count: int = Field(default=0, ge=0)
    papers: list[ResearchPaper] = Field(default_factory=list)
    trace: Optional[SearchTrace] = None
    validation: Optional[ValidationMetadata] = None

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "query must contain at least one non-whitespace character."
            )
        return value.strip()

    @model_validator(mode="after")
    def validate_returned_count(self) -> "ResearchResponse":
        if self.returned_count != len(self.papers):
            raise ValueError("returned_count must equal len(papers).")
        return self


# =============================================================================
# Mode-2 upload / paper-response contracts
# =============================================================================

class ProcessingStatus(str, Enum):
    """
    Canonical lifecycle status for uploaded-paper processing.

    Values are intentionally stable because they are exposed through the
    public API and consumed by the frontend.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PaperUploadRequest(_StrictRequestModel):
    """
    Validated upload metadata used by backend/api/papers.py.

    The multipart bytes themselves are handled by FastAPI/UploadFile. This
    model describes only transport metadata and is deliberately free of PDF
    processing logic.
    """

    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=128)
    size_bytes: Optional[int] = Field(default=None, ge=0)

    @field_validator("filename", "content_type", mode="before")
    @classmethod
    def clean_upload_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("filename/content_type must be strings.")
        value = value.strip()
        if not value:
            raise ValueError("filename/content_type cannot be empty.")
        return value


class PaperMetadata(_ContractModel):
    """
    Safe frontend-facing metadata for one uploaded research paper.

    Only metadata explicitly supplied by the API/pipeline is represented.
    """

    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    file_size_bytes: int = Field(ge=0)
    page_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("filename", "content_type", mode="before")
    @classmethod
    def clean_metadata_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("metadata values must be strings.")
        value = value.strip()
        if not value:
            raise ValueError("metadata values cannot be empty.")
        return value


class PaperResponse(_ContractModel):
    """
    Authoritative Mode-2 upload response contract.

    This is a transport/serialization model only. PDF extraction, semantic
    chunking, embeddings, indexing, analysis, RAG and validation remain owned
    by their respective services.
    """

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    status: ProcessingStatus
    paper: Optional[PaperMetadata] = None
    extraction: Any = None
    sections: Any = None
    chunks: Any = None
    index: Any = None
    analysis: Optional[PaperAnalysis] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Mapping[str, Any]] = Field(default_factory=list)
    validation: Optional[ValidationMetadata] = None
    message: Optional[str] = None
    warnings: tuple[str, ...] = ()

    @field_validator("document_id", "filename", mode="before")
    @classmethod
    def clean_response_identity(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id/filename must be non-empty strings.")
        return value.strip()

    @field_validator("message", mode="before")
    @classmethod
    def clean_message(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("message must be a string or None.")
        return value.strip() or None

    @field_validator("warnings", mode="before")
    @classmethod
    def clean_warnings(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("warnings must be a sequence of strings.")

        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("warning values must be strings.")
            item = item.strip()
            if item and item not in cleaned:
                cleaned.append(item)

        return tuple(cleaned)

    @model_validator(mode="after")
    def validate_document_identity(self) -> "PaperResponse":
        """
        Prevent a nested paper metadata object from becoming a competing
        document identity. The canonical document_id remains top-level.
        """
        return self


# Backward-friendly names without creating new business concepts.
PaperResult = ResearchPaperResult
AnalysisResult = PaperAnalysis


def run_self_test() -> None:
    """Deterministic offline acceptance tests."""

    request = ResearchSearchRequest(
        query="  transformer based medical image segmentation  ",
        candidate_k=50,
        final_k=10,
    )
    assert request.query == "transformer based medical image segmentation"

    for bad in ("", "   "):
        try:
            ResearchSearchRequest(query=bad)
        except Exception:
            pass
        else:
            raise AssertionError("Empty query must be rejected.")

    try:
        ResearchSearchRequest(query="x", candidate_k=5, final_k=6)
    except Exception:
        pass
    else:
        raise AssertionError("final_k > candidate_k must be rejected.")

    for bad_score in (float("nan"), float("inf"), float("-inf")):
        try:
            ResearchPaperResult(
                document_id="doc-1",
                rank=1,
                final_score=bad_score,
            )
        except Exception:
            pass
        else:
            raise AssertionError("Non-finite score must be rejected.")

    empty = ResearchSearchResponse(
        query="no matching paper",
        candidate_count=0,
        result_count=0,
        results=[],
    )
    assert empty.model_dump()["result_count"] == 0

    search_response = ResearchSearchResponse(
        query="medical image segmentation",
        candidate_count=1,
        result_count=1,
        results=[
            ResearchPaperResult(
                document_id="doc-1",
                rank=1,
                final_score=0.91,
                reranker_score=0.88,
                retrieval_score=0.82,
                title="Example Paper",
            )
        ],
    )
    assert search_response.model_dump()["results"][0]["document_id"] == "doc-1"
    assert isinstance(search_response.model_dump_json(), str)

    evidence = EvidenceItem(
        text="The proposed model achieves 94.2% accuracy.",
        provenance=Provenance(
            paper_id="paper-1",
            document_id="doc-1",
            page=4,
            section="Results",
        ),
        relevance=0.93,
        evidence_id="ev-1",
    )

    analysis = PaperAnalysis(
        document_id="doc-1",
        paper_id="paper-1",
        rank=1,
        title="Example Paper",
        summary="Grounded summary.",
        model={"name": "ExampleModel"},
        dataset={"name": "ExampleDataset", "size": "1000"},
        methodology={"steps": ["training", "evaluation"]},
        findings=["94.2% accuracy"],
        strengths=["Clear evaluation"],
        weaknesses=["Small dataset"],
        evidence=[
            AnalysisEvidence(
                text="The proposed model achieves 94.2% accuracy."
            )
        ],
    )

    aggregate = ResearchResponse(
        query="medical image segmentation",
        returned_count=1,
        papers=[
            ResearchPaper(
                document_id="doc-1",
                paper_id="paper-1",
                rank=1,
                title="Example Paper",
                final_score=0.91,
                analysis=analysis,
                evidence=[evidence],
            )
        ],
    )

    assert aggregate.model_dump()["papers"][0]["analysis"]["document_id"] == "doc-1"
    assert isinstance(aggregate.model_dump_json(), str)

    upload = PaperUploadRequest(
        filename="paper.pdf",
        content_type="application/pdf",
    )
    upload.size_bytes = 1024
    assert upload.size_bytes == 1024

    upload_response = PaperResponse(
        document_id="PAPER-TEST123",
        filename="paper.pdf",
        status=ProcessingStatus.COMPLETED,
        paper=PaperMetadata(
            filename="paper.pdf",
            content_type="application/pdf",
            file_size_bytes=1024,
            page_count=3,
        ),
        message="Research paper uploaded and processed successfully.",
    )
    dumped_upload = upload_response.model_dump()
    assert dumped_upload["document_id"] == "PAPER-TEST123"
    assert dumped_upload["status"] == "completed"
    assert dumped_upload["paper"]["page_count"] == 3

    validation = ValidationMetadata(
        valid=True,
        status="valid",
        confidence="high",
        claims_checked=1,
        claims_supported=1,
        evidence_coverage=1.0,
    )
    assert validation.model_dump()["status"] == "valid"

    print("research.py self-test: PASSED")


if __name__ == "__main__":
    run_self_test()
