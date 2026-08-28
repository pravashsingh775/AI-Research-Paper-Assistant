"""
High-level research-paper summary analysis for the AI Research Paper Assistant.

Architectural boundary
----------------------
    rag/pipeline.py
        -> retrieval
        -> context
        -> prompt
        -> LLM client
        -> llm/parser.py
        -> validation

This module is intentionally a *normalization/analysis boundary* for summary
data. It does not perform retrieval, embeddings, FAISS access, prompt
construction, LLM calls, PDF reading, or evidence verification.

The existing prompt schema is the source of truth for LLM summary fields:
    summary
    problem
    methodology
    dataset
    model
    key_findings
    limitations
    grounded
    evidence

The analyzer preserves those fields and attaches paper/document provenance
needed by application and frontend layers.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Public constants
# ============================================================================

TASK_TYPE = "summary"
SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})

# These strings are deliberately conservative. They are status labels for
# normalization, not claims of factual correctness.
SUCCESS = "success"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
INVALID_INPUT = "invalid_input"
VALIDATION_ERROR = "validation_error"


# ============================================================================
# Exceptions
# ============================================================================


class SummaryError(RuntimeError):
    """Base summary-analysis error."""


class SummaryInputError(SummaryError, ValueError):
    """Input does not satisfy the summary analyzer contract."""


class SummaryEvidenceError(SummaryError, ValueError):
    """Evidence/provenance is missing or structurally unusable."""


class SummaryTaskError(SummaryError, ValueError):
    """The supplied task/scope is incompatible with summary analysis."""


# ============================================================================
# Public result type
# ============================================================================


@dataclass(frozen=True)
class SummaryResult:
    """
    Machine-readable, immutable summary result.

    `structured_output` is the parser-owned summary payload and is deliberately
    retained rather than reimplementing the LLM schema in this class.

    Top-level convenience properties expose the fields most commonly consumed
    by APIs/frontends without changing the underlying parser contract.
    """

    status: str
    success: bool
    grounded: bool

    paper_id: Optional[str]
    document_id: Optional[str]
    title: Optional[str]

    structured_output: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    citations: tuple[Mapping[str, Any], ...]

    scope: str = "research"
    task_type: str = TASK_TYPE
    evidence_count: int = 0
    source_count: int = 0
    citation_count: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("summary"))

    @property
    def problem(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("problem"))

    @property
    def methodology(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("methodology"))

    @property
    def dataset(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("dataset"))

    @property
    def model(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("model"))

    @property
    def key_findings(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("key_findings"))

    @property
    def limitations(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("limitations"))

    @property
    def overall_takeaway(self) -> Optional[str]:
        """
        The current project schema does not define an `overall_takeaway`
        field. Do not synthesize one here; return only an explicitly supplied
        future-compatible field.
        """
        return _optional_string(
            self.structured_output.get("overall_takeaway")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe application representation."""
        return {
            "status": self.status,
            "success": self.success,
            "grounded": self.grounded,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "title": self.title,
            "scope": self.scope,
            "task_type": self.task_type,
            "summary": self.summary,
            "problem": self.problem,
            "methodology": self.methodology,
            "dataset": self.dataset,
            "model": self.model,
            "key_findings": list(self.key_findings),
            "limitations": list(self.limitations),
            "overall_takeaway": self.overall_takeaway,
            "structured_output": _json_copy(self.structured_output),
            "evidence": _json_copy(list(self.evidence)),
            "citations": _json_copy(list(self.citations)),
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "citation_count": self.citation_count,
            "diagnostics": _json_copy(self.diagnostics),
        }


# ============================================================================
# Internal normalization helpers
# ============================================================================


_MISSING = object()


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def _required_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SummaryInputError(
            f"{field_name} must be a non-empty string."
        )
    return value.strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return ()

    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
    return tuple(result)


def _json_copy(value: Any) -> Any:
    """
    Copy JSON-compatible values and reject arbitrary Python objects.

    This protects the result boundary from mutable caller-owned structures and
    avoids accidentally serializing implementation objects.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise SummaryInputError(
                "Summary data contains a non-finite numeric value."
            )
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _json_copy(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_copy(child) for child in value]

    raise SummaryInputError(
        "Summary data contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # A copied ordinary dict is intentionally used instead of a mapping proxy
    # so the object remains straightforward to JSON serialize downstream.
    copied = _json_copy(value)
    if not isinstance(copied, dict):
        raise SummaryInputError("Structured output must be an object.")
    return copied


# ============================================================================
# Provenance extraction
# ============================================================================


_PROVENANCE_FIELDS = (
    "paper_id",
    "document_id",
    "chunk_id",
    "page",
    "section",
    "citation",
)


def _extract_identity(
    *,
    paper: Any,
    structured_output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    scope: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve identity from explicit paper/document metadata or evidence.

    The analyzer never invents identifiers. If identity is absent, it remains
    None except where the caller's explicit scope makes an identity mandatory.
    """
    paper_id = _first_string(
        _read(paper, "paper_id"),
        _read(paper, "id"),
        _read(structured_output, "paper_id"),
    )
    document_id = _first_string(
        _read(paper, "document_id"),
        _read(paper, "id") if scope == "uploaded" else None,
        _read(structured_output, "document_id"),
    )
    title = _first_string(
        _read(paper, "title"),
        _read(structured_output, "title"),
    )

    evidence_paper_ids = {
        str(item["paper_id"])
        for item in evidence
        if isinstance(item.get("paper_id"), str)
        and item["paper_id"].strip()
    }
    evidence_document_ids = {
        str(item["document_id"])
        for item in evidence
        if isinstance(item.get("document_id"), str)
        and item["document_id"].strip()
    }

    if paper_id is None and len(evidence_paper_ids) == 1:
        paper_id = next(iter(evidence_paper_ids))

    if document_id is None and len(evidence_document_ids) == 1:
        document_id = next(iter(evidence_document_ids))

    return paper_id, document_id, title


def _first_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_evidence(
    evidence: Any,
) -> tuple[Mapping[str, Any], ...]:
    if evidence is None:
        return ()

    if not isinstance(evidence, Sequence) or isinstance(
        evidence,
        (str, bytes, bytearray),
    ):
        raise SummaryEvidenceError(
            "Evidence must be a sequence of evidence objects."
        )

    normalized: list[Mapping[str, Any]] = []

    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise SummaryEvidenceError(
                f"Evidence item {index} must be an object."
            )

        copied = _immutable_mapping(item)
        normalized.append(copied)

    return tuple(normalized)


def _extract_evidence(
    *,
    parsed_response: Any,
    evidence: Any,
    pipeline_response: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Prefer authoritative ContextResult/pipeline evidence over the LLM's
    evidence-reference list when available.

    The LLM evidence list is still preserved inside structured_output. This
    method does not verify references; validation/evidence.py owns that job.
    """
    if evidence is not None:
        return _normalize_evidence(evidence)

    pipeline_evidence = _read(pipeline_response, "evidence", _MISSING)
    if pipeline_evidence is not _MISSING and pipeline_evidence is not None:
        return _normalize_evidence(pipeline_evidence)

    parsed_evidence = _read(parsed_response, "evidence", _MISSING)
    if parsed_evidence is not _MISSING:
        return _normalize_evidence(parsed_evidence)

    if isinstance(parsed_response, Mapping):
        return _normalize_evidence(
            parsed_response.get("evidence", ())
        )

    return ()


# ============================================================================
# Evidence/source safety
# ============================================================================


def _source_ids(
    evidence: Sequence[Mapping[str, Any]],
) -> set[str]:
    ids: set[str] = set()

    for item in evidence:
        for key in ("paper_id", "document_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                ids.add(value.strip())

    return ids


def _validate_single_paper_scope(
    *,
    evidence: Sequence[Mapping[str, Any]],
    paper_id: Optional[str],
    document_id: Optional[str],
    scope: str,
) -> None:
    """
    Prevent normal single-paper summaries from silently mixing documents.

    Comparison is explicitly exempt because it has its own task contract.
    """
    if scope == "comparison":
        return

    if not evidence:
        return

    identity_values: set[str] = set()

    for item in evidence:
        paper_value = item.get("paper_id")
        document_value = item.get("document_id")

        if isinstance(paper_value, str) and paper_value.strip():
            identity_values.add(paper_value.strip())
        elif isinstance(document_value, str) and document_value.strip():
            identity_values.add(document_value.strip())

    if len(identity_values) > 1:
        raise SummaryEvidenceError(
            "Single-paper summary received evidence from multiple "
            f"documents/papers: {sorted(identity_values)!r}."
        )

    if paper_id and any(
        isinstance(item.get("paper_id"), str)
        and item["paper_id"].strip()
        and item["paper_id"].strip() != paper_id
        for item in evidence
    ):
        raise SummaryEvidenceError(
            "Evidence contains a paper_id different from the requested "
            f"paper_id={paper_id!r}."
        )

    if document_id and any(
        isinstance(item.get("document_id"), str)
        and item["document_id"].strip()
        and item["document_id"].strip() != document_id
        for item in evidence
    ):
        raise SummaryEvidenceError(
            "Evidence contains a document_id different from the requested "
            f"document_id={document_id!r}."
        )


def _count_citations(
    evidence: Sequence[Mapping[str, Any]],
    structured_output: Mapping[str, Any],
) -> int:
    """
    Count explicit citation/provenance references without inventing them.

    This is metadata only; it is not a citation-quality score.
    """
    explicit = structured_output.get("citations")
    if isinstance(explicit, Sequence) and not isinstance(
        explicit,
        (str, bytes, bytearray),
    ):
        return len(explicit)

    count = 0
    for item in evidence:
        if item.get("citation") is not None:
            count += 1

    return count


# ============================================================================
# Main analyzer
# ============================================================================


class SummaryAnalyzer:
    """
    Deterministic high-level summary normalizer.

    It consumes output already produced by the project's RAG/LLM/parser
    pipeline. It never calls an LLM or retrieval component.

    Primary API:
        analyze(paper, parsed_response=..., evidence=..., ...)

    The `pipeline_response` parameter is supported so the analyzer can also be
    used directly after RAGPipeline without coupling to its implementation.
    """

    def __init__(
        self,
        *,
        require_evidence_for_success: bool = True,
        allow_insufficient_evidence: bool = True,
    ) -> None:
        if not isinstance(require_evidence_for_success, bool):
            raise TypeError(
                "require_evidence_for_success must be a boolean."
            )
        if not isinstance(allow_insufficient_evidence, bool):
            raise TypeError(
                "allow_insufficient_evidence must be a boolean."
            )

        self.require_evidence_for_success = require_evidence_for_success
        self.allow_insufficient_evidence = allow_insufficient_evidence

    def analyze(
        self,
        paper: Any = None,
        *,
        evidence: Any = None,
        parsed_response: Any = None,
        pipeline_response: Any = None,
        scope: Optional[str] = None,
        task_type: str = TASK_TYPE,
    ) -> SummaryResult:
        """
        Normalize one already-generated summary.

        Parameters
        ----------
        paper:
            Paper/document metadata object or mapping. Optional when identity
            can be recovered from evidence/pipeline data.

        evidence:
            Authoritative ContextResult evidence when available.

        parsed_response:
            Parser-owned structured output. This is the preferred source of
            summary content.

        pipeline_response:
            Optional PipelineResponse wrapper. Its structured_output and
            evidence fields are consumed without duplicating pipeline logic.

        scope:
            research | uploaded | comparison. If omitted, it is recovered
            from pipeline_response where available.

        task_type:
            Must be "summary". Comparison is intentionally not implemented by
            this module.
        """
        started = time.perf_counter()

        normalized_task = self._validate_task_type(task_type)
        normalized_scope = self._resolve_scope(scope, pipeline_response)

        structured = self._extract_structured_output(
            parsed_response=parsed_response,
            pipeline_response=pipeline_response,
        )

        normalized_evidence = _extract_evidence(
            parsed_response=parsed_response,
            evidence=evidence,
            pipeline_response=pipeline_response,
        )

        paper_id, document_id, title = _extract_identity(
            paper=paper,
            structured_output=structured,
            evidence=normalized_evidence,
            scope=normalized_scope,
        )

        if normalized_scope == "uploaded" and document_id is None:
            raise SummaryInputError(
                "Uploaded-paper summary requires document identity."
            )

        _validate_single_paper_scope(
            evidence=normalized_evidence,
            paper_id=paper_id,
            document_id=document_id,
            scope=normalized_scope,
        )

        grounded = self._resolve_grounded(
            structured_output=structured,
            pipeline_response=pipeline_response,
        )

        status, success = self._resolve_status(
            structured_output=structured,
            evidence=normalized_evidence,
            pipeline_response=pipeline_response,
            grounded=grounded,
        )

        source_count = len(_source_ids(normalized_evidence))
        citation_count = _count_citations(
            normalized_evidence,
            structured,
        )

        diagnostics = {
            "evidence_count": len(normalized_evidence),
            "source_count": source_count,
            "citation_count": citation_count,
            "processing_seconds": time.perf_counter() - started,
        }

        logger.info(
            "Summary analysis completed: task=%s scope=%s "
            "paper_id=%s document_id=%s status=%s evidence_count=%d",
            normalized_task,
            normalized_scope,
            paper_id,
            document_id,
            status,
            len(normalized_evidence),
        )

        return SummaryResult(
            status=status,
            success=success,
            grounded=grounded,
            paper_id=paper_id,
            document_id=document_id,
            title=title,
            structured_output=structured,
            evidence=normalized_evidence,
            citations=tuple(
                item
                for item in normalized_evidence
                if item.get("citation") is not None
            ),
            scope=normalized_scope,
            task_type=normalized_task,
            evidence_count=len(normalized_evidence),
            source_count=source_count,
            citation_count=citation_count,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Validation/resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_task_type(task_type: Any) -> str:
        if not isinstance(task_type, str):
            raise SummaryTaskError("task_type must be a string.")

        normalized = task_type.strip().lower()
        if normalized != TASK_TYPE:
            raise SummaryTaskError(
                "SummaryAnalyzer only handles task_type='summary'; "
                f"received {task_type!r}."
            )

        return normalized

    @staticmethod
    def _resolve_scope(
        scope: Optional[str],
        pipeline_response: Any,
    ) -> str:
        candidate = scope

        if candidate is None:
            candidate = _read(pipeline_response, "scope", None)

        if candidate is None:
            candidate = "research"

        if not isinstance(candidate, str):
            raise SummaryInputError("scope must be a string.")

        normalized = candidate.strip().lower()
        if normalized not in SUPPORTED_SCOPES:
            raise SummaryInputError(
                f"Unsupported scope={candidate!r}. "
                f"Expected one of {sorted(SUPPORTED_SCOPES)!r}."
            )

        return normalized

    @staticmethod
    def _extract_structured_output(
        *,
        parsed_response: Any,
        pipeline_response: Any,
    ) -> Mapping[str, Any]:
        candidate = parsed_response

        if candidate is None:
            candidate = _read(
                pipeline_response,
                "structured_output",
                None,
            )

        if candidate is None:
            raise SummaryInputError(
                "A parsed structured summary is required. "
                "summary.py does not generate one."
            )

        # ParsedLLMOutput and similar wrappers implement Mapping.
        if isinstance(candidate, Mapping):
            data = candidate
        else:
            # Some parser implementations expose the actual payload as
            # `.data` or `.structured_output`.
            data = _read(candidate, "data", _MISSING)
            if data is _MISSING:
                data = _read(candidate, "structured_output", _MISSING)

            if not isinstance(data, Mapping):
                raise SummaryInputError(
                    "Parsed response must be a mapping or expose a "
                    "mapping via data/structured_output."
                )

        copied = _immutable_mapping(data)

        if not copied:
            raise SummaryInputError(
                "Parsed summary output cannot be empty."
            )

        return copied

    @staticmethod
    def _resolve_grounded(
        *,
        structured_output: Mapping[str, Any],
        pipeline_response: Any,
    ) -> bool:
        value = structured_output.get("grounded")

        if isinstance(value, bool):
            return value

        pipeline_grounded = _read(
            pipeline_response,
            "grounded",
            None,
        )
        if isinstance(pipeline_grounded, bool):
            return pipeline_grounded

        # Fail closed. Evidence presence alone is NOT proof of grounding.
        return False

    def _resolve_status(
        self,
        *,
        structured_output: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        pipeline_response: Any,
        grounded: bool,
    ) -> tuple[str, bool]:
        pipeline_status = _read(
            pipeline_response,
            "status",
            None,
        )

        if isinstance(pipeline_status, Enum):
            pipeline_status = pipeline_status.value

        if isinstance(pipeline_status, str):
            normalized_pipeline_status = pipeline_status.strip().lower()
            if normalized_pipeline_status == INSUFFICIENT_EVIDENCE:
                return INSUFFICIENT_EVIDENCE, False

        if not evidence and self.require_evidence_for_success:
            if self.allow_insufficient_evidence:
                return INSUFFICIENT_EVIDENCE, False
            raise SummaryEvidenceError(
                "Summary contains no source evidence."
            )

        # The parser schema requires grounded to be explicit. Do not turn
        # evidence existence into grounded=True.
        if self.require_evidence_for_success and not grounded:
            return VALIDATION_ERROR, False

        if not self._has_summary_content(structured_output):
            return INVALID_INPUT, False

        return SUCCESS, True

    @staticmethod
    def _has_summary_content(
        structured_output: Mapping[str, Any],
    ) -> bool:
        value = structured_output.get("summary")
        return isinstance(value, str) and bool(value.strip())


# ============================================================================
# Batch support for Mode 1
# ============================================================================


class ResearchSummaryAnalyzer:
    """
    Deterministic batch facade for independently summarizing selected papers.

    Each item is processed independently. Evidence is never pooled across
    papers. Comparison is not silently performed here.
    """

    def __init__(
        self,
        analyzer: Optional[SummaryAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or SummaryAnalyzer()

    def analyze_papers(
        self,
        papers: Sequence[Any],
        *,
        responses: Sequence[Any],
        evidences: Optional[Sequence[Any]] = None,
    ) -> tuple[SummaryResult, ...]:
        if not isinstance(papers, Sequence) or isinstance(
            papers,
            (str, bytes, bytearray),
        ):
            raise SummaryInputError("papers must be a sequence.")

        if not isinstance(responses, Sequence) or isinstance(
            responses,
            (str, bytes, bytearray),
        ):
            raise SummaryInputError("responses must be a sequence.")

        if len(papers) != len(responses):
            raise SummaryInputError(
                "papers and responses must contain the same number of items."
            )

        if evidences is None:
            evidence_items: Sequence[Any] = [None] * len(papers)
        else:
            if not isinstance(evidences, Sequence) or isinstance(
                evidences,
                (str, bytes, bytearray),
            ):
                raise SummaryInputError(
                    "evidences must be a sequence when supplied."
                )
            if len(evidences) != len(papers):
                raise SummaryInputError(
                    "papers, responses, and evidences must have matching "
                    "lengths."
                )
            evidence_items = evidences

        results: list[SummaryResult] = []

        for index, (paper, response, item_evidence) in enumerate(
            zip(papers, responses, evidence_items)
        ):
            try:
                result = self.analyzer.analyze(
                    paper,
                    parsed_response=response,
                    evidence=item_evidence,
                    scope="research",
                )
            except SummaryError as exc:
                raise SummaryError(
                    f"Failed to analyze paper at batch index {index}."
                ) from exc

            results.append(result)

        return tuple(results)


# ============================================================================
# Backward-compatible functional APIs
# ============================================================================


_DEFAULT_ANALYZER = SummaryAnalyzer()


def analyze_summary(
    paper: Any = None,
    *,
    evidence: Any = None,
    parsed_response: Any = None,
    pipeline_response: Any = None,
    scope: Optional[str] = None,
) -> SummaryResult:
    """Functional convenience wrapper around SummaryAnalyzer."""
    return _DEFAULT_ANALYZER.analyze(
        paper,
        evidence=evidence,
        parsed_response=parsed_response,
        pipeline_response=pipeline_response,
        scope=scope,
    )


# A concise alias for application/service imports.
analyze = analyze_summary


# ============================================================================
# Model-free self-test
# ============================================================================


def run_self_test() -> None:
    """
    Run deterministic summary tests without LLM, retrieval, FAISS, PDF, or
    network access.
    """

    evidence_a = (
        {
            "evidence_id": "paper_001::chunk_001",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "chunk_id": "chunk_001",
            "section": "Abstract",
            "page": 1,
            "text": (
                "The paper proposes a deep learning method for image "
                "classification."
            ),
        },
        {
            "evidence_id": "paper_001::chunk_002",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "chunk_id": "chunk_002",
            "section": "Results",
            "page": 5,
            "text": "The authors report 99.71% accuracy.",
        },
    )

    parsed_a = {
        "summary": (
            "The paper proposes a deep learning method for image "
            "classification and reports 99.71% accuracy."
        ),
        "problem": "Image classification.",
        "methodology": "Deep learning.",
        "dataset": "Not found in the provided evidence.",
        "model": "Not found in the provided evidence.",
        "key_findings": ["The authors report 99.71% accuracy."],
        "limitations": [],
        "grounded": True,
        "evidence": [
            {
                "evidence_number": 1,
                "chunk_id": "chunk_001",
                "document_id": "paper_001",
                "paper_id": "paper_001",
                "page": 1,
                "section": "Abstract",
            },
            {
                "evidence_number": 2,
                "chunk_id": "chunk_002",
                "document_id": "paper_001",
                "paper_id": "paper_001",
                "page": 5,
                "section": "Results",
            },
        ],
    }

    analyzer = SummaryAnalyzer()

    # 1. Valid research-discovery summary.
    result = analyzer.analyze(
        {
            "paper_id": "paper_001",
            "title": "Synthetic Paper",
        },
        parsed_response=parsed_a,
        evidence=evidence_a,
        scope="research",
    )

    assert result.status == SUCCESS
    assert result.success is True
    assert result.grounded is True
    assert result.paper_id == "paper_001"
    assert result.summary is not None
    assert "99.71%" in result.summary
    assert result.evidence_count == 2
    assert result.source_count == 1
    assert result.to_dict()["evidence"][1]["page"] == 5

    # 2. Uploaded-paper identity is mandatory.
    uploaded = analyzer.analyze(
        {
            "document_id": "paper_001",
            "title": "Uploaded Paper",
        },
        parsed_response=parsed_a,
        evidence=evidence_a,
        scope="uploaded",
    )
    assert uploaded.document_id == "paper_001"

    # 3. Missing evidence becomes explicit insufficient evidence.
    insufficient = analyzer.analyze(
        {"paper_id": "paper_001"},
        parsed_response=parsed_a,
        evidence=(),
        scope="research",
    )
    assert insufficient.status == INSUFFICIENT_EVIDENCE
    assert insufficient.success is False
    assert insufficient.grounded is True

    # 4. Evidence exists but grounded=false cannot be reported as success.
    ungrounded = dict(parsed_a)
    ungrounded["grounded"] = False
    ungrounded_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        parsed_response=ungrounded,
        evidence=evidence_a,
        scope="research",
    )
    assert ungrounded_result.status == VALIDATION_ERROR
    assert ungrounded_result.success is False
    assert ungrounded_result.grounded is False

    # 5. Missing summary content is rejected as invalid.
    missing_summary = dict(parsed_a)
    missing_summary["summary"] = ""
    missing_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        parsed_response=missing_summary,
        evidence=evidence_a,
        scope="research",
    )
    assert missing_result.status == INVALID_INPUT
    assert missing_result.success is False

    # 6. Paper A/B evidence cannot be silently mixed.
    evidence_mixed = (
        evidence_a[0],
        {
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "chunk_id": "chunk_x",
            "page": 2,
            "section": "Results",
        },
    )
    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            parsed_response=parsed_a,
            evidence=evidence_mixed,
            scope="research",
        )
    except SummaryEvidenceError:
        pass
    else:
        raise AssertionError("Cross-paper evidence was silently accepted.")

    # 7. Numerical value is preserved exactly; no rounding/conversion.
    assert "99.71%" in result.summary
    assert "99.71%" in result.key_findings[0]

    # 8. Provenance is preserved.
    assert result.evidence[0]["document_id"] == "paper_001"
    assert result.evidence[0]["chunk_id"] == "chunk_001"
    assert result.evidence[0]["page"] == 1
    assert result.evidence[0]["section"] == "Abstract"

    # 9. No duplicate LLM/retrieval calls: this module has no provider,
    # embedding, FAISS, or retrieval imports.
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_import_tokens = (
        "faiss",
        "sentence_transformers",
        "transformers",
        "torch",
        "openai",
        "anthropic",
        "google",
        "fitz",
        "pymupdf",
    )
    lowered = source.lower()
    assert not any(
        f"import {token}" in lowered
        or f"from {token}" in lowered
        for token in forbidden_import_tokens
    )

    # 10. Determinism.
    repeat = analyzer.analyze(
        {"paper_id": "paper_001", "title": "Synthetic Paper"},
        parsed_response=parsed_a,
        evidence=evidence_a,
        scope="research",
    )
    first = result.to_dict()
    second = repeat.to_dict()

    # Processing time is intentionally excluded from the deterministic
    # comparison.
    first["diagnostics"] = {}
    second["diagnostics"] = {}
    assert first == second

    # 11. Multi-paper batch keeps papers isolated.
    parsed_b = dict(parsed_a)
    parsed_b["summary"] = "Paper B reports a transformer-based classifier."
    parsed_b["problem"] = "Classification."
    evidence_b = (
        {
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "chunk_id": "chunk_b1",
            "page": 2,
            "section": "Methodology",
        },
    )

    batch = ResearchSummaryAnalyzer(analyzer).analyze_papers(
        [
            {"paper_id": "paper_001"},
            {"paper_id": "paper_002"},
        ],
        responses=[parsed_a, parsed_b],
        evidences=[evidence_a, evidence_b],
    )

    assert len(batch) == 2
    assert batch[0].paper_id == "paper_001"
    assert batch[1].paper_id == "paper_002"
    assert batch[0].evidence[0]["paper_id"] == "paper_001"
    assert batch[1].evidence[0]["paper_id"] == "paper_002"

    # 12. Comparison scope is not silently treated as single-paper analysis.
    comparison = analyzer.analyze(
        None,
        parsed_response=parsed_a,
        evidence=evidence_mixed,
        scope="comparison",
    )
    assert comparison.scope == "comparison"

    print("SummaryAnalyzer self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()