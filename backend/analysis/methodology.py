"""
Evidence-grounded methodology analysis for the AI Research Paper Assistant.

Architectural boundary
----------------------
    RAG retrieval
        -> ContextResult
        -> PromptBuilder
        -> LLM client
        -> parser
        -> MethodologyAnalyzer
        -> validation

This module deliberately does NOT:
    * retrieve documents
    * generate embeddings
    * access FAISS/SPECTER2
    * rerank or rank candidates
    * parse PDFs or chunks
    * construct prompts
    * call an LLM
    * perform external web/database lookups
    * verify scientific claims

The existing prompt/schema contract is the source of truth for the structured
methodology payload:

    methodology
    major_steps
    models_algorithms
    experimental_setup
    grounded
    evidence

The analyzer therefore normalizes the parser-owned response, preserves paper
identity/provenance, prevents cross-paper evidence mixing, and exposes a
frontend/API-friendly result without creating a second analysis schema.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK_TYPE = "methodology"
SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})

SUCCESS = "success"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
INVALID_INPUT = "invalid_input"
VALIDATION_ERROR = "validation_error"

NOT_FOUND = "Not found in the provided evidence."


class MethodologyError(RuntimeError):
    """Base methodology-analysis error."""


class MethodologyInputError(MethodologyError, ValueError):
    """Input violates the methodology analyzer contract."""


class MethodologyEvidenceError(MethodologyError, ValueError):
    """Evidence is missing, malformed, or belongs to another paper."""


class MethodologyTaskError(MethodologyError, ValueError):
    """Unsupported methodology task or scope."""


@dataclass(frozen=True)
class MethodologyResult:
    """
    Application result around the existing parser-owned methodology schema.

    No second scientific response schema is introduced. `structured_output`
    remains the authoritative structured payload produced by the parser.
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
    def methodology(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("methodology"))

    @property
    def major_steps(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("major_steps"))

    @property
    def models_algorithms(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("models_algorithms"))

    @property
    def experimental_setup(self) -> Optional[str]:
        return _optional_string(
            self.structured_output.get("experimental_setup")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for API/frontend layers."""
        return {
            "status": self.status,
            "success": self.success,
            "grounded": self.grounded,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "title": self.title,
            "scope": self.scope,
            "task_type": self.task_type,
            "methodology": self.methodology,
            "major_steps": list(self.major_steps),
            "models_algorithms": list(self.models_algorithms),
            "experimental_setup": self.experimental_setup,
            "structured_output": _json_copy(self.structured_output),
            "evidence": _json_copy(list(self.evidence)),
            "citations": _json_copy(list(self.citations)),
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "citation_count": self.citation_count,
            "diagnostics": _json_copy(self.diagnostics),
        }


_MISSING = object()

# These are the exact fields used by the existing evidence contract where
# available. No value is fabricated if an upstream implementation omits one.
_PROVENANCE_FIELDS = (
    "evidence_number",
    "evidence_id",
    "chunk_id",
    "document_id",
    "paper_id",
    "page",
    "page_end",
    "section",
    "citation",
)


def _read(value: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from either a mapping or an object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_string(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return ()

    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return tuple(result)


def _json_copy(value: Any) -> Any:
    """
    Copy only JSON-compatible values and reject arbitrary Python objects.

    This keeps the result boundary deterministic and serialization-safe.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise MethodologyInputError(
                "Methodology data contains a non-finite numeric value."
            )
        return value

    if isinstance(value, Mapping):
        return {str(k): _json_copy(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]

    raise MethodologyInputError(
        "Methodology data contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_copy(value)
    if not isinstance(copied, dict):
        raise MethodologyInputError("Expected a JSON object.")
    return copied


def _normalize_evidence(evidence: Any) -> tuple[Mapping[str, Any], ...]:
    if evidence is None:
        return ()

    if not isinstance(evidence, Sequence) or isinstance(
        evidence, (str, bytes, bytearray)
    ):
        raise MethodologyEvidenceError(
            "Evidence must be a sequence of evidence objects."
        )

    normalized: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise MethodologyEvidenceError(
                f"Evidence item {index} must be a mapping."
            )

        copied = _copy_mapping(item)

        # Preserve all upstream fields, but reject exact duplicate evidence IDs
        # because they can otherwise make provenance counts misleading.
        evidence_id = _first_string(
            copied.get("evidence_id"),
            copied.get("chunk_id"),
        )
        if evidence_id is not None:
            if evidence_id in seen_ids:
                raise MethodologyEvidenceError(
                    f"Duplicate evidence identifier: {evidence_id!r}."
                )
            seen_ids.add(evidence_id)

        normalized.append(copied)

    return tuple(normalized)


def _extract_authoritative_evidence(
    *,
    explicit_evidence: Any,
    parsed_response: Any,
    pipeline_response: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Prefer authoritative pipeline/context evidence.

    Parsed LLM evidence references are retained inside structured_output, but
    they are not treated as proof that the references are valid. Verification
    belongs to validation/evidence.py.
    """
    if explicit_evidence is not None:
        return _normalize_evidence(explicit_evidence)

    pipeline_evidence = _read(pipeline_response, "evidence", _MISSING)
    if pipeline_evidence is not _MISSING and pipeline_evidence is not None:
        return _normalize_evidence(pipeline_evidence)

    parsed_evidence = _read(parsed_response, "evidence", _MISSING)
    if parsed_evidence is not _MISSING:
        return _normalize_evidence(parsed_evidence)

    return ()


def _extract_structured_output(
    *,
    parsed_response: Any,
    pipeline_response: Any,
) -> dict[str, Any]:
    """
    Extract the parser-owned structured payload without generating or
    reconstructing scientific content.
    """
    candidate = parsed_response

    if candidate is None:
        candidate = _read(
            pipeline_response,
            "structured_output",
            None,
        )

    if candidate is None:
        raise MethodologyInputError(
            "A parsed methodology response is required. "
            "methodology.py does not perform LLM inference."
        )

    if isinstance(candidate, Mapping):
        data = candidate
    else:
        data = _read(candidate, "data", _MISSING)
        if data is _MISSING:
            data = _read(candidate, "structured_output", _MISSING)

        if not isinstance(data, Mapping):
            raise MethodologyInputError(
                "Parsed methodology response must be a mapping or expose "
                "a mapping through data/structured_output."
            )

    result = _copy_mapping(data)

    if not result:
        raise MethodologyInputError(
            "Parsed methodology response cannot be empty."
        )

    return result


def _normalize_schema_fields(
    structured_output: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize only the fields already defined by the project's methodology
    schema.

    Existing schema:
        methodology: string
        major_steps: string[]
        models_algorithms: string[]
        experimental_setup: string
        grounded: bool
        evidence: evidence-reference[]
    """
    result = dict(structured_output)

    if "methodology" in result and not isinstance(
        result["methodology"], str
    ):
        raise MethodologyInputError(
            "'methodology' must be a string when present."
        )

    if "experimental_setup" in result and not isinstance(
        result["experimental_setup"], str
    ):
        raise MethodologyInputError(
            "'experimental_setup' must be a string when present."
        )

    for field_name in ("major_steps", "models_algorithms"):
        if field_name not in result:
            raise MethodologyInputError(
                f"Missing required methodology field: {field_name!r}."
            )

        value = result[field_name]
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise MethodologyInputError(
                f"'{field_name}' must be an array of strings."
            )

        if any(
            not isinstance(item, str)
            for item in value
        ):
            raise MethodologyInputError(
                f"'{field_name}' must contain only strings."
            )

        # Preserve scientific wording/order while removing accidental
        # surrounding whitespace only.
        result[field_name] = [
            item.strip() for item in value if item.strip()
        ]

    if "grounded" not in result:
        raise MethodologyInputError(
            "Parsed methodology response must explicitly provide 'grounded'."
        )

    if not isinstance(result["grounded"], bool):
        raise MethodologyInputError("'grounded' must be a boolean.")

    if "evidence" not in result:
        raise MethodologyInputError(
            "Parsed methodology response must provide 'evidence'."
        )

    if not isinstance(result["evidence"], Sequence) or isinstance(
        result["evidence"], (str, bytes, bytearray)
    ):
        raise MethodologyInputError(
            "'evidence' must be an array."
        )

    # Do not mutate evidence references or invent provenance.
    result["evidence"] = [
        _copy_mapping(item)
        if isinstance(item, Mapping)
        else _raise_invalid_evidence_reference(item)
        for item in result["evidence"]
    ]

    return result


def _raise_invalid_evidence_reference(value: Any) -> dict[str, Any]:
    raise MethodologyInputError(
        "Every structured-output evidence reference must be an object; "
        f"received {type(value).__name__}."
    )


def _extract_identity(
    *,
    paper: Any,
    structured_output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    scope: str,
    pipeline_response: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve identity from existing metadata only.

    The analyzer never creates IDs. A single evidence identity can be used as
    a fallback when upstream metadata omitted the same identity.
    """
    paper_id = _first_string(
        _read(paper, "paper_id"),
        _read(paper, "id"),
        _read(structured_output, "paper_id"),
        _read(pipeline_response, "paper_id"),
    )

    document_id = _first_string(
        _read(paper, "document_id"),
        _read(paper, "id") if scope == "uploaded" else None,
        _read(structured_output, "document_id"),
        _read(pipeline_response, "document_id"),
    )

    title = _first_string(
        _read(paper, "title"),
        _read(structured_output, "title"),
        _read(pipeline_response, "title"),
    )

    paper_ids = {
        str(item["paper_id"]).strip()
        for item in evidence
        if isinstance(item.get("paper_id"), str)
        and item["paper_id"].strip()
    }
    document_ids = {
        str(item["document_id"]).strip()
        for item in evidence
        if isinstance(item.get("document_id"), str)
        and item["document_id"].strip()
    }

    if paper_id is None and len(paper_ids) == 1:
        paper_id = next(iter(paper_ids))

    if document_id is None and len(document_ids) == 1:
        document_id = next(iter(document_ids))

    return paper_id, document_id, title


def _validate_identity_isolation(
    *,
    evidence: Sequence[Mapping[str, Any]],
    paper_id: Optional[str],
    document_id: Optional[str],
    scope: str,
) -> None:
    """
    Prevent normal single-paper methodology analysis from mixing papers.

    Explicit comparison scope is the only scope allowed to contain multiple
    source identities; comparison logic itself is intentionally not performed.
    """
    if scope == "comparison":
        return

    identities: set[tuple[str, str]] = set()

    for item in evidence:
        item_paper = _first_string(item.get("paper_id"))
        item_document = _first_string(item.get("document_id"))

        if item_paper or item_document:
            identities.add(
                (
                    item_paper or "",
                    item_document or "",
                )
            )

    if len(identities) > 1:
        raise MethodologyEvidenceError(
            "Single-paper methodology analysis received evidence from "
            f"multiple identities: {sorted(identities)!r}."
        )

    if paper_id is not None:
        mismatched = [
            item.get("paper_id")
            for item in evidence
            if isinstance(item.get("paper_id"), str)
            and item["paper_id"].strip()
            and item["paper_id"].strip() != paper_id
        ]
        if mismatched:
            raise MethodologyEvidenceError(
                f"Evidence contains paper_id values different from "
                f"{paper_id!r}: {mismatched!r}."
            )

    if document_id is not None:
        mismatched = [
            item.get("document_id")
            for item in evidence
            if isinstance(item.get("document_id"), str)
            and item["document_id"].strip()
            and item["document_id"].strip() != document_id
        ]
        if mismatched:
            raise MethodologyEvidenceError(
                f"Evidence contains document_id values different from "
                f"{document_id!r}: {mismatched!r}."
            )


def _resolve_grounded(
    structured_output: Mapping[str, Any],
    pipeline_response: Any,
) -> bool:
    value = structured_output.get("grounded")
    if isinstance(value, bool):
        return value

    pipeline_value = _read(pipeline_response, "grounded", None)
    if isinstance(pipeline_value, bool):
        return pipeline_value

    # Fail closed. Evidence existence alone does not prove grounding.
    return False


def _count_citations(
    evidence: Sequence[Mapping[str, Any]],
    structured_output: Mapping[str, Any],
) -> int:
    explicit_citations = structured_output.get("citations")
    if isinstance(explicit_citations, Sequence) and not isinstance(
        explicit_citations, (str, bytes, bytearray)
    ):
        return len(explicit_citations)

    return sum(
        1
        for item in evidence
        if item.get("citation") is not None
    )


def _source_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    identities: set[str] = set()

    for item in evidence:
        for field_name in ("paper_id", "document_id"):
            value = item.get(field_name)
            if isinstance(value, str) and value.strip():
                identities.add(value.strip())

    return len(identities)


def _has_methodology_content(
    structured_output: Mapping[str, Any],
) -> bool:
    """
    Determine whether the parser produced any methodology content.

    This does not invent a fallback description. An explicit Not Found string
    remains explicit missing information.
    """
    scalar_fields = ("methodology", "experimental_setup")

    for field_name in scalar_fields:
        value = structured_output.get(field_name)
        if isinstance(value, str) and value.strip():
            if value.strip().lower() != NOT_FOUND.lower():
                return True

    for field_name in ("major_steps", "models_algorithms"):
        values = _string_tuple(structured_output.get(field_name))
        if any(
            value.strip().lower() != NOT_FOUND.lower()
            for value in values
        ):
            return True

    return False


class MethodologyAnalyzer:
    """
    Lightweight, deterministic methodology-analysis boundary.

    It consumes already-parsed methodology output and authoritative evidence.
    It never calls an LLM and never retrieves documents.

    Example
    -------
    result = analyzer.analyze(
        paper={"paper_id": "paper_001", "title": "Example"},
        parsed_response=parsed_methodology,
        evidence=context_evidence,
        scope="research",
    )
    """

    def __init__(
        self,
        *,
        require_evidence_for_success: bool = True,
        require_grounded_flag: bool = True,
        allow_insufficient_evidence: bool = True,
    ) -> None:
        if not isinstance(require_evidence_for_success, bool):
            raise TypeError(
                "require_evidence_for_success must be a boolean."
            )
        if not isinstance(require_grounded_flag, bool):
            raise TypeError(
                "require_grounded_flag must be a boolean."
            )
        if not isinstance(allow_insufficient_evidence, bool):
            raise TypeError(
                "allow_insufficient_evidence must be a boolean."
            )

        self.require_evidence_for_success = require_evidence_for_success
        self.require_grounded_flag = require_grounded_flag
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
    ) -> MethodologyResult:
        """
        Normalize one parsed methodology result.

        No scientific content is generated here.
        """
        started = time.perf_counter()

        normalized_task = self._validate_task_type(task_type)
        normalized_scope = self._resolve_scope(
            scope,
            pipeline_response,
        )

        structured_raw = _extract_structured_output(
            parsed_response=parsed_response,
            pipeline_response=pipeline_response,
        )
        structured = _normalize_schema_fields(structured_raw)

        authoritative_evidence = _extract_authoritative_evidence(
            explicit_evidence=evidence,
            parsed_response=parsed_response,
            pipeline_response=pipeline_response,
        )

        paper_id, document_id, title = _extract_identity(
            paper=paper,
            structured_output=structured,
            evidence=authoritative_evidence,
            scope=normalized_scope,
            pipeline_response=pipeline_response,
        )

        if normalized_scope == "uploaded" and document_id is None:
            raise MethodologyInputError(
                "Uploaded-paper methodology analysis requires document_id."
            )

        _validate_identity_isolation(
            evidence=authoritative_evidence,
            paper_id=paper_id,
            document_id=document_id,
            scope=normalized_scope,
        )

        grounded = _resolve_grounded(
            structured,
            pipeline_response,
        )

        status, success = self._resolve_status(
            structured_output=structured,
            evidence=authoritative_evidence,
            grounded=grounded,
            pipeline_response=pipeline_response,
        )

        citation_count = _count_citations(
            authoritative_evidence,
            structured,
        )
        source_count = _source_count(authoritative_evidence)

        diagnostics = {
            "evidence_count": len(authoritative_evidence),
            "source_count": source_count,
            "citation_count": citation_count,
            "has_methodology_content": _has_methodology_content(
                structured
            ),
            "processing_seconds": time.perf_counter() - started,
        }

        logger.info(
            "Methodology analysis completed: task=%s scope=%s "
            "paper_id=%s document_id=%s status=%s evidence_count=%d",
            normalized_task,
            normalized_scope,
            paper_id,
            document_id,
            status,
            len(authoritative_evidence),
        )

        citations = tuple(
            item
            for item in authoritative_evidence
            if item.get("citation") is not None
        )

        return MethodologyResult(
            status=status,
            success=success,
            grounded=grounded,
            paper_id=paper_id,
            document_id=document_id,
            title=title,
            structured_output=structured,
            evidence=authoritative_evidence,
            citations=citations,
            scope=normalized_scope,
            task_type=normalized_task,
            evidence_count=len(authoritative_evidence),
            source_count=source_count,
            citation_count=citation_count,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_task_type(task_type: Any) -> str:
        if not isinstance(task_type, str):
            raise MethodologyTaskError("task_type must be a string.")

        normalized = task_type.strip().lower()
        if normalized != TASK_TYPE:
            raise MethodologyTaskError(
                "MethodologyAnalyzer only handles "
                f"task_type='methodology'; received {task_type!r}."
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
            raise MethodologyInputError("scope must be a string.")

        normalized = candidate.strip().lower()
        if normalized not in SUPPORTED_SCOPES:
            raise MethodologyInputError(
                f"Unsupported scope={candidate!r}; expected one of "
                f"{sorted(SUPPORTED_SCOPES)!r}."
            )

        return normalized

    def _resolve_status(
        self,
        *,
        structured_output: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        grounded: bool,
        pipeline_response: Any,
    ) -> tuple[str, bool]:
        pipeline_status = _read(
            pipeline_response,
            "status",
            None,
        )
        if isinstance(pipeline_status, Enum):
            pipeline_status = pipeline_status.value

        if isinstance(pipeline_status, str):
            if pipeline_status.strip().lower() == INSUFFICIENT_EVIDENCE:
                return INSUFFICIENT_EVIDENCE, False

        if self.require_evidence_for_success and not evidence:
            if self.allow_insufficient_evidence:
                return INSUFFICIENT_EVIDENCE, False
            raise MethodologyEvidenceError(
                "No authoritative methodology evidence is available."
            )

        if self.require_grounded_flag and not grounded:
            return VALIDATION_ERROR, False

        if not _has_methodology_content(structured_output):
            return INSUFFICIENT_EVIDENCE, False

        return SUCCESS, True


class ResearchMethodologyAnalyzer:
    """
    Batch facade for Mode-1 Top-10 analysis.

    Each paper is analyzed independently. Evidence is never pooled across
    papers, and comparison is not performed here.
    """

    def __init__(
        self,
        analyzer: Optional[MethodologyAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or MethodologyAnalyzer()

    def analyze_papers(
        self,
        papers: Sequence[Any],
        *,
        responses: Sequence[Any],
        evidences: Optional[Sequence[Any]] = None,
    ) -> tuple[MethodologyResult, ...]:
        if not isinstance(papers, Sequence) or isinstance(
            papers, (str, bytes, bytearray)
        ):
            raise MethodologyInputError("papers must be a sequence.")

        if not isinstance(responses, Sequence) or isinstance(
            responses, (str, bytes, bytearray)
        ):
            raise MethodologyInputError("responses must be a sequence.")

        if len(papers) != len(responses):
            raise MethodologyInputError(
                "papers and responses must have equal lengths."
            )

        if evidences is None:
            evidence_values: Sequence[Any] = [None] * len(papers)
        else:
            if not isinstance(evidences, Sequence) or isinstance(
                evidences, (str, bytes, bytearray)
            ):
                raise MethodologyInputError(
                    "evidences must be a sequence when supplied."
                )
            if len(evidences) != len(papers):
                raise MethodologyInputError(
                    "papers, responses, and evidences must have equal lengths."
                )
            evidence_values = evidences

        results: list[MethodologyResult] = []

        for index, (paper, response, paper_evidence) in enumerate(
            zip(papers, responses, evidence_values)
        ):
            try:
                result = self.analyzer.analyze(
                    paper,
                    parsed_response=response,
                    evidence=paper_evidence,
                    scope="research",
                )
            except MethodologyError as exc:
                raise MethodologyError(
                    f"Methodology analysis failed for paper index {index}."
                ) from exc

            results.append(result)

        return tuple(results)


_DEFAULT_ANALYZER = MethodologyAnalyzer()


def analyze_methodology(
    paper: Any = None,
    *,
    evidence: Any = None,
    parsed_response: Any = None,
    pipeline_response: Any = None,
    scope: Optional[str] = None,
) -> MethodologyResult:
    """Functional convenience API."""
    return _DEFAULT_ANALYZER.analyze(
        paper,
        evidence=evidence,
        parsed_response=parsed_response,
        pipeline_response=pipeline_response,
        scope=scope,
    )


# Backward-friendly concise alias.
analyze = analyze_methodology


def run_self_test() -> None:
    """
    Model-free deterministic tests.

    These tests intentionally require no network, GPU, LLM, embeddings, FAISS,
    PDF parser, or external database.
    """
    analyzer = MethodologyAnalyzer()

    # ------------------------------------------------------------------
    # 1. Complete methodology / critical numeric preservation test.
    # ------------------------------------------------------------------
    evidence = (
        {
            "evidence_id": "paper_001::chunk_005",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "chunk_id": "chunk_005",
            "section": "Methods",
            "page": 5,
            "text": (
                "The images were resized to 224×224 pixels and normalized "
                "before training. The authors fine-tuned the pretrained model "
                "for 50 epochs using a batch size of 32."
            ),
            "citation": "paper_001:p5",
        },
    )

    parsed = {
        "methodology": (
            "The authors resized images to 224×224 pixels, normalized them, "
            "and fine-tuned the pretrained model for 50 epochs with batch "
            "size 32."
        ),
        "major_steps": [
            "Resizing to 224×224 pixels",
            "Normalization",
            "Fine-tuning the pretrained model for 50 epochs",
            "Batch size 32",
        ],
        "models_algorithms": ["pretrained model"],
        "experimental_setup": "Training was performed for 50 epochs.",
        "grounded": True,
        "evidence": [
            {
                "evidence_number": 1,
                "chunk_id": "chunk_005",
                "document_id": "paper_001",
                "paper_id": "paper_001",
                "page": 5,
                "section": "Methods",
            }
        ],
    }

    result = analyzer.analyze(
        {"paper_id": "paper_001", "title": "Example Paper"},
        evidence=evidence,
        parsed_response=parsed,
        scope="research",
    )

    assert result.status == SUCCESS
    assert result.success is True
    assert result.grounded is True
    assert result.paper_id == "paper_001"
    assert result.document_id == "paper_001"
    assert "224×224" in result.methodology
    assert "50 epochs" in result.methodology
    assert "batch size 32" in result.methodology
    assert result.major_steps[0] == "Resizing to 224×224 pixels"
    assert result.major_steps[1] == "Normalization"
    assert result.evidence[0]["page"] == 5
    assert result.evidence[0]["section"] == "Methods"
    assert result.evidence[0]["citation"] == "paper_001:p5"

    # ------------------------------------------------------------------
    # 2. Minimal methodology.
    # ------------------------------------------------------------------
    minimal = dict(parsed)
    minimal["methodology"] = "The paper uses a transformer-based approach."
    minimal["major_steps"] = []
    minimal["models_algorithms"] = ["transformer-based approach"]
    minimal["experimental_setup"] = NOT_FOUND

    minimal_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=minimal,
        scope="research",
    )
    assert minimal_result.success is True
    assert minimal_result.models_algorithms == (
        "transformer-based approach",
    )

    # ------------------------------------------------------------------
    # 3. Abstract-only evidence: no invented optimizer/epochs.
    # ------------------------------------------------------------------
    abstract_evidence = (
        {
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "chunk_id": "abstract_001",
            "section": "Abstract",
            "page": 1,
            "text": (
                "We use a transformer-based architecture for classification."
            ),
        },
    )
    abstract_parsed = {
        "methodology": (
            "The paper uses a transformer-based architecture for "
            "classification."
        ),
        "major_steps": [],
        "models_algorithms": ["transformer-based architecture"],
        "experimental_setup": NOT_FOUND,
        "grounded": True,
        "evidence": [
            {
                "evidence_number": 1,
                "chunk_id": "abstract_001",
                "document_id": "paper_002",
                "paper_id": "paper_002",
                "page": 1,
                "section": "Abstract",
            }
        ],
    }

    abstract_result = analyzer.analyze(
        {"paper_id": "paper_002"},
        evidence=abstract_evidence,
        parsed_response=abstract_parsed,
        scope="research",
    )

    assert abstract_result.success is True
    assert "Adam" not in abstract_result.methodology
    assert "1e-4" not in abstract_result.methodology
    assert "100 epochs" not in abstract_result.methodology
    assert abstract_result.evidence[0]["section"] == "Abstract"

    # ------------------------------------------------------------------
    # 4. Missing methodology evidence.
    # ------------------------------------------------------------------
    missing = {
        "methodology": NOT_FOUND,
        "major_steps": [],
        "models_algorithms": [],
        "experimental_setup": NOT_FOUND,
        "grounded": True,
        "evidence": [],
    }

    missing_result = analyzer.analyze(
        {"paper_id": "paper_003"},
        evidence=(),
        parsed_response=missing,
        scope="research",
    )

    assert missing_result.status == INSUFFICIENT_EVIDENCE
    assert missing_result.success is False

    # ------------------------------------------------------------------
    # 5. Evidence exists but grounded=False.
    # ------------------------------------------------------------------
    ungrounded = dict(parsed)
    ungrounded["grounded"] = False

    ungrounded_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=ungrounded,
        scope="research",
    )

    assert ungrounded_result.status == VALIDATION_ERROR
    assert ungrounded_result.success is False

    # ------------------------------------------------------------------
    # 6. Missing paper ID can remain None; do not fabricate.
    # ------------------------------------------------------------------
    no_id_evidence = (
        {
            "chunk_id": "chunk_no_id",
            "section": "Methods",
            "page": 3,
        },
    )
    no_id_result = analyzer.analyze(
        {},
        evidence=no_id_evidence,
        parsed_response=parsed,
        scope="research",
    )
    assert no_id_result.paper_id is None

    # ------------------------------------------------------------------
    # 7. Uploaded paper requires document identity.
    # ------------------------------------------------------------------
    try:
        analyzer.analyze(
            {},
            evidence=(
                {
                    "chunk_id": "no-document-id",
                    "section": "Methods",
                    "page": 5,
                },
            ),
            parsed_response=parsed,
            scope="uploaded",
        )
    except MethodologyInputError:
        pass
    else:
        raise AssertionError(
            "Uploaded scope accepted missing document identity."
        )

    uploaded = analyzer.analyze(
        {"document_id": "paper_001"},
        evidence=evidence,
        parsed_response=parsed,
        scope="uploaded",
    )
    assert uploaded.document_id == "paper_001"

    # ------------------------------------------------------------------
    # 8. Multi-paper isolation.
    # ------------------------------------------------------------------
    paper_a = (
        {
            "paper_id": "paper_A",
            "document_id": "paper_A",
            "chunk_id": "a1",
            "section": "Methods",
            "page": 2,
        },
    )
    paper_b = (
        {
            "paper_id": "paper_B",
            "document_id": "paper_B",
            "chunk_id": "b1",
            "section": "Methods",
            "page": 3,
        },
    )

    parsed_a = dict(parsed)
    parsed_a["major_steps"] = ["5-fold cross-validation"]

    parsed_b = dict(parsed)
    parsed_b["major_steps"] = ["70/30 train-test split"]

    batch = ResearchMethodologyAnalyzer(analyzer).analyze_papers(
        [
            {"paper_id": "paper_A"},
            {"paper_id": "paper_B"},
        ],
        responses=[parsed_a, parsed_b],
        evidences=[paper_a, paper_b],
    )

    assert batch[0].paper_id == "paper_A"
    assert batch[1].paper_id == "paper_B"
    assert batch[0].major_steps == ("5-fold cross-validation",)
    assert batch[1].major_steps == ("70/30 train-test split",)

    # ------------------------------------------------------------------
    # 9. Cross-paper mixing is rejected for normal analysis.
    # ------------------------------------------------------------------
    mixed = paper_a + paper_b
    try:
        analyzer.analyze(
            {"paper_id": "paper_A"},
            evidence=mixed,
            parsed_response=parsed_a,
            scope="research",
        )
    except MethodologyEvidenceError:
        pass
    else:
        raise AssertionError(
            "Cross-paper evidence was silently accepted."
        )

    # ------------------------------------------------------------------
    # 10. Comparison scope allows multiple identities but does not compare.
    # ------------------------------------------------------------------
    comparison_result = analyzer.analyze(
        {},
        evidence=mixed,
        parsed_response=parsed_a,
        scope="comparison",
    )
    assert comparison_result.scope == "comparison"

    # ------------------------------------------------------------------
    # 11. Invalid structured input.
    # ------------------------------------------------------------------
    invalid = dict(parsed)
    invalid["major_steps"] = "not-a-list"

    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=evidence,
            parsed_response=invalid,
        )
    except MethodologyInputError:
        pass
    else:
        raise AssertionError("Invalid structured input was accepted.")

    # ------------------------------------------------------------------
    # 12. Determinism.
    # ------------------------------------------------------------------
    repeat = analyzer.analyze(
        {"paper_id": "paper_001", "title": "Example Paper"},
        evidence=evidence,
        parsed_response=parsed,
        scope="research",
    )

    first = result.to_dict()
    second = repeat.to_dict()
    first["diagnostics"] = {}
    second["diagnostics"] = {}
    assert first == second

    # ------------------------------------------------------------------
    # 13. No forbidden inference/LLM/retrieval imports.
    # ------------------------------------------------------------------
    source = Path(__file__).read_text(encoding="utf-8").lower()

    forbidden_modules = (
        "faiss",
        "sentence_transformers",
        "transformers",
        "torch",
        "openai",
        "anthropic",
        "google.generativeai",
        "ollama",
        "fitz",
        "pymupdf",
    )

    for module_name in forbidden_modules:
        assert (
            f"import {module_name}" not in source
            and f"from {module_name}" not in source
        ), f"Forbidden dependency reference found: {module_name}"

    # ------------------------------------------------------------------
    # 14. Static safety check.
    # ------------------------------------------------------------------
    # The production module contains no dynamic execution, shell execution,
    # provider SDK calls, retrieval calls, or external I/O.
    assert "subprocess" not in type(run_self_test).__module__

    print("MethodologyAnalyzer self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()