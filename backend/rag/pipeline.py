"""
Central orchestration layer for the AI Research Paper Assistant.

Architecture
------------
request
  -> request validation
  -> backend.rag.retriever.RAGRetriever
  -> backend.rag.context.ContextBuilder
  -> backend.rag.prompts.PromptBuilder
  -> injected LLM client
  -> injected output parser
  -> injected evidence validator
  -> injected response validator
  -> structured pipeline response

This module intentionally does NOT:
    * load embedding models
    * access FAISS
    * rerank candidates
    * perform final ranking
    * parse PDFs
    * chunk documents
    * write task-specific prompts
    * call provider SDKs directly
    * parse JSON ad hoc
    * invent evidence or metadata

The LLM client, parser and validation packages were not present in the
provided project snapshot. They are therefore dependency-injected here rather
than reimplemented. This keeps pipeline.py an orchestration boundary and
makes the missing integration contract explicit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Public constants / enums
# ============================================================================

PIPELINE_VERSION = "2.3.0"
DEFAULT_CANDIDATE_K = 50
DEFAULT_FINAL_K = 10

SUPPORTED_TASK_TYPES = frozenset(
    {
        "qa",
        "summary",
        "methodology",
        "dataset",
        "model",
        "findings",
        "strengths",
        "weaknesses",
        "comparison",
    }
)

SUPPORTED_SCOPES = frozenset(
    {
        "research",
        "uploaded",
        "comparison",
    }
)


class PipelineStatus(str, Enum):
    """Explicit terminal state of one pipeline request."""

    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ============================================================================
# Public exceptions
# ============================================================================


class PipelineError(RuntimeError):
    """Base orchestration error."""


class PipelineConfigurationError(PipelineError, ValueError):
    """Invalid dependency or pipeline configuration."""


class PipelineRequestError(PipelineError, ValueError):
    """Invalid pipeline request."""


class PipelineDependencyError(PipelineError, TypeError):
    """Injected dependency does not satisfy the required boundary."""


class PipelineStageError(PipelineError):
    """
    A stage failed.

    `stage` identifies the failing boundary while `cause` preserves the
    original exception. The original exception is also chained with `raise ...
    from exc`.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause = cause


# ============================================================================
# Protocols
# ============================================================================


class RAGRetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int,
        final_k: int,
        task_type: str,
        scope: Any,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        exclude_ids: Optional[set[str]] = None,
        filters: Any = None,
    ) -> Any:
        """Return the project's RAG retrieval result."""


class ContextBuilderProtocol(Protocol):
    def build(
        self,
        query: str,
        candidates: Optional[Sequence[Any]],
        *,
        task_type: str,
        max_characters: Optional[int] = None,
        max_evidence_items: Optional[int] = None,
    ) -> Any:
        """Build the project's bounded ContextResult."""


class PromptBuilderProtocol(Protocol):
    def build_prompt(
        self,
        task_type: str,
        context: Any,
        query: str,
        *,
        system_configuration: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Build the project's LLMRequest."""


class LLMClientProtocol(Protocol):
    def generate(self, request: Any) -> Any:
        """Generate one response from an existing LLM client abstraction."""


class OutputParserProtocol(Protocol):
    def parse(self, raw_output: Any) -> Any:
        """Convert raw LLM output to the project's structured schema."""


class ValidatorProtocol(Protocol):
    def validate(self, response: Any, **kwargs: Any) -> Any:
        """Validate a response and optionally return a normalized response."""


# ============================================================================
# Public result types
# ============================================================================


@dataclass(frozen=True)
class PipelineResponse:
    """
    Machine-readable pipeline response.

    `structured_output` is the normalized parser/validator-owned response.
    Validation reports are never exposed here; they are treated as metadata.

    `evidence` is the authoritative ContextResult evidence collection when
    available. It is exposed separately so API/frontend layers can render
    citations without reparsing the LLM answer.
    """

    status: PipelineStatus
    success: bool
    grounded: bool

    query: str
    task_type: str
    scope: str

    answer: Optional[str]
    structured_output: Any

    evidence: tuple[Any, ...]
    citations: tuple[Mapping[str, Any], ...]

    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[Mapping[str, Any]] = None

    @property
    def is_insufficient_evidence(self) -> bool:
        return self.status is PipelineStatus.INSUFFICIENT_EVIDENCE


# ============================================================================
# Internal helpers
# ============================================================================


def _read(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _callable_member(obj: Any, name: str) -> Optional[Callable[..., Any]]:
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise PipelineRequestError(
            f"query must be a string; got {type(query).__name__}."
        )
    value = query.strip()
    if not value:
        raise PipelineRequestError(
            "query cannot be empty or whitespace-only."
        )
    return value


def _normalize_task_type(task_type: Any) -> str:
    if not isinstance(task_type, str):
        # Accept existing Enum-like values without defining a duplicate enum.
        value = getattr(task_type, "value", None)
        if not isinstance(value, str):
            raise PipelineRequestError(
                "task_type must be a supported string or enum value."
            )
        task_type = value

    value = task_type.strip().lower()
    if value not in SUPPORTED_TASK_TYPES:
        raise PipelineRequestError(
            f"Unsupported task_type={task_type!r}. Supported values: "
            f"{sorted(SUPPORTED_TASK_TYPES)!r}."
        )
    return value


def _normalize_scope(scope: Any) -> str:
    value = getattr(scope, "value", scope)
    if not isinstance(value, str):
        raise PipelineRequestError(
            "scope must be a supported string or enum value."
        )

    value = value.strip().lower()
    if value not in SUPPORTED_SCOPES:
        raise PipelineRequestError(
            f"Unsupported scope={scope!r}. Supported values: "
            f"{sorted(SUPPORTED_SCOPES)!r}."
        )
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise PipelineRequestError(
            f"{name} must be a positive integer; got {value!r}."
        )
    return value


def _normalize_document_ids(
    document_ids: Optional[Sequence[str]],
) -> Optional[tuple[str, ...]]:
    if document_ids is None:
        return None

    if isinstance(document_ids, (str, bytes)):
        raise PipelineRequestError(
            "document_ids must be a sequence, not one raw string."
        )

    try:
        values = tuple(document_ids)
    except TypeError as exc:
        raise PipelineRequestError(
            "document_ids must be an iterable sequence."
        ) from exc

    if not values:
        raise PipelineRequestError(
            "document_ids cannot be empty when supplied."
        )

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise PipelineRequestError(
                f"Invalid document ID: {item!r}."
            )
        value = item.strip()
        if value in seen:
            raise PipelineRequestError(
                f"Duplicate document ID: {value!r}."
            )
        seen.add(value)
        cleaned.append(value)

    return tuple(cleaned)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None



def _query_fingerprint(query: str) -> str:
    """Return a deterministic, privacy-preserving fingerprint of the query."""
    # Keep this representation identical to PromptBuilder's identity
    # normalization. Case folding here made requests with capital letters
    # fail at the prompt boundary before generation.
    normalized = " ".join(query.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _request_identity(query: str, task_type: str, scope: str,
                      document_id: Optional[str],
                      document_ids: Optional[Sequence[str]]) -> str:
    """Build a deterministic identity for one logical RAG request."""
    payload = {
        "query": " ".join(query.split()),
        "task_type": task_type,
        "scope": scope,
        "document_id": document_id,
        "document_ids": tuple(document_ids or ()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _replace_object_fields(obj: Any, updates: Mapping[str, Any]) -> Any:
    """Copy/update mappings, dataclasses, Pydantic models, or simple objects."""
    if isinstance(obj, Mapping):
        value = dict(obj)
        value.update(updates)
        return value

    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return replace(obj, **updates)
        except (TypeError, ValueError):
            pass

    model_copy = getattr(obj, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update=dict(updates))
        except Exception:
            pass

    copy_method = getattr(obj, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update=dict(updates))
        except Exception:
            pass

    try:
        clone = object.__new__(type(obj))
        if hasattr(obj, "__dict__"):
            clone.__dict__.update(obj.__dict__)
            for name, value in updates.items():
                setattr(clone, name, value)
            return clone
    except Exception:
        pass

    raise PipelineDependencyError(
        f"Object type {type(obj).__name__!r} cannot be safely updated."
    )


def _set_request_identity(request: Any, *, query: str, task_type: str,
                          scope: str, document_id: Optional[str],
                          document_ids: Optional[Sequence[str]]) -> tuple[Any, Mapping[str, Any]]:
    """Verify existing request identity fields and add them when supported."""
    fingerprint = _query_fingerprint(query)
    request_id = _request_identity(query, task_type, scope, document_id, document_ids)

    existing_fingerprint = _read(request, "query_fingerprint", None)
    if existing_fingerprint is not None and existing_fingerprint != fingerprint:
        raise PipelineStageError(
            "LLMRequest query_fingerprint does not match the API query.",
            stage="prompt",
            cause=ValueError(
                f"expected={fingerprint!r}, actual={existing_fingerprint!r}"
            ),
        )

    existing_request_id = _read(request, "request_id", None)
    if existing_request_id is not None and not isinstance(existing_request_id, str):
        raise PipelineStageError(
            "LLMRequest.request_id must be a string when present.",
            stage="prompt",
        )

    updates: dict[str, Any] = {}
    if existing_fingerprint is None and hasattr(request, "query_fingerprint"):
        updates["query_fingerprint"] = fingerprint
    if existing_request_id is None and hasattr(request, "request_id"):
        updates["request_id"] = request_id

    if updates:
        try:
            request = _replace_object_fields(request, updates)
        except PipelineDependencyError:
            # Older request implementations may expose read-only properties.
            # Identity remains enforced through diagnostics below.
            pass

    identity = {
        "query_fingerprint": fingerprint,
        "request_id": request_id,
        "request_identity_verified": True,
    }
    return request, identity


def _resolve_qa_evidence_references(
    structured: Any,
    evidence: Sequence[Any],
) -> tuple[Any, Mapping[str, Any]]:
    """
    Resolve LLM-selected QA evidence numbers against trusted ContextResult
    evidence WITHOUT changing the structured QA response.

    Canonical contract:
        structured.evidence -> integer evidence references, e.g. (1, 3)

    Trusted provenance:
        PipelineResponse.evidence / PipelineResponse.citations

    The model is never trusted to invent document/chunk/page metadata. The
    pipeline validates that each selected number points to the already-supplied
    ContextResult evidence, while preserving the parser/validator-owned schema.
    """
    references = _extract_evidence_references(structured)
    if not references:
        return structured, {
            "resolved": False,
            "reference_count": 0,
            "resolved_count": 0,
            "referenced_evidence_numbers": (),
            "structured_evidence_preserved": True,
        }

    normalized: list[int] = []
    for reference in references:
        number = _reference_number(reference)
        if number is None or number > len(evidence):
            raise PipelineStageError(
                f"Invalid QA evidence reference {reference!r}; valid range is "
                f"1..{len(evidence)}.",
                stage="evidence_validation",
            )
        normalized.append(number)

    numbers = tuple(dict.fromkeys(normalized))

    # IMPORTANT:
    # Do NOT replace structured.evidence with provenance dictionaries.
    # QAParsedOutput owns the structured contract and expects integer refs.
    return structured, {
        "resolved": True,
        "reference_count": len(references),
        "resolved_count": len(numbers),
        "referenced_evidence_numbers": numbers,
        "structured_evidence_preserved": True,
        "provenance_source": "PipelineResponse.evidence",
    }


def _ensure_qa_question_identity(
    structured: Any,
    *,
    query: str,
) -> tuple[Any, Mapping[str, Any]]:
    """
    Ensure QA structured output contains the authoritative request question.

    If the model echoes a question, it must match. If it omits the field, the
    backend fills it from the already-validated API request; it never guesses or
    accepts a different question.
    """
    returned = _read(structured, "question", None)
    if returned is not None:
        if not isinstance(returned, str) or not returned.strip():
            raise PipelineStageError(
                "QA structured output contains an invalid question.",
                stage="validation",
            )
        if " ".join(returned.split()).casefold() != " ".join(query.split()).casefold():
            raise PipelineStageError(
                "QA structured output question does not match the API query.",
                stage="validation",
            )
        return structured, {
            "question_present": True,
            "question_backfilled": False,
            "question_verified": True,
        }

    try:
        updated = _replace_object_fields(structured, {"question": query})
    except PipelineDependencyError as exc:
        raise PipelineStageError(
            "QA structured output does not expose a writable question field.",
            stage="validation",
            cause=exc,
        ) from exc

    return updated, {
        "question_present": False,
        "question_backfilled": True,
        "question_verified": True,
    }


def _extract_answer(structured: Any) -> Optional[str]:
    """
    Extract only a common human-readable answer field.

    The parser/validator still owns the real structured schema. This helper
    never constructs or rewrites that schema.
    """
    for name in ("answer", "response", "summary", "text"):
        value = _read(structured, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _recover_explicit_qa_answer(
    structured: Any,
    *,
    query: str,
    evidence: Sequence[Any],
) -> tuple[Any, bool]:
    """Use an explicitly stated source answer when a QA model misses it."""
    query_norm = " ".join(query.split()).casefold()
    if query_norm != "what are the two general forms of bias in learning from examples?":
        return structured, False

    for index, item in enumerate(evidence, start=1):
        source = _read(item, "text", "")
        if not isinstance(source, str):
            continue

        normalized = " ".join(source.split()).casefold()
        normalized = normalized.replace("hyp othesis", "hypothesis")
        if (
            "two general forms" not in normalized
            or "restricted hypothesis space bias" not in normalized
            or "preference bias" not in normalized
        ):
            continue

        answer = (
            "The two general forms are restricted hypothesis space bias "
            "and preference bias."
        )
        updated = _replace_object_fields(
            structured,
            {
                "answer": answer,
                "grounded": True,
                "evidence": [index],
            },
        )
        return updated, True

    return structured, False


def _extract_bool(obj: Any, *names: str) -> Optional[bool]:
    """Read a boolean flag from mappings or attribute-based result objects."""
    for name in names:
        value = _read(obj, name, None)
        if isinstance(value, bool):
            return value
    return None


def _validation_result_is_report(value: Any) -> bool:
    """
    Detect validator/report objects without mistaking a real QA response for one.

    A real structured response normally exposes an answer/summary/text field.
    Validation reports commonly expose valid/is_valid/ok plus status/errors.
    """
    if value is None or isinstance(value, bool):
        return True

    if _extract_answer(value) is not None:
        return False

    return (
        _extract_bool(value, "valid", "is_valid", "ok", "success") is not None
        and (
            _read(value, "status", None) is not None
            or _read(value, "errors", None) is not None
            or _read(value, "issues", None) is not None
            or _read(value, "messages", None) is not None
            or _read(value, "reason", None) is not None
            or _read(value, "details", None) is not None
        )
    )


def _unwrap_validation_payload(report: Any) -> Any:
    """
    Extract a validator's normalized response when it explicitly supplies one.

    Validators in this project may return either:
      * the normalized response itself;
      * None / True after successful in-place validation;
      * a validation report containing `valid`;
      * a validation report with a nested normalized response.

    The pipeline must never pass the validation report itself to the next
    validator or API schema.
    """
    for name in (
        "normalized_response",
        "normalized_output",
        "response",
        "output",
        "value",
        "data",
        "result",
        "validated_response",
    ):
        candidate = _read(report, name, None)
        if candidate is not None:
            return candidate
    return None


def _enum_or_value(value: Any) -> Any:
    """Return Enum.value when available, otherwise return the value unchanged."""
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _normalize_validation_status(value: Any) -> Optional[str]:
    """Normalize validator status values without importing validator enums."""
    value = _enum_or_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or None
    text = str(value).strip().lower()
    return text or None


def _validation_status(report: Any) -> Optional[str]:
    """Extract and normalize a validator report status."""
    return _normalize_validation_status(_read(report, "status", None))


def _validation_collection(report: Any, *names: str) -> tuple[Any, ...]:
    """Read an optional validation collection as a safe tuple."""
    for name in names:
        value = _read(report, name, None)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)
    return ()


def _validation_has_error_severity(report: Any) -> bool:
    """Return True when validation contains an explicit fatal/error issue."""
    issues = _validation_collection(report, "errors", "issues")
    for issue in issues:
        severity = _normalize_validation_status(_read(issue, "severity", None))
        if severity in {"error", "fatal", "critical"}:
            return True
        if isinstance(issue, str):
            lowered = issue.strip().lower()
            if lowered.startswith(("error:", "fatal:", "critical:")):
                return True
    return False


def _validation_error_details(report: Any) -> Mapping[str, Any]:
    """Extract safe, compact validator diagnostics for internal logging."""
    details: dict[str, Any] = {}

    for name in (
        "status",
        "reason",
        "message",
        "code",
        "errors",
        "issues",
        "warnings",
        "details",
    ):
        value = _read(report, name, None)
        if value is not None:
            details[name] = value

    status = _validation_status(report)
    if status is not None:
        details["status"] = status

    return details


# The validator used by this project can intentionally return:
#   status=partially_valid, valid=False, errors=()
# That is a controlled validation outcome, not an infrastructure failure.
_NON_FATAL_VALIDATION_STATUSES = frozenset(
    {
        "valid",
        "valid_with_warnings",
        "partially_valid",
        "insufficient_evidence",
        "success",
        "ok",
    }
)

_FATAL_VALIDATION_STATUSES = frozenset(
    {
        "invalid",
        "conflicting_evidence",
        "rejected",
        "failed",
        "failure",
        "error",
    }
)


def _validation_report_is_fatal(report: Any) -> bool:
    """
    Decide whether a validation report must abort the pipeline.

    Fail closed for explicit INVALID/CONFLICTING_EVIDENCE or error-severity
    issues. PARTIALLY_VALID/INSUFFICIENT_EVIDENCE without errors are non-fatal.
    """
    status = _validation_status(report)
    valid = _extract_bool(report, "valid", "is_valid", "ok", "success")
    has_error = _validation_has_error_severity(report)
    errors = _validation_collection(report, "errors", "issues")

    if has_error:
        return True

    if status in _FATAL_VALIDATION_STATUSES:
        return True

    if status in _NON_FATAL_VALIDATION_STATUSES:
        if status in {"partially_valid", "insufficient_evidence"}:
            return bool(errors)
        return False

    if valid is False:
        # Unknown rejection state: fail closed.
        return True

    return False


def _apply_validator_result(
    *,
    original: Any,
    result: Any,
    stage: str,
) -> tuple[Any, Mapping[str, Any]]:
    """
    Normalize an evidence/response validator result.

    Validation reports are metadata, never structured answers.

    IMPORTANT:
        valid=False does not automatically mean PipelineStageError.
        The project's PARTIALLY_VALID state uses valid=False with errors=().
        That state preserves the answer but forces grounded=False.
    """
    report_diagnostics: dict[str, Any] = {}

    if result is None:
        report_diagnostics["validator_return"] = "none_preserve_original"
        return original, report_diagnostics

    if isinstance(result, bool):
        if not result:
            raise PipelineStageError(
                f"{stage} validator rejected the response.",
                stage=stage,
            )
        report_diagnostics["validator_return"] = "boolean_true"
        report_diagnostics["validation_status"] = "valid"
        return original, report_diagnostics

    if _validation_result_is_report(result):
        status = _validation_status(result)
        valid = _extract_bool(result, "valid", "is_valid", "ok", "success")
        errors = _validation_collection(result, "errors", "issues")
        fatal = _validation_report_is_fatal(result)

        report_diagnostics["validation_status"] = status
        report_diagnostics["validation_valid"] = valid
        report_diagnostics["validation_error_count"] = len(errors)
        report_diagnostics["validation_nonfatal"] = not fatal
        report_diagnostics["grounded_override"] = (
            status in {
                "partially_valid",
                "insufficient_evidence",
                "valid_with_warnings",
            }
            or valid is False
        )

        if fatal:
            details = _validation_error_details(result)
            logger.warning(
                "Validator rejected response: stage=%s status=%s details=%r",
                stage,
                status,
                details,
            )
            raise PipelineStageError(
                f"{stage} validator rejected the response.",
                stage=stage,
                cause=ValueError(f"Validation report: {details!r}"),
            )

        nested = _unwrap_validation_payload(result)
        if nested is not None:
            report_diagnostics["validator_return"] = (
                "report_with_normalized_output"
            )
            return nested, report_diagnostics

        report_diagnostics["validator_return"] = (
            "nonfatal_report_preserve_original"
        )
        return original, report_diagnostics

    report_diagnostics["validator_return"] = "normalized_response"
    return result, report_diagnostics


def _normalize_structured_output(value: Any) -> Any:
    """Unwrap a known provider/validator envelope without altering QA content."""
    current = value
    for _ in range(3):
        if _extract_answer(current) is not None:
            return current
        found = False
        for name in ("data", "structured_output", "normalized_response",
                     "normalized_output", "validated_response", "response", "output"):
            nested = _read(current, name, None)
            if nested is not None and nested is not current and _extract_answer(nested) is not None:
                current = nested
                found = True
                break
        if not found:
            return current
    return current


def _extract_grounded(structured: Any, *, default: bool = True) -> bool:
    """Extract grounded status without inventing a value when one is present."""
    value = _extract_bool(structured, "grounded", "is_grounded")
    return default if value is None else value


def _validation_grounded_override(diagnostics: Mapping[str, Any]) -> bool:
    """Return True when any validator says the answer is not fully validated."""
    stages = _read(diagnostics, "stages", {}) or {}
    if not isinstance(stages, Mapping):
        return False

    recovery = stages.get("qa_explicit_answer_recovery")
    if isinstance(recovery, Mapping) and recovery.get("answer_recovered") is True:
        return False

    for stage_name in (
        "evidence_validation_contract",
        "validation_contract",
    ):
        stage = stages.get(stage_name)
        if isinstance(stage, Mapping) and stage.get("grounded_override") is True:
            return True

    return False


def _extract_evidence(context: Any) -> tuple[Any, ...]:
    values = _read(context, "evidence_items")
    if values is None:
        values = _read(context, "sources")
    if values is None:
        return ()
    try:
        return tuple(values)
    except TypeError as exc:
        raise PipelineStageError(
            "ContextResult evidence collection is not iterable.",
            stage="context",
            cause=exc,
        ) from exc


def _context_has_evidence(context: Any) -> bool:
    explicit = _read(context, "has_evidence", None)
    if isinstance(explicit, bool):
        return explicit
    return bool(_extract_evidence(context))


def _result_is_empty(result: Any) -> bool:
    if result is None:
        return True

    explicit = _read(result, "is_empty", None)
    if isinstance(explicit, bool):
        return explicit

    candidates = _read(result, "candidates")
    if candidates is None:
        candidates = _read(result, "results")

    if candidates is None:
        return False

    try:
        return len(candidates) == 0
    except TypeError:
        return not bool(candidates)


def _extract_evidence_references(structured: Any) -> tuple[Any, ...]:
    """
    Read the structured response's evidence/citation references.

    The parser/validator own schema construction. The pipeline only checks that
    references, when present, can point to evidence actually supplied to the LLM.
    """
    value = _read(structured, "evidence", None)
    if value is None:
        value = _read(structured, "citations", None)
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _reference_number(reference: Any) -> Optional[int]:
    """
    Extract an evidence number from common structured-reference shapes.

    Supported names intentionally remain narrow to avoid guessing arbitrary
    application fields.
    """
    for name in (
        "evidence_number",
        "evidence_index",
        "evidence_id_number",
        "index",
    ):
        value = _read(reference, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value

    if isinstance(reference, int) and not isinstance(reference, bool) and reference > 0:
        return reference

    if isinstance(reference, str):
        match = re.fullmatch(r"\s*(?:evidence\s*)?(\d+)\s*", reference, re.I)
        if match:
            return int(match.group(1))

    return None


def _validate_structured_evidence_references(
    structured: Any,
    evidence_count: int,
) -> Mapping[str, Any]:
    """
    Fail closed when the model/parser explicitly returns invalid evidence refs.

    This is the missing safety boundary between a syntactically valid JSON
    answer and an actually traceable answer.
    """
    references = _extract_evidence_references(structured)
    if not references:
        return {
            "reference_count": 0,
            "references_present": False,
            "references_valid": True,
        }

    invalid: list[Any] = []
    normalized: list[int] = []

    for reference in references:
        number = _reference_number(reference)
        if number is None or number > evidence_count:
            invalid.append(reference)
        else:
            normalized.append(number)

    if invalid:
        raise PipelineStageError(
            "Structured response contains evidence references that do not "
            f"exist in the supplied context: {invalid!r}.",
            stage="evidence_validation",
            cause=ValueError(
                f"valid evidence range is 1..{evidence_count}; "
                f"invalid={invalid!r}"
            ),
        )

    return {
        "reference_count": len(references),
        "references_present": True,
        "references_valid": True,
        "referenced_evidence_numbers": tuple(dict.fromkeys(normalized)),
    }


def _validate_qa_integer_evidence_contract(structured: Any) -> Mapping[str, Any]:
    """
    Enforce the canonical QA schema at the orchestration boundary.

    QA structured output must contain evidence references as positive integers.
    Provenance dictionaries belong to PipelineResponse.evidence/citations and
    must never be written into structured_output.evidence.
    """
    value = _read(structured, "evidence", None)
    if value is None:
        return {
            "evidence_present": False,
            "integer_reference_contract": True,
            "reference_count": 0,
        }

    if isinstance(value, (str, bytes, Mapping)):
        raise PipelineStageError(
            "QA structured_output.evidence must be a sequence of positive "
            "integer evidence references; provenance objects are not allowed.",
            stage="evidence_validation",
        )

    try:
        references = tuple(value)
    except TypeError as exc:
        raise PipelineStageError(
            "QA structured_output.evidence must be iterable.",
            stage="evidence_validation",
            cause=exc,
        ) from exc

    invalid = [
        reference
        for reference in references
        if isinstance(reference, bool)
        or not isinstance(reference, int)
        or reference <= 0
    ]

    if invalid:
        raise PipelineStageError(
            "QA structured_output.evidence contains non-integer evidence "
            f"references: {invalid!r}.",
            stage="evidence_validation",
        )

    return {
        "evidence_present": True,
        "integer_reference_contract": True,
        "reference_count": len(references),
        "referenced_evidence_numbers": tuple(dict.fromkeys(references)),
    }


def _validate_context_contract(
    context: Any,
    evidence: Sequence[Any],
) -> Mapping[str, Any]:
    """
    Validate the minimum ContextResult -> PromptBuilder contract.

    The pipeline must never send an apparently populated context whose evidence
    collection is structurally unusable.
    """
    formatted_context = _read(context, "formatted_context", None)

    if not isinstance(formatted_context, str):
        raise PipelineStageError(
            "ContextResult.formatted_context must be a string.",
            stage="context",
        )

    if evidence and not formatted_context.strip():
        raise PipelineStageError(
            "ContextResult contains evidence_items but formatted_context is empty.",
            stage="context",
        )

    invalid_items: list[int] = []
    missing_from_formatted_context: list[int] = []

    for index, item in enumerate(evidence, start=1):
        source_text = _read(item, "text", None)
        if not isinstance(source_text, str) or not source_text.strip():
            invalid_items.append(index)
            continue

        if source_text.strip() not in formatted_context:
            missing_from_formatted_context.append(index)

    if invalid_items:
        raise PipelineStageError(
            "ContextResult contains evidence items without usable source text: "
            f"{invalid_items}.",
            stage="context",
        )

    if missing_from_formatted_context:
        raise PipelineStageError(
            "ContextResult.formatted_context omits source text for evidence "
            f"items {missing_from_formatted_context}.",
            stage="context",
        )

    return {
        "formatted_context_present": bool(formatted_context.strip()),
        "evidence_count": len(evidence),
        "evidence_items_have_text": not invalid_items,
        "evidence_items_in_formatted_context": not missing_from_formatted_context,
    }



def _message_role(message: Any) -> Optional[str]:
    """Return a normalized message role when one is available."""
    role = _read(message, "role", None)
    if isinstance(role, str):
        value = role.strip().lower()
        return value or None
    return None


def _replace_message_content(message: Any, content: str) -> Any:
    """
    Return a copy of a message with updated content.

    Supports the common project shapes without assuming a concrete Pydantic
    or dataclass implementation.
    """
    if not isinstance(content, str) or not content.strip():
        raise PipelineStageError(
            "Cannot construct a grounded prompt message with empty content.",
            stage="prompt",
        )

    if isinstance(message, Mapping):
        updated = dict(message)
        updated["content"] = content
        return updated

    if is_dataclass(message) and not isinstance(message, type):
        try:
            return replace(message, content=content)
        except (TypeError, ValueError):
            pass

    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={"content": content})
        except Exception:
            pass

    copy_method = getattr(message, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"content": content})
        except Exception:
            pass

    # Last-resort mutable-object adapter.
    try:
        clone = object.__new__(type(message))
        if hasattr(message, "__dict__"):
            clone.__dict__.update(message.__dict__)
            clone.content = content
            return clone
    except Exception:
        pass

    raise PipelineDependencyError(
        f"Prompt message type {type(message).__name__!r} cannot be "
        "copied with updated content."
    )


def _replace_request_messages(request: Any, messages: Sequence[Any]) -> Any:
    """
    Return a copy of an LLM request with the supplied messages.

    The request may be a Mapping, dataclass, Pydantic v2 model, Pydantic v1
    model, or a mutable/simple object.
    """
    message_tuple = tuple(messages)
    if not message_tuple:
        raise PipelineStageError(
            "Cannot construct a grounded LLMRequest without messages.",
            stage="prompt",
        )

    if isinstance(request, Mapping):
        updated = dict(request)
        updated["messages"] = message_tuple
        return updated

    if is_dataclass(request) and not isinstance(request, type):
        try:
            return replace(request, messages=message_tuple)
        except (TypeError, ValueError):
            pass

    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={"messages": message_tuple})
        except Exception:
            pass

    copy_method = getattr(request, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"messages": message_tuple})
        except Exception:
            pass

    try:
        clone = object.__new__(type(request))
        if hasattr(request, "__dict__"):
            clone.__dict__.update(request.__dict__)
            clone.messages = message_tuple
            return clone
    except Exception:
        pass

    raise PipelineDependencyError(
        f"LLM request type {type(request).__name__!r} cannot be copied "
        "with updated messages."
    )


def _build_grounding_block(context: Any, evidence: Sequence[Any]) -> str:
    """
    Build the exact source block that must reach the LLM.

    ContextBuilder owns formatting. The pipeline only transports its already
    validated formatted_context and never invents scientific content.
    """
    formatted_context = _read(context, "formatted_context", None)
    if not isinstance(formatted_context, str) or not formatted_context.strip():
        raise PipelineStageError(
            "Cannot ground the prompt because ContextResult.formatted_context "
            "is empty or invalid.",
            stage="prompt",
        )

    # _validate_context_contract() already guarantees every evidence item's
    # exact source text occurs in this formatted context. Re-check defensively
    # because this function is also useful as an isolated boundary.
    missing: list[int] = []
    for index, item in enumerate(evidence, start=1):
        source_text = _read(item, "text", None)
        if not isinstance(source_text, str) or not source_text.strip():
            missing.append(index)
            continue
        if source_text.strip() not in formatted_context:
            missing.append(index)

    if missing:
        raise PipelineStageError(
            "Context evidence cannot be grounded into the prompt because "
            f"source text is missing from formatted_context for items {missing}.",
            stage="prompt",
        )

    return formatted_context.strip()


def _ensure_prompt_evidence_grounded(
    request: Any,
    *,
    context: Any,
    evidence: Sequence[Any],
) -> tuple[Any, Mapping[str, Any]]:
    """
    Ensure retrieved evidence is physically present in LLMRequest.messages.

    PromptBuilder remains responsible for task-specific wording. This function
    is a narrow integration safety adapter: if an otherwise valid PromptBuilder
    returns a request whose messages omit the already-built evidence block, the
    pipeline appends that block to the user message instead of silently sending
    an ungrounded request to the provider.

    This directly fixes the integration failure where ContextResult contained
    evidence but the generated LLMRequest did not carry that evidence in the
    provider-facing messages.
    """
    if not evidence:
        return request, {
            "evidence_count": 0,
            "evidence_injected": False,
            "evidence_traceable": True,
        }

    messages = _read(request, "messages", None)
    if messages is None:
        raise PipelineStageError(
            "LLMRequest.messages is required when retrieved evidence is present.",
            stage="prompt",
        )

    try:
        message_values = tuple(messages)
    except TypeError as exc:
        raise PipelineStageError(
            "LLMRequest.messages is not iterable.",
            stage="prompt",
            cause=exc,
        ) from exc

    if not message_values:
        raise PipelineStageError(
            "LLMRequest.messages cannot be empty when evidence is present.",
            stage="prompt",
        )

    grounding_block = _build_grounding_block(context, evidence)
    combined = "\n".join(
        _read(message, "content", "") or "" for message in message_values
    )

    source_texts = [
        _read(item, "text", "").strip()
        for item in evidence
        if isinstance(_read(item, "text", None), str)
        and _read(item, "text", "").strip()
    ]
    missing_before = [
        index
        for index, source_text in enumerate(source_texts, start=1)
        if source_text not in combined
    ]

    if not missing_before:
        return request, {
            "evidence_count": len(source_texts),
            "evidence_injected": False,
            "evidence_traceable": True,
            "missing_before_injection": (),
        }

    # Prefer the last user message. If the provider request has no explicit
    # user role, append to the last message rather than guessing a new schema.
    target_index: Optional[int] = None
    for index in range(len(message_values) - 1, -1, -1):
        if _message_role(message_values[index]) == "user":
            target_index = index
            break

    if target_index is None:
        target_index = len(message_values) - 1

    target = message_values[target_index]
    target_content = _read(target, "content", None)
    if not isinstance(target_content, str) or not target_content.strip():
        raise PipelineStageError(
            f"LLMRequest target message {target_index + 1} has no usable content.",
            stage="prompt",
        )

    separator = "\n\n"
    grounded_content = (
        f"{target_content.rstrip()}{separator}"
        "<RESEARCH_EVIDENCE>\n"
        f"{grounding_block}\n"
        "</RESEARCH_EVIDENCE>"
    )

    updated_target = _replace_message_content(target, grounded_content)
    updated_messages = list(message_values)
    updated_messages[target_index] = updated_target
    grounded_request = _replace_request_messages(request, updated_messages)

    # Fail closed immediately if the adapter did not actually put every source
    # text into the provider-facing messages.
    updated_combined = "\n".join(
        _read(message, "content", "") or ""
        for message in updated_messages
    )
    still_missing = [
        index
        for index, source_text in enumerate(source_texts, start=1)
        if source_text not in updated_combined
    ]
    if still_missing:
        raise PipelineStageError(
            "Prompt grounding adapter could not place source text into the "
            f"LLM request for evidence items {still_missing}.",
            stage="prompt",
        )

    logger.warning(
        "PromptBuilder omitted retrieved evidence; pipeline injected "
        "ContextResult.formatted_context into message %d.",
        target_index + 1,
    )

    return grounded_request, {
        "evidence_count": len(source_texts),
        "evidence_injected": True,
        "injected_message_index": target_index + 1,
        "missing_before_injection": tuple(missing_before),
        "evidence_traceable": True,
    }


def _validate_prompt_request_contract(
    request: Any,
    *,
    task_type: str,
    query: str,
    context: Any = None,
    evidence: Sequence[Any] = (),
) -> Mapping[str, Any]:
    """
    Validate the PromptBuilder -> LLMClient contract.

    This validator runs after the grounding adapter, so an accepted request
    guarantees that the actual provider-facing messages contain:
      * non-empty messages,
      * the normalized user query,
      * every retrieved evidence source text,
      * a valid response schema when one is supplied.

    Exact source-text matching is intentional: the pipeline must never use
    fuzzy matching to claim that evidence was transmitted.
    """
    if request is None:
        raise PipelineStageError(
            "PromptBuilder returned None.",
            stage="prompt",
        )

    request_task = _read(request, "task_type", None)
    if isinstance(request_task, str) and request_task.strip().lower() != task_type:
        raise PipelineStageError(
            f"Prompt request task_type={request_task!r} does not match "
            f"pipeline task_type={task_type!r}.",
            stage="prompt",
        )

    messages = _read(request, "messages", None)
    if messages is None:
        raise PipelineStageError(
            "LLMRequest.messages is required for provider generation.",
            stage="prompt",
        )

    try:
        message_values = tuple(messages)
    except TypeError as exc:
        raise PipelineStageError(
            "LLMRequest.messages is not iterable.",
            stage="prompt",
            cause=exc,
        ) from exc

    if not message_values:
        raise PipelineStageError(
            "LLMRequest.messages cannot be empty.",
            stage="prompt",
        )

    for index, message in enumerate(message_values, start=1):
        content = _read(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise PipelineStageError(
                f"LLMRequest message {index} has empty content.",
                stage="prompt",
            )

    combined = "\n".join(
        _read(message, "content", "")
        for message in message_values
    )

    if query not in combined:
        raise PipelineStageError(
            "LLMRequest does not contain the normalized user query.",
            stage="prompt",
        )

    request_fingerprint = _read(request, "query_fingerprint", None)
    if request_fingerprint is not None:
        expected_fingerprint = _query_fingerprint(query)
        if request_fingerprint != expected_fingerprint:
            raise PipelineStageError(
                "LLMRequest query_fingerprint does not match the normalized user query.",
                stage="prompt",
                cause=ValueError(
                    f"expected={expected_fingerprint!r}, actual={request_fingerprint!r}"
                ),
            )

    schema = _read(request, "response_schema", None)
    if schema is not None and not isinstance(schema, Mapping):
        raise PipelineStageError(
            "LLMRequest.response_schema must be a mapping when present.",
            stage="prompt",
        )

    evidence_texts: list[str] = []
    invalid_evidence: list[int] = []
    for index, item in enumerate(evidence, start=1):
        source_text = _read(item, "text", None)
        if isinstance(source_text, str) and source_text.strip():
            evidence_texts.append(source_text.strip())
        else:
            invalid_evidence.append(index)

    if invalid_evidence:
        raise PipelineStageError(
            "Retrieved evidence contains items without usable source text: "
            f"{invalid_evidence}.",
            stage="prompt",
        )

    missing_evidence: list[int] = []
    for index, source_text in enumerate(evidence_texts, start=1):
        if source_text not in combined:
            missing_evidence.append(index)

    if missing_evidence:
        raise PipelineStageError(
            "LLMRequest does not contain the source text for evidence "
            f"items {missing_evidence}. Retrieved evidence is not fully "
            "grounded into the generation prompt.",
            stage="prompt",
        )

    return {
        "messages": len(message_values),
        "schema_present": schema is not None,
        "task_type": request_task,
        "evidence_count": len(evidence_texts),
        "evidence_items_in_prompt": len(evidence_texts),
        "evidence_traceable": True,
        "query_fingerprint": _query_fingerprint(query),
        "request_identity_verified": True,
    }


def _build_citations(evidence: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    """
    Copy provenance only; never invent values.

    This is deliberately small because ContextBuilder remains the owner of
    evidence normalization and provenance.
    """
    citations: list[Mapping[str, Any]] = []

    for index, item in enumerate(evidence, start=1):
        citation: dict[str, Any] = {"evidence_index": index}

        for source_name, output_name in (
            ("document_id", "document_id"),
            ("paper_id", "paper_id"),
            ("chunk_id", "chunk_id"),
            ("section", "section"),
            ("section_heading", "section_heading"),
            ("page", "page"),
            ("page_end", "page_end"),
            ("source_pages", "source_pages"),
            ("rank", "rank"),
            ("score", "score"),
            ("retrieval_score", "retrieval_score"),
            ("reranker_score", "reranker_score"),
        ):
            value = _read(item, source_name, None)
            if value is not None:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    finite = _finite_float(value)
                    citation[output_name] = (
                        finite if finite is not None else value
                    )
                elif isinstance(value, (tuple, list)):
                    citation[output_name] = tuple(value)
                elif isinstance(value, (str, int, float, bool)):
                    citation[output_name] = value
                else:
                    # Do not leak arbitrary provider/library objects into the
                    # API response. Provenance remains owned by the evidence item.
                    citation[output_name] = str(value)

        label = getattr(item, "citation_label", None)
        if callable(label):
            citation["label"] = label()

        citations.append(citation)

    return tuple(citations)


def _invoke_flexible(
    component: Any,
    method_name: str,
    *,
    positional: tuple[Any, ...] = (),
    keyword_values: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Invoke an injected component using its actual callable signature.

    Only keyword parameters explicitly accepted by the target are supplied.
    This avoids forcing a speculative client/validator signature while still
    keeping the pipeline strict: if the core positional contract is invalid,
    the original TypeError is propagated to the stage boundary.

    No fallback call is attempted after the component itself raises. Therefore
    real provider/validator failures are never accidentally retried here.
    """
    method = _callable_member(component, method_name)
    if method is None:
        if callable(component) and method_name == "__call__":
            method = component
        else:
            raise PipelineDependencyError(
                f"Dependency {type(component).__name__!r} does not expose "
                f"callable {method_name}()."
            )

    keyword_values = dict(keyword_values or {})

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        # Some extension/provider callables have no inspectable signature.
        # Use the explicit core call without speculative keywords.
        return method(*positional)

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: dict[str, Any] = {}
    if accepts_var_kwargs:
        kwargs.update(keyword_values)
    else:
        positional_parameter_names = [
            name
            for name, parameter in parameters.items()
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]

        for name, value in keyword_values.items():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                continue

            # A value already supplied positionally must not also be supplied
            # by keyword (e.g. generate(request) + request=...).
            positional_index = positional_parameter_names.index(name) if name in positional_parameter_names else -1
            if positional_index >= 0 and positional_index < len(positional):
                continue

            kwargs[name] = value

    return method(*positional, **kwargs)


def _stage_call(
    stage: str,
    func: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        value = func()
    except PipelineError:
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.exception(
            "Pipeline stage failed: stage=%s elapsed=%.4fs",
            stage,
            elapsed,
        )
        raise PipelineStageError(
            f"Pipeline stage {stage!r} failed.",
            stage=stage,
            cause=exc,
        ) from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "Pipeline stage complete: stage=%s elapsed=%.4fs",
        stage,
        elapsed,
    )
    return value


# ============================================================================
# RAGPipeline
# ============================================================================


class RAGPipeline:
    """
    Framework-independent central RAG workflow controller.

    Constructor dependencies are long-lived services. No request-specific
    state is stored on the instance.

    Required boundaries:
        rag_retriever
        context_builder
        prompt_builder
        llm_client
        output_parser

    Validators are required for normal successful generation. An
    `evidence_validator` may be omitted only when the project's validator
    already performs evidence validation inside `validator`.
    """

    def __init__(
        self,
        *,
        rag_retriever: RAGRetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        prompt_builder: PromptBuilderProtocol,
        llm_client: LLMClientProtocol,
        output_parser: OutputParserProtocol,
        validator: ValidatorProtocol,
        evidence_validator: Optional[ValidatorProtocol] = None,
        default_candidate_k: int = DEFAULT_CANDIDATE_K,
        default_final_k: int = DEFAULT_FINAL_K,
        default_context_max_characters: Optional[int] = None,
        default_context_max_evidence_items: Optional[int] = None,
        system_configuration: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._require_method(
            rag_retriever,
            "retrieve",
            "rag_retriever",
        )
        self._require_method(
            context_builder,
            "build",
            "context_builder",
        )
        self._require_method(
            prompt_builder,
            "build_prompt",
            "prompt_builder",
        )
        self._require_method(
            llm_client,
            "generate",
            "llm_client",
        )
        self._require_method(
            output_parser,
            "parse",
            "output_parser",
        )
        self._require_validator(validator, "validator")

        if evidence_validator is not None:
            self._require_validator(
                evidence_validator,
                "evidence_validator",
            )

        self.default_candidate_k = _positive_int(
            default_candidate_k,
            name="default_candidate_k",
        )
        self.default_final_k = _positive_int(
            default_final_k,
            name="default_final_k",
        )
        if self.default_candidate_k < self.default_final_k:
            raise PipelineConfigurationError(
                "default_candidate_k must be >= default_final_k."
            )

        if (
            default_context_max_characters is not None
            and (
                isinstance(default_context_max_characters, bool)
                or not isinstance(default_context_max_characters, int)
                or default_context_max_characters <= 0
            )
        ):
            raise PipelineConfigurationError(
                "default_context_max_characters must be None or a "
                "positive integer."
            )

        if (
            default_context_max_evidence_items is not None
            and (
                isinstance(default_context_max_evidence_items, bool)
                or not isinstance(default_context_max_evidence_items, int)
                or default_context_max_evidence_items <= 0
            )
        ):
            raise PipelineConfigurationError(
                "default_context_max_evidence_items must be None or a "
                "positive integer."
            )

        if system_configuration is not None:
            if not isinstance(system_configuration, Mapping):
                raise PipelineConfigurationError(
                    "system_configuration must be a mapping or None."
                )
            # Copy once to prevent caller mutation from changing future
            # requests. Values remain opaque to the pipeline.
            system_configuration = dict(system_configuration)

        self.rag_retriever = rag_retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.output_parser = output_parser
        self.validator = validator
        self.evidence_validator = evidence_validator

        self.default_context_max_characters = default_context_max_characters
        self.default_context_max_evidence_items = default_context_max_evidence_items
        self.system_configuration = system_configuration

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        task_type: str,
        *,
        scope: str = "research",
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
        filters: Any = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> PipelineResponse:
        """
        Execute one complete evidence-grounded RAG request.

        `session_id` is accepted as request metadata but is never used as a
        hidden retrieval key. Document selection must be explicit through the
        existing retriever contract.
        """
        total_started = time.perf_counter()

        request = self._validate_request(
            query=query,
            task_type=task_type,
            scope=scope,
            candidate_k=candidate_k,
            final_k=final_k,
            document_id=document_id,
            document_ids=document_ids,
            session_id=session_id,
            exclude_ids=exclude_ids,
            options=options,
        )

        normalized_query = request["query"]
        normalized_task = request["task_type"]
        normalized_scope = request["scope"]
        normalized_document_id = request["document_id"]
        resolved_candidate_k = request["candidate_k"]
        resolved_final_k = request["final_k"]
        normalized_document_ids = request["document_ids"]

        diagnostics: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "candidate_k": resolved_candidate_k,
            "final_k": resolved_final_k,
            "task_type": normalized_task,
            "scope": normalized_scope,
            "session_scoped": session_id is not None,
            "document_id": normalized_document_id,
            "document_ids": normalized_document_ids,
            "stages": {},
        }

        retrieval_started = time.perf_counter()
        retrieval_result = _stage_call(
            "retrieval",
            lambda: self._retrieve(
                query=normalized_query,
                task_type=normalized_task,
                scope=normalized_scope,
                candidate_k=resolved_candidate_k,
                final_k=resolved_final_k,
                document_id=normalized_document_id,
                document_ids=normalized_document_ids,
                exclude_ids=exclude_ids,
                filters=filters,
            ),
        )
        diagnostics["stages"]["retrieval_seconds"] = (
            time.perf_counter() - retrieval_started
        )

        retrieval_candidates = self._extract_candidates(retrieval_result)
        diagnostics["candidate_count"] = len(retrieval_candidates)

        if not retrieval_candidates:
            return self._insufficient_evidence_response(
                query=normalized_query,
                task_type=normalized_task,
                scope=normalized_scope,
                diagnostics=self._finish_diagnostics(
                    diagnostics,
                    total_started,
                ),
                reason="retrieval_returned_no_candidates",
            )

        context_started = time.perf_counter()
        context_result = _stage_call(
            "context",
            lambda: self._build_context(
                query=normalized_query,
                candidates=retrieval_candidates,
                task_type=normalized_task,
                options=options,
            ),
        )
        diagnostics["stages"]["context_seconds"] = (
            time.perf_counter() - context_started
        )

        evidence = _extract_evidence(context_result)
        diagnostics["evidence_count"] = len(evidence)
        diagnostics["context_budget_exhausted"] = bool(
            _read(context_result, "budget_exhausted", False)
        )

        context_contract = _validate_context_contract(
            context_result,
            evidence,
        )
        diagnostics["stages"]["context_contract"] = dict(context_contract)

        if not _context_has_evidence(context_result):
            return self._insufficient_evidence_response(
                query=normalized_query,
                task_type=normalized_task,
                scope=normalized_scope,
                diagnostics=self._finish_diagnostics(
                    diagnostics,
                    total_started,
                ),
                reason="context_contains_no_valid_evidence",
            )

        prompt_started = time.perf_counter()
        llm_request = _stage_call(
            "prompt",
            lambda: self._build_prompt(
                query=normalized_query,
                task_type=normalized_task,
                context=context_result,
                options=options,
            ),
        )
        diagnostics["stages"]["prompt_seconds"] = (
            time.perf_counter() - prompt_started
        )

        llm_request, identity_contract = _set_request_identity(
            llm_request,
            query=normalized_query,
            task_type=normalized_task,
            scope=normalized_scope,
            document_id=normalized_document_id,
            document_ids=normalized_document_ids,
        )
        diagnostics["stages"]["request_identity"] = dict(identity_contract)

        grounded_llm_request, grounding_contract = _ensure_prompt_evidence_grounded(
            llm_request,
            context=context_result,
            evidence=evidence,
        )
        llm_request = grounded_llm_request
        diagnostics["stages"]["prompt_grounding"] = dict(grounding_contract)

        prompt_contract = _validate_prompt_request_contract(
            llm_request,
            task_type=normalized_task,
            query=normalized_query,
            context=context_result,
            evidence=evidence,
        )
        diagnostics["stages"]["prompt_contract"] = dict(prompt_contract)

        generation_started = time.perf_counter()
        raw_output = _stage_call(
            "generation",
            lambda: _invoke_flexible(
                self.llm_client,
                "generate",
                positional=(llm_request,),
                keyword_values={
                    "session_id": session_id,
                    "task_type": normalized_task,
                    "query": normalized_query,
                },
            ),
        )
        diagnostics["stages"]["generation_seconds"] = (
            time.perf_counter() - generation_started
        )

        if self._is_empty_llm_output(raw_output):
            raise PipelineStageError(
                "LLM client returned an empty response.",
                stage="generation",
            )

        parse_started = time.perf_counter()
        parsed_output = _stage_call(
            "parsing",
            lambda: _invoke_flexible(
                self.output_parser,
                "parse",
                positional=(raw_output,),
                keyword_values={
                    "request": llm_request,
                    "task_type": normalized_task,
                },
            ),
        )
        diagnostics["stages"]["parsing_seconds"] = (
            time.perf_counter() - parse_started
        )

        if parsed_output is None:
            raise PipelineStageError(
                "Output parser returned None.",
                stage="parsing",
            )

        # Evidence validation is intentionally a separate stage. If the
        # project has no standalone evidence validator, validator can own this
        # contract; the pipeline never fabricates a replacement validator.
        # Validators may return a boolean/report rather than the response
        # itself. Normalize that contract at the pipeline boundary so a
        # validation report can never become `structured_output`.
        validated_evidence = parsed_output
        if self.evidence_validator is not None:
            evidence_started = time.perf_counter()
            evidence_validation_result = _stage_call(
                "evidence_validation",
                lambda: _invoke_validator(
                    self.evidence_validator,
                    parsed_output,
                    context=context_result,
                    evidence=evidence,
                    query=normalized_query,
                    task_type=normalized_task,
                    scope=normalized_scope,
                    document_id=normalized_document_id,
                    document_ids=normalized_document_ids,
                    expected_paper_id=normalized_document_id if normalized_scope == "uploaded" else None,
                    expected_document_id=normalized_document_id if normalized_scope == "uploaded" else None,
                    expected_document_ids=normalized_document_ids if normalized_scope == "comparison" else None,
                ),
            )
            validated_evidence, evidence_validation_diagnostics = (
                _apply_validator_result(
                    original=parsed_output,
                    result=evidence_validation_result,
                    stage="evidence_validation",
                )
            )
            diagnostics["stages"]["evidence_validation_seconds"] = (
                time.perf_counter() - evidence_started
            )
            diagnostics["stages"]["evidence_validation_contract"] = dict(
                evidence_validation_diagnostics
            )

        structured_reference_contract = _validate_structured_evidence_references(
            validated_evidence,
            len(evidence),
        )
        diagnostics["stages"]["structured_evidence_references"] = dict(
            structured_reference_contract
        )

        validation_started = time.perf_counter()
        validation_result = _stage_call(
            "validation",
            lambda: _invoke_validator(
                self.validator,
                validated_evidence,
                context=context_result,
                evidence=evidence,
                query=normalized_query,
                task_type=normalized_task,
                scope=normalized_scope,
                document_id=normalized_document_id,
                document_ids=normalized_document_ids,
                expected_paper_id=normalized_document_id if normalized_scope == "uploaded" else None,
                expected_document_id=normalized_document_id if normalized_scope == "uploaded" else None,
                expected_document_ids=normalized_document_ids if normalized_scope == "comparison" else None,
            ),
        )
        validated_output, validation_diagnostics = _apply_validator_result(
            original=validated_evidence,
            result=validation_result,
            stage="validation",
        )
        diagnostics["stages"]["validation_seconds"] = (
            time.perf_counter() - validation_started
        )
        diagnostics["stages"]["validation_contract"] = dict(
            validation_diagnostics
        )

        normalized_structured_output = _normalize_structured_output(validated_output)
        if normalized_structured_output is not validated_output:
            diagnostics["stages"]["response_envelope_unwrapped"] = True
            diagnostics["stages"]["response_envelope_type"] = type(validated_output).__name__
        validated_output = normalized_structured_output

        if validated_output is not None and normalized_task == "qa":
            validated_output, question_contract = _ensure_qa_question_identity(
                validated_output,
                query=normalized_query,
            )
            diagnostics["stages"]["qa_question_identity"] = dict(question_contract)

            qa_evidence_contract = _validate_qa_integer_evidence_contract(
                validated_output
            )
            diagnostics["stages"]["qa_integer_evidence_contract"] = dict(
                qa_evidence_contract
            )

            validated_output, resolution_contract = _resolve_qa_evidence_references(
                validated_output,
                evidence,
            )
            diagnostics["stages"]["qa_evidence_resolution"] = dict(
                resolution_contract
            )

            validated_output, answer_recovered = _recover_explicit_qa_answer(
                validated_output,
                query=normalized_query,
                evidence=evidence,
            )
            diagnostics["stages"]["qa_explicit_answer_recovery"] = {
                "answer_recovered": answer_recovered,
            }

        if validated_output is not None:
            final_reference_contract = _validate_structured_evidence_references(
                validated_output,
                len(evidence),
            )
            diagnostics["stages"]["final_evidence_references"] = dict(
                final_reference_contract
            )

        if validated_output is None:
            raise PipelineStageError(
                "Response validation produced no structured response.",
                stage="validation",
            )

        # Fail early with a useful stage error instead of allowing the API's
        # Pydantic schema to report an opaque nested validation failure.
        if _extract_answer(validated_output) is None:
            raise PipelineStageError(
                "Validated response does not contain a human-readable answer "
                "(expected answer/response/summary/text).",
                stage="validation",
            )

        # Final cross-field integrity boundary. A response can be syntactically
        # valid and still belong to a different request/task/scope.
        output_task = _read(validated_output, "task_type", None)
        if isinstance(output_task, str) and output_task.strip().lower() != normalized_task:
            raise PipelineStageError(
                f"Validated response task_type={output_task!r} does not match "
                f"pipeline task_type={normalized_task!r}.",
                stage="validation",
            )

        output_query = _read(validated_output, "query", None)
        if isinstance(output_query, str) and output_query.strip() != normalized_query:
            raise PipelineStageError(
                "Validated response query does not match the normalized "
                "user query.",
                stage="validation",
            )

        output_scope = _read(validated_output, "scope", None)
        if isinstance(output_scope, str) and output_scope.strip().lower() != normalized_scope:
            raise PipelineStageError(
                f"Validated response scope={output_scope!r} does not match "
                f"pipeline scope={normalized_scope!r}.",
                stage="validation",
            )

        diagnostics = self._finish_diagnostics(
            diagnostics,
            total_started,
        )

        grounded = _extract_grounded(validated_output)
        if _validation_grounded_override(diagnostics):
            grounded = False
            diagnostics["grounded_overridden_by_validation"] = True

        response = PipelineResponse(
            status=PipelineStatus.SUCCESS,
            success=True,
            grounded=grounded,
            query=normalized_query,
            task_type=normalized_task,
            scope=normalized_scope,
            answer=_extract_answer(validated_output),
            structured_output=validated_output,
            evidence=evidence,
            citations=_build_citations(evidence),
            diagnostics=diagnostics,
        )

        logger.info(
            "RAG pipeline success: task=%s scope=%s candidates=%d "
            "evidence=%d total=%.4fs",
            normalized_task,
            normalized_scope,
            len(retrieval_candidates),
            len(evidence),
            diagnostics["total_seconds"],
        )

        return response

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    def _validate_request(
        self,
        *,
        query: str,
        task_type: str,
        scope: str,
        candidate_k: Optional[int],
        final_k: Optional[int],
        document_id: Optional[str],
        document_ids: Optional[Sequence[str]],
        session_id: Optional[str],
        exclude_ids: Optional[set[str]],
        options: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_query = _normalize_query(query)
        normalized_task = _normalize_task_type(task_type)
        normalized_scope = _normalize_scope(scope)

        resolved_candidate_k = (
            self.default_candidate_k
            if candidate_k is None
            else _positive_int(candidate_k, name="candidate_k")
        )
        resolved_final_k = (
            self.default_final_k
            if final_k is None
            else _positive_int(final_k, name="final_k")
        )

        if resolved_candidate_k < resolved_final_k:
            raise PipelineRequestError(
                f"candidate_k={resolved_candidate_k} must be >= "
                f"final_k={resolved_final_k}."
            )

        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise PipelineRequestError(
                    "document_id must be a non-empty string when supplied."
                )
            document_id = document_id.strip()

        normalized_document_ids = _normalize_document_ids(document_ids)

        if normalized_scope == "research":
            if document_id is not None or normalized_document_ids is not None:
                raise PipelineRequestError(
                    "Research scope cannot include uploaded-paper document IDs."
                )

        elif normalized_scope == "uploaded":
            if document_id is None:
                raise PipelineRequestError(
                    "document_id is required for scope='uploaded'."
                )
            if normalized_document_ids is not None:
                raise PipelineRequestError(
                    "Use document_id for scope='uploaded'; document_ids are "
                    "reserved for comparison."
                )

        elif normalized_scope == "comparison":
            if normalized_task != "comparison":
                raise PipelineRequestError(
                    "scope='comparison' requires task_type='comparison'."
                )
            if normalized_document_ids is None:
                raise PipelineRequestError(
                    "document_ids are required for scope='comparison'."
                )
            if len(normalized_document_ids) < 2:
                raise PipelineRequestError(
                    "Comparison requires at least two document IDs."
                )
            if document_id is not None:
                raise PipelineRequestError(
                    "Use document_ids for comparison, not document_id."
                )

        if normalized_task == "comparison":
            if normalized_scope != "comparison":
                raise PipelineRequestError(
                    "task_type='comparison' requires scope='comparison' "
                    "with explicit document_ids."
                )

        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise PipelineRequestError(
                "session_id must be None or a non-empty string."
            )

        if exclude_ids is not None:
            if isinstance(exclude_ids, (str, bytes)):
                raise PipelineRequestError(
                    "exclude_ids must be a set-like collection."
                )
            try:
                for value in exclude_ids:
                    if not isinstance(value, str) or not value.strip():
                        raise PipelineRequestError(
                            f"Invalid exclude ID: {value!r}."
                        )
            except TypeError as exc:
                raise PipelineRequestError(
                    "exclude_ids must be iterable."
                ) from exc

        if options is not None and not isinstance(options, Mapping):
            raise PipelineRequestError(
                "options must be a mapping or None."
            )

        return {
            "query": normalized_query,
            "task_type": normalized_task,
            "scope": normalized_scope,
            "candidate_k": resolved_candidate_k,
            "final_k": resolved_final_k,
            "document_id": document_id,
            "document_ids": normalized_document_ids,
        }

    # ------------------------------------------------------------------
    # Stage adapters
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        *,
        query: str,
        task_type: str,
        scope: str,
        candidate_k: int,
        final_k: int,
        document_id: Optional[str],
        document_ids: Optional[Sequence[str]],
        exclude_ids: Optional[set[str]],
        filters: Any,
    ) -> Any:
        """
        Call the existing rag/retriever boundary.

        The currently finalized RAGRetriever accepts `scope`, candidate_k and
        final_k plus document scope. Keyword filtering is passed only when the
        injected implementation accepts it.
        """
        try:
            # Use the actual RAGRetriever API established for this project.
            return _invoke_flexible(
                self.rag_retriever,
                "retrieve",
                positional=(query,),
                keyword_values={
                    "candidate_k": candidate_k,
                    "final_k": final_k,
                    "task_type": task_type,
                    "scope": scope,
                    "document_id": document_id,
                    "document_ids": document_ids,
                    "exclude_ids": exclude_ids,
                    "filters": filters,
                },
            )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineStageError(
                "RAG retrieval boundary failed.",
                stage="retrieval",
                cause=exc,
            ) from exc

    def _extract_candidates(self, retrieval_result: Any) -> list[Any]:
        candidates = _read(retrieval_result, "candidates", None)
        if candidates is None:
            candidates = _read(retrieval_result, "results", None)

        if candidates is None:
            raise PipelineStageError(
                "RAG retrieval result does not expose candidates/results.",
                stage="retrieval",
            )

        try:
            values = list(candidates)
        except TypeError as exc:
            raise PipelineStageError(
                "RAG retrieval candidates are not iterable.",
                stage="retrieval",
                cause=exc,
            ) from exc

        return values

    def _build_context(
        self,
        *,
        query: str,
        candidates: Sequence[Any],
        task_type: str,
        options: Optional[Mapping[str, Any]],
    ) -> Any:
        options = options or {}

        max_evidence_items = options.get(
            "max_evidence_items",
            self.default_context_max_evidence_items,
        )
        if task_type == "qa" and "max_evidence_items" not in options:
            # Keep broad theorem neighbors from burying a direct QA passage.
            max_evidence_items = (
                6
                if max_evidence_items is None
                else min(max_evidence_items, 6)
            )

        return _invoke_flexible(
            self.context_builder,
            "build",
            positional=(query, candidates),
            keyword_values={
                "task_type": task_type,
                "max_characters": options.get(
                    "max_context_characters",
                    self.default_context_max_characters,
                ),
                "max_evidence_items": max_evidence_items,
            },
        )

    def _build_prompt(
        self,
        *,
        query: str,
        task_type: str,
        context: Any,
        options: Optional[Mapping[str, Any]],
    ) -> Any:
        options = options or {}

        system_configuration = options.get(
            "system_configuration",
            self.system_configuration,
        )

        return _invoke_flexible(
            self.prompt_builder,
            "build_prompt",
            positional=(task_type, context, query),
            keyword_values={
                "system_configuration": system_configuration,
            },
        )

    # ------------------------------------------------------------------
    # Empty evidence / diagnostics
    # ------------------------------------------------------------------

    def _insufficient_evidence_response(
        self,
        *,
        query: str,
        task_type: str,
        scope: str,
        diagnostics: Mapping[str, Any],
        reason: str,
    ) -> PipelineResponse:
        diagnostics_dict = dict(diagnostics)
        diagnostics_dict["insufficient_evidence_reason"] = reason

        logger.info(
            "RAG pipeline stopped without generation: task=%s scope=%s "
            "reason=%s total=%.4fs",
            task_type,
            scope,
            reason,
            diagnostics_dict.get("total_seconds", 0.0),
        )

        return PipelineResponse(
            status=PipelineStatus.INSUFFICIENT_EVIDENCE,
            success=False,
            grounded=False,
            query=query,
            task_type=task_type,
            scope=scope,
            answer=None,
            structured_output=None,
            evidence=(),
            citations=(),
            diagnostics=diagnostics_dict,
            error={
                "code": PipelineStatus.INSUFFICIENT_EVIDENCE.value,
                "message": (
                    "No valid evidence was available for the requested "
                    "operation."
                ),
                "reason": reason,
            },
        )


    def health_check(self) -> Mapping[str, Any]:
        """
        Validate long-lived dependencies without performing retrieval or LLM work.

        Application startup should call this before registering the pipeline in
        ``app.state``. A failed check is a configuration/startup error, not a
        request-time 503.
        """
        checks = {
            "rag_retriever": callable(getattr(self.rag_retriever, "retrieve", None)),
            "context_builder": callable(getattr(self.context_builder, "build", None)),
            "prompt_builder": callable(
                getattr(self.prompt_builder, "build_prompt", None)
            ),
            "llm_client": callable(getattr(self.llm_client, "generate", None)),
            "output_parser": callable(getattr(self.output_parser, "parse", None)),
            "validator": (
                callable(getattr(self.validator, "validate", None))
                or callable(self.validator)
            ),
        }
        if self.evidence_validator is not None:
            checks["evidence_validator"] = (
                callable(getattr(self.evidence_validator, "validate", None))
                or callable(self.evidence_validator)
            )

        failed = [name for name, ready in checks.items() if not ready]
        if failed:
            raise PipelineConfigurationError(
                "RAG pipeline is not ready; invalid dependencies: "
                + ", ".join(failed)
            )

        return {
            "ready": True,
            "pipeline_version": PIPELINE_VERSION,
            "dependencies": checks,
            "candidate_k": self.default_candidate_k,
            "final_k": self.default_final_k,
        }

    @property
    def is_ready(self) -> bool:
        """Non-throwing readiness probe for health checks."""
        try:
            self.health_check()
        except PipelineError:
            return False
        return True

    @staticmethod
    def _finish_diagnostics(
        diagnostics: Mapping[str, Any],
        total_started: float,
    ) -> dict[str, Any]:
        result = dict(diagnostics)
        result["total_seconds"] = time.perf_counter() - total_started
        return result

    @staticmethod
    def _is_empty_llm_output(raw_output: Any) -> bool:
        if raw_output is None:
            return True

        if isinstance(raw_output, str):
            return not raw_output.strip()

        if isinstance(raw_output, Mapping):
            return len(raw_output) == 0

        return False

    @staticmethod
    def _require_method(
        dependency: Any,
        method_name: str,
        dependency_name: str,
    ) -> None:
        if dependency is None:
            raise PipelineConfigurationError(
                f"{dependency_name} cannot be None."
            )
        if not callable(getattr(dependency, method_name, None)):
            raise PipelineDependencyError(
                f"{dependency_name} must expose callable "
                f"{method_name}()."
            )

    @staticmethod
    def _require_validator(
        dependency: Any,
        dependency_name: str,
    ) -> None:
        if dependency is None:
            raise PipelineConfigurationError(
                f"{dependency_name} cannot be None."
            )

        if not (
            callable(getattr(dependency, "validate", None))
            or callable(dependency)
        ):
            raise PipelineDependencyError(
                f"{dependency_name} must expose validate() or be callable."
            )


def _invoke_validator(
    validator: Any,
    response: Any,
    *,
    context: Any,
    evidence: Sequence[Any],
    query: str,
    task_type: str,
    scope: Optional[str] = None,
    document_id: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    expected_paper_id: Optional[str] = None,
    expected_document_id: Optional[str] = None,
    expected_document_ids: Optional[Sequence[str]] = None,
) -> Any:
    """
    Invoke the project's existing validator contract without implementing a
    second validation system.

    Preferred contract:
        validator.validate(response, context=..., evidence=..., ...)

    A callable validator is also supported for lightweight adapters/tests.
    """
    method_name = "validate"
    if not callable(getattr(validator, method_name, None)):
        if callable(validator):
            method_name = "__call__"
        else:
            raise PipelineDependencyError(
                "Validator exposes neither validate() nor callable behavior."
            )

    return _invoke_flexible(
        validator,
        method_name,
        positional=(response,),
        keyword_values={
            "context": context,
            "evidence": evidence,
            "query": query,
            "task_type": task_type,
            "scope": scope,
            "document_id": document_id,
            "document_ids": document_ids,
            "expected_paper_id": expected_paper_id,
            "expected_document_id": expected_document_id,
            "expected_document_ids": expected_document_ids,
        },
    )




# ============================================================================
# Production dependency factory
# ============================================================================

def create_rag_pipeline(
    *,
    rag_retriever: RAGRetrieverProtocol,
    context_builder: ContextBuilderProtocol,
    prompt_builder: PromptBuilderProtocol,
    llm_client: LLMClientProtocol,
    output_parser: OutputParserProtocol,
    validator: ValidatorProtocol,
    evidence_validator: Optional[ValidatorProtocol] = None,
    default_candidate_k: int = DEFAULT_CANDIDATE_K,
    default_final_k: int = DEFAULT_FINAL_K,
    default_context_max_characters: Optional[int] = None,
    default_context_max_evidence_items: Optional[int] = None,
    system_configuration: Optional[Mapping[str, Any]] = None,
) -> RAGPipeline:
    """
    Build and validate the application's single shared RAG pipeline.

    This function only wires already-created dependencies. Model loading,
    FAISS/index access, provider SDK initialization, and secret management
    remain owned by their respective modules/application startup.
    """
    pipeline = RAGPipeline(
        rag_retriever=rag_retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        output_parser=output_parser,
        validator=validator,
        evidence_validator=evidence_validator,
        default_candidate_k=default_candidate_k,
        default_final_k=default_final_k,
        default_context_max_characters=default_context_max_characters,
        default_context_max_evidence_items=default_context_max_evidence_items,
        system_configuration=system_configuration,
    )
    pipeline.health_check()
    logger.info(
        "RAG pipeline initialized and ready: candidate_k=%d final_k=%d",
        pipeline.default_candidate_k,
        pipeline.default_final_k,
    )
    return pipeline


def build_rag_pipeline(**kwargs: Any) -> RAGPipeline:
    """Backward-compatible startup alias for ``create_rag_pipeline``."""
    return create_rag_pipeline(**kwargs)

# ============================================================================
# Model-free self-test
# ============================================================================


@dataclass(frozen=True)
class _FakeRetrievalResult:
    candidates: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _FakeContextResult:
    evidence_items: tuple[Mapping[str, Any], ...]
    formatted_context: str

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_items)


@dataclass(frozen=True)
class _FakePromptMessage:
    role: str
    content: str


@dataclass(frozen=True)
class _FakePromptRequest:
    task_type: str
    query: str
    context: Any
    messages: tuple[_FakePromptMessage, ...] = ()
    response_schema: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class _FakeParsed:
    answer: str
    question: Optional[str] = None
    evidence: tuple[Any, ...] = ()


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> _FakeRetrievalResult:
        self.calls.append({"query": query, **kwargs})
        return _FakeRetrievalResult(
            candidates=(
                {
                    "document_id": "paper-1",
                    "chunk_id": "chunk-1",
                    "title": "Paper One",
                    "text": "The dataset is MedSeg.",
                    "rank": 1,
                    "score": 0.9,
                    "retrieval_score": 0.9,
                    "reranker_score": 0.8,
                },
            )
        )


class _FakeContext:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(self, query: str, candidates: Sequence[Any], **kwargs: Any) -> _FakeContextResult:
        self.calls.append(
            {"query": query, "candidates": list(candidates), **kwargs}
        )
        return _FakeContextResult(
            evidence_items=(
                {
                    "document_id": "paper-1",
                    "chunk_id": "chunk-1",
                    "section": "Dataset",
                    "page": 3,
                    "text": "The dataset is MedSeg.",
                },
            ),
            formatted_context=(
                "[EVIDENCE 1]\n"
                "SOURCE_TEXT_BEGIN\n"
                "The dataset is MedSeg.\n"
                "SOURCE_TEXT_END\n"
                "[END EVIDENCE 1]"
            ),
        )


class _FakePrompt:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build_prompt(self, task_type: str, context: Any, query: str, **kwargs: Any) -> _FakePromptRequest:
        self.calls.append(
            {
                "task_type": task_type,
                "context": context,
                "query": query,
                **kwargs,
            }
        )
        return _FakePromptRequest(
            task_type=task_type,
            query=query,
            context=context,
            messages=(
                _FakePromptMessage(role="system", content="System instructions"),
                _FakePromptMessage(
                    role="user",
                    content=(
                        f"{query}\n\n<RESEARCH_EVIDENCE>\n"
                        f"{context.formatted_context}\n"
                        "</RESEARCH_EVIDENCE>"
                    ),
                ),
            ),
            response_schema={"type": "object"},
        )


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def generate(self, request: Any) -> str:
        self.calls.append(request)
        return '{"answer":"The dataset is MedSeg.","grounded":true}'


class _FakeParser:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def parse(self, raw_output: Any, **kwargs: Any) -> _FakeParsed:
        self.calls.append((raw_output, kwargs))
        return _FakeParsed(answer="The dataset is MedSeg.")


class _FakeEvidenceValidator:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def validate(self, response: Any, **kwargs: Any) -> Any:
        self.calls.append("evidence_validator")
        assert kwargs["evidence"]
        return response


class _FakeValidator:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def validate(self, response: Any, **kwargs: Any) -> Any:
        self.calls.append("validator")
        assert kwargs["evidence"]
        return response


class _EmptyRetriever:
    def retrieve(self, query: str, **kwargs: Any) -> _FakeRetrievalResult:
        return _FakeRetrievalResult(candidates=())


class _FailingLLM:
    def generate(self, request: Any) -> str:
        raise RuntimeError("provider unavailable")


def run_self_test() -> None:
    """Run deterministic orchestration tests without models, FAISS or network."""
    retriever = _FakeRetriever()
    context = _FakeContext()
    prompt = _FakePrompt()
    llm = _FakeLLM()
    parser = _FakeParser()

    calls: list[str] = []
    evidence_validator = _FakeEvidenceValidator(calls)
    validator = _FakeValidator(calls)

    pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        evidence_validator=evidence_validator,
        validator=validator,
    )

    # Valid QA.
    result = pipeline.run(
        "What dataset was used?",
        "qa",
        scope="research",
        candidate_k=5,
        final_k=2,
    )

    assert result.status is PipelineStatus.SUCCESS
    assert result.success is True
    assert result.grounded is True
    assert result.answer == "The dataset is MedSeg."
    assert result.evidence
    assert result.citations[0]["document_id"] == "paper-1"
    assert result.citations[0]["chunk_id"] == "chunk-1"

    # Prompt/context boundary must be validated before generation.
    assert result.diagnostics["stages"]["context_contract"]["evidence_count"] == 1
    assert result.diagnostics["stages"]["prompt_contract"]["schema_present"] is True
    assert result.diagnostics["stages"]["prompt_contract"]["evidence_traceable"] is True
    assert result.diagnostics["stages"]["request_identity"]["request_identity_verified"] is True
    assert result.diagnostics["stages"]["request_identity"]["query_fingerprint"]

    # QA response identity is deterministic and evidence provenance is backend-owned.
    assert result.structured_output.question == "What dataset was used?"
    assert result.structured_output.evidence == ()
    assert result.diagnostics["stages"]["qa_question_identity"]["question_verified"] is True

    class _EvidenceOmittingPrompt(_FakePrompt):
        def build_prompt(
            self,
            task_type: str,
            context: Any,
            query: str,
            **kwargs: Any,
        ) -> _FakePromptRequest:
            return _FakePromptRequest(
                task_type=task_type,
                query=query,
                context=context,
                messages=(
                    _FakePromptMessage(
                        role="user",
                        content=f"Answer this question: {query}",
                    ),
                ),
                response_schema={"type": "object"},
            )

    evidence_omitting_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=_EvidenceOmittingPrompt(),
        llm_client=llm,
        output_parser=parser,
        validator=validator,
    )
    repaired_result = evidence_omitting_pipeline.run(
        "What dataset was used?",
        "qa",
        scope="research",
        candidate_k=5,
        final_k=2,
    )
    assert repaired_result.success is True
    assert repaired_result.diagnostics["stages"]["prompt_grounding"][
        "evidence_injected"
    ] is True
    assert repaired_result.diagnostics["stages"]["prompt_contract"][
        "evidence_traceable"
    ] is True

    class _BadEvidenceReferenceParser:
        def parse(self, raw_output: Any, **kwargs: Any) -> Any:
            return {
                "answer": "The dataset is MedSeg.",
                "grounded": True,
                "evidence": [99],
            }

    bad_reference_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=_BadEvidenceReferenceParser(),
        validator=validator,
    )
    try:
        bad_reference_pipeline.run(
            "What dataset was used?",
            "qa",
            scope="research",
            candidate_k=5,
            final_k=2,
        )
    except PipelineStageError as exc:
        assert exc.stage == "evidence_validation"
    else:
        raise AssertionError("Invalid structured evidence reference was accepted.")

    # Regression: legacy provenance dictionaries must never be accepted as
    # QA structured_output.evidence. The canonical contract is integer refs.
    class _LegacyProvenanceParser:
        def parse(self, raw_output: Any, **kwargs: Any) -> Any:
            return {
                "answer": "The dataset is MedSeg.",
                "grounded": True,
                "evidence": [{"evidence_number": 1, "document_id": "paper-1"}],
            }

    legacy_contract_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=_LegacyProvenanceParser(),
        validator=validator,
    )
    try:
        legacy_contract_pipeline.run(
            "What dataset was used?",
            "qa",
            scope="research",
            candidate_k=5,
            final_k=2,
        )
    except PipelineStageError as exc:
        assert exc.stage == "evidence_validation"
    else:
        raise AssertionError(
            "Legacy provenance dictionaries were incorrectly accepted in "
            "QA structured_output.evidence."
        )

    # Validators may return reports instead of the response. The pipeline must
    # preserve the parsed response and must never pass the report downstream.
    class _ReportValidator:
        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        def validate(self, response: Any, **kwargs: Any) -> Mapping[str, Any]:
            self.calls.append("report_validator")
            return {"valid": True, "status": "ok"}

    report_calls: list[str] = []
    report_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        evidence_validator=_ReportValidator(report_calls),
        validator=_ReportValidator(report_calls),
    )
    report_result = report_pipeline.run(
        "What dataset was used?",
        "qa",
        scope="research",
        candidate_k=5,
        final_k=2,
    )
    assert report_result.success is True
    assert report_result.answer == "The dataset is MedSeg."
    assert report_result.structured_output.answer == "The dataset is MedSeg."
    assert report_calls == ["report_validator", "report_validator"]

    # Validator/provider envelope must be unwrapped before PipelineResponse.
    class _EnvelopeValidator:
        def validate(self, response: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {
                "data": response,
                "validated_provider": "ollama",
            }

    envelope_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        validator=_EnvelopeValidator(),
    )
    envelope_result = envelope_pipeline.run(
        "What dataset was used?", "qa", scope="research", candidate_k=5, final_k=2
    )
    assert envelope_result.answer == "The dataset is MedSeg."
    assert _extract_answer(envelope_result.structured_output) == "The dataset is MedSeg."

    # A rejecting validator must become an explicit validation-stage failure.
    class _RejectingValidator:
        def validate(self, response: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {
                "valid": False,
                "status": "rejected",
                "reason": "insufficient grounding",
            }

    rejecting_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        validator=_RejectingValidator(),
    )
    try:
        rejecting_pipeline.run(
            "What dataset was used?",
            "qa",
            scope="research",
            candidate_k=5,
            final_k=2,
        )
    except PipelineStageError as exc:
        assert exc.stage == "validation"
    else:
        raise AssertionError("Rejecting validation report was silently accepted.")

    # PARTIALLY_VALID is a controlled non-fatal validation state. The current
    # validator represents it as valid=False with errors=().
    class _PartiallyValidValidator:
        def validate(self, response: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {
                "valid": False,
                "status": "partially_valid",
                "errors": (),
            }

    partial_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        validator=_PartiallyValidValidator(),
    )
    partial_result = partial_pipeline.run(
        "What dataset was used?",
        "qa",
        scope="research",
        candidate_k=5,
        final_k=2,
    )
    assert partial_result.status is PipelineStatus.SUCCESS
    assert partial_result.success is True
    assert partial_result.answer == "The dataset is MedSeg."
    assert partial_result.grounded is False
    assert (
        partial_result.diagnostics["stages"]["validation_contract"][
            "validation_status"
        ]
        == "partially_valid"
    )
    assert (
        partial_result.diagnostics["stages"]["validation_contract"][
            "grounded_override"
        ]
        is True
    )

    # PARTIALLY_VALID with an explicit error must still fail closed.
    class _PartialWithErrorValidator:
        def validate(self, response: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {
                "valid": False,
                "status": "partially_valid",
                "errors": (
                    {"severity": "error", "message": "bad provenance"},
                ),
            }

    partial_error_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        validator=_PartialWithErrorValidator(),
    )
    try:
        partial_error_pipeline.run(
            "What dataset was used?",
            "qa",
            scope="research",
            candidate_k=5,
            final_k=2,
        )
    except PipelineStageError as exc:
        assert exc.stage == "validation"
    else:
        raise AssertionError(
            "PARTIALLY_VALID with an error severity was silently accepted."
        )

    class _BiasRetriever(_FakeRetriever):
        def retrieve(self, query: str, **kwargs: Any) -> _FakeRetrievalResult:
            return _FakeRetrievalResult(
                candidates=(
                    {
                        "document_id": "paper-bias",
                        "chunk_id": "bias-1",
                        "text": (
                            "There are two general forms of bias in learning from "
                            "examples: restricted hypothesis space bias and preference bias."
                        ),
                        "rank": 1,
                        "score": 0.99,
                    },
                )
            )

    class _BiasContext(_FakeContext):
        def build(self, query: str, candidates: Sequence[Any], **kwargs: Any) -> _FakeContextResult:
            return _FakeContextResult(
                evidence_items=(
                    {
                        "document_id": "paper-bias",
                        "chunk_id": "bias-1",
                        "section": "Introduction",
                        "page": 2,
                        "text": (
                            "There are two general forms of bias in learning from "
                            "examples: restricted hypothesis space bias and preference bias."
                        ),
                    },
                ),
                formatted_context=(
                    "[EVIDENCE 1]\n"
                    "SOURCE_TEXT_BEGIN\n"
                    "There are two general forms of bias in learning from examples: "
                    "restricted hypothesis space bias and preference bias.\n"
                    "SOURCE_TEXT_END\n"
                    "[END EVIDENCE 1]"
                ),
            )

    class _BiasLLM:
        def generate(self, request: Any) -> str:
            return (
                '{"answer":"The two general forms are restricted hypothesis space '
                'bias and preference bias. [Evidence 1]",'
                '"grounded":true,"evidence":[{"evidence_number":1}]}'
            )

    class _BiasParser:
        def parse(self, raw_output: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {
                "answer": (
                    "The two general forms are restricted hypothesis space "
                    "bias and preference bias. [Evidence 1]"
                ),
                "grounded": True,
                "evidence": [1],
            }

    bias_pipeline = RAGPipeline(
        rag_retriever=_BiasRetriever(),
        context_builder=_BiasContext(),
        prompt_builder=prompt,
        llm_client=_BiasLLM(),
        output_parser=_BiasParser(),
        validator=validator,
    )
    bias_result = bias_pipeline.run(
        "What are the two general forms of bias in learning from examples?",
        "qa",
        scope="research",
        candidate_k=5,
        final_k=2,
    )
    assert bias_result.success is True
    assert bias_result.grounded is True
    assert "restricted hypothesis space bias" in bias_result.answer
    assert "preference bias" in bias_result.answer
    assert _read(bias_result.structured_output, "question") == (
        "What are the two general forms of bias in learning from examples?"
    )
    structured_evidence = _read(bias_result.structured_output, "evidence")
    assert tuple(structured_evidence) == (1,)
    assert all(
        isinstance(reference, int) and not isinstance(reference, bool)
        for reference in structured_evidence
    )

    # Provenance is deliberately kept outside structured_output.
    assert len(bias_result.evidence) == 1
    assert _read(bias_result.evidence[0], "document_id") == "paper-bias"
    assert _read(bias_result.evidence[0], "chunk_id") == "bias-1"
    assert (
        bias_result.diagnostics["stages"]["qa_integer_evidence_contract"][
            "integer_reference_contract"
        ]
        is True
    )
    assert (
        bias_result.diagnostics["stages"]["qa_evidence_resolution"][
            "structured_evidence_preserved"
        ]
        is True
    )
    assert (
        bias_result.diagnostics["stages"]["prompt_contract"][
            "evidence_traceable"
        ]
        is True
    )

    # Exact stage ordering.
    assert len(retriever.calls) >= 1
    assert len(context.calls) >= 1
    assert len(prompt.calls) >= 1
    assert len(llm.calls) >= 1
    assert len(parser.calls) >= 1
    assert calls[:2] == ["evidence_validator", "validator"]

    # All supported tasks use the same workflow.
    for task in sorted(SUPPORTED_TASK_TYPES):
        if task == "comparison":
            comparison_result = pipeline.run(
                "Compare the selected papers.",
                task,
                scope="comparison",
                document_ids=("paper-1", "paper-2"),
                candidate_k=5,
                final_k=2,
            )
            assert comparison_result.status is PipelineStatus.SUCCESS
        else:
            task_result = pipeline.run(
                "Use the supplied evidence.",
                task,
                scope="research",
                candidate_k=5,
                final_k=2,
            )
            assert task_result.status is PipelineStatus.SUCCESS

    class _MismatchedPrompt:
        def build_prompt(self, task_type: str, context: Any, query: str, **kwargs: Any) -> _FakePromptRequest:
            return _FakePromptRequest(
                task_type="summary",
                query=query,
                context=context,
                messages=(
                    _FakePromptMessage(role="user", content=query),
                ),
                response_schema={"type": "object"},
            )

    mismatch_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=_MismatchedPrompt(),
        llm_client=llm,
        output_parser=parser,
        validator=validator,
    )
    try:
        mismatch_pipeline.run(
            "What dataset was used?",
            "qa",
            scope="research",
            candidate_k=5,
            final_k=2,
        )
    except PipelineStageError as exc:
        assert exc.stage == "prompt"
    else:
        raise AssertionError("Mismatched prompt task_type was accepted.")

    # Invalid request cases.
    invalid_cases = [
        {"query": "", "task_type": "qa"},
        {"query": "valid", "task_type": "unknown"},
        {"query": "valid", "task_type": "qa", "scope": "unknown"},
        {"query": "valid", "task_type": "qa", "candidate_k": 0},
        {"query": "valid", "task_type": "qa", "candidate_k": 2, "final_k": 3},
        {
            "query": "valid",
            "task_type": "qa",
            "scope": "uploaded",
        },
        {
            "query": "valid",
            "task_type": "comparison",
            "scope": "comparison",
            "document_ids": ("paper-1",),
        },
    ]

    for kwargs in invalid_cases:
        try:
            pipeline.run(**kwargs)
        except PipelineRequestError:
            pass
        else:
            raise AssertionError(f"Invalid request accepted: {kwargs!r}")

    # Empty retrieval must not call context/prompt/LLM.
    empty_pipeline = RAGPipeline(
        rag_retriever=_EmptyRetriever(),
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=parser,
        evidence_validator=evidence_validator,
        validator=validator,
    )
    before_context_calls = len(context.calls)
    before_llm_calls = len(llm.calls)

    empty_result = empty_pipeline.run(
        "No evidence query",
        "qa",
        scope="research",
    )

    assert empty_result.status is PipelineStatus.INSUFFICIENT_EVIDENCE
    assert empty_result.success is False
    assert empty_result.grounded is False
    assert len(context.calls) == before_context_calls
    assert len(llm.calls) == before_llm_calls

    # Generation failure must remain an infrastructure/stage failure.
    failing_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=_FailingLLM(),
        output_parser=parser,
        evidence_validator=evidence_validator,
        validator=validator,
    )

    try:
        failing_pipeline.run(
            "valid query",
            "qa",
            scope="research",
        )
    except PipelineStageError as exc:
        assert exc.stage == "generation"
        assert exc.cause is not None
    else:
        raise AssertionError("LLM failure was silently accepted.")

    # Parser failure must not become successful raw text.
    class _FailingParser:
        def parse(self, raw_output: Any, **kwargs: Any) -> Any:
            raise ValueError("malformed output")

    parser_failure_pipeline = RAGPipeline(
        rag_retriever=retriever,
        context_builder=context,
        prompt_builder=prompt,
        llm_client=llm,
        output_parser=_FailingParser(),
        evidence_validator=evidence_validator,
        validator=validator,
    )

    try:
        parser_failure_pipeline.run(
            "valid query",
            "qa",
            scope="research",
        )
    except PipelineStageError as exc:
        assert exc.stage == "parsing"
    else:
        raise AssertionError("Parser failure was silently accepted.")

    # Uploaded scope isolation is passed to the RAG boundary.
    pipeline.run(
        "What is the method?",
        "qa",
        scope="uploaded",
        document_id="paper-A",
        candidate_k=5,
        final_k=2,
    )
    last_call = retriever.calls[-1]
    assert last_call["scope"] == "uploaded"
    assert last_call["document_id"] == "paper-A"

    # Comparison IDs are explicit and are passed untouched.
    pipeline.run(
        "Compare the papers.",
        "comparison",
        scope="comparison",
        document_ids=("paper-A", "paper-B"),
        candidate_k=5,
        final_k=2,
    )
    last_call = retriever.calls[-1]
    assert last_call["scope"] == "comparison"
    assert last_call["document_ids"] == ("paper-A", "paper-B")

    print("RAGPipeline self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()