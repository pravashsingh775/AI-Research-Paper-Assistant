"""
Evidence-grounded model/algorithm analysis for the AI Research Paper Assistant.

ARCHITECTURAL RESPONSIBILITY
----------------------------
This module is the post-parser analysis boundary for model information.

Expected pipeline:

    retrieval/RAG
        -> context
        -> prompt
        -> LLM client
        -> parser
        -> ModelAnalyzer
        -> validation
        -> frontend/API

This module intentionally does NOT:
    * retrieve papers/chunks
    * generate embeddings
    * access FAISS/SPECTER2
    * rerank/rank papers
    * parse PDFs
    * build prompts
    * call an LLM
    * calculate model performance
    * compare which model is better
    * use web/external model knowledge

IMPORTANT PROJECT CONTRACT
--------------------------
The existing prompt/schema contract defines the model payload as:

    model_name
    architecture
    major_components
    training_approach
    hyperparameters
    baselines
    grounded
    evidence

This module reuses that contract. It does NOT introduce a competing top-level
LLM response schema.

Because the current schema does not expose separate first-class fields for
framework, backbone, pretraining dataset, fine-tuning flag, arbitrary model
roles, or component relationships, those details are preserved only through
the existing fields when the parser provides them. This module never invents
new semantic fields or derives unsupported facts.
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

TASK_TYPE = "model"
SUCCESS = "success"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
INVALID_INPUT = "invalid_input"
VALIDATION_ERROR = "validation_error"

NOT_FOUND = "Not found in the provided evidence."

SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})


class ModelAnalysisError(RuntimeError):
    """Base exception for model-analysis failures."""


class ModelInputError(ModelAnalysisError, ValueError):
    """Invalid model-analysis input or parsed payload."""


class ModelEvidenceError(ModelAnalysisError, ValueError):
    """Evidence/provenance is malformed or crosses paper boundaries."""


class ModelTaskError(ModelAnalysisError, ValueError):
    """Unsupported task type or analysis scope."""


@dataclass(frozen=True)
class ModelAnalysisResult:
    """
    Machine-readable result around the existing project schema.

    `structured_output` is authoritative and keeps the exact parser-owned
    fields. Convenience properties expose those same fields without creating
    a second response schema.
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
    def model_name(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("model_name"))

    @property
    def architecture(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("architecture"))

    @property
    def major_components(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("major_components"))

    @property
    def training_approach(self) -> Optional[str]:
        return _optional_string(
            self.structured_output.get("training_approach")
        )

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("hyperparameters"))

    @property
    def baselines(self) -> tuple[str, ...]:
        return _string_tuple(self.structured_output.get("baselines"))

    @property
    def proposed_or_primary_model(self) -> Optional[str]:
        """
        Existing-schema interpretation only.

        The parser's `model_name` field is the primary extracted model field.
        This property intentionally does NOT claim that the model is
        "proposed" unless the upstream evidence/parser has established that.
        """
        return self.model_name

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
            "model_name": self.model_name,
            "architecture": self.architecture,
            "major_components": list(self.major_components),
            "training_approach": self.training_approach,
            "hyperparameters": list(self.hyperparameters),
            "baselines": list(self.baselines),
            "structured_output": _json_copy(self.structured_output),
            "evidence": _json_copy(list(self.evidence)),
            "citations": _json_copy(list(self.citations)),
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "citation_count": self.citation_count,
            "diagnostics": _json_copy(self.diagnostics),
        }


def _read(value: Any, name: str, default: Any = None) -> Any:
    """Read a value from either a mapping or a normal Python object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_string(*values: Any) -> Optional[str]:
    for value in values:
        normalized = _optional_string(value)
        if normalized is not None:
            return normalized
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return ()

    result: list[str] = []
    for item in value:
        normalized = _optional_string(item)
        if normalized is not None:
            result.append(normalized)
    return tuple(result)


def _json_copy(value: Any) -> Any:
    """Copy only JSON-compatible values; reject arbitrary Python objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelInputError(
                "Model-analysis data contains a non-finite number."
            )
        return value

    if isinstance(value, Mapping):
        return {str(k): _json_copy(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]

    raise ModelInputError(
        "Model-analysis data contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_copy(value)
    if not isinstance(copied, dict):
        raise ModelInputError("Expected a JSON object.")
    return copied


def _normalize_evidence(
    evidence: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Preserve authoritative evidence exactly enough for provenance.

    The project schema requires:
        evidence_number, chunk_id, document_id, paper_id, page, section

    Additional upstream metadata is retained when present.
    """
    if evidence is None:
        return ()

    if not isinstance(evidence, Sequence) or isinstance(
        evidence, (str, bytes, bytearray)
    ):
        raise ModelEvidenceError(
            "Evidence must be a sequence of evidence objects."
        )

    normalized: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ModelEvidenceError(
                f"Evidence item {index} must be an object."
            )

        copied = _copy_mapping(item)

        evidence_id = _first_string(
            copied.get("evidence_id"),
            copied.get("chunk_id"),
        )
        if evidence_id is not None:
            if evidence_id in seen_ids:
                raise ModelEvidenceError(
                    f"Duplicate evidence identifier: {evidence_id!r}."
                )
            seen_ids.add(evidence_id)

        normalized.append(copied)

    return tuple(normalized)


def _extract_structured_output(
    *,
    parsed_response: Any,
    pipeline_response: Any,
) -> dict[str, Any]:
    """
    Extract the parser-owned model payload.

    This function never performs inference and never calls an LLM.
    """
    candidate = parsed_response

    if candidate is None:
        candidate = _read(
            pipeline_response,
            "structured_output",
            None,
        )

    if candidate is None:
        raise ModelInputError(
            "A parsed model response is required. "
            "model.py does not perform LLM inference."
        )

    if isinstance(candidate, Mapping):
        data = candidate
    else:
        data = _read(candidate, "data", None)
        if data is None:
            data = _read(candidate, "structured_output", None)

        if not isinstance(data, Mapping):
            raise ModelInputError(
                "Parsed model response must be a mapping or expose "
                "a mapping through data/structured_output."
            )

    result = _copy_mapping(data)

    if not result:
        raise ModelInputError("Parsed model response cannot be empty.")

    return result


def _normalize_schema_fields(
    structured_output: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate the exact existing `research_paper_model` schema.

    Required fields:
        model_name
        architecture
        major_components
        training_approach
        hyperparameters
        baselines
        grounded
        evidence

    No extra top-level scientific fields are fabricated.
    """
    result = dict(structured_output)

    required_scalar_fields = (
        "model_name",
        "architecture",
        "training_approach",
    )

    for field_name in required_scalar_fields:
        if field_name not in result:
            raise ModelInputError(
                f"Missing required model field: {field_name!r}."
            )
        if not isinstance(result[field_name], str):
            raise ModelInputError(
                f"'{field_name}' must be a string."
            )

    required_array_fields = (
        "major_components",
        "hyperparameters",
        "baselines",
    )

    for field_name in required_array_fields:
        if field_name not in result:
            raise ModelInputError(
                f"Missing required model field: {field_name!r}."
            )

        value = result[field_name]
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ModelInputError(
                f"'{field_name}' must be an array of strings."
            )

        if any(not isinstance(item, str) for item in value):
            raise ModelInputError(
                f"'{field_name}' must contain only strings."
            )

        result[field_name] = [
            item.strip()
            for item in value
            if item.strip()
        ]

    if "grounded" not in result:
        raise ModelInputError(
            "Parsed model response must explicitly provide 'grounded'."
        )

    if not isinstance(result["grounded"], bool):
        raise ModelInputError("'grounded' must be a boolean.")

    if "evidence" not in result:
        raise ModelInputError(
            "Parsed model response must provide 'evidence'."
        )

    evidence_refs = result["evidence"]
    if not isinstance(evidence_refs, Sequence) or isinstance(
        evidence_refs, (str, bytes, bytearray)
    ):
        raise ModelInputError("'evidence' must be an array.")

    normalized_refs: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_refs):
        if not isinstance(item, Mapping):
            raise ModelInputError(
                f"Evidence reference {index} must be an object."
            )
        normalized_refs.append(_copy_mapping(item))

    result["evidence"] = normalized_refs

    return result


def _extract_authoritative_evidence(
    *,
    explicit_evidence: Any,
    parsed_response: Any,
    pipeline_response: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Prefer pipeline/context evidence over LLM-provided references.

    LLM evidence references are claims about provenance; the authoritative
    evidence object should come from the upstream RAG/context layer when
    available.
    """
    if explicit_evidence is not None:
        return _normalize_evidence(explicit_evidence)

    pipeline_evidence = _read(
        pipeline_response,
        "evidence",
        None,
    )
    if pipeline_evidence is not None:
        return _normalize_evidence(pipeline_evidence)

    parsed_evidence = _read(
        parsed_response,
        "evidence",
        None,
    )
    if parsed_evidence is not None:
        return _normalize_evidence(parsed_evidence)

    return ()


def _extract_identity(
    *,
    paper: Any,
    structured_output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    scope: str,
    pipeline_response: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve existing identity metadata without creating identifiers."""
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
        value.strip()
        for item in evidence
        for value in [item.get("paper_id")]
        if isinstance(value, str) and value.strip()
    }

    document_ids = {
        value.strip()
        for item in evidence
        for value in [item.get("document_id")]
        if isinstance(value, str) and value.strip()
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
    Prevent cross-paper evidence contamination in normal single-paper mode.

    Explicit comparison scope can contain multiple identities, but comparison
    itself is not performed by this module.
    """
    if scope == "comparison":
        return

    identities: set[tuple[str, str]] = set()

    for item in evidence:
        item_paper = _optional_string(item.get("paper_id"))
        item_document = _optional_string(item.get("document_id"))

        if item_paper or item_document:
            identities.add(
                (
                    item_paper or "",
                    item_document or "",
                )
            )

    if len(identities) > 1:
        raise ModelEvidenceError(
            "Single-paper model analysis received evidence from multiple "
            f"identities: {sorted(identities)!r}."
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
            raise ModelEvidenceError(
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
            raise ModelEvidenceError(
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

    pipeline_value = _read(
        pipeline_response,
        "grounded",
        None,
    )
    if isinstance(pipeline_value, bool):
        return pipeline_value

    # Fail closed: evidence presence alone is not equivalent to grounding.
    return False


def _count_citations(
    evidence: Sequence[Mapping[str, Any]],
) -> int:
    return sum(
        1
        for item in evidence
        if item.get("citation") is not None
    )


def _source_count(
    evidence: Sequence[Mapping[str, Any]],
) -> int:
    identities: set[str] = set()

    for item in evidence:
        for field_name in ("paper_id", "document_id"):
            value = item.get(field_name)
            if isinstance(value, str) and value.strip():
                identities.add(value.strip())

    return len(identities)


def _contains_meaningful_model_information(
    structured_output: Mapping[str, Any],
) -> bool:
    """
    Determine whether the parser found model information.

    `baselines` and `major_components` can be useful evidence even when
    model_name is unavailable. However, explicit Not Found strings are not
    treated as extracted facts.
    """
    scalar_fields = (
        "model_name",
        "architecture",
        "training_approach",
    )

    for field_name in scalar_fields:
        value = _optional_string(structured_output.get(field_name))
        if value is not None and value.lower() != NOT_FOUND.lower():
            return True

    for field_name in (
        "major_components",
        "hyperparameters",
        "baselines",
    ):
        values = _string_tuple(structured_output.get(field_name))
        if any(
            value.lower() != NOT_FOUND.lower()
            for value in values
        ):
            return True

    return False


def _validate_llm_evidence_references(
    *,
    structured_output: Mapping[str, Any],
    authoritative_evidence: Sequence[Mapping[str, Any]],
) -> None:
    """
    Validate that parser evidence references do not point to a different
    document when both sides expose an identity.

    Full semantic claim verification remains the responsibility of
    validation/evidence.py and validation/validator.py.
    """
    references = structured_output.get("evidence", [])
    if not authoritative_evidence or not references:
        return

    authoritative_chunk_ids = {
        value.strip()
        for item in authoritative_evidence
        for value in [item.get("chunk_id")]
        if isinstance(value, str) and value.strip()
    }

    authoritative_document_ids = {
        value.strip()
        for item in authoritative_evidence
        for value in [item.get("document_id")]
        if isinstance(value, str) and value.strip()
    }

    authoritative_paper_ids = {
        value.strip()
        for item in authoritative_evidence
        for value in [item.get("paper_id")]
        if isinstance(value, str) and value.strip()
    }

    for reference in references:
        ref_chunk = _optional_string(reference.get("chunk_id"))
        ref_document = _optional_string(reference.get("document_id"))
        ref_paper = _optional_string(reference.get("paper_id"))

        if (
            ref_chunk is not None
            and authoritative_chunk_ids
            and ref_chunk not in authoritative_chunk_ids
        ):
            raise ModelEvidenceError(
                f"Parser evidence reference points to unavailable chunk "
                f"{ref_chunk!r}."
            )

        if (
            ref_document is not None
            and authoritative_document_ids
            and ref_document not in authoritative_document_ids
        ):
            raise ModelEvidenceError(
                f"Parser evidence reference points to document "
                f"{ref_document!r}, outside the supplied evidence."
            )

        if (
            ref_paper is not None
            and authoritative_paper_ids
            and ref_paper not in authoritative_paper_ids
        ):
            raise ModelEvidenceError(
                f"Parser evidence reference points to paper "
                f"{ref_paper!r}, outside the supplied evidence."
            )


class ModelAnalyzer:
    """
    Deterministic post-parser model-analysis boundary.

    The analyzer consumes:
        * parsed LLM model output
        * already-prepared evidence
        * document/paper metadata

    It does not perform model extraction by itself. Model extraction is
    performed by the existing RAG -> prompt -> LLM -> parser path.
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

        self.require_evidence_for_success = (
            require_evidence_for_success
        )
        self.require_grounded_flag = require_grounded_flag
        self.allow_insufficient_evidence = (
            allow_insufficient_evidence
        )

    def analyze(
        self,
        paper: Any = None,
        *,
        evidence: Any = None,
        parsed_response: Any = None,
        pipeline_response: Any = None,
        scope: Optional[str] = None,
        task_type: str = TASK_TYPE,
    ) -> ModelAnalysisResult:
        """
        Normalize one parser-produced model result.

        No scientific information is generated here.
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
            raise ModelInputError(
                "Uploaded-paper model analysis requires document_id."
            )

        _validate_identity_isolation(
            evidence=authoritative_evidence,
            paper_id=paper_id,
            document_id=document_id,
            scope=normalized_scope,
        )

        _validate_llm_evidence_references(
            structured_output=structured,
            authoritative_evidence=authoritative_evidence,
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

        citation_count = _count_citations(authoritative_evidence)
        source_count = _source_count(authoritative_evidence)

        diagnostics = {
            "evidence_count": len(authoritative_evidence),
            "source_count": source_count,
            "citation_count": citation_count,
            "model_present": (
                structured.get("model_name") != NOT_FOUND
            ),
            "baseline_count": len(
                _string_tuple(structured.get("baselines"))
            ),
            "processing_seconds": time.perf_counter() - started,
        }

        citations = tuple(
            item
            for item in authoritative_evidence
            if item.get("citation") is not None
        )

        logger.info(
            "Model analysis completed: task=%s scope=%s paper_id=%s "
            "document_id=%s status=%s evidence_count=%d",
            normalized_task,
            normalized_scope,
            paper_id,
            document_id,
            status,
            len(authoritative_evidence),
        )

        return ModelAnalysisResult(
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
            raise ModelTaskError("task_type must be a string.")

        normalized = task_type.strip().lower()
        if normalized != TASK_TYPE:
            raise ModelTaskError(
                "ModelAnalyzer only handles task_type='model'; "
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
            candidate = _read(
                pipeline_response,
                "scope",
                None,
            )

        if candidate is None:
            candidate = "research"

        if not isinstance(candidate, str):
            raise ModelInputError("scope must be a string.")

        normalized = candidate.strip().lower()

        if normalized not in SUPPORTED_SCOPES:
            raise ModelInputError(
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
            if (
                pipeline_status.strip().lower()
                == INSUFFICIENT_EVIDENCE
            ):
                return INSUFFICIENT_EVIDENCE, False

        if self.require_evidence_for_success and not evidence:
            if self.allow_insufficient_evidence:
                return INSUFFICIENT_EVIDENCE, False

            raise ModelEvidenceError(
                "No authoritative model evidence is available."
            )

        if self.require_grounded_flag and not grounded:
            return VALIDATION_ERROR, False

        if not _contains_meaningful_model_information(
            structured_output
        ):
            return INSUFFICIENT_EVIDENCE, False

        return SUCCESS, True


class ResearchModelAnalyzer:
    """
    Batch facade for independent Top-N research-paper analysis.

    Each paper is processed independently. Evidence is never pooled.
    """

    def __init__(
        self,
        analyzer: Optional[ModelAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or ModelAnalyzer()

    def analyze_papers(
        self,
        papers: Sequence[Any],
        *,
        responses: Sequence[Any],
        evidences: Optional[Sequence[Any]] = None,
    ) -> tuple[ModelAnalysisResult, ...]:
        if not isinstance(papers, Sequence) or isinstance(
            papers, (str, bytes, bytearray)
        ):
            raise ModelInputError("papers must be a sequence.")

        if not isinstance(responses, Sequence) or isinstance(
            responses, (str, bytes, bytearray)
        ):
            raise ModelInputError("responses must be a sequence.")

        if len(papers) != len(responses):
            raise ModelInputError(
                "papers and responses must have equal lengths."
            )

        if evidences is None:
            evidence_values: Sequence[Any] = [None] * len(papers)
        else:
            if not isinstance(evidences, Sequence) or isinstance(
                evidences, (str, bytes, bytearray)
            ):
                raise ModelInputError(
                    "evidences must be a sequence when supplied."
                )

            if len(evidences) != len(papers):
                raise ModelInputError(
                    "papers, responses, and evidences must have equal "
                    "lengths."
                )

            evidence_values = evidences

        results: list[ModelAnalysisResult] = []

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
            except ModelAnalysisError as exc:
                raise ModelAnalysisError(
                    f"Model analysis failed for paper index {index}."
                ) from exc

            results.append(result)

        return tuple(results)


_DEFAULT_ANALYZER = ModelAnalyzer()


def analyze_model(
    paper: Any = None,
    *,
    evidence: Any = None,
    parsed_response: Any = None,
    pipeline_response: Any = None,
    scope: Optional[str] = None,
) -> ModelAnalysisResult:
    """Functional convenience API."""
    return _DEFAULT_ANALYZER.analyze(
        paper,
        evidence=evidence,
        parsed_response=parsed_response,
        pipeline_response=pipeline_response,
        scope=scope,
    )


# Short alias for existing callers that prefer a generic analyzer function.
analyze = analyze_model


def run_self_test() -> None:
    """
    Deterministic tests for the model-analysis boundary.

    No internet, LLM API, API key, GPU, FAISS, embeddings, or PDF parser is
    required.
    """
    analyzer = ModelAnalyzer()

    # ---------------------------------------------------------------
    # 1. Single primary model + provenance.
    # ---------------------------------------------------------------
    evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "chunk_006",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "page": 6,
            "section": "Methods",
            "citation": "paper_001:p6",
            "text": (
                "We use a ResNet-50 backbone followed by a custom "
                "attention module and classification head. The backbone "
                "was initialized with ImageNet-pretrained weights."
            ),
        },
    )

    parsed = {
        "model_name": "ResNet-50",
        "architecture": "CNN backbone with a custom classification head",
        "major_components": [
            "ResNet-50 backbone",
            "custom attention module",
            "classification head",
        ],
        "training_approach": "ImageNet-pretrained backbone initialization",
        "hyperparameters": [],
        "baselines": [],
        "grounded": True,
        "evidence": [
            {
                "evidence_number": 1,
                "chunk_id": "chunk_006",
                "document_id": "paper_001",
                "paper_id": "paper_001",
                "page": 6,
                "section": "Methods",
            }
        ],
    }

    result = analyzer.analyze(
        {
            "paper_id": "paper_001",
            "document_id": "paper_001",
            "title": "Example Model Paper",
        },
        evidence=evidence,
        parsed_response=parsed,
        scope="research",
    )

    assert result.status == SUCCESS
    assert result.success is True
    assert result.grounded is True
    assert result.model_name == "ResNet-50"
    assert result.architecture == (
        "CNN backbone with a custom classification head"
    )
    assert result.major_components == (
        "ResNet-50 backbone",
        "custom attention module",
        "classification head",
    )
    assert result.baselines == ()
    assert result.evidence[0]["page"] == 6
    assert result.evidence[0]["section"] == "Methods"
    assert result.evidence[0]["citation"] == "paper_001:p6"

    # ---------------------------------------------------------------
    # 2. Multiple models with proposed/primary + baselines.
    # ---------------------------------------------------------------
    parsed_multi = dict(parsed)
    parsed_multi["model_name"] = "Model-X"
    parsed_multi["baselines"] = [
        "ResNet-50",
        "EfficientNet-B0",
    ]

    multi = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=parsed_multi,
    )

    assert multi.model_name == "Model-X"
    assert multi.proposed_or_primary_model == "Model-X"
    assert multi.baselines == (
        "ResNet-50",
        "EfficientNet-B0",
    )

    # ---------------------------------------------------------------
    # 3. Exact variant preservation.
    # ---------------------------------------------------------------
    variant = dict(parsed)
    variant["model_name"] = "YOLOv8m"

    variant_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=variant,
    )

    assert variant_result.model_name == "YOLOv8m"
    assert variant_result.model_name != "YOLOv8"

    # ---------------------------------------------------------------
    # 4. No-inference test.
    # ---------------------------------------------------------------
    no_inference = dict(parsed)
    no_inference["model_name"] = "transformer-based classifier"
    no_inference["architecture"] = "Not found in the provided evidence."
    no_inference["major_components"] = []
    no_inference["training_approach"] = (
        "Not found in the provided evidence."
    )
    no_inference["hyperparameters"] = []
    no_inference["baselines"] = []
    no_inference["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "abstract_001",
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "page": 1,
            "section": "Abstract",
        }
    ]

    no_inference_result = analyzer.analyze(
        {"paper_id": "paper_002"},
        evidence=(
            {
                "evidence_number": 1,
                "chunk_id": "abstract_001",
                "document_id": "paper_002",
                "paper_id": "paper_002",
                "page": 1,
                "section": "Abstract",
            },
        ),
        parsed_response=no_inference,
    )

    assert no_inference_result.model_name == (
        "transformer-based classifier"
    )
    assert no_inference_result.architecture is not None
    assert "BERT" not in no_inference_result.model_name
    assert "12 layers" not in no_inference_result.model_name
    assert "AdamW" not in no_inference_result.model_name

    # ---------------------------------------------------------------
    # 5. Missing model evidence.
    # ---------------------------------------------------------------
    missing = {
        "model_name": NOT_FOUND,
        "architecture": NOT_FOUND,
        "major_components": [],
        "training_approach": NOT_FOUND,
        "hyperparameters": [],
        "baselines": [],
        "grounded": True,
        "evidence": [],
    }

    missing_result = analyzer.analyze(
        {"paper_id": "paper_003"},
        evidence=(),
        parsed_response=missing,
    )

    assert missing_result.status == INSUFFICIENT_EVIDENCE
    assert missing_result.success is False

    # ---------------------------------------------------------------
    # 6. Abstract-only evidence must not create configuration.
    # ---------------------------------------------------------------
    abstract = dict(no_inference)
    abstract["model_name"] = "Transformer"
    abstract["architecture"] = "Transformer-based classifier"
    abstract["hyperparameters"] = []
    abstract["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "abstract_004",
            "document_id": "paper_004",
            "paper_id": "paper_004",
            "page": 1,
            "section": "Abstract",
        }
    ]

    abstract_result = analyzer.analyze(
        {"paper_id": "paper_004"},
        evidence=(
            {
                "evidence_number": 1,
                "chunk_id": "abstract_004",
                "document_id": "paper_004",
                "paper_id": "paper_004",
                "page": 1,
                "section": "Abstract",
            },
        ),
        parsed_response=abstract,
    )

    assert abstract_result.hyperparameters == ()
    assert "BERT" not in abstract_result.model_name

    # ---------------------------------------------------------------
    # 7. Numeric configuration preservation.
    # ---------------------------------------------------------------
    numeric = dict(parsed)
    numeric["hyperparameters"] = [
        "12 layers",
        "768 hidden dimensions",
        "8 attention heads",
        "224×224 input",
        "25M parameters",
    ]

    numeric_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=numeric,
    )

    assert numeric_result.hyperparameters == (
        "12 layers",
        "768 hidden dimensions",
        "8 attention heads",
        "224×224 input",
        "25M parameters",
    )

    # ---------------------------------------------------------------
    # 8. Fine-tuning/pretraining wording preservation.
    # ---------------------------------------------------------------
    training = dict(parsed)
    training["model_name"] = "BERT-base"
    training["training_approach"] = (
        "The pretrained BERT model was fine-tuned on the target dataset."
    )

    training_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=training,
    )

    assert "fine-tuned" in training_result.training_approach
    assert "pretrained" in training_result.training_approach

    # ---------------------------------------------------------------
    # 9. Invalid structured input.
    # ---------------------------------------------------------------
    invalid = dict(parsed)
    invalid["major_components"] = "not-a-list"

    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=evidence,
            parsed_response=invalid,
        )
    except ModelInputError:
        pass
    else:
        raise AssertionError(
            "Invalid structured model input was accepted."
        )

    # ---------------------------------------------------------------
    # 10. grounded=False fails closed.
    # ---------------------------------------------------------------
    ungrounded = dict(parsed)
    ungrounded["grounded"] = False

    ungrounded_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=evidence,
        parsed_response=ungrounded,
    )

    assert ungrounded_result.status == VALIDATION_ERROR
    assert ungrounded_result.success is False

    # ---------------------------------------------------------------
    # 11. Parser evidence reference outside authoritative context.
    # ---------------------------------------------------------------
    wrong_reference = dict(parsed)
    wrong_reference["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "chunk_DOES_NOT_EXIST",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "page": 6,
            "section": "Methods",
        }
    ]

    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=evidence,
            parsed_response=wrong_reference,
        )
    except ModelEvidenceError:
        pass
    else:
        raise AssertionError(
            "Unavailable parser evidence reference was accepted."
        )

    # ---------------------------------------------------------------
    # 12. Multi-paper isolation.
    # ---------------------------------------------------------------
    paper_a_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "a1",
            "document_id": "paper_A",
            "paper_id": "paper_A",
            "page": 2,
            "section": "Methods",
        },
    )
    paper_b_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "b1",
            "document_id": "paper_B",
            "paper_id": "paper_B",
            "page": 3,
            "section": "Methods",
        },
    )

    parsed_a = dict(parsed)
    parsed_a["model_name"] = "BERT"
    parsed_a["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "a1",
            "document_id": "paper_A",
            "paper_id": "paper_A",
            "page": 2,
            "section": "Methods",
        }
    ]

    parsed_b = dict(parsed)
    parsed_b["model_name"] = "LSTM"
    parsed_b["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "b1",
            "document_id": "paper_B",
            "paper_id": "paper_B",
            "page": 3,
            "section": "Methods",
        }
    ]

    batch = ResearchModelAnalyzer(analyzer).analyze_papers(
        [
            {"paper_id": "paper_A"},
            {"paper_id": "paper_B"},
        ],
        responses=[parsed_a, parsed_b],
        evidences=[paper_a_evidence, paper_b_evidence],
    )

    assert batch[0].model_name == "BERT"
    assert batch[1].model_name == "LSTM"
    assert batch[0].paper_id == "paper_A"
    assert batch[1].paper_id == "paper_B"

    # ---------------------------------------------------------------
    # 13. Cross-paper contamination is rejected.
    # ---------------------------------------------------------------
    try:
        analyzer.analyze(
            {"paper_id": "paper_A"},
            evidence=paper_a_evidence + paper_b_evidence,
            parsed_response=parsed_a,
        )
    except ModelEvidenceError:
        pass
    else:
        raise AssertionError(
            "Cross-paper model evidence was silently accepted."
        )

    # ---------------------------------------------------------------
    # 14. Uploaded-paper identity is mandatory.
    # ---------------------------------------------------------------
    uploaded_evidence_without_id = (
        {
            "evidence_number": 1,
            "chunk_id": "upload_chunk",
            "document_id": None,
            "paper_id": None,
            "page": 2,
            "section": "Methods",
        },
    )

    try:
        analyzer.analyze(
            {},
            evidence=uploaded_evidence_without_id,
            parsed_response=parsed,
            scope="uploaded",
        )
    except ModelInputError:
        pass
    else:
        raise AssertionError(
            "Uploaded scope accepted missing document identity."
        )

    # ---------------------------------------------------------------
    # 15. Comparison scope permits multiple identities but does not
    #     compare models.
    # ---------------------------------------------------------------
    comparison = analyzer.analyze(
        {},
        evidence=paper_a_evidence + paper_b_evidence,
        parsed_response=parsed_a,
        scope="comparison",
    )

    assert comparison.scope == "comparison"
    assert comparison.model_name == "BERT"

    # ---------------------------------------------------------------
    # 16. Determinism.
    # ---------------------------------------------------------------
    repeat = analyzer.analyze(
        {
            "paper_id": "paper_001",
            "document_id": "paper_001",
            "title": "Example Model Paper",
        },
        evidence=evidence,
        parsed_response=parsed,
    )

    first = result.to_dict()
    second = repeat.to_dict()
    first["diagnostics"] = {}
    second["diagnostics"] = {}

    assert first == second

    # ---------------------------------------------------------------
    # 17. Runtime dependency boundary.
    # ---------------------------------------------------------------
    # This module does not import retrieval, embedding, PDF, LLM-provider,
    # or frontend libraries. The test itself avoids importing those modules.
    forbidden_runtime_names = {
        "faiss",
        "torch",
        "transformers",
        "sentence_transformers",
        "openai",
        "anthropic",
        "fitz",
        "pymupdf",
        "streamlit",
    }
    assert forbidden_runtime_names.isdisjoint(globals())
    assert "subprocess" not in globals()
    assert "eval" not in globals()
    assert "exec" not in globals()

    print("ModelAnalyzer self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()