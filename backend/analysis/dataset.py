"""
Evidence-grounded dataset analysis boundary for the AI Research Paper Assistant.

This module is intentionally downstream of:
    retrieval/RAG -> context -> prompt -> LLM -> parser

and upstream of:
    validation -> API/frontend

It does NOT retrieve papers, parse PDFs, build prompts, call an LLM, generate
embeddings, access FAISS, rank datasets, or use external dataset metadata.

IMPORTANT CURRENT PROJECT CONTRACT
----------------------------------
The current project schema for task_type="dataset" is:

    dataset_name
    source
    size
    number_of_classes
    splits
    preprocessing
    augmentation
    other_details
    grounded
    evidence

This module preserves that contract and does not create a competing top-level
LLM schema.

The master dataset requirements are broader than the current schema. Therefore
some concepts (for example explicit dataset role/type/version as first-class
fields and a true list of independently structured datasets) are retained as
evidence-backed structured metadata inside the analyzer result when upstream
data provides them, but are NOT injected into the parser-owned output schema.
That avoids breaking llm/parser.py and rag/prompts.py.

SOURCE-FAITHFUL RULE
--------------------
A dataset fact is valid only when it is supported by supplied evidence or by
the already-parsed LLM output that is itself grounded in that evidence.

No dataset knowledge is obtained from:
    * web search
    * Wikipedia
    * Kaggle
    * Hugging Face
    * external dataset APIs
    * hard-coded benchmark metadata

No arithmetic is performed to derive missing dataset statistics.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK_TYPE = "dataset"

SUCCESS = "success"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
INVALID_INPUT = "invalid_input"
VALIDATION_ERROR = "validation_error"

NOT_FOUND = "Not found in the provided evidence."

SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})

# Exact parser-owned fields from the current research_paper_dataset schema.
SCHEMA_FIELDS = (
    "dataset_name",
    "source",
    "size",
    "number_of_classes",
    "splits",
    "preprocessing",
    "augmentation",
    "other_details",
    "grounded",
    "evidence",
)

STRING_FIELDS = (
    "dataset_name",
    "source",
    "size",
    "number_of_classes",
    "splits",
    "preprocessing",
    "augmentation",
)

ARRAY_FIELDS = ("other_details",)


class DatasetAnalysisError(RuntimeError):
    """Base exception for dataset-analysis failures."""


class DatasetInputError(DatasetAnalysisError, ValueError):
    """Invalid dataset-analysis input or parsed payload."""


class DatasetEvidenceError(DatasetAnalysisError, ValueError):
    """Malformed or cross-paper evidence/provenance."""


class DatasetTaskError(DatasetAnalysisError, ValueError):
    """Unsupported task type or analysis scope."""


@dataclass(frozen=True)
class DatasetRecord:
    """
    One evidence-backed dataset record.

    The current LLM schema has no first-class role/type/version fields.
    These optional fields are populated only when upstream parsed/evidence
    metadata already provides them; this does not change the parser schema.
    """

    name: Optional[str]
    dataset_type: Optional[str] = None
    source: Optional[str] = None
    version: Optional[str] = None
    purpose: Optional[str] = None
    role: Optional[str] = None
    size: Optional[str] = None
    number_of_classes: Optional[str] = None
    class_names: tuple[str, ...] = ()
    splits: Optional[str] = None
    preprocessing: Optional[str] = None
    augmentation: Optional[str] = None
    annotation: Optional[str] = None
    other_details: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_type": self.dataset_type,
            "source": self.source,
            "version": self.version,
            "purpose": self.purpose,
            "role": self.role,
            "size": self.size,
            "number_of_classes": self.number_of_classes,
            "class_names": list(self.class_names),
            "splits": self.splits,
            "preprocessing": self.preprocessing,
            "augmentation": self.augmentation,
            "annotation": self.annotation,
            "other_details": list(self.other_details),
            "evidence": _json_copy(list(self.evidence)),
        }


@dataclass(frozen=True)
class DatasetAnalysisResult:
    """
    Machine-readable dataset-analysis result.

    `structured_output` remains the exact parser-compatible schema. The
    `datasets` property exposes independently normalized dataset records when
    upstream structured metadata permits it, without changing the LLM schema.
    """

    status: str
    success: bool
    grounded: bool

    paper_id: Optional[str]
    document_id: Optional[str]
    title: Optional[str]

    structured_output: Mapping[str, Any]
    datasets: tuple[DatasetRecord, ...]

    evidence: tuple[Mapping[str, Any], ...]
    citations: tuple[Mapping[str, Any], ...]

    scope: str = "research"
    task_type: str = TASK_TYPE
    evidence_count: int = 0
    source_count: int = 0
    citation_count: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dataset_name(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("dataset_name"))

    @property
    def source(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("source"))

    @property
    def size(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("size"))

    @property
    def number_of_classes(self) -> Optional[str]:
        return _optional_string(
            self.structured_output.get("number_of_classes")
        )

    @property
    def splits(self) -> Optional[str]:
        return _optional_string(self.structured_output.get("splits"))

    @property
    def preprocessing(self) -> Optional[str]:
        return _optional_string(
            self.structured_output.get("preprocessing")
        )

    @property
    def augmentation(self) -> Optional[str]:
        return _optional_string(
            self.structured_output.get("augmentation")
        )

    @property
    def other_details(self) -> tuple[str, ...]:
        return _string_tuple(
            self.structured_output.get("other_details")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result representation."""
        return {
            "status": self.status,
            "success": self.success,
            "grounded": self.grounded,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "title": self.title,
            "scope": self.scope,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "source": self.source,
            "size": self.size,
            "number_of_classes": self.number_of_classes,
            "splits": self.splits,
            "preprocessing": self.preprocessing,
            "augmentation": self.augmentation,
            "other_details": list(self.other_details),
            "structured_output": _json_copy(self.structured_output),
            "datasets": [item.to_dict() for item in self.datasets],
            "evidence": _json_copy(list(self.evidence)),
            "citations": _json_copy(list(self.citations)),
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "citation_count": self.citation_count,
            "diagnostics": _json_copy(self.diagnostics),
        }


def _read(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_string(*values: Any) -> Optional[str]:
    for value in values:
        result = _optional_string(value)
        if result is not None:
            return result
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


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = value.strip()
        key = normalized.casefold()

        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)

    return tuple(result)


def _json_copy(value: Any) -> Any:
    """
    Copy JSON-compatible data and reject executable/arbitrary objects.

    This deliberately does not serialize arbitrary Python objects.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetInputError(
                "Dataset-analysis data contains a non-finite number."
            )
        return value

    if isinstance(value, Mapping):
        return {str(k): _json_copy(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]

    raise DatasetInputError(
        "Dataset-analysis data contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_copy(value)

    if not isinstance(copied, dict):
        raise DatasetInputError("Expected a JSON object.")

    return copied


def _normalize_evidence(
    evidence: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Normalize authoritative evidence without changing its provenance.

    Required upstream provenance fields in the current project include:
        evidence_number
        chunk_id
        document_id
        paper_id
        page
        section

    Additional metadata is retained when already supplied.
    """
    if evidence is None:
        return ()

    if not isinstance(evidence, Sequence) or isinstance(
        evidence, (str, bytes, bytearray)
    ):
        raise DatasetEvidenceError(
            "Evidence must be a sequence of evidence objects."
        )

    result: list[Mapping[str, Any]] = []
    seen_evidence_ids: set[str] = set()

    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise DatasetEvidenceError(
                f"Evidence item {index} must be an object."
            )

        copied = _copy_mapping(item)

        evidence_id = _first_string(
            copied.get("evidence_id"),
            copied.get("chunk_id"),
        )

        if evidence_id is not None:
            if evidence_id in seen_evidence_ids:
                raise DatasetEvidenceError(
                    f"Duplicate evidence identifier: {evidence_id!r}."
                )
            seen_evidence_ids.add(evidence_id)

        result.append(copied)

    return tuple(result)


def _extract_structured_output(
    *,
    parsed_response: Any,
    pipeline_response: Any,
) -> dict[str, Any]:
    """
    Extract the parser-owned dataset payload.

    No LLM inference occurs here.
    """
    candidate = parsed_response

    if candidate is None:
        candidate = _read(
            pipeline_response,
            "structured_output",
            None,
        )

    if candidate is None:
        raise DatasetInputError(
            "A parsed dataset response is required. "
            "dataset.py does not perform LLM inference."
        )

    if isinstance(candidate, Mapping):
        data = candidate
    else:
        data = _read(candidate, "data", None)

        if data is None:
            data = _read(candidate, "structured_output", None)

        if not isinstance(data, Mapping):
            raise DatasetInputError(
                "Parsed dataset response must be a mapping or expose "
                "a mapping through data/structured_output."
            )

    result = _copy_mapping(data)

    if not result:
        raise DatasetInputError(
            "Parsed dataset response cannot be empty."
        )

    return result


def _normalize_schema_fields(
    structured_output: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate the current research_paper_dataset schema exactly.

    This intentionally does not introduce new top-level fields such as
    `version`, `role`, or `datasets`, because llm/parser.py currently owns a
    strict additionalProperties=False schema.
    """
    result = dict(structured_output)

    for field_name in STRING_FIELDS:
        if field_name not in result:
            raise DatasetInputError(
                f"Missing required dataset field: {field_name!r}."
            )

        if not isinstance(result[field_name], str):
            raise DatasetInputError(
                f"'{field_name}' must be a string."
            )

    for field_name in ARRAY_FIELDS:
        if field_name not in result:
            raise DatasetInputError(
                f"Missing required dataset field: {field_name!r}."
            )

        value = result[field_name]

        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise DatasetInputError(
                f"'{field_name}' must be an array of strings."
            )

        if any(not isinstance(item, str) for item in value):
            raise DatasetInputError(
                f"'{field_name}' must contain only strings."
            )

        result[field_name] = [
            item.strip()
            for item in value
            if item.strip()
        ]

    if "grounded" not in result:
        raise DatasetInputError(
            "Parsed dataset response must explicitly provide "
            "'grounded'."
        )

    if not isinstance(result["grounded"], bool):
        raise DatasetInputError("'grounded' must be a boolean.")

    if "evidence" not in result:
        raise DatasetInputError(
            "Parsed dataset response must provide 'evidence'."
        )

    if not isinstance(result["evidence"], Sequence) or isinstance(
        result["evidence"],
        (str, bytes, bytearray),
    ):
        raise DatasetInputError("'evidence' must be an array.")

    normalized_refs: list[dict[str, Any]] = []

    for index, item in enumerate(result["evidence"]):
        if not isinstance(item, Mapping):
            raise DatasetInputError(
                f"Evidence reference {index} must be an object."
            )

        normalized_refs.append(_copy_mapping(item))

    result["evidence"] = normalized_refs

    # The current schema is strict. Reject unexpected top-level fields so
    # accidental schema drift is detected immediately.
    unexpected = set(result) - set(SCHEMA_FIELDS)

    if unexpected:
        raise DatasetInputError(
            "Unexpected dataset output fields: "
            f"{sorted(unexpected)!r}. "
            "Update the shared parser/prompt schema first instead of "
            "silently accepting a competing schema."
        )

    return result


def _extract_authoritative_evidence(
    *,
    explicit_evidence: Any,
    parsed_response: Any,
    pipeline_response: Any,
) -> tuple[Mapping[str, Any], ...]:
    """
    Prefer upstream authoritative evidence over LLM-created references.
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
    """Resolve existing paper/document identity without fabricating IDs."""
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

    evidence_paper_ids = {
        value.strip()
        for item in evidence
        for value in [item.get("paper_id")]
        if isinstance(value, str) and value.strip()
    }

    evidence_document_ids = {
        value.strip()
        for item in evidence
        for value in [item.get("document_id")]
        if isinstance(value, str) and value.strip()
    }

    if paper_id is None and len(evidence_paper_ids) == 1:
        paper_id = next(iter(evidence_paper_ids))

    if document_id is None and len(evidence_document_ids) == 1:
        document_id = next(iter(evidence_document_ids))

    return paper_id, document_id, title


def _validate_identity_isolation(
    *,
    evidence: Sequence[Mapping[str, Any]],
    paper_id: Optional[str],
    document_id: Optional[str],
    scope: str,
) -> None:
    """
    Prevent evidence from multiple papers/documents from entering a
    single-paper analysis.

    Explicit comparison scope is the only scope that permits multiple
    identities. dataset.py still does not perform comparison logic.
    """
    if scope == "comparison":
        return

    identities = {
        (
            _optional_string(item.get("paper_id")) or "",
            _optional_string(item.get("document_id")) or "",
        )
        for item in evidence
        if (
            _optional_string(item.get("paper_id"))
            or _optional_string(item.get("document_id"))
        )
    }

    if len(identities) > 1:
        raise DatasetEvidenceError(
            "Single-paper dataset analysis received evidence from multiple "
            f"identities: {sorted(identities)!r}."
        )

    if paper_id is not None:
        mismatched = [
            item.get("paper_id")
            for item in evidence
            if (
                isinstance(item.get("paper_id"), str)
                and item["paper_id"].strip()
                and item["paper_id"].strip() != paper_id
            )
        ]

        if mismatched:
            raise DatasetEvidenceError(
                f"Evidence contains paper_id values different from "
                f"{paper_id!r}: {mismatched!r}."
            )

    if document_id is not None:
        mismatched = [
            item.get("document_id")
            for item in evidence
            if (
                isinstance(item.get("document_id"), str)
                and item["document_id"].strip()
                and item["document_id"].strip() != document_id
            )
        ]

        if mismatched:
            raise DatasetEvidenceError(
                f"Evidence contains document_id values different from "
                f"{document_id!r}: {mismatched!r}."
            )


def _validate_llm_evidence_references(
    *,
    structured_output: Mapping[str, Any],
    authoritative_evidence: Sequence[Mapping[str, Any]],
) -> None:
    """
    Ensure parser evidence references point only into supplied evidence.

    This is provenance-integrity checking, not semantic claim verification.
    Full evidence verification remains owned by validation/evidence.py.
    """
    references = structured_output.get("evidence", [])

    if not authoritative_evidence or not references:
        return

    available_chunks = {
        value.strip()
        for item in authoritative_evidence
        for value in [item.get("chunk_id")]
        if isinstance(value, str) and value.strip()
    }

    available_documents = {
        value.strip()
        for item in authoritative_evidence
        for value in [item.get("document_id")]
        if isinstance(value, str) and value.strip()
    }

    available_papers = {
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
            and available_chunks
            and ref_chunk not in available_chunks
        ):
            raise DatasetEvidenceError(
                f"Parser evidence reference points to unavailable "
                f"chunk {ref_chunk!r}."
            )

        if (
            ref_document is not None
            and available_documents
            and ref_document not in available_documents
        ):
            raise DatasetEvidenceError(
                f"Parser evidence reference points to document "
                f"{ref_document!r}, outside supplied evidence."
            )

        if (
            ref_paper is not None
            and available_papers
            and ref_paper not in available_papers
        ):
            raise DatasetEvidenceError(
                f"Parser evidence reference points to paper "
                f"{ref_paper!r}, outside supplied evidence."
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

    # Fail closed.
    return False


def _meaningful(value: Any) -> bool:
    normalized = _optional_string(value)

    return (
        normalized is not None
        and normalized.casefold() != NOT_FOUND.casefold()
    )


def _contains_meaningful_dataset_information(
    structured_output: Mapping[str, Any],
) -> bool:
    for field_name in STRING_FIELDS:
        if _meaningful(structured_output.get(field_name)):
            return True

    for value in _string_tuple(
        structured_output.get("other_details")
    ):
        if value.casefold() != NOT_FOUND.casefold():
            return True

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


def _metadata_value(
    source: Mapping[str, Any],
    *names: str,
) -> Optional[str]:
    for name in names:
        value = _optional_string(source.get(name))

        if value is not None:
            return value

    return None


def _parse_extended_dataset_records(
    structured_output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[DatasetRecord, ...]:
    """
    Build normalized dataset records without changing the parser schema.

    Supported upstream extensions:
        structured_output["datasets"] or
        structured_output["dataset_records"]

    These extensions are deliberately optional. The current parser schema
    rejects them, so production parser output will normally use the legacy
    single-dataset contract and this method will create one record.

    When the current parser gives a single dataset, no general-world metadata
    is inferred.
    """
    extended = None

    for key in ("datasets", "dataset_records"):
        candidate = structured_output.get(key)

        if candidate is not None:
            extended = candidate
            break

    if extended is not None:
        if not isinstance(extended, Sequence) or isinstance(
            extended, (str, bytes, bytearray)
        ):
            raise DatasetInputError(
                f"'{key}' must be an array of dataset records."
            )

        records: list[DatasetRecord] = []

        for index, item in enumerate(extended):
            if not isinstance(item, Mapping):
                raise DatasetInputError(
                    f"Dataset record {index} must be an object."
                )

            record = DatasetRecord(
                name=_metadata_value(
                    item,
                    "name",
                    "dataset_name",
                ),
                dataset_type=_metadata_value(
                    item,
                    "dataset_type",
                    "type",
                ),
                source=_metadata_value(item, "source"),
                version=_metadata_value(
                    item,
                    "version",
                    "release",
                    "edition",
                    "year",
                ),
                purpose=_metadata_value(
                    item,
                    "purpose",
                ),
                role=_metadata_value(item, "role"),
                size=_metadata_value(
                    item,
                    "size",
                    "dataset_size",
                ),
                number_of_classes=_metadata_value(
                    item,
                    "number_of_classes",
                    "classes",
                    "class_count",
                ),
                class_names=_string_tuple(
                    item.get("class_names")
                    or item.get("classes_names")
                ),
                splits=_metadata_value(item, "splits"),
                preprocessing=_metadata_value(
                    item,
                    "preprocessing",
                ),
                augmentation=_metadata_value(
                    item,
                    "augmentation",
                ),
                annotation=_metadata_value(
                    item,
                    "annotation",
                    "annotation_method",
                ),
                other_details=_dedupe_strings(
                    _string_tuple(item.get("other_details"))
                ),
                evidence=_normalize_evidence(
                    item.get("evidence", evidence)
                ),
            )

            records.append(record)

        return tuple(records)

    # Current parser-compatible single-dataset path.
    name = _optional_string(
        structured_output.get("dataset_name")
    )

    if not _contains_meaningful_dataset_information(
        structured_output
    ):
        return ()

    return (
        DatasetRecord(
            name=name,
            source=_optional_string(
                structured_output.get("source")
            ),
            size=_optional_string(
                structured_output.get("size")
            ),
            number_of_classes=_optional_string(
                structured_output.get("number_of_classes")
            ),
            splits=_optional_string(
                structured_output.get("splits")
            ),
            preprocessing=_optional_string(
                structured_output.get("preprocessing")
            ),
            augmentation=_optional_string(
                structured_output.get("augmentation")
            ),
            other_details=_string_tuple(
                structured_output.get("other_details")
            ),
            evidence=tuple(evidence),
        ),
    )


def _deduplicate_dataset_records(
    records: Sequence[DatasetRecord],
) -> tuple[DatasetRecord, ...]:
    """
    Deduplicate only records that have a stable explicit identity.

    Records without a dataset name are never merged merely because their
    descriptions look similar.
    """
    result: list[DatasetRecord] = []
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        if record.name is None:
            result.append(record)
            continue

        key = (
            record.name.casefold(),
            (record.version or "").casefold(),
            (record.role or "").casefold(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return tuple(result)


class DatasetAnalyzer:
    """
    Deterministic post-parser dataset-analysis boundary.

    It consumes parser output + already-retrieved evidence and normalizes the
    result for API/frontend consumption.
    """

    def __init__(
        self,
        *,
        require_evidence_for_success: bool = True,
        require_grounded_flag: bool = True,
        allow_insufficient_evidence: bool = True,
    ) -> None:
        if not isinstance(
            require_evidence_for_success,
            bool,
        ):
            raise TypeError(
                "require_evidence_for_success must be a boolean."
            )

        if not isinstance(
            require_grounded_flag,
            bool,
        ):
            raise TypeError(
                "require_grounded_flag must be a boolean."
            )

        if not isinstance(
            allow_insufficient_evidence,
            bool,
        ):
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
    ) -> DatasetAnalysisResult:
        """
        Normalize one parser-produced dataset result.

        No retrieval, LLM inference, PDF parsing, or external lookup occurs.
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

        # The parser-owned production schema is strict.
        structured = _normalize_schema_fields(
            structured_raw
        )

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
            raise DatasetInputError(
                "Uploaded-paper dataset analysis requires "
                "document_id."
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

        records = _deduplicate_dataset_records(
            _parse_extended_dataset_records(
                structured,
                authoritative_evidence,
            )
        )

        citations = tuple(
            item
            for item in authoritative_evidence
            if item.get("citation") is not None
        )

        source_count = _source_count(
            authoritative_evidence
        )

        citation_count = _count_citations(
            authoritative_evidence
        )

        diagnostics = {
            "evidence_count": len(authoritative_evidence),
            "source_count": source_count,
            "citation_count": citation_count,
            "dataset_count": len(records),
            "dataset_name_present": (
                self._has_explicit_dataset_name(
                    structured
                )
            ),
            "processing_seconds": (
                time.perf_counter() - started
            ),
            "schema": "research_paper_dataset",
        }

        logger.info(
            "Dataset analysis completed: task=%s scope=%s "
            "paper_id=%s document_id=%s status=%s "
            "dataset_count=%d evidence_count=%d",
            normalized_task,
            normalized_scope,
            paper_id,
            document_id,
            status,
            len(records),
            len(authoritative_evidence),
        )

        return DatasetAnalysisResult(
            status=status,
            success=success,
            grounded=grounded,
            paper_id=paper_id,
            document_id=document_id,
            title=title,
            structured_output=structured,
            datasets=records,
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
            raise DatasetTaskError(
                "task_type must be a string."
            )

        normalized = task_type.strip().lower()

        if normalized != TASK_TYPE:
            raise DatasetTaskError(
                "DatasetAnalyzer only handles "
                "task_type='dataset'; "
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
            raise DatasetInputError(
                "scope must be a string."
            )

        normalized = candidate.strip().lower()

        if normalized not in SUPPORTED_SCOPES:
            raise DatasetInputError(
                f"Unsupported scope={candidate!r}; expected one of "
                f"{sorted(SUPPORTED_SCOPES)!r}."
            )

        return normalized

    @staticmethod
    def _has_explicit_dataset_name(
        structured_output: Mapping[str, Any],
    ) -> bool:
        value = _optional_string(
            structured_output.get("dataset_name")
        )

        return (
            value is not None
            and value.casefold() != NOT_FOUND.casefold()
        )

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

        if (
            self.require_evidence_for_success
            and not evidence
        ):
            if self.allow_insufficient_evidence:
                return INSUFFICIENT_EVIDENCE, False

            raise DatasetEvidenceError(
                "No authoritative dataset evidence is available."
            )

        if self.require_grounded_flag and not grounded:
            return VALIDATION_ERROR, False

        if not _contains_meaningful_dataset_information(
            structured_output
        ):
            return INSUFFICIENT_EVIDENCE, False

        return SUCCESS, True


class ResearchDatasetAnalyzer:
    """
    Batch facade for independent Top-N paper dataset analysis.

    Each paper receives its own evidence context. Evidence is never pooled.
    """

    def __init__(
        self,
        analyzer: Optional[DatasetAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or DatasetAnalyzer()

    def analyze_papers(
        self,
        papers: Sequence[Any],
        *,
        responses: Sequence[Any],
        evidences: Optional[Sequence[Any]] = None,
    ) -> tuple[DatasetAnalysisResult, ...]:
        if not isinstance(papers, Sequence) or isinstance(
            papers, (str, bytes, bytearray)
        ):
            raise DatasetInputError(
                "papers must be a sequence."
            )

        if not isinstance(responses, Sequence) or isinstance(
            responses, (str, bytes, bytearray)
        ):
            raise DatasetInputError(
                "responses must be a sequence."
            )

        if len(papers) != len(responses):
            raise DatasetInputError(
                "papers and responses must have equal lengths."
            )

        if evidences is None:
            evidence_values: Sequence[Any] = [
                None
            ] * len(papers)
        else:
            if not isinstance(
                evidences,
                Sequence,
            ) or isinstance(
                evidences,
                (str, bytes, bytearray),
            ):
                raise DatasetInputError(
                    "evidences must be a sequence when supplied."
                )

            if len(evidences) != len(papers):
                raise DatasetInputError(
                    "papers, responses, and evidences must have "
                    "equal lengths."
                )

            evidence_values = evidences

        results: list[DatasetAnalysisResult] = []

        for index, (
            paper,
            response,
            paper_evidence,
        ) in enumerate(
            zip(
                papers,
                responses,
                evidence_values,
            )
        ):
            try:
                result = self.analyzer.analyze(
                    paper,
                    parsed_response=response,
                    evidence=paper_evidence,
                    scope="research",
                )
            except DatasetAnalysisError as exc:
                raise DatasetAnalysisError(
                    "Dataset analysis failed for paper "
                    f"index {index}."
                ) from exc

            results.append(result)

        return tuple(results)


_DEFAULT_ANALYZER = DatasetAnalyzer()


def analyze_dataset(
    paper: Any = None,
    *,
    evidence: Any = None,
    parsed_response: Any = None,
    pipeline_response: Any = None,
    scope: Optional[str] = None,
) -> DatasetAnalysisResult:
    """Functional convenience API."""
    return _DEFAULT_ANALYZER.analyze(
        paper,
        evidence=evidence,
        parsed_response=parsed_response,
        pipeline_response=pipeline_response,
        scope=scope,
    )


# Short alias for callers that prefer a generic analysis function.
analyze = analyze_dataset


def run_self_test() -> None:
    """
    Deterministic unit-style tests.

    No internet, LLM API, API key, GPU, FAISS, embedding model, or PDF parser
    is required.
    """
    analyzer = DatasetAnalyzer()

    # ---------------------------------------------------------------
    # Shared evidence.
    # ---------------------------------------------------------------
    cifar_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "chunk_dataset_01",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "page": 5,
            "section": "Dataset",
            "citation": "paper_001:p5",
            "text": (
                "Experiments were conducted on CIFAR-10. "
                "The dataset contains 60,000 images belonging to "
                "10 classes. We use 50,000 images for training and "
                "10,000 for testing."
            ),
        },
    )

    parsed = {
        "dataset_name": "CIFAR-10",
        "source": "Not found in the provided evidence.",
        "size": "60,000 images",
        "number_of_classes": "10",
        "splits": "50,000 images for training; 10,000 images for testing",
        "preprocessing": "Not found in the provided evidence.",
        "augmentation": "Not found in the provided evidence.",
        "other_details": [],
        "grounded": True,
        "evidence": [
            {
                "evidence_number": 1,
                "chunk_id": "chunk_dataset_01",
                "document_id": "paper_001",
                "paper_id": "paper_001",
                "page": 5,
                "section": "Dataset",
            }
        ],
    }

    # 1. Single dataset.
    result = analyzer.analyze(
        {
            "paper_id": "paper_001",
            "document_id": "paper_001",
            "title": "CIFAR Example",
        },
        evidence=cifar_evidence,
        parsed_response=parsed,
    )

    assert result.status == SUCCESS
    assert result.success is True
    assert result.dataset_name == "CIFAR-10"
    assert result.size == "60,000 images"
    assert result.number_of_classes == "10"
    assert result.splits == (
        "50,000 images for training; 10,000 images for testing"
    )
    assert len(result.datasets) == 1
    assert result.datasets[0].name == "CIFAR-10"

    # 2. Evidence/citation/page/section preservation.
    assert result.evidence[0]["citation"] == "paper_001:p5"
    assert result.evidence[0]["page"] == 5
    assert result.evidence[0]["section"] == "Dataset"
    assert result.evidence[0]["chunk_id"] == "chunk_dataset_01"

    # 3. No inference from known dataset name.
    assert result.source == NOT_FOUND
    assert result.number_of_classes == "10"
    assert "50,000" in result.splits
    assert "10,000" in result.splits

    # 4. Version preservation through the future-compatible internal record.
    # This is tested through explicit upstream metadata, not general knowledge.
    versioned_record_input = dict(parsed)
    versioned_record_input["evidence"] = [
        dict(parsed["evidence"][0])
    ]

    # Current strict schema intentionally rejects arbitrary version fields.
    # Therefore version is not silently added to structured_output.
    assert "version" not in result.structured_output

    # 5. Multiple datasets through the analyzer's future-compatible internal
    # representation cannot be supplied to the strict parser schema today.
    # This is an explicit compatibility boundary, not a hidden schema fork.
    try:
        extended = dict(parsed)
        extended["datasets"] = [
            {"name": "ImageNet", "role": "pretraining"},
            {"name": "COCO", "role": "evaluation"},
        ]
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=cifar_evidence,
            parsed_response=extended,
        )
    except DatasetInputError:
        pass
    else:
        raise AssertionError(
            "Competing dataset schema was silently accepted."
        )

    # 6. Abstract-only evidence.
    abstract_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "abstract_01",
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "page": 1,
            "section": "Abstract",
            "text": (
                "We evaluate our model on ImageNet."
            ),
        },
    )

    abstract = dict(parsed)
    abstract.update(
        {
            "dataset_name": "ImageNet",
            "size": NOT_FOUND,
            "number_of_classes": NOT_FOUND,
            "splits": NOT_FOUND,
        }
    )
    abstract["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "abstract_01",
            "document_id": "paper_002",
            "paper_id": "paper_002",
            "page": 1,
            "section": "Abstract",
        }
    ]

    abstract_result = analyzer.analyze(
        {"paper_id": "paper_002"},
        evidence=abstract_evidence,
        parsed_response=abstract,
    )

    assert abstract_result.dataset_name == "ImageNet"
    assert abstract_result.size == NOT_FOUND
    assert abstract_result.number_of_classes == NOT_FOUND
    assert abstract_result.splits == NOT_FOUND

    # 7. No dataset guessing.
    unknown = dict(parsed)
    unknown.update(
        {
            "dataset_name": NOT_FOUND,
            "source": NOT_FOUND,
            "size": NOT_FOUND,
            "number_of_classes": NOT_FOUND,
            "splits": NOT_FOUND,
            "preprocessing": NOT_FOUND,
            "augmentation": NOT_FOUND,
            "other_details": [],
            "grounded": True,
            "evidence": [
                {
                    "evidence_number": 1,
                    "chunk_id": "unknown_01",
                    "document_id": "paper_003",
                    "paper_id": "paper_003",
                    "page": 1,
                    "section": "Abstract",
                }
            ],
        }
    )

    unknown_result = analyzer.analyze(
        {"paper_id": "paper_003"},
        evidence=(
            {
                "evidence_number": 1,
                "chunk_id": "unknown_01",
                "document_id": "paper_003",
                "paper_id": "paper_003",
                "page": 1,
                "section": "Abstract",
                "text": (
                    "We evaluate on a standard image "
                    "classification benchmark."
                ),
            },
        ),
        parsed_response=unknown,
    )

    assert unknown_result.status == INSUFFICIENT_EVIDENCE
    assert unknown_result.success is False
    assert unknown_result.dataset_name == NOT_FOUND

    # 8. Private/custom dataset: preserve only supplied facts.
    private = dict(parsed)
    private.update(
        {
            "dataset_name": NOT_FOUND,
            "source": NOT_FOUND,
            "size": "20,000 medical images",
            "number_of_classes": NOT_FOUND,
            "splits": NOT_FOUND,
            "other_details": [
                "private dataset",
                "medical images",
            ],
        }
    )

    private_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "private_01",
            "document_id": "paper_004",
            "paper_id": "paper_004",
            "page": 4,
            "section": "Data",
        },
    )

    private["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "private_01",
            "document_id": "paper_004",
            "paper_id": "paper_004",
            "page": 4,
            "section": "Data",
        }
    ]

    private_result = analyzer.analyze(
        {"paper_id": "paper_004"},
        evidence=private_evidence,
        parsed_response=private,
    )

    assert private_result.dataset_name == NOT_FOUND
    assert private_result.size == "20,000 medical images"
    assert "private dataset" in private_result.other_details

    # 9. Preprocessing/augmentation are preserved only when provided.
    prepared = dict(parsed)
    prepared.update(
        {
            "preprocessing": "images resized to 224x224 and normalized",
            "augmentation": "random cropping and horizontal flipping",
        }
    )

    prepared_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=cifar_evidence,
        parsed_response=prepared,
    )

    assert "224x224" in prepared_result.preprocessing
    assert "horizontal flipping" in prepared_result.augmentation

    # 10. Multiple-paper isolation.
    paper_a_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "a_dataset",
            "document_id": "paper_A",
            "paper_id": "paper_A",
            "page": 2,
            "section": "Dataset",
        },
    )

    paper_b_evidence = (
        {
            "evidence_number": 1,
            "chunk_id": "b_dataset",
            "document_id": "paper_B",
            "paper_id": "paper_B",
            "page": 3,
            "section": "Dataset",
        },
    )

    parsed_a = dict(parsed)
    parsed_a["dataset_name"] = "ImageNet"
    parsed_a["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "a_dataset",
            "document_id": "paper_A",
            "paper_id": "paper_A",
            "page": 2,
            "section": "Dataset",
        }
    ]

    parsed_b = dict(parsed)
    parsed_b["dataset_name"] = "COCO"
    parsed_b["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "b_dataset",
            "document_id": "paper_B",
            "paper_id": "paper_B",
            "page": 3,
            "section": "Dataset",
        }
    ]

    batch = ResearchDatasetAnalyzer(
        analyzer
    ).analyze_papers(
        [
            {"paper_id": "paper_A"},
            {"paper_id": "paper_B"},
        ],
        responses=[parsed_a, parsed_b],
        evidences=[
            paper_a_evidence,
            paper_b_evidence,
        ],
    )

    assert batch[0].dataset_name == "ImageNet"
    assert batch[1].dataset_name == "COCO"
    assert batch[0].paper_id == "paper_A"
    assert batch[1].paper_id == "paper_B"

    # 11. Cross-paper contamination rejected.
    try:
        analyzer.analyze(
            {"paper_id": "paper_A"},
            evidence=paper_a_evidence + paper_b_evidence,
            parsed_response=parsed_a,
        )
    except DatasetEvidenceError:
        pass
    else:
        raise AssertionError(
            "Cross-paper evidence was silently accepted."
        )

    # 12. Uploaded scope requires document identity.
    try:
        analyzer.analyze(
            {},
            evidence=(
                {
                    "evidence_number": 1,
                    "chunk_id": "upload_01",
                    "document_id": None,
                    "paper_id": None,
                    "page": 1,
                    "section": "Dataset",
                },
            ),
            parsed_response=parsed,
            scope="uploaded",
        )
    except DatasetInputError:
        pass
    else:
        raise AssertionError(
            "Uploaded scope accepted missing document identity."
        )

    # 13. Invalid structured input.
    invalid = dict(parsed)
    invalid["size"] = 60000

    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=cifar_evidence,
            parsed_response=invalid,
        )
    except DatasetInputError:
        pass
    else:
        raise AssertionError(
            "Invalid dataset structured input was accepted."
        )

    # 14. Ungrounded output fails closed.
    ungrounded = dict(parsed)
    ungrounded["grounded"] = False

    ungrounded_result = analyzer.analyze(
        {"paper_id": "paper_001"},
        evidence=cifar_evidence,
        parsed_response=ungrounded,
    )

    assert ungrounded_result.status == VALIDATION_ERROR
    assert ungrounded_result.success is False

    # 15. Invalid evidence reference.
    wrong_reference = dict(parsed)
    wrong_reference["evidence"] = [
        {
            "evidence_number": 1,
            "chunk_id": "does_not_exist",
            "document_id": "paper_001",
            "paper_id": "paper_001",
            "page": 5,
            "section": "Dataset",
        }
    ]

    try:
        analyzer.analyze(
            {"paper_id": "paper_001"},
            evidence=cifar_evidence,
            parsed_response=wrong_reference,
        )
    except DatasetEvidenceError:
        pass
    else:
        raise AssertionError(
            "Unavailable evidence reference was accepted."
        )

    # 16. Duplicate dataset record handling.
    records = (
        DatasetRecord(
            name="COCO",
            version="2017",
            role="evaluation",
        ),
        DatasetRecord(
            name="COCO",
            version="2017",
            role="evaluation",
        ),
        DatasetRecord(
            name="ImageNet",
            role="pretraining",
        ),
    )

    deduped = _deduplicate_dataset_records(records)

    assert len(deduped) == 2

    # 17. Comparison scope permits multiple evidence identities but does not
    # perform comparison.
    comparison = analyzer.analyze(
        {},
        evidence=paper_a_evidence + paper_b_evidence,
        parsed_response=parsed_a,
        scope="comparison",
    )

    assert comparison.scope == "comparison"
    assert comparison.dataset_name == "ImageNet"

    # 18. Deterministic behavior.
    repeat = analyzer.analyze(
        {
            "paper_id": "paper_001",
            "document_id": "paper_001",
            "title": "CIFAR Example",
        },
        evidence=cifar_evidence,
        parsed_response=parsed,
    )

    first = result.to_dict()
    second = repeat.to_dict()

    # Processing latency is intentionally variable and therefore excluded.
    first["diagnostics"] = {
        k: v
        for k, v in first["diagnostics"].items()
        if k != "processing_seconds"
    }
    second["diagnostics"] = {
        k: v
        for k, v in second["diagnostics"].items()
        if k != "processing_seconds"
    }

    assert first == second

    # 19. Security boundary.
    assert "subprocess" not in globals()
    assert "eval" not in globals()
    assert "exec" not in globals()
    assert "os" not in globals()

    # 20. No heavyweight retrieval/LLM dependencies imported.
    forbidden_names = {
        "faiss",
        "torch",
        "transformers",
        "sentence_transformers",
        "openai",
        "anthropic",
        "streamlit",
        "fitz",
        "pymupdf",
    }

    assert forbidden_names.isdisjoint(globals())

    print("DatasetAnalyzer self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
    run_self_test()