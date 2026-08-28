"""
Evidence-grounded strength analysis for the AI Research Paper Assistant.

Primary responsibility
----------------------
Identify scientifically defensible strengths from supplied paper evidence and
previously extracted structured analysis.

Architectural boundary
----------------------
This module:
    - does not call an LLM
    - does not build prompts
    - does not retrieve documents
    - does not load FAISS
    - does not generate embeddings
    - does not read PDFs
    - does not access external knowledge
    - does not compare papers
    - does not assign an arbitrary paper-quality score

The LLM/parser schema currently used by the project defines:

    strengths: list[str]
    author_stated_strengths: list[str]
    inferred_strengths: list[str]
    grounded: bool
    evidence: list[object]

This file preserves that contract while also exposing richer internal
StrengthItem objects for application code that needs category/reason/evidence/
confidence/source provenance.

The implementation is deterministic and intentionally conservative:
unsupported strengths are omitted rather than guessed.
"""

from __future__ import annotations

import logging
from pathlib import Path
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK_TYPE = "strengths"

SUCCESS = "success"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
VALIDATION_ERROR = "validation_error"

NOT_FOUND = "Not found in the provided evidence."
UNAVAILABLE = "Not available in the provided content."
UNDETERMINABLE = "Cannot be reliably determined from the available content."

SUPPORTED_SCOPES = frozenset({"research", "uploaded", "comparison"})
CONFIDENCES = frozenset({"high", "medium", "low"})

# Higher values mean stronger evidence quality.
_EVIDENCE_QUALITY = {
    "quantitative": 5,
    "comparative": 5,
    "direct": 4,
    "methodological": 3,
    "author_claim": 1,
    "unknown": 0,
}

# Explicitly disallowed promotional / absolute language unless it is clearly
# preserved as an attributed author claim.
_OVERCLAIM_TERMS = (
    "revolutionary",
    "groundbreaking",
    "perfect",
    "flawless",
    "unbeatable",
    "guaranteed",
    "universally superior",
    "universally better",
    "best paper",
    "completely robust",
)

# Phrases that usually identify an author claim rather than independent
# evidence. These are never sufficient for an inferred strength by themselves.
_AUTHOR_CLAIM_PATTERNS = (
    r"\bthe authors?\s+(?:claim|argue|state|assert|suggest|report|describe)\b",
    r"\bwe\s+(?:claim|argue|state|assert|believe|suggest)\b",
    r"\bour\s+(?:method|model|approach)\s+(?:is|achieves|outperforms)\b",
    r"\bthe proposed\s+(?:method|model|approach)\s+is\s+(?:highly|very|extremely)\b",
)

# Evidence patterns. These are intentionally conservative and operate only on
# supplied text. They do not import domain knowledge or query the web.
_PATTERNS = {
    "problem_definition": (
        r"\bresearch (?:question|problem)\b",
        r"\bwe address\b",
        r"\bwe investigate\b",
        r"\bwe study\b",
        r"\bthe (?:aim|objective|goal) of (?:this|the) (?:study|work|paper)\b",
        r"\bthis paper (?:aims|investigates|addresses)\b",
        r"\bresearch gap\b",
    ),
    "contribution": (
        r"\bwe (?:propose|present|introduce)\b",
        r"\bwe introduce\b",
        r"\bmain contribution\b",
        r"\bour contribution\b",
        r"\bthis work introduces\b",
        r"\bnew (?:framework|method|approach|dataset|benchmark|analysis)\b",
    ),
    "methodology": (
        r"\bexperimental (?:design|setup|procedure)\b",
        r"\bmethodology\b",
        r"\bwe (?:train|evaluate|compare|conduct|perform)\b",
        r"\bthe proposed (?:method|approach|pipeline)\b",
        r"\bworkflow\b",
        r"\bpipeline\b",
        r"\bpreprocessing\b",
        r"\bexperimental protocol\b",
    ),
    "multi_dataset": (
        r"\b(?:three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:benchmark\s+)?datasets?\b",
        r"\bmultiple datasets?\b",
        r"\bacross (?:multiple|several|three|four|five|\d+) datasets?\b",
        r"\bdatasets?\s+(?:a|b|c)\b",
    ),
    "benchmark_dataset": (
        r"\bbenchmark dataset\b",
        r"\bstandard benchmark\b",
        r"\bbenchmark datasets\b",
    ),
    "dataset_split": (
        r"\btrain(?:ing)?[/-](?:test|validation)\b",
        r"\btraining set\b.*\btest(?:ing)? set\b",
        r"\bvalidation set\b",
        r"\bheld[- ]out test\b",
        r"\btest split\b",
    ),
    "multiple_baselines": (
        r"\b(?:compared|evaluated|tested)\s+(?:against|with)\s+"
        r"(?:\w+\s+){0,5}(?:baseline|baselines)\b",
        r"\bmultiple baselines?\b",
        r"\bfive established baseline\b",
        r"\bthree established baseline\b",
        r"\bfour established baseline\b",
        r"\bcompared with .*baselines?\b",
        r"\bcompared against .*baselines?\b",
    ),
    "comparison": (
        r"\bcompared (?:with|against) (?:the )?(?:baseline|baselines|"
        r"previous|prior|existing|competing)\b",
        r"\boutperform(?:s|ed)?\b",
        r"\bimprov(?:e|ed|es|ement)\s+(?:over|upon)\b",
        r"\bhigher .* than .*baseline\b",
        r"\blower .* than .*baseline\b",
        r"\bbetter .* than .*baseline\b",
    ),
    "multiple_metrics": (
        r"\bmultiple (?:evaluation )?metrics\b",
        r"\baccuracy\b.*\bprecision\b.*\brecall\b",
        r"\bprecision\b.*\brecall\b.*\bf1\b",
        r"\b(?:accuracy|precision|recall|f1|auc)\b"
        r".*\b(?:accuracy|precision|recall|f1|auc)\b",
    ),
    "statistical": (
        r"\bstatistical significance\b",
        r"\bp\s*[<=>]\s*0\.\d+",
        r"\bconfidence interval\b",
        r"\b95%\s+ci\b",
        r"\bstandard deviation\b",
        r"\bstatistically significant\b",
    ),
    "ablation": (
        r"\bablation (?:study|analysis|experiment)\b",
        r"\bablation\b.*\bmodule\b",
        r"\bremoving .* reduced\b",
        r"\bwithout .* component\b.*\bperformance\b",
    ),
    "robustness": (
        r"\brobustness\b",
        r"\bgaussian noise\b",
        r"\bnoise levels?\b",
        r"\bperturb(?:ation|ations)\b",
        r"\bdomain shift\b",
        r"\badversarial\b",
        r"\bdifferent environments?\b",
        r"\bvarying (?:noise|conditions|parameters)\b",
    ),
    "cross_domain": (
        r"\bcross[- ]domain\b",
        r"\bacross domains?\b",
        r"\bdomain generalization\b",
        r"\bdifferent domains?\b",
    ),
    "external_validation": (
        r"\bexternal validation\b",
        r"\bexternal test(?:ing)?\b",
        r"\bindependent test(?:ing)? set\b",
        r"\bheld[- ]out external\b",
        r"\bexternal dataset\b",
    ),
    "repeated_experiments": (
        r"\brepeated experiments?\b",
        r"\bmultiple runs?\b",
        r"\brepeated runs?\b",
        r"\b\d+\s+runs?\b",
        r"\bmean\s*[±+/-]\s*(?:std|standard deviation)\b",
    ),
    "error_analysis": (
        r"\berror analysis\b",
        r"\bfailure cases?\b",
        r"\banalysis of errors\b",
        r"\berror cases?\b",
    ),
    "code": (
        r"\bsource code\b.*\b(?:available|released|public)\b",
        r"\bcode\b.*\b(?:available|released|publicly available)\b",
        r"\bgithub\b",
        r"\brepository\b.*\b(?:available|public)\b",
    ),
    "dataset_availability": (
        r"\bdataset\b.*\b(?:available|publicly available|released)\b",
        r"\bpublic dataset\b",
        r"\bdataset can be downloaded\b",
    ),
    "implementation_details": (
        r"\bimplementation details\b",
        r"\bhyperparameters?\b",
        r"\bconfiguration files?\b",
        r"\bexperimental configuration\b",
        r"\bsupplementary material\b",
        r"\bsupplementary information\b",
    ),
    "efficiency": (
        r"\binference latency\b",
        r"\binference time\b",
        r"\bthroughput\b",
        r"\bflops\b",
        r"\bmemory (?:usage|consumption)\b",
        r"\bparameter count\b",
        r"\btraining time\b",
        r"\bcomputational cost\b",
        r"\bresource (?:constraint|constrained)\b",
    ),
    "real_world": (
        r"\breal[- ]world\b",
        r"\bclinical\b",
        r"\bdeployment\b",
        r"\bdeployed\b",
        r"\bin practice\b",
        r"\bproduction environment\b",
        r"\brealistic (?:setting|scenario|data)\b",
    ),
    "qualitative": (
        r"\bqualitative evaluation\b",
        r"\bqualitative analysis\b",
        r"\buser study\b",
        r"\bhuman evaluation\b",
        r"\bcase stud(?:y|ies)\b",
    ),
}

_CATEGORY_ORDER = (
    "performance",
    "experimental",
    "evaluation",
    "methodology",
    "generalization",
    "reproducibility",
    "dataset",
    "model",
    "problem_definition",
    "contribution",
    "practical_relevance",
)

_PRIORITY = {
    "quantitative": 6,
    "comparative": 6,
    "direct": 5,
    "methodological": 4,
    "author_claim": 1,
    "unknown": 0,
}


class StrengthAnalysisError(RuntimeError):
    """Base exception for strength-analysis failures."""


class StrengthInputError(StrengthAnalysisError, ValueError):
    """Malformed paper or structured-analysis input."""


class StrengthEvidenceError(StrengthAnalysisError, ValueError):
    """Malformed, missing, or cross-paper evidence."""


class StrengthTaskError(StrengthAnalysisError, ValueError):
    """Unsupported task type or analysis scope."""


class EvidenceType(str, Enum):
    QUANTITATIVE = "quantitative"
    COMPARATIVE = "comparative"
    DIRECT = "direct"
    METHODOLOGICAL = "methodological"
    AUTHOR_CLAIM = "author_claim"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceRecord:
    """Normalized evidence with provenance preserved."""

    text: str
    source: str
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    evidence_number: Optional[int] = None
    paper_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[Any] = None
    section: Optional[str] = None
    citation: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "evidence_type": self.evidence_type.value,
        }

        if self.evidence_number is not None:
            result["evidence_number"] = self.evidence_number
        if self.paper_id is not None:
            result["paper_id"] = self.paper_id
        if self.document_id is not None:
            result["document_id"] = self.document_id
        if self.chunk_id is not None:
            result["chunk_id"] = self.chunk_id
        if self.page is not None:
            result["page"] = self.page
        if self.section is not None:
            result["section"] = self.section
        if self.citation is not None:
            result["citation"] = self.citation
        if self.metadata:
            result["metadata"] = _json_copy(self.metadata)

        return result


@dataclass(frozen=True)
class StrengthItem:
    """
    One evidence-backed strength.

    This richer representation is internal/application-facing. The parser
    compatibility payload remains string-list based.
    """

    category: str
    statement: str
    reason: str
    evidence: tuple[EvidenceRecord, ...]
    confidence: str
    evidence_type: str
    source_locations: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "statement": self.statement,
            "reason": self.reason,
            "evidence": [
                item.to_dict() for item in self.evidence
            ],
            "confidence": self.confidence,
            "evidence_type": self.evidence_type,
            "source_locations": _json_copy(
                list(self.source_locations)
            ),
        }


@dataclass(frozen=True)
class StrengthAnalysisResult:
    """Final machine-readable strength analysis."""

    status: str
    success: bool
    grounded: bool
    paper_id: Optional[str]
    document_id: Optional[str]
    title: Optional[str]
    strengths: tuple[StrengthItem, ...]
    author_stated_strengths: tuple[str, ...]
    inferred_strengths: tuple[str, ...]
    evidence: tuple[EvidenceRecord, ...]
    scope: str = "research"
    task_type: str = TASK_TYPE
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Return an application-ready representation.

        The `structured_output` member is exactly compatible with the current
        parser strengths schema.
        """
        structured_output = {
            "strengths": [
                item.statement for item in self.strengths
            ],
            "author_stated_strengths": list(
                self.author_stated_strengths
            ),
            "inferred_strengths": list(
                self.inferred_strengths
            ),
            "grounded": self.grounded,
            "evidence": [
                _evidence_ref(item)
                for item in self.evidence
            ],
        }

        return {
            "status": self.status,
            "success": self.success,
            "grounded": self.grounded,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "title": self.title,
            "scope": self.scope,
            "task_type": self.task_type,
            "strengths": [
                item.to_dict() for item in self.strengths
            ],
            "author_stated_strengths": list(
                self.author_stated_strengths
            ),
            "inferred_strengths": list(
                self.inferred_strengths
            ),
            "evidence": [
                item.to_dict() for item in self.evidence
            ],
            "structured_output": structured_output,
            "diagnostics": _json_copy(self.diagnostics),
        }

    # Convenient compatibility properties.
    @property
    def strength_statements(self) -> tuple[str, ...]:
        return tuple(item.statement for item in self.strengths)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(item.category for item in self.strengths)


def _read(value: Any, name: str, default: Any = None) -> Any:
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
        value,
        (str, bytes, bytearray),
    ):
        return ()

    result: list[str] = []

    for item in value:
        normalized = _optional_string(item)

        if normalized is not None:
            result.append(normalized)

    return tuple(result)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = value.strip()

        if not normalized:
            continue

        key = re.sub(r"\s+", " ", normalized).casefold()

        if key not in seen:
            seen.add(key)
            result.append(normalized)

    return tuple(result)


def _json_copy(value: Any) -> Any:
    """Copy only JSON-safe data; reject arbitrary objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrengthInputError(
                "Strength-analysis data contains a non-finite number."
            )
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _json_copy(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_copy(child) for child in value]

    raise StrengthInputError(
        "Unsupported value type in strength-analysis data: "
        f"{type(value).__name__}."
    )


def _normalize_scope(scope: Any) -> str:
    if scope is None:
        return "research"

    if not isinstance(scope, str):
        raise StrengthInputError("scope must be a string.")

    normalized = scope.strip().lower()

    if normalized not in SUPPORTED_SCOPES:
        raise StrengthInputError(
            f"Unsupported scope={scope!r}; expected one of "
            f"{sorted(SUPPORTED_SCOPES)!r}."
        )

    return normalized


def _normalize_task_type(task_type: Any) -> str:
    if not isinstance(task_type, str):
        raise StrengthTaskError(
            "task_type must be a string."
        )

    normalized = task_type.strip().lower()

    if normalized != TASK_TYPE:
        raise StrengthTaskError(
            "StrengthAnalyzer only handles "
            f"task_type={TASK_TYPE!r}; received {task_type!r}."
        )

    return normalized


def _normalize_paper_identity(
    paper: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    paper_id = _first_string(
        _read(paper, "paper_id"),
        _read(paper, "document_id"),
        _read(paper, "id"),
    )

    document_id = _first_string(
        _read(paper, "document_id"),
        _read(paper, "paper_id"),
        _read(paper, "id"),
    )

    title = _first_string(
        _read(paper, "title"),
    )

    return paper_id, document_id, title


def _normalize_evidence(
    value: Any,
    *,
    default_paper_id: Optional[str] = None,
    default_document_id: Optional[str] = None,
) -> tuple[EvidenceRecord, ...]:
    if value is None:
        return ()

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise StrengthEvidenceError(
            "Evidence must be a sequence."
        )

    records: list[EvidenceRecord] = []

    for index, item in enumerate(value):
        if isinstance(item, EvidenceRecord):
            record = item
        elif isinstance(item, Mapping):
            text = _first_string(
                item.get("text"),
                item.get("evidence"),
                item.get("content"),
            )

            if text is None:
                raise StrengthEvidenceError(
                    f"Evidence item {index} has no text."
                )

            source = _first_string(
                item.get("source"),
                item.get("section"),
                item.get("location"),
            ) or "provided_content"

            raw_type = _optional_string(
                item.get("evidence_type")
            )

            try:
                evidence_type = EvidenceType(
                    raw_type or EvidenceType.UNKNOWN.value
                )
            except ValueError:
                evidence_type = EvidenceType.UNKNOWN

            evidence_number = item.get("evidence_number")

            if (
                evidence_number is not None
                and (
                    isinstance(evidence_number, bool)
                    or not isinstance(evidence_number, int)
                )
            ):
                raise StrengthEvidenceError(
                    f"Evidence item {index} has invalid "
                    "'evidence_number'."
                )

            metadata = item.get("metadata", {})

            if not isinstance(metadata, Mapping):
                raise StrengthEvidenceError(
                    f"Evidence item {index} metadata must be an object."
                )

            record = EvidenceRecord(
                text=text,
                source=source,
                evidence_type=evidence_type,
                evidence_number=evidence_number,
                paper_id=_first_string(
                    item.get("paper_id"),
                    default_paper_id,
                ),
                document_id=_first_string(
                    item.get("document_id"),
                    default_document_id,
                ),
                chunk_id=_optional_string(
                    item.get("chunk_id")
                ),
                page=item.get("page"),
                section=_optional_string(
                    item.get("section")
                ),
                citation=_optional_string(
                    item.get("citation")
                ),
                metadata=_json_copy(metadata),
            )
        else:
            raise StrengthEvidenceError(
                f"Evidence item {index} must be an object."
            )

        records.append(record)

    return tuple(records)


def _infer_evidence_type(
    text: str,
    explicit: Optional[EvidenceType] = None,
) -> EvidenceType:
    if explicit is not None and explicit != EvidenceType.UNKNOWN:
        return explicit

    lowered = text.casefold()

    if _matches_any(
        lowered,
        (
            r"\b(?:accuracy|precision|recall|f1|auc|mAP|rmse|mae)\b"
            r"\s*(?:=|of|was|were)\s*[-+]?\d",
            r"\b\d+(?:\.\d+)?%\b",
            r"\bp\s*[<=>]\s*0\.\d+",
            r"\bconfidence interval\b",
        ),
    ):
        return EvidenceType.QUANTITATIVE

    if _matches_any(
        lowered,
        _PATTERNS["comparison"],
    ):
        return EvidenceType.COMPARATIVE

    if _matches_any(
        lowered,
        _AUTHOR_CLAIM_PATTERNS,
    ):
        return EvidenceType.AUTHOR_CLAIM

    if _matches_any(
        lowered,
        _PATTERNS["methodology"],
    ):
        return EvidenceType.METHODOLOGICAL

    return EvidenceType.DIRECT


def _is_author_claim(text: str) -> bool:
    lowered = text.casefold()
    return _matches_any(
        lowered,
        _AUTHOR_CLAIM_PATTERNS,
    )


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _extract_candidate_evidence_from_paper(
    paper: Any,
    *,
    paper_id: Optional[str],
    document_id: Optional[str],
) -> tuple[EvidenceRecord, ...]:
    """
    Consume already-supplied evidence/content.

    Priority:
        1. explicit evidence
        2. content/context/chunks/sections
        3. abstract/summary
        4. structured analysis fields

    No external knowledge is used.
    """
    explicit = _read(paper, "evidence", None)

    if explicit is not None:
        records = _normalize_evidence(
            explicit,
            default_paper_id=paper_id,
            default_document_id=document_id,
        )

        if records:
            return tuple(
                EvidenceRecord(
                    text=item.text,
                    source=item.source,
                    evidence_type=_infer_evidence_type(
                        item.text,
                        item.evidence_type,
                    ),
                    evidence_number=item.evidence_number,
                    paper_id=item.paper_id,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    page=item.page,
                    section=item.section,
                    citation=item.citation,
                    metadata=item.metadata,
                )
                for item in records
            )

    candidates: list[EvidenceRecord] = []

    content = _read(paper, "content", None)

    if isinstance(content, str) and content.strip():
        candidates.append(
            EvidenceRecord(
                text=content.strip(),
                source="provided_content",
                evidence_type=_infer_evidence_type(content),
                paper_id=paper_id,
                document_id=document_id,
            )
        )

    # RAG/context representations that may already exist on the paper object.
    for field_name in ("context", "chunks", "sections"):
        value = _read(paper, field_name, None)

        if value is None:
            continue

        if isinstance(value, str) and value.strip():
            candidates.append(
                EvidenceRecord(
                    text=value.strip(),
                    source=field_name,
                    evidence_type=_infer_evidence_type(value),
                    paper_id=paper_id,
                    document_id=document_id,
                )
            )
            continue

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            try:
                nested = _normalize_evidence(
                    value,
                    default_paper_id=paper_id,
                    default_document_id=document_id,
                )
            except StrengthEvidenceError:
                nested = ()

            candidates.extend(
                EvidenceRecord(
                    text=item.text,
                    source=item.source,
                    evidence_type=_infer_evidence_type(
                        item.text,
                        item.evidence_type,
                    ),
                    evidence_number=item.evidence_number,
                    paper_id=item.paper_id,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    page=item.page,
                    section=item.section,
                    citation=item.citation,
                    metadata=item.metadata,
                )
                for item in nested
            )

    # Abstract/summary is legitimate source material, but only if supplied.
    for field_name, source_name in (
        ("abstract", "abstract"),
        ("summary", "summary"),
    ):
        value = _read(paper, field_name, None)

        if isinstance(value, str) and value.strip():
            candidates.append(
                EvidenceRecord(
                    text=value.strip(),
                    source=source_name,
                    evidence_type=_infer_evidence_type(value),
                    paper_id=paper_id,
                    document_id=document_id,
                )
            )

    # Previously extracted structured analysis may be supplied by the caller.
    # We do not reinterpret missing values or inject general knowledge.
    for field_name in (
        "summary_analysis",
        "methodology",
        "model",
        "dataset",
        "findings",
        "analysis",
    ):
        value = _read(paper, field_name, None)

        if value is None:
            continue

        texts = _structured_text_values(value)

        for text in texts:
            candidates.append(
                EvidenceRecord(
                    text=text,
                    source=field_name,
                    evidence_type=_infer_evidence_type(text),
                    paper_id=paper_id,
                    document_id=document_id,
                )
            )

    return _dedupe_evidence(candidates)


def _structured_text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()

    if isinstance(value, Mapping):
        result: list[str] = []

        for key, child in value.items():
            if key in {
                "evidence",
                "citations",
                "diagnostics",
            }:
                continue

            if isinstance(child, str) and child.strip():
                result.append(child.strip())
            elif isinstance(child, Sequence) and not isinstance(
                child,
                (str, bytes, bytearray),
            ):
                result.extend(
                    item.strip()
                    for item in child
                    if isinstance(item, str) and item.strip()
                )

        return tuple(result)

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )

    return ()


def _dedupe_evidence(
    evidence: Sequence[EvidenceRecord],
) -> tuple[EvidenceRecord, ...]:
    result: list[EvidenceRecord] = []
    seen: set[tuple[str, str, Optional[str], Optional[Any]]] = set()

    for item in evidence:
        key = (
            re.sub(r"\s+", " ", item.text).casefold(),
            item.source.casefold(),
            item.chunk_id,
            item.page,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return tuple(result)


def _validate_evidence_identity(
    evidence: Sequence[EvidenceRecord],
    *,
    paper_id: Optional[str],
    document_id: Optional[str],
    scope: str,
) -> None:
    if scope == "comparison":
        return

    identities: set[tuple[str, str]] = set()

    for item in evidence:
        item_paper = item.paper_id or ""
        item_document = item.document_id or ""

        if item_paper or item_document:
            identities.add(
                (item_paper, item_document)
            )

    if len(identities) > 1:
        raise StrengthEvidenceError(
            "Evidence from multiple papers/documents was supplied to a "
            "single-paper strength analysis."
        )

    if paper_id is not None:
        for item in evidence:
            if (
                item.paper_id is not None
                and item.paper_id != paper_id
            ):
                raise StrengthEvidenceError(
                    f"Evidence paper_id={item.paper_id!r} does not "
                    f"match paper_id={paper_id!r}."
                )

    if document_id is not None:
        for item in evidence:
            if (
                item.document_id is not None
                and item.document_id != document_id
            ):
                raise StrengthEvidenceError(
                    f"Evidence document_id={item.document_id!r} does "
                    f"not match document_id={document_id!r}."
                )


def _evidence_ref(
    evidence: EvidenceRecord,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in (
        ("evidence_number", evidence.evidence_number),
        ("chunk_id", evidence.chunk_id),
        ("document_id", evidence.document_id),
        ("paper_id", evidence.paper_id),
        ("page", evidence.page),
        ("section", evidence.section),
        ("citation", evidence.citation),
    ):
        if value is not None:
            result[key] = value

    return result


def _source_locations(
    evidence: Sequence[EvidenceRecord],
) -> tuple[Mapping[str, Any], ...]:
    locations: list[Mapping[str, Any]] = []

    for item in evidence:
        location = _evidence_ref(item)

        if location:
            locations.append(location)

    return tuple(locations)


def _contains_quantitative(text: str) -> bool:
    return _infer_evidence_type(text) == EvidenceType.QUANTITATIVE


def _contains_comparison(text: str) -> bool:
    return _matches_any(
        text,
        _PATTERNS["comparison"],
    )


def _contains_author_claim(text: str) -> bool:
    return _is_author_claim(text)


def _supporting_evidence(
    evidence: Sequence[EvidenceRecord],
    patterns: Sequence[str],
    *,
    require_non_author_claim: bool = True,
) -> tuple[EvidenceRecord, ...]:
    result: list[EvidenceRecord] = []

    for item in evidence:
        if not _matches_any(item.text, patterns):
            continue

        if (
            require_non_author_claim
            and item.evidence_type == EvidenceType.AUTHOR_CLAIM
        ):
            continue

        result.append(item)

    return tuple(result)


def _best_confidence(
    evidence: Sequence[EvidenceRecord],
) -> str:
    if not evidence:
        return "low"

    qualities = [
        _EVIDENCE_QUALITY.get(
            item.evidence_type.value,
            0,
        )
        for item in evidence
    ]

    maximum = max(qualities)
    count = len(evidence)

    if maximum >= 5:
        return "high"

    if maximum >= 4 or count >= 2:
        return "medium"

    return "low"


def _best_evidence_type(
    evidence: Sequence[EvidenceRecord],
) -> str:
    if not evidence:
        return EvidenceType.UNKNOWN.value

    best = max(
        evidence,
        key=lambda item: _EVIDENCE_QUALITY.get(
            item.evidence_type.value,
            0,
        ),
    )

    return best.evidence_type.value


def _has_overclaim_language(text: str) -> bool:
    lowered = text.casefold()
    return any(
        term in lowered
        for term in _OVERCLAIM_TERMS
    )


def _safe_statement(text: str) -> str:
    """
    Remove accidental marketing language from generated rule-based statements.

    This function only normalizes our own deterministic templates; it never
    rewrites source evidence.
    """
    normalized = re.sub(r"\s+", " ", text).strip()

    for term in _OVERCLAIM_TERMS:
        normalized = re.sub(
            rf"\b{re.escape(term)}\b",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = re.sub(r"\s+([,.])", r"\1", normalized)

    return normalized.strip()


def _numeric_comparison_details(
    evidence: Sequence[EvidenceRecord],
) -> Optional[dict[str, str]]:
    """
    Extract explicit proposed-vs-baseline numeric facts without calculating
    improvements.

    Accepted forms are deliberately narrow:
        Proposed F1 = 0.91; Baseline F1 = 0.86
        F1 0.91 vs 0.86
        accuracy of 94.2% versus 91.1%

    The function never computes a delta.
    """
    for item in evidence:
        text = item.text

        patterns = (
            (
                r"(?:proposed|ours|our method|our model)\s+"
                r"(?P<metric>\b(?:accuracy|precision|recall|f1|auc|map|"
                r"mAP|rmse|mae)\b)"
                r"\s*(?:=|of|was)\s*"
                r"(?P<proposed>\d+(?:\.\d+)?%?)"
                r".{0,80}?"
                r"(?:baseline|previous|prior|competing)\s+"
                r"(?:\w+\s+)?"
                r"(?P<metric2>\b(?:accuracy|precision|recall|f1|auc|map|"
                r"mAP|rmse|mae)\b)"
                r"\s*(?:=|of|was)\s*"
                r"(?P<baseline>\d+(?:\.\d+)?%?)"
            ),
            (
                r"(?P<metric>\b(?:accuracy|precision|recall|f1|auc|map|"
                r"mAP|rmse|mae)\b)"
                r".{0,30}?"
                r"(?:proposed|ours|our method|our model)"
                r".{0,20}?"
                r"(?P<proposed>\d+(?:\.\d+)?%?)"
                r".{0,60}?"
                r"(?:baseline|previous|prior|competing)"
                r".{0,20}?"
                r"(?P<baseline>\d+(?:\.\d+)?%?)"
            ),
            (
                r"(?P<metric>\b(?:accuracy|precision|recall|f1|auc|map|"
                r"mAP|rmse|mae)\b)"
                r"\s*(?P<proposed>\d+(?:\.\d+)?%?)"
                r"\s*(?:vs\.?|versus|compared with|against)\s*"
                r"(?P<baseline>\d+(?:\.\d+)?%?)"
            ),
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:
                return {
                    "metric": match.group("metric"),
                    "proposed": match.group("proposed"),
                    "baseline": match.group("baseline"),
                }

    return None


def _numeric_strength(
    evidence: Sequence[EvidenceRecord],
) -> Optional[StrengthItem]:
    details = _numeric_comparison_details(evidence)

    if details is None:
        return None

    # Only call it a performance improvement when the source itself establishes
    # the comparison direction. We do not calculate a difference.
    comparison_evidence = tuple(
        item
        for item in evidence
        if _contains_comparison(item.text)
        and _contains_quantitative(item.text)
    )

    if not comparison_evidence:
        return None

    statement = (
        f"The proposed method achieves higher "
        f"{details['metric']} performance than the reported baseline."
    )

    reason = (
        "The paper reports an explicit quantitative comparison between "
        "the proposed method and a baseline."
    )

    return StrengthItem(
        category="performance",
        statement=statement,
        reason=reason,
        evidence=comparison_evidence,
        confidence=_best_confidence(comparison_evidence),
        evidence_type=_best_evidence_type(comparison_evidence),
        source_locations=_source_locations(
            comparison_evidence
        ),
    )


def _candidate_strengths(
    evidence: Sequence[EvidenceRecord],
) -> list[StrengthItem]:
    candidates: list[StrengthItem] = []

    def add(
        category: str,
        statement: str,
        reason: str,
        patterns: Sequence[str],
        *,
        minimum_evidence: int = 1,
    ) -> None:
        supporting = _supporting_evidence(evidence, patterns)

        if len(supporting) < minimum_evidence:
            return

        clean_statement = _safe_statement(statement)

        if not clean_statement:
            return

        if _has_overclaim_language(clean_statement):
            return

        candidates.append(
            StrengthItem(
                category=category,
                statement=clean_statement,
                reason=reason,
                evidence=supporting,
                confidence=_best_confidence(supporting),
                evidence_type=_best_evidence_type(supporting),
                source_locations=_source_locations(
                    supporting
                ),
            )
        )

    # ---------------------------------------------------------------
    # Problem definition
    # ---------------------------------------------------------------
    add(
        "problem_definition",
        "The paper clearly defines the research problem or objective.",
        "The supplied evidence explicitly states the research problem, "
        "objective, question, or identified gap.",
        _PATTERNS["problem_definition"],
    )

    # ---------------------------------------------------------------
    # Contribution
    # ---------------------------------------------------------------
    add(
        "contribution",
        "The paper clearly describes its research contribution.",
        "The supplied evidence explicitly identifies a proposed approach, "
        "contribution, framework, method, dataset, benchmark, or analysis.",
        _PATTERNS["contribution"],
    )

    # ---------------------------------------------------------------
    # Methodology
    # ---------------------------------------------------------------
    add(
        "methodology",
        "The study provides an explicitly described research or experimental procedure.",
        "The evidence describes the experimental design, procedure, pipeline, "
        "or methodological steps rather than merely naming a model.",
        _PATTERNS["methodology"],
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    add(
        "dataset",
        "The approach is evaluated across multiple datasets.",
        "The paper explicitly reports evaluation on multiple datasets.",
        _PATTERNS["multi_dataset"],
    )

    add(
        "dataset",
        "The study uses benchmark data for empirical evaluation.",
        "The supplied evidence explicitly identifies benchmark or standard "
        "benchmark datasets.",
        _PATTERNS["benchmark_dataset"],
    )

    add(
        "dataset",
        "The evaluation uses an explicitly defined train/test or validation split.",
        "The paper describes how training, validation, and/or testing data "
        "are separated.",
        _PATTERNS["dataset_split"],
    )

    # ---------------------------------------------------------------
    # Model / baseline
    # ---------------------------------------------------------------
    add(
        "model",
        "The proposed approach is compared with multiple reported baselines.",
        "The evidence explicitly describes comparisons against multiple "
        "baseline methods.",
        _PATTERNS["multiple_baselines"],
    )

    # ---------------------------------------------------------------
    # Experimental design
    # ---------------------------------------------------------------
    add(
        "experimental",
        "The study includes comparative experiments against existing methods.",
        "The evidence reports direct comparisons with baselines, prior, "
        "existing, or competing methods.",
        _PATTERNS["comparison"],
    )

    add(
        "experimental",
        "The study includes an ablation analysis of model or method components.",
        "The evidence explicitly reports an ablation study or component-removal "
        "experiment.",
        _PATTERNS["ablation"],
    )

    add(
        "experimental",
        "The study evaluates performance under varying conditions.",
        "The paper reports experiments involving noise, perturbations, domain "
        "shift, parameter changes, adversarial conditions, or other controlled "
        "condition changes.",
        _PATTERNS["robustness"],
    )

    add(
        "experimental",
        "The study includes cross-domain evaluation.",
        "The evidence explicitly reports evaluation across different domains.",
        _PATTERNS["cross_domain"],
    )

    add(
        "experimental",
        "The study includes external validation.",
        "The evidence explicitly identifies an external or independent "
        "validation/testing setting.",
        _PATTERNS["external_validation"],
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    add(
        "evaluation",
        "The study evaluates performance using multiple metrics.",
        "The evidence reports multiple evaluation measures.",
        _PATTERNS["multiple_metrics"],
    )

    add(
        "evaluation",
        "The study includes quantitative statistical evaluation.",
        "The evidence reports statistical significance, confidence intervals, "
        "standard deviation, or repeated quantitative measurements.",
        _PATTERNS["statistical"],
    )

    add(
        "evaluation",
        "The study includes repeated experimental evaluation.",
        "The paper reports repeated runs or repeated experiments rather than "
        "a single reported run.",
        _PATTERNS["repeated_experiments"],
    )

    add(
        "evaluation",
        "The study includes error analysis of failure cases.",
        "The evidence explicitly describes error analysis or examination of "
        "failure cases.",
        _PATTERNS["error_analysis"],
    )

    add(
        "evaluation",
        "The study includes qualitative evaluation.",
        "The paper explicitly reports qualitative analysis, human evaluation, "
        "a user study, or case studies.",
        _PATTERNS["qualitative"],
    )

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------
    add(
        "reproducibility",
        "The availability of source code supports reproducibility.",
        "The evidence states that source code, a public repository, or code "
        "implementation is available.",
        _PATTERNS["code"],
    )

    add(
        "reproducibility",
        "The paper provides implementation details that support reproducibility.",
        "The supplied evidence includes implementation details, hyperparameters, "
        "configuration, or supplementary material.",
        _PATTERNS["implementation_details"],
    )

    add(
        "reproducibility",
        "The dataset availability supports reproducibility of the evaluation.",
        "The evidence explicitly states that the dataset is public or available.",
        _PATTERNS["dataset_availability"],
    )

    # ---------------------------------------------------------------
    # Practical relevance
    # ---------------------------------------------------------------
    add(
        "practical_relevance",
        "The study includes evidence of practical or real-world evaluation.",
        "The evidence explicitly describes real-world settings, deployment, "
        "clinical evaluation, or practical use.",
        _PATTERNS["real_world"],
    )

    add(
        "practical_relevance",
        "The study reports efficiency-related evaluation.",
        "The paper reports measurable latency, throughput, memory, FLOPs, "
        "parameter count, training time, or computational/resource usage.",
        _PATTERNS["efficiency"],
    )

    numeric_item = _numeric_strength(evidence)

    if numeric_item is not None:
        candidates.append(numeric_item)

    return candidates


def _deduplicate_strengths(
    strengths: Sequence[StrengthItem],
) -> tuple[StrengthItem, ...]:
    """
    Merge duplicate conceptual strengths.

    Example:
        multiple datasets
        evaluated on three datasets

    are represented as one dataset strength when they map to the same category
    and normalized statement.
    """
    result: list[StrengthItem] = []
    index_by_key: dict[tuple[str, str], int] = {}

    for item in strengths:
        normalized_statement = re.sub(
            r"\s+",
            " ",
            item.statement,
        ).strip().casefold()

        # Collapse obvious paraphrase variants.
        normalized_statement = (
            normalized_statement
            .replace(
                "the approach is evaluated across multiple datasets",
                "multi_dataset_evaluation",
            )
            .replace(
                "the study is evaluated across multiple datasets",
                "multi_dataset_evaluation",
            )
            .replace(
                "the method is tested on multiple datasets",
                "multi_dataset_evaluation",
            )
            .replace(
                "the study includes comparative experiments against existing methods",
                "baseline_comparison",
            )
        )

        key = (
            item.category,
            normalized_statement,
        )

        existing_index = index_by_key.get(key)

        if existing_index is None:
            index_by_key[key] = len(result)
            result.append(item)
            continue

        existing = result[existing_index]

        merged_evidence = _dedupe_evidence(
            existing.evidence + item.evidence
        )

        merged = StrengthItem(
            category=existing.category,
            statement=existing.statement,
            reason=existing.reason,
            evidence=merged_evidence,
            confidence=_best_confidence(merged_evidence),
            evidence_type=_best_evidence_type(
                merged_evidence
            ),
            source_locations=_source_locations(
                merged_evidence
            ),
        )

        result[existing_index] = merged

    return tuple(result)


def _rank_strengths(
    strengths: Sequence[StrengthItem],
) -> tuple[StrengthItem, ...]:
    category_priority = {
        category: len(_CATEGORY_ORDER) - index
        for index, category in enumerate(
            _CATEGORY_ORDER
        )
    }

    return tuple(
        sorted(
            strengths,
            key=lambda item: (
                -_PRIORITY.get(
                    item.evidence_type,
                    0,
                ),
                -category_priority.get(
                    item.category,
                    0,
                ),
                -len(item.evidence),
                item.statement.casefold(),
            ),
        )
    )


def _extract_author_claims(
    evidence: Sequence[EvidenceRecord],
) -> tuple[str, ...]:
    claims: list[str] = []

    for item in evidence:
        if (
            item.evidence_type != EvidenceType.AUTHOR_CLAIM
            and not _is_author_claim(item.text)
        ):
            continue

        text = re.sub(r"\s+", " ", item.text).strip()

        if not text:
            continue

        # Do not expose entire long passages as "strengths". Preserve a concise
        # sentence only when the supplied evidence itself is already short.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        for sentence in sentences:
            normalized = sentence.strip()

            if not normalized:
                continue

            if _matches_any(
                normalized,
                _AUTHOR_CLAIM_PATTERNS,
            ):
                claims.append(normalized)

    return _dedupe_strings(claims)


def _validate_parser_compatible_output(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "strengths",
        "author_stated_strengths",
        "inferred_strengths",
        "grounded",
        "evidence",
    }

    if set(data) != expected:
        missing = expected - set(data)
        extra = set(data) - expected

        raise StrengthInputError(
            "Strength structured output schema mismatch. "
            f"missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}."
        )

    for field_name in (
        "strengths",
        "author_stated_strengths",
        "inferred_strengths",
    ):
        value = data[field_name]

        if not isinstance(value, list) or any(
            not isinstance(item, str)
            for item in value
        ):
            raise StrengthInputError(
                f"'{field_name}' must be an array of strings."
            )

    if not isinstance(data["grounded"], bool):
        raise StrengthInputError(
            "'grounded' must be boolean."
        )

    if not isinstance(data["evidence"], list):
        raise StrengthInputError(
            "'evidence' must be an array."
        )

    return _json_copy(data)


def _extract_previous_strengths(
    paper: Any,
) -> tuple[str, ...]:
    """
    Preserve explicit previously-extracted strengths when supplied.

    They are NOT automatically treated as independently verified strengths.
    The caller can provide them for deduplication/context; evidence-grounded
    candidates are still required for the inferred `strengths` list.
    """
    candidates: list[str] = []

    for field_name in (
        "strengths",
        "strength",
        "author_stated_strengths",
    ):
        value = _read(paper, field_name, None)

        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            candidates.extend(
                item
                for item in value
                if isinstance(item, str)
            )

    return _dedupe_strings(candidates)


def _build_result(
    *,
    paper_id: Optional[str],
    document_id: Optional[str],
    title: Optional[str],
    scope: str,
    evidence: tuple[EvidenceRecord, ...],
    strengths: tuple[StrengthItem, ...],
    author_stated_strengths: tuple[str, ...],
    status: str,
    success: bool,
    grounded: bool,
    started: float,
) -> StrengthAnalysisResult:
    inferred = tuple(
        item.statement
        for item in strengths
    )

    diagnostics = {
        "strength_count": len(strengths),
        "evidence_count": len(evidence),
        "author_claim_count": len(
            author_stated_strengths
        ),
        "inferred_strength_count": len(inferred),
        "processing_seconds": (
            time.perf_counter() - started
        ),
        "analysis_status": status,
    }

    return StrengthAnalysisResult(
        status=status,
        success=success,
        grounded=grounded,
        paper_id=paper_id,
        document_id=document_id,
        title=title,
        strengths=strengths,
        author_stated_strengths=author_stated_strengths,
        inferred_strengths=inferred,
        evidence=evidence,
        scope=scope,
        task_type=TASK_TYPE,
        diagnostics=diagnostics,
    )


class StrengthAnalyzer:
    """
    Deterministic, evidence-first strength analyzer.

    Primary integration API:
        analyze(paper)

    Extended API:
        analyze(
            paper,
            summary=...,
            methodology=...,
            model=...,
            dataset=...,
            findings=...,
            evidence=...,
            scope=...,
        )

    The extended fields are optional compatibility hooks for future/current
    orchestrators that already expose structured component outputs. They do
    not trigger new retrieval or LLM work.
    """

    def __init__(
        self,
        *,
        require_evidence: bool = True,
        allow_author_claims: bool = True,
    ) -> None:
        if not isinstance(require_evidence, bool):
            raise TypeError(
                "require_evidence must be boolean."
            )

        if not isinstance(
            allow_author_claims,
            bool,
        ):
            raise TypeError(
                "allow_author_claims must be boolean."
            )

        self.require_evidence = require_evidence
        self.allow_author_claims = (
            allow_author_claims
        )

    def analyze(
        self,
        paper: Any,
        *,
        summary: Any = None,
        methodology: Any = None,
        model: Any = None,
        dataset: Any = None,
        findings: Any = None,
        evidence: Any = None,
        scope: Optional[str] = None,
        task_type: str = TASK_TYPE,
    ) -> StrengthAnalysisResult:
        started = time.perf_counter()

        normalized_task = _normalize_task_type(
            task_type
        )
        normalized_scope = _normalize_scope(
            scope
        )

        paper_id, document_id, title = (
            _normalize_paper_identity(paper)
        )

        if (
            paper_id is None
            and document_id is None
        ):
            raise StrengthInputError(
                "Paper must provide paper_id or document_id."
            )

        # Merge optional component outputs into a shallow analysis view. This
        # is already-produced information; no component is executed here.
        analysis_view = paper

        if any(
            value is not None
            for value in (
                summary,
                methodology,
                model,
                dataset,
                findings,
            )
        ):
            if isinstance(paper, Mapping):
                analysis_view = dict(paper)
            else:
                analysis_view = {
                    "paper_id": paper_id,
                    "document_id": document_id,
                    "title": title,
                }

            for key, value in (
                ("summary_analysis", summary),
                ("methodology", methodology),
                ("model", model),
                ("dataset", dataset),
                ("findings", findings),
            ):
                if value is not None:
                    analysis_view[key] = value

        if evidence is None:
            normalized_evidence = (
                _extract_candidate_evidence_from_paper(
                    analysis_view,
                    paper_id=paper_id,
                    document_id=document_id,
                )
            )
        else:
            normalized_evidence = _normalize_evidence(
                evidence,
                default_paper_id=paper_id,
                default_document_id=document_id,
            )

        # Explicit caller evidence takes precedence, but structured component
        # evidence may be carried inside those components and can be extracted
        # without invoking them.
        if evidence is not None:
            normalized_evidence = _dedupe_evidence(
                normalized_evidence
                + _extract_component_evidence(
                    (
                        summary,
                        methodology,
                        model,
                        dataset,
                        findings,
                    ),
                    paper_id=paper_id,
                    document_id=document_id,
                )
            )

        _validate_evidence_identity(
            normalized_evidence,
            paper_id=paper_id,
            document_id=document_id,
            scope=normalized_scope,
        )

        if self.require_evidence and not normalized_evidence:
            return _build_result(
                paper_id=paper_id,
                document_id=document_id,
                title=title,
                scope=normalized_scope,
                evidence=(),
                strengths=(),
                author_stated_strengths=(),
                status=INSUFFICIENT_EVIDENCE,
                success=False,
                grounded=False,
                started=started,
            )

        author_claims = (
            _extract_author_claims(
                normalized_evidence
            )
            if self.allow_author_claims
            else ()
        )

        candidates = _candidate_strengths(
            normalized_evidence
        )

        # Remove strengths whose only supporting evidence is an author claim.
        candidates = [
            item
            for item in candidates
            if any(
                evidence_item.evidence_type
                != EvidenceType.AUTHOR_CLAIM
                for evidence_item in item.evidence
            )
        ]

        deduplicated = _deduplicate_strengths(
            candidates
        )

        ranked = _rank_strengths(
            deduplicated
        )

        # The output is grounded if there is at least one actual evidence-backed
        # strength. If the paper contains evidence but no defensible strength,
        # fail closed with insufficient_evidence.
        if not ranked:
            return _build_result(
                paper_id=paper_id,
                document_id=document_id,
                title=title,
                scope=normalized_scope,
                evidence=normalized_evidence,
                strengths=(),
                author_stated_strengths=author_claims,
                status=INSUFFICIENT_EVIDENCE,
                success=False,
                grounded=False,
                started=started,
            )

        result = _build_result(
            paper_id=paper_id,
            document_id=document_id,
            title=title,
            scope=normalized_scope,
            evidence=normalized_evidence,
            strengths=ranked,
            author_stated_strengths=author_claims,
            status=SUCCESS,
            success=True,
            grounded=True,
            started=started,
        )

        # Final schema contract check.
        structured = {
            "strengths": list(
                result.strength_statements
            ),
            "author_stated_strengths": list(
                result.author_stated_strengths
            ),
            "inferred_strengths": list(
                result.inferred_strengths
            ),
            "grounded": result.grounded,
            "evidence": [
                _evidence_ref(item)
                for item in result.evidence
            ],
        }

        _validate_parser_compatible_output(
            structured
        )

        logger.info(
            "Strength analysis completed: "
            "paper_id=%s document_id=%s strength_count=%d "
            "evidence_count=%d status=%s",
            paper_id,
            document_id,
            len(ranked),
            len(normalized_evidence),
            result.status,
        )

        return result


class ResearchStrengthAnalyzer:
    """Batch facade for independent Mode-1 / multi-upload analysis."""

    def __init__(
        self,
        analyzer: Optional[StrengthAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or StrengthAnalyzer()

    def analyze_papers(
        self,
        papers: Sequence[Any],
        *,
        evidences: Optional[Sequence[Any]] = None,
    ) -> tuple[StrengthAnalysisResult, ...]:
        if not isinstance(
            papers,
            Sequence,
        ) or isinstance(
            papers,
            (str, bytes, bytearray),
        ):
            raise StrengthInputError(
                "papers must be a sequence."
            )

        if evidences is None:
            evidence_values: Sequence[Any] = (
                [None] * len(papers)
            )
        else:
            if not isinstance(
                evidences,
                Sequence,
            ) or isinstance(
                evidences,
                (str, bytes, bytearray),
            ):
                raise StrengthInputError(
                    "evidences must be a sequence."
                )

            if len(evidences) != len(papers):
                raise StrengthInputError(
                    "papers and evidences must have equal lengths."
                )

            evidence_values = evidences

        seen_paper_ids: set[str] = set()
        results: list[StrengthAnalysisResult] = []

        for index, (
            paper,
            paper_evidence,
        ) in enumerate(
            zip(
                papers,
                evidence_values,
            )
        ):
            paper_id, document_id, _ = (
                _normalize_paper_identity(paper)
            )

            identity = paper_id or document_id

            if identity is None:
                raise StrengthInputError(
                    f"Paper at index {index} has no identity."
                )

            if identity in seen_paper_ids:
                raise StrengthInputError(
                    f"Duplicate paper/document identity: {identity!r}."
                )

            seen_paper_ids.add(identity)

            results.append(
                self.analyzer.analyze(
                    paper,
                    evidence=paper_evidence,
                    scope="research",
                )
            )

        return tuple(results)


def _extract_component_evidence(
    components: Sequence[Any],
    *,
    paper_id: Optional[str],
    document_id: Optional[str],
) -> tuple[EvidenceRecord, ...]:
    result: list[EvidenceRecord] = []

    for component in components:
        if component is None:
            continue

        if isinstance(component, StrengthAnalysisResult):
            result.extend(component.evidence)
            continue

        component_evidence = _read(
            component,
            "evidence",
            None,
        )

        if component_evidence is None:
            continue

        try:
            result.extend(
                _normalize_evidence(
                    component_evidence,
                    default_paper_id=paper_id,
                    default_document_id=document_id,
                )
            )
        except StrengthEvidenceError:
            # The authoritative validation layer will reject malformed
            # component evidence. This module does not invent a replacement.
            continue

    return tuple(result)


_DEFAULT_ANALYZER = StrengthAnalyzer()


def analyze_strengths(
    paper: Any,
    *,
    summary: Any = None,
    methodology: Any = None,
    model: Any = None,
    dataset: Any = None,
    findings: Any = None,
    evidence: Any = None,
    scope: Optional[str] = None,
) -> StrengthAnalysisResult:
    """Functional convenience wrapper."""
    return _DEFAULT_ANALYZER.analyze(
        paper,
        summary=summary,
        methodology=methodology,
        model=model,
        dataset=dataset,
        findings=findings,
        evidence=evidence,
        scope=scope,
    )


# Common short alias.
analyze = analyze_strengths


def run_self_test() -> None:
    """Comprehensive offline tests; no LLM, internet, GPU, or FAISS."""

    def make_evidence(
        text: str,
        *,
        paper_id: str = "paper_001",
        page: int = 7,
        section: str = "Experiments",
        chunk_id: str = "chunk_001",
        source: str = "paper",
    ) -> tuple[dict[str, Any], ...]:
        return (
            {
                "evidence_number": 1,
                "chunk_id": chunk_id,
                "document_id": paper_id,
                "paper_id": paper_id,
                "page": page,
                "section": section,
                "source": source,
                "text": text,
            },
        )

    base_paper = {
        "paper_id": "paper_001",
        "document_id": "paper_001",
        "title": "Evidence-Grounded Research Paper",
    }

    # ------------------------------------------------------------------
    # 1. Clear methodological strength.
    # ------------------------------------------------------------------
    result = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The experimental design evaluates the proposed method "
            "using a controlled experimental protocol with training, "
            "validation, and testing stages."
        ),
    )

    assert result.success is True
    assert any(
        item.category == "methodology"
        for item in result.strengths
    )

    # ------------------------------------------------------------------
    # 2. Multiple-dataset strength.
    # ------------------------------------------------------------------
    multi_dataset = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The proposed method was evaluated on three benchmark "
            "datasets: Dataset A, Dataset B, and Dataset C."
        ),
    )

    assert any(
        "multiple datasets" in item.statement
        for item in multi_dataset.strengths
    )

    # ------------------------------------------------------------------
    # 3. Multiple baseline comparison.
    # ------------------------------------------------------------------
    baselines = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The method was compared against five established baseline "
            "methods in the experimental evaluation."
        ),
    )

    assert any(
        item.category == "model"
        for item in baselines.strengths
    )

    # ------------------------------------------------------------------
    # 4. Quantitative performance improvement.
    # ------------------------------------------------------------------
    quantitative = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The proposed F1 = 0.91, compared with the baseline F1 = 0.86."
        ),
    )

    assert any(
        item.category == "performance"
        for item in quantitative.strengths
    )
    assert any(
        "higher F1" in item.statement
        for item in quantitative.strengths
    )

    # ------------------------------------------------------------------
    # 5. Multiple metrics.
    # ------------------------------------------------------------------
    metrics = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "Performance was evaluated using accuracy, precision, recall, "
            "and F1 score."
        ),
    )

    assert any(
        item.category == "evaluation"
        for item in metrics.strengths
    )

    # ------------------------------------------------------------------
    # 6. Ablation.
    # ------------------------------------------------------------------
    ablation = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "An ablation study was performed by removing module X and "
            "measuring its effect on F1."
        ),
    )

    assert any(
        "ablation" in item.statement.casefold()
        for item in ablation.strengths
    )

    # ------------------------------------------------------------------
    # 7. Robustness.
    # ------------------------------------------------------------------
    robustness = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "Performance was evaluated under Gaussian noise levels "
            "of sigma=0.1, 0.2, and 0.3."
        ),
    )

    assert any(
        "varying conditions" in item.statement
        for item in robustness.strengths
    )

    assert not any(
        item.statement == "The model is robust."
        for item in robustness.strengths
    )

    # ------------------------------------------------------------------
    # 8. External validation.
    # ------------------------------------------------------------------
    external = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The model was evaluated on an independent external test set."
        ),
    )

    assert any(
        item.category == "experimental"
        and "external validation" in item.statement.casefold()
        for item in external.strengths
    )

    # ------------------------------------------------------------------
    # 9. Reproducibility implementation details.
    # ------------------------------------------------------------------
    reproducibility = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The paper provides implementation details, hyperparameters, "
            "and configuration files."
        ),
    )

    assert any(
        item.category == "reproducibility"
        for item in reproducibility.strengths
    )

    # ------------------------------------------------------------------
    # 10. Code availability.
    # ------------------------------------------------------------------
    code = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "Source code is publicly available in a GitHub repository."
        ),
    )

    assert any(
        "source code" in item.statement.casefold()
        for item in code.strengths
    )

    # ------------------------------------------------------------------
    # 11. Dataset availability.
    # ------------------------------------------------------------------
    dataset_available = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The dataset is publicly available for download."
        ),
    )

    assert any(
        "dataset availability" in item.statement.casefold()
        or "dataset" in item.statement.casefold()
        for item in dataset_available.strengths
    )

    # ------------------------------------------------------------------
    # 12. Practical relevance.
    # ------------------------------------------------------------------
    practical = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The system was evaluated in a real-world deployment setting."
        ),
    )

    assert any(
        item.category == "practical_relevance"
        for item in practical.strengths
    )

    # ------------------------------------------------------------------
    # 13. Efficiency.
    # ------------------------------------------------------------------
    efficiency = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The evaluation reports inference latency and throughput."
        ),
    )

    assert any(
        "efficiency" in item.statement.casefold()
        for item in efficiency.strengths
    )

    # ------------------------------------------------------------------
    # 14. Cross-domain.
    # ------------------------------------------------------------------
    cross_domain = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The approach was evaluated across two different domains."
        ),
    )

    assert any(
        "cross-domain" in item.statement.casefold()
        for item in cross_domain.strengths
    )

    # ------------------------------------------------------------------
    # 15. Error analysis.
    # ------------------------------------------------------------------
    error_analysis = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The paper includes error analysis of failure cases."
        ),
    )

    assert any(
        "error analysis" in item.statement.casefold()
        for item in error_analysis.strengths
    )

    # ------------------------------------------------------------------
    # 16. Statistical evaluation.
    # ------------------------------------------------------------------
    statistical = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The improvement was statistically significant with p < 0.05 "
            "and a 95% confidence interval."
        ),
    )

    assert any(
        "statistical" in item.statement.casefold()
        for item in statistical.strengths
    )

    # ------------------------------------------------------------------
    # 17. Multiple strengths.
    # ------------------------------------------------------------------
    multiple = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The proposed method was evaluated on three benchmark datasets "
            "and compared with five established baseline methods. "
            "Performance was evaluated using accuracy, precision, recall, "
            "and F1."
        ),
    )

    assert len(multiple.strengths) >= 3

    # ------------------------------------------------------------------
    # 18. Duplicate handling.
    # ------------------------------------------------------------------
    duplicate = StrengthAnalyzer().analyze(
        base_paper,
        evidence=(
            {
                **make_evidence(
                    "The proposed method was evaluated on three datasets."
                )[0],
                "chunk_id": "chunk_a",
            },
            {
                **make_evidence(
                    "The proposed method was evaluated on three datasets."
                )[0],
                "chunk_id": "chunk_b",
            },
        ),
    )

    dataset_strengths = [
        item
        for item in duplicate.strengths
        if item.category == "dataset"
        and "multiple datasets" in item.statement
    ]

    assert len(dataset_strengths) == 1
    assert len(dataset_strengths[0].evidence) == 2

    # ------------------------------------------------------------------
    # 19. Categorization.
    # ------------------------------------------------------------------
    assert all(
        item.category in set(_CATEGORY_ORDER)
        for item in multiple.strengths
    )

    # ------------------------------------------------------------------
    # 20. Evidence preservation.
    # ------------------------------------------------------------------
    assert multiple.evidence[0].text
    assert multiple.evidence[0].chunk_id == "chunk_001"

    # ------------------------------------------------------------------
    # 21. Page preservation.
    # ------------------------------------------------------------------
    assert multiple.evidence[0].page == 7

    # ------------------------------------------------------------------
    # 22. Section preservation.
    # ------------------------------------------------------------------
    assert multiple.evidence[0].section == "Experiments"

    # ------------------------------------------------------------------
    # 23. Paper identity preservation.
    # ------------------------------------------------------------------
    assert multiple.paper_id == "paper_001"
    assert multiple.document_id == "paper_001"

    # ------------------------------------------------------------------
    # 24. Multiple-paper isolation.
    # ------------------------------------------------------------------
    paper_a = {
        "paper_id": "paper_A",
        "document_id": "paper_A",
    }
    paper_b = {
        "paper_id": "paper_B",
        "document_id": "paper_B",
    }

    result_a = StrengthAnalyzer().analyze(
        paper_a,
        evidence=make_evidence(
            "The method was compared with three baselines.",
            paper_id="paper_A",
        ),
    )

    result_b = StrengthAnalyzer().analyze(
        paper_b,
        evidence=make_evidence(
            "The study includes an ablation study.",
            paper_id="paper_B",
        ),
    )

    assert result_a.paper_id == "paper_A"
    assert result_b.paper_id == "paper_B"
    assert all(
        item.paper_id == "paper_A"
        for item in result_a.evidence
    )
    assert all(
        item.paper_id == "paper_B"
        for item in result_b.evidence
    )

    # Cross-paper contamination must fail.
    try:
        StrengthAnalyzer().analyze(
            paper_a,
            evidence=(
                *make_evidence(
                    "The method was compared with three baselines.",
                    paper_id="paper_A",
                ),
                *make_evidence(
                    "The study includes an ablation study.",
                    paper_id="paper_B",
                    chunk_id="chunk_B",
                ),
            ),
        )
    except StrengthEvidenceError:
        pass
    else:
        raise AssertionError(
            "Cross-paper evidence was silently accepted."
        )

    # ------------------------------------------------------------------
    # 25. Multiple uploaded-paper isolation.
    # ------------------------------------------------------------------
    batch = ResearchStrengthAnalyzer().analyze_papers(
        [paper_a, paper_b],
        evidences=[
            make_evidence(
                "The method was compared with three baselines.",
                paper_id="paper_A",
            ),
            make_evidence(
                "The study includes an ablation study.",
                paper_id="paper_B",
            ),
        ],
    )

    assert len(batch) == 2
    assert batch[0].paper_id == "paper_A"
    assert batch[1].paper_id == "paper_B"

    # ------------------------------------------------------------------
    # 26. Missing evidence.
    # ------------------------------------------------------------------
    missing = StrengthAnalyzer().analyze(
        base_paper,
        evidence=(),
    )

    assert missing.status == INSUFFICIENT_EVIDENCE
    assert missing.success is False
    assert missing.strengths == ()

    # ------------------------------------------------------------------
    # 27. Unsupported novelty claim.
    # ------------------------------------------------------------------
    novelty = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The authors state that the proposed method is novel."
        ),
    )

    assert not any(
        "novel" in item.statement.casefold()
        for item in novelty.strengths
    )
    assert novelty.author_stated_strengths

    # ------------------------------------------------------------------
    # 28. Unsupported robustness claim.
    # ------------------------------------------------------------------
    unsupported_robustness = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The authors claim that their model is highly robust."
        ),
    )

    assert not any(
        "robust" in item.statement.casefold()
        for item in unsupported_robustness.strengths
    )
    assert unsupported_robustness.author_stated_strengths

    # ------------------------------------------------------------------
    # 29. Unsupported SOTA claim.
    # ------------------------------------------------------------------
    unsupported_sota = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "The authors state that their method achieves state-of-the-art "
            "performance."
        ),
    )

    assert not any(
        "state-of-the-art" in item.statement.casefold()
        for item in unsupported_sota.strengths
    )

    # ------------------------------------------------------------------
    # 30. No hallucinated strengths.
    # ------------------------------------------------------------------
    hallucination = StrengthAnalyzer().analyze(
        base_paper,
        evidence=make_evidence(
            "We propose a new deep learning model."
        ),
    )

    assert hallucination.status == SUCCESS
    assert hallucination.strengths
    assert not any(
        item.category in {
            "performance",
            "generalization",
            "reproducibility",
        }
        for item in hallucination.strengths
    )

    # ------------------------------------------------------------------
    # 31. No unnecessary LLM call.
    # ------------------------------------------------------------------
    import ast

    source_text = Path(__file__).read_text(
        encoding="utf-8"
    )
    syntax_tree = ast.parse(source_text)

    imported_modules: set[str] = set()

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(
                    node.module.split(".")[0]
                )

    forbidden_modules = {
        "openai",
        "anthropic",
        "ollama",
        "torch",
        "transformers",
        "faiss",
        "fitz",
        "pymupdf",
        "sentence_transformers",
    }

    assert imported_modules.isdisjoint(
        forbidden_modules
    )

    # No dynamic/shell execution APIs are imported.
    assert "subprocess" not in imported_modules
    assert "os" not in imported_modules

    # ------------------------------------------------------------------
    # 34. Deterministic output.
    # ------------------------------------------------------------------
    deterministic_evidence = make_evidence(
        "The proposed method was evaluated on three benchmark datasets "
        "and compared with five established baseline methods."
    )

    first = StrengthAnalyzer().analyze(
        base_paper,
        evidence=deterministic_evidence,
    ).to_dict()

    second = StrengthAnalyzer().analyze(
        base_paper,
        evidence=deterministic_evidence,
    ).to_dict()

    first["diagnostics"] = {}
    second["diagnostics"] = {}

    assert first == second

    # ------------------------------------------------------------------
    # 35. Parser-compatible schema.
    # ------------------------------------------------------------------
    structured = multiple.to_dict()[
        "structured_output"
    ]

    _validate_parser_compatible_output(
        structured
    )

    assert set(structured) == {
        "strengths",
        "author_stated_strengths",
        "inferred_strengths",
        "grounded",
        "evidence",
    }

    # ------------------------------------------------------------------
    # 36. Explicit component integration.
    # ------------------------------------------------------------------
    component = StrengthAnalyzer().analyze(
        base_paper,
        findings={
            "key_findings": [
                "The proposed method was evaluated on three datasets."
            ],
            "evidence": [
                {
                    "text": (
                        "The proposed method was evaluated on three "
                        "datasets."
                    ),
                    "source": "findings",
                    "paper_id": "paper_001",
                    "document_id": "paper_001",
                    "page": 8,
                    "section": "Results",
                }
            ],
        },
        evidence=(),
    )

    assert component.strengths
    assert any(
        item.category == "dataset"
        for item in component.strengths
    )

    # ------------------------------------------------------------------
    # 37. Comparison scope permits multiple identities but does not compare.
    # ------------------------------------------------------------------
    comparison = StrengthAnalyzer().analyze(
        {"paper_id": "paper_A", "document_id": "paper_A"},
        scope="comparison",
        evidence=(
            {
                **make_evidence(
                    "The method was compared with multiple baselines.",
                    paper_id="paper_A",
                )[0],
                "document_id": "paper_A",
            },
            {
                **make_evidence(
                    "The study includes an ablation study.",
                    paper_id="paper_B",
                    chunk_id="chunk_B",
                )[0],
                "document_id": "paper_B",
            },
        ),
    )

    assert comparison.scope == "comparison"
    assert comparison.strengths

    # ------------------------------------------------------------------
    # 38. Security checks.
    # ------------------------------------------------------------------
    assert "subprocess" not in imported_modules
    assert "os" not in imported_modules

    # ------------------------------------------------------------------
    # 39. No arbitrary quality-score / best-paper output fields.
    # ------------------------------------------------------------------
    defined_names = {
        node.id
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Name)
    }

    assert "quality_score" not in defined_names
    assert "best_paper" not in defined_names

    print("StrengthAnalyzer self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
    run_self_test()