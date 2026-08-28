
"""
Evidence-grounded research-paper findings analysis.

Primary responsibility:
    Extract what a paper actually reports as experimental/observational findings.

Design goals:
    - evidence-first
    - deterministic
    - paper-isolated
    - provenance-preserving
    - numerically conservative
    - compatible with mapping/list/dataclass/Pydantic-like inputs
    - no retrieval, embeddings, FAISS, PDF processing, or LLM calls
    - safe for downstream strengths/weaknesses/validation/frontend layers

This module deliberately does not judge whether a result is good or bad.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import logging
import math
import re
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class FindingsAnalysisError(Exception):
    """Base error for findings analysis."""


class FindingsInputError(FindingsAnalysisError):
    """Raised when the supplied paper/evidence cannot be interpreted safely."""


class FindingsEvidenceError(FindingsAnalysisError):
    """Raised when evidence violates the findings evidence contract."""


class FindingsTaskError(FindingsAnalysisError):
    """Raised when analysis execution fails."""


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------

class FindingType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    COMPARATIVE = "comparative"
    PERFORMANCE = "performance"
    ABLATION = "ablation"
    ROBUSTNESS = "robustness"
    ERROR_ANALYSIS = "error_analysis"
    STATISTICAL = "statistical"
    EFFICIENCY = "efficiency"
    COMPUTATIONAL = "computational"
    USER_STUDY = "user_study"
    OBSERVATION = "observation"
    NEGATIVE = "negative"


class EvidenceType(str, Enum):
    REPORTED_RESULT = "reported_result"
    EXPERIMENTALLY_OBSERVED = "experimentally_observed"
    AUTHOR_CLAIM = "author_claim"
    DERIVED_RESULT = "derived_result"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Small immutable-ish data records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    paper_id: Optional[str] = None
    table: Optional[str] = None
    figure: Optional[str] = None
    paragraph: Optional[str] = None
    evidence_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
        }


@dataclass(frozen=True)
class EvidenceRecord:
    text: str
    source: str = "provided_evidence"
    evidence_id: Optional[str] = None
    paper_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    table: Optional[str] = None
    figure: Optional[str] = None
    paragraph: Optional[str] = None
    evidence_type: EvidenceType = EvidenceType.REPORTED_RESULT
    confidence: Confidence = Confidence.HIGH

    def location(self) -> SourceLocation:
        return SourceLocation(
            page=self.page,
            section=self.section,
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            paper_id=self.paper_id,
            table=self.table,
            figure=self.figure,
            paragraph=self.paragraph,
            evidence_id=self.evidence_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "evidence_id": self.evidence_id,
            "paper_id": self.paper_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "section": self.section,
            "table": self.table,
            "figure": self.figure,
            "paragraph": self.paragraph,
            "evidence_type": self.evidence_type.value,
            "confidence": self.confidence.value,
            "source_location": self.location().to_dict(),
        }


@dataclass
class FindingItem:
    finding_id: str
    paper_id: str
    finding_type: str
    statement: str
    evidence: list[EvidenceRecord]
    evidence_type: str
    confidence: str
    metric: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    baseline: Optional[str] = None
    baseline_value: Optional[float] = None
    proposed_value: Optional[float] = None
    comparison: Optional[str] = None
    absolute_improvement: Optional[float] = None
    relative_improvement_percent: Optional[float] = None
    derived_fields: list[str] = field(default_factory=list)
    experimental_condition: Optional[str] = None
    source_locations: list[SourceLocation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "paper_id": self.paper_id,
            "type": self.finding_type,
            "finding_type": self.finding_type,
            "statement": self.statement,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "dataset": self.dataset,
            "split": self.split,
            "baseline": self.baseline,
            "baseline_value": self.baseline_value,
            "proposed_value": self.proposed_value,
            "comparison": self.comparison,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement_percent": self.relative_improvement_percent,
            "derived_fields": list(self.derived_fields),
            "experimental_condition": self.experimental_condition,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "source": (
                self.source_locations[0].to_dict()
                if self.source_locations
                else {}
            ),
            "source_locations": [x.to_dict() for x in self.source_locations],
        }


@dataclass
class FindingsAnalysisResult:
    paper_id: str
    findings: list[FindingItem]
    status: str = "success"
    evidence_count: int = 0
    processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "findings": [x.to_dict() for x in self.findings],
            "status": self.status,
            "evidence_count": self.evidence_count,
            "finding_count": len(self.findings),
            "processing_time": self.processing_time,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Generic safe extraction helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def _read(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if is_dataclass(obj):
        try:
            return getattr(obj, key, default)
        except Exception:
            return default
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _first_string(obj: Any, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = _read(obj, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        return tuple(
            x for x in (
                _optional_string(value.get("name")),
                _optional_string(value.get("label")),
                _optional_string(value.get("text")),
                _optional_string(value.get("value")),
            )
            if x
        )
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            text = _optional_string(item)
            if text:
                result.append(text)
        return tuple(result)
    return ()


def _json_safe_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(x) for x in value]
    if is_dataclass(value):
        return _json_safe_copy(asdict(value))
    return str(value)


def _normalize_identity(paper: Any) -> tuple[str, Optional[str]]:
    paper_id = _first_string(
        paper,
        ("paper_id", "id", "document_id", "doc_id", "paperId"),
    )
    document_id = _first_string(
        paper,
        ("document_id", "doc_id", "documentId"),
    )
    if not paper_id:
        paper_id = document_id or "paper_unknown"
    return paper_id, document_id


def _normalize_page(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\b(\d{1,5})\b", value)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _normalize_confidence(value: Any) -> Confidence:
    text = str(value or "").strip().lower()
    if text == "low":
        return Confidence.LOW
    if text == "medium":
        return Confidence.MEDIUM
    return Confidence.HIGH


def _is_author_claim(text: str) -> bool:
    lowered = text.casefold()
    patterns = (
        r"\bthe authors?\s+(?:claim|argue|state|assert|suggest|report|"
        r"describe|propose|introduce|conclude)\b",
        r"\bour\s+(?:method|model|approach|results)\s+(?:is|are|"
        r"highly|significantly|particularly)\b",
        r"\bwe\s+(?:claim|argue|believe|assert)\b",
        r"\b(?:novel|innovative|powerful|effective|superior|robust)\s+"
        r"(?:method|model|approach|framework|architecture)\b",
    )
    return any(re.search(p, lowered) for p in patterns)


def _infer_evidence_type(text: str, explicit: Any = None) -> EvidenceType:
    if explicit:
        value = str(explicit).strip().lower()
        for item in EvidenceType:
            if item.value == value:
                return item
    if _is_author_claim(text):
        return EvidenceType.AUTHOR_CLAIM
    return EvidenceType.REPORTED_RESULT


def _normalize_evidence(item: Any, paper_id: str, document_id: Optional[str]) -> Optional[EvidenceRecord]:
    if isinstance(item, EvidenceRecord):
        if item.paper_id and item.paper_id != paper_id:
            return None
        return item

    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return EvidenceRecord(
            text=text,
            paper_id=paper_id,
            document_id=document_id,
            evidence_type=_infer_evidence_type(text),
        )

    if not isinstance(item, Mapping) and not is_dataclass(item):
        return None

    text = _first_string(item, ("text", "evidence", "content", "snippet", "quote", "source_text"))
    if not text:
        return None

    item_paper = _first_string(item, ("paper_id", "paperId", "document_id"))
    if item_paper and item_paper != paper_id:
        return None

    return EvidenceRecord(
        text=text,
        source=_first_string(item, ("source", "origin", "source_type")) or "provided_evidence",
        evidence_id=_first_string(item, ("evidence_id", "citation_id", "id", "evidenceId")),
        paper_id=paper_id,
        document_id=_first_string(item, ("document_id", "documentId")) or document_id,
        chunk_id=_first_string(item, ("chunk_id", "chunkId")),
        page=_normalize_page(_read(item, "page", _read(item, "page_number", None))),
        section=_first_string(item, ("section", "section_name")),
        table=_first_string(item, ("table", "table_id", "table_name")),
        figure=_first_string(item, ("figure", "figure_id", "figure_name")),
        paragraph=_first_string(item, ("paragraph", "paragraph_id")),
        evidence_type=_infer_evidence_type(
            text,
            _read(item, "evidence_type", _read(item, "type", None)),
        ),
        confidence=_normalize_confidence(_read(item, "confidence", "high")),
    )


def _dedupe_evidence(items: Sequence[EvidenceRecord]) -> list[EvidenceRecord]:
    seen: set[tuple[Any, ...]] = set()
    result: list[EvidenceRecord] = []
    for item in items:
        key = (
            item.paper_id,
            re.sub(r"\s+", " ", item.text.casefold()).strip(),
            item.page,
            item.section,
            item.table,
            item.figure,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _collect_texts(value: Any, *, max_depth: int = 4, _depth: int = 0) -> list[Any]:
    if value is None or _depth > max_depth:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, Mapping):
        result: list[Any] = []
        priority = (
            "evidence", "findings", "results", "key_findings", "content",
            "text", "statement", "observations", "experiments",
            "comparisons", "metrics", "analysis",
        )
        for key in priority:
            if key in value:
                result.extend(_collect_texts(value[key], max_depth=max_depth, _depth=_depth + 1))
        if not result:
            for v in value.values():
                result.extend(_collect_texts(v, max_depth=max_depth, _depth=_depth + 1))
        return result
    if isinstance(value, Iterable):
        result: list[Any] = []
        for item in value:
            result.extend(_collect_texts(item, max_depth=max_depth, _depth=_depth + 1))
        return result
    return []


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

_EVIDENCE_KEYS = (
    "evidence",
    "evidence_items",
    "evidence_records",
    "supporting_evidence",
    "retrieved_evidence",
    "rag_evidence",
    "citations",
    "sources",
    "results",
    "findings",
    "key_findings",
)


def _extract_evidence(
    paper: Any,
    evidence: Any,
    *,
    paper_id: str,
    document_id: Optional[str],
) -> list[EvidenceRecord]:
    candidates: list[Any] = []

    if evidence is not None:
        # Preserve structured evidence records first; generic recursive
        # traversal is only a fallback for unstructured containers.
        if isinstance(evidence, (str, EvidenceRecord, Mapping)):
            candidates.append(evidence)
        elif isinstance(evidence, (list, tuple)):
            candidates.extend(evidence)
        else:
            candidates.extend(_collect_texts(evidence, max_depth=6))

        if isinstance(evidence, Mapping):
            for key in _EVIDENCE_KEYS:
                if key in evidence:
                    value = evidence[key]
                    if isinstance(value, (list, tuple)):
                        candidates.extend(value)
                    else:
                        candidates.append(value)

    for key in _EVIDENCE_KEYS:
        value = _read(paper, key, None)
        if value is not None:
            candidates.append(value)

    # Existing analysis components may expose their own evidence.
    for component_name in (
        "summary",
        "methodology",
        "model",
        "dataset",
        "findings",
        "strengths",
        "weaknesses",
        "analysis",
        "rag",
        "context",
    ):
        component = _read(paper, component_name, None)
        if component is not None:
            for key in ("evidence", "supporting_evidence", "results"):
                value = _read(component, key, None)
                if value is not None:
                    candidates.append(value)

    normalized: list[EvidenceRecord] = []
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            for sub in candidate:
                record = _normalize_evidence(sub, paper_id, document_id)
                if record:
                    normalized.append(record)
        else:
            record = _normalize_evidence(candidate, paper_id, document_id)
            if record:
                normalized.append(record)

    return _dedupe_evidence(normalized)


# ---------------------------------------------------------------------------
# Pattern and parsing utilities
# ---------------------------------------------------------------------------

_NUMBER = r"(?:[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
_NUM_RANGE = rf"{_NUMBER}(?:\s*(?:±|\+/-|to|–|-)\s*{_NUMBER})?"

_METRIC_ALIASES = (
    "accuracy", "acc",
    "precision", "prec",
    "recall", "sensitivity", "specificity",
    "f1", "f1-score", "f1 score",
    "auc", "roc-auc", "roc auc",
    "map", "mAP",
    "bleu", "rouge", "perplexity",
    "loss", "mae", "mse", "rmse", "r2", "r²",
    "latency", "throughput",
    "memory", "parameters", "parameter count", "flops",
    "fidelity", "ndcg", "hit rate", "mrr",
)


def _metric_from_text(text: str) -> Optional[str]:
    ordered = sorted(_METRIC_ALIASES, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(x) for x in ordered) + r")\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1)
    normalized = raw.lower()
    if normalized in {"acc"}:
        return "accuracy"
    if normalized in {"prec"}:
        return "precision"
    if normalized in {"f1", "f1-score", "f1 score"}:
        return "F1"
    if normalized in {"roc-auc", "roc auc"}:
        return "ROC-AUC"
    if normalized == "map":
        return "mAP"
    if normalized == "r2":
        return "R²"
    return raw


def _unit_from_text(text: str, metric: Optional[str] = None) -> Optional[str]:
    if re.search(r"(?:%|\bpercent\b|\bpercentage\b)", text, re.I):
        return "%"
    match = re.search(
        rf"{_NUMBER}\s*(ms|milliseconds?|s|sec(?:onds?)?|"
        rf"gb|mb|kb|samples?|parameters?|flops?|gflops?|"
        rf"hz|fps)\b",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1)
    if metric in {"accuracy", "precision", "recall", "F1", "AUC", "ROC-AUC", "mAP"}:
        # Do not silently convert 0.91 to 91%.
        return None
    return None


def _parse_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        match = re.search(_NUMBER, value.replace(",", ""))
        if match:
            try:
                number = float(match.group(0))
                return number if math.isfinite(number) else None
            except ValueError:
                return None
    return None


def _parse_metric_value(text: str, metric: Optional[str]) -> Optional[float]:
    if not metric:
        return None
    metric_pattern = re.escape(metric).replace(r"\ ", r"\s+")
    match = re.search(
        rf"{metric_pattern}\s*(?:=|:|is|of|was|were|achieved)?\s*"
        rf"({_NUMBER})\s*(?:%|percent|percentage)?",
        text,
        flags=re.I,
    )
    if match:
        return _parse_number(match.group(1))
    # Generic number close to a metric.
    idx = text.casefold().find(metric.casefold())
    if idx >= 0:
        window = text[max(0, idx - 20): idx + 100]
        match = re.search(_NUMBER, window)
        if match:
            return _parse_number(match.group(0))
    return None


def _extract_all_metric_values(text: str) -> list[tuple[str, float, Optional[str]]]:
    results: list[tuple[str, float, Optional[str]]] = []
    for metric in sorted(_METRIC_ALIASES, key=len, reverse=True):
        if not re.search(r"\b" + re.escape(metric) + r"\b", text, re.I):
            continue
        value = _parse_metric_value(text, metric)
        if value is None:
            continue
        normalized_metric = _metric_from_text(metric)
        results.append((normalized_metric or metric, value, _unit_from_text(text, normalized_metric)))
    # Preserve order but remove duplicate metric/value pairs.
    seen: set[tuple[str, float, Optional[str]]] = set()
    return [x for x in results if not (x in seen or seen.add(x))]


def _extract_dataset(text: str) -> Optional[str]:
    patterns = (
        r"\b(?:on|using|evaluated on|tested on|trained on)\s+"
        r"([A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,4})",
        r"\bdataset\s*[:=]?\s*([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,3})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip(" ,.;:()[]")
            if value and value.lower() not in {"the", "a", "an"}:
                return value
    return None


def _extract_split(text: str) -> Optional[str]:
    match = re.search(
        r"\b(train|training|validation|val|test|testing|development|dev)\s+split\b",
        text,
        re.I,
    )
    if match:
        return match.group(1)
    return None


def _extract_condition(text: str) -> Optional[str]:
    patterns = (
        r"\bunder\s+([^.;]{3,100})",
        r"\bat\s+(?:noise|snr|temperature|resolution|batch size|"
        r"learning rate|compression|sampling rate)\s*[^.;]{0,80}",
        r"\bwith\s+([^.;]{3,100})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()
    return None


def _extract_baseline(text: str) -> Optional[str]:
    patterns = (
        r"\b(?:baseline|baselines)\s*(?:method|model)?\s*[:=]?\s*"
        r"([A-Za-z0-9_.+\-/ ]{1,80})",
        r"\bcompared\s+(?:with|to|against)\s+"
        r"([A-Za-z0-9_.+\-/ ]{1,80})",
        r"\bversus\s+([A-Za-z0-9_.+\-/ ]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidate = match.group(1).strip(" ,.;:()[]")
            candidate = re.split(
                r"\b(?:on|with|at|achieved|and|whereas|while)\b",
                candidate,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
            if candidate:
                return candidate
    return None


def _extract_named_value(text: str, label: str) -> Optional[float]:
    match = re.search(
        rf"\b{re.escape(label)}\b\s*(?:=|:|is|was|of)?\s*({_NUMBER})"
        rf"\s*(?:%|percent|percentage)?",
        text,
        flags=re.I,
    )
    return _parse_number(match.group(1)) if match else None


def _extract_comparison_values(text: str, metric: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    proposed = None
    baseline = None

    proposed_labels = ("proposed", "ours", "our model", "our method", "method")
    baseline_labels = ("baseline", "baseline method", "baseline model", "previous", "prior")

    for label in proposed_labels:
        proposed = _extract_named_value(text, label)
        if proposed is not None:
            break
    for label in baseline_labels:
        baseline = _extract_named_value(text, label)
        if baseline is not None:
            break

    if proposed is None and baseline is None and metric:
        metric_match = re.search(
            rf"\\b{re.escape(metric)}\\b.{{0,160}}",
            text,
            re.I,
        )
        _ = metric_match

    # Explicit "X compared with Y" where numbers are adjacent to comparison.
    if proposed is None or baseline is None:
        nums = [
            _parse_number(x)
            for x in re.findall(_NUMBER, text)
        ]
        nums = [x for x in nums if x is not None]
        if len(nums) >= 2 and re.search(
            r"\b(?:compared|versus|vs\.?|baseline|outperformed|higher|lower)\b",
            text,
            re.I,
        ):
            if proposed is None:
                proposed = nums[0]
            if baseline is None:
                baseline = nums[1]

    return proposed, baseline


def _safe_round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 10)


def _absolute_improvement(proposed: float, baseline: float) -> float:
    return _safe_round(proposed - baseline)  # type: ignore[return-value]


def _relative_improvement(proposed: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return _safe_round((proposed - baseline) / abs(baseline) * 100.0)


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _format_value(value: Optional[float], unit: Optional[str]) -> str:
    if value is None:
        return ""
    suffix = unit or ""
    return f"{_format_number(value)}{suffix}"


def _has_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _has_numeric_signal(text: str) -> bool:
    return bool(re.search(_NUMBER, text))


def _has_comparison_signal(text: str) -> bool:
    return _has_any(text, (
        r"\bcompared\s+(?:with|to|against)\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\boutperform(?:ed|s)?\b",
        r"\bhigher\s+(?:than|performance|score|accuracy|f1)\b",
        r"\blower\s+(?:than|performance|score|accuracy|f1)\b",
        r"\bbetter\s+than\b",
        r"\bworse\s+than\b",
        r"\bbaseline\b",
        r"\bprior\s+(?:work|method|model)\b",
    ))


def _has_ablation_signal(text: str) -> bool:
    return _has_any(text, (
        r"\bablation\b",
        r"\bremoving\s+(?:module|component|feature|layer)\b",
        r"\bwithout\s+(?:module|component|feature|layer)\b",
        r"\bw/o\b",
        r"\bwithout\s+.+?\breduced\b",
        r"\bremove(?:d|ing)?\s+.+?\bperformance\b",
    ))


def _has_robustness_signal(text: str) -> bool:
    return _has_any(text, (
        r"\brobust(?:ness)?\b",
        r"\bnoise\b",
        r"\bperturb(?:ation|ed)?\b",
        r"\bdisturbance\b",
        r"\bcorruption\b",
        r"\bunder\s+varying\b",
        r"\bunder\s+different\s+conditions\b",
        r"\bsensitivity\b",
    ))


def _has_error_signal(text: str) -> bool:
    return _has_any(text, (
        r"\berror\s+analysis\b",
        r"\bmost\s+errors?\b",
        r"\bcommon\s+errors?\b",
        r"\bmisclassif(?:ication|ied)\b",
        r"\bfalse\s+(?:positive|negative)s?\b",
        r"\berror\s+pattern\b",
        r"\bfailure\s+cases?\b",
    ))


def _has_statistical_signal(text: str) -> bool:
    return _has_any(text, (
        r"\bp\s*[<=>]\s*",
        r"\bconfidence\s+interval\b",
        r"\b95%\s*ci\b",
        r"\bstatistically\s+significant\b",
        r"\bstandard\s+deviation\b",
        r"\bstd\.?\b",
        r"\bstandard\s+error\b",
        r"\beffect\s+size\b",
        r"\b±\b",
    ))


def _has_efficiency_signal(text: str) -> bool:
    return _has_any(text, (
        r"\blatency\b",
        r"\bthroughput\b",
        r"\binference\s+time\b",
        r"\bruntime\b",
        r"\bmemory\s+(?:usage|consumption)\b",
        r"\bparameters?\b",
        r"\bflops\b",
        r"\bgflops\b",
        r"\bcomputational\s+(?:cost|complexity)\b",
        r"\benergy\s+(?:consumption|usage)\b",
    ))


def _has_user_study_signal(text: str) -> bool:
    return _has_any(text, (
        r"\buser\s+study\b",
        r"\bparticipants?\b",
        r"\bsubjects?\b",
        r"\buser\s+evaluation\b",
        r"\bhuman\s+evaluation\b",
        r"\bquestionnaire\b",
        r"\bsurvey\b",
    ))


def _has_negative_signal(text: str) -> bool:
    return _has_any(text, (
        r"\bdid\s+not\s+(?:outperform|improve|increase|decrease)\b",
        r"\bno\s+(?:significant|measurable|observable)\s+(?:difference|improvement|change)\b",
        r"\bnot\s+statistically\s+significant\b",
        r"\bperformance\s+degrad(?:ed|es|ation)\b",
        r"\bperformance\s+decreased\b",
        r"\bfailed\s+to\s+(?:improve|outperform|increase)\b",
        r"\bno\s+measurable\s+improvement\b",
    ))


def _has_result_verb(text: str) -> bool:
    return _has_any(text, (
        r"\bachiev(?:ed|es)\b",
        r"\bobtain(?:ed|s)\b",
        r"\breport(?:ed|s)\b",
        r"\bmeasur(?:ed|es)\b",
        r"\bobserv(?:ed|es)\b",
        r"\bshow(?:ed|s)\b",
        r"\bdemonstrat(?:ed|es)\b",
        r"\bresult(?:ed|s)\b",
        r"\bimprov(?:ed|es)\b",
        r"\bdecreas(?:ed|es)\b",
        r"\bincreas(?:ed|es)\b",
        r"\bperform(?:ed|s)\b",
    ))


def _claim_strength(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"\bcauses?\b|\bcausal\b|\bproves?\b|\bproven\b", lowered):
        return "strong"
    if re.search(r"\bdemonstrates?\b|\bshows?\b|\bconfirmed?\b", lowered):
        return "strong"
    if re.search(r"\bindicates?\b|\bsuggests?\b|\bassociated\b|\bcorrelat", lowered):
        return "qualified"
    return "reported"


# ---------------------------------------------------------------------------
# Structured evidence extraction from already-parsed analysis outputs
# ---------------------------------------------------------------------------

def _structured_findings_from_component(component: Any, paper_id: str, document_id: Optional[str]) -> list[EvidenceRecord]:
    if component is None:
        return []

    result: list[EvidenceRecord] = []

    if isinstance(component, Mapping):
        for key in (
            "findings",
            "results",
            "key_findings",
            "observations",
            "evidence",
            "supporting_evidence",
            "experimental_results",
        ):
            value = component.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                values = value
            else:
                values = [value]
            for item in values:
                record = _normalize_evidence(item, paper_id, document_id)
                if record:
                    result.append(record)

    elif isinstance(component, (list, tuple)):
        for item in component:
            record = _normalize_evidence(item, paper_id, document_id)
            if record:
                result.append(record)

    return result


def _build_evidence_pool(
    paper: Any,
    evidence: Any,
    *,
    paper_id: str,
    document_id: Optional[str],
) -> list[EvidenceRecord]:
    pool = _extract_evidence(
        paper,
        evidence,
        paper_id=paper_id,
        document_id=document_id,
    )

    for component_name in (
        "findings",
        "results",
        "analysis",
        "rag_context",
        "context",
    ):
        component = _read(paper, component_name, None)
        pool.extend(
            _structured_findings_from_component(
                component,
                paper_id,
                document_id,
            )
        )

    return _dedupe_evidence(pool)


# ---------------------------------------------------------------------------
# Candidate finding extraction
# ---------------------------------------------------------------------------

def _candidate_type(text: str) -> Optional[FindingType]:
    if not _has_result_verb(text) and not _has_numeric_signal(text):
        # Explicit error/statistical/qualitative observations may not use
        # standard result verbs.
        if not (
            _has_error_signal(text)
            or _has_negative_signal(text)
            or _has_statistical_signal(text)
            or _has_any(text, (
                r"\bassociated\s+with\b",
                r"\bcorrelat(?:ed|es|ion)\b",
                r"\brelated\s+to\b",
            ))
        ):
            return None

    if _has_ablation_signal(text):
        return FindingType.ABLATION
    if _has_error_signal(text):
        return FindingType.ERROR_ANALYSIS
    if _has_user_study_signal(text):
        return FindingType.USER_STUDY
    if _has_statistical_signal(text):
        return FindingType.STATISTICAL
    if _has_robustness_signal(text):
        return FindingType.ROBUSTNESS
    if _has_efficiency_signal(text):
        return FindingType.EFFICIENCY
    if _has_comparison_signal(text):
        return FindingType.COMPARATIVE
    if _has_numeric_signal(text):
        metric = _metric_from_text(text)
        if metric:
            return FindingType.QUANTITATIVE
    if _has_negative_signal(text):
        return FindingType.NEGATIVE
    return FindingType.QUALITATIVE


def _confidence_for(evidence: Sequence[EvidenceRecord], derived: bool = False) -> Confidence:
    if derived:
        return Confidence.MEDIUM if evidence else Confidence.LOW
    if not evidence:
        return Confidence.LOW
    if any(x.evidence_type == EvidenceType.AUTHOR_CLAIM for x in evidence):
        return Confidence.MEDIUM
    if any(
        x.page is not None
        or x.table is not None
        or x.figure is not None
        for x in evidence
    ):
        return Confidence.HIGH
    return Confidence.HIGH


def _source_locations(evidence: Sequence[EvidenceRecord]) -> list[SourceLocation]:
    result: list[SourceLocation] = []
    seen: set[tuple[Any, ...]] = set()
    for item in evidence:
        loc = item.location()
        key = tuple(sorted(loc.to_dict().items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(loc)
    return result


def _finding_id(paper_id: str, finding_type: str, statement: str) -> str:
    raw = f"{paper_id}|{finding_type}|{statement.casefold().strip()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"finding_{digest}"


def _statement_for_quantitative(
    text: str,
    metric: Optional[str],
    value: Optional[float],
    unit: Optional[str],
    dataset: Optional[str],
    split: Optional[str],
) -> str:
    if metric and value is not None:
        statement = f"The study reports {metric} = {_format_value(value, unit)}"
        if dataset:
            statement += f" on {dataset}"
        if split:
            statement += f" {split} split"
        return statement + "."
    return text.strip()


def _comparison_statement(
    text: str,
    metric: Optional[str],
    proposed: Optional[float],
    baseline: Optional[float],
    unit: Optional[str],
) -> tuple[str, Optional[float], Optional[float]]:
    if proposed is not None and baseline is not None and metric:
        absolute = _absolute_improvement(proposed, baseline)
        statement = (
            f"The proposed method reports {metric} = "
            f"{_format_value(proposed, unit)} compared with "
            f"{_format_value(baseline, unit)} for the reported baseline."
        )
        return statement, absolute, _relative_improvement(proposed, baseline)

    # Preserve qualitative comparison exactly at the evidence level.
    return text.strip(), None, None


def _ablation_statement(text: str, metric: Optional[str]) -> str:
    # Explicit numbers are preferred.
    numbers = [_parse_number(x) for x in re.findall(_NUMBER, text)]
    numbers = [x for x in numbers if x is not None]
    if metric and len(numbers) >= 2 and re.search(
        r"\b(?:removing|without|w/o)\b", text, re.I
    ):
        return (
            f"Removing the reported component changed {metric} from "
            f"{_format_number(numbers[0])} to {_format_number(numbers[1])}."
        )
    return text.strip()


def _preserve_claim_strength(text: str) -> str:
    # Do not rewrite qualified scientific claims into stronger causal claims.
    return text.strip()


def _is_defensible_finding(text: str, evidence: Sequence[EvidenceRecord]) -> bool:
    if not text.strip() or not evidence:
        return False
    if all(x.evidence_type == EvidenceType.AUTHOR_CLAIM for x in evidence):
        # Author claims are not rejected universally; they can be preserved
        # as findings only when they explicitly describe an observed result.
        return bool(
            _has_numeric_signal(text)
            or _has_any(text, (
                r"\bresults?\b",
                r"\bexperiments?\b",
                r"\bevaluation\b",
                r"\bobserved\b",
                r"\bmeasured\b",
            ))
        )
    return True


def _make_candidate(
    evidence: EvidenceRecord,
    paper_id: str,
) -> Optional[FindingItem]:
    text = re.sub(r"\s+", " ", evidence.text).strip()
    if not _is_defensible_finding(text, [evidence]):
        return None

    finding_type = _candidate_type(text)
    if finding_type is None:
        return None

    metric = _metric_from_text(text)
    unit = _unit_from_text(text, metric)
    value = _parse_metric_value(text, metric)
    dataset = _extract_dataset(text)
    split = _extract_split(text)
    condition = _extract_condition(text)
    baseline_name = _extract_baseline(text)

    proposed, baseline = _extract_comparison_values(text, metric)

    absolute = None
    relative = None
    derived_fields: list[str] = []

    statement = _preserve_claim_strength(text)

    if finding_type == FindingType.COMPARATIVE:
        statement, absolute, relative = _comparison_statement(
            text,
            metric,
            proposed,
            baseline,
            unit,
        )
        if absolute is not None:
            derived_fields.append("absolute_improvement")
        if relative is not None:
            derived_fields.append("relative_improvement_percent")

    elif finding_type == FindingType.ABLATION:
        statement = _ablation_statement(text, metric)

    elif finding_type == FindingType.QUANTITATIVE and metric and value is not None:
        statement = _statement_for_quantitative(
            text,
            metric,
            value,
            unit,
            dataset,
            split,
        )

    if _has_negative_signal(text):
        if finding_type in {
            FindingType.QUANTITATIVE,
            FindingType.COMPARATIVE,
        }:
            pass
        else:
            finding_type = FindingType.NEGATIVE

    evidence_type = evidence.evidence_type.value
    confidence = _confidence_for([evidence], derived=bool(derived_fields))

    return FindingItem(
        finding_id=_finding_id(paper_id, finding_type.value, statement),
        paper_id=paper_id,
        finding_type=finding_type.value,
        statement=statement,
        evidence=[evidence],
        evidence_type=evidence_type,
        confidence=confidence.value,
        metric=metric,
        value=value,
        unit=unit,
        dataset=dataset,
        split=split,
        baseline=baseline_name,
        baseline_value=baseline,
        proposed_value=proposed,
        comparison=(
            "proposed_vs_baseline"
            if proposed is not None and baseline is not None
            else None
        ),
        absolute_improvement=absolute,
        relative_improvement_percent=relative,
        derived_fields=derived_fields,
        experimental_condition=condition,
        source_locations=_source_locations([evidence]),
    )


def _extract_explicit_structured_finding(
    item: Any,
    paper_id: str,
    document_id: Optional[str],
) -> Optional[FindingItem]:
    if not isinstance(item, Mapping) and not is_dataclass(item):
        return None

    statement = _first_string(item, ("statement", "finding", "text", "description"))
    if not statement:
        return None

    evidence_raw = _read(item, "evidence", None)
    evidence: list[EvidenceRecord] = []
    if evidence_raw is not None:
        values = evidence_raw if isinstance(evidence_raw, (list, tuple)) else [evidence_raw]
        for raw in values:
            normalized = _normalize_evidence(raw, paper_id, document_id)
            if normalized:
                evidence.append(normalized)

    if not evidence:
        # The structured finding itself can serve as supplied evidence only
        # when it contains a source/provenance signal.
        has_source = any(
            _read(item, key, None) is not None
            for key in ("source", "page", "section", "chunk_id", "evidence_id")
        )
        if has_source:
            normalized = _normalize_evidence(
                {
                    "text": statement,
                    "source": _read(item, "source", "structured_analysis"),
                    "page": _read(item, "page", None),
                    "section": _read(item, "section", None),
                    "chunk_id": _read(item, "chunk_id", None),
                    "evidence_id": _read(item, "evidence_id", None),
                },
                paper_id,
                document_id,
            )
            if normalized:
                evidence.append(normalized)

    if not evidence:
        return None

    raw_type = _first_string(item, ("type", "finding_type", "category"))
    finding_type = (
        raw_type.lower()
        if raw_type
        else (_candidate_type(statement) or FindingType.QUALITATIVE).value
    )

    metric = _optional_string(_read(item, "metric", None))
    value = _parse_number(_read(item, "value", None))
    unit = _optional_string(_read(item, "unit", None))
    dataset = _optional_string(_read(item, "dataset", None))
    split = _optional_string(_read(item, "split", None))
    baseline = _optional_string(_read(item, "baseline", None))
    proposed = _parse_number(_read(item, "proposed_value", None))
    baseline_value = _parse_number(_read(item, "baseline_value", None))

    absolute = _parse_number(_read(item, "absolute_improvement", None))
    relative = _parse_number(_read(item, "relative_improvement_percent", None))
    derived_fields = list(_string_tuple(_read(item, "derived_fields", None)))

    # If explicit numeric values exist, validate the derived values instead of
    # trusting an ungrounded number.
    if proposed is not None and baseline_value is not None:
        expected_abs = _absolute_improvement(proposed, baseline_value)
        if absolute is not None and not math.isclose(absolute, expected_abs, rel_tol=1e-9, abs_tol=1e-9):
            raise FindingsEvidenceError(
                "Structured finding contains an inconsistent absolute improvement."
            )
        absolute = expected_abs
        if "absolute_improvement" not in derived_fields:
            derived_fields.append("absolute_improvement")

        expected_rel = _relative_improvement(proposed, baseline_value)
        if relative is not None and expected_rel is not None and not math.isclose(
            relative,
            expected_rel,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise FindingsEvidenceError(
                "Structured finding contains an inconsistent relative improvement."
            )
        if expected_rel is not None:
            relative = expected_rel
            if "relative_improvement_percent" not in derived_fields:
                derived_fields.append("relative_improvement_percent")

    confidence = _confidence_for(evidence, derived=bool(derived_fields))

    return FindingItem(
        finding_id=_finding_id(paper_id, finding_type, statement),
        paper_id=paper_id,
        finding_type=finding_type,
        statement=statement.strip(),
        evidence=evidence,
        evidence_type=(
            _first_string(item, ("evidence_type",))
            or evidence[0].evidence_type.value
        ),
        confidence=(
            _first_string(item, ("confidence",))
            or confidence.value
        ),
        metric=metric,
        value=value,
        unit=unit,
        dataset=dataset,
        split=split,
        baseline=baseline,
        baseline_value=baseline_value,
        proposed_value=proposed,
        comparison=_optional_string(_read(item, "comparison", None)),
        absolute_improvement=absolute,
        relative_improvement_percent=relative,
        derived_fields=derived_fields,
        experimental_condition=_optional_string(
            _read(item, "experimental_condition", None)
        ),
        source_locations=_source_locations(evidence),
    )


# ---------------------------------------------------------------------------
# Deduplication and validation
# ---------------------------------------------------------------------------

def _finding_key(item: FindingItem) -> tuple[Any, ...]:
    normalized_statement = re.sub(
        r"\s+",
        " ",
        item.statement.casefold().strip(),
    )
    return (
        item.paper_id,
        item.finding_type,
        item.metric.casefold() if item.metric else None,
        item.dataset.casefold() if item.dataset else None,
        item.value,
        item.proposed_value,
        item.baseline_value,
        normalized_statement,
    )


def _merge_evidence(a: FindingItem, b: FindingItem) -> FindingItem:
    merged = list(a.evidence)
    existing = {
        (
            x.text.casefold(),
            x.page,
            x.section,
            x.table,
            x.figure,
        )
        for x in merged
    }
    for item in b.evidence:
        key = (
            item.text.casefold(),
            item.page,
            item.section,
            item.table,
            item.figure,
        )
        if key not in existing:
            merged.append(item)
            existing.add(key)

    a.evidence = merged
    a.source_locations = _source_locations(merged)
    if a.confidence == Confidence.LOW.value and b.confidence != Confidence.LOW.value:
        a.confidence = b.confidence
    return a


def _deduplicate_findings(items: Sequence[FindingItem]) -> list[FindingItem]:
    result: list[FindingItem] = []
    index: dict[tuple[Any, ...], int] = {}

    for item in items:
        key = _finding_key(item)
        if key in index:
            result[index[key]] = _merge_evidence(result[index[key]], item)
            continue

        # A conservative secondary duplicate rule for repeated identical
        # quantitative observations expressed with slightly different text.
        secondary = (
            item.paper_id,
            item.finding_type,
            item.metric,
            item.dataset,
            item.value,
            item.proposed_value,
            item.baseline_value,
        )
        if (
            item.metric is not None
            and any(
                x.paper_id == item.paper_id
                and x.finding_type == item.finding_type
                and x.metric == item.metric
                and x.dataset == item.dataset
                and x.value == item.value
                and x.proposed_value == item.proposed_value
                and x.baseline_value == item.baseline_value
                for x in result
            )
        ):
            for existing in result:
                if (
                    existing.paper_id == item.paper_id
                    and existing.finding_type == item.finding_type
                    and existing.metric == item.metric
                    and existing.dataset == item.dataset
                    and existing.value == item.value
                    and existing.proposed_value == item.proposed_value
                    and existing.baseline_value == item.baseline_value
                ):
                    _merge_evidence(existing, item)
                    break
            continue

        index[key] = len(result)
        result.append(item)

    return result


def _validate_finding(item: FindingItem, paper_id: str) -> None:
    if item.paper_id != paper_id:
        raise FindingsEvidenceError("Finding paper identity does not match input paper.")
    if not item.statement.strip():
        raise FindingsEvidenceError("Finding statement is empty.")
    if not item.evidence:
        raise FindingsEvidenceError("Finding has no evidence.")

    for evidence in item.evidence:
        if evidence.paper_id and evidence.paper_id != paper_id:
            raise FindingsEvidenceError("Evidence from another paper was attached.")
        if not evidence.text.strip():
            raise FindingsEvidenceError("Finding evidence contains empty text.")

    if item.proposed_value is not None and item.baseline_value is not None:
        expected_abs = _absolute_improvement(
            item.proposed_value,
            item.baseline_value,
        )
        if item.absolute_improvement is not None and not math.isclose(
            item.absolute_improvement,
            expected_abs,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise FindingsEvidenceError("Absolute improvement is numerically inconsistent.")

        if item.relative_improvement_percent is not None:
            expected_rel = _relative_improvement(
                item.proposed_value,
                item.baseline_value,
            )
            if expected_rel is not None and not math.isclose(
                item.relative_improvement_percent,
                expected_rel,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise FindingsEvidenceError("Relative improvement is numerically inconsistent.")

    # A finding cannot claim external comparison merely because its text
    # contains generic quality language.
    if re.search(
        r"\bstate[- ]of[- ]the[- ]art\b|\buniversally\s+superior\b|\bperfect(?:ly)?\s+general",
        item.statement,
        re.I,
    ):
        raise FindingsEvidenceError(
            "Unsupported quality/generalization claim detected in finding."
        )


def _rank_findings(items: Sequence[FindingItem]) -> list[FindingItem]:
    def score(item: FindingItem) -> tuple[int, int, int, str]:
        evidence_score = {
            EvidenceType.REPORTED_RESULT.value: 4,
            EvidenceType.EXPERIMENTALLY_OBSERVED.value: 4,
            EvidenceType.DERIVED_RESULT.value: 3,
            EvidenceType.AUTHOR_CLAIM.value: 1,
        }.get(item.evidence_type, 1)

        type_score = {
            FindingType.COMPARATIVE.value: 8,
            FindingType.ABLATION.value: 8,
            FindingType.STATISTICAL.value: 7,
            FindingType.ROBUSTNESS.value: 7,
            FindingType.QUANTITATIVE.value: 6,
            FindingType.ERROR_ANALYSIS.value: 6,
            FindingType.EFFICIENCY.value: 5,
            FindingType.COMPUTATIONAL.value: 5,
            FindingType.NEGATIVE.value: 5,
            FindingType.QUALITATIVE.value: 3,
            FindingType.OBSERVATION.value: 3,
        }.get(item.finding_type, 2)

        provenance_score = sum(
            1
            for e in item.evidence
            if e.page is not None or e.section or e.table or e.figure
        )
        return (
            evidence_score,
            type_score,
            provenance_score,
            item.finding_id,
        )

    return sorted(items, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------

class FindingsAnalyzer:
    """
    Evidence-grounded findings analyzer.

    It accepts an existing paper object and/or already available evidence.
    It never performs retrieval, embedding, PDF processing, or LLM calls.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict

    def analyze(
        self,
        paper: Any,
        *,
        evidence: Any = None,
    ) -> FindingsAnalysisResult:
        started = time.perf_counter()

        paper_id, document_id = _normalize_identity(paper)
        if not paper_id:
            raise FindingsInputError("paper_id/document_id is required.")

        pool = _build_evidence_pool(
            paper,
            evidence,
            paper_id=paper_id,
            document_id=document_id,
        )

        candidates: list[FindingItem] = []

        # First preserve structured findings already produced upstream.
        structured_sources: list[Any] = []
        for key in ("findings", "results", "analysis"):
            value = _read(paper, key, None)
            if value is not None:
                structured_sources.append(value)
        if isinstance(evidence, Mapping):
            for key in ("findings", "results"):
                if key in evidence:
                    structured_sources.append(evidence[key])

        for source in structured_sources:
            values = source if isinstance(source, (list, tuple)) else [source]
            for value in values:
                item = _extract_explicit_structured_finding(
                    value,
                    paper_id,
                    document_id,
                )
                if item:
                    candidates.append(item)

        # Then extract atomic findings from evidence snippets.
        for record in pool:
            metric_values = _extract_all_metric_values(record.text)

            # Preserve each explicitly reported metric as an atomic finding.
            # This prevents Accuracy/Precision/Recall/F1 from being collapsed
            # into one opaque result.
            if len(metric_values) > 1:
                for metric_name, metric_value, metric_unit in metric_values:
                    metric_text = (
                        f"{metric_name} = {_format_value(metric_value, metric_unit)}"
                    )
                    dataset = _extract_dataset(record.text)
                    split = _extract_split(record.text)
                    statement = _statement_for_quantitative(
                        record.text,
                        metric_name,
                        metric_value,
                        metric_unit,
                        dataset,
                        split,
                    )
                    item = FindingItem(
                        finding_id=_finding_id(
                            paper_id,
                            FindingType.QUANTITATIVE.value,
                            statement,
                        ),
                        paper_id=paper_id,
                        finding_type=FindingType.QUANTITATIVE.value,
                        statement=statement,
                        evidence=[record],
                        evidence_type=record.evidence_type.value,
                        confidence=_confidence_for([record]).value,
                        metric=metric_name,
                        value=metric_value,
                        unit=metric_unit,
                        dataset=dataset,
                        split=split,
                        experimental_condition=_extract_condition(record.text),
                        source_locations=_source_locations([record]),
                    )
                    candidates.append(item)
                continue

            item = _make_candidate(record, paper_id)
            if item:
                candidates.append(item)

        candidates = _deduplicate_findings(candidates)

        validated: list[FindingItem] = []
        errors: list[str] = []

        for item in candidates:
            try:
                _validate_finding(item, paper_id)
                validated.append(item)
            except FindingsEvidenceError as exc:
                if self.strict:
                    errors.append(str(exc))
                else:
                    logger.warning("Discarding invalid finding: %s", exc)

        # Never expose an invalid finding. A strict run can still return
        # valid findings plus errors so the orchestrator can isolate failure.
        findings = _rank_findings(validated)

        status = "success"
        if not findings and pool:
            status = "insufficient_evidence"
        elif not pool:
            status = "insufficient_evidence"
        elif errors and findings:
            status = "partial"
        elif errors and not findings:
            status = "failed"

        elapsed = time.perf_counter() - started

        logger.info(
            "Findings analysis completed: paper_id=%s finding_count=%d "
            "evidence_count=%d status=%s",
            paper_id,
            len(findings),
            len(pool),
            status,
        )

        return FindingsAnalysisResult(
            paper_id=paper_id,
            findings=findings,
            status=status,
            evidence_count=len(pool),
            processing_time=elapsed,
            errors=errors,
        )

    # Compatibility aliases used by common project orchestrators.
    def analyze_paper(self, paper: Any, *, evidence: Any = None) -> FindingsAnalysisResult:
        return self.analyze(paper, evidence=evidence)

    def __call__(self, paper: Any, *, evidence: Any = None) -> FindingsAnalysisResult:
        return self.analyze(paper, evidence=evidence)


# Common alias for dependency injection.
ResearchFindingsAnalyzer = FindingsAnalyzer


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def analyze_findings(
    paper: Any,
    *,
    evidence: Any = None,
    strict: bool = True,
) -> FindingsAnalysisResult:
    return FindingsAnalyzer(strict=strict).analyze(
        paper,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Self-test suite
# ---------------------------------------------------------------------------

def _ev(text: str, *, page: int | None = None, section: str | None = "Results") -> dict[str, Any]:
    return {
        "text": text,
        "source": "mock",
        "paper_id": "paper_test",
        "page": page,
        "section": section,
    }


def run_self_test() -> None:
    analyzer = FindingsAnalyzer()

    # 1. Basic quantitative finding.
    result = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("The proposed model achieved 94.2% accuracy on Dataset A test split.", page=8)],
    )
    assert result.status == "success"
    assert result.findings
    f = result.findings[0]
    assert f.metric == "accuracy"
    assert math.isclose(f.value or 0, 94.2)
    assert f.unit == "%"
    assert f.dataset == "Dataset A"
    assert f.split == "test"

    # 2. Multiple metrics must remain atomic.
    multi = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev(
            "The model achieved Accuracy = 94.2%, Precision = 92.8%, "
            "Recall = 93.7%, and F1 = 93.2%."
        )],
    )
    metrics = {x.metric for x in multi.findings}
    assert {"accuracy", "precision", "recall", "F1"} <= metrics

    # 3. Baseline + percentage points.
    comparison = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev(
            "Proposed Accuracy = 94.2% compared with Baseline = 91.8%.",
            page=8,
        )],
    )
    assert any(
        math.isclose(x.absolute_improvement or 0, 2.4)
        for x in comparison.findings
    )
    comp = next(x for x in comparison.findings if x.absolute_improvement is not None)
    assert math.isclose(comp.relative_improvement_percent or 0, (2.4 / 91.8) * 100)

    # 4. Qualitative improvement must not invent a number.
    qualitative = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("The proposed method improved performance over the baseline.")],
    )
    assert qualitative.findings
    assert not any(x.absolute_improvement is not None for x in qualitative.findings)

    # 5. Negative result.
    negative = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("The proposed model did not outperform the baseline.")],
    )
    assert negative.findings
    assert any("did not outperform" in x.statement.casefold() for x in negative.findings)

    # 6. Ablation.
    ablation = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev(
            "Ablation study: Full model F1 = 0.91; without module A F1 = 0.86."
        )],
    )
    assert any(x.finding_type == "ablation" for x in ablation.findings)
    assert not any("essential" in x.statement.casefold() for x in ablation.findings)

    # 7. Causality preservation.
    causal = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("X was associated with Y.")],
    )
    assert causal.findings
    assert not any("caused" in x.statement.casefold() for x in causal.findings)

    # 8. Claim strength preservation.
    claim = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("The results suggest improved robustness.")],
    )
    assert claim.findings
    assert any("suggest" in x.statement.casefold() for x in claim.findings)

    # 9. Statistical result.
    stats = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("Method A achieved F1 = 0.91 ± 0.02; p < 0.05.")],
    )
    assert any(
        x.finding_type == "statistical" or x.metric == "F1"
        for x in stats.findings
    )

    # 10. Error analysis.
    errors = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("Most errors occurred for class B.")],
    )
    assert any(x.finding_type == "error_analysis" for x in errors.findings)

    # 11. Robustness.
    robust = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev(
            "Performance was evaluated under Gaussian noise levels of σ=0.1, 0.2 and 0.3."
        )],
    )
    assert any(x.finding_type == "robustness" for x in robust.findings)

    # 12. Efficiency.
    efficiency = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("Inference latency was 12 ms with throughput of 80 FPS.")],
    )
    assert efficiency.findings

    # 13. Provenance.
    prov = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("Accuracy = 94.2% in Table 2.", page=8)],
    )
    assert prov.findings
    assert any(
        loc.page == 8
        for item in prov.findings
        for loc in item.source_locations
    )

    # 14. Multiple papers must be isolated.
    a = analyzer.analyze(
        {"paper_id": "paper_A"},
        evidence=[{
            "text": "Accuracy = 90%",
            "paper_id": "paper_A",
        }],
    )
    b = analyzer.analyze(
        {"paper_id": "paper_B"},
        evidence=[{
            "text": "Accuracy = 80%",
            "paper_id": "paper_B",
        }],
    )
    assert all(x.paper_id == "paper_A" for x in a.findings)
    assert all(x.paper_id == "paper_B" for x in b.findings)
    assert not any(
        x.value == 80
        for x in a.findings
    )

    # 15. Wrong-paper evidence is rejected/ignored.
    wrong = analyzer.analyze(
        {"paper_id": "paper_A"},
        evidence=[{
            "text": "Accuracy = 80%",
            "paper_id": "paper_B",
        }],
    )
    assert wrong.status == "insufficient_evidence"
    assert not wrong.findings

    # 16. No hallucinated numbers.
    no_number = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("The proposed method improved accuracy.")],
    )
    assert no_number.findings
    assert all(x.value is None for x in no_number.findings)

    # 17. Derived structured value validation.
    structured = analyzer.analyze(
        {
            "paper_id": "paper_test",
            "findings": [{
                "type": "comparative",
                "statement": "The proposed model achieved 94.2% accuracy compared with 91.8% for the baseline.",
                "metric": "accuracy",
                "proposed_value": 94.2,
                "baseline_value": 91.8,
                "evidence": [_ev(
                    "The proposed model achieved 94.2% accuracy compared with 91.8% for the baseline."
                )],
            }],
        }
    )
    sf = structured.findings[0]
    assert math.isclose(sf.absolute_improvement or 0, 2.4)
    assert "absolute_improvement" in sf.derived_fields

    # 18. Duplicate chunks are merged, not multiplied.
    dup = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[
            _ev("Accuracy = 94.2%", page=8),
            _ev("Accuracy = 94.2%", page=8),
            _ev("Accuracy = 94.2%", page=9),
        ],
    )
    assert len(dup.findings) == 1
    assert len(dup.findings[0].evidence) == 2

    # 19. No quality judgment.
    quality = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("This is a powerful model.")],
    )
    assert not quality.findings

    # 20. Determinism.
    d1 = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("F1 = 0.91 on Dataset A.")],
    ).to_dict()
    d2 = analyzer.analyze(
        {"paper_id": "paper_test"},
        evidence=[_ev("F1 = 0.91 on Dataset A.")],
    ).to_dict()
    # Ignore latency because it is intentionally runtime-dependent.
    d1["processing_time"] = 0
    d2["processing_time"] = 0
    assert d1["findings"] == d2["findings"]

    print("FindingsAnalyzer self-test: PASSED")


if __name__ == "__main__":
    run_self_test()