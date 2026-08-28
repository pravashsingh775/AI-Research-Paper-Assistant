"""
Final response quality gate for the AI Research Paper Assistant.

Responsibility boundary
-----------------------
`evidence.py` validates individual claims against supplied evidence.
This module validates the *complete structured response*: structure,
provenance, claim coverage, cross-section consistency, cross-paper
isolation, citations, and final safety policy.

This module deliberately performs no retrieval, embedding, PDF parsing,
LLM calls, web access, or dataset loading.
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

try:
    from .evidence import (
        EvidenceItem,
        EvidenceValidator,
        SupportStatus,
        Confidence,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from evidence import (
        EvidenceItem,
        EvidenceValidator,
        SupportStatus,
        Confidence,
    )

logger = logging.getLogger(__name__)


class FinalValidationStatus(str, Enum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"
    path: Optional[str] = None
    paper_id: Optional[str] = None
    claim: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimDecision:
    claim: str
    claim_type: str
    status: str
    confidence: str
    supported: bool
    critical: bool
    reason: str
    paper_id: Optional[str] = None
    document_id: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()
    path: Optional[str] = None


@dataclass(frozen=True)
class FinalValidationResult:
    valid: bool
    status: FinalValidationStatus
    confidence: str
    validated_content: Any
    rejected_content: tuple[dict[str, Any], ...]
    warnings: tuple[ValidationIssue, ...]
    errors: tuple[ValidationIssue, ...]
    claims_checked: int
    claims_supported: int
    claims_unsupported: int
    claims_contradicted: int
    evidence_coverage: float
    paper_integrity: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_decisions: tuple[ClaimDecision, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CRITICAL_FIELDS = {
    "model", "dataset", "findings", "methodology", "answer",
}
_ANALYTICAL_FIELDS = {"strengths", "weaknesses"}
_CONTAINER_FIELDS = {
    "summary", "methodology", "model", "dataset", "findings",
    "strengths", "weaknesses", "answer",
}
_NON_FACTUAL_RE = re.compile(
    r"^(?:here(?:'s| is)|below|in summary|overall|note that|"
    r"the following|this section|based on the (?:paper|evidence))\b",
    re.I,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*%|"
    r"\s*(?:ms|s|m|gb|mb|k|m|b))?(?![\w.])",
    re.I,
)
_METRIC_RE = re.compile(
    r"\b(accuracy|precision|recall|f1(?:[-\s]?score)?|mAP|auc|"
    r"bleu|rouge|rmse|mae|mse|latency|parameters?|params?)\b",
    re.I,
)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            k: v for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("text", "claim", "answer", "content", "value", "name"):
            if key in value and isinstance(value[key], (str, int, float)):
                return str(value[key]).strip()
        return " ".join(_text(v) for v in value.values() if v is not None).strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_text(v) for v in value).strip()
    return str(value).strip()


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _iter_evidence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "evidence" in value:
            return _iter_evidence(value["evidence"])
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _claim_is_factual(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    if _NON_FACTUAL_RE.search(text.strip()):
        return False
    return bool(
        _NUMBER_RE.search(text)
        or _METRIC_RE.search(text)
        or re.search(
            r"\b(?:uses?|utilizes?|trained|evaluated|tested|achieves?|"
            r"contains?|consists?|proposes?|reports?|dataset|model|"
            r"method|architecture|experiment|finding|result|limitation|"
            r"optimizer|adam|sgd|outperforms?|compared|accuracy|f1|precision|recall)\b",
            text,
            re.I,
        )
    )


def _claim_type(path: str) -> str:
    leaf = path.split(".")[0].lower()
    if leaf in {"answer", "qa"}:
        return "qa"
    if leaf in _ANALYTICAL_FIELDS:
        return leaf
    if leaf in _CONTAINER_FIELDS:
        return leaf
    return "general"


def _critical(path: str, claim: str) -> bool:
    leaf = path.split(".")[0].lower()
    if leaf in _CRITICAL_FIELDS:
        return True
    return bool(
        re.search(
            r"\b(?:accuracy|precision|recall|f1|mAP|auc|dataset|"
            r"model|baseline|finding|result|outperform)\b",
            claim,
            re.I,
        )
    )


def _walk_claims(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, str):
        if _claim_is_factual(value):
            yield path or "response", value.strip()
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            if key_s in {
                "citations", "evidence", "validation", "metadata",
                "source_metadata", "provenance", "confidence",
            }:
                continue
            child_path = f"{path}.{key_s}" if path else key_s
            yield from _walk_claims(child, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for i, child in enumerate(value):
            yield from _walk_claims(child, f"{path}[{i}]")
        return
    if isinstance(value, (int, float, bool)):
        if path:
            yield path, str(value)


def _extract_paper_ids(evidence: Iterable[Any]) -> set[str]:
    ids: set[str] = set()
    for item in evidence:
        provenance = _get(item, "provenance")
        paper = _get(provenance, "paper_id") or _get(item, "paper_id")
        if paper:
            ids.add(str(paper))
    return ids


def _extract_document_ids(evidence: Iterable[Any]) -> set[str]:
    ids: set[str] = set()
    for item in evidence:
        provenance = _get(item, "provenance")
        doc = _get(provenance, "document_id") or _get(item, "document_id")
        if doc:
            ids.add(str(doc))
    return ids


def _identity(value: Any) -> tuple[Optional[str], Optional[str]]:
    return (
        str(_get(value, "paper_id")) if _get(value, "paper_id") is not None else None,
        str(_get(value, "document_id")) if _get(value, "document_id") is not None else None,
    )


def _evidence_from_result(result: Any) -> list[Any]:
    return list(_get(result, "evidence", ()) or ())


def _evidence_id(item: Any) -> Optional[str]:
    value = _get(item, "evidence_id")
    return str(value) if value is not None else None


def _normalize_number(value: str) -> Optional[float]:
    raw = value.replace(",", "").strip().lower()
    raw = re.sub(r"\s+", "", raw)
    multiplier = 1.0
    if raw.endswith("k"):
        multiplier = 1_000.0
        raw = raw[:-1]
    elif raw.endswith("m"):
        multiplier = 1_000_000.0
        raw = raw[:-1]
    elif raw.endswith("b"):
        multiplier = 1_000_000_000.0
        raw = raw[:-1]
    raw = raw.rstrip("%")
    try:
        return float(raw) * multiplier
    except ValueError:
        return None


def _numeric_signature(text: str) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    numbers = list(_NUMBER_RE.finditer(text))
    for number in numbers:
        value = _normalize_number(number.group(0))
        if value is None:
            continue
        preceding = text[max(0, number.start() - 80):number.start()]
        metric_match = list(_METRIC_RE.finditer(preceding))
        metric = metric_match[-1].group(1).lower() if metric_match else "number"
        pairs.append((metric, value))
    return tuple(pairs)


def _flatten_evidence(response: Any) -> list[Any]:
    raw = _get(response, "evidence")
    if raw is None:
        raw = _get(response, "validation")
    if isinstance(raw, Mapping):
        for key in ("results", "evidence", "items"):
            if key in raw:
                raw = raw[key]
                break
    return _iter_evidence(raw)


class ResponseValidator:
    """Deterministic final quality gate."""

    def __init__(self, evidence_validator: Optional[EvidenceValidator] = None):
        self.evidence_validator = evidence_validator or EvidenceValidator()

    def validate(
        self,
        response: Any,
        *,
        evidence: Optional[Sequence[Any]] = None,
        mode: Optional[str] = None,
        expected_paper_id: Optional[str] = None,
        expected_document_id: Optional[str] = None,
    ) -> FinalValidationResult:
        started = time.perf_counter()
        warnings: list[ValidationIssue] = []
        errors: list[ValidationIssue] = []
        decisions: list[ClaimDecision] = []

        payload = _plain(response)
        if not isinstance(payload, Mapping):
            errors.append(ValidationIssue(
                "malformed_response",
                "Final response must be a mapping/object.",
                "error",
            ))
            return self._build(
                payload, (), warnings, errors, decisions, started,
                paper_integrity=False,
            )

        if not payload:
            errors.append(ValidationIssue(
                "empty_response", "Response is empty.", "error",
            ))

        mode_value = (mode or _get(payload, "mode") or "").lower() or None
        response_paper, response_document = _identity(payload)
        expected_paper_id = expected_paper_id or response_paper
        expected_document_id = expected_document_id or response_document

        if expected_paper_id is None and self._contains_factual_claims(payload):
            errors.append(ValidationIssue(
                "missing_paper_id",
                "Paper identity is missing for a paper-grounded response.",
                "error",
            ))

        supplied_evidence = list(evidence) if evidence is not None else _flatten_evidence(payload)
        paper_ids = _extract_paper_ids(supplied_evidence)
        document_ids = _extract_document_ids(supplied_evidence)

        paper_integrity = True

        if expected_paper_id and paper_ids and paper_ids != {expected_paper_id}:
            paper_integrity = False
            errors.append(ValidationIssue(
                "cross_paper_contamination",
                f"Evidence is associated with papers {sorted(paper_ids)!r}, "
                f"but the response targets {expected_paper_id!r}.",
                "error",
                paper_id=expected_paper_id,
            ))

        if expected_document_id and document_ids and expected_document_id not in document_ids:
            paper_integrity = False
            errors.append(ValidationIssue(
                "wrong_document_id",
                f"Evidence does not belong to document {expected_document_id!r}.",
                "error",
                paper_id=expected_paper_id,
            ))

        if mode_value in {"mode1", "research_discovery", "discovery"}:
            self._validate_top10(payload, warnings, errors)
        elif mode_value in {"mode2", "my_paper", "uploaded_paper", "qa"}:
            self._validate_uploaded_context(payload, warnings, errors)

        self._validate_citations(payload, supplied_evidence, expected_paper_id, warnings, errors)
        self._validate_internal_consistency(payload, warnings, errors)

        claims = list(_walk_claims(payload))
        for path, claim in claims:
            ctype = _claim_type(path)
            critical = _critical(path, claim)
            claim_evidence = self._evidence_for_claim(
                claim, supplied_evidence, payload,
            )

            if not claim_evidence:
                decisions.append(ClaimDecision(
                    claim, ctype, "insufficient_evidence", "low", False,
                    critical, "No supplied evidence was available.",
                    expected_paper_id, expected_document_id, (), path,
                ))
                continue

            try:
                ev_result = self.evidence_validator.validate_claim(
                    claim,
                    claim_evidence,
                    claim_type=ctype,
                    paper_id=expected_paper_id,
                    document_id=expected_document_id,
                )
            except TypeError:
                ev_result = self.evidence_validator.validate_claim(
                    claim, claim_evidence,
                )

            status = _enum_value(_get(ev_result, "status", "unsupported"))
            confidence = _enum_value(_get(ev_result, "confidence", "low"))
            result_evidence = _evidence_from_result(ev_result)
            ids = tuple(
                x for x in (_evidence_id(e) for e in result_evidence) if x
            )
            supported = status == "supported"

            decisions.append(ClaimDecision(
                claim=claim,
                claim_type=ctype,
                status=status,
                confidence=confidence,
                supported=supported,
                critical=critical,
                reason=str(_get(ev_result, "reason", "")),
                paper_id=_get(ev_result, "paper_id") or expected_paper_id,
                document_id=_get(ev_result, "document_id") or expected_document_id,
                evidence_ids=ids,
                path=path,
            ))

            if status in {"unsupported", "insufficient_evidence"}:
                warnings.append(ValidationIssue(
                    "unsupported_claim",
                    f"Claim is not sufficiently supported: {claim}",
                    "error" if critical else "warning",
                    path, expected_paper_id, claim, ids,
                ))
            elif status in {"contradicted"}:
                errors.append(ValidationIssue(
                    "contradicted_claim",
                    f"Claim is contradicted by supplied evidence: {claim}",
                    "error", path, expected_paper_id, claim, ids,
                ))
            elif status in {"conflicting_evidence"}:
                errors.append(ValidationIssue(
                    "conflicting_evidence",
                    f"Evidence conflicts for claim: {claim}",
                    "error" if critical else "warning",
                    path, expected_paper_id, claim, ids,
                ))
            elif status == "partially_supported":
                warnings.append(ValidationIssue(
                    "partially_supported_claim",
                    f"Claim is only partially supported: {claim}",
                    "warning", path, expected_paper_id, claim, ids,
                ))

        self._validate_numeric_consistency(payload, supplied_evidence, warnings, errors)

        return self._build(
            payload, supplied_evidence, warnings, errors, decisions,
            started, paper_integrity=paper_integrity,
        )

    def validate_response(self, response: Any, **kwargs: Any) -> FinalValidationResult:
        return self.validate(response, **kwargs)

    def _contains_factual_claims(self, payload: Mapping[str, Any]) -> bool:
        return any(True for _ in _walk_claims(payload))

    def _evidence_for_claim(
        self,
        claim: str,
        evidence: Sequence[Any],
        payload: Mapping[str, Any],
    ) -> list[Any]:
        attached = []
        for item in evidence:
            text = _text(_get(item, "text", item))
            if not text:
                continue
            attached.append(item)
        return attached

    def _validate_top10(
        self,
        payload: Mapping[str, Any],
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        papers = (
            _get(payload, "papers")
            or _get(payload, "top_papers")
            or _get(payload, "results")
        )
        if papers is None:
            return
        if not isinstance(papers, Sequence) or isinstance(papers, (str, bytes)):
            errors.append(ValidationIssue(
                "invalid_top10_structure",
                "Mode-1 paper results must be a list.",
                "error",
            ))
            return
        if len(papers) != 10:
            warnings.append(ValidationIssue(
                "top10_count",
                f"Expected TOP 10 results, received {len(papers)}.",
                "warning",
            ))

        seen: set[str] = set()
        for index, paper in enumerate(papers):
            pid = _get(paper, "paper_id")
            if not pid:
                errors.append(ValidationIssue(
                    "missing_paper_id",
                    f"TOP-10 item {index} has no paper_id.",
                    "error",
                    f"papers[{index}]",
                ))
                continue
            pid = str(pid)
            if pid in seen:
                errors.append(ValidationIssue(
                    "duplicate_paper",
                    f"Duplicate paper_id in TOP-10: {pid}.",
                    "error",
                    f"papers[{index}]",
                    pid,
                ))
            seen.add(pid)

    def _validate_uploaded_context(
        self,
        payload: Mapping[str, Any],
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        document_id = _get(payload, "document_id")
        if not document_id and _get(payload, "answer"):
            warnings.append(ValidationIssue(
                "missing_document_id",
                "Uploaded-paper/Q&A response has no document_id.",
                "warning",
            ))

    def _validate_citations(
        self,
        payload: Mapping[str, Any],
        evidence: Sequence[Any],
        expected_paper_id: Optional[str],
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        citations = _get(payload, "citations")
        if citations is None:
            return
        if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
            errors.append(ValidationIssue(
                "malformed_citations",
                "Citations must be a list when present.",
                "error",
            ))
            return

        evidence_ids = {
            _evidence_id(item) for item in evidence if _evidence_id(item)
        }
        for index, citation in enumerate(citations):
            cid = (
                _get(citation, "evidence_id")
                or _get(citation, "citation_id")
                or _get(citation, "id")
                if not isinstance(citation, str)
                else citation
            )
            if cid and evidence_ids and str(cid) not in evidence_ids:
                errors.append(ValidationIssue(
                    "citation_mismatch",
                    f"Citation {cid!r} does not map to supplied evidence.",
                    "error",
                    f"citations[{index}]",
                    expected_paper_id,
                ))

            citation_paper = _get(citation, "paper_id") if not isinstance(citation, str) else None
            if citation_paper and expected_paper_id and str(citation_paper) != expected_paper_id:
                errors.append(ValidationIssue(
                    "citation_wrong_paper",
                    f"Citation points to {citation_paper!r}, expected {expected_paper_id!r}.",
                    "error",
                    f"citations[{index}]",
                    expected_paper_id,
                ))

    def _validate_internal_consistency(
        self,
        payload: Mapping[str, Any],
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        model_mentions = (
            self._section_values(payload, "model")
            + self._section_values(payload, "summary")
            + self._section_values(payload, "methodology")
        )
        dataset_mentions = (
            self._section_values(payload, "dataset")
            + self._section_values(payload, "summary")
            + self._section_values(payload, "findings")
        )
        methodology_mentions = self._section_values(payload, "methodology")

        self._detect_section_conflict(
            model_mentions, "model", warnings, errors,
        )
        self._detect_section_conflict(
            dataset_mentions, "dataset", warnings, errors,
        )

        if model_mentions and methodology_mentions:
            models = self._normalized_entities(model_mentions)
            methods = self._normalized_entities(methodology_mentions)
            if models and methods and not (models & methods):
                if any(self._looks_like_model(x) for x in model_mentions) and any(
                    self._looks_like_model(x) for x in methodology_mentions
                ):
                    errors.append(ValidationIssue(
                        "internal_inconsistency",
                        "Model and methodology sections contain incompatible model identities.",
                        "error",
                    ))

    def _section_values(self, payload: Mapping[str, Any], key: str) -> list[str]:
        value = _get(payload, key)
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [x for x in (_text(v) for v in value.values()) if x]
        if isinstance(value, Sequence):
            return [x for x in (_text(v) for v in value) if x]
        return [_text(value)]

    def _normalized_entities(self, values: Sequence[str]) -> set[str]:
        text = " ".join(values).lower()
        text = re.sub(r"[\-_]", " ", text)
        aliases = {
            "resnet 50": "resnet50",
            "resnet-50": "resnet50",
            "vit": "vit",
            "vision transformer": "vit",
            "efficientnet b0": "efficientnetb0",
            "efficientnet-b0": "efficientnetb0",
        }
        for src, dst in aliases.items():
            text = text.replace(src, dst)
        return {
            token for token in re.findall(
                r"\b(?:resnet\d+|vit|efficientnetb\d+|yolov\d+[a-z]*)\b",
                text,
            )
        }

    def _looks_like_model(self, text: str) -> bool:
        return bool(re.search(
            r"\b(?:resnet\d+|vit|vision transformer|efficientnet|"
            r"yolov\d+|bert|roberta|transformer|cnn|rnn|lstm)\b",
            text, re.I,
        ))

    def _detect_section_conflict(
        self,
        values: Sequence[str],
        label: str,
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        entities = self._normalized_entities(values)
        if len(entities) > 1:
            errors.append(ValidationIssue(
                f"{label}_inconsistency",
                f"Multiple incompatible {label} identities were reported: {sorted(entities)}.",
                "error",
            ))

    def _validate_numeric_consistency(
        self,
        payload: Mapping[str, Any],
        evidence: Sequence[Any],
        warnings: list[ValidationIssue],
        errors: list[ValidationIssue],
    ) -> None:
        section_signatures: dict[str, set[tuple[str, float]]] = {}
        for section in ("summary", "findings", "model", "dataset", "methodology"):
            value = _get(payload, section)
            if value is None:
                continue
            sig = set(_numeric_signature(_text(value)))
            if sig:
                section_signatures[section] = sig

        all_pairs: list[tuple[str, str, tuple[str, float]]] = []
        sections = list(section_signatures)
        for i, left in enumerate(sections):
            for right in sections[i + 1:]:
                left_metrics = {m for m, _ in section_signatures[left]}
                right_metrics = {m for m, _ in section_signatures[right]}
                for metric in left_metrics & right_metrics:
                    lv = {v for m, v in section_signatures[left] if m == metric}
                    rv = {v for m, v in section_signatures[right] if m == metric}
                    if lv and rv and lv.isdisjoint(rv):
                        all_pairs.append((left, right, (metric, min(lv))))
        for left, right, (metric, value) in all_pairs:
            errors.append(ValidationIssue(
                "numerical_inconsistency",
                f"{metric} values differ between {left} and {right}.",
                "error",
            ))

    def _build(
        self,
        payload: Any,
        evidence: Sequence[Any],
        warnings: Sequence[ValidationIssue],
        errors: Sequence[ValidationIssue],
        decisions: Sequence[ClaimDecision],
        started: float,
        *,
        paper_integrity: bool,
    ) -> FinalValidationResult:
        checked = len(decisions)
        supported = sum(d.status == "supported" for d in decisions)
        unsupported = sum(
            d.status in {"unsupported", "insufficient_evidence"}
            for d in decisions
        )
        contradicted = sum(d.status == "contradicted" for d in decisions)
        effective = sum(
            d.status not in {"insufficient_evidence"} for d in decisions
        )
        coverage = supported / effective if effective else 0.0

        critical_bad = any(
            d.critical and d.status in {
                "unsupported", "insufficient_evidence",
                "contradicted", "conflicting_evidence",
            }
            for d in decisions
        )

        has_conflict = any(i.code == "conflicting_evidence" for i in errors)
        has_contradiction = any(
            i.code in {
                "contradicted_claim", "cross_paper_contamination",
                "wrong_document_id", "citation_wrong_paper",
            }
            for i in errors
        )
        has_partial = any(d.status == "partially_supported" for d in decisions)

        if has_conflict:
            status = FinalValidationStatus.CONFLICTING_EVIDENCE
        elif has_contradiction or critical_bad:
            status = FinalValidationStatus.INVALID
        elif checked == 0 and self._contains_factual_claims(payload if isinstance(payload, Mapping) else {}):
            status = FinalValidationStatus.INSUFFICIENT_EVIDENCE
        elif has_partial or unsupported or contradicted:
            status = FinalValidationStatus.PARTIALLY_VALID
        elif warnings:
            status = FinalValidationStatus.VALID_WITH_WARNINGS
        else:
            status = FinalValidationStatus.VALID

        valid = status in {
            FinalValidationStatus.VALID,
            FinalValidationStatus.VALID_WITH_WARNINGS,
        }

        confidence = "high"
        if status in {
            FinalValidationStatus.PARTIALLY_VALID,
            FinalValidationStatus.INSUFFICIENT_EVIDENCE,
        }:
            confidence = "medium"
        if status in {
            FinalValidationStatus.INVALID,
            FinalValidationStatus.CONFLICTING_EVIDENCE,
        }:
            confidence = "low"

        rejected = tuple({
            "path": d.path,
            "claim": d.claim,
            "status": d.status,
            "reason": d.reason,
        } for d in decisions if not d.supported)

        elapsed = time.perf_counter() - started
        paper_id = _get(payload, "paper_id") if isinstance(payload, Mapping) else None

        logger.info(
            "Final response validation complete: status=%s paper_id=%s "
            "claims=%d supported=%d unsupported=%d contradicted=%d processing_time=%.4fs",
            status.value,
            paper_id,
            checked,
            supported,
            unsupported,
            contradicted,
            elapsed,
        )

        return FinalValidationResult(
            valid=valid,
            status=status,
            confidence=confidence,
            validated_content=payload,
            rejected_content=rejected,
            warnings=tuple(warnings),
            errors=tuple(errors),
            claims_checked=checked,
            claims_supported=supported,
            claims_unsupported=unsupported,
            claims_contradicted=contradicted,
            evidence_coverage=round(coverage, 6),
            paper_integrity=paper_integrity,
            metadata={
                "validation_version": "1.0",
                "evidence_count": len(evidence),
                "processing_time": elapsed,
            },
            claim_decisions=tuple(decisions),
        )


Validator = ResponseValidator
FinalResponseValidator = ResponseValidator


def validate_response(
    response: Any,
    *,
    evidence: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> FinalValidationResult:
    return ResponseValidator().validate(response, evidence=evidence, **kwargs)


def _fixture_evidence(text: str, paper_id: str = "paper_001") -> EvidenceItem:
    try:
        from .evidence import Provenance  # type: ignore
    except ImportError:  # pragma: no cover
        from evidence import Provenance  # type: ignore
    return EvidenceItem(
        text=text,
        provenance=Provenance(
            paper_id=paper_id,
            document_id=paper_id,
            page=1,
            section="Results",
        ),
        evidence_id=hashlib.sha256(text.encode()).hexdigest()[:12],
    )


def run_self_test() -> None:
    validator = ResponseValidator()

    evidence = [_fixture_evidence("Our model achieves 94.2% accuracy.")]
    result = validator.validate(
        {
            "paper_id": "paper_001",
            "summary": "The proposed model achieves 94.2% accuracy.",
            "evidence": evidence,
        },
        evidence=evidence,
        expected_paper_id="paper_001",
    )
    assert result.status in {
        FinalValidationStatus.VALID,
        FinalValidationStatus.VALID_WITH_WARNINGS,
    }

    evidence = [_fixture_evidence("Our model achieves 94.2% accuracy.")]
    result = validator.validate(
        {
            "paper_id": "paper_001",
            "summary": "The proposed model achieves 99.2% accuracy.",
        },
        evidence=evidence,
        expected_paper_id="paper_001",
    )
    assert result.status == FinalValidationStatus.INVALID

    evidence = [_fixture_evidence("The model achieves 94.2% accuracy.")]
    result = validator.validate(
        {
            "paper_id": "paper_001",
            "summary": "The model achieves 94.2% F1.",
        },
        evidence=evidence,
        expected_paper_id="paper_001",
    )
    assert result.status == FinalValidationStatus.INVALID

    evidence = [_fixture_evidence("Accuracy is 94%.", "paper_A")]
    result = validator.validate(
        {
            "paper_id": "paper_B",
            "summary": "Accuracy is 94%.",
        },
        evidence=evidence,
        expected_paper_id="paper_B",
    )
    assert result.status == FinalValidationStatus.INVALID
    assert not result.paper_integrity

    result = validator.validate(
        {
            "paper_id": "paper_001",
            "answer": "The optimizer used was Adam.",
        },
        evidence=[],
        expected_paper_id="paper_001",
    )
    assert result.status in {
        FinalValidationStatus.INVALID,
        FinalValidationStatus.INSUFFICIENT_EVIDENCE,
    }

    result = validator.validate({
        "paper_id": "paper_001",
        "summary": "The model uses ResNet-50.",
        "model": "The model uses ViT.",
    }, evidence=[])
    assert any(
        issue.code == "model_inconsistency"
        for issue in result.errors
    )

    e = [_fixture_evidence("The model achieves 94% accuracy.")]
    payload = {"paper_id": "paper_001", "summary": "The model achieves 94% accuracy."}
    a = validator.validate(payload, evidence=e, expected_paper_id="paper_001")
    b = validator.validate(payload, evidence=e, expected_paper_id="paper_001")
    assert a.status == b.status
    assert a.claims_checked == b.claims_checked
    assert [d.status for d in a.claim_decisions] == [d.status for d in b.claim_decisions]

    print("ResponseValidator self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()