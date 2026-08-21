from __future__ import annotations

import asyncio
import datetime as dt
import enum
import json
import logging
import os
import pathlib
import re
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from src.rag.prompt_builder import MultiPromptPayload
except Exception:
    MultiPromptPayload = Any

try:
    from src.rag.context_manager import OptimizedContextPackage
except Exception:
    OptimizedContextPackage = Any


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LLMProvider(str, enum.Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"
    DEEPSEEK = "DEEPSEEK"
    OLLAMA = "OLLAMA"
    LOCAL_FALLBACK = "LOCAL_FALLBACK"


class ExecutionMode(str, enum.Enum):
    SINGLE_MODEL = "SINGLE_MODEL"
    AGENTIC_JUDGE_LOOP = "AGENTIC_JUDGE_LOOP"
    MULTI_MODEL_ENSEMBLE = "MULTI_MODEL_ENSEMBLE"


class OrchestratorTaskIntent(str, enum.Enum):
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    METHOD_COMPARISON = "METHOD_COMPARISON"
    RESEARCH_GAP = "RESEARCH_GAP"
    PATENT_DRAFTING = "PATENT_DRAFTING"
    SUMMARIZATION = "SUMMARIZATION"


TaskIntent = OrchestratorTaskIntent


# =============================================================================
# CONFIGURATION
# =============================================================================

class OrchestratorConfig:
    VERSION = "9.0.0-PROD-ULTIMATE"

    EMBEDDING_MODEL = os.getenv("ORCHESTRATOR_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    SEMANTIC_SUPPORT_THRESHOLD = float(os.getenv("ORCHESTRATOR_SEMANTIC_THRESHOLD", "0.50"))
    STRONG_SUPPORT_THRESHOLD = float(os.getenv("ORCHESTRATOR_STRONG_THRESHOLD", "0.65"))
    MAX_EVIDENCE_BLOCKS = 40
    MAX_CLAIMS = 40

    DEFAULT_MODE = ExecutionMode.AGENTIC_JUDGE_LOOP
    MAX_REFINEMENT_LOOPS = 2
    REQUEST_TIMEOUT_SECONDS = 45
    MAX_RETRIES = 2

    QUALITY_PASS_THRESHOLD = 90.0
    GROUNDING_PASS_THRESHOLD = 90.0
    CITATION_PASS_THRESHOLD = 90.0
    MAX_UNSUPPORTED_CLAIMS_FOR_PASS = 0

    OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1/chat/completions")
    ANTHROPIC_ENDPOINT = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1/messages")
    GEMINI_ENDPOINT = os.getenv(
        "GEMINI_API_BASE",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    )
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", f"{OLLAMA_BASE_URL}/api/chat")
    OLLAMA_TAGS_ENDPOINT = os.getenv("OLLAMA_TAGS_ENDPOINT", f"{OLLAMA_BASE_URL}/api/tags")
    OLLAMA_HEALTH_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "0.75"))
    OLLAMA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180"))
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "512"))
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_AUTO_DISCOVER_MODEL = os.getenv("OLLAMA_AUTO_DISCOVER_MODEL", "1").lower() not in {"0", "false", "no"}

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    MODEL_PRICING: Dict[str, Tuple[float, float]] = {
        "gpt-4o": (0.0025, 0.0100),
        "gpt-4o-mini": (0.00015, 0.0006),
        "claude-3-5-sonnet-20240620": (0.0030, 0.0150),
        "gemini-1.5-pro": (0.00125, 0.0050),
        "deepseek-chat": (0.00014, 0.00028),
        "llama3": (0.0, 0.0),
        "ollama-local": (0.0, 0.0),
        "local-fallback": (0.0, 0.0),
    }

    MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
        "claude-3-5-sonnet-20240620": {
            "provider": LLMProvider.ANTHROPIC,
            "reasoning": 98.0,
            "latency": 72.0,
            "cost": 55.0,
            "context": 100.0,
        },
        "gpt-4o": {
            "provider": LLMProvider.OPENAI,
            "reasoning": 95.0,
            "latency": 85.0,
            "cost": 60.0,
            "context": 95.0,
        },
        "deepseek-chat": {
            "provider": LLMProvider.DEEPSEEK,
            "reasoning": 90.0,
            "latency": 80.0,
            "cost": 95.0,
            "context": 85.0,
        },
        "ollama-local": {
            "provider": LLMProvider.OLLAMA,
            "reasoning": 82.0,
            "latency": 88.0,
            "cost": 100.0,
            "context": 88.0,
        },
        "local-fallback": {
            "provider": LLMProvider.LOCAL_FALLBACK,
            "reasoning": 55.0,
            "latency": 100.0,
            "cost": 100.0,
            "context": 45.0,
        },
    }

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    REPORTS_DIR = pathlib.Path(
        os.getenv(
            "ORCHESTRATOR_REPORT_DIR",
            str(PROJECT_ROOT / "outputs" / "reports" / "llm_orchestrator"),
        )
    )
    PLOTS_DIR = REPORTS_DIR / "plots"
    LOGS_DIR = REPORTS_DIR / "logs"


# =============================================================================
# LOGGING
# =============================================================================

def setup_directories() -> None:
    for path in (
        OrchestratorConfig.REPORTS_DIR,
        OrchestratorConfig.PLOTS_DIR,
        OrchestratorConfig.LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    setup_directories()
    logger = logging.getLogger("LLMOrchestrator")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.FileHandler(
            OrchestratorConfig.LOGS_DIR / "orchestrator.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvidenceBlock:
    ref_id: str
    text: str
    title: str = ""
    paper_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimVerification:
    claim: str
    citations: List[str]
    best_ref: Optional[str]
    semantic_similarity: float
    lexical_overlap: float
    support_score: float
    status: str
    reason: str


@dataclass
class SemanticGroundingReport:
    grounding_score: float
    total_claims: int
    grounded_claims_count: int
    partial_claims_count: int
    unsupported_claims_count: int
    unsupported_claims: List[str]
    supported_claims: List[str]
    verified_citations: List[str]
    invalid_citations: List[str]
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    evidence_blocks: int = 0
    evidence_coverage_pct: float = 0.0

    @property
    def citation_anchors_found(self) -> List[str]:
        return self.verified_citations


@dataclass
class QualityBreakdown:
    grounding_score: float
    citation_correctness: float
    coverage_score: float
    evidence_usage: float
    consistency_score: float
    readability_score: float
    completeness_score: float
    final_quality_score: float

    @property
    def quality_score(self) -> float:
        return self.final_quality_score


@dataclass
class LLMModelSpec:
    provider: LLMProvider
    model_name: str
    temperature: float
    max_tokens: int
    routing_score: float = 0.0


@dataclass
class CostReport:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    pricing_model: str
    cache_hit: bool = False


@dataclass
class ExecutionTraceRecord:
    execution_id: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    mode: str = ""
    routed_provider: str = ""
    routed_model: str = ""
    routing_scores: Dict[str, float] = field(default_factory=dict)
    routing_reason: str = ""
    judge_critique_applied: bool = False
    refinement_loops: int = 0
    ensemble_models_called: List[str] = field(default_factory=list)
    retries: int = 0
    fallback_occurred: bool = False
    fallback_reason: str = ""
    latency_ms: float = 0.0
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrchestratedResponse:
    execution_id: str
    query: str
    task_type: OrchestratorTaskIntent
    generated_response: str
    model_spec: LLMModelSpec
    cost_report: CostReport
    grounding_report: SemanticGroundingReport
    quality: QualityBreakdown
    execution_trace: ExecutionTraceRecord

    @property
    def quality_score(self) -> float:
        return self.quality.final_quality_score

    @property
    def generated_text(self) -> str:
        return self.generated_response


# =============================================================================
# GENERIC OBJECT / CONTEXT ADAPTERS
# =============================================================================

def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(x) for x in value if _stringify(x))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if is_dataclass(value):
        return json.dumps(asdict(value), ensure_ascii=False, default=str)
    return str(value)


def _extract_evidence(context_package: Any = None, prompt_payload: Any = None) -> List[EvidenceBlock]:
    raw_chunks: List[Any] = []

    for source in (context_package, prompt_payload):
        chunks = _get_attr_or_key(source, "chunks", None)
        if chunks:
            if isinstance(chunks, (list, tuple)):
                raw_chunks.extend(chunks)

    blocks: List[EvidenceBlock] = []

    for idx, chunk in enumerate(raw_chunks, start=1):
        text = (
            _get_attr_or_key(chunk, "chunk_text", "")
            or _get_attr_or_key(chunk, "text", "")
            or _get_attr_or_key(chunk, "content", "")
        )
        text = _stringify(text).strip()
        if not text:
            continue

        ref = (
            _get_attr_or_key(chunk, "ref_id", "")
            or _get_attr_or_key(chunk, "citation", "")
            or _get_attr_or_key(chunk, "chunk_id", "")
        )
        ref = str(ref).strip()
        ref_match = re.search(r"(Ref-\d+)", ref, flags=re.I)
        ref_id = ref_match.group(1) if ref_match else f"Ref-{idx}"

        title = _stringify(_get_attr_or_key(chunk, "title", "")).strip()
        paper_id = _stringify(_get_attr_or_key(chunk, "paper_id", "")).strip()

        blocks.append(
            EvidenceBlock(
                ref_id=ref_id,
                text=text,
                title=title,
                paper_id=paper_id,
                metadata={
                    "similarity_score": _get_attr_or_key(chunk, "similarity_score", None),
                    "confidence_score": _get_attr_or_key(chunk, "confidence_score", None),
                    "publication_year": _get_attr_or_key(chunk, "publication_year", None),
                },
            )
        )

    if not blocks:
        context_text = (
            _stringify(_get_attr_or_key(context_package, "formatted_context", ""))
            or _stringify(_get_attr_or_key(context_package, "context_text", ""))
            or _stringify(_get_attr_or_key(prompt_payload, "context_prompt", ""))
            or _stringify(_get_attr_or_key(prompt_payload, "formatted_context", ""))
        )

        if context_text.strip():
            parts = [
                p.strip()
                for p in re.split(r"\n\s*\n", context_text)
                if len(p.strip()) >= 30
            ]
            if not parts:
                parts = [context_text.strip()]

            for idx, part in enumerate(parts[: OrchestratorConfig.MAX_EVIDENCE_BLOCKS], start=1):
                match = re.search(r"\[(Ref-\d+)\]", part, flags=re.I)
                ref_id = match.group(1) if match else f"Ref-{idx}"
                blocks.append(EvidenceBlock(ref_id=ref_id, text=part))

    unique: Dict[str, EvidenceBlock] = {}
    for block in blocks:
        unique.setdefault(block.ref_id, block)

    return list(unique.values())[: OrchestratorConfig.MAX_EVIDENCE_BLOCKS]


def _format_evidence(blocks: Sequence[EvidenceBlock]) -> str:
    if not blocks:
        return ""
    return "\n\n".join(
        f"[{b.ref_id}] {b.title + ': ' if b.title else ''}{b.text}"
        for b in blocks
    )


# =============================================================================
# CLAIM / TEXT PROCESSING
# =============================================================================

_CITATION_RE = re.compile(r"\[(Ref-\d+)\]", re.I)


def _split_sentences(text: str) -> List[str]:
    cleaned_text = re.sub(r"(?m)^\s*#{1,6}\s*[^\n]+\n+", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9#*])", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) >= 18]


def _extract_claims(text: str) -> List[str]:
    claims: List[str] = []
    for sentence in _split_sentences(text):
        s = re.sub(r"^#+\s*", "", sentence).strip()
        if len(s) < 25:
            continue
        if s.endswith(":") and len(s.split()) < 12:
            continue
        if re.fullmatch(r"[*_`#\-\s\d.]+", s):
            continue
        claims.append(s)
        if len(claims) >= OrchestratorConfig.MAX_CLAIMS:
            break
    return claims


def _tokens(text: str) -> set:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on",
        "with", "is", "are", "was", "were", "be", "by", "as", "that",
        "this", "than", "from", "at", "it", "their", "its", "into",
        "can", "may", "more", "less", "such", "using", "used",
    }
    return {
        t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", (text or "").lower())
        if t not in stop
    }


def _lexical_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# =============================================================================
# SEMANTIC GROUNDING
# =============================================================================

class SemanticGroundingEngine:
    def __init__(self, embedding_model_name: str = OrchestratorConfig.EMBEDDING_MODEL):
        self.encoder = None
        self.model_name = embedding_model_name

        if SentenceTransformer is not None:
            try:
                self.encoder = SentenceTransformer(embedding_model_name)
                logger.info("Semantic grounding encoder initialized: %s", embedding_model_name)
            except Exception as exc:
                logger.warning("Semantic encoder unavailable; using lexical grounding fallback: %s", exc)

    def _semantic_matrix(
        self,
        claims: Sequence[str],
        evidence: Sequence[str],
    ) -> Optional[np.ndarray]:
        if self.encoder is None or not claims or not evidence:
            return None
        try:
            claim_vecs = self.encoder.encode(
                list(claims), normalize_embeddings=True, convert_to_numpy=True
            )
            evidence_vecs = self.encoder.encode(
                list(evidence), normalize_embeddings=True, convert_to_numpy=True
            )
            return np.matmul(claim_vecs, evidence_vecs.T)
        except Exception as exc:
            logger.warning("Embedding verification failed; using lexical fallback: %s", exc)
            return None

    def verify_grounding(
        self,
        response_text: str,
        evidence: Union[str, Sequence[EvidenceBlock]],
    ) -> SemanticGroundingReport:
        if isinstance(evidence, str):
            blocks = [
                EvidenceBlock(ref_id=f"Ref-{i}", text=p)
                for i, p in enumerate(
                    [x.strip() for x in re.split(r"\n\s*\n", evidence) if x.strip()],
                    start=1,
                )
            ]
        else:
            blocks = list(evidence)

        claims = _extract_claims(response_text)

        if not claims:
            return SemanticGroundingReport(
                grounding_score=0.0,
                total_claims=0,
                grounded_claims_count=0,
                partial_claims_count=0,
                unsupported_claims_count=0,
                unsupported_claims=[],
                supported_claims=[],
                verified_citations=[],
                invalid_citations=[],
                evidence_blocks=len(blocks),
                evidence_coverage_pct=0.0,
            )

        if not blocks:
            unsupported = claims[:]
            return SemanticGroundingReport(
                grounding_score=0.0,
                total_claims=len(claims),
                grounded_claims_count=0,
                partial_claims_count=0,
                unsupported_claims_count=len(claims),
                unsupported_claims=unsupported,
                supported_claims=[],
                verified_citations=[],
                invalid_citations=sorted(set(_CITATION_RE.findall(response_text))),
                claim_verifications=[
                    ClaimVerification(
                        claim=c,
                        citations=list(dict.fromkeys(_CITATION_RE.findall(c))),
                        best_ref=None,
                        semantic_similarity=0.0,
                        lexical_overlap=0.0,
                        support_score=0.0,
                        status="UNSUPPORTED",
                        reason="No evidence was supplied to the orchestrator.",
                    )
                    for c in claims
                ],
                evidence_blocks=0,
                evidence_coverage_pct=0.0,
            )

        ref_map = {b.ref_id.lower(): b for b in blocks}
        evidence_texts = [b.text for b in blocks]
        matrix = self._semantic_matrix(claims, evidence_texts)

        verifications: List[ClaimVerification] = []
        supported: List[str] = []
        unsupported: List[str] = []
        verified_citations: set = set()
        invalid_citations: set = set()
        grounded = 0
        partial = 0

        for i, claim in enumerate(claims):
            cited = list(dict.fromkeys(_CITATION_RE.findall(claim)))

            if not cited:
                normalized_response = re.sub(r"\n\s*\n", " <PARAGRAPH> ", response_text)
                normalized_response = re.sub(r"\s+", " ", normalized_response).strip()
                normalized_claim = re.sub(r"\s+", " ", claim).strip()
                probe = normalized_claim[:100]
                claim_pos = normalized_response.find(probe)
                if claim_pos >= 0:
                    para_start = normalized_response.rfind("<PARAGRAPH>", 0, claim_pos)
                    para_start = 0 if para_start < 0 else para_start + len("<PARAGRAPH>")
                    para_end = normalized_response.find("<PARAGRAPH>", claim_pos)
                    if para_end < 0:
                        para_end = len(normalized_response)
                    paragraph_window = normalized_response[para_start:para_end]
                    cited = list(dict.fromkeys(_CITATION_RE.findall(paragraph_window)))

            valid_cited_blocks: List[EvidenceBlock] = []

            for ref in cited:
                block = ref_map.get(ref.lower())
                if block is None:
                    invalid_citations.add(ref)
                else:
                    valid_cited_blocks.append(block)

            if matrix is not None:
                sims = matrix[i]
                best_index = int(np.argmax(sims))
                global_sem = float(sims[best_index])
            else:
                lexical_scores = [_lexical_overlap(claim, b.text) for b in blocks]
                best_index = int(np.argmax(lexical_scores))
                global_sem = 0.0

            if valid_cited_blocks:
                cited_indices = [
                    blocks.index(block) for block in valid_cited_blocks if block in blocks
                ]
                if matrix is not None and cited_indices:
                    cited_scores = [float(matrix[i][j]) for j in cited_indices]
                    cited_index = cited_indices[int(np.argmax(cited_scores))]
                    semantic = max(cited_scores)
                    best_ref = blocks[cited_index].ref_id
                else:
                    scored_cited = [
                        (b.ref_id, _lexical_overlap(claim, b.text), b)
                        for b in valid_cited_blocks
                    ]
                    best_ref, semantic, _ = max(
                        scored_cited,
                        key=lambda item: item[1],
                    )
            else:
                best_ref = blocks[best_index].ref_id if blocks else None
                semantic = global_sem

            best_block = ref_map.get(best_ref.lower()) if best_ref else blocks[best_index]
            lexical = _lexical_overlap(claim, best_block.text) if best_block else 0.0

            if matrix is not None:
                support = 0.65 * semantic + 0.35 * lexical
            else:
                claim_terms = _tokens(claim)
                evidence_terms = _tokens(best_block.text if best_block else "")
                intersection = len(claim_terms & evidence_terms)
                union = max(1, len(claim_terms | evidence_terms))
                jaccard = intersection / union
                support = 0.65 * lexical + 0.35 * jaccard

            has_valid_citation = bool(valid_cited_blocks)
            if has_valid_citation:
                verified_citations.update(b.ref_id for b in valid_cited_blocks)

            if has_valid_citation and support >= OrchestratorConfig.STRONG_SUPPORT_THRESHOLD:
                status = "VERIFIED"
                grounded += 1
                reason = "Valid citation mapped to evidence with strong semantic/lexical support."
                supported.append(claim)
            elif has_valid_citation and support >= OrchestratorConfig.SEMANTIC_SUPPORT_THRESHOLD:
                status = "VERIFIED"
                grounded += 1
                reason = "Valid citation mapped to evidence with acceptable semantic support."
                supported.append(claim)
            else:
                status = "UNSUPPORTED"
                unsupported.append(claim)
                reason = (
                    "No valid citation/evidence support reached the required threshold."
                    if not has_valid_citation
                    else "Cited evidence does not sufficiently support the claim."
                )

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    citations=cited,
                    best_ref=best_ref,
                    semantic_similarity=round(semantic, 4),
                    lexical_overlap=round(lexical, 4),
                    support_score=round(support, 4),
                    status=status,
                    reason=reason,
                )
            )

        weighted_support = (
            grounded * 1.0 + partial * 0.85
        ) / max(1, len(claims))
        grounding_score = round(weighted_support * 100.0, 1)

        used_refs = {
            v.best_ref
            for v in verifications
            if v.status in {"VERIFIED", "PARTIAL"} and v.best_ref
        }
        evidence_coverage = round(
            min(100.0, len(used_refs) / max(1, len(blocks)) * 100.0), 1
        )

        return SemanticGroundingReport(
            grounding_score=grounding_score,
            total_claims=len(claims),
            grounded_claims_count=grounded,
            partial_claims_count=partial,
            unsupported_claims_count=len(unsupported),
            unsupported_claims=unsupported,
            supported_claims=supported,
            verified_citations=sorted(verified_citations),
            invalid_citations=sorted(invalid_citations),
            claim_verifications=verifications,
            evidence_blocks=len(blocks),
            evidence_coverage_pct=evidence_coverage,
        )


# =============================================================================
# ROUTING
# =============================================================================

class ExplainableModelRouter:
    TASK_WEIGHTS = {
        OrchestratorTaskIntent.METHOD_COMPARISON: (0.50, 0.15, 0.10, 0.25),
        OrchestratorTaskIntent.RESEARCH_GAP: (0.50, 0.10, 0.10, 0.30),
        OrchestratorTaskIntent.LITERATURE_REVIEW: (0.45, 0.15, 0.10, 0.30),
        OrchestratorTaskIntent.PATENT_DRAFTING: (0.40, 0.20, 0.10, 0.30),
        OrchestratorTaskIntent.QUESTION_ANSWERING: (0.35, 0.25, 0.15, 0.25),
        OrchestratorTaskIntent.SUMMARIZATION: (0.30, 0.30, 0.15, 0.25),
    }

    MODEL_NAMES = {
        LLMProvider.OPENAI: OrchestratorConfig.OPENAI_MODEL,
        LLMProvider.ANTHROPIC: OrchestratorConfig.ANTHROPIC_MODEL,
        LLMProvider.DEEPSEEK: OrchestratorConfig.DEEPSEEK_MODEL,
        LLMProvider.GEMINI: OrchestratorConfig.GEMINI_MODEL,
        LLMProvider.OLLAMA: "ollama-local",
        LLMProvider.LOCAL_FALLBACK: "local-fallback",
    }

    @staticmethod
    def _ollama_reachable() -> bool:
        try:
            from urllib.request import urlopen
            with urlopen(
                OrchestratorConfig.OLLAMA_TAGS_ENDPOINT,
                timeout=OrchestratorConfig.OLLAMA_HEALTH_TIMEOUT_SECONDS,
            ) as response:
                return 200 <= int(response.status) < 300
        except Exception:
            return False

    @classmethod
    def available_model_names(cls) -> List[str]:
        names: List[str] = []

        if os.getenv("OPENAI_API_KEY"):
            names.append(OrchestratorConfig.OPENAI_MODEL)
        if os.getenv("ANTHROPIC_API_KEY"):
            names.append(OrchestratorConfig.ANTHROPIC_MODEL)
        if os.getenv("DEEPSEEK_API_KEY"):
            names.append(OrchestratorConfig.DEEPSEEK_MODEL)
        if os.getenv("GEMINI_API_KEY"):
            names.append(OrchestratorConfig.GEMINI_MODEL)

        if cls._ollama_reachable():
            names.append("ollama-local")

        names.append("local-fallback")
        return names

    @classmethod
    def calculate_routing_scores(
        cls,
        task_type: OrchestratorTaskIntent,
        prompt_tokens: int,
        available_models: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        weights = cls.TASK_WEIGHTS.get(task_type, (0.35, 0.25, 0.15, 0.25))
        available = set(available_models or cls.available_model_names())

        scores: Dict[str, float] = {}
        for model_name, profile in OrchestratorConfig.MODEL_PROFILES.items():
            if model_name not in available:
                continue

            context = float(profile["context"])
            if prompt_tokens > 32000:
                context *= 0.80

            score = (
                weights[0] * float(profile["reasoning"])
                + weights[1] * float(profile["latency"])
                + weights[2] * float(profile["cost"])
                + weights[3] * context
            )
            scores[model_name] = round(score, 2)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def _spec_for(
        cls,
        provider: LLMProvider,
        model_name: str,
        score: float,
        task_type: OrchestratorTaskIntent,
    ) -> LLMModelSpec:
        temperature = 0.05 if task_type != OrchestratorTaskIntent.PATENT_DRAFTING else 0.15
        max_tokens = 4096
        if task_type == OrchestratorTaskIntent.LITERATURE_REVIEW:
            max_tokens = 6144
        return LLMModelSpec(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            routing_score=score,
        )

    @classmethod
    def route(
        cls,
        task_type: OrchestratorTaskIntent,
        prompt_tokens: int,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> Tuple[LLMModelSpec, Dict[str, float]]:
        available = cls.available_model_names()
        scores = cls.calculate_routing_scores(task_type, prompt_tokens, available)

        if preferred_provider in {LLMProvider.LOCAL_FALLBACK, LLMProvider.OLLAMA}:
            if "ollama-local" in available:
                return cls._spec_for(
                    LLMProvider.OLLAMA,
                    OrchestratorConfig.OLLAMA_MODEL,
                    scores.get("ollama-local", 0.0),
                    task_type,
                ), scores
            return cls._spec_for(
                LLMProvider.LOCAL_FALLBACK,
                "local-fallback",
                scores.get("local-fallback", 0.0),
                task_type,
            ), scores

        if preferred_provider is not None:
            for name, profile in OrchestratorConfig.MODEL_PROFILES.items():
                if profile["provider"] == preferred_provider and name in available:
                    return cls._spec_for(
                        preferred_provider, name, scores.get(name, 0.0), task_type
                    ), scores

        cloud_available = any(
            name in available
            for name in (
                OrchestratorConfig.OPENAI_MODEL,
                OrchestratorConfig.ANTHROPIC_MODEL,
                OrchestratorConfig.DEEPSEEK_MODEL,
                OrchestratorConfig.GEMINI_MODEL,
            )
        )
        if not cloud_available:
            if "ollama-local" in available:
                return cls._spec_for(
                    LLMProvider.OLLAMA,
                    OrchestratorConfig.OLLAMA_MODEL,
                    scores.get("ollama-local", 0.0),
                    task_type,
                ), scores
            return cls._spec_for(
                LLMProvider.LOCAL_FALLBACK,
                "local-fallback",
                scores.get("local-fallback", 0.0),
                task_type,
            ), scores

        if not scores:
            return cls._spec_for(
                LLMProvider.LOCAL_FALLBACK,
                "local-fallback",
                0.0,
                task_type,
            ), scores

        best_name = next(iter(scores))
        profile = OrchestratorConfig.MODEL_PROFILES[best_name]
        return cls._spec_for(
            profile["provider"], best_name, scores[best_name], task_type
        ), scores


# =============================================================================
# DYNAMIC LLM CLIENT
# =============================================================================

class DynamicLLMClient:
    @staticmethod
    async def generate_async(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(OrchestratorConfig.MAX_RETRIES + 1):
            try:
                if spec.provider == LLMProvider.OLLAMA:
                    return await DynamicLLMClient._call_ollama(
                        system_prompt, user_prompt, spec
                    )
                if spec.provider == LLMProvider.OPENAI:
                    key = os.getenv("OPENAI_API_KEY", "")
                    if not key:
                        raise RuntimeError("OPENAI_API_KEY is not configured.")
                    return await DynamicLLMClient._call_openai_compatible(
                        system_prompt, user_prompt, spec, key,
                        OrchestratorConfig.OPENAI_ENDPOINT
                    )
                if spec.provider == LLMProvider.DEEPSEEK:
                    key = os.getenv("DEEPSEEK_API_KEY", "")
                    if not key:
                        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
                    return await DynamicLLMClient._call_openai_compatible(
                        system_prompt, user_prompt, spec, key,
                        OrchestratorConfig.DEEPSEEK_ENDPOINT
                    )
                if spec.provider == LLMProvider.ANTHROPIC:
                    key = os.getenv("ANTHROPIC_API_KEY", "")
                    if not key:
                        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
                    return await DynamicLLMClient._call_anthropic(
                        system_prompt, user_prompt, spec, key
                    )
                if spec.provider == LLMProvider.GEMINI:
                    key = os.getenv("GEMINI_API_KEY", "")
                    if not key:
                        raise RuntimeError("GEMINI_API_KEY is not configured.")
                    return await DynamicLLMClient._call_gemini(
                        system_prompt, user_prompt, spec, key
                    )
                return DynamicLLMClient._generate_local_fallback(user_prompt)
            except Exception as exc:
                last_error = exc
                if attempt < OrchestratorConfig.MAX_RETRIES:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise RuntimeError(str(last_error) if last_error else "LLM generation failed.")

    @staticmethod
    async def _call_openai_compatible(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
        api_key: str,
        endpoint: str,
    ) -> str:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": spec.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        }

        timeout = aiohttp.ClientTimeout(total=OrchestratorConfig.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"{spec.provider.value} HTTP {resp.status}: {body[:500]}")
                data = json.loads(body)
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    raise RuntimeError(f"{spec.provider.value} returned empty content.")
                return str(content).strip()

    @staticmethod
    async def _call_anthropic(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
        api_key: str,
    ) -> str:
        import aiohttp

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": spec.model_name,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        timeout = aiohttp.ClientTimeout(total=OrchestratorConfig.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                OrchestratorConfig.ANTHROPIC_ENDPOINT,
                json=payload,
                headers=headers,
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"ANTHROPIC HTTP {resp.status}: {body[:500]}")
                data = json.loads(body)
                content = data.get("content", [])
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                result = "\n".join(text_parts).strip()
                if not result:
                    raise RuntimeError("Anthropic returned empty content.")
                return result

    @staticmethod
    async def _call_gemini(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
        api_key: str,
    ) -> str:
        import aiohttp

        endpoint = OrchestratorConfig.GEMINI_ENDPOINT.format(model=spec.model_name)
        endpoint = f"{endpoint}?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": spec.temperature,
                "maxOutputTokens": spec.max_tokens,
            },
        }

        timeout = aiohttp.ClientTimeout(total=OrchestratorConfig.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"GEMINI HTTP {resp.status}: {body[:500]}")
                data = json.loads(body)
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Gemini returned no candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                result = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                ).strip()
                if not result:
                    raise RuntimeError("Gemini returned empty content.")
                return result

    @staticmethod
    def _resolve_ollama_model_sync() -> str:
        requested = OrchestratorConfig.OLLAMA_MODEL
        if not OrchestratorConfig.OLLAMA_AUTO_DISCOVER_MODEL:
            return requested

        try:
            from urllib.request import urlopen
            with urlopen(
                OrchestratorConfig.OLLAMA_TAGS_ENDPOINT,
                timeout=OrchestratorConfig.OLLAMA_HEALTH_TIMEOUT_SECONDS,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            installed = {
                str(item.get("name", "")).strip()
                for item in payload.get("models", [])
                if item.get("name")
            }
            if requested in installed:
                return requested

            candidates = [
                requested,
                "qwen2.5:7b",
                "qwen2.5:3b",
                "llama3.1:8b",
                "llama3.2:3b",
                "mistral:7b",
                "llama3:8b",
            ]
            for candidate in candidates:
                if candidate in installed:
                    logger.info("Configured model '%s' unavailable; using '%s'.", requested, candidate)
                    return candidate
        except Exception:
            pass

        return requested

    @staticmethod
    async def _call_ollama(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
    ) -> str:
        import aiohttp

        model_name = (
            spec.model_name
            if spec.model_name not in {"ollama-local", "local-fallback"}
            else await asyncio.to_thread(DynamicLLMClient._resolve_ollama_model_sync)
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": OrchestratorConfig.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": spec.temperature,
                "num_predict": spec.max_tokens,
                "num_ctx": OrchestratorConfig.OLLAMA_NUM_CTX,
                "num_batch": OrchestratorConfig.OLLAMA_NUM_BATCH,
            },
        }

        timeout = aiohttp.ClientTimeout(total=OrchestratorConfig.OLLAMA_REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                OrchestratorConfig.OLLAMA_ENDPOINT,
                json=payload,
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"OLLAMA HTTP {resp.status}: {body[:700]}")
                data = json.loads(body)
                result = str(data.get("message", {}).get("content", "")).strip()
                if not result:
                    raise RuntimeError("Ollama returned empty content.")
                return result

    @staticmethod
    def _generate_local_fallback(user_prompt: str) -> str:
        query_match = re.search(r"(?:QUERY|Question)\s*:\s*(.+)", user_prompt, flags=re.I)
        query = query_match.group(1).strip() if query_match else "the requested research question"

        evidence_match = re.search(
            r"VERIFIABLE EVIDENCE:\s*(.*?)(?:\n\nCITATION RULE:|\Z)",
            user_prompt,
            flags=re.I | re.S,
        )
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""

        blocks: List[Tuple[str, str]] = []
        for part in re.split(r"\n\s*\n", evidence_text):
            part = part.strip()
            if not part:
                continue
            ref = re.search(r"\[(Ref-\d+)\]", part, flags=re.I)
            if not ref:
                continue
            ref_id = ref.group(1)
            body = re.sub(r"^\[Ref-\d+\]\s*", "", part, flags=re.I)
            if ": " in body and len(body.split(": ", 1)[0].split()) <= 10:
                body = body.split(": ", 1)[1].strip()
            blocks.append((ref_id, body))

        if not blocks:
            return f"The query is: {query}. The retrieved evidence is insufficient to establish this claim."

        paragraphs: List[str] = []
        for ref_id, body in blocks[:4]:
            sentences = _split_sentences(body)
            excerpt = " ".join(sentences[:2]) if sentences else body
            paragraphs.append(f"{excerpt} [{ref_id}].")

        return "### Evidence-Grounded Research Analysis\n\n" + "\n\n".join(
            f"{idx}. {paragraph}" for idx, paragraph in enumerate(paragraphs, start=1)
        )


# =============================================================================
# ENSEMBLE + JUDGE
# =============================================================================

class MultiModelEnsembleEngine:
    @staticmethod
    async def run_ensemble(
        system_prompt: str,
        user_prompt: str,
        specs: Sequence[LLMModelSpec],
    ) -> Tuple[str, List[str]]:
        tasks = [
            DynamicLLMClient.generate_async(system_prompt, user_prompt, spec)
            for spec in specs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: List[Tuple[LLMModelSpec, str]] = []
        for spec, result in zip(specs, results):
            if isinstance(result, str) and len(result.strip()) >= 80:
                valid.append((spec, result.strip()))

        if not valid:
            return DynamicLLMClient._generate_local_fallback(user_prompt), ["local-fallback"]

        evaluator = SemanticGroundingEngine()
        scored: List[Tuple[float, str, str]] = []

        for spec, response in valid:
            report = evaluator.verify_grounding(response, user_prompt)
            score = report.grounding_score + min(20.0, len(report.verified_citations) * 4.0)
            scored.append((score, spec.model_name, response))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][2], [name for _, name, _ in scored]


class JudgeModelSelfReflector:
    @staticmethod
    async def refine(
        candidate_response: str,
        user_prompt: str,
        grounding_report: SemanticGroundingReport,
        spec: LLMModelSpec,
    ) -> str:
        if (
            grounding_report.grounding_score >= OrchestratorConfig.GROUNDING_PASS_THRESHOLD
            and grounding_report.unsupported_claims_count <= OrchestratorConfig.MAX_UNSUPPORTED_CLAIMS_FOR_PASS
        ):
            return candidate_response

        unsupported = "\n".join(f"- {claim}" for claim in grounding_report.unsupported_claims[:10])

        prompt = (
            "You are a strict scientific evidence auditor.\n\n"
            "Rewrite the candidate response so that every factual claim is directly supported "
            "by the supplied evidence. Do not invent citations or numerical results. "
            "If the evidence does not provide sufficient information, explicitly state: "
            "'The retrieved evidence is insufficient to establish this claim.' "
            "Remove unsupported claims rather than guessing.\n\n"
            f"USER / EVIDENCE PROMPT:\n{user_prompt}\n\n"
            f"CANDIDATE RESPONSE:\n{candidate_response}\n\n"
            f"UNSUPPORTED CLAIMS:\n{unsupported}\n\n"
            "Return only the corrected answer."
        )

        try:
            return await DynamicLLMClient.generate_async(
                "You are a strict scientific peer reviewer.",
                prompt,
                spec,
            )
        except Exception:
            return candidate_response


# =============================================================================
# QUALITY EVALUATION
# =============================================================================

class QualityEvaluator:
    @staticmethod
    def _readability(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0

        score = 90.0
        if 120 <= len(words) <= 700:
            score += 5.0
        if re.search(r"\n\s*[-*]\s+", text) or re.search(r"\n\s*\d+\.", text):
            score += 3.0
        if len(max(text.splitlines(), key=len, default="")) > 240:
            score -= 5.0
        return round(max(0.0, min(100.0, score)), 1)

    @staticmethod
    def evaluate(
        query: str,
        response: str,
        report: SemanticGroundingReport,
        task_type: OrchestratorTaskIntent,
    ) -> QualityBreakdown:
        if report.total_claims == 0:
            return QualityBreakdown(0, 0, 0, 0, 0, 0, 0, 0)

        verified = [v for v in report.claim_verifications if v.status == "VERIFIED"]

        citation_correctness = round(
            len(verified) / max(1, report.total_claims) * 100.0, 1
        )

        coverage = report.evidence_coverage_pct

        evidence_usage = round(
            (
                len(
                    [
                        v
                        for v in report.claim_verifications
                        if v.best_ref and v.status in {"VERIFIED", "PARTIAL"}
                    ]
                )
                / max(1, report.total_claims)
            )
            * 100.0,
            1,
        )

        consistency = round(
            max(
                0.0,
                100.0
                - (report.unsupported_claims_count * 30.0)
                - (len(report.invalid_citations) * 20.0),
            ),
            1,
        )

        readability = QualityEvaluator._readability(response)

        words = len(response.split())
        completeness = 100.0
        if words < 80:
            completeness -= 25.0
        if task_type == OrchestratorTaskIntent.METHOD_COMPARISON:
            required_terms = ["compare", "trade-off", "architecture"]
            matched = sum(1 for term in required_terms if term in response.lower())
            completeness = min(100.0, completeness - (3 - matched) * 8.0)

        final = (
            report.grounding_score * 0.30
            + citation_correctness * 0.20
            + coverage * 0.15
            + evidence_usage * 0.10
            + consistency * 0.10
            + readability * 0.08
            + completeness * 0.07
        )

        if report.unsupported_claims_count > 0:
            final = min(final, 95.0)
        if report.invalid_citations:
            final = min(final, 95.0)

        return QualityBreakdown(
            grounding_score=round(report.grounding_score, 1),
            citation_correctness=round(citation_correctness, 1),
            coverage_score=round(coverage, 1),
            evidence_usage=round(evidence_usage, 1),
            consistency_score=round(consistency, 1),
            readability_score=round(readability, 1),
            completeness_score=round(completeness, 1),
            final_quality_score=round(min(100.0, final), 1),
        )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class LLMOrchestratorEngine:
    def __init__(self) -> None:
        setup_directories()
        self.grounding_engine = SemanticGroundingEngine()

    @staticmethod
    def _build_prompts(
        query: str,
        prompt_payload: Any,
        evidence_blocks: Sequence[EvidenceBlock],
    ) -> Tuple[str, str]:
        system_prompt = (
            "You are an expert scientific research assistant. "
            "Answer strictly and exclusively from the supplied evidence. "
            "Every factual claim must include a valid [Ref-X] citation that directly supports it. "
            "EVIDENCE BOUNDARY: Do not use world knowledge or extrapolation. "
            "If evidence is insufficient to establish a claim, explicitly state: "
            "'The retrieved evidence is insufficient to establish this claim.'"
        )

        if isinstance(prompt_payload, str):
            user_prompt = prompt_payload
        elif prompt_payload is not None:
            user_prompt = (
                _stringify(_get_attr_or_key(prompt_payload, "final_prompt", ""))
                or _stringify(_get_attr_or_key(prompt_payload, "user_prompt", ""))
                or query
            )
            supplied_system = _stringify(
                _get_attr_or_key(prompt_payload, "system_prompt", "")
            ).strip()
            if supplied_system:
                system_prompt = supplied_system + "\n\n" + system_prompt
        else:
            user_prompt = query

        evidence_text = _format_evidence(evidence_blocks)

        if evidence_text:
            user_prompt = (
                f"QUERY:\n{query}\n\n"
                f"INSTRUCTIONS:\n{user_prompt}\n\n"
                f"VERIFIABLE EVIDENCE:\n{evidence_text}\n\n"
                "CITATION RULE: Append explicit [Ref-X] citation tags directly to every individual claim sentence using the anchors present in VERIFIABLE EVIDENCE."
            )
        else:
            user_prompt = (
                f"QUERY:\n{query}\n\n"
                f"INSTRUCTIONS:\n{user_prompt}\n\n"
                "No evidence was supplied. State that evidence is insufficient."
            )

        return system_prompt, user_prompt

    async def orchestrate_async(
        self,
        query: str,
        prompt_payload: Optional[Union[MultiPromptPayload, str]] = None,
        context_package: Optional[OptimizedContextPackage] = None,
        task_type: OrchestratorTaskIntent = OrchestratorTaskIntent.QUESTION_ANSWERING,
        mode: ExecutionMode = OrchestratorConfig.DEFAULT_MODE,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> OrchestratedResponse:
        start = time.perf_counter()
        execution_id = f"EXEC-{uuid.uuid4().hex[:8].upper()}"
        stage_latencies: Dict[str, float] = {}

        t0 = time.perf_counter()
        evidence_blocks = _extract_evidence(context_package, prompt_payload)
        system_prompt, user_prompt = self._build_prompts(
            query, prompt_payload, evidence_blocks
        )
        prompt_tokens = max(1, int(len(user_prompt.split()) * 1.33))
        stage_latencies["context_normalization_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        t0 = time.perf_counter()
        spec, routing_scores = ExplainableModelRouter.route(
            task_type,
            prompt_tokens,
            preferred_provider,
        )
        stage_latencies["routing_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        t0 = time.perf_counter()
        models_called: List[str] = []
        fallback_occurred = False
        fallback_reason = ""
        retries = 0

        try:
            if mode == ExecutionMode.MULTI_MODEL_ENSEMBLE:
                specs = [spec]
                available = ExplainableModelRouter.available_model_names()
                if (
                    OrchestratorConfig.DEEPSEEK_MODEL in available
                    and spec.provider != LLMProvider.DEEPSEEK
                ):
                    specs.append(
                        LLMModelSpec(
                            LLMProvider.DEEPSEEK,
                            OrchestratorConfig.DEEPSEEK_MODEL,
                            0.10,
                            3072,
                            routing_scores.get(OrchestratorConfig.DEEPSEEK_MODEL, 0.0),
                        )
                    )
                if "ollama-local" in available and spec.provider != LLMProvider.OLLAMA:
                    specs.append(
                        LLMModelSpec(
                            LLMProvider.OLLAMA,
                            OrchestratorConfig.OLLAMA_MODEL,
                            0.10,
                            3072,
                            routing_scores.get("ollama-local", 0.0),
                        )
                    )
                raw_response, models_called = await MultiModelEnsembleEngine.run_ensemble(
                    system_prompt,
                    user_prompt,
                    specs,
                )
            else:
                raw_response = await DynamicLLMClient.generate_async(
                    system_prompt,
                    user_prompt,
                    spec,
                )
                models_called = [spec.model_name]
        except Exception as exc:
            fallback_occurred = True
            fallback_reason = str(exc)[:500]
            retries = OrchestratorConfig.MAX_RETRIES
            logger.warning(
                "Primary LLM backend unavailable; switching to deterministic local fallback: %s",
                fallback_reason,
            )
            raw_response = DynamicLLMClient._generate_local_fallback(user_prompt)
            models_called.append("local-fallback")
            spec = LLMModelSpec(
                provider=LLMProvider.LOCAL_FALLBACK,
                model_name="local-fallback",
                temperature=0.0,
                max_tokens=4096,
                routing_score=0.0,
            )

        stage_latencies["llm_execution_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        t0 = time.perf_counter()
        grounding = self.grounding_engine.verify_grounding(
            raw_response,
            evidence_blocks,
        )
        stage_latencies["grounding_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        judge_applied = False
        refinement_loops = 0

        if mode == ExecutionMode.AGENTIC_JUDGE_LOOP:
            while (
                refinement_loops < OrchestratorConfig.MAX_REFINEMENT_LOOPS
                and (
                    grounding.grounding_score < OrchestratorConfig.GROUNDING_PASS_THRESHOLD
                    or grounding.unsupported_claims_count > 0
                    or grounding.invalid_citations
                )
            ):
                t0 = time.perf_counter()
                judge_applied = True
                refinement_loops += 1

                raw_response = await JudgeModelSelfReflector.refine(
                    raw_response,
                    user_prompt,
                    grounding,
                    spec,
                )
                grounding = self.grounding_engine.verify_grounding(
                    raw_response,
                    evidence_blocks,
                )

                stage_latencies[f"judge_refinement_{refinement_loops}_ms"] = round(
                    (time.perf_counter() - t0) * 1000.0, 2
                )

        t0 = time.perf_counter()
        completion_tokens = max(1, int(len(raw_response.split()) * 1.33))
        pricing = OrchestratorConfig.MODEL_PRICING.get(spec.model_name, (0.0, 0.0))
        estimated_cost = round(
            (prompt_tokens / 1000.0) * pricing[0]
            + (completion_tokens / 1000.0) * pricing[1],
            6,
        )

        cost_report = CostReport(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=estimated_cost,
            pricing_model=spec.model_name,
        )

        quality = QualityEvaluator.evaluate(
            query=query,
            response=raw_response,
            report=grounding,
            task_type=task_type,
        )
        stage_latencies["quality_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        total_ms = round((time.perf_counter() - start) * 1000.0, 2)

        trace = ExecutionTraceRecord(
            execution_id=execution_id,
            mode=mode.value,
            routed_provider=spec.provider.value,
            routed_model=spec.model_name,
            routing_scores=routing_scores,
            routing_reason=(
                f"Task={task_type.value}; prompt_tokens={prompt_tokens}; "
                f"evidence_blocks={len(evidence_blocks)}"
            ),
            judge_critique_applied=judge_applied,
            refinement_loops=refinement_loops,
            ensemble_models_called=models_called,
            retries=retries,
            fallback_occurred=fallback_occurred,
            fallback_reason=fallback_reason,
            latency_ms=total_ms,
            stage_latencies_ms=stage_latencies,
        )

        result = OrchestratedResponse(
            execution_id=execution_id,
            query=query,
            task_type=task_type,
            generated_response=raw_response,
            model_spec=spec,
            cost_report=cost_report,
            grounding_report=grounding,
            quality=quality,
            execution_trace=trace,
        )

        logger.info(
            "Orchestration [%s] complete in %.2f ms. Routed to [%s] "
            "(Score: %.2f). Quality: %.1f/100. Grounding: %.1f%%.",
            execution_id,
            total_ms,
            spec.model_name,
            spec.routing_score,
            quality.final_quality_score,
            grounding.grounding_score,
        )
        return result

    def orchestrate(
        self,
        query: str,
        prompt_payload: Optional[Union[MultiPromptPayload, str]] = None,
        context_package: Optional[OptimizedContextPackage] = None,
        task_type: OrchestratorTaskIntent = OrchestratorTaskIntent.QUESTION_ANSWERING,
        mode: ExecutionMode = OrchestratorConfig.DEFAULT_MODE,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> OrchestratedResponse:
        return asyncio.run(
            self.orchestrate_async(
                query=query,
                prompt_payload=prompt_payload,
                context_package=context_package,
                task_type=task_type,
                mode=mode,
                preferred_provider=preferred_provider,
            )
        )


# =============================================================================
# EXPORTS & VALIDATION
# =============================================================================

def export_orchestrated_response(
    res: OrchestratedResponse,
    file_prefix: str = "orchestration_response",
) -> Tuple[str, str, str]:
    setup_directories()

    json_path = OrchestratorConfig.REPORTS_DIR / f"{file_prefix}.json"
    md_path = OrchestratorConfig.REPORTS_DIR / f"{file_prefix}.md"
    txt_path = OrchestratorConfig.REPORTS_DIR / f"{file_prefix}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, ensure_ascii=False, default=str)

    q = res.quality
    md = (
        f"# LLM ORCHESTRATION EXECUTION REPORT [{res.execution_id}]\n\n"
        f"**Query:** {res.query}\n\n"
        f"**Task:** `{res.task_type.value}`\n\n"
        f"**Model:** `{res.model_spec.model_name}` "
        f"({res.model_spec.provider.value})\n\n"
        f"## Quality Breakdown\n"
        f"- Final quality: **{q.final_quality_score}/100**\n"
        f"- Grounding: {q.grounding_score}%\n"
        f"- Citation correctness: {q.citation_correctness}%\n"
        f"- Evidence coverage: {q.coverage_score}%\n"
        f"- Evidence usage: {q.evidence_usage}%\n"
        f"- Consistency: {q.consistency_score}%\n"
        f"- Readability: {q.readability_score}%\n"
        f"- Completeness: {q.completeness_score}%\n\n"
        f"## Grounding\n"
        f"- Claims: {res.grounding_report.total_claims}\n"
        f"- Verified: {res.grounding_report.grounded_claims_count}\n"
        f"- Partial: {res.grounding_report.partial_claims_count}\n"
        f"- Unsupported: {res.grounding_report.unsupported_claims_count}\n"
        f"- Verified citations: {', '.join(res.grounding_report.verified_citations) or 'None'}\n\n"
        f"## Generated Response\n\n{res.generated_response}\n"
    )

    md_path.write_text(md, encoding="utf-8")
    txt_path.write_text(res.generated_response, encoding="utf-8")

    return str(json_path), str(md_path), str(txt_path)


def _validation_evidence() -> List[EvidenceBlock]:
    return [
        EvidenceBlock(
            ref_id="Ref-1",
            title="Transformer attention evidence",
            text=(
                "Vision Transformer architectures divide an image into patches and "
                "use self-attention to model relationships between patches. "
                "Self-attention can connect distant image regions rather than relying "
                "only on a local convolutional receptive field."
            ),
        ),
        EvidenceBlock(
            ref_id="Ref-2",
            title="Autonomous perception trade-off evidence",
            text=(
                "For autonomous perception, transformer-based backbones can provide "
                "global context modeling, while convolutional backbones are often "
                "computationally efficient because local convolutions exploit spatial "
                "structure. Therefore, the practical comparison is a trade-off "
                "involving accuracy, context modeling, memory, and latency."
            ),
        ),
        EvidenceBlock(
            ref_id="Ref-3",
            title="Evidence limitation",
            text=(
                "Architecture comparisons should not claim universal superiority "
                "without specifying the dataset, task, implementation, hardware, "
                "training configuration, and evaluation metric."
            ),
        ),
    ]


class _DeterministicValidationClient:
    @staticmethod
    async def generate(
        system_prompt: str,
        user_prompt: str,
        spec: LLMModelSpec,
    ) -> str:
        return (
            "### Evidence-Grounded Comparison\n\n"
            "Vision Transformer architectures divide an image into patches and "
            "use self-attention to model relationships between patches [Ref-1]. "
            "Self-attention can connect distant image regions rather than relying "
            "only on a local convolutional receptive field [Ref-1].\n\n"
            "For autonomous perception, transformer-based backbones can provide "
            "global context modeling, while convolutional backbones are often "
            "computationally efficient because local convolutions exploit spatial "
            "structure [Ref-2]. Therefore, the practical comparison is a trade-off "
            "involving accuracy, context modeling, memory, and latency [Ref-2].\n\n"
            "Architecture comparisons should not claim universal superiority "
            "without specifying the dataset, task, implementation, hardware, "
            "training configuration, and evaluation metric [Ref-3]."
        )


def validate_orchestrator(orchestrator: LLMOrchestratorEngine) -> pd.DataFrame:
    logger.info("Running Automated LLM Orchestrator Quality Validation Suite...")

    evidence = _validation_evidence()
    evidence_text = _format_evidence(evidence)

    response_text = asyncio.run(
        _DeterministicValidationClient.generate(
            "scientific system",
            evidence_text,
            LLMModelSpec(
                LLMProvider.LOCAL_FALLBACK,
                "validation-fixture",
                0.0,
                2048,
            ),
        )
    )

    report = orchestrator.grounding_engine.verify_grounding(
        response_text,
        evidence,
    )
    quality = QualityEvaluator.evaluate(
        query="Compare attention mechanisms and vision transformers for autonomous perception",
        response=response_text,
        report=report,
        task_type=OrchestratorTaskIntent.METHOD_COMPARISON,
    )

    records = [
        {
            "validation_test": "response_generation",
            "expected": "> 100 characters",
            "actual": f"{len(response_text)} characters",
            "status": "PASS" if len(response_text) > 100 else "FAIL",
        },
        {
            "validation_test": "claim_extraction",
            "expected": ">= 3 claims",
            "actual": f"{report.total_claims} claims",
            "status": "PASS" if report.total_claims >= 3 else "FAIL",
        },
        {
            "validation_test": "semantic_grounding",
            "expected": f">= {OrchestratorConfig.GROUNDING_PASS_THRESHOLD}%",
            "actual": f"{report.grounding_score}%",
            "status": "PASS" if report.grounding_score >= OrchestratorConfig.GROUNDING_PASS_THRESHOLD else "FAIL",
        },
        {
            "validation_test": "citation_correctness",
            "expected": ">= 90% valid cited claims",
            "actual": f"{quality.citation_correctness}%",
            "status": "PASS" if quality.citation_correctness >= 90.0 else "FAIL",
        },
        {
            "validation_test": "unsupported_claims",
            "expected": "0",
            "actual": str(report.unsupported_claims_count),
            "status": "PASS" if report.unsupported_claims_count == 0 else "FAIL",
        },
        {
            "validation_test": "invalid_citations",
            "expected": "0",
            "actual": str(len(report.invalid_citations)),
            "status": "PASS" if not report.invalid_citations else "FAIL",
        },
        {
            "validation_test": "evidence_coverage",
            "expected": ">= 90%",
            "actual": f"{quality.coverage_score}%",
            "status": "PASS" if quality.coverage_score >= 90.0 else "FAIL",
        },
        {
            "validation_test": "final_quality",
            "expected": "All thresholds met",
            "actual": f"quality={quality.final_quality_score}/100",
            "status": "PASS" if quality.final_quality_score >= 90.0 else "FAIL",
        },
    ]

    df = pd.DataFrame(records)
    df.to_csv(OrchestratorConfig.REPORTS_DIR / "validation_report.csv", index=False)
    return df


def generate_orchestrator_visualizations(res: OrchestratedResponse) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    setup_directories()
    q = res.quality
    names = ["Grounding", "Citation", "Coverage", "Evidence", "Consistency", "Readability", "Completeness", "Final"]
    values = [q.grounding_score, q.citation_correctness, q.coverage_score, q.evidence_usage, q.consistency_score, q.readability_score, q.completeness_score, q.final_quality_score]

    plt.figure(figsize=(10, 5))
    plt.bar(names, values, edgecolor="black")
    plt.ylim(0, 105)
    plt.ylabel("Score")
    plt.title("LLM Orchestrator Quality Breakdown")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OrchestratorConfig.PLOTS_DIR / "quality_breakdown.png", dpi=250)
    plt.close()


def main() -> None:
    setup_directories()
    orchestrator = LLMOrchestratorEngine()
    test_query = "Compare attention mechanisms and vision transformers for autonomous perception"
    evidence = _validation_evidence()

    demo_prompt = {
        "final_prompt": "Compare the mechanisms, perception implications, and practical trade-offs using only the supplied evidence.",
        "system_prompt": "You are an expert AI research assistant. Use only the supplied evidence.",
        "chunks": [{"ref_id": b.ref_id, "chunk_text": b.text, "title": b.title} for b in evidence],
    }

    result = orchestrator.orchestrate(
        query=test_query,
        prompt_payload=demo_prompt,
        task_type=OrchestratorTaskIntent.METHOD_COMPARISON,
        mode=ExecutionMode.AGENTIC_JUDGE_LOOP,
        preferred_provider=LLMProvider.OLLAMA,
    )

    export_orchestrated_response(result, file_prefix="vision_transformers_orchestration")
    generate_orchestrator_visualizations(result)
    validation_df = validate_orchestrator(orchestrator)
    all_passed = bool((validation_df["status"] == "PASS").all())

    logger.info("========== LLM ORCHESTRATOR SUMMARY ==========")
    logger.info("Execution ID         : %s", result.execution_id)
    logger.info("Final Quality Score  : %.1f/100", result.quality.final_quality_score)
    logger.info("Semantic Grounding   : %.1f%%", result.grounding_report.grounding_score)
    logger.info("Citation Correctness : %.1f%%", result.quality.citation_correctness)
    logger.info("Validation Suite     : %s", "ALL CHECKS PASSED" if all_passed else "VALIDATION FAILED")
    logger.info("===============================================")


if __name__ == "__main__":
    main()