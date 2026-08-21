from __future__ import annotations

import collections
import datetime as dt
import enum
import json
import logging
import math
import os
import pathlib
import re
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore

try:
    from src.rag.semantic_search_engine import SemanticSearchEngine
except ImportError:
    try:
        from semantic_search_engine import SemanticSearchEngine
    except ImportError:
        SemanticSearchEngine = None  # type: ignore


# =============================================================================
# Utility functions
# =============================================================================

def estimate_token_count(text: str) -> int:
    """Conservative token estimate.

    This is intentionally a budget estimator, not a tokenizer replacement.
    For exact LLM accounting, downstream orchestration should use the target
    model tokenizer.
    """
    if not text or not isinstance(text, str):
        return 0
    words = text.strip().split()
    if not words:
        return 0
    return max(1, int(math.ceil(len(words) / 0.75)))


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]{2,}\b", text.lower()))


def lexical_overlap(query: str, text: str) -> float:
    q = tokenize(query)
    d = tokenize(text)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def jaccard_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# =============================================================================
# Configuration
# =============================================================================

class TaskType(str, enum.Enum):
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    PAPER_COMPARISON = "PAPER_COMPARISON"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    RESEARCH_PROPOSAL = "RESEARCH_PROPOSAL"
    PATENT_SEARCH = "PATENT_SEARCH"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    RESEARCH_GAP = "RESEARCH_GAP"
    SURVEY_WRITING = "SURVEY_WRITING"
    EXPERIMENTAL_DESIGN = "EXPERIMENTAL_DESIGN"
    DATASET_RECOMMENDATION = "DATASET_RECOMMENDATION"


class RetrievalProfile(str, enum.Enum):
    PRECISION_ORIENTED = "PRECISION_ORIENTED"
    COVERAGE_ORIENTED = "COVERAGE_ORIENTED"
    NOVELTY_ORIENTED = "NOVELTY_ORIENTED"
    PRIOR_ART = "PRIOR_ART"


@dataclass
class RetrieverConfig:
    """Centralized runtime configuration.

    Values can be overridden through environment variables without changing
    the Python integration contract.
    """

    PIPELINE_VERSION: str = "4.0.0-PROD"

    DEFAULT_TOP_K_PAPERS: int = 15
    DEFAULT_TOP_K_CHUNKS: int = 10
    DEFAULT_TOKEN_BUDGET: int = 4000

    # Retrieve many candidates before reranking.
    CANDIDATE_MULTIPLIER: int = 5
    MIN_CANDIDATE_K: int = 50
    MAX_CANDIDATE_K: int = 100

    # Cross encoder
    RERANKER_MODEL: str = os.getenv(
        "RETRIEVER_RERANKER_MODEL",
        "BAAI/bge-reranker-base",
    )
    RERANK_BATCH_SIZE: int = int(os.getenv("RETRIEVER_RERANK_BATCH", "16"))
    ENABLE_CROSS_ENCODER: bool = os.getenv(
        "RETRIEVER_ENABLE_CROSS_ENCODER", "1"
    ).lower() not in {"0", "false", "no"}

    # Evidence selection
    MAX_CHUNKS_PER_PAPER: int = 2
    JACCARD_DEDUP_THRESHOLD: float = 0.82
    SEMANTIC_DUPLICATE_THRESHOLD: float = 0.92

    # Quality scoring. These are intentionally moderate: similarity remains
    # dominant while quality/completeness and task-aware recency contribute.
    SIMILARITY_WEIGHT: float = 0.42
    RERANK_WEIGHT: float = 0.28
    COMPLETENESS_WEIGHT: float = 0.15
    RECENCY_WEIGHT: float = 0.10
    DIVERSITY_BONUS_WEIGHT: float = 0.05

    # Scientific reasoning
    MIN_CONSENSUS_SOURCES: int = 3

    # Paths
    PROJECT_ROOT: str = os.getenv(
        "AI_RPA_PROJECT_ROOT",
        str(pathlib.Path(__file__).resolve().parents[3]),
    )
    REPORTS_DIR: str = os.getenv(
        "RETRIEVER_REPORTS_DIR",
        str(
            pathlib.Path(
                os.getenv(
                    "AI_RPA_PROJECT_ROOT",
                    str(pathlib.Path(__file__).resolve().parents[3]),
                )
            )
            / "outputs"
            / "reports"
            / "retriever"
        ),
    )

    @property
    def PLOTS_DIR(self) -> str:
        return str(pathlib.Path(self.REPORTS_DIR) / "plots")

    @property
    def LOGS_DIR(self) -> str:
        return str(pathlib.Path(self.REPORTS_DIR) / "logs")


CONFIG = RetrieverConfig()


# =============================================================================
# Metadata schema
# =============================================================================

COLUMN_MAP: Dict[str, List[str]] = {
    "paper_id": ["Paper ID", "paper_id", "id", "arxiv_id"],
    "title": ["Title", "title", "paper_title"],
    "abstract": [
        "Abstract",
        "abstract",
        "summary",
        "Summary",
        "combined_text",
        "Combined Text",
        "search_text",
    ],
    "combined_text": [
        "combined_text",
        "Combined Text",
        "search_text",
        "full_text",
        "text",
        "Abstract",
        "abstract",
        "summary",
    ],
    "authors": ["Authors", "authors", "first_author"],
    "category": ["Category", "category", "primary_category", "category_code"],
    "broad_domain": ["Broad Domain", "broad_domain", "domain"],
    "publication_year": [
        "Publication Year",
        "publication_year",
        "published_year",
        "year",
    ],
    "rank": ["Rank", "rank", "dense_rank"],
    "similarity_score": ["Similarity Score", "similarity_score", "score"],
    "confidence_score": [
        "Confidence Score (%)",
        "confidence_score",
        "confidence",
    ],
}


def get_field(row: pd.Series, field_key: str, default_val: Any = "") -> Any:
    aliases = COLUMN_MAP.get(field_key, [field_key])
    for col in aliases:
        if col in row.index:
            value = row[col]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value or value.upper() == "N/A":
                    continue
            return value
    return default_val


def get_evidence_text(row: pd.Series) -> str:
    """Prefer actual combined text/abstract and avoid title-only evidence."""
    candidates = [
        ("combined_text", get_field(row, "combined_text", "")),
        ("abstract", get_field(row, "abstract", "")),
    ]

    title = normalize_text(get_field(row, "title", ""))
    best = ""

    for _, value in candidates:
        value = normalize_text(value)
        if value and len(value) > len(best):
            best = value

    # If the dataset's combined_text is only the title, keep title as a last
    # resort but mark the chunk as low-completeness later.
    if not best:
        best = title

    return best


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("RetrieverPipelineEngine")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        pathlib.Path(CONFIG.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(CONFIG.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.FileHandler(
            pathlib.Path(CONFIG.LOGS_DIR) / "retriever_pipeline.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def setup_directories() -> None:
    pathlib.Path(CONFIG.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(CONFIG.PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(CONFIG.LOGS_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data contracts
# =============================================================================

@dataclass
class QueryUnderstandingResult:
    original_query: str
    expanded_query: str
    task_type: TaskType
    retrieval_profile: RetrievalProfile
    entities: List[str]
    concepts: List[str]
    confidence: float = 0.0


@dataclass
class EvidenceChunk:
    paper_id: str
    chunk_id: str
    title: str
    authors: str
    category: str
    broad_domain: str
    publication_year: int
    chunk_text: str
    section_name: str
    original_rank: int
    similarity_score: float
    confidence_score: float
    word_count: int
    estimated_tokens: int
    evidence_quality_score: float = 0.0
    rerank_score: float = 0.0
    lexical_score: float = 0.0
    completeness_score: float = 0.0
    selection_reason: str = ""


@dataclass
class ConflictReport:
    has_conflicts: bool
    conflict_summary: str
    supporting_evidence_ids: List[str]
    opposing_evidence_ids: List[str]
    potential_reasons: List[str]
    confidence: float


@dataclass
class ConsensusReport:
    consensus_level: str
    consensus_percentage: float
    supporting_paper_ids: List[str]
    opposing_paper_ids: List[str]


@dataclass
class CoverageReport:
    overall_coverage_pct: float
    topic_scores: Dict[str, float]
    missing_topics: List[str]
    iterative_queries_suggested: List[str]


@dataclass
class RetrievalIRMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    map_score: float
    ndcg_at_k: float
    coverage_score: float
    diversity_score: float
    citation_density: float
    evaluation_mode: str = "PROXY_ESTIMATE"


@dataclass
class ReproducibilitySnapshot:
    retriever_version: str
    pipeline_version: str
    timestamp: str
    config_snapshot: Dict[str, Any]
    evidence_ids: List[str]
    random_seed: int = 42


@dataclass
class EvidencePackage:
    query: str
    query_understanding: QueryUnderstandingResult
    chunks: List[EvidenceChunk]
    formatted_context: str
    conflict_report: ConflictReport
    consensus_report: ConsensusReport
    coverage_report: CoverageReport
    relationship_graph: Dict[str, Any]
    confidence_calibration: Dict[str, float]
    ir_metrics: RetrievalIRMetrics
    retrieval_trace: Dict[str, Any]
    reproducibility: ReproducibilitySnapshot


# =============================================================================
# Query understanding
# =============================================================================

class QueryUnderstandingEngine:
    ONTOLOGY_EXPANSIONS: Dict[str, List[str]] = {
        "vit": ["Vision Transformer", "ViT", "Transformer Encoder", "Self Attention"],
        "rag": ["Retrieval Augmented Generation", "Dense Retrieval", "Knowledge Grounding"],
        "llm": ["Large Language Model", "Foundation Model", "Transformer Language Model"],
        "yolo": ["You Only Look Once", "Real-Time Object Detection", "Bounding Box Detection"],
        "gnn": ["Graph Neural Network", "Graph Convolution", "Message Passing"],
        "cnn": ["Convolutional Neural Network", "Convolution", "Visual Feature Extractor"],
        "nlp": ["Natural Language Processing", "Language Modeling", "Text Representation"],
        "ocr": ["Optical Character Recognition", "Text Recognition", "Document Understanding"],
    }

    TASK_PATTERNS: Dict[TaskType, Tuple[str, ...]] = {
        TaskType.RESEARCH_GAP: (
            "research gap", "gap", "unsolved", "open problem", "limitation",
            "what is missing", "future work",
        ),
        TaskType.LITERATURE_REVIEW: (
            "literature review", "systematic review", "survey", "overview",
            "state of the art", "state-of-the-art", "recent advances",
        ),
        TaskType.PAPER_COMPARISON: (
            "compare", "comparison", "versus", "vs", "difference between",
            "better than", "trade-off",
        ),
        TaskType.PATENT_SEARCH: ("patent", "prior art", "prior-art"),
        TaskType.TREND_ANALYSIS: (
            "trend", "trends", "evolution", "over time", "recent developments",
        ),
        TaskType.DATASET_RECOMMENDATION: (
            "dataset", "datasets", "benchmark", "corpus",
        ),
        TaskType.EXPERIMENTAL_DESIGN: (
            "experiment", "experimental design", "ablation", "evaluation protocol",
        ),
        TaskType.RESEARCH_PROPOSAL: (
            "research proposal", "propose", "proposed approach", "research idea",
        ),
        TaskType.SURVEY_WRITING: (
            "write a survey", "survey paper", "survey article",
        ),
    }

    @classmethod
    def process_query(cls, query: str) -> QueryUnderstandingResult:
        original = normalize_text(query)
        q_lower = original.lower()

        words = re.findall(r"\b\w+\b", q_lower)

        expansions: List[str] = [original]
        for word in words:
            expansions.extend(cls.ONTOLOGY_EXPANSIONS.get(word, []))

        # Phrase-level scientific expansion.
        phrase_expansions = {
            "vision transformers": [
                "ViT", "visual transformer", "transformer-based vision",
                "self-attention for vision",
            ],
            "autonomous perception": [
                "autonomous driving perception",
                "autonomous vehicle perception",
                "BEV perception",
                "sensor fusion",
                "3D object detection",
            ],
            "attention mechanisms": [
                "self-attention",
                "multi-head attention",
                "cross-attention",
            ],
        }

        for phrase, values in phrase_expansions.items():
            if phrase in q_lower:
                expansions.extend(values)

        expanded = " ".join(dict.fromkeys(x for x in expansions if x))

        scores: Dict[TaskType, float] = collections.defaultdict(float)

        for task, patterns in cls.TASK_PATTERNS.items():
            for pattern in patterns:
                if pattern in q_lower:
                    scores[task] += 1.0

        # Questions that contain comparison language receive comparison intent.
        if re.search(r"\b(compare|versus|vs|difference|trade[- ]off)\b", q_lower):
            scores[TaskType.PAPER_COMPARISON] += 2.0

        # "What datasets/results/limitations..." should not be ordinary QA
        # when the query is clearly asking about a scientific dimension.
        if re.search(r"\b(dataset|benchmark)\b", q_lower):
            scores[TaskType.DATASET_RECOMMENDATION] += 1.0

        if scores:
            task = max(scores, key=scores.get)
            confidence = min(1.0, scores[task] / max(2.0, sum(scores.values())))
        else:
            task = TaskType.QUESTION_ANSWERING
            confidence = 0.75

        profile_map = {
            TaskType.RESEARCH_GAP: RetrievalProfile.NOVELTY_ORIENTED,
            TaskType.LITERATURE_REVIEW: RetrievalProfile.COVERAGE_ORIENTED,
            TaskType.SURVEY_WRITING: RetrievalProfile.COVERAGE_ORIENTED,
            TaskType.PAPER_COMPARISON: RetrievalProfile.PRECISION_ORIENTED,
            TaskType.PATENT_SEARCH: RetrievalProfile.PRIOR_ART,
            TaskType.TREND_ANALYSIS: RetrievalProfile.COVERAGE_ORIENTED,
            TaskType.DATASET_RECOMMENDATION: RetrievalProfile.COVERAGE_ORIENTED,
            TaskType.RESEARCH_PROPOSAL: RetrievalProfile.COVERAGE_ORIENTED,
            TaskType.EXPERIMENTAL_DESIGN: RetrievalProfile.PRECISION_ORIENTED,
            TaskType.QUESTION_ANSWERING: RetrievalProfile.PRECISION_ORIENTED,
        }

        # Entity extraction is deliberately conservative.
        stop = {
            "about", "which", "where", "what", "when", "whose", "these",
            "those", "using", "with", "from", "into", "than", "between",
        }
        entities = []
        for word in words:
            if len(word) > 4 and word not in stop:
                entities.append(word.capitalize())

        entities = list(dict.fromkeys(entities))[:20]

        return QueryUnderstandingResult(
            original_query=original,
            expanded_query=expanded,
            task_type=task,
            retrieval_profile=profile_map[task],
            entities=entities,
            concepts=list(dict.fromkeys(expansions))[:30],
            confidence=round(confidence, 4),
        )


# =============================================================================
# Reranking
# =============================================================================

class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        papers_df: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        raise NotImplementedError


class HybridReranker(BaseReranker):
    """Deterministic reranker used when a neural cross encoder is unavailable."""

    def rerank(
        self,
        query: str,
        papers_df: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        if papers_df.empty:
            return papers_df.copy()

        df = papers_df.copy()

        semantic = [
            safe_float(get_field(row, "similarity_score", 0.0))
            for _, row in df.iterrows()
        ]
        semantic_norm = minmax(semantic)

        rerank_scores = []
        lexical_scores = []

        for i, (_, row) in enumerate(df.iterrows()):
            text = (
                normalize_text(get_field(row, "title", ""))
                + " "
                + get_evidence_text(row)
            )
            lex = lexical_overlap(query, text)
            lexical_scores.append(lex)

            base = semantic_norm[i] if i < len(semantic_norm) else 0.5
            score = 0.75 * base + 0.25 * lex
            rerank_scores.append(score)

        df["_rerank_score"] = rerank_scores
        df["_lexical_score"] = lexical_scores

        return (
            df.sort_values("_rerank_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker with deterministic hybrid fallback."""

    def __init__(
        self,
        model_name: str = CONFIG.RERANKER_MODEL,
        batch_size: int = CONFIG.RERANK_BATCH_SIZE,
        enabled: bool = CONFIG.ENABLE_CROSS_ENCODER,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.enabled = enabled
        self.model = None
        self.backend = "HYBRID_FALLBACK"
        self.load_error = ""

    def _lazy_load(self) -> bool:
        if not self.enabled:
            return False

        if self.model is not None:
            return True

        if CrossEncoder is None:
            self.load_error = "sentence-transformers.CrossEncoder is unavailable"
            return False

        try:
            logger.info(
                "Loading Cross-Encoder reranker: %s",
                self.model_name,
            )
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
            )
            self.backend = "CROSS_ENCODER"
            return True
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning(
                "Cross-Encoder unavailable; using hybrid fallback. Reason: %s",
                exc,
            )
            self.model = None
            self.backend = "HYBRID_FALLBACK"
            return False

    def rerank(
        self,
        query: str,
        papers_df: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        if papers_df.empty:
            return papers_df.copy()

        if not self._lazy_load():
            return HybridReranker().rerank(query, papers_df, top_k)

        df = papers_df.copy()

        pairs = []
        lexical_scores = []

        for _, row in df.iterrows():
            title = normalize_text(get_field(row, "title", ""))
            text = get_evidence_text(row)

            # Keep the cross encoder input bounded.
            evidence = text[:5000]
            candidate = f"{title}\n{evidence}".strip()
            pairs.append((query, candidate))
            lexical_scores.append(lexical_overlap(query, candidate))

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = np.asarray(scores, dtype=float).reshape(-1)

            # CrossEncoder scores are model-dependent and not guaranteed to
            # be probabilities. Min-max normalization is only for blending.
            ce_norm = minmax(scores.tolist())

            # Small lexical component makes exact scientific terms matter
            # without dominating semantic relevance.
            hybrid = [
                0.88 * ce + 0.12 * lex
                for ce, lex in zip(ce_norm, lexical_scores)
            ]

            df["_rerank_score"] = hybrid
            df["_cross_encoder_score"] = scores
            df["_lexical_score"] = lexical_scores

            return (
                df.sort_values("_rerank_score", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )

        except Exception as exc:
            logger.warning(
                "Cross-Encoder scoring failed; using hybrid fallback: %s",
                exc,
            )
            self.backend = "HYBRID_FALLBACK"
            self.load_error = str(exc)
            return HybridReranker().rerank(query, papers_df, top_k)


# =============================================================================
# Adaptive chunking
# =============================================================================

class AdaptiveDocumentChunker:
    """Creates evidence chunks from abstract/combined text.

    The previous implementation could silently reduce a paper to its title.
    This version strongly prefers combined_text and preserves enough context
    for downstream LLM grounding.
    """

    SECTION_PATTERNS = (
        "abstract",
        "introduction",
        "method",
        "methodology",
        "experiment",
        "experiments",
        "results",
        "discussion",
        "limitation",
        "conclusion",
        "dataset",
    )

    @classmethod
    def _infer_section(cls, text: str) -> str:
        low = text.lower()
        for section in cls.SECTION_PATTERNS:
            if re.search(rf"\b{re.escape(section)}\b", low):
                return section.title()
        return "Abstract"

    @classmethod
    def chunk_paper(
        cls,
        paper_row: pd.Series,
        max_words: int = 180,
        overlap: int = 35,
    ) -> List[EvidenceChunk]:
        title = normalize_text(get_field(paper_row, "title", "Untitled Paper"))
        paper_id = normalize_text(get_field(paper_row, "paper_id", "UNKNOWN"))
        authors = normalize_text(get_field(paper_row, "authors", "N/A"))
        category = normalize_text(get_field(paper_row, "category", "N/A"))
        broad_domain = normalize_text(get_field(paper_row, "broad_domain", "N/A"))

        pub_year = safe_int(get_field(paper_row, "publication_year", 0), 0)
        orig_rank = safe_int(get_field(paper_row, "rank", 0), 0)
        sim_score = safe_float(get_field(paper_row, "similarity_score", 0.0))
        conf_score = safe_float(get_field(paper_row, "confidence_score", 0.0))

        text = get_evidence_text(paper_row)
        text = normalize_text(text)

        if not text:
            text = title

        # Remove accidental repeated title at the beginning.
        if title and text.lower().startswith(title.lower()):
            remainder = text[len(title):].strip(" :-\n")
            if len(remainder) > 40:
                text = remainder

        words = text.split()

        if len(words) <= max_words:
            raw_chunks = [text]
        else:
            raw_chunks = []
            step = max(1, max_words - overlap)
            for start in range(0, len(words), step):
                chunk_words = words[start:start + max_words]
                if len(chunk_words) < 25:
                    break
                raw_chunks.append(" ".join(chunk_words))
                if start + max_words >= len(words):
                    break

        chunks: List[EvidenceChunk] = []

        for index, chunk_text in enumerate(raw_chunks, 1):
            word_count = len(chunk_text.split())
            if word_count < 5:
                continue

            chunks.append(
                EvidenceChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_C{index:02d}",
                    title=title,
                    authors=authors,
                    category=category,
                    broad_domain=broad_domain,
                    publication_year=pub_year,
                    chunk_text=chunk_text,
                    section_name=cls._infer_section(chunk_text),
                    original_rank=orig_rank,
                    similarity_score=sim_score,
                    confidence_score=conf_score,
                    word_count=word_count,
                    estimated_tokens=estimate_token_count(chunk_text),
                )
            )

        return chunks


# =============================================================================
# Evidence quality
# =============================================================================

class ScientificQualityAnalyzer:
    @staticmethod
    def evaluate_and_score(
        chunk: EvidenceChunk,
        query: str,
        task_type: TaskType,
        current_year: int = 2026,
    ) -> float:
        sim = clamp((chunk.similarity_score + 1.0) / 2.0)

        # Confidence is generated by the semantic search engine. We use it as
        # a supporting signal, not as ground truth.
        confidence = clamp(chunk.confidence_score / 100.0)

        rerank = clamp(chunk.rerank_score)

        lex = clamp(chunk.lexical_score)

        # Completeness: reward actual abstract/combined-text evidence and
        # penalize title-like snippets.
        text = chunk.chunk_text
        words = chunk.word_count
        title_like = words <= max(20, len(chunk.title.split()) + 5)

        completeness = 0.35
        if words >= 40:
            completeness += 0.25
        if words >= 100:
            completeness += 0.20
        if not title_like:
            completeness += 0.20
        completeness = clamp(completeness)

        # Recency is task dependent.
        age = max(0, current_year - chunk.publication_year) if chunk.publication_year else 10

        if task_type in {
            TaskType.TREND_ANALYSIS,
            TaskType.RESEARCH_GAP,
        }:
            decay = 0.075
        elif task_type in {
            TaskType.LITERATURE_REVIEW,
            TaskType.SURVEY_WRITING,
        }:
            decay = 0.025
        else:
            decay = 0.035

        recency = math.exp(-decay * age)

        # If cross encoder is unavailable, its score may be 0. HybridReranker
        # still supplies a meaningful score, so no special-case penalty.
        score = (
            CONFIG.SIMILARITY_WEIGHT * sim
            + CONFIG.RERANK_WEIGHT * rerank
            + CONFIG.COMPLETENESS_WEIGHT * completeness
            + CONFIG.RECENCY_WEIGHT * recency
            + CONFIG.DIVERSITY_BONUS_WEIGHT * lex
        )

        chunk.lexical_score = round(lex, 4)
        chunk.completeness_score = round(completeness, 4)
        chunk.evidence_quality_score = round(clamp(score), 4)

        return chunk.evidence_quality_score


# =============================================================================
# Deduplication and diversity
# =============================================================================

class EvidenceDeduplicatorAndDiversifier:
    @staticmethod
    def process(
        chunks: List[EvidenceChunk],
        max_chunks_per_paper: int = CONFIG.MAX_CHUNKS_PER_PAPER,
    ) -> List[EvidenceChunk]:
        if not chunks:
            return []

        ranked = sorted(
            chunks,
            key=lambda x: (
                x.evidence_quality_score,
                x.similarity_score,
                x.rerank_score,
            ),
            reverse=True,
        )

        selected: List[EvidenceChunk] = []
        paper_counts: Dict[str, int] = collections.defaultdict(int)

        for chunk in ranked:
            if paper_counts[chunk.paper_id] >= max_chunks_per_paper:
                continue

            duplicate = False

            for previous in selected:
                # Same paper + highly overlapping text.
                lexical_jaccard = jaccard_similarity(
                    chunk.chunk_text,
                    previous.chunk_text,
                )
                if lexical_jaccard >= CONFIG.JACCARD_DEDUP_THRESHOLD:
                    duplicate = True
                    break

                # Approximate semantic duplicate protection. We deliberately
                # do not claim this is embedding similarity; it is lexical
                # fallback similarity.
                if (
                    chunk.title
                    and previous.title
                    and chunk.title.lower() == previous.title.lower()
                    and lexical_jaccard >= 0.70
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(chunk)
            paper_counts[chunk.paper_id] += 1

        return selected


# =============================================================================
# Scientific reasoning
# =============================================================================

class ScientificReasoningSuite:
    """Evidence-level reasoning.

    These methods intentionally report conservative conclusions. Keyword
    heuristics are not treated as proof of scientific contradiction.
    """

    CONTRADICTION_PATTERNS = (
        r"\bcontradict",
        r"\bcontrary to\b",
        r"\binconsistent with\b",
        r"\bfailed to\b",
        r"\bno significant improvement\b",
        r"\bdoes not improve\b",
        r"\bdid not improve\b",
        r"\bworse than\b",
    )

    SUPPORT_PATTERNS = (
        r"\boutperform",
        r"\bsuperior\b",
        r"\bimprov",
        r"\bbetter performance\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
    )

    @classmethod
    def analyze_conflicts(cls, chunks: List[EvidenceChunk]) -> ConflictReport:
        if len(chunks) < 2:
            return ConflictReport(
                False,
                "Insufficient independent evidence for conflict analysis.",
                [c.paper_id for c in chunks],
                [],
                [],
                0.0,
            )

        opposing: Set[str] = set()
        supporting: Set[str] = set()

        for chunk in chunks:
            text = chunk.chunk_text.lower()

            if any(re.search(p, text) for p in cls.CONTRADICTION_PATTERNS):
                opposing.add(chunk.paper_id)

            if any(re.search(p, text) for p in cls.SUPPORT_PATTERNS):
                supporting.add(chunk.paper_id)

        # A conflict is only flagged when both positive and negative signals
        # appear across independent papers. This is still a "potential"
        # conflict, not a validated contradiction.
        has_conflicts = bool(opposing and supporting)

        if has_conflicts:
            summary = (
                "Potential empirical disagreement detected from language "
                "patterns across retrieved sources. This requires claim-level "
                "verification before being treated as a scientific contradiction."
            )
            confidence = 0.55
        else:
            summary = (
                "No evidence-level contradiction was established by the "
                "conservative conflict detector."
            )
            confidence = 0.20

        return ConflictReport(
            has_conflicts=has_conflicts,
            conflict_summary=summary,
            supporting_evidence_ids=sorted(supporting),
            opposing_evidence_ids=sorted(opposing),
            potential_reasons=[
                "Different datasets or benchmarks",
                "Different experimental settings",
                "Different model/hyperparameter choices",
            ],
            confidence=confidence,
        )

    @staticmethod
    def evaluate_consensus(chunks: List[EvidenceChunk]) -> ConsensusReport:
        unique = list(dict.fromkeys(c.paper_id for c in chunks))

        if not unique:
            return ConsensusReport("NO_CONSENSUS", 0.0, [], [])

        # Consensus here means independent source support, not scientific truth.
        source_count = len(unique)
        if source_count >= 6:
            level = "STRONG"
        elif source_count >= 3:
            level = "MODERATE"
        elif source_count >= 2:
            level = "WEAK"
        else:
            level = "INSUFFICIENT"

        # Cap the score below 100 unless actual claim-level verification exists.
        percentage = min(90.0, round(source_count / 6.0 * 90.0, 1))

        return ConsensusReport(
            consensus_level=level,
            consensus_percentage=percentage,
            supporting_paper_ids=unique,
            opposing_paper_ids=[],
        )

    @staticmethod
    def analyze_coverage(
        query: str,
        chunks: List[EvidenceChunk],
    ) -> CoverageReport:
        if not chunks:
            return CoverageReport(
                0.0,
                {},
                ["Architecture", "Datasets", "Results", "Limitations", "Applications"],
                [],
            )

        text = " ".join(c.chunk_text for c in chunks).lower()

        dimensions = {
            "Architecture": (
                "architecture", "model", "transformer", "network",
                "encoder", "attention", "algorithm",
            ),
            "Datasets": (
                "dataset", "benchmark", "data", "corpus", "training set",
            ),
            "Results": (
                "accuracy", "result", "performance", "f1", "precision",
                "recall", "map", "score", "improved",
            ),
            "Limitations": (
                "limitation", "drawback", "failure", "bottleneck",
                "challenge", "future work",
            ),
            "Applications": (
                "application", "deployment", "use case", "autonomous",
                "industry", "real-world",
            ),
        }

        scores: Dict[str, float] = {}
        missing: List[str] = []

        for dimension, keywords in dimensions.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            score = round(min(100.0, hits / len(keywords) * 100.0), 1)
            scores[dimension] = score
            if score < 20.0:
                missing.append(dimension)

        overall = round(float(np.mean(list(scores.values()))), 1)

        queries = [
            f"{dimension} of {query}"
            for dimension in missing
        ]

        return CoverageReport(
            overall_coverage_pct=overall,
            topic_scores=scores,
            missing_topics=missing,
            iterative_queries_suggested=queries,
        )

    @staticmethod
    def build_relationship_graph(
        chunks: List[EvidenceChunk],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        for chunk in chunks:
            nodes.append(
                {
                    "id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title[:120],
                    "quality": chunk.evidence_quality_score,
                }
            )

        # Connect papers only when there is measurable lexical overlap.
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if chunks[i].paper_id == chunks[j].paper_id:
                    relation = "SAME_PAPER"
                    score = 1.0
                else:
                    score = jaccard_similarity(
                        chunks[i].chunk_text,
                        chunks[j].chunk_text,
                    )
                    if score >= 0.35:
                        relation = "TOPICALLY_RELATED"
                    else:
                        continue

                edges.append(
                    {
                        "source": chunks[i].chunk_id,
                        "target": chunks[j].chunk_id,
                        "relation": relation,
                        "similarity": round(score, 4),
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }


# =============================================================================
# IR evaluation
# =============================================================================

class RetrievalEvaluator:
    """IR metrics with an explicit benchmark/proxy distinction.

    `qrels` format:
        {
            "query text": {
                "paper_id_1": 2,
                "paper_id_2": 1,
                ...
            }
        }

    Relevance grades follow the usual convention:
        0 = irrelevant
        1 = relevant
        2 = highly relevant
    """

    @staticmethod
    def _dcg(relevances: Sequence[float]) -> float:
        return sum(
            rel / math.log2(index + 2)
            for index, rel in enumerate(relevances)
        )

    @classmethod
    def evaluate_with_qrels(
        cls,
        query: str,
        chunks: List[EvidenceChunk],
        qrels: Dict[str, Dict[str, int]],
        k: Optional[int] = None,
    ) -> RetrievalIRMetrics:
        k = k or len(chunks)
        judgments = qrels.get(query, {})

        if not judgments:
            return cls.evaluate_proxy(chunks)

        retrieved = [c.paper_id for c in chunks[:k]]
        grades = [float(judgments.get(pid, 0)) for pid in retrieved]

        relevant_total = sum(1 for grade in judgments.values() if grade > 0)
        relevant_retrieved = sum(1 for grade in grades if grade > 0)

        precision = relevant_retrieved / max(1, len(grades))
        recall = relevant_retrieved / max(1, relevant_total)

        rr = 0.0
        for rank, grade in enumerate(grades, 1):
            if grade > 0:
                rr = 1.0 / rank
                break

        # AP
        running = 0.0
        hits = 0
        for rank, grade in enumerate(grades, 1):
            if grade > 0:
                hits += 1
                running += hits / rank
        ap = running / max(1, relevant_total)

        dcg = cls._dcg(grades)
        ideal = sorted(judgments.values(), reverse=True)[:k]
        idcg = cls._dcg([float(x) for x in ideal])
        ndcg = dcg / idcg if idcg else 0.0

        unique = len(set(retrieved))
        diversity = unique / max(1, len(retrieved))
        token_count = sum(c.estimated_tokens for c in chunks[:k])
        citation_density = unique / max(1, token_count) * 1000.0

        return RetrievalIRMetrics(
            recall_at_k=round(recall, 4),
            precision_at_k=round(precision, 4),
            mrr=round(rr, 4),
            map_score=round(ap, 4),
            ndcg_at_k=round(ndcg, 4),
            coverage_score=round(recall * 100.0, 2),
            diversity_score=round(diversity, 4),
            citation_density=round(citation_density, 2),
            evaluation_mode="BENCHMARK_QRELS",
        )

    @staticmethod
    def evaluate_proxy(
        chunks: List[EvidenceChunk],
    ) -> RetrievalIRMetrics:
        """Proxy metrics when no relevance judgments are supplied.

        These values are intentionally labelled PROXY_ESTIMATE and should not
        be reported as benchmark retrieval accuracy.
        """
        if not chunks:
            return RetrievalIRMetrics(
                0, 0, 0, 0, 0, 0, 0, 0, "PROXY_ESTIMATE"
            )

        k = len(chunks)

        # Ranking quality proxy: normalized quality in the returned set.
        quality = [clamp(c.evidence_quality_score) for c in chunks]
        avg_quality = float(np.mean(quality))

        # Reciprocal rank proxy based on first strong item.
        rr = 0.0
        for rank, q in enumerate(quality, 1):
            if q >= 0.70:
                rr = 1.0 / rank
                break

        # Diversity is real and measurable without labels.
        unique = len(set(c.paper_id for c in chunks))
        diversity = unique / max(1, k)

        # Coverage proxy is deliberately not presented as recall.
        coverage_proxy = avg_quality * 100.0

        # Self-consistency of quality ordering, not true nDCG.
        sorted_quality = sorted(quality, reverse=True)
        dcg = sum(
            q / math.log2(i + 2)
            for i, q in enumerate(quality)
        )
        idcg = sum(
            q / math.log2(i + 2)
            for i, q in enumerate(sorted_quality)
        )
        ndcg_proxy = dcg / idcg if idcg else 0.0

        token_count = sum(c.estimated_tokens for c in chunks)
        citation_density = unique / max(1, token_count) * 1000.0

        return RetrievalIRMetrics(
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=round(rr, 4),
            map_score=0.0,
            ndcg_at_k=round(ndcg_proxy, 4),
            coverage_score=round(coverage_proxy, 2),
            diversity_score=round(diversity, 4),
            citation_density=round(citation_density, 2),
            evaluation_mode="PROXY_ESTIMATE",
        )

    @classmethod
    def evaluate(
        cls,
        query: str,
        chunks: List[EvidenceChunk],
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        k: Optional[int] = None,
    ) -> Tuple[RetrievalIRMetrics, Dict[str, float]]:
        if qrels and query in qrels:
            metrics = cls.evaluate_with_qrels(query, chunks, qrels, k)
        else:
            metrics = cls.evaluate_proxy(chunks)

        if chunks:
            avg_conf = float(
                np.mean([clamp(c.confidence_score / 100.0) for c in chunks])
            ) * 100.0
            avg_quality = float(
                np.mean([clamp(c.evidence_quality_score) for c in chunks])
            ) * 100.0
        else:
            avg_conf = 0.0
            avg_quality = 0.0

        calibration = {
            "Architecture": round(clamp((avg_conf * 0.60 + avg_quality * 0.40) / 100) * 100, 1),
            "Methods": round(clamp((avg_conf * 0.55 + avg_quality * 0.45) / 100) * 100, 1),
            "Datasets": round(clamp((avg_conf * 0.45 + avg_quality * 0.55) / 100) * 100, 1),
            "Results": round(clamp((avg_conf * 0.50 + avg_quality * 0.50) / 100) * 100, 1),
        }

        return metrics, calibration


# =============================================================================
# Core pipeline
# =============================================================================

class RetrieverPipeline:
    """Production-grade evidence retriever.

    Public API compatibility:
        RetrieverPipeline(search_engine=None, reranker=None)
        retrieve(query, top_k_papers, top_k_chunks, token_budget,
                  category, broad_domain, year_range)
    """

    def __init__(
        self,
        search_engine: Optional[Any] = None,
        reranker: Optional[BaseReranker] = None,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        setup_directories()

        self.qrels = qrels

        if search_engine is not None:
            self.search_engine = search_engine
        elif SemanticSearchEngine is not None:
            logger.info("Initializing underlying SemanticSearchEngine instance...")
            self.search_engine = SemanticSearchEngine()
        else:
            self.search_engine = None

        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = CrossEncoderReranker()

        self.last_reranker_backend = "UNKNOWN"

    @staticmethod
    def _candidate_k(top_k_papers: int) -> int:
        return min(
            CONFIG.MAX_CANDIDATE_K,
            max(
                CONFIG.MIN_CANDIDATE_K,
                top_k_papers * CONFIG.CANDIDATE_MULTIPLIER,
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k_papers: int = CONFIG.DEFAULT_TOP_K_PAPERS,
        top_k_chunks: int = CONFIG.DEFAULT_TOP_K_CHUNKS,
        token_budget: int = CONFIG.DEFAULT_TOKEN_BUDGET,
        category: Optional[str] = None,
        broad_domain: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> EvidencePackage:
        started = time.perf_counter()

        query = normalize_text(query)
        if not query:
            raise ValueError("query must be a non-empty string")

        top_k_papers = max(1, int(top_k_papers))
        top_k_chunks = max(1, int(top_k_chunks))
        token_budget = max(128, int(token_budget))

        trace: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Stage 1: Query understanding
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        qu = QueryUnderstandingEngine.process_query(query)
        trace["query_understanding_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # ------------------------------------------------------------------
        # Stage 2: Candidate retrieval
        # ------------------------------------------------------------------
        candidate_k = self._candidate_k(top_k_papers)

        t0 = time.perf_counter()

        if self.search_engine is None:
            search_df = pd.DataFrame()
        else:
            search_df = self.search_engine.search(
                query=qu.expanded_query,
                top_k=candidate_k,
                category=category,
                broad_domain=broad_domain,
                year_range=year_range,
            )

        trace["search_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        trace["candidate_k"] = candidate_k
        trace["candidate_count"] = int(len(search_df))

        if search_df.empty:
            logger.warning("Empty semantic search results returned.")
            return self._build_empty_package(
                query=query,
                qu=qu,
                token_budget=token_budget,
                trace=trace,
            )

        # ------------------------------------------------------------------
        # Stage 3: Neural/hybrid reranking
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        reranked_df = self.reranker.rerank(
            query=query,
            papers_df=search_df,
            top_k=top_k_papers,
        )

        # Capture backend when using our CrossEncoderReranker.
        self.last_reranker_backend = getattr(
            self.reranker,
            "backend",
            self.reranker.__class__.__name__,
        )

        trace["rerank_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        trace["reranker_backend"] = self.last_reranker_backend

        # ------------------------------------------------------------------
        # Stage 4: Chunking
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        raw_chunks: List[EvidenceChunk] = []

        for _, row in reranked_df.iterrows():
            raw_chunks.extend(
                AdaptiveDocumentChunker.chunk_paper(row)
            )

        trace["chunk_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        trace["raw_chunk_count"] = len(raw_chunks)

        # Propagate reranker scores to chunks.
        for chunk in raw_chunks:
            matching = reranked_df[
                reranked_df.apply(
                    lambda r: normalize_text(
                        get_field(r, "paper_id", "")
                    ) == chunk.paper_id,
                    axis=1,
                )
            ]

            if not matching.empty:
                row = matching.iloc[0]
                chunk.rerank_score = safe_float(
                    row.get("_rerank_score", 0.0)
                )
                chunk.lexical_score = safe_float(
                    row.get("_lexical_score", 0.0)
                )

        # ------------------------------------------------------------------
        # Stage 5: Evidence quality
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        for chunk in raw_chunks:
            ScientificQualityAnalyzer.evaluate_and_score(
                chunk=chunk,
                query=query,
                task_type=qu.task_type,
            )

        trace["quality_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # ------------------------------------------------------------------
        # Stage 6: Deduplication + diversity
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        diverse_chunks = EvidenceDeduplicatorAndDiversifier.process(
            raw_chunks
        )

        trace["dedup_diversity_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # ------------------------------------------------------------------
        # Stage 7: Token-budget aware evidence selection
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        retained: List[EvidenceChunk] = []
        consumed = 0

        for chunk in diverse_chunks:
            if not retained:
                if chunk.estimated_tokens > token_budget:
                    # Hard truncate the first evidence rather than returning
                    # an empty context.
                    words = chunk.chunk_text.split()
                    allowed_words = max(20, int(token_budget * 0.75))
                    chunk.chunk_text = " ".join(words[:allowed_words])
                    chunk.word_count = len(chunk.chunk_text.split())
                    chunk.estimated_tokens = estimate_token_count(
                        chunk.chunk_text
                    )

            if consumed + chunk.estimated_tokens <= token_budget:
                retained.append(chunk)
                consumed += chunk.estimated_tokens

            if len(retained) >= top_k_chunks:
                break

        trace["budget_fit_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        trace["consumed_tokens"] = consumed
        trace["token_budget"] = token_budget

        # ------------------------------------------------------------------
        # Stage 8: Scientific reasoning
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        conflict = ScientificReasoningSuite.analyze_conflicts(retained)
        consensus = ScientificReasoningSuite.evaluate_consensus(retained)
        coverage = ScientificReasoningSuite.analyze_coverage(
            query,
            retained,
        )
        graph = ScientificReasoningSuite.build_relationship_graph(
            retained
        )

        trace["reasoning_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # ------------------------------------------------------------------
        # Stage 9: Selection explanations + formatting
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        for rank, chunk in enumerate(retained, 1):
            evidence_source = (
                "cross-encoder reranking"
                if self.last_reranker_backend == "CROSS_ENCODER"
                else "hybrid semantic/lexical reranking"
            )

            chunk.selection_reason = (
                f"Selected as evidence rank #{rank}; "
                f"quality={chunk.evidence_quality_score:.4f}, "
                f"semantic={chunk.similarity_score:.4f}, "
                f"rerank={chunk.rerank_score:.4f}; "
                f"source={evidence_source}."
            )

        formatted_context = self.assemble_formatted_context(retained)

        trace["formatting_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # ------------------------------------------------------------------
        # Stage 10: IR evaluation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        metrics, calibration = RetrievalEvaluator.evaluate(
            query=query,
            chunks=retained,
            qrels=self.qrels,
            k=top_k_chunks,
        )

        trace["evaluation_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        trace["evaluation_mode"] = metrics.evaluation_mode

        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        total_ms = round(
            (time.perf_counter() - started) * 1000, 2
        )
        trace["total_ms"] = total_ms

        repro = ReproducibilitySnapshot(
            retriever_version=CONFIG.PIPELINE_VERSION,
            pipeline_version=CONFIG.PIPELINE_VERSION,
            timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_snapshot={
                "top_k_papers": top_k_papers,
                "candidate_k": candidate_k,
                "top_k_chunks": top_k_chunks,
                "token_budget": token_budget,
                "category": category,
                "broad_domain": broad_domain,
                "year_range": year_range,
                "reranker_backend": self.last_reranker_backend,
                "reranker_model": getattr(
                    self.reranker,
                    "model_name",
                    None,
                ),
            },
            evidence_ids=[c.chunk_id for c in retained],
        )

        package = EvidencePackage(
            query=query,
            query_understanding=qu,
            chunks=retained,
            formatted_context=formatted_context,
            conflict_report=conflict,
            consensus_report=consensus,
            coverage_report=coverage,
            relationship_graph=graph,
            confidence_calibration=calibration,
            ir_metrics=metrics,
            retrieval_trace=trace,
            reproducibility=repro,
        )

        logger.info(
            "Retriever Pipeline completed in %.2f ms. "
            "Candidates=%d, Reranked=%d, Evidence=%d, Tokens=%d/%d, "
            "Reranker=%s, Evaluation=%s",
            total_ms,
            len(search_df),
            len(reranked_df),
            len(retained),
            consumed,
            token_budget,
            self.last_reranker_backend,
            metrics.evaluation_mode,
        )

        return package

    @staticmethod
    def assemble_formatted_context(
        chunks: List[EvidenceChunk],
    ) -> str:
        if not chunks:
            return "NO RELEVANT SCIENTIFIC EVIDENCE FOUND."

        blocks = [
            "# SCIENTIFIC EVIDENCE CONTEXT PACKAGE",
            "Use only the evidence below for grounded claims.",
            "Do not infer experimental results that are not present in the excerpts.",
            "",
        ]

        for index, chunk in enumerate(chunks, 1):
            blocks.append(
                "\n".join(
                    [
                        f"--- EVIDENCE BLOCK [{index}] ---",
                        f"Evidence ID: {chunk.chunk_id}",
                        f"Paper ID: {chunk.paper_id}",
                        f"Title: {chunk.title}",
                        f"Authors: {chunk.authors}",
                        f"Category: {chunk.category}",
                        f"Domain: {chunk.broad_domain}",
                        f"Publication Year: {chunk.publication_year}",
                        f"Section: {chunk.section_name}",
                        f"Similarity Score: {chunk.similarity_score:.4f}",
                        f"Rerank Score: {chunk.rerank_score:.4f}",
                        f"Evidence Quality Score: {chunk.evidence_quality_score:.4f}",
                        f"Excerpt:\n\"{chunk.chunk_text}\"",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _build_empty_package(
        query: str,
        qu: QueryUnderstandingResult,
        token_budget: int,
        trace: Optional[Dict[str, Any]] = None,
    ) -> EvidencePackage:
        metrics = RetrievalIRMetrics(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "PROXY_ESTIMATE",
        )

        repro = ReproducibilitySnapshot(
            retriever_version=CONFIG.PIPELINE_VERSION,
            pipeline_version=CONFIG.PIPELINE_VERSION,
            timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_snapshot={"token_budget": token_budget},
            evidence_ids=[],
        )

        trace = trace or {}
        trace["total_ms"] = trace.get("total_ms", 0.0)

        return EvidencePackage(
            query=query,
            query_understanding=qu,
            chunks=[],
            formatted_context="NO RELEVANT SCIENTIFIC EVIDENCE FOUND.",
            conflict_report=ConflictReport(
                False, "No evidence", [], [], [], 0.0
            ),
            consensus_report=ConsensusReport(
                "NO_CONSENSUS", 0.0, [], []
            ),
            coverage_report=CoverageReport(
                0.0, {}, [], []
            ),
            relationship_graph={
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            },
            confidence_calibration={},
            ir_metrics=metrics,
            retrieval_trace=trace,
            reproducibility=repro,
        )


# =============================================================================
# Export
# =============================================================================

def export_evidence_package(
    pkg: EvidencePackage,
    file_prefix: str = "evidence_package",
) -> Tuple[str, str, str]:
    setup_directories()

    json_path = pathlib.Path(CONFIG.REPORTS_DIR) / f"{file_prefix}.json"
    csv_path = pathlib.Path(CONFIG.REPORTS_DIR) / f"{file_prefix}_chunks.csv"
    md_path = pathlib.Path(CONFIG.REPORTS_DIR) / f"{file_prefix}.md"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(pkg), file, indent=2, ensure_ascii=False)

    if pkg.chunks:
        pd.DataFrame(
            [asdict(chunk) for chunk in pkg.chunks]
        ).to_csv(csv_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)

    md = [
        "# SCIENTIFIC EVIDENCE RETRIEVAL REPORT",
        "",
        f"**Query:** {pkg.query}",
        "",
        f"**Task:** `{pkg.query_understanding.task_type.value}`",
        "",
        f"**Retrieval Profile:** `{pkg.query_understanding.retrieval_profile.value}`",
        "",
        f"**Reranker:** `{pkg.retrieval_trace.get('reranker_backend', 'N/A')}`",
        "",
        f"**Evaluation Mode:** `{pkg.ir_metrics.evaluation_mode}`",
        "",
        "## Retrieval Metrics",
        "",
        f"- Precision@K: {pkg.ir_metrics.precision_at_k}",
        f"- Recall@K: {pkg.ir_metrics.recall_at_k}",
        f"- MRR: {pkg.ir_metrics.mrr}",
        f"- MAP: {pkg.ir_metrics.map_score}",
        f"- nDCG@K: {pkg.ir_metrics.ndcg_at_k}",
        f"- Diversity: {pkg.ir_metrics.diversity_score}",
        f"- Coverage: {pkg.ir_metrics.coverage_score}",
        "",
        "## Coverage",
        "",
        f"- Overall: {pkg.coverage_report.overall_coverage_pct}%",
        f"- Missing topics: {', '.join(pkg.coverage_report.missing_topics) or 'None'}",
        "",
        "## Evidence Context",
        "",
        "```text",
        pkg.formatted_context,
        "```",
    ]

    with open(md_path, "w", encoding="utf-8") as file:
        file.write("\n".join(md))

    logger.info(
        "Exported evidence artifacts to '%s', '%s', '%s'.",
        json_path,
        csv_path,
        md_path,
    )

    return str(json_path), str(csv_path), str(md_path)


# =============================================================================
# Validation
# =============================================================================

def validate_retriever_pipeline(
    pipeline: RetrieverPipeline,
) -> pd.DataFrame:
    """Integration validation, not a claim of scientific benchmark accuracy."""
    logger.info("Running Automated Retriever Quality Validation Suite...")

    query = "Attention mechanisms and vision transformers for autonomous perception"
    package = pipeline.retrieve(
        query=query,
        top_k_papers=10,
        top_k_chunks=8,
        token_budget=3000,
        broad_domain="Computer Vision",
    )

    records = []

    def add(name: str, expected: str, actual: str, passed: bool) -> None:
        records.append(
            {
                "validation_test": name,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if passed else "FAIL",
            }
        )

    add(
        "non_empty_evidence_package",
        "At least 1 chunk",
        f"{len(package.chunks)} chunks",
        len(package.chunks) > 0,
    )

    ids = [chunk.chunk_id for chunk in package.chunks]
    add(
        "unique_chunk_ids",
        "No duplicate IDs",
        f"{len(ids) - len(set(ids))} duplicates",
        len(ids) == len(set(ids)),
    )

    paper_ids = [chunk.paper_id for chunk in package.chunks]
    add(
        "paper_diversity",
        "At least 2 unique papers when available",
        f"{len(set(paper_ids))} unique papers",
        len(set(paper_ids)) >= min(2, len(package.chunks)),
    )

    add(
        "token_budget",
        "<= configured budget",
        f"{sum(c.estimated_tokens for c in package.chunks)} tokens",
        sum(c.estimated_tokens for c in package.chunks) <= 3000,
    )

    add(
        "quality_scores_valid",
        "All scores in [0,1]",
        "Valid" if all(
            0.0 <= c.evidence_quality_score <= 1.0
            for c in package.chunks
        ) else "Invalid",
        all(
            0.0 <= c.evidence_quality_score <= 1.0
            for c in package.chunks
        ),
    )

    add(
        "formatted_context",
        "Non-empty grounded context",
        f"{len(package.formatted_context)} characters",
        bool(package.formatted_context.strip()),
    )

    add(
        "evidence_attribution",
        "Every chunk has paper_id + title",
        "Valid" if all(
            c.paper_id and c.title
            for c in package.chunks
        ) else "Invalid",
        all(c.paper_id and c.title for c in package.chunks),
    )

    add(
        "evaluation_transparency",
        "Evaluation mode explicitly recorded",
        package.ir_metrics.evaluation_mode,
        package.ir_metrics.evaluation_mode
        in {"PROXY_ESTIMATE", "BENCHMARK_QRELS"},
    )

    df = pd.DataFrame(records)
    df.to_csv(
        pathlib.Path(CONFIG.REPORTS_DIR) / "validation_report.csv",
        index=False,
    )

    return df


# =============================================================================
# Optional lightweight analytics
# =============================================================================

def generate_visual_analytics(package: EvidencePackage) -> None:
    """Generate plots only when matplotlib is available.

    Plotting is intentionally optional so API startup does not depend on it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("Matplotlib unavailable; skipping visual analytics.")
        return

    setup_directories()

    metrics = package.ir_metrics

    names = [
        "Precision",
        "Recall",
        "MRR",
        "MAP",
        "nDCG",
        "Diversity",
    ]

    values = [
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.mrr,
        metrics.map_score,
        metrics.ndcg_at_k,
        metrics.diversity_score,
    ]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values)
    plt.ylim(0, 1.1)
    plt.title("Retriever Evaluation Metrics")
    plt.ylabel("Score")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        pathlib.Path(CONFIG.PLOTS_DIR) / "retriever_metrics.png",
        dpi=250,
    )
    plt.close()

    if package.coverage_report.topic_scores:
        plt.figure(figsize=(9, 5))
        topics = list(package.coverage_report.topic_scores.keys())
        scores = list(package.coverage_report.topic_scores.values())
        plt.barh(topics, scores)
        plt.xlim(0, 100)
        plt.title("Scientific Evidence Coverage")
        plt.xlabel("Coverage (%)")
        plt.tight_layout()
        plt.savefig(
            pathlib.Path(CONFIG.PLOTS_DIR) / "topic_coverage.png",
            dpi=250,
        )
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_directories()

    logger.info(
        "Initializing Production Evidence Retriever Pipeline v%s...",
        CONFIG.PIPELINE_VERSION,
    )

    pipeline = RetrieverPipeline()

    test_query = (
        "Attention mechanisms and vision transformers for autonomous perception"
    )

    logger.info(
        "\nExecuting Retriever Test Query: '%s'",
        test_query,
    )

    package = pipeline.retrieve(
        query=test_query,
        top_k_papers=10,
        top_k_chunks=8,
        token_budget=3000,
        broad_domain="Computer Vision",
    )

    export_evidence_package(
        package,
        file_prefix="vision_transformers_retrieval",
    )

    validation = validate_retriever_pipeline(pipeline)
    generate_visual_analytics(package)

    all_passed = (
        not validation.empty
        and (validation["status"] == "PASS").all()
    )

    logger.info(
        "\n========== EVIDENCE RETRIEVER PIPELINE READY =========="
    )
    logger.info(
        "Pipeline Version      : %s",
        CONFIG.PIPELINE_VERSION,
    )
    logger.info(
        "Task Classification   : %s",
        package.query_understanding.task_type.value,
    )
    logger.info(
        "Retrieval Profile     : %s",
        package.query_understanding.retrieval_profile.value,
    )
    logger.info(
        "Candidate Pool        : %s",
        package.retrieval_trace.get("candidate_count", 0),
    )
    logger.info(
        "Chunks Yielded        : %s",
        len(package.chunks),
    )
    logger.info(
        "Reranker Backend      : %s",
        package.retrieval_trace.get("reranker_backend"),
    )
    logger.info(
        "Overall Coverage      : %.1f%%",
        package.coverage_report.overall_coverage_pct,
    )
    logger.info(
        "nDCG@K / Proxy        : %.4f",
        package.ir_metrics.ndcg_at_k,
    )
    logger.info(
        "Evaluation Mode       : %s",
        package.ir_metrics.evaluation_mode,
    )
    logger.info(
        "Consensus Level       : %s",
        package.consensus_report.consensus_level,
    )
    logger.info(
        "Validation Suite      : %s",
        "ALL CHECKS PASSED" if all_passed else "VALIDATION FAILED",
    )
    logger.info(
        "Artifacts Saved To    : %s",
        CONFIG.REPORTS_DIR,
    )
    logger.info(
        "========================================================\n"
    )


if __name__ == "__main__":
    main()