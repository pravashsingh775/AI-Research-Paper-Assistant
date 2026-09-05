from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from apps.api.app.core.evidence_state import supports_full_text_qa
from apps.api.app.services.embeddings import embed, get_embedding_provider

STOP_WORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def content_words(text: str) -> set[str]:
    """Extract substantive words excluding common stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOP_WORDS}


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None
    section: str
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class RAGConfig:
    dense_weight: float = 0.55
    lexical_weight: float = 0.45
    section_priority_boost: float = 0.65
    background_penalty: float = 0.65
    dataset_signal_boost: float = 0.08
    default_retrieval_limit: int = 6


DEFAULT_RAG_CONFIG = RAGConfig()

QUESTION_SIGNALS: dict[str, set[str]] = {
    "dataset": {
        "dataset",
        "datasets",
        "data",
        "corpus",
        "benchmark",
        "participants",
        "samples",
        "records",
        "subjects",
    },
    "methodology": {
        "method",
        "methodology",
        "approach",
        "procedure",
        "pipeline",
        "technique",
    },
    "preprocessing": {
        "preprocess",
        "preprocessing",
        "cleaning",
        "tokenization",
        "normalization",
        "filtering",
    },
    "model": {
        "model",
        "architecture",
        "network",
        "algorithm",
        "encoder",
        "transformer",
        "llm",
        "backbone",
    },
    "training": {
        "training",
        "trained",
        "optimizer",
        "epoch",
        "learning",
        "hyperparameter",
        "loss",
    },
    "evaluation": {
        "evaluation",
        "evaluated",
        "metric",
        "metrics",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "bleu",
        "rouge",
        "rmse",
        "mae",
    },
    "results": {
        "result",
        "results",
        "findings",
        "performance",
        "improvement",
        "outperform",
        "achieve",
    },
    "strengths": {
        "strength",
        "strengths",
        "advantage",
        "advantages",
        "contribution",
        "contributions",
        "benefit",
    },
    "limitations": {
        "limitation",
        "limitations",
        "weakness",
        "weaknesses",
        "drawback",
        "bottleneck",
    },
    "future_work": {"future", "future work", "next", "directions"},
    "metadata": {"author", "authors", "year", "venue", "journal", "conference"},
}

QUESTION_TYPE_PRIORITY: list[str] = [
    "future_work",
    "preprocessing",
    "training",
    "dataset",
    "strengths",
    "limitations",
    "model",
    "evaluation",
    "results",
    "methodology",
    "metadata",
]

PRIMARY_SECTION_MAP: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset", "data", "benchmark", "corpus", "materials"),
    "methodology": ("methodology", "method", "approach", "pipeline", "study design"),
    "preprocessing": ("preprocessing", "preprocess", "tokenization", "cleaning"),
    "model": ("architecture", "model", "backbone", "network"),
    "training": ("training", "hyperparameter", "optimization"),
    "evaluation": ("evaluation", "metric", "metrics", "experiment"),
    "results": ("result", "results", "findings", "performance"),
    "strengths": ("strength", "strengths", "contribution", "contributions"),
    "limitations": ("limitation", "limitations", "weakness", "weaknesses"),
    "future_work": ("future", "future work"),
}

SECONDARY_SECTION_MAP: dict[str, tuple[str, ...]] = {
    "dataset": ("method", "experimental", "evaluation"),
    "methodology": ("architecture", "implementation", "procedure"),
    "model": ("method", "implementation"),
    "training": ("method", "implementation"),
    "evaluation": ("result", "performance", "discussion"),
    "results": ("experiment", "evaluation", "discussion"),
    "strengths": ("discussion", "conclusion"),
    "limitations": ("discussion", "conclusion"),
    "future_work": ("conclusion", "discussion"),
}

PRIMARY_STUDY_SECTIONS: tuple[str, ...] = (
    "method",
    "methodology",
    "study design",
    "design",
    "data",
    "dataset",
    "experiments",
    "results",
    "evaluation",
    "implementation",
)

BACKGROUND_SECTIONS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "background",
    "related",
    "related work",
    "reference",
    "references",
    "prior work",
)


def classify_question(question: str) -> str:
    lowered = question.lower()
    terms = content_words(lowered)
    if "future work" in lowered:
        return "future_work"
    scores = {kind: len(terms & words) for kind, words in QUESTION_SIGNALS.items()}
    max_score = max(scores.values(), default=0)
    if max_score == 0:
        return "general"
    # Tie break by specific domain priority
    for p in QUESTION_TYPE_PRIORITY:
        if scores.get(p, 0) == max_score:
            return p
    return max(scores, key=scores.get)


def evidence_supports(
    question: str, chunks: list[Chunk], question_type: str | None = None
) -> bool:
    if not chunks:
        return False
    kind = question_type or classify_question(question)
    q_words = content_words(question)
    if not q_words:
        return False

    all_chunk_words = set().union(*[content_words(c.text) for c in chunks])

    # For general queries, require substantive overlap of at least 2 content words (or all if short)
    if kind in {"general", "metadata"}:
        matched = q_words & all_chunk_words
        threshold = min(2, len(q_words))
        return len(matched) >= threshold

    # For targeted academic questions:
    signals = QUESTION_SIGNALS.get(kind, set())
    has_signal = any(signals & content_words(chunk.text) for chunk in chunks)
    if not has_signal:
        return False

    # Check if question specifies concrete topic constraints (terms outside general domain signals)
    generic_question_terms = signals | {
        "used",
        "use",
        "study",
        "paper",
        "main",
        "primary",
        "work",
        "author",
        "authors",
        "what",
        "which",
        "how",
    }
    specific_topic_terms = q_words - generic_question_terms
    if specific_topic_terms:
        if not (specific_topic_terms & all_chunk_words):
            return False

    return True


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def chunk_text(text: str, page: int | None = None, max_words: int = 220) -> list[Chunk]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    heading = (
        r"abstract|introduction|background|related\s+work|prior\s+work|study\s+design|design|"
        r"datasets?|data\s+(?:source|collection|preprocessing)?|preprocessing|tokenization|"
        r"materials?\s+and\s+methods|method(?:ology|s)?|training|model(?:\s+architecture)?|"
        r"experiments?|results?|evaluation|strengths?|discussion|conclusion|limitations?|future(?:\s+work)?|references"
    )
    sections = re.split(
        rf"\n\s*(?=(?:{heading})\s*(?:\n|$))", normalized, flags=re.I | re.M
    )
    chunks: list[Chunk] = []
    index = 0
    for raw_section in sections:
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n|\n", raw_section)
            if part.strip()
        ]
        section = (
            (
                paragraphs[0][:100]
                if paragraphs and len(paragraphs[0].split()) <= 12
                else "Body"
            )
            .strip()
            .title()
        )
        if re.fullmatch(r"P\s*A\s*G\s*E\s*\|?\s*\d+", section, flags=re.I):
            section = "Body"
        body = " ".join(paragraphs[1:] if section != "Body" else paragraphs)
        words = body.split()
        for start in range(0, len(words), max_words):
            snippet = " ".join(words[start : start + max_words]).strip()
            if snippet:
                chunks.append(Chunk(snippet, page, section, index, embed(snippet)))
                index += 1
    return chunks


def is_primary_study_section(section: str) -> bool:
    sec_lower = section.lower()
    return any(p in sec_lower for p in PRIMARY_STUDY_SECTIONS) and not any(
        b in sec_lower for b in BACKGROUND_SECTIONS
    )


def hybrid_retrieve(
    chunks: list[Chunk],
    query: str,
    limit: int | None = None,
    config: RAGConfig | None = None,
) -> list[Chunk]:
    cfg = config or DEFAULT_RAG_CONFIG
    max_results = limit if limit is not None else cfg.default_retrieval_limit

    query_terms = content_words(query)
    query_vector = embed(query)
    question_type = classify_question(query)
    scored: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        terms = content_words(chunk.text)
        lexical = len(query_terms & terms) / max(len(query_terms), 1)
        dense = cosine(query_vector, chunk.embedding)
        section = chunk.section.lower()
        section_boost = 0.0

        if any(
            label in section for label in PRIMARY_SECTION_MAP.get(question_type, ())
        ):
            section_boost += cfg.section_priority_boost
        elif any(
            label in section for label in SECONDARY_SECTION_MAP.get(question_type, ())
        ):
            section_boost += cfg.section_priority_boost * 0.4

        # Suppress background sections for empirical questions
        if question_type not in {"general", "metadata"} and any(
            label in section for label in BACKGROUND_SECTIONS
        ):
            section_boost -= cfg.background_penalty

        text_boost = 0.0
        if question_type == "dataset":
            evidence_signals = {
                "electronic",
                "medical",
                "records",
                "randomly",
                "selected",
                "patients",
                "participants",
                "sample",
                "database",
                "dataset",
                "data",
                "corpus",
                "benchmark",
            }
            text_boost = cfg.dataset_signal_boost * len(evidence_signals & terms)

        total_score = (
            cfg.dense_weight * dense
            + cfg.lexical_weight * lexical
            + section_boost
            + text_boost
        )
        scored.append((total_score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:max_results]]
