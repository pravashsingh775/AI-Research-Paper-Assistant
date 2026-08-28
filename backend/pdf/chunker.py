"""
Production-quality semantic chunking for structured scientific papers.

Pipeline boundary
-----------------
StructuredResearchPaper
        |
        v
    SectionAwareChunker
        |
        v
ChunkedResearchPaper
        |
        v
Embeddings / Vector Store / RAG

This module intentionally does NOT:
- reopen or parse PDFs
- detect sections
- OCR documents
- generate embeddings
- call an LLM
- summarize or paraphrase source text
- retrieve or rerank chunks
- create FAISS indexes

Core strategy
-------------
section -> subsection boundary -> paragraph -> sentence -> token fallback

The implementation is deterministic, source-faithful, configurable, and
designed to be independent from the downstream embedding model.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# ============================================================================
# Public constants
# ============================================================================

VALID_CHUNKING_STATUS = frozenset(
    {"success", "partial", "failed"}
)

VALID_CHUNKING_STRATEGIES = frozenset(
    {"hierarchical", "section_semantic"}
)

VALID_CHUNKING_METHODS = frozenset(
    {
        "paragraph",
        "paragraph_group",
        "sentence",
        "sentence_group",
        "token_fallback",
    }
)

DEFAULT_SECTION_TYPES = frozenset(
    {
        "title",
        "abstract",
        "keywords",
        "introduction",
        "background",
        "related_work",
        "literature_review",
        "preliminaries",
        "problem_statement",
        "research_questions",
        "methodology",
        "methods",
        "dataset",
        "model",
        "architecture",
        "implementation",
        "experimental_setup",
        "experiments",
        "results",
        "evaluation",
        "discussion",
        "results_discussion",
        "limitations",
        "future_work",
        "conclusion",
        "acknowledgements",
        "references",
        "appendix",
        "other",
        "unknown",
    }
)


# ============================================================================
# Exceptions
# ============================================================================


class SemanticChunkingError(RuntimeError):
    """Base exception for semantic chunking."""


class InvalidChunkingInputError(
    SemanticChunkingError,
    ValueError,
):
    """Raised when StructuredResearchPaper violates required invariants."""


class ChunkingOutputValidationError(
    SemanticChunkingError,
):
    """Raised when generated chunks violate output invariants."""


# ============================================================================
# Token counting
# ============================================================================


class TokenCounter(Protocol):
    """Minimal tokenizer/counting contract."""

    def count(self, text: str) -> int:
        """Return a deterministic token count for text."""

    def tokens(self, text: str) -> Sequence[str]:
        """Return tokens suitable for deterministic fallback splitting."""

    def decode(self, tokens: Sequence[str]) -> str:
        """Reconstruct text from fallback tokens."""


class RegexTokenCounter:
    """
    Lightweight deterministic fallback tokenizer.

    This is intentionally NOT a claim of model-token equivalence. It provides
    a stable approximation when the actual downstream tokenizer is unavailable.

    The tokenizer keeps:
    - words
    - numbers/decimals
    - common citation tokens
    - mathematical symbols
    - punctuation

    as distinct token-like units.
    """

    _TOKEN_RE = re.compile(
        r"""
        \[[^\]\n]{1,80}\]                         # [12], [1, 3], [Smith2024]
        |https?://[^\s]+                          # URLs
        |[A-Za-z]+\.[A-Za-z]+\.?                  # e.g., i.e., U.S.
        |\d+(?:\.\d+)?%                           # percentages
        |\d+(?:\.\d+)?(?:[eE][+-]?\d+)?           # scientific numbers
        |[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+        # model/data names
        |[A-Za-z]+(?:'[A-Za-z]+)?                 # words
        |[^\w\s]                                  # punctuation/symbol
        """,
        flags=re.VERBOSE | re.UNICODE,
    )

    def tokens(self, text: str) -> Sequence[str]:
        return tuple(
            match.group(0)
            for match in self._TOKEN_RE.finditer(text)
        )

    def count(self, text: str) -> int:
        return len(self.tokens(text))

    def decode(self, tokens: Sequence[str]) -> str:
        """
        Reconstruct readable text from fallback tokens.

        This is used only when an individual sentence exceeds the configured
        maximum. It deliberately produces source-derived fragments rather than
        generating/paraphrasing text.
        """
        if not tokens:
            return ""

        output: list[str] = []

        no_space_before = {
            ".",
            ",",
            ";",
            ":",
            "!",
            "?",
            "%",
            ")",
            "]",
            "}",
            "'",
            "’",
            "”",
        }

        no_space_after = {
            "(",
            "[",
            "{",
            "“",
            "‘",
        }

        for token in tokens:
            if not output:
                output.append(token)
                continue

            previous = output[-1]

            if (
                token in no_space_before
                or previous in no_space_after
            ):
                output.append(token)
            else:
                output.append(" ")
                output.append(token)

        return "".join(output)


class CallableTokenCounter:
    """
    Adapter for an externally supplied tokenizer.

    `counter` may be any callable returning an integer token count.
    Optional `tokenize`/`decode` functions enable token fallback.
    """

    def __init__(
        self,
        counter: Callable[[str], int],
        *,
        tokenize: Optional[
            Callable[[str], Sequence[str]]
        ] = None,
        decode: Optional[
            Callable[[Sequence[str]], str]
        ] = None,
    ) -> None:
        if not callable(counter):
            raise TypeError(
                "counter must be callable."
            )

        self._counter = counter
        self._tokenize = tokenize
        self._decode = decode

    def count(self, text: str) -> int:
        value = int(self._counter(text))

        if value < 0:
            raise ValueError(
                "Token counter returned a negative count."
            )

        return value

    def tokens(self, text: str) -> Sequence[str]:
        if self._tokenize is None:
            return RegexTokenCounter().tokens(text)

        return tuple(
            self._tokenize(text)
        )

    def decode(
        self,
        tokens: Sequence[str],
    ) -> str:
        if self._decode is None:
            return RegexTokenCounter().decode(tokens)

        return self._decode(tokens)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for hierarchical semantic chunking.

    Defaults are engineering baselines, not empirically optimal values.
    Retrieval/Q&A experiments should tune them later.
    """

    min_chunk_tokens: int = 80
    target_chunk_tokens: int = 300
    max_chunk_tokens: int = 500

    overlap_tokens: int = 50
    overlap_min_tokens: int = 12

    include_references: bool = False
    preserve_subsections: bool = True

    strategy: str = "hierarchical"

    # Paragraph grouping.
    max_paragraphs_per_chunk: int = 4

    # Sentence fallback.
    min_sentence_tokens_for_overlap: int = 8

    # Prevent a tiny final fragment from becoming an isolated chunk.
    tiny_fragment_ratio: float = 0.35

    # Reference handling.
    reference_chunk_target_tokens: int = 220
    reference_chunk_max_tokens: int = 350

    # Text quality.
    normalize_whitespace: bool = True
    preserve_blank_lines: bool = False

    # PDF-extraction quality guards. These do not rewrite scientific content;
    # they prevent obviously pathological extraction artifacts from silently
    # becoming a tiny, apparently-valid vector index.
    validate_source_quality: bool = True
    fail_on_suspicious_source: bool = True
    suspicious_min_chars: int = 300
    suspicious_short_line_ratio: float = 0.70
    suspicious_toc_line_ratio: float = 0.35
    suspicious_fragment_line_ratio: float = 0.20
    suspicious_single_chunk_chars: int = 650

    # Page metadata behavior.
    page_mapping_mode: str = "section"

    # Diagnostic / integrity behavior.
    fail_fast: bool = False
    fail_on_section_error: bool = True
    require_full_section_coverage: bool = True
    detect_duplicate_chunks: bool = True

    def __post_init__(self) -> None:
        if self.min_chunk_tokens < 1:
            raise ValueError(
                "min_chunk_tokens must be >= 1."
            )

        if self.target_chunk_tokens < self.min_chunk_tokens:
            raise ValueError(
                "target_chunk_tokens must be >= min_chunk_tokens."
            )

        if self.max_chunk_tokens < self.target_chunk_tokens:
            raise ValueError(
                "max_chunk_tokens must be >= target_chunk_tokens."
            )

        if self.overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens must be >= 0."
            )

        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_chunk_tokens."
            )

        if self.overlap_min_tokens < 0:
            raise ValueError(
                "overlap_min_tokens must be >= 0."
            )

        if self.max_paragraphs_per_chunk < 1:
            raise ValueError(
                "max_paragraphs_per_chunk must be >= 1."
            )

        if self.suspicious_min_chars < 1:
            raise ValueError("suspicious_min_chars must be >= 1.")

        if self.suspicious_single_chunk_chars < 1:
            raise ValueError("suspicious_single_chunk_chars must be >= 1.")

        for name, value in (
            ("suspicious_short_line_ratio", self.suspicious_short_line_ratio),
            ("suspicious_toc_line_ratio", self.suspicious_toc_line_ratio),
            ("suspicious_fragment_line_ratio", self.suspicious_fragment_line_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if not 0.0 < self.tiny_fragment_ratio <= 1.0:
            raise ValueError(
                "tiny_fragment_ratio must be in (0, 1]."
            )

        if self.strategy not in VALID_CHUNKING_STRATEGIES:
            raise ValueError(
                f"Unsupported chunking strategy: {self.strategy!r}"
            )

        if self.page_mapping_mode not in {
            "section",
            "metadata",
        }:
            raise ValueError(
                "page_mapping_mode must be 'section' or 'metadata'."
            )


# ============================================================================
# Public output models
# ============================================================================


@dataclass(frozen=True)
class SemanticChunk:
    """
    Source-faithful retrieval chunk.

    `text` is never generated, summarized, or paraphrased by this module.
    """

    chunk_id: str
    document_id: str
    section_id: str
    section_type: str
    section_heading: Optional[str]
    section_level: int

    text: str

    start_page: Optional[int]
    end_page: Optional[int]
    source_pages: tuple[int, ...]

    parent_section_id: Optional[str]

    chunk_index: int
    token_count: int
    char_count: int
    overlap_with_previous: int

    chunking_method: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError(
                "chunk_id cannot be empty."
            )

        if not self.document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if not self.section_id:
            raise ValueError(
                "section_id cannot be empty."
            )

        if not isinstance(self.section_type, str) or not self.section_type.strip():
            raise ValueError("section_type must be a non-empty string.")

        if not self.text.strip():
            raise ValueError(
                "Chunk text cannot be empty."
            )

        if self.section_level < 1:
            raise ValueError(
                "section_level must be >= 1."
            )

        if self.chunk_index < 0:
            raise ValueError(
                "chunk_index must be >= 0."
            )

        if self.token_count < 1:
            raise ValueError(
                "token_count must be >= 1."
            )

        if self.char_count != len(self.text):
            raise ValueError(
                "char_count must equal len(text)."
            )

        if self.overlap_with_previous < 0:
            raise ValueError(
                "overlap_with_previous cannot be negative."
            )

        if self.chunking_method not in VALID_CHUNKING_METHODS:
            raise ValueError(
                f"Unsupported chunking_method: "
                f"{self.chunking_method!r}"
            )

        if (
            tuple(self.source_pages)
            != tuple(
                sorted(
                    set(self.source_pages)
                )
            )
        ):
            raise ValueError(
                "source_pages must be sorted and unique."
            )

        if self.start_page is None or self.end_page is None:
            if self.source_pages:
                raise ValueError(
                    "source_pages must be empty when page bounds are "
                    "unknown."
                )
        else:
            if self.start_page > self.end_page:
                raise ValueError(
                    "start_page cannot exceed end_page."
                )

            if self.source_pages:
                if (
                    self.source_pages[0]
                    != self.start_page
                    or self.source_pages[-1]
                    != self.end_page
                ):
                    raise ValueError(
                        "source_pages do not match page bounds."
                    )


@dataclass(frozen=True)
class ChunkingDiagnostics:
    """Engineering diagnostics, not model-quality metrics."""

    total_chunks: int
    average_token_count: float
    median_token_count: float
    min_token_count: int
    max_token_count: int
    chunks_per_section: Mapping[str, int]
    average_overlap_tokens: float
    max_overlap_tokens: int
    oversized_chunks: int
    tiny_chunks: int
    duplicate_chunks: int
    token_fallback_chunks: int
    page_mapping_mode: str
    input_char_count: int = 0
    input_section_count: int = 0
    input_page_count: int = 0
    declared_page_count: int = 0
    missing_source_pages: tuple[int, ...] = ()
    source_coverage_ratio: float = 0.0
    suspicious_source: bool = False
    source_quality_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkedResearchPaper:
    """Complete output of semantic chunking."""

    document_id: str
    chunks: tuple[SemanticChunk, ...]
    chunk_count: int
    chunking_status: str
    warnings: tuple[str, ...] = ()
    diagnostics: Optional[ChunkingDiagnostics] = None

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if self.chunk_count != len(self.chunks):
            raise ValueError(
                "chunk_count must equal len(chunks)."
            )

        if self.chunking_status not in VALID_CHUNKING_STATUS:
            raise ValueError(
                f"Invalid chunking_status: "
                f"{self.chunking_status!r}"
            )

        indices = [
            chunk.chunk_index
            for chunk in self.chunks
        ]

        if indices != list(
            range(
                len(self.chunks)
            )
        ):
            raise ValueError(
                "Chunk indices must be contiguous and document ordered."
            )

        ids = [
            chunk.chunk_id
            for chunk in self.chunks
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Chunk IDs must be unique."
            )


# ============================================================================
# Lightweight compatible section protocol
# ============================================================================


class PaperSectionLike(Protocol):
    section_id: str
    document_id: str
    section_type: str
    original_heading: Optional[str]
    normalized_heading: str
    level: int
    text: str
    start_page: Optional[int]
    end_page: Optional[int]
    source_pages: Sequence[int]
    parent_section_id: Optional[str]
    child_section_ids: Sequence[str]
    confidence: Optional[float]


class StructuredResearchPaperLike(Protocol):
    document_id: str
    title: Optional[str]
    sections: Sequence[PaperSectionLike]
    section_count: int
    parsing_status: str
    warnings: Sequence[str]
    page_count: Optional[int]
    total_pages: Optional[int]


# ============================================================================
# Internal units
# ============================================================================


@dataclass(frozen=True)
class _Paragraph:
    text: str
    index: int


@dataclass(frozen=True)
class _Sentence:
    text: str
    index: int


@dataclass(frozen=True)
class _DraftChunk:
    """
    Internal chunk before global chunk indexing/metadata finalization.
    """

    document_id: str
    section: PaperSectionLike
    text: str
    method: str
    token_count: int
    overlap_text: str = ""


# ============================================================================
# Scientific sentence splitter
# ============================================================================


class ScientificSentenceSplitter:
    """
    Conservative, dependency-free sentence splitter.

    It protects common scientific abbreviations, decimal numbers, citation
    forms, initials, URLs, and common Latin abbreviations before detecting
    terminal sentence boundaries.

    This is intentionally not a general NLP parser. It is a deterministic
    boundary splitter for chunk-size control.
    """

    _PROTECTED_PERIOD = "<prd>"

    _ABBREVIATIONS = frozenset(
        {
            "e.g.",
            "i.e.",
            "etc.",
            "et al.",
            "vs.",
            "fig.",
            "figs.",
            "eq.",
            "eqs.",
            "sec.",
            "secs.",
            "ref.",
            "refs.",
            "no.",
            "nos.",
            "dr.",
            "prof.",
            "mr.",
            "mrs.",
            "ms.",
            "inc.",
            "ltd.",
            "dept.",
            "approx.",
            "min.",
            "max.",
            "avg.",
            "std.",
            "st.",
            "u.s.",
            "u.k.",
            "a.m.",
            "p.m.",
        }
    )

    _URL_RE = re.compile(
        r"https?://[^\s]+",
        flags=re.IGNORECASE,
    )

    _DECIMAL_RE = re.compile(
        r"(?<=\d)\.(?=\d)"
    )

    _INITIAL_RE = re.compile(
        r"\b[A-Z]\."
    )

    _CITATION_RE = re.compile(
        r"\[(?:[^\]\n]{1,80})\]"
    )

    _ENUMERATION_RE = re.compile(
        r"(?P<prefix>(?:^|[\s])"
        r"(?:\d+\.|[A-Za-z]\.))"
        r"\s+"
    )

    _TERMINAL_RE = re.compile(
        r"""
        (?P<end>
            [.!?]+
            (?:["'”’)\]}]+)?
        )
        (?=
            \s+
            (?:["'“‘(\[]*[\w$]
            |$
        )
        )
        """,
        flags=re.VERBOSE | re.UNICODE,
    )

    def split(self, text: str) -> list[str]:
        text = text.strip()

        if not text:
            return []

        protected = self._protect(text)

        sentences: list[str] = []
        start = 0

        for match in self._TERMINAL_RE.finditer(
            protected
        ):
            end = match.end()

            candidate = protected[
                start:end
            ].strip()

            if not candidate:
                continue

            if not self._is_safe_boundary(
                protected,
                match.start(),
                match.end(),
            ):
                continue

            sentences.append(
                self._restore(
                    candidate
                )
            )

            start = end

        tail = protected[start:].strip()

        if tail:
            sentences.append(
                self._restore(tail)
            )

        return self._postprocess(
            sentences
        )

    def _protect(self, text: str) -> str:
        value = text

        # Protect URLs first.
        value = self._protect_matches(
            value,
            self._URL_RE,
        )

        # Protect citations containing periods.
        value = self._protect_matches(
            value,
            self._CITATION_RE,
        )

        # Protect decimal points.
        value = self._DECIMAL_RE.sub(
            self._PROTECTED_PERIOD,
            value,
        )

        # Protect initials.
        value = self._INITIAL_RE.sub(
            lambda match: match.group(0).replace(
                ".",
                self._PROTECTED_PERIOD,
            ),
            value,
        )

        # Protect known abbreviations.
        for abbreviation in sorted(
            self._ABBREVIATIONS,
            key=len,
            reverse=True,
        ):
            pattern = re.compile(
                re.escape(abbreviation),
                flags=re.IGNORECASE,
            )

            value = pattern.sub(
                lambda match: match.group(0).replace(
                    ".",
                    self._PROTECTED_PERIOD,
                ),
                value,
            )

        return value

    def _restore(self, text: str) -> str:
        return text.replace(
            self._PROTECTED_PERIOD,
            ".",
        ).strip()

    @staticmethod
    def _protect_matches(
        text: str,
        pattern: re.Pattern[str],
    ) -> str:
        return pattern.sub(
            lambda match: match.group(0).replace(
                ".",
                ScientificSentenceSplitter._PROTECTED_PERIOD,
            ),
            text,
        )

    @staticmethod
    def _is_safe_boundary(
        text: str,
        start: int,
        end: int,
    ) -> bool:
        fragment = text[
            max(0, start - 80):min(
                len(text),
                end + 80,
            )
        ]

        # A terminal period immediately followed by another digit was already
        # protected for decimals, but this protects uncommon scientific
        # notation patterns too.
        if re.search(
            r"\d\.\d",
            fragment,
        ):
            absolute_decimal = re.search(
                r"\d\.\d",
                fragment,
            )

            if absolute_decimal:
                local_start = (
                    max(0, start - 80)
                    + absolute_decimal.start()
                )

                if local_start <= start <= (
                    local_start
                    + len(
                        absolute_decimal.group(0)
                    )
                ):
                    return False

        # Do not split inside a citation.
        if (
            fragment.count("[")
            > fragment.count("]")
        ):
            return False

        return True

    @staticmethod
    def _postprocess(
        sentences: Sequence[str],
    ) -> list[str]:
        result: list[str] = []

        for sentence in sentences:
            value = sentence.strip()

            if not value:
                continue

            # A leading citation fragment should remain attached to the
            # sentence that contains it.
            if result and (
                len(value) <= 3
                and not re.search(
                    r"[.!?]$",
                    value,
                )
            ):
                result[-1] = (
                    f"{result[-1]} {value}"
                ).strip()
            else:
                result.append(value)

        return result


# ============================================================================
# Main chunker
# ============================================================================


class SectionAwareSemanticChunker:
    """
    Deterministic hierarchical semantic chunker.

    The class accepts an already parsed StructuredResearchPaper. It does not
    know how to open or interpret PDFs and does not rediscover sections.
    """

    def __init__(
        self,
        *,
        config: Optional[ChunkingConfig] = None,
        token_counter: Optional[TokenCounter] = None,
        sentence_splitter: Optional[
            ScientificSentenceSplitter
        ] = None,
    ) -> None:
        self.config = (
            config
            or ChunkingConfig()
        )
        self.token_counter = (
            token_counter
            or RegexTokenCounter()
        )
        self.sentence_splitter = (
            sentence_splitter
            or ScientificSentenceSplitter()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        paper: StructuredResearchPaperLike,
    ) -> ChunkedResearchPaper:
        """
        Convert a StructuredResearchPaper into retrieval-ready chunks.

        Failure isolation:
        - invalid document-level input raises InvalidChunkingInputError
        - an individual malformed section becomes a warning and does not
          destroy the rest of the paper unless fail_fast=True
        """
        self._validate_paper(
            paper
        )

        logger.info(
            "Starting semantic chunking: document_id=%s sections=%d",
            paper.document_id,
            len(paper.sections),
        )

        warnings: list[str] = []
        drafts: list[_DraftChunk] = []

        # Every non-empty, indexable section must generate at least one draft.
        # This prevents a section-level failure from silently shrinking a
        # complete research paper into a tiny partial chunk set.
        eligible_section_ids = {
            section.section_id
            for section in paper.sections
            if (
                isinstance(section.text, str)
                and section.text.strip()
                and not (
                    section.section_type == "references"
                    and not self.config.include_references
                )
            )
        }
        generated_section_ids: set[str] = set()

        source_quality_warnings = self._assess_source_quality(paper)
        warnings.extend(source_quality_warnings)

        if (
            source_quality_warnings
            and self.config.validate_source_quality
            and self.config.fail_on_suspicious_source
        ):
            raise InvalidChunkingInputError(
                "Suspicious/low-quality extracted source detected before "
                "chunking. Refusing to create a misleading retrieval index. "
                + " | ".join(source_quality_warnings)
            )

        for section_index, section in enumerate(
            paper.sections
        ):
            try:
                section_drafts = self._chunk_section(
                    document_id=paper.document_id,
                    section=section,
                    section_index=section_index,
                )

                drafts.extend(section_drafts)

                if section_drafts:
                    generated_section_ids.add(section.section_id)
                elif section.section_id in eligible_section_ids:
                    message = (
                        f"Non-empty section {section.section_id!r} "
                        f"({section.section_type}) generated zero chunks."
                    )
                    warnings.append(message)

                    if (
                        self.config.fail_fast
                        or self.config.fail_on_section_error
                    ):
                        raise SemanticChunkingError(message)

            except Exception as exc:
                message = (
                    f"Section {section.section_id!r} "
                    f"({section.section_type}) failed during chunking: "
                    f"{type(exc).__name__}: {exc}"
                )

                logger.exception(
                    "Section chunking failure: document_id=%s section=%s",
                    paper.document_id,
                    section.section_id,
                )

                warnings.append(message)

                # Production mode fails closed: source content must never be
                # silently dropped because one section failed.
                if (
                    self.config.fail_fast
                    or self.config.fail_on_section_error
                ):
                    raise SemanticChunkingError(
                        message
                    ) from exc

        missing_section_ids = eligible_section_ids - generated_section_ids
        if (
            self.config.require_full_section_coverage
            and missing_section_ids
        ):
            ordered_missing = [
                section.section_id
                for section in paper.sections
                if section.section_id in missing_section_ids
            ]
            raise ChunkingOutputValidationError(
                "Chunking lost one or more non-empty source sections: "
                f"{ordered_missing!r}. Refusing to create a partial index."
            )

        chunks: list[SemanticChunk] = []

        duplicate_keys: set[str] = set()
        duplicate_count = 0

        for global_index, draft in enumerate(
            drafts
        ):
            chunk = self._materialize_chunk(
                draft=draft,
                global_index=global_index,
                previous_chunk=(
                    chunks[-1]
                    if chunks
                    else None
                ),
            )

            duplicate_key = self._normalized_text_key(
                chunk.text
            )

            if (
                self.config.detect_duplicate_chunks
                and duplicate_key in duplicate_keys
            ):
                duplicate_count += 1
                warnings.append(
                    "Duplicate normalized chunk text detected at "
                    f"chunk index {global_index}; chunk retained because "
                    "silently deleting source-derived content is unsafe."
                )

            duplicate_keys.add(
                duplicate_key
            )
            chunks.append(
                chunk
            )

        diagnostics = self._build_diagnostics(
            chunks=chunks,
            duplicate_count=duplicate_count,
            paper=paper,
            source_quality_warnings=source_quality_warnings,
        )

        status = self._determine_status(
            paper=paper,
            chunks=chunks,
            warnings=warnings,
            diagnostics=diagnostics,
        )

        warnings = self._dedupe(
            [
                *paper.warnings,
                *warnings,
            ]
        )

        result = ChunkedResearchPaper(
            document_id=paper.document_id,
            chunks=tuple(chunks),
            chunk_count=len(chunks),
            chunking_status=status,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
        )

        self._validate_output(
            paper=paper,
            result=result,
        )

        logger.info(
            "Semantic chunking completed: document_id=%s "
            "chunks=%d status=%s",
            paper.document_id,
            result.chunk_count,
            result.chunking_status,
        )

        return result

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_paper(
        self,
        paper: StructuredResearchPaperLike,
    ) -> None:
        if paper is None:
            raise InvalidChunkingInputError(
                "paper cannot be None."
            )

        document_id = getattr(
            paper,
            "document_id",
            None,
        )

        if (
            not isinstance(document_id, str)
            or not document_id.strip()
        ):
            raise InvalidChunkingInputError(
                "paper.document_id must be a non-empty string."
            )

        sections = getattr(
            paper,
            "sections",
            None,
        )

        if sections is None:
            raise InvalidChunkingInputError(
                "paper.sections is required."
            )

        section_count = getattr(
            paper,
            "section_count",
            None,
        )

        if (
            not isinstance(section_count, int)
            or section_count != len(sections)
        ):
            raise InvalidChunkingInputError(
                "paper.section_count does not match sections length."
            )

        section_ids: set[str] = set()

        previous_start_page: Optional[int] = None

        for index, section in enumerate(
            sections
        ):
            self._validate_section(
                section,
                index=index,
                document_id=document_id,
            )

            if section.section_id in section_ids:
                raise InvalidChunkingInputError(
                    f"Duplicate section_id: "
                    f"{section.section_id!r}"
                )

            section_ids.add(
                section.section_id
            )

            if (
                previous_start_page is not None
                and section.start_page is not None
                and section.start_page
                < previous_start_page
            ):
                raise InvalidChunkingInputError(
                    "Section order is not document ordered."
                )

            if section.start_page is not None:
                previous_start_page = (
                    section.start_page
                )

    @staticmethod
    def _validate_section(
        section: PaperSectionLike,
        *,
        index: int,
        document_id: str,
    ) -> None:
        required = (
            "section_id",
            "document_id",
            "section_type",
            "level",
            "text",
        )

        for field_name in required:
            if not hasattr(
                section,
                field_name,
            ):
                raise InvalidChunkingInputError(
                    f"Section {index} is missing "
                    f"{field_name!r}."
                )

        if section.document_id != document_id:
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} belongs to "
                f"{section.document_id!r}, expected {document_id!r}."
            )

        if not isinstance(
            section.section_id,
            str,
        ) or not section.section_id.strip():
            raise InvalidChunkingInputError(
                f"Section {index} has invalid section_id."
            )

        if (
            not isinstance(section.section_type, str)
            or not section.section_type.strip()
        ):
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} has invalid section_type."
            )

        if section.section_type not in DEFAULT_SECTION_TYPES:
            # Do not reject future canonical types from the upstream parser
            # too aggressively. Unknown types are preserved as supplied.
            logger.warning(
                "Unknown upstream section_type=%r for section=%s; "
                "preserving it.",
                section.section_type,
                section.section_id,
            )

        if not isinstance(
            section.level,
            int,
        ) or section.level < 1:
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} has invalid level."
            )

        if not isinstance(
            section.text,
            str,
        ):
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r}.text must be a string."
            )

        if (
            section.start_page is not None
            and (
                not isinstance(
                    section.start_page,
                    int,
                )
                or section.start_page < 1
            )
        ):
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} has invalid start_page."
            )

        if (
            section.end_page is not None
            and (
                not isinstance(
                    section.end_page,
                    int,
                )
                or section.end_page < 1
            )
        ):
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} has invalid end_page."
            )

        if (
            section.start_page is not None
            and section.end_page is not None
            and section.start_page > section.end_page
        ):
            raise InvalidChunkingInputError(
                f"Section {section.section_id!r} has reversed page bounds."
            )

        pages = tuple(
            section.source_pages
            or ()
        )

        if pages:
            if pages != tuple(
                sorted(
                    set(pages)
                )
            ):
                raise InvalidChunkingInputError(
                    f"Section {section.section_id!r} source_pages "
                    "must be sorted and unique."
                )

            if (
                section.start_page is not None
                and pages[0] != section.start_page
            ):
                raise InvalidChunkingInputError(
                    f"Section {section.section_id!r} source_pages "
                    "do not match start_page."
                )

            if (
                section.end_page is not None
                and pages[-1] != section.end_page
            ):
                raise InvalidChunkingInputError(
                    f"Section {section.section_id!r} source_pages "
                    "do not match end_page."
                )

    # ------------------------------------------------------------------
    # Source quality / extraction diagnostics
    # ------------------------------------------------------------------

    def _assess_source_quality(
        self,
        paper: StructuredResearchPaperLike,
    ) -> list[str]:
        """Detect common PDF-extraction failures before vector indexing.

        The chunker cannot recover missing PDF pages or perform OCR. Its job here
        is to fail closed when the supplied structured text has strong signs of
        being a table of contents, broken line extraction, or an unexpectedly
        tiny source. This prevents a 40-page paper from silently becoming a
        one-vector FAISS index.
        """
        if not self.config.validate_source_quality:
            return []

        sections = [
            section
            for section in paper.sections
            if isinstance(section.text, str) and section.text.strip()
            and not (
                section.section_type == "references"
                and not self.config.include_references
            )
        ]

        if not sections:
            return []

        text = "\n".join(section.text for section in sections).strip()
        if len(text) < self.config.suspicious_min_chars:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        warnings: list[str] = []
        short_lines = sum(1 for line in lines if len(line) <= 60)
        toc_lines = sum(1 for line in lines if self._looks_like_toc_line(line))
        fragment_lines = sum(
            1 for line in lines if self._looks_like_extraction_fragment(line)
        )

        short_ratio = short_lines / len(lines)
        toc_ratio = toc_lines / len(lines)
        fragment_ratio = fragment_lines / len(lines)

        if toc_ratio >= self.config.suspicious_toc_line_ratio:
            warnings.append(
                f"TOC-like extraction pattern detected (ratio={toc_ratio:.2f})."
            )

        if fragment_ratio >= self.config.suspicious_fragment_line_ratio:
            warnings.append(
                f"Broken PDF line/word fragmentation detected "
                f"(ratio={fragment_ratio:.2f})."
            )

        if (
            len(lines) >= 8
            and short_ratio >= self.config.suspicious_short_line_ratio
            and toc_ratio > 0
        ):
            warnings.append(
                f"Highly fragmented short-line extraction detected "
                f"(ratio={short_ratio:.2f})."
            )

        # A suspiciously small corpus with multiple source pages is especially
        # dangerous: it often means only a TOC/first page reached the chunker.
        page_count = len(
            {
                int(page)
                for section in sections
                for page in (section.source_pages or ())
                if isinstance(page, int) and page > 0
            }
        )
        if page_count >= 5 and len(text) < self.config.suspicious_single_chunk_chars:
            warnings.append(
                f"Only {len(text)} extracted characters were supplied for "
                f"{page_count} source pages; likely incomplete extraction."
            )

        # A document may contain plenty of characters while its section parser
        # still exposes only a subset of pages. Always surface the mismatch.
        declared_page_count = getattr(paper, "page_count", None)
        if declared_page_count is None:
            declared_page_count = getattr(paper, "total_pages", None)

        if (
            isinstance(declared_page_count, int)
            and not isinstance(declared_page_count, bool)
            and declared_page_count > 0
            and page_count > 0
            and page_count < declared_page_count
        ):
            warnings.append(
                f"Structured paper exposes source-page coverage "
                f"{page_count}/{declared_page_count}; "
                "section parsing may have omitted document content."
            )

        return self._dedupe(warnings)

    @staticmethod
    def _looks_like_toc_line(line: str) -> bool:
        """Return True for common table-of-contents/listing patterns."""
        value = line.strip()
        if not value:
            return False
        if re.search(r"(?:\.{2,}|…{2,})\s*\d{1,4}$", value):
            return True
        if re.search(r"\s+\d{1,4}$", value) and len(value) <= 100:
            lowered = value.lower()
            if any(
                marker in lowered
                for marker in (
                    "abstract", "introduction", "method", "methodology",
                    "results", "discussion", "conclusion", "references",
                    "appendix", "dataset", "experiments",
                )
            ):
                return True
        return False

    @staticmethod
    def _looks_like_extraction_fragment(line: str) -> bool:
        """Detect isolated word fragments commonly caused by PDF line wrapping."""
        value = line.strip()
        if len(value) == 1 and value.isalpha() and value.islower():
            return True
        # Two/three-letter scientific terms (AI, ML, CNN, etc.) are common and
        # must not be classified as extraction failures merely because they are
        # short. The isolated lowercase-letter signature is the reliable case.
        return False

    @staticmethod
    def _repair_extracted_line_breaks(lines: Sequence[str]) -> list[str]:
        """Repair only unambiguous word splits created at PDF line boundaries.

        Examples: a hyphenated line break such as ``compu-`` + ``tation`` is
        reconstructed as ``computation``. A very strong PDF artifact such as a
        line ending in an alphabetic fragment followed by a single lowercase
        character is also joined. This is source reconstruction, not paraphrasing.
        """
        if not lines:
            return []

        result: list[str] = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                i += 1
                continue

            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if (
                    current.endswith("-")
                    and re.fullmatch(r"[A-Za-z]+", nxt)
                ):
                    current = current[:-1] + nxt
                    i += 2
                    result.append(current)
                    continue

                if (
                    re.search(r"[A-Za-z]{4,}$", current)
                    and re.fullmatch(r"[a-z]", nxt)
                ):
                    current = current + nxt
                    i += 2
                    result.append(current)
                    continue

            result.append(current)
            i += 1

        return result

    # ------------------------------------------------------------------
    # Section chunking
    # ------------------------------------------------------------------

    def _chunk_section(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        section_index: int,
    ) -> list[_DraftChunk]:
        text = self._normalize_source_text(
            section.text
        )

        if not text:
            logger.warning(
                "Empty section encountered: %s",
                section.section_id,
            )
            return []

        if (
            section.section_type == "references"
            and not self.config.include_references
        ):
            logger.debug(
                "References excluded from chunking: %s",
                section.section_id,
            )
            return []

        paragraphs = self._split_paragraphs(
            text
        )

        if not paragraphs:
            return []

        if (
            section.section_type == "references"
        ):
            return self._chunk_references(
                document_id=document_id,
                section=section,
                paragraphs=paragraphs,
            )

        return self._chunk_hierarchical(
            document_id=document_id,
            section=section,
            paragraphs=paragraphs,
        )

    def _chunk_hierarchical(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        paragraphs: Sequence[_Paragraph],
    ) -> list[_DraftChunk]:
        drafts: list[_DraftChunk] = []

        current_paragraphs: list[_Paragraph] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self._count(
                paragraph.text
            )

            # A normal paragraph that fits can remain intact.
            if paragraph_tokens <= self.config.max_chunk_tokens:
                if not current_paragraphs:
                    current_paragraphs = [
                        paragraph
                    ]
                    current_tokens = (
                        paragraph_tokens
                    )
                    continue

                combined_text = self._join_paragraphs(
                    [
                        p.text
                        for p in (
                            *current_paragraphs,
                            paragraph,
                        )
                    ]
                )

                combined_tokens = self._count(
                    combined_text
                )

                if (
                    combined_tokens
                    <= self.config.target_chunk_tokens
                    and len(current_paragraphs)
                    < self.config.max_paragraphs_per_chunk
                ):
                    current_paragraphs.append(
                        paragraph
                    )
                    current_tokens = (
                        combined_tokens
                    )
                    continue

                # If the current group is below the target but adding the
                # paragraph would still fit under max, allow it. This gives
                # useful multi-paragraph context without exceeding limits.
                if (
                    combined_tokens
                    <= self.config.max_chunk_tokens
                    and len(current_paragraphs)
                    < self.config.max_paragraphs_per_chunk
                    and current_tokens
                    < self.config.min_chunk_tokens
                ):
                    current_paragraphs.append(
                        paragraph
                    )
                    current_tokens = (
                        combined_tokens
                    )
                    continue

                drafts.append(
                    self._draft_from_paragraphs(
                        document_id=document_id,
                        section=section,
                        paragraphs=current_paragraphs,
                        method=(
                            "paragraph"
                            if len(current_paragraphs) == 1
                            else "paragraph_group"
                        ),
                    )
                )

                current_paragraphs = [
                    paragraph
                ]
                current_tokens = (
                    paragraph_tokens
                )
                continue

            # Long paragraph:
            # flush current group first, then recursively sentence-chunk.
            if current_paragraphs:
                drafts.append(
                    self._draft_from_paragraphs(
                        document_id=document_id,
                        section=section,
                        paragraphs=current_paragraphs,
                        method=(
                            "paragraph"
                            if len(current_paragraphs) == 1
                            else "paragraph_group"
                        ),
                    )
                )
                current_paragraphs = []
                current_tokens = 0

            logger.warning(
                "Long paragraph requires sentence splitting: "
                "section=%s paragraph_index=%d tokens=%d",
                section.section_id,
                paragraph.index,
                paragraph_tokens,
            )

            drafts.extend(
                self._chunk_long_paragraph(
                    document_id=document_id,
                    section=section,
                    paragraph=paragraph,
                )
            )

        if current_paragraphs:
            drafts.append(
                self._draft_from_paragraphs(
                    document_id=document_id,
                    section=section,
                    paragraphs=current_paragraphs,
                    method=(
                        "paragraph"
                        if len(current_paragraphs) == 1
                        else "paragraph_group"
                    ),
                )
            )

        return self._merge_tiny_drafts(
            drafts=drafts,
            document_id=document_id,
            section=section,
        )

    def _chunk_long_paragraph(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        paragraph: _Paragraph,
    ) -> list[_DraftChunk]:
        sentences = self.sentence_splitter.split(
            paragraph.text
        )

        if not sentences:
            return self._token_fallback(
                document_id=document_id,
                section=section,
                text=paragraph.text,
            )

        sentence_units = [
            _Sentence(
                text=text,
                index=index,
            )
            for index, text in enumerate(
                sentences
            )
        ]

        drafts: list[_DraftChunk] = []

        current: list[_Sentence] = []
        current_tokens = 0

        for sentence in sentence_units:
            sentence_tokens = self._count(
                sentence.text
            )

            if sentence_tokens > self.config.max_chunk_tokens:
                if current:
                    drafts.append(
                        self._draft_from_sentences(
                            document_id=document_id,
                            section=section,
                            sentences=current,
                        )
                    )
                    current = []
                    current_tokens = 0

                drafts.extend(
                    self._token_fallback(
                        document_id=document_id,
                        section=section,
                        text=sentence.text,
                    )
                )
                continue

            if not current:
                current = [
                    sentence
                ]
                current_tokens = (
                    sentence_tokens
                )
                continue

            candidate_text = self._join_sentences(
                [
                    s.text
                    for s in (
                        *current,
                        sentence,
                    )
                ]
            )

            candidate_tokens = self._count(
                candidate_text
            )

            if (
                candidate_tokens
                <= self.config.target_chunk_tokens
            ):
                current.append(
                    sentence
                )
                current_tokens = (
                    candidate_tokens
                )
                continue

            if (
                candidate_tokens
                <= self.config.max_chunk_tokens
                and current_tokens
                < self.config.min_chunk_tokens
            ):
                current.append(
                    sentence
                )
                current_tokens = (
                    candidate_tokens
                )
                continue

            drafts.append(
                self._draft_from_sentences(
                    document_id=document_id,
                    section=section,
                    sentences=current,
                )
            )

            # Controlled sentence overlap within the same section only.
            overlap = self._select_sentence_overlap(
                current
            )

            current = [
                *overlap,
                sentence,
            ]

            current_tokens = self._count(
                self._join_sentences(
                    [
                        s.text
                        for s in current
                    ]
                )
            )

            # If overlap itself makes the next chunk too large, discard
            # overlap rather than violating the hard max.
            if (
                current_tokens
                > self.config.max_chunk_tokens
            ):
                current = [
                    sentence
                ]
                current_tokens = (
                    sentence_tokens
                )

        if current:
            drafts.append(
                self._draft_from_sentences(
                    document_id=document_id,
                    section=section,
                    sentences=current,
                )
            )

        return drafts

    # ------------------------------------------------------------------
    # Reference chunking
    # ------------------------------------------------------------------

    def _chunk_references(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        paragraphs: Sequence[_Paragraph],
    ) -> list[_DraftChunk]:
        """
        References are kept separate from normal semantic chunks.

        We prefer each extracted reference paragraph/entry as an atomic unit.
        If an entry is huge, sentence/token fallback applies. Multiple small
        entries may be grouped up to the configured reference target.
        """
        drafts: list[_DraftChunk] = []

        current: list[_Paragraph] = []
        current_tokens = 0

        for paragraph in paragraphs:
            tokens = self._count(
                paragraph.text
            )

            if tokens > self.config.reference_chunk_max_tokens:
                if current:
                    drafts.append(
                        self._draft_from_paragraphs(
                            document_id=document_id,
                            section=section,
                            paragraphs=current,
                            method=(
                                "paragraph"
                                if len(current) == 1
                                else "paragraph_group"
                            ),
                        )
                    )
                    current = []
                    current_tokens = 0

                drafts.extend(
                    self._chunk_long_paragraph(
                        document_id=document_id,
                        section=section,
                        paragraph=paragraph,
                    )
                )
                continue

            if not current:
                current = [
                    paragraph
                ]
                current_tokens = tokens
                continue

            combined = self._join_paragraphs(
                [
                    p.text
                    for p in (
                        *current,
                        paragraph,
                    )
                ]
            )
            combined_tokens = self._count(
                combined
            )

            if (
                combined_tokens
                <= self.config.reference_chunk_target_tokens
            ):
                current.append(
                    paragraph
                )
                current_tokens = (
                    combined_tokens
                )
                continue

            drafts.append(
                self._draft_from_paragraphs(
                    document_id=document_id,
                    section=section,
                    paragraphs=current,
                    method=(
                        "paragraph"
                        if len(current) == 1
                        else "paragraph_group"
                    ),
                )
            )

            current = [
                paragraph
            ]
            current_tokens = tokens

        if current:
            drafts.append(
                self._draft_from_paragraphs(
                    document_id=document_id,
                    section=section,
                    paragraphs=current,
                    method=(
                        "paragraph"
                        if len(current) == 1
                        else "paragraph_group"
                    ),
                )
            )

        return drafts

    # ------------------------------------------------------------------
    # Draft creation
    # ------------------------------------------------------------------

    def _draft_from_paragraphs(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        paragraphs: Sequence[_Paragraph],
        method: str,
    ) -> _DraftChunk:
        text = self._join_paragraphs(
            [
                paragraph.text
                for paragraph in paragraphs
            ]
        )

        return _DraftChunk(
            document_id=document_id,
            section=section,
            text=text,
            method=method,
            token_count=self._count(
                text
            ),
        )

    def _draft_from_sentences(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        sentences: Sequence[_Sentence],
    ) -> _DraftChunk:
        text = self._join_sentences(
            [
                sentence.text
                for sentence in sentences
            ]
        )

        return _DraftChunk(
            document_id=document_id,
            section=section,
            text=text,
            method=(
                "sentence"
                if len(sentences) == 1
                else "sentence_group"
            ),
            token_count=self._count(
                text
            ),
            overlap_text=self._detect_overlap_text(
                sentences
            ),
        )

    def _token_fallback(
        self,
        *,
        document_id: str,
        section: PaperSectionLike,
        text: str,
    ) -> list[_DraftChunk]:
        """
        Last-resort hard split for an indivisible oversized sentence.

        We split only at tokenizer units. No text is generated.
        """
        tokens = tuple(
            self.token_counter.tokens(
                text
            )
        )

        if not tokens:
            return []

        logger.warning(
            "Token fallback required: section=%s tokens=%d",
            section.section_id,
            len(tokens),
        )

        max_tokens = (
            self.config.max_chunk_tokens
        )

        overlap = min(
            self.config.overlap_tokens,
            max_tokens - 1,
        )

        chunks: list[_DraftChunk] = []

        start = 0

        while start < len(tokens):
            end = min(
                start + max_tokens,
                len(tokens),
            )

            chunk_tokens = tokens[
                start:end
            ]

            chunk_text = (
                self.token_counter.decode(
                    chunk_tokens
                ).strip()
            )

            if chunk_text:
                overlap_text = ""

                if (
                    chunks
                    and overlap > 0
                ):
                    overlap_start = max(
                        start - overlap,
                        0,
                    )

                    overlap_tokens = tokens[
                        overlap_start:start
                    ]

                    overlap_text = (
                        self.token_counter.decode(
                            overlap_tokens
                        ).strip()
                    )

                chunks.append(
                    _DraftChunk(
                        document_id=document_id,
                        section=section,
                        text=chunk_text,
                        method="token_fallback",
                        token_count=self._count(
                            chunk_text
                        ),
                        overlap_text=overlap_text,
                    )
                )

            if end >= len(tokens):
                break

            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    # ------------------------------------------------------------------
    # Tiny chunk handling
    # ------------------------------------------------------------------

    def _merge_tiny_drafts(
        self,
        *,
        drafts: Sequence[_DraftChunk],
        document_id: str,
        section: PaperSectionLike,
    ) -> list[_DraftChunk]:
        """
        Merge tiny fragments only inside the same section.

        We never merge across sections. A tiny chunk is retained when no safe
        neighboring merge can be made without exceeding the hard maximum.
        """
        if len(drafts) <= 1:
            return list(drafts)

        result: list[_DraftChunk] = []

        for draft in drafts:
            if not result:
                result.append(
                    draft
                )
                continue

            if not self._is_tiny(
                draft.token_count
            ):
                result.append(
                    draft
                )
                continue

            previous = result[-1]

            if (
                previous.section.section_id
                != section.section_id
            ):
                result.append(
                    draft
                )
                continue

            combined = self._join_paragraphs(
                [
                    previous.text,
                    draft.text,
                ]
            )

            combined_tokens = self._count(
                combined
            )

            if (
                combined_tokens
                <= self.config.max_chunk_tokens
            ):
                result[-1] = _DraftChunk(
                    document_id=document_id,
                    section=section,
                    text=combined,
                    method="paragraph_group",
                    token_count=combined_tokens,
                    overlap_text="",
                )
            else:
                result.append(
                    draft
                )

        return result

    def _is_tiny(
        self,
        token_count: int,
    ) -> bool:
        return (
            token_count
            < self.config.min_chunk_tokens
            * self.config.tiny_fragment_ratio
        )

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def _select_sentence_overlap(
        self,
        sentences: Sequence[_Sentence],
    ) -> list[_Sentence]:
        if self.config.overlap_tokens <= 0:
            return []

        selected: list[_Sentence] = []
        total = 0

        for sentence in reversed(
            sentences
        ):
            tokens = self._count(
                sentence.text
            )

            if (
                selected
                and total + tokens
                > self.config.overlap_tokens
            ):
                break

            if (
                not selected
                and tokens
                > self.config.overlap_tokens
            ):
                # A large sentence cannot be partially overlapped at the
                # sentence level. The next chunk starts cleanly.
                break

            selected.append(
                sentence
            )
            total += tokens

            if total >= (
                self.config.overlap_min_tokens
            ):
                break

        selected.reverse()
        return selected

    @staticmethod
    def _detect_overlap_text(
        sentences: Sequence[_Sentence],
    ) -> str:
        # This metadata is intentionally not inferred as a numeric score.
        # The materialized chunk computes actual token overlap against the
        # previous chunk in document order.
        return ""

    def _calculate_overlap(
        self,
        *,
        previous: Optional[SemanticChunk],
        current_text: str,
        current_section_id: str,
    ) -> int:
        if (
            previous is None
            or self.config.overlap_tokens <= 0
        ):
            return 0

        # Hard section boundary: overlap is never allowed across sections.
        if previous.section_id != current_section_id:
            return 0

        previous_tokens = list(
            self.token_counter.tokens(
                previous.text
            )
        )

        current_tokens = list(
            self.token_counter.tokens(
                current_text
            )
        )

        if not previous_tokens or not current_tokens:
            return 0

        max_overlap = min(
            self.config.overlap_tokens,
            len(previous_tokens),
            len(current_tokens),
        )

        # Longest suffix of previous == prefix of current.
        for size in range(
            max_overlap,
            0,
            -1,
        ):
            if (
                previous_tokens[-size:]
                == current_tokens[:size]
            ):
                return size

        return 0

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_chunk(
        self,
        *,
        draft: _DraftChunk,
        global_index: int,
        previous_chunk: Optional[SemanticChunk],
    ) -> SemanticChunk:
        text = self._normalize_source_text(
            draft.text
        )

        token_count = self._count(
            text
        )

        if token_count <= 0:
            raise ChunkingOutputValidationError(
                "A non-empty draft produced zero tokens."
            )

        if (
            token_count
            > self.config.max_chunk_tokens
        ):
            # The only legitimate exception should be impossible after
            # fallback. Fail loudly rather than silently violating the max.
            raise ChunkingOutputValidationError(
                f"Chunk exceeds max_chunk_tokens: "
                f"{token_count} > "
                f"{self.config.max_chunk_tokens}."
            )

        section = draft.section

        overlap = self._calculate_overlap(
            previous=previous_chunk,
            current_text=text,
            current_section_id=section.section_id,
        )

        chunk_id = self._make_chunk_id(
            document_id=draft.document_id,
            section=section,
            global_index=global_index,
            text=text,
        )

        source_pages = self._source_pages_for_chunk(
            section=section
        )

        start_page, end_page = (
            self._page_bounds(
                source_pages
            )
        )

        metadata = {
            "document_id": draft.document_id,
            "chunk_id": chunk_id,
            "section_id": section.section_id,
            "section_type": section.section_type,
            "section_heading": (
                section.original_heading
            ),
            "section_level": section.level,
            "parent_section_id": (
                section.parent_section_id
            ),
            "source_pages": list(
                source_pages
            ),
            "start_page": start_page,
            "end_page": end_page,
            "chunk_index": global_index,
            "token_count": token_count,
            "char_count": len(text),
            "overlap_with_previous": overlap,
            "chunking_method": draft.method,
            "chunking_strategy": self.config.strategy,
            "page_mapping_mode": (
                self.config.page_mapping_mode
            ),
            "source_fidelity": "source_derived",
        }

        if (
            section.confidence is not None
        ):
            metadata[
                "section_confidence"
            ] = section.confidence

        if (
            section.section_type
            == "references"
        ):
            metadata[
                "retrieval_role"
            ] = "reference"

        return SemanticChunk(
            chunk_id=chunk_id,
            document_id=draft.document_id,
            section_id=section.section_id,
            section_type=section.section_type,
            section_heading=(
                section.original_heading
            ),
            section_level=section.level,
            text=text,
            start_page=start_page,
            end_page=end_page,
            source_pages=tuple(
                source_pages
            ),
            parent_section_id=(
                section.parent_section_id
            ),
            chunk_index=global_index,
            token_count=token_count,
            char_count=len(text),
            overlap_with_previous=overlap,
            chunking_method=draft.method,
            metadata=metadata,
        )

    @staticmethod
    def _make_chunk_id(
        *,
        document_id: str,
        section: PaperSectionLike,
        global_index: int,
        text: str,
    ) -> str:
        """
        Stable ID derived from source identity, structural location and text.

        It intentionally does not use UUID/randomness.
        """
        material = "|".join(
            (
                document_id,
                section.section_id,
                str(global_index),
                section.section_type,
                str(section.level),
                text,
            )
        )

        digest = hashlib.sha256(
            material.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()[:24]

        return (
            f"chk_{global_index:06d}_{digest}"
        )

    # ------------------------------------------------------------------
    # Text / paragraph processing
    # ------------------------------------------------------------------

    def _normalize_source_text(
        self,
        text: str,
    ) -> str:
        value = unicodedata.normalize(
            "NFC",
            text,
        )

        value = value.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        if self.config.normalize_whitespace:
            # Preserve newlines and tabs only structurally. Collapse spaces
            # within a line; this does not rewrite scientific content.
            value = "\n".join(
                re.sub(
                    r"[ \t]+",
                    " ",
                    line,
                ).strip()
                for line in value.split(
                    "\n"
                )
            )

        # Collapse excessive blank lines but retain paragraph boundaries.
        value = re.sub(
            r"\n[ \t]*\n(?:[ \t]*\n)+",
            "\n\n",
            value,
        )

        return value.strip()

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[_Paragraph]:
        """
        Conservative paragraph detection.

        Blank lines are the strongest signal. If extraction collapsed all
        blank lines, a second pass recognizes common paragraph indentation
        only when it is unambiguous.
        """
        raw_parts = re.split(
            r"\n\s*\n",
            text,
        )

        paragraphs: list[str] = []

        for part in raw_parts:
            value = part.strip()

            if not value:
                continue

            paragraphs.append(
                value
            )

        if len(paragraphs) > 1:
            return [
                _Paragraph(
                    text=value,
                    index=index,
                )
                for index, value in enumerate(
                    paragraphs
                )
            ]

        # Some PDF extractors collapse paragraphs into single newlines.
        # Preserve line-based structure only when several non-empty lines are
        # reasonably paragraph-like; do not aggressively split scientific
        # equations, lists, captions, or headings.
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) <= 1:
            return [
                _Paragraph(
                    text=text.strip(),
                    index=0,
                )
            ]

        reconstructed = self._reconstruct_paragraphs(
            lines
        )

        return [
            _Paragraph(
                text=value,
                index=index,
            )
            for index, value in enumerate(
                reconstructed
            )
        ]

    def _reconstruct_paragraphs(
        self,
        lines: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        current: list[str] = []
        repaired_lines = self._repair_extracted_line_breaks(lines)

        for line in repaired_lines:
            if self._looks_like_standalone_block(
                line
            ):
                if current:
                    result.append(
                        " ".join(current).strip()
                    )
                    current = []

                result.append(
                    line
                )
                continue

            current.append(
                line
            )

            if self._looks_like_paragraph_end(
                line
            ):
                result.append(
                    " ".join(current).strip()
                )
                current = []

        if current:
            result.append(
                " ".join(current).strip()
            )

        return [
            value
            for value in result
            if value
        ]

    @staticmethod
    def _looks_like_standalone_block(
        line: str,
    ) -> bool:
        lowered = line.lower()

        if re.match(
            r"^(?:figure|fig\.?|table|tab\.?|algorithm)\s*\d+",
            lowered,
        ):
            return True

        if re.match(
            r"^(?:appendix|references|bibliography)\b",
            lowered,
        ):
            return True

        if re.match(
            r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])[\.)]\s+\S+",
            line,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    @staticmethod
    def _looks_like_paragraph_end(
        line: str,
    ) -> bool:
        return bool(
            re.search(
                r"[.!?](?:[\"'”’)\]]+)?$",
                line,
            )
        )

    # ------------------------------------------------------------------
    # Joining
    # ------------------------------------------------------------------

    @staticmethod
    def _join_paragraphs(
        paragraphs: Sequence[str],
    ) -> str:
        return "\n\n".join(
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ).strip()

    @staticmethod
    def _join_sentences(
        sentences: Sequence[str],
    ) -> str:
        return " ".join(
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ).strip()

    # ------------------------------------------------------------------
    # Token count
    # ------------------------------------------------------------------

    def _count(
        self,
        text: str,
    ) -> int:
        return int(
            self.token_counter.count(
                text
            )
        )

    # ------------------------------------------------------------------
    # Page mapping
    # ------------------------------------------------------------------

    def _source_pages_for_chunk(
        self,
        *,
        section: PaperSectionLike,
    ) -> tuple[int, ...]:
        """
        Preserve page traceability supplied by section_parser.

        The current PaperSection contract gives page information at section
        granularity, not character/paragraph granularity. Therefore the safe
        default is to preserve all source pages belonging to the section
        instead of inventing false precision.

        A future extractor/section-parser version can provide paragraph-level
        page spans and this method can be replaced without changing the chunk
        schema.
        """
        pages = tuple(
            sorted(
                set(
                    int(page)
                    for page in (
                        section.source_pages
                        or ()
                    )
                    if isinstance(
                        page,
                        int,
                    )
                    and page > 0
                )
            )
        )

        if pages:
            return pages

        if (
            section.start_page is not None
            and section.end_page is not None
        ):
            return tuple(
                range(
                    section.start_page,
                    section.end_page + 1,
                )
            )

        return ()

    @staticmethod
    def _page_bounds(
        pages: Sequence[int],
    ) -> tuple[
        Optional[int],
        Optional[int],
    ]:
        if not pages:
            return None, None

        return (
            pages[0],
            pages[-1],
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _build_diagnostics(
        self,
        *,
        chunks: Sequence[SemanticChunk],
        duplicate_count: int,
        paper: StructuredResearchPaperLike,
        source_quality_warnings: Sequence[str],
    ) -> ChunkingDiagnostics:
        input_char_count = sum(
            len(section.text.strip())
            for section in paper.sections
            if isinstance(section.text, str)
        )
        input_page_count = len(
            {
                int(page)
                for section in paper.sections
                for page in (section.source_pages or ())
                if isinstance(page, int) and page > 0
            }
        )
        suspicious_source = bool(source_quality_warnings)

        declared_page_count = getattr(paper, "page_count", None)
        if declared_page_count is None:
            declared_page_count = getattr(paper, "total_pages", None)
        if (
            not isinstance(declared_page_count, int)
            or isinstance(declared_page_count, bool)
            or declared_page_count < 0
        ):
            declared_page_count = 0

        actual_pages = {
            int(page)
            for section in paper.sections
            for page in (section.source_pages or ())
            if isinstance(page, int) and page > 0
        }

        missing_source_pages = (
            tuple(
                page
                for page in range(1, declared_page_count + 1)
                if page not in actual_pages
            )
            if declared_page_count > 0
            else ()
        )

        source_coverage_ratio = (
            round(
                min(input_page_count / declared_page_count, 1.0),
                6,
            )
            if declared_page_count > 0
            else 0.0
        )

        if not chunks:
            return ChunkingDiagnostics(
                total_chunks=0,
                average_token_count=0.0,
                median_token_count=0.0,
                min_token_count=0,
                max_token_count=0,
                chunks_per_section={},
                average_overlap_tokens=0.0,
                max_overlap_tokens=0,
                oversized_chunks=0,
                tiny_chunks=0,
                duplicate_chunks=duplicate_count,
                token_fallback_chunks=0,
                page_mapping_mode=(
                    self.config.page_mapping_mode
                ),
                input_char_count=input_char_count,
                input_section_count=len(paper.sections),
                input_page_count=input_page_count,
                declared_page_count=declared_page_count,
                missing_source_pages=missing_source_pages,
                source_coverage_ratio=source_coverage_ratio,
                suspicious_source=suspicious_source,
                source_quality_warnings=tuple(source_quality_warnings),
            )

        token_counts = [
            chunk.token_count
            for chunk in chunks
        ]

        overlap_counts = [
            chunk.overlap_with_previous
            for chunk in chunks
        ]

        chunks_per_section: dict[
            str,
            int,
        ] = {}

        for chunk in chunks:
            chunks_per_section[
                chunk.section_id
            ] = (
                chunks_per_section.get(
                    chunk.section_id,
                    0,
                )
                + 1
            )

        return ChunkingDiagnostics(
            total_chunks=len(chunks),
            average_token_count=round(
                statistics.mean(
                    token_counts
                ),
                3,
            ),
            median_token_count=round(
                statistics.median(
                    token_counts
                ),
                3,
            ),
            min_token_count=min(
                token_counts
            ),
            max_token_count=max(
                token_counts
            ),
            chunks_per_section=dict(
                chunks_per_section
            ),
            average_overlap_tokens=round(
                statistics.mean(
                    overlap_counts
                ),
                3,
            ),
            max_overlap_tokens=max(
                overlap_counts
            ),
            oversized_chunks=sum(
                token_count
                > self.config.max_chunk_tokens
                for token_count in token_counts
            ),
            tiny_chunks=sum(
                self._is_tiny(
                    token_count
                )
                for token_count in token_counts
            ),
            duplicate_chunks=duplicate_count,
            token_fallback_chunks=sum(
                chunk.chunking_method
                == "token_fallback"
                for chunk in chunks
            ),
            input_char_count=input_char_count,
            input_section_count=len(paper.sections),
            input_page_count=input_page_count,
            declared_page_count=declared_page_count,
            missing_source_pages=missing_source_pages,
            source_coverage_ratio=source_coverage_ratio,
            suspicious_source=suspicious_source,
            source_quality_warnings=tuple(source_quality_warnings),
            page_mapping_mode=(
                self.config.page_mapping_mode
            ),
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _determine_status(
        self,
        *,
        paper: StructuredResearchPaperLike,
        chunks: Sequence[SemanticChunk],
        warnings: Sequence[str],
        diagnostics: ChunkingDiagnostics,
    ) -> str:
        if not chunks:
            if any(
                section.text.strip()
                and not (
                    section.section_type
                    == "references"
                    and not self.config.include_references
                )
                for section in paper.sections
            ):
                return "partial"

            return "success"

        if warnings:
            return "partial"

        # Source-quality warnings are deliberately surfaced as partial even when
        # fail_on_suspicious_source is disabled for debugging/backward compatibility.
        if diagnostics.suspicious_source:
            return "partial"

        # Token fallback means the semantic unit could not be kept intact and
        # had to be divided below the sentence boundary. This is an expected
        # edge case, but should remain visible as partial processing.
        if diagnostics.token_fallback_chunks > 0:
            return "partial"

        if paper.parsing_status in {
            "partial",
            "failed",
        }:
            return "partial"

        return "success"

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        *,
        paper: StructuredResearchPaperLike,
        result: ChunkedResearchPaper,
    ) -> None:
        if result.document_id != paper.document_id:
            raise ChunkingOutputValidationError(
                "Output document_id does not match input."
            )

        section_ids = {
            section.section_id
            for section in paper.sections
        }

        eligible_section_ids = {
            section.section_id
            for section in paper.sections
            if (
                isinstance(section.text, str)
                and section.text.strip()
                and not (
                    section.section_type == "references"
                    and not self.config.include_references
                )
            )
        }
        output_section_ids = {
            chunk.section_id
            for chunk in result.chunks
        }

        if self.config.require_full_section_coverage:
            missing_sections = eligible_section_ids - output_section_ids
            if missing_sections:
                ordered_missing = [
                    section.section_id
                    for section in paper.sections
                    if section.section_id in missing_sections
                ]
                raise ChunkingOutputValidationError(
                    "Output is missing non-empty source sections: "
                    f"{ordered_missing!r}."
                )

        previous_chunk: Optional[
            SemanticChunk
        ] = None

        for expected_index, chunk in enumerate(
            result.chunks
        ):
            if chunk.chunk_index != expected_index:
                raise ChunkingOutputValidationError(
                    "Chunk indices are not contiguous."
                )

            if chunk.document_id != paper.document_id:
                raise ChunkingOutputValidationError(
                    f"Chunk {chunk.chunk_id} has wrong document_id."
                )

            if chunk.section_id not in section_ids:
                raise ChunkingOutputValidationError(
                    f"Chunk {chunk.chunk_id} references "
                    "an unknown section."
                )

            if (
                chunk.token_count
                > self.config.max_chunk_tokens
            ):
                raise ChunkingOutputValidationError(
                    f"Chunk {chunk.chunk_id} exceeds "
                    "max_chunk_tokens."
                )

            actual_count = self._count(
                chunk.text
            )

            if actual_count != chunk.token_count:
                raise ChunkingOutputValidationError(
                    f"Token count mismatch for {chunk.chunk_id}: "
                    f"stored={chunk.token_count}, "
                    f"actual={actual_count}."
                )

            if chunk.char_count != len(
                chunk.text
            ):
                raise ChunkingOutputValidationError(
                    f"Character count mismatch for "
                    f"{chunk.chunk_id}."
                )

            if previous_chunk is not None:
                if (
                    chunk.section_id
                    != previous_chunk.section_id
                    and chunk.overlap_with_previous
                    != 0
                ):
                    raise ChunkingOutputValidationError(
                        "Overlap crossed a section boundary."
                    )

                if (
                    chunk.chunk_index
                    <= previous_chunk.chunk_index
                ):
                    raise ChunkingOutputValidationError(
                        "Chunk order is invalid."
                    )

            previous_chunk = chunk

        if result.diagnostics is not None:
            diagnostics = result.diagnostics
            if (
                diagnostics.declared_page_count > 0
                and diagnostics.input_page_count > 0
                and diagnostics.input_page_count
                < diagnostics.declared_page_count
            ):
                raise ChunkingOutputValidationError(
                    "Chunked source does not cover the declared document "
                    f"page count: covered={diagnostics.input_page_count}, "
                    f"declared={diagnostics.declared_page_count}, "
                    f"missing_pages={diagnostics.missing_source_pages[:25]!r}."
                )

        if (
            result.diagnostics is not None
            and result.diagnostics.oversized_chunks != 0
        ):
            raise ChunkingOutputValidationError(
                "Diagnostics report oversized chunks."
            )

        non_reference_text = any(
            isinstance(section.text, str)
            and section.text.strip()
            and not (
                section.section_type == "references"
                and not self.config.include_references
            )
            for section in paper.sections
        )
        if non_reference_text and result.chunk_count == 0:
            raise ChunkingOutputValidationError(
                "Non-empty source sections produced zero retrieval chunks."
            )

        if result.diagnostics is not None:
            if result.diagnostics.total_chunks != result.chunk_count:
                raise ChunkingOutputValidationError(
                    "Diagnostics total_chunks does not match chunk_count."
                )
            if result.diagnostics.input_section_count != len(paper.sections):
                raise ChunkingOutputValidationError(
                    "Diagnostics input_section_count is inconsistent."
                )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_text_key(
        text: str,
    ) -> str:
        value = unicodedata.normalize(
            "NFC",
            text,
        ).lower()

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value

    @staticmethod
    def _dedupe(
        values: Iterable[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = str(value).strip()

            if not value or value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result


# ============================================================================
# Functional API
# ============================================================================


def semantic_chunk(
    paper: StructuredResearchPaperLike,
    *,
    config: Optional[ChunkingConfig] = None,
    token_counter: Optional[TokenCounter] = None,
    sentence_splitter: Optional[
        ScientificSentenceSplitter
    ] = None,
) -> ChunkedResearchPaper:
    """
    Functional wrapper around SectionAwareSemanticChunker.
    """
    return SectionAwareSemanticChunker(
        config=config,
        token_counter=token_counter,
        sentence_splitter=sentence_splitter,
    ).chunk(
        paper
    )


# ============================================================================
# Synthetic test fixtures
# ============================================================================


@dataclass(frozen=True)
class _TestSection:
    section_id: str
    document_id: str
    section_type: str
    original_heading: Optional[str]
    normalized_heading: str
    level: int
    text: str
    start_page: Optional[int]
    end_page: Optional[int]
    source_pages: tuple[int, ...]
    parent_section_id: Optional[str]
    child_section_ids: tuple[str, ...]
    confidence: Optional[float] = 0.99


@dataclass(frozen=True)
class _TestPaper:
    document_id: str
    title: Optional[str]
    sections: tuple[_TestSection, ...]
    section_count: int
    parsing_status: str = "success"
    warnings: tuple[str, ...] = ()
    page_count: Optional[int] = None


def _make_section(
    section_id: str,
    section_type: str,
    text: str,
    *,
    level: int = 1,
    page_start: Optional[int] = 1,
    page_end: Optional[int] = 1,
    parent: Optional[str] = None,
) -> _TestSection:
    pages = (
        tuple(
            range(
                page_start,
                page_end + 1,
            )
        )
        if (
            page_start is not None
            and page_end is not None
        )
        else ()
    )

    heading = section_type.replace(
        "_",
        " ",
    ).title()

    return _TestSection(
        section_id=section_id,
        document_id="doc_test_001",
        section_type=section_type,
        original_heading=heading,
        normalized_heading=section_type,
        level=level,
        text=text,
        start_page=page_start,
        end_page=page_end,
        source_pages=pages,
        parent_section_id=parent,
        child_section_ids=(),
    )


def _make_paper(
    sections: Sequence[_TestSection],
    *,
    document_id: str = "doc_test_001",
    parsing_status: str = "success",
) -> _TestPaper:
    return _TestPaper(
        document_id=document_id,
        title="Synthetic Scientific Paper",
        sections=tuple(sections),
        section_count=len(sections),
        parsing_status=parsing_status,
        warnings=(),
    )


# ============================================================================
# Self-test suite
# ============================================================================


def run_self_test() -> None:
    """
    Execute comprehensive model-free tests.

    No PDF, GPU, model download, internet, FAISS, SPECTER2 or LLM is needed.
    """
    # ------------------------------------------------------------------
    # 1. Small abstract stays intact.
    # ------------------------------------------------------------------
    paper = _make_paper(
        [
            _make_section(
                "sec_abs",
                "abstract",
                (
                    "We propose a retrieval system for scientific "
                    "documents. The system preserves evidence and source "
                    "traceability."
                ),
            )
        ]
    )

    chunked = semantic_chunk(
        paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=0,
        ),
    )

    assert chunked.chunk_count == 1
    assert chunked.chunks[0].chunking_method == "paragraph"
    assert "source traceability" in chunked.chunks[0].text

    # ------------------------------------------------------------------
    # 2. Multiple paragraphs can form a coherent group.
    # ------------------------------------------------------------------
    multi_para = _make_paper(
        [
            _make_section(
                "sec_intro",
                "introduction",
                (
                    "The first paragraph establishes the research "
                    "problem and motivation.\n\n"
                    "The second paragraph describes the scientific "
                    "context and why retrieval quality matters.\n\n"
                    "The third paragraph states the contribution."
                ),
            )
        ]
    )

    multi_result = semantic_chunk(
        multi_para,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=0,
        ),
    )

    assert multi_result.chunk_count == 1
    assert multi_result.chunks[0].chunking_method == (
        "paragraph_group"
    )

    # ------------------------------------------------------------------
    # 3. Section boundaries are hard.
    # ------------------------------------------------------------------
    boundary_paper = _make_paper(
        [
            _make_section(
                "sec_intro",
                "introduction",
                (
                    "Introduction paragraph explaining the research "
                    "problem and motivation."
                ),
            ),
            _make_section(
                "sec_method",
                "methodology",
                (
                    "Methodology paragraph describing the training "
                    "procedure and model configuration."
                ),
            ),
        ]
    )

    boundary_result = semantic_chunk(
        boundary_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=50,
        ),
    )

    assert boundary_result.chunk_count == 2
    assert boundary_result.chunks[0].section_id == "sec_intro"
    assert boundary_result.chunks[1].section_id == "sec_method"
    assert boundary_result.chunks[1].overlap_with_previous == 0
    assert "training procedure" not in (
        boundary_result.chunks[0].text
    )

    # ------------------------------------------------------------------
    # 4. Nested subsections remain independently identifiable.
    # ------------------------------------------------------------------
    nested_paper = _make_paper(
        [
            _make_section(
                "sec_method",
                "methodology",
                (
                    "The overall methodology defines the experimental "
                    "pipeline."
                ),
                level=1,
            ),
            _make_section(
                "sec_dataset",
                "dataset",
                (
                    "The dataset contains annotated scientific images "
                    "collected from public repositories."
                ),
                level=2,
                parent="sec_method",
            ),
            _make_section(
                "sec_model",
                "model",
                (
                    "The model uses a transformer encoder with "
                    "multi-head attention."
                ),
                level=2,
                parent="sec_method",
            ),
        ]
    )

    nested_result = semantic_chunk(
        nested_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=0,
        ),
    )

    assert [
        chunk.section_id
        for chunk in nested_result.chunks
    ] == [
        "sec_method",
        "sec_dataset",
        "sec_model",
    ]

    # ------------------------------------------------------------------
    # 5. Long paragraph -> sentence fallback.
    # ------------------------------------------------------------------
    long_text = (
        "Sentence one describes the experimental setting. "
        "Sentence two describes the dataset construction. "
        "Sentence three describes model training. "
        "Sentence four describes the optimization procedure. "
        "Sentence five describes evaluation. "
        "Sentence six describes the reported metrics."
    )

    long_paper = _make_paper(
        [
            _make_section(
                "sec_long",
                "methodology",
                long_text,
            )
        ]
    )

    long_result = semantic_chunk(
        long_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=10,
            max_chunk_tokens=16,
            overlap_tokens=3,
        ),
    )

    assert long_result.chunk_count >= 2
    assert all(
        chunk.token_count <= 16
        for chunk in long_result.chunks
    )

    # ------------------------------------------------------------------
    # 6. Scientific abbreviations must not split incorrectly.
    # ------------------------------------------------------------------
    scientific = _make_paper(
        [
            _make_section(
                "sec_sci",
                "results",
                (
                    "The model achieved 94.7% accuracy, e.g. on the "
                    "held-out set. Fig. 2 shows the corresponding result. "
                    "The value was 0.95 in the second experiment."
                ),
            )
        ]
    )

    scientific_result = semantic_chunk(
        scientific,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=0,
        ),
    )

    assert scientific_result.chunk_count == 1

    # ------------------------------------------------------------------
    # 7. Token fallback for one indivisible long sentence.
    # ------------------------------------------------------------------
    long_sentence = (
        "This sentence contains "
        + " ".join(
            f"technical_term_{i}"
            for i in range(80)
        )
        + "."
    )

    fallback_paper = _make_paper(
        [
            _make_section(
                "sec_fallback",
                "methodology",
                long_sentence,
            )
        ]
    )

    fallback_result = semantic_chunk(
        fallback_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=15,
            max_chunk_tokens=20,
            overlap_tokens=3,
        ),
    )

    assert fallback_result.chunk_count >= 3
    assert any(
        chunk.chunking_method == "token_fallback"
        for chunk in fallback_result.chunks
    )
    assert all(
        chunk.token_count <= 20
        for chunk in fallback_result.chunks
    )

    # ------------------------------------------------------------------
    # 8. Reference exclusion.
    # ------------------------------------------------------------------
    reference_paper = _make_paper(
        [
            _make_section(
                "sec_ref",
                "references",
                (
                    "[1] A. Author. A scientific paper. Journal, 2024.\n\n"
                    "[2] B. Author. Another paper. Conference, 2025."
                ),
            )
        ]
    )

    excluded = semantic_chunk(
        reference_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            include_references=False,
        ),
    )

    assert excluded.chunk_count == 0
    assert excluded.chunking_status == "success"

    # ------------------------------------------------------------------
    # 9. Reference inclusion.
    # ------------------------------------------------------------------
    included = semantic_chunk(
        reference_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            include_references=True,
        ),
    )

    assert included.chunk_count >= 1
    assert all(
        chunk.section_type == "references"
        for chunk in included.chunks
    )

    # ------------------------------------------------------------------
    # 10. Page traceability.
    # ------------------------------------------------------------------
    page_paper = _make_paper(
        [
            _make_section(
                "sec_pages",
                "methodology",
                (
                    "This section starts on page four. "
                    "It continues with experimental details."
                ),
                page_start=4,
                page_end=6,
            )
        ]
    )

    page_result = semantic_chunk(
        page_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
        ),
    )

    page_chunk = page_result.chunks[0]

    assert page_chunk.start_page == 4
    assert page_chunk.end_page == 6
    assert page_chunk.source_pages == (
        4,
        5,
        6,
    )

    # ------------------------------------------------------------------
    # 11. Missing page metadata.
    # ------------------------------------------------------------------
    missing_pages = _make_paper(
        [
            _make_section(
                "sec_missing_pages",
                "results",
                "Results text without page mapping.",
                page_start=None,
                page_end=None,
            )
        ]
    )

    missing_result = semantic_chunk(
        missing_pages,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
        ),
    )

    assert missing_result.chunks[0].source_pages == ()
    assert missing_result.chunks[0].start_page is None
    assert missing_result.chunks[0].end_page is None

    # ------------------------------------------------------------------
    # 12. Source fidelity: numbers, model names and citations remain.
    # ------------------------------------------------------------------
    source_text = (
        "ResNet-50 achieved 94.7% accuracy on the Laboro-Tomato "
        "dataset [12]. The reported F1 score was 0.921."
    )

    source_paper = _make_paper(
        [
            _make_section(
                "sec_source",
                "results",
                source_text,
            )
        ]
    )

    source_result = semantic_chunk(
        source_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
        ),
    )

    source_chunk = source_result.chunks[0]
    assert "ResNet-50" in source_chunk.text
    assert "94.7%" in source_chunk.text
    assert "[12]" in source_chunk.text
    assert "0.921" in source_chunk.text

    # ------------------------------------------------------------------
    # 13. Deterministic IDs.
    # ------------------------------------------------------------------
    deterministic_config = ChunkingConfig(
        min_chunk_tokens=5,
        target_chunk_tokens=20,
        max_chunk_tokens=30,
        overlap_tokens=3,
    )

    deterministic_a = semantic_chunk(
        long_paper,
        config=deterministic_config,
    )
    deterministic_b = semantic_chunk(
        long_paper,
        config=deterministic_config,
    )

    assert [
        chunk.chunk_id
        for chunk in deterministic_a.chunks
    ] == [
        chunk.chunk_id
        for chunk in deterministic_b.chunks
    ]

    assert [
        chunk.text
        for chunk in deterministic_a.chunks
    ] == [
        chunk.text
        for chunk in deterministic_b.chunks
    ]

    # ------------------------------------------------------------------
    # 14. Duplicate text detection does not silently delete content.
    # ------------------------------------------------------------------
    duplicate_paper = _make_paper(
        [
            _make_section(
                "sec_dup",
                "discussion",
                (
                    "The same scientific observation is repeated here.\n\n"
                    "The same scientific observation is repeated here."
                ),
            )
        ]
    )

    duplicate_result = semantic_chunk(
        duplicate_paper,
        config=ChunkingConfig(
            min_chunk_tokens=2,
            target_chunk_tokens=5,
            max_chunk_tokens=20,
            overlap_tokens=0,
        ),
    )

    assert duplicate_result.chunk_count == 2
    assert duplicate_result.diagnostics is not None
    assert duplicate_result.diagnostics.duplicate_chunks >= 1

    # ------------------------------------------------------------------
    # 15. Empty section.
    # ------------------------------------------------------------------
    empty_paper = _make_paper(
        [
            _make_section(
                "sec_empty",
                "abstract",
                "",
            ),
            _make_section(
                "sec_valid",
                "introduction",
                "Valid introduction content.",
            ),
        ]
    )

    empty_result = semantic_chunk(
        empty_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
        ),
    )

    assert empty_result.chunk_count == 1
    assert empty_result.chunks[0].section_id == (
        "sec_valid"
    )

    # ------------------------------------------------------------------
    # 16. Partial failure isolation via an intentionally broken section.
    # ------------------------------------------------------------------
    class BrokenSection:
        section_id = "broken"
        document_id = "doc_test_001"
        section_type = "results"
        original_heading = "Results"
        normalized_heading = "results"
        level = 1
        text = 123  # invalid
        start_page = 1
        end_page = 1
        source_pages = (1,)
        parent_section_id = None
        child_section_ids = ()
        confidence = 0.5

    broken_paper = _TestPaper(
        document_id="doc_test_001",
        title="Broken Test",
        sections=(
            _make_section(
                "sec_good",
                "introduction",
                "A valid section survives.",
            ),
            BrokenSection(),  # type: ignore[arg-type]
        ),
        section_count=2,
        parsing_status="success",
    )

    # Invalid section metadata is a document input contract violation and
    # should fail before processing rather than silently corrupting output.
    try:
        semantic_chunk(
            broken_paper
        )
    except InvalidChunkingInputError:
        pass
    else:
        raise AssertionError(
            "Malformed section was not rejected."
        )

    # ------------------------------------------------------------------
    # 17. Unicode scientific text.
    # ------------------------------------------------------------------
    unicode_paper = _make_paper(
        [
            _make_section(
                "sec_unicode",
                "methodology",
                (
                    "The naïve baseline uses α = 0.95 and "
                    "temperature ΔT = 2°C. Résumé-level metadata is "
                    "not rewritten."
                ),
            )
        ]
    )

    unicode_result = semantic_chunk(
        unicode_paper,
        config=ChunkingConfig(
            min_chunk_tokens=5,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
        ),
    )

    assert "α" in unicode_result.chunks[0].text
    assert "0.95" in unicode_result.chunks[0].text
    assert "2°C" in unicode_result.chunks[0].text

    # ------------------------------------------------------------------
    # 18. Invalid document.
    # ------------------------------------------------------------------
    try:
        semantic_chunk(None)  # type: ignore[arg-type]
    except InvalidChunkingInputError:
        pass
    else:
        raise AssertionError(
            "None input was not rejected."
        )

    # ------------------------------------------------------------------
    # 19. Hard maximum validation.
    # ------------------------------------------------------------------
    assert all(
        chunk.token_count
        <= deterministic_config.max_chunk_tokens
        for chunk in deterministic_a.chunks
    )

    # ------------------------------------------------------------------
    # 20. P0 regression: full 46-page document coverage.
    # ------------------------------------------------------------------
    full_sections = tuple(
        _make_section(
            f"sec_page_{page}",
            "results",
            (
                f"Scientific content for source page {page}. "
                "The experiment evaluates retrieval quality, evidence "
                "preservation, source traceability, model performance, "
                "and reproducibility across the complete research paper."
            ),
            page_start=page,
            page_end=page,
        )
        for page in range(1, 47)
    )

    full_paper = _TestPaper(
        document_id="doc_test_001",
        title="46 Page Synthetic Paper",
        sections=full_sections,
        section_count=46,
        parsing_status="success",
        warnings=(),
        page_count=46,
    )

    full_result = semantic_chunk(
        full_paper,
        config=ChunkingConfig(
            min_chunk_tokens=10,
            target_chunk_tokens=35,
            max_chunk_tokens=50,
            overlap_tokens=5,
            require_full_section_coverage=True,
        ),
    )

    assert full_result.chunk_count >= 46
    assert {
        chunk.section_id for chunk in full_result.chunks
    } == {
        section.section_id for section in full_paper.sections
    }
    assert {
        page
        for chunk in full_result.chunks
        for page in chunk.source_pages
    } == set(range(1, 47))
    assert full_result.diagnostics is not None
    assert full_result.diagnostics.input_page_count == 46
    assert full_result.diagnostics.declared_page_count == 46
    assert full_result.diagnostics.source_coverage_ratio == 1.0

    incomplete_paper = _TestPaper(
        document_id="doc_test_001",
        title="Incomplete 46 Page Paper",
        sections=full_sections[:5],
        section_count=5,
        parsing_status="success",
        warnings=(),
        page_count=46,
    )

    try:
        semantic_chunk(
            incomplete_paper,
            config=ChunkingConfig(
                min_chunk_tokens=10,
                target_chunk_tokens=35,
                max_chunk_tokens=50,
                overlap_tokens=5,
                require_full_section_coverage=True,
            ),
        )
    except InvalidChunkingInputError:
        pass
    else:
        raise AssertionError(
            "Incomplete 46-page source was accepted."
        )

    # ------------------------------------------------------------------
    # 20. No cross-section overlap.
    # ------------------------------------------------------------------
    assert all(
        (
            index == 0
            or chunk.section_id
            != deterministic_a.chunks[
                index - 1
            ].section_id
            or chunk.overlap_with_previous
            >= 0
        )
        for index, chunk in enumerate(
            deterministic_a.chunks
        )
    )

    print(
        "SemanticChunker self-test: PASSED "
        f"({len(boundary_result.chunks)} boundary chunks, "
        f"{len(fallback_result.chunks)} fallback chunks)"
    )


# ============================================================================
# Module entry point
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    run_self_test()