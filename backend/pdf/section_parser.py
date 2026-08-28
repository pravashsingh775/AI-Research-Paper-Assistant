"""
Production-quality scientific research-paper section parser.

Responsibility
--------------
This module performs STRUCTURE EXTRACTION only:

    ExtractedDocument -> StructuredResearchPaper

It does not perform PDF reading, OCR, summarization, semantic analysis,
embeddings, vector search, RAG, Q&A, ranking, or scientific-quality analysis.

Design goals
------------
- deterministic and local by default
- conservative heading detection
- multiple independent heading signals
- support IEEE / ACM / arXiv / journal-style numbering
- preserve original heading and section text
- reconstruct subsection hierarchy
- preserve document/page/source traceability
- deterministic section IDs
- explicit uncertainty and partial-success handling
- no global mutable parser state
- easy unit testing without PDF/LLM/GPU/internet

Expected extractor interface
-----------------------------
The parser consumes an object compatible with backend.pdf.extractor's
ExtractedDocument / ExtractedPage:

    document_id
    filename
    page_count
    pages
        page_number
        text
        char_count
        word_count
        extraction_status
        blocks (optional)
    full_text
    metadata
    extraction_status

The parser intentionally does not import or reopen the PDF.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# ============================================================================
# Public constants
# ============================================================================

VALID_PARSING_STATUSES = frozenset(
    {"success", "partial", "failed"}
)

VALID_SECTION_LEVELS = frozenset(range(1, 10))

# Canonical section types. "unknown" is deliberately separate from "other":
# unknown means classification evidence is weak; other means the heading is
# confidently a section but has no safe canonical mapping.
CANONICAL_SECTION_TYPES = frozenset(
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


class SectionParsingError(RuntimeError):
    """Base error for section parsing."""


class InvalidExtractedDocumentError(
    SectionParsingError,
    ValueError,
):
    """Raised when the extractor output violates parser invariants."""


class SectionOutputValidationError(SectionParsingError):
    """Raised when the parser creates an internally invalid output."""


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True)
class SectionParserConfig:
    """
    Configuration for deterministic section detection.

    Scores are heuristic engineering signals, not probabilities and are not
    statistically calibrated confidence values.
    """

    # Candidate-level thresholds.
    heading_accept_threshold: float = 0.64
    strong_heading_threshold: float = 0.82
    classification_threshold: float = 0.72
    ambiguity_margin: float = 0.10

    # Text constraints.
    min_heading_chars: int = 2
    max_heading_chars: int = 180
    max_heading_words: int = 18
    max_body_like_heading_chars: int = 120

    # Numbered headings receive stronger structural evidence.
    numbered_heading_bonus: float = 0.18
    known_heading_bonus: float = 0.24
    isolated_line_bonus: float = 0.14
    capitalization_bonus: float = 0.06
    short_heading_bonus: float = 0.05
    page_position_bonus: float = 0.04

    # Body/caption/list penalties.
    sentence_penalty: float = 0.28
    caption_penalty: float = 0.55
    equation_penalty: float = 0.50
    list_penalty: float = 0.35
    body_phrase_penalty: float = 0.26
    punctuation_penalty: float = 0.08

    # Conservative unnumbered detection.
    require_known_vocab_for_weak_unnumbered: bool = True

    # Reference handling.
    references_terminal_bias: float = 0.08
    stop_subheading_detection_in_references: bool = True

    # Optional extraction blocks can improve candidate quality.
    use_text_blocks: bool = True

    # If a page has extremely little text, candidates on it are not promoted
    # merely because a line looks short.
    minimum_page_chars_for_weak_heading: int = 80

    # Prevent pathological repeated scanning.
    max_candidate_lines_per_page: int = 5000

    # Robustness / recovery.
    # A real PDF may contain no reliably detectable headings even though the
    # extracted text is perfectly usable. Never discard that content.
    preserve_preamble: bool = True
    preserve_unsectioned_document: bool = True

    # Some PDF layouts wrap a heading over two physical lines, e.g.
    # "2 Proposed" / "Method". Try a bounded structural recovery.
    detect_two_line_headings: bool = True

    # PDF text extraction can split one heading word across physical lines,
    # e.g. "EXAM" / "PLES" -> "EXAMPLES". Recovery is deliberately bounded
    # and never mutates source page text.
    detect_word_fragment_headings: bool = True
    word_fragment_max_chars: int = 10
    word_fragment_min_first_chars: int = 3
    word_fragment_min_second_chars: int = 2
    word_fragment_max_second_chars: int = 8
    word_fragment_bonus: float = 0.16

    # Table-of-contents protection. TOC entries can be textually identical to
    # real headings, so detection is performed at page level before headings.
    detect_table_of_contents: bool = True
    toc_scan_page_limit: int = 12
    toc_min_known_entries: int = 3
    toc_max_page_chars: int = 4000
    toc_navigation_density_threshold: float = 0.45
    uppercase_unknown_heading_accept: bool = True

    # A fallback section is structural preservation, not a semantic claim.
    fallback_section_confidence: Optional[float] = None

    # Strongly reconstructed/isolated headings may be structurally valid while
    # having no safe canonical semantic label. Such headings are preserved as
    # "other" instead of "unknown". The threshold is deliberately below the
    # global strong-heading threshold because recovered headings already carry
    # an explicit structural recovery bonus.
    unknown_heading_fallback_threshold: float = 0.75
    enable_unknown_heading_fallback: bool = True

    def __post_init__(self) -> None:
        for name in (
            "heading_accept_threshold",
            "strong_heading_threshold",
            "classification_threshold",
            "ambiguity_margin",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if self.heading_accept_threshold > self.strong_heading_threshold:
            raise ValueError(
                "heading_accept_threshold cannot exceed "
                "strong_heading_threshold."
            )

        if self.min_heading_chars < 1:
            raise ValueError("min_heading_chars must be >= 1.")

        if self.max_heading_chars < self.min_heading_chars:
            raise ValueError(
                "max_heading_chars must be >= min_heading_chars."
            )

        if self.max_heading_words < 1:
            raise ValueError("max_heading_words must be >= 1.")

        if self.max_body_like_heading_chars < self.min_heading_chars:
            raise ValueError(
                "max_body_like_heading_chars is too small."
            )

        if self.minimum_page_chars_for_weak_heading < 0:
            raise ValueError(
                "minimum_page_chars_for_weak_heading must be >= 0."
            )

        if self.max_candidate_lines_per_page < 1:
            raise ValueError(
                "max_candidate_lines_per_page must be >= 1."
            )

        if self.toc_scan_page_limit < 1:
            raise ValueError(
                "toc_scan_page_limit must be >= 1."
            )

        if self.toc_min_known_entries < 1:
            raise ValueError(
                "toc_min_known_entries must be >= 1."
            )

        if self.toc_max_page_chars < 1:
            raise ValueError(
                "toc_max_page_chars must be >= 1."
            )

        if not 0.0 <= self.toc_navigation_density_threshold <= 1.0:
            raise ValueError(
                "toc_navigation_density_threshold must be between 0 and 1."
            )

        if self.fallback_section_confidence is not None and not (
            0.0 <= self.fallback_section_confidence <= 1.0
        ):
            raise ValueError(
                "fallback_section_confidence must be between 0 and 1 "
                "when provided."
            )

        if not 0.0 <= self.unknown_heading_fallback_threshold <= 1.0:
            raise ValueError(
                "unknown_heading_fallback_threshold must be between 0 and 1."
            )


@dataclass(frozen=True)
class HeadingCandidate:
    """
    Internal/public diagnostic representation of a possible heading.

    confidence is a heuristic score. It is intentionally not described as a
    calibrated probability.
    """

    page_number: int
    text: str
    normalized_heading: str
    section_type: str
    level: int
    confidence: float
    classification_confidence: float
    start_offset: int
    end_offset: int
    line_index: int
    block_index: Optional[int] = None
    numbering: Optional[str] = None
    numbering_depth: Optional[int] = None
    score_components: Mapping[str, float] = field(
        default_factory=dict
    )
    warnings: tuple[str, ...] = ()

    @property
    def is_numbered(self) -> bool:
        return bool(self.numbering)


@dataclass(frozen=True)
class PaperSection:
    """
    A source-preserving logical section of a research paper.

    section_type is a structural classification only. It does not mean the
    parser has interpreted the science contained in the section.
    """

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
    confidence: Optional[float]

    def __post_init__(self) -> None:
        if not self.section_id:
            raise ValueError("section_id cannot be empty.")

        if not self.document_id:
            raise ValueError("document_id cannot be empty.")

        if self.section_type not in CANONICAL_SECTION_TYPES:
            raise ValueError(
                f"Unsupported section_type: {self.section_type!r}"
            )

        if not self.normalized_heading:
            raise ValueError(
                "normalized_heading cannot be empty."
            )

        if self.level not in VALID_SECTION_LEVELS:
            raise ValueError(
                f"Invalid section level: {self.level}"
            )

        pages = tuple(self.source_pages)

        if pages != tuple(sorted(set(pages))):
            raise ValueError(
                "source_pages must be sorted and unique."
            )

        if self.start_page is None or self.end_page is None:
            if pages:
                raise ValueError(
                    "source_pages cannot be populated when page bounds "
                    "are missing."
                )
        else:
            if self.start_page > self.end_page:
                raise ValueError(
                    "start_page cannot exceed end_page."
                )

            if pages and (
                pages[0] != self.start_page
                or pages[-1] != self.end_page
            ):
                raise ValueError(
                    "source_pages do not match section page bounds."
                )

        if self.confidence is not None and not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class StructuredResearchPaper:
    """
    Complete structural representation consumed by the chunker.

    title is metadata/structure only. It is never generated by summarization.
    """

    document_id: str
    title: Optional[str]
    sections: tuple[PaperSection, ...]
    section_count: int
    parsing_status: str
    warnings: tuple[str, ...] = ()
    detected_headings: tuple[HeadingCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id cannot be empty.")

        if self.section_count != len(self.sections):
            raise ValueError(
                "section_count must match sections length."
            )

        if self.parsing_status not in VALID_PARSING_STATUSES:
            raise ValueError(
                f"Invalid parsing_status: "
                f"{self.parsing_status!r}"
            )

        ids = [
            section.section_id
            for section in self.sections
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Section IDs must be unique."
            )


# ============================================================================
# Extractor-compatible protocols
# ============================================================================


class ExtractedPageLike(Protocol):
    page_number: int
    text: str


class ExtractedDocumentLike(Protocol):
    document_id: str
    page_count: int
    pages: Sequence[ExtractedPageLike]
    full_text: str
    metadata: Mapping[str, Any]
    extraction_status: str


# ============================================================================
# Alias maps
# ============================================================================


def _clean_alias(value: str) -> str:
    """
    Normalize a heading label for alias matching only.

    Section content is never passed through this function.
    """
    value = unicodedata.normalize("NFC", value)
    value = value.strip().lower()

    # Remove common heading numbering and appendix letter prefixes.
    value = re.sub(
        r"^(?:(?:\d+\.)+\d*|[ivxlcdm]+\.|[a-z]\.)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.rstrip(":")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .:-")

    # Normalize typographic variants for matching.
    value = value.replace("&", " and ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# Exact aliases intentionally kept maintainable rather than enormous.
SECTION_ALIASES: Mapping[str, str] = {
    # Abstract / front matter
    "abstract": "abstract",
    "abstracts": "abstract",
    "keywords": "keywords",
    "key words": "keywords",
    "key words and phrases": "keywords",
    "index terms": "keywords",
    "index terms—": "keywords",
    "index terms:": "keywords",

    # Introduction / context
    "introduction": "introduction",
    "background": "background",
    "background and motivation": "background",
    "motivation": "background",
    "overview": "other",
    "related work": "related_work",
    "related works": "related_work",
    "literature review": "literature_review",
    "literature survey": "literature_review",
    "state of the art": "related_work",
    "preliminaries": "preliminaries",
    "preliminaries and notation": "preliminaries",
    "problem statement": "problem_statement",
    "research problem": "problem_statement",
    "research questions": "research_questions",
    "research question": "research_questions",

    # Method
    "method": "methodology",
    "methods": "methodology",
    "methodology": "methodology",
    "materials and methods": "methodology",
    "materials methods": "methodology",
    "proposed method": "methodology",
    "proposed methodology": "methodology",
    "proposed approach": "methodology",
    "approach": "other",
    "framework": "methodology",
    "system design": "methodology",
    "design and methodology": "methodology",
    "experimental setup": "experimental_setup",
    "experimental settings": "experimental_setup",
    "experimental configuration": "experimental_setup",
    "implementation": "implementation",
    "implementation details": "implementation",

    # Data/model
    "dataset": "dataset",
    "datasets": "dataset",
    "data": "dataset",
    "data collection": "dataset",
    "data preparation": "dataset",
    "data preprocessing": "dataset",
    "preprocessing": "dataset",
    "model": "model",
    "models": "model",
    "model architecture": "architecture",
    "architecture": "architecture",
    "network architecture": "architecture",
    "system architecture": "architecture",

    # Experiments/results
    "experiment": "experiments",
    "experiments": "experiments",
    "experimental evaluation": "evaluation",
    "evaluation": "evaluation",
    "evaluation metrics": "evaluation",
    "results": "results",
    "experimental results": "results",
    "empirical results": "results",
    "findings": "results",
    "discussion": "discussion",
    "results and discussion": "results_discussion",
    "results discussion": "results_discussion",
    "results and analysis": "results_discussion",
    "analysis and discussion": "results_discussion",

    # Ending
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "concluding remarks": "conclusion",
    "future work": "future_work",
    "future directions": "future_work",
    "future research": "future_work",
    "limitations": "limitations",
    "limitations and future work": "limitations",
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",

    # References
    "references": "references",
    "bibliography": "references",
    "works cited": "references",
    "literature cited": "references",

    # Appendix
    "appendix": "appendix",
    "appendices": "appendix",
    "supplementary material": "appendix",
    "supplementary materials": "appendix",
    "supplementary information": "appendix",
}


# Known multiword aliases need phrase matching before generic token matching.
KNOWN_SECTION_LABELS = frozenset(
    SECTION_ALIASES.keys()
)

# Titles that are dangerous to classify from weak evidence.
FALSE_HEADING_EXACT = frozenset(
    {
        "input",
        "output",
        "note",
        "notes",
        "where",
        "proof",
        "example",
        "examples",
        "algorithm",
        "algorithm 1",
        "algorithm 2",
        "figure",
        "table",
        "et al",
    }
)


# ============================================================================
# Compiled patterns
# ============================================================================


# Numeric:
#   1 Introduction
#   1. Introduction
#   2.1 Dataset
#   2.1.1 Training
_NUMERIC_RE = re.compile(
    r"^\s*"
    r"(?P<num>\d+(?:\.\d+)*)"
    r"(?:[\.)])?"
    r"\s+"
    r"(?P<label>.+?)"
    r"\s*$"
)

# Roman:
#   I. INTRODUCTION
#   II. METHODS
_ROMAN_RE = re.compile(
    r"^\s*"
    r"(?P<num>[IVXLCDM]+)"
    r"\.\s+"
    r"(?P<label>.+?)"
    r"\s*$",
    flags=re.IGNORECASE,
)

# Lettered:
#   A. Introduction
#   B. PROPOSED METHOD
_LETTER_RE = re.compile(
    r"^\s*"
    r"(?P<num>[A-Z])"
    r"\.\s+"
    r"(?P<label>.+?)"
    r"\s*$",
)

# Appendix A / Appendix B.
_APPENDIX_RE = re.compile(
    r"^\s*appendix(?:\s+|\s*[-:]\s*)"
    r"(?P<label>[A-Z])?"
    r"\s*(?P<rest>.*)$",
    flags=re.IGNORECASE,
)

# Common captions.
_CAPTION_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:figure|fig\.?)\s*\d+[a-z]?"
    r"|(?:table|tab\.?)\s*\d+[a-z]?"
    r"|(?:algorithm)\s*\d+[a-z]?"
    r")"
    r"(?:\s*[:.\-]\s*|\s+|$)",
    flags=re.IGNORECASE,
)

# Equation-ish lines.
_EQUATION_RE = re.compile(
    r"^\s*(?:"
    r"\(?\d+(?:\.\d+)*\)?\s*"
    r"|"
    r"(?:[A-Za-z]\s*=\s*.+)"
    r"|"
    r"(?:.*[∑∫√≤≥≈≠→←↔±×÷].*)"
    r")\s*$"
)

# Bullets / numbered list items that are not section numbering.
_LIST_RE = re.compile(
    r"^\s*(?:"
    r"[-*•▪◦‣]\s+"
    r"|"
    r"\(\d+\)\s+"
    r"|"
    r"\d+\)\s+"
    r"|"
    r"[a-z]\)\s+"
    r")",
    flags=re.IGNORECASE,
)

# Sentence-like ending. We deliberately do not treat a colon alone as proof
# of body text because "Keywords:" is a legitimate heading-like construct.
_SENTENCE_END_RE = re.compile(
    r"[.!?](?:[\"'”’)\]]+)?$"
)

# Leading/trailing punctuation noise.
_HEADING_EDGE_RE = re.compile(
    r"^[\s:;,.]+|[\s;,.]+$"
)

# Whitespace.
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Numbering after normalization.
_NUMBERING_ONLY_RE = re.compile(
    r"^(?P<num>(?:\d+\.)*\d+|[IVXLCDM]+|[A-Z])$",
    flags=re.IGNORECASE,
)


# ============================================================================
# Internal candidate record
# ============================================================================


@dataclass(frozen=True)
class _LineRecord:
    page_number: int
    text: str
    start_offset: int
    end_offset: int
    line_index: int
    block_index: Optional[int]
    previous_blank: bool
    next_blank: bool
    page_char_count: int
    page_line_count: int


@dataclass(frozen=True)
class _ClassifiedHeading:
    section_type: str
    confidence: float
    ambiguous: bool
    warning: Optional[str]


# ============================================================================
# Public parser
# ============================================================================


class SectionParser:
    """
    Deterministic scientific-paper section parser.

    Usage
    -----
    parser = SectionParser()
    paper = parser.parse(extracted_document)

    The parser never opens the PDF. It consumes only the extractor result.
    """

    def __init__(
        self,
        *,
        config: Optional[SectionParserConfig] = None,
    ) -> None:
        self.config = config or SectionParserConfig()

        # Instance-local immutable lookup data. No shared mutable state.
        self._aliases = {
            _clean_alias(key): value
            for key, value in SECTION_ALIASES.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        document: ExtractedDocumentLike,
    ) -> StructuredResearchPaper:
        """
        Parse one ExtractedDocument into a structured research paper.

        The method is deterministic for identical input/configuration.
        """
        logger.info(
            "Starting section parsing: document_id=%s",
            getattr(document, "document_id", None),
        )

        self._validate_document(document)

        document_id = document.document_id

        try:
            page_records = self._build_page_records(document)

            if not page_records:
                raise InvalidExtractedDocumentError(
                    "ExtractedDocument contains no usable pages."
                )

            toc_pages = self._detect_toc_pages(
                page_records=page_records,
            )

            candidates = self._detect_heading_candidates(
                document=document,
                page_records=page_records,
                excluded_pages=toc_pages,
            )

            candidates = self._postprocess_candidates(
                candidates
            )

            title = self._detect_title(
                document=document,
                page_records=page_records,
                candidates=candidates,
            )

            sections = self._build_sections(
                document=document,
                page_records=page_records,
                candidates=candidates,
                excluded_pages=toc_pages,
            )

            sections = self._reconstruct_hierarchy(
                sections
            )

            status, warnings = self._determine_status(
                document=document,
                candidates=candidates,
                sections=sections,
            )

            warnings = self._dedupe_warnings(warnings)

            result = StructuredResearchPaper(
                document_id=document_id,
                title=title,
                sections=tuple(sections),
                section_count=len(sections),
                parsing_status=status,
                warnings=tuple(warnings),
                detected_headings=tuple(candidates),
            )

            self._validate_output(
                result=result,
                document=document,
            )

            logger.info(
                "Section parsing completed: document_id=%s "
                "sections=%d status=%s",
                document_id,
                result.section_count,
                result.parsing_status,
            )

            return result

        except SectionParsingError:
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected section parser failure: "
                "document_id=%s",
                document_id,
            )
            raise SectionParsingError(
                "Unexpected failure during section parsing."
            ) from exc

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_document(
        document: ExtractedDocumentLike,
    ) -> None:
        if document is None:
            raise InvalidExtractedDocumentError(
                "document cannot be None."
            )

        document_id = getattr(
            document,
            "document_id",
            None,
        )

        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidExtractedDocumentError(
                "ExtractedDocument.document_id is required."
            )

        page_count = getattr(
            document,
            "page_count",
            None,
        )

        if not isinstance(page_count, int) or page_count <= 0:
            raise InvalidExtractedDocumentError(
                "ExtractedDocument.page_count must be a positive integer."
            )

        pages = getattr(
            document,
            "pages",
            None,
        )

        if pages is None:
            raise InvalidExtractedDocumentError(
                "ExtractedDocument.pages is required."
            )

        if len(pages) != page_count:
            raise InvalidExtractedDocumentError(
                "page_count does not match pages length."
            )

        page_numbers: list[int] = []

        for page in pages:
            number = getattr(
                page,
                "page_number",
                None,
            )

            if not isinstance(number, int) or number <= 0:
                raise InvalidExtractedDocumentError(
                    "Every extracted page must have a positive "
                    "1-based page_number."
                )

            text = getattr(page, "text", None)

            if not isinstance(text, str):
                raise InvalidExtractedDocumentError(
                    f"Page {number} text must be a string."
                )

            page_status = getattr(page, "extraction_status", None)
            if page_status is not None and page_status not in {
                "success",
                "partial",
                "failed",
                "ocr_required",
            }:
                raise InvalidExtractedDocumentError(
                    f"Page {number} has invalid extraction_status "
                    f"{page_status!r}."
                )

            declared_char_count = getattr(page, "char_count", None)
            if declared_char_count is not None:
                if not isinstance(declared_char_count, int) or declared_char_count < 0:
                    raise InvalidExtractedDocumentError(
                        f"Page {number} has invalid char_count."
                    )
                if declared_char_count != len(text):
                    raise InvalidExtractedDocumentError(
                        f"Page {number} char_count does not match text length."
                    )

            page_numbers.append(number)

        expected = list(
            range(1, page_count + 1)
        )

        if page_numbers != expected:
            raise InvalidExtractedDocumentError(
                "Extracted pages must be ordered exactly 1..N."
            )

        full_text = getattr(
            document,
            "full_text",
            None,
        )

        if not isinstance(full_text, str):
            raise InvalidExtractedDocumentError(
                "ExtractedDocument.full_text must be a string."
            )

        extraction_status = getattr(
            document,
            "extraction_status",
            None,
        )

        if extraction_status not in {
            "success",
            "partial",
            "ocr_required",
            "failed",
        }:
            raise InvalidExtractedDocumentError(
                "Invalid ExtractedDocument.extraction_status."
            )

    # ------------------------------------------------------------------
    # Page representation
    # ------------------------------------------------------------------

    def _build_page_records(
        self,
        document: ExtractedDocumentLike,
    ) -> list[list[_LineRecord]]:
        """
        Convert each page into offset-aware line records.

        Offsets are page-local and refer to the exact normalized page text
        used by the parser. We intentionally do not rewrite the page content
        beyond newline/whitespace normalization required for structural
        analysis.
        """
        result: list[list[_LineRecord]] = []

        for page in document.pages:
            text = self._normalize_page_structure_text(
                page.text
            )

            lines = text.splitlines()

            # Preserve an empty-page representation.
            if not lines:
                result.append([])
                continue

            page_records: list[_LineRecord] = []

            cursor = 0

            for line_index, line in enumerate(lines):
                start = cursor
                end = start + len(line)

                previous_blank = (
                    line_index == 0
                    or not lines[line_index - 1].strip()
                )

                next_blank = (
                    line_index == len(lines) - 1
                    or not lines[line_index + 1].strip()
                )

                block_index = None

                # Optional block mapping is best-effort. The parser never
                # requires block metadata.
                if self.config.use_text_blocks:
                    block_index = self._find_block_index_for_line(
                        page=page,
                        line=line,
                    )

                page_records.append(
                    _LineRecord(
                        page_number=page.page_number,
                        text=line,
                        start_offset=start,
                        end_offset=end,
                        line_index=line_index,
                        block_index=block_index,
                        previous_blank=previous_blank,
                        next_blank=next_blank,
                        page_char_count=len(text),
                        page_line_count=len(lines),
                    )
                )

                cursor = end + 1  # splitlines removed "\n"

            result.append(page_records)

        return result

    @staticmethod
    def _normalize_page_structure_text(
        text: str,
    ) -> str:
        """
        Minimal structural normalization.

        Scientific section content is never normalized through this function
        after it is assigned to a section; this is only used to reason about
        line boundaries.
        """
        text = unicodedata.normalize(
            "NFC",
            text.replace("\r\n", "\n").replace("\r", "\n"),
        )

        # Remove non-semantic zero-width formatting characters while keeping
        # ZWNJ/ZWJ and variation selector behavior intact.
        chars: list[str] = []

        for char in text:
            if unicodedata.category(char) == "Cf":
                if ord(char) not in {
                    0x200C,
                    0x200D,
                    0xFE0F,
                }:
                    continue
            chars.append(char)

        text = "".join(chars)

        # Trailing whitespace does not affect source meaning and makes
        # candidate matching more stable.
        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        return text

    @staticmethod
    def _find_block_index_for_line(
        *,
        page: Any,
        line: str,
    ) -> Optional[int]:
        blocks = getattr(
            page,
            "blocks",
            None,
        )

        if not blocks:
            return None

        normalized_line = line.strip()

        if not normalized_line:
            return None

        for block in blocks:
            block_text = getattr(
                block,
                "text",
                None,
            )

            if not isinstance(block_text, str):
                continue

            if normalized_line in block_text:
                return getattr(
                    block,
                    "block_index",
                    None,
                )

        return None

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _detect_toc_pages(
        self,
        *,
        page_records: Sequence[Sequence[_LineRecord]],
    ) -> set[int]:
        """
        Detect likely table-of-contents pages before heading extraction.

        TOC entries can be indistinguishable from real section headings.
        Detection therefore uses page-level evidence and never removes source
        text; it only prevents TOC lines from becoming structural headings.
        """
        if not self.config.detect_table_of_contents:
            return set()

        toc_pages: set[int] = set()

        for page_index, records in enumerate(
            page_records[: self.config.toc_scan_page_limit]
        ):
            if not records:
                continue

            nonempty = [
                record.text.strip()
                for record in records
                if record.text.strip()
            ]
            if not nonempty:
                continue

            page_text = "\n".join(nonempty)
            if len(page_text) > self.config.toc_max_page_chars:
                continue

            normalized = [_clean_alias(line) for line in nonempty]
            marker_present = any(
                line in {
                    "contents",
                    "table of contents",
                    "table contents",
                    "content",
                }
                for line in normalized
            )

            known_entries = sum(
                1 for line in normalized
                if line in self._aliases
            )

            navigation_like = sum(
                1
                for line in nonempty
                if self._looks_like_toc_navigation_line(line)
            )
            navigation_density = (
                navigation_like / max(1, len(nonempty))
            )

            if known_entries < self.config.toc_min_known_entries:
                continue

            if marker_present:
                toc_pages.add(page_index + 1)
                continue

            # Marker-less TOCs require stronger navigation evidence and one
            # additional known section label to avoid suppressing body pages.
            if (
                navigation_density
                >= self.config.toc_navigation_density_threshold
                and known_entries >= self.config.toc_min_known_entries + 1
            ):
                toc_pages.add(page_index + 1)

        if toc_pages:
            logger.info(
                "Detected likely table-of-contents page(s): %s",
                sorted(toc_pages),
            )

        return toc_pages

    @staticmethod
    def _looks_like_toc_navigation_line(text: str) -> bool:
        value = text.strip()
        if not value:
            return False

        # Dotted leaders: "Introduction ........ 3"
        if re.search(r"\.{2,}\s*\d+\s*$", value):
            return True

        # Extracted TOCs sometimes lose dotted leaders but retain page numbers.
        if re.search(r"\s+\d{1,4}\s*$", value):
            return True

        if re.search(r"\.{2,}\s*$", value):
            return True

        return False

    def _detect_heading_candidates(
        self,
        *,
        document: ExtractedDocumentLike,
        page_records: Sequence[Sequence[_LineRecord]],
        excluded_pages: Optional[set[int]] = None,
    ) -> list[HeadingCandidate]:
        """
        Detect structural headings with bounded two-line recovery.

        Recovery has two distinct modes:

        1. normal wrapped headings:
           "1 Proposed" + "Methodology" -> "1 Proposed Methodology"

        2. PDF word-fragment headings:
           "THEORETICAL RESULTS ON LEARNING FROM EXAM" +
           "PLES" -> "THEORETICAL RESULTS ON LEARNING FROM EXAMPLES"

        The second mode is especially important for legacy PDFs where the text
        extractor inserts a physical line break in the middle of a word. Source
        page text is never rewritten; only the diagnostic heading candidate uses
        the reconstructed label.
        """
        candidates: list[HeadingCandidate] = []
        references_detected = False
        excluded_pages = excluded_pages or set()

        for page_index, original_lines in enumerate(page_records):
            page_number = page_index + 1

            if page_number in excluded_pages:
                logger.debug(
                    "Skipping heading detection on excluded structural page %d.",
                    page_number,
                )
                continue

            lines = list(original_lines)

            if len(lines) > self.config.max_candidate_lines_per_page:
                logger.warning(
                    "Page %d exceeds candidate-line limit; truncating heading scan.",
                    page_index + 1,
                )
                lines = lines[: self.config.max_candidate_lines_per_page]

            line_index = 0
            while line_index < len(lines):
                record = lines[line_index]

                if not record.text.strip():
                    line_index += 1
                    continue

                candidate = self._evaluate_line(record)

                # Once References is reached, reference entries must remain body
                # text. Only a strong Appendix may start a later section.
                if references_detected:
                    if (
                        candidate is not None
                        and candidate.section_type == "appendix"
                        and candidate.confidence
                        >= self.config.strong_heading_threshold
                    ):
                        candidates.append(candidate)
                    line_index += 1
                    continue

                # Try recovery BEFORE emitting the individual candidates.
                # This is the key fix for "EXAM" / blank / "PLES": the old
                # implementation required both physical lines to be adjacent
                # without a blank line, which is not true for this PDF.
                recovered = None
                consumed = 1

                if (
                    self.config.detect_two_line_headings
                    and line_index + 1 < len(lines)
                ):
                    next_record = lines[line_index + 1]

                    # Case A: immediate next physical line.
                    immediate_pair = bool(next_record.text.strip())

                    if immediate_pair:
                        recovered = self._recover_two_line_heading(
                            record,
                            next_record,
                        )

                    # Case B: one blank line between fragments. This is common
                    # in legacy extraction when a visual line break becomes a
                    # paragraph separator. We only permit it for a high-signal
                    # word fragment pair.
                    if (
                        recovered is None
                        and line_index + 2 < len(lines)
                        and not next_record.text.strip()
                    ):
                        after_blank = lines[line_index + 2]
                        if after_blank.text.strip():
                            recovered = self._recover_two_line_heading(
                                record,
                                after_blank,
                                allow_blank_gap=True,
                            )
                            if recovered is not None:
                                consumed = 3

                if recovered is not None:
                    candidates.append(recovered)

                    if (
                        recovered.section_type == "references"
                        and recovered.confidence
                        >= self.config.strong_heading_threshold
                        and self.config.stop_subheading_detection_in_references
                    ):
                        references_detected = True

                    # The recovered candidate spans the physical records that
                    # produced it. Do not emit the individual fragments.
                    line_index += consumed
                    continue

                if candidate is not None:
                    candidates.append(candidate)

                    if (
                        candidate.section_type == "references"
                        and candidate.confidence
                        >= self.config.strong_heading_threshold
                        and self.config.stop_subheading_detection_in_references
                    ):
                        references_detected = True

                line_index += 1

        logger.debug(
            "Detected %d raw heading candidates.",
            len(candidates),
        )
        return candidates

    def _recover_two_line_heading(
        self,
        first: _LineRecord,
        second: _LineRecord,
        *,
        allow_blank_gap: bool = False,
    ) -> Optional[HeadingCandidate]:
        """
        Attempt a conservative two-line heading reconstruction.

        The method first tries a normal space-joined heading. If that is not
        structurally convincing, it tries a word-fragment join without a space.
        Word-fragment recovery is only enabled for strongly structural pairs.
        """
        first_text = first.text.strip()
        second_text = second.text.strip()

        if not first_text or not second_text:
            return None

        if len(first_text) > 120 or len(second_text) > 120:
            return None

        # A normal wrapped heading should be physically adjacent unless the
        # extractor has clearly produced an isolated heading fragment.
        if not allow_blank_gap:
            if first.next_blank or second.previous_blank:
                # This can still be a word-fragment pair if both lines are
                # isolated; permit that path below.
                if not self._looks_like_word_fragment_pair(first, second):
                    return None
        else:
            if not self._looks_like_word_fragment_pair(first, second):
                return None

        # ------------------------------------------------------------------
        # 1. Word-fragment reconstruction: EXAM + PLES -> EXAMPLES
        # ------------------------------------------------------------------
        if (
            self.config.detect_word_fragment_headings
            and self._looks_like_word_fragment_pair(first, second)
        ):
            joined = self._combine_heading_records(
                first,
                second,
                separator="",
            )
            if joined is not None:
                candidate = self._evaluate_line(
                    joined,
                    extra_score=self.config.word_fragment_bonus,
                    recovery_kind="word_fragment",
                )
                if candidate is not None and self._is_recovered_heading_confident(
                    candidate,
                    recovery_kind="word_fragment",
                ):
                    return candidate

        # ------------------------------------------------------------------
        # 2. Normal wrapped heading: "1 Proposed" + "Methodology"
        # ------------------------------------------------------------------
        if allow_blank_gap:
            return None

        if not self._looks_like_normal_wrapped_heading(first, second):
            return None

        combined = self._combine_heading_records(
            first,
            second,
            separator=" ",
        )
        if combined is None:
            return None

        combined_candidate = self._evaluate_line(combined)
        if combined_candidate is None:
            return None

        if (
            combined_candidate.is_numbered
            or combined_candidate.section_type not in {"unknown", "other"}
        ):
            return combined_candidate

        return None

    @staticmethod
    def _combine_heading_records(
        first: _LineRecord,
        second: _LineRecord,
        *,
        separator: str = " ",
    ) -> Optional[_LineRecord]:
        first_text = first.text.strip()
        second_text = second.text.strip()

        if not first_text or not second_text:
            return None

        if len(first_text) > 120 or len(second_text) > 120:
            return None

        combined_text = f"{first_text}{separator}{second_text}"
        if len(combined_text) > 180:
            return None

        return _LineRecord(
            page_number=first.page_number,
            text=combined_text,
            start_offset=first.start_offset,
            end_offset=second.end_offset,
            line_index=first.line_index,
            block_index=(
                first.block_index
                if first.block_index is not None
                else second.block_index
            ),
            previous_blank=first.previous_blank,
            next_blank=second.next_blank,
            page_char_count=first.page_char_count,
            page_line_count=first.page_line_count,
        )

    def _looks_like_normal_wrapped_heading(
        self,
        first: _LineRecord,
        second: _LineRecord,
    ) -> bool:
        """
        Return True only when the pair has structural heading evidence.

        A numbered first line plus a heading-like second line is the strongest
        common case. We also allow two isolated short lines when at least one
        line maps confidently to known section vocabulary.
        """
        first_text = first.text.strip()
        second_text = second.text.strip()

        if not first_text or not second_text:
            return False

        first_numbering, first_label, _ = self._parse_heading_numbering(first_text)
        second_numbering, second_label, _ = self._parse_heading_numbering(
            second_text
        )

        if second_numbering is not None:
            return False

        first_class = self._classify_heading(
            label=first_label,
            numbering=first_numbering,
            record=first,
        )
        second_class = self._classify_heading(
            label=second_label,
            numbering=second_numbering,
            record=second,
        )

        if first_numbering is not None:
            return (
                self._word_count(second_text) <= self.config.max_heading_words
                and not _SENTENCE_END_RE.search(second_text)
                and (
                    second_class.section_type != "unknown"
                    or self._capitalization_signal(second_label) >= 0.75
                )
            )

        if first.previous_blank and second.next_blank:
            return (
                first_class.section_type != "unknown"
                and second_class.section_type != "unknown"
            )

        return False

    def _looks_like_word_fragment_pair(
        self,
        first: _LineRecord,
        second: _LineRecord,
    ) -> bool:
        """
        Detect likely mid-word extraction fragments without using a dictionary.

        This deliberately favors high precision:
        - both fragments must be alphabetic,
        - both must look like isolated heading text,
        - both are short,
        - at least one fragment has a common word-ending pattern, or the
          concatenation is a known high-frequency scientific heading token.

        It is therefore not a generic line-joining mechanism.
        """
        if not self.config.detect_word_fragment_headings:
            return False

        first_text = first.text.strip()
        second_text = second.text.strip()

        if not first_text or not second_text:
            return False

        if len(first_text) < self.config.word_fragment_min_first_chars:
            return False

        if len(second_text) < self.config.word_fragment_min_second_chars:
            return False

        if len(second_text) > self.config.word_fragment_max_second_chars:
            return False

        # A fragment is the final word of the first physical line. The line
        # itself may be a long heading, e.g. "THEORETICAL ... EXAM".
        first_words = re.findall(
            r"[^\W\d_]+",
            first_text,
            flags=re.UNICODE,
        )
        second_words = re.findall(
            r"[^\W\d_]+",
            second_text,
            flags=re.UNICODE,
        )

        if not first_words or len(second_words) != 1:
            return False

        first_fragment = first_words[-1]
        second_fragment = second_words[0]

        if len(first_fragment) > self.config.word_fragment_max_chars:
            return False
        if len(first_fragment) < self.config.word_fragment_min_first_chars:
            return False
        if len(second_fragment) < self.config.word_fragment_min_second_chars:
            return False
        if len(second_fragment) > self.config.word_fragment_max_second_chars:
            return False

        # There must be no punctuation or numeric token at the split boundary.
        if not re.fullmatch(
            r"[^\W\d_]+",
            first_fragment,
            flags=re.UNICODE,
        ):
            return False
        if not re.fullmatch(
            r"[^\W\d_]+",
            second_fragment,
            flags=re.UNICODE,
        ):
            return False

        # Mid-word extraction normally preserves the typography of an ALL-CAPS
        # heading. Mixed-case normal prose is intentionally rejected.
        if not (
            self._capitalization_signal(first_fragment) >= 0.95
            and self._capitalization_signal(second_fragment) >= 0.95
        ):
            return False

        # The two records must look like separate heading fragments, not prose.
        if not (
            (first.previous_blank and first.next_blank)
            and (second.previous_blank and second.next_blank)
        ):
            return False

        combined = f"{first_fragment}{second_fragment}".lower()

        # Common scientific/technical words frequently broken by PDF extraction.
        known_fragment_words = {
            "examples",
            "methodology",
            "introduction",
            "conclusions",
            "acknowledgments",
            "acknowledgements",
            "experimental",
            "experiments",
            "implementation",
            "evaluation",
            "architecture",
            "preliminaries",
            "applications",
            "optimization",
            "classification",
            "generalization",
            "representation",
            "representations",
            "computation",
            "computational",
            "complexity",
            "information",
            "investigation",
            "organization",
            "organizations",
        }

        if combined in known_fragment_words:
            return True

        # Generic morphology fallback. This catches words such as
        # "METHODOLOGY" and "INTRODUCTION" without requiring a dictionary.
        common_tails = (
            "tion",
            "sion",
            "ment",
            "ness",
            "ity",
            "ies",
            "ing",
            "ed",
            "ology",
            "logy",
            "ical",
            "ance",
            "ence",
            "ation",
            "ations",
            "ization",
            "izations",
            "ative",
            "ously",
            "able",
            "ible",
            "ious",
            "ive",
            "ary",
            "ory",
            "ers",
            "ors",
            "ples",
        )

        return any(
            combined.endswith(tail)
            and len(combined) >= len(tail) + 3
            for tail in common_tails
        )

    @staticmethod
    def _is_recovered_heading_confident(
        candidate: HeadingCandidate,
        *,
        recovery_kind: str,
    ) -> bool:
        if recovery_kind != "word_fragment":
            return True

        # Word-fragment reconstruction must itself produce a structurally
        # plausible heading. Do not allow a reconstruction to turn arbitrary
        # uppercase prose into a section boundary.
        return (
            candidate.confidence >= 0.64
            and candidate.section_type != "unknown"
            or (
                candidate.confidence >= 0.72
                and candidate.section_type == "unknown"
            )
        )

    def _evaluate_line(
        self,
        record: _LineRecord,
        *,
        extra_score: float = 0.0,
        recovery_kind: Optional[str] = None,
    ) -> Optional[HeadingCandidate]:
        raw = record.text
        stripped = raw.strip()

        if not stripped:
            return None

        # Avoid treating a page's whole paragraph as a heading.
        if len(stripped) < self.config.min_heading_chars:
            return None

        if len(stripped) > self.config.max_heading_chars:
            return None

        word_count = self._word_count(stripped)

        if word_count > self.config.max_heading_words:
            return None

        caption_match = bool(
            _CAPTION_RE.match(stripped)
        )

        if caption_match:
            return None

        if self._looks_like_equation(stripped):
            return None

        if self._looks_like_list_item(stripped):
            return None

        numbering, label, numbering_depth = (
            self._parse_heading_numbering(
                stripped
            )
        )

        classification = self._classify_heading(
            label=label,
            numbering=numbering,
            record=record,
        )

        score, components = self._heading_score(
            stripped=stripped,
            label=label,
            numbering=numbering,
            classification=classification,
            record=record,
        )

        if extra_score:
            score = max(0.0, min(1.0, score + extra_score))
            components = dict(components)
            components[f"{recovery_kind or 'recovery'}_bonus"] = extra_score

        # Numbering is strong structural evidence even when the label is
        # domain-specific (for example "2.2.1 Training Strategy"). Allow a
        # slightly lower threshold for such headings, but keep the threshold
        # high enough to reject ordinary numbered prose/list items.
        effective_threshold = self.config.heading_accept_threshold

        if (
            numbering is not None
            and classification.section_type == "unknown"
        ):
            effective_threshold = max(
                0.58,
                self.config.heading_accept_threshold - 0.06,
            )

        if score < effective_threshold:
            return None

        # If an unnumbered heading is weak and not known vocabulary, reject it
        # rather than hallucinating structure.
        if (
            numbering is None
            and classification.section_type == "unknown"
            and self.config.require_known_vocab_for_weak_unnumbered
            and score < self.config.strong_heading_threshold
        ):
            # Classic technical reports often use unnumbered ALL-CAPS
            # headings. Accept only a strongly structural form: isolated,
            # short, uppercase, and not sentence-like.
            uppercase_structural = (
                self.config.uppercase_unknown_heading_accept
                and record.previous_blank
                and record.next_blank
                and self._capitalization_signal(label) >= 1.0
                and self._word_count(label) <= 10
                and not _SENTENCE_END_RE.search(stripped)
                and not stripped.endswith(":")
            )

            if not uppercase_structural:
                return None

        # ------------------------------------------------------------------
        # Structural fallback classification
        # ------------------------------------------------------------------
        # A heading can be structurally strong even when its semantic label is
        # outside our canonical vocabulary. In that case "unknown" would
        # incorrectly signal unresolved structure and would downgrade an
        # otherwise valid parse to "partial".
        #
        # We only apply this fallback when:
        #   * semantic classification is still unknown,
        #   * the candidate is structurally strong,
        #   * the heading is either a recovered word-fragment heading or a
        #     conservative uppercase/isolated heading.
        #
        # Importantly, classification_confidence remains low: "other" here is
        # a structural-preservation bucket, NOT a semantic claim.
        if (
            self.config.enable_unknown_heading_fallback
            and classification.section_type == "unknown"
            and score >= self.config.unknown_heading_fallback_threshold
            and (
                recovery_kind == "word_fragment"
                or (
                    record.previous_blank
                    and record.next_blank
                    and self._capitalization_signal(label) >= 1.0
                    and self._word_count(label) <= 10
                )
            )
        ):
            classification = _ClassifiedHeading(
                section_type="other",
                confidence=classification.confidence,
                ambiguous=False,
                warning=None,
            )

        normalized = self._normalize_heading_label(
            label
        )

        if not normalized:
            return None

        level = self._infer_level(
            numbering=numbering,
            numbering_depth=numbering_depth,
            record=record,
            label=label,
        )

        warning_list: list[str] = []

        if classification.ambiguous:
            warning_list.append(
                f"Ambiguous section classification for "
                f"heading {stripped!r} on page "
                f"{record.page_number}."
            )

        return HeadingCandidate(
            page_number=record.page_number,
            text=stripped,
            normalized_heading=normalized,
            section_type=classification.section_type,
            level=level,
            confidence=round(
                max(0.0, min(1.0, score)),
                6,
            ),
            classification_confidence=round(
                classification.confidence,
                6,
            ),
            start_offset=record.start_offset,
            end_offset=record.end_offset,
            line_index=record.line_index,
            block_index=record.block_index,
            numbering=numbering,
            numbering_depth=numbering_depth,
            score_components=dict(components),
            warnings=tuple(warning_list),
        )

    # ------------------------------------------------------------------
    # Heading classification
    # ------------------------------------------------------------------

    def _classify_heading(
        self,
        *,
        label: str,
        numbering: Optional[str],
        record: _LineRecord,
    ) -> _ClassifiedHeading:
        cleaned = _clean_alias(label)

        # Direct alias match.
        if cleaned in self._aliases:
            section_type = self._aliases[cleaned]

            if section_type == "other":
                # "Approach", "Framework", etc. are legitimate structural
                # headings but their scientific role is document-dependent.
                # Preserve them as "other" and explicitly mark the mapping
                # uncertainty rather than silently calling them methodology.
                if cleaned in {
                    "approach",
                    "framework",
                    "design",
                    "analysis",
                }:
                    return _ClassifiedHeading(
                        section_type="other",
                        confidence=0.58,
                        ambiguous=True,
                        warning=(
                            f"Heading {label!r} is structurally plausible "
                            "but cannot be mapped confidently to a "
                            "canonical section type."
                        ),
                    )

                return _ClassifiedHeading(
                    section_type="other",
                    confidence=0.74,
                    ambiguous=False,
                    warning=None,
                )

            if cleaned == "overview":
                # "Overview" is a valid structural section label. It is not
                # safe to reinterpret it as Introduction/Background, but using
                # the explicit "other" bucket does not constitute a parsing
                # ambiguity.
                return _ClassifiedHeading(
                    section_type="other",
                    confidence=0.74,
                    ambiguous=False,
                    warning=None,
                )

            return _ClassifiedHeading(
                section_type=section_type,
                confidence=0.98,
                ambiguous=False,
                warning=None,
            )

        # Appendix variants.
        appendix_match = _APPENDIX_RE.match(
            record.text.strip()
        )

        if appendix_match:
            return _ClassifiedHeading(
                section_type="appendix",
                confidence=0.96,
                ambiguous=False,
                warning=None,
            )

        # Conservative fuzzy matching only for headings that are close to a
        # known alias. This avoids a huge brittle synonym dictionary.
        best_type: Optional[str] = None
        best_ratio = 0.0
        second_ratio = 0.0

        for alias, section_type in self._aliases.items():
            if section_type in {
                "other",
                "unknown",
            }:
                continue

            ratio = SequenceMatcher(
                None,
                cleaned,
                alias,
            ).ratio()

            if ratio > best_ratio:
                second_ratio = best_ratio
                best_ratio = ratio
                best_type = section_type
            elif ratio > second_ratio:
                second_ratio = ratio

        if (
            best_type is not None
            and best_ratio >= 0.91
            and (best_ratio - second_ratio)
            >= self.config.ambiguity_margin
        ):
            return _ClassifiedHeading(
                section_type=best_type,
                confidence=min(
                    0.96,
                    0.78 + 0.18 * best_ratio,
                ),
                ambiguous=False,
                warning=None,
            )

        # Phrase-level semantic-ish aliases that are still deterministic.
        lowered = cleaned

        if (
            "results" in lowered
            and "discussion" in lowered
        ):
            return _ClassifiedHeading(
                section_type="results_discussion",
                confidence=0.90,
                ambiguous=False,
                warning=None,
            )

        if (
            "material" in lowered
            and "method" in lowered
        ):
            return _ClassifiedHeading(
                section_type="methodology",
                confidence=0.91,
                ambiguous=False,
                warning=None,
            )

        if (
            "experimental" in lowered
            and (
                "setup" in lowered
                or "setting" in lowered
                or "configuration" in lowered
            )
        ):
            return _ClassifiedHeading(
                section_type="experimental_setup",
                confidence=0.91,
                ambiguous=False,
                warning=None,
            )

        if (
            "future" in lowered
            and (
                "work" in lowered
                or "direction" in lowered
                or "research" in lowered
            )
        ):
            return _ClassifiedHeading(
                section_type="future_work",
                confidence=0.89,
                ambiguous=False,
                warning=None,
            )

        # Explicitly avoid forcing ambiguous headings such as "Approach".
        ambiguous_labels = {
            "approach",
            "framework",
            "design",
            "analysis",
            "method",
        }

        if lowered in ambiguous_labels:
            return _ClassifiedHeading(
                section_type="other",
                confidence=0.58,
                ambiguous=True,
                warning=(
                    f"Heading {label!r} is structurally plausible but "
                    "cannot be mapped confidently to a canonical "
                    "section type."
                ),
            )

        return _ClassifiedHeading(
            section_type="unknown",
            confidence=0.40,
            ambiguous=True,
            warning=(
                f"Heading {label!r} could not be mapped confidently "
                "to a canonical section type."
            ),
        )

    # ------------------------------------------------------------------
    # Heading score
    # ------------------------------------------------------------------

    def _heading_score(
        self,
        *,
        stripped: str,
        label: str,
        numbering: Optional[str],
        classification: _ClassifiedHeading,
        record: _LineRecord,
    ) -> tuple[float, Mapping[str, float]]:
        components: dict[str, float] = {}

        # Start from a neutral base. Scores are evidence aggregation, not
        # probabilities.
        score = 0.35

        if numbering:
            score += self.config.numbered_heading_bonus
            components["numbering"] = (
                self.config.numbered_heading_bonus
            )
        else:
            components["numbering"] = 0.0

        if (
            classification.section_type
            not in {"unknown"}
        ):
            score += self.config.known_heading_bonus
            components["known_heading"] = (
                self.config.known_heading_bonus
            )
        else:
            components["known_heading"] = 0.0

        isolated = (
            record.previous_blank
            and record.next_blank
        )

        if isolated:
            score += self.config.isolated_line_bonus
            components["isolated_line"] = (
                self.config.isolated_line_bonus
            )
        else:
            components["isolated_line"] = 0.0

        capitalization = self._capitalization_signal(
            label
        )

        capitalization_bonus = (
            self.config.capitalization_bonus
            * capitalization
        )

        score += capitalization_bonus
        components["capitalization"] = capitalization_bonus

        word_count = self._word_count(stripped)

        if 1 <= word_count <= 8:
            score += self.config.short_heading_bonus
            components["shortness"] = (
                self.config.short_heading_bonus
            )
        else:
            components["shortness"] = 0.0

        # Early-page headings are common for Abstract/Introduction, while
        # late-page headings can be Conclusion/References/Appendix. This is a
        # weak signal only.
        if record.page_line_count > 0:
            relative_line = (
                record.line_index
                / max(1, record.page_line_count - 1)
            )

            if relative_line <= 0.15 or relative_line >= 0.85:
                score += self.config.page_position_bonus
                components["page_position"] = (
                    self.config.page_position_bonus
                )
            else:
                components["page_position"] = 0.0

        # Body-like sentence penalty.
        if _SENTENCE_END_RE.search(stripped):
            score -= self.config.sentence_penalty
            components["sentence_penalty"] = (
                -self.config.sentence_penalty
            )
        else:
            components["sentence_penalty"] = 0.0

        # Colon at end is not automatically bad because Abstract:/Keywords:
        # are common. Penalize only non-known headings.
        if stripped.endswith(":"):
            if classification.section_type in {
                "unknown",
                "other",
            }:
                score -= self.config.punctuation_penalty
                components["punctuation_penalty"] = (
                    -self.config.punctuation_penalty
                )
            else:
                components["punctuation_penalty"] = 0.0
        else:
            components["punctuation_penalty"] = 0.0

        # Long body-like unnumbered text is dangerous.
        if (
            numbering is None
            and len(stripped)
            > self.config.max_body_like_heading_chars
        ):
            score -= self.config.body_phrase_penalty
            components["body_phrase_penalty"] = (
                -self.config.body_phrase_penalty
            )
        else:
            components["body_phrase_penalty"] = 0.0

        # If page text is extremely short, weak unnumbered candidates are
        # unreliable.
        if (
            numbering is None
            and record.page_char_count
            < self.config.minimum_page_chars_for_weak_heading
            and classification.section_type
            in {"unknown", "other"}
        ):
            score -= self.config.body_phrase_penalty
            components["sparse_page_penalty"] = (
                -self.config.body_phrase_penalty
            )
        else:
            components["sparse_page_penalty"] = 0.0

        return (
            max(0.0, min(1.0, score)),
            components,
        )

    # ------------------------------------------------------------------
    # Numbering / normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_heading_numbering(
        text: str,
    ) -> tuple[
        Optional[str],
        str,
        Optional[int],
    ]:
        match = _NUMERIC_RE.match(text)

        if match:
            number = match.group("num")
            label = match.group("label").strip()

            # Avoid interpreting list-like "1) Something" as section
            # numbering unless the surrounding structure later supports it.
            if ")" in text[: text.find(label)]:
                return (
                    None,
                    text.strip(),
                    None,
                )

            depth = number.count(".") + 1

            return number, label, depth

        match = _ROMAN_RE.match(text)

        if match:
            number = match.group("num").upper()
            label = match.group("label").strip()

            # A Roman numeral is only accepted when followed by a meaningful
            # label; single "I." is not a heading.
            if not label:
                return None, text.strip(), None

            return number, label, 1

        match = _LETTER_RE.match(text)

        if match:
            number = match.group("num")
            label = match.group("label").strip()

            return number, label, 1

        appendix_match = _APPENDIX_RE.match(text)

        if appendix_match:
            letter = appendix_match.group("label")
            rest = appendix_match.group("rest").strip()

            if letter:
                numbering = letter.upper()
                label = (
                    f"Appendix {letter.upper()}"
                    if not rest
                    else f"Appendix {letter.upper()} {rest}"
                )
            else:
                numbering = None
                label = (
                    "Appendix"
                    if not rest
                    else f"Appendix {rest}"
                )

            return numbering, label, 1

        return None, text.strip(), None

    @staticmethod
    def _normalize_heading_label(
        label: str,
    ) -> str:
        value = unicodedata.normalize(
            "NFC",
            label,
        ).strip()

        # Remove numbering again because some special forms place numbering
        # inside the label.
        value = re.sub(
            r"^(?:(?:\d+\.)+\d*|[IVXLCDM]+\.|[A-Z]\.)\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = value.strip(" \t\r\n:;-.")

        value = value.lower()

        # Keep Unicode letters/digits and meaningful punctuation temporarily.
        value = value.replace("&", " and ")

        # Canonical slug-like label for stable comparison/metadata.
        value = re.sub(
            r"[^\w\s-]",
            " ",
            value,
            flags=re.UNICODE,
        )

        value = re.sub(
            r"[\s_-]+",
            "_",
            value,
        )

        return value.strip("_")

    # ------------------------------------------------------------------
    # Heading-level inference
    # ------------------------------------------------------------------

    def _infer_level(
        self,
        *,
        numbering: Optional[str],
        numbering_depth: Optional[int],
        record: _LineRecord,
        label: str,
    ) -> int:
        if numbering_depth is not None:
            return max(
                1,
                min(
                    9,
                    numbering_depth,
                ),
            )

        # Roman and lettered headings are conventionally top-level in many
        # IEEE/ACM documents, so default to level 1.
        if numbering and (
            numbering.isalpha()
            or numbering.upper() == numbering
        ):
            return 1

        # Without numbering, use a conservative top-level level. Later
        # hierarchy reconstruction can infer a deeper level from consistency
        # with neighboring headings only when evidence exists.
        return 1

    # ------------------------------------------------------------------
    # Candidate postprocessing
    # ------------------------------------------------------------------

    def _postprocess_candidates(
        self,
        candidates: Sequence[HeadingCandidate],
    ) -> list[HeadingCandidate]:
        if not candidates:
            return []

        # Preserve document order.
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.page_number,
                candidate.start_offset,
            ),
        )

        # Remove exact duplicate candidates that can arise when the same line
        # is represented by multiple extractor blocks.
        deduped: list[HeadingCandidate] = []

        seen: set[tuple[int, int, str]] = set()

        for candidate in ordered:
            key = (
                candidate.page_number,
                candidate.start_offset,
                candidate.normalized_heading,
            )

            if key in seen:
                continue

            seen.add(key)
            deduped.append(candidate)

        # If a two-line recovery candidate spans a shorter candidate starting
        # on its second line, keep the recovered combined heading and suppress
        # the nested fragment. This prevents "2 Proposed" + "Methodology" from
        # becoming two sections after recovery.
        spanning = [
            candidate
            for candidate in deduped
            if candidate.end_offset > candidate.start_offset
        ]
        nested_suppressed: list[HeadingCandidate] = []
        for candidate in deduped:
            is_nested = any(
                other is not candidate
                and other.page_number == candidate.page_number
                and other.start_offset <= candidate.start_offset
                and other.end_offset >= candidate.end_offset
                and (
                    other.start_offset < candidate.start_offset
                    or other.end_offset > candidate.end_offset
                )
                and (
                    other.is_numbered
                    or other.section_type not in {"unknown", "other"}
                )
                for other in spanning
            )
            if not is_nested:
                nested_suppressed.append(candidate)

        deduped = nested_suppressed

        # Reject obvious duplicate repeated headers/footers only when the
        # exact heading appears repeatedly on many pages and has no strong
        # section semantics. We preserve known section headings.
        frequency: dict[str, int] = {}

        for candidate in deduped:
            frequency[
                candidate.normalized_heading
            ] = (
                frequency.get(
                    candidate.normalized_heading,
                    0,
                )
                + 1
            )

        final: list[HeadingCandidate] = []

        for candidate in deduped:
            repeated = (
                frequency[candidate.normalized_heading] >= 3
            )

            if (
                repeated
                and candidate.section_type
                in {"unknown", "other"}
                and not candidate.is_numbered
            ):
                logger.debug(
                    "Suppressing likely repeated "
                    "header/footer candidate: %r",
                    candidate.text,
                )
                continue

            final.append(candidate)

        # Enforce a terminal References boundary if confidence is high.
        references_indices = [
            index
            for index, candidate in enumerate(final)
            if (
                candidate.section_type == "references"
                and candidate.confidence
                >= self.config.heading_accept_threshold
            )
        ]

        if references_indices:
            first_reference = references_indices[0]
            prefix = final[: first_reference + 1]
            reference_candidate = final[first_reference]

            # No later heading should be allowed to split the References
            # section unless it is clearly an Appendix.
            suffix = [
                candidate
                for candidate in final[first_reference + 1 :]
                if candidate.section_type == "appendix"
                and candidate.confidence
                >= self.config.strong_heading_threshold
            ]

            final = prefix + suffix

            # Preserve reference candidate even if it was the only one.
            if reference_candidate not in final:
                final.insert(
                    first_reference,
                    reference_candidate,
                )

        logger.debug(
            "Postprocessed heading candidates: %d",
            len(final),
        )

        return final

    # ------------------------------------------------------------------
    # Title detection
    # ------------------------------------------------------------------

    def _detect_title(
        self,
        *,
        document: ExtractedDocumentLike,
        page_records: Sequence[Sequence[_LineRecord]],
        candidates: Sequence[HeadingCandidate],
    ) -> Optional[str]:
        """
        Detect a title conservatively.

        Priority:
        1. reliable PDF metadata title
        2. first-page candidate-like title before Abstract/Introduction
        3. otherwise None

        We never treat Abstract/Introduction as the title.
        """
        metadata = getattr(
            document,
            "metadata",
            {},
        ) or {}

        metadata_title = metadata.get("title")

        if isinstance(metadata_title, str):
            metadata_title = metadata_title.strip()

            if self._is_reliable_title(
                metadata_title
            ):
                return metadata_title

        if not page_records:
            return None

        first_page = page_records[0]

        # Find first major section. Candidate titles must occur before it.
        major_boundary_index: Optional[int] = None

        for index, candidate in enumerate(candidates):
            if candidate.page_number != 1:
                break

            if candidate.section_type in {
                "abstract",
                "keywords",
                "introduction",
            }:
                major_boundary_index = index
                break

        boundary_offset = None

        if major_boundary_index is not None:
            boundary_offset = candidates[
                major_boundary_index
            ].start_offset

        title_lines: list[str] = []

        for record in first_page:
            stripped = record.text.strip()

            if not stripped:
                if title_lines:
                    break
                continue

            if (
                boundary_offset is not None
                and record.start_offset >= boundary_offset
            ):
                break

            if _CAPTION_RE.match(stripped):
                continue

            if self._looks_like_equation(stripped):
                continue

            # Avoid selecting authors/affiliations by common structural
            # markers. This is intentionally conservative.
            lowered = stripped.lower()

            if (
                "@" in stripped
                or "university" in lowered
                or "institute" in lowered
                or "department" in lowered
                or "corresponding author" in lowered
            ):
                if title_lines:
                    break
                continue

            if (
                len(stripped) <= 150
                and self._word_count(stripped) <= 20
            ):
                title_lines.append(stripped)

            if len(title_lines) >= 4:
                break

        if not title_lines:
            return None

        candidate_title = " ".join(
            title_lines
        ).strip()

        if not self._is_reliable_title(
            candidate_title
        ):
            return None

        return candidate_title

    @staticmethod
    def _is_reliable_title(
        title: str,
    ) -> bool:
        normalized = title.strip().lower()

        if not normalized:
            return False

        if normalized in {
            "abstract",
            "introduction",
            "keywords",
            "index terms",
            "references",
            "bibliography",
        }:
            return False

        if len(title) < 4:
            return False

        if len(title) > 400:
            return False

        # PDF metadata is frequently polluted by source/build paths such as
        # "/nfs/.../techreport.dvi". These are not paper titles.
        if re.search(
            r"(?:^|[\\/])[^\\/\n]{1,120}\.(?:pdf|dvi|ps|tex|aux|log)$",
            title,
            flags=re.IGNORECASE,
        ):
            return False

        if re.match(
            r"^(?:[A-Za-z]:[\\/]|/|\\\\)",
            title,
        ):
            return False

        if _CAPTION_RE.match(title):
            return False

        if _SENTENCE_END_RE.search(title):
            # Titles can end in punctuation, but a full sentence is more
            # likely body text.
            return False

        return True

    # ------------------------------------------------------------------
    # Section construction
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        *,
        document: ExtractedDocumentLike,
        page_records: Sequence[Sequence[_LineRecord]],
        candidates: Sequence[HeadingCandidate],
        excluded_pages: Optional[set[int]] = None,
    ) -> list[PaperSection]:
        pages_text = [
            self._normalize_page_structure_text(page.text)
            for page in document.pages
        ]
        excluded_pages = excluded_pages or set()

        if not candidates:
            if not self.config.preserve_unsectioned_document:
                return []

            text = "\n\n".join(
                page.strip("\n")
                for page in pages_text
                if page.strip()
            ).strip()

            if not text:
                return []

            return [
                PaperSection(
                    section_id=self._make_fallback_section_id(
                        document.document_id,
                        "document_content",
                    ),
                    document_id=document.document_id,
                    section_type="other",
                    original_heading=None,
                    normalized_heading="document_content",
                    level=1,
                    text=text,
                    start_page=1,
                    end_page=document.page_count,
                    source_pages=tuple(range(1, document.page_count + 1)),
                    parent_section_id=None,
                    child_section_ids=(),
                    confidence=self.config.fallback_section_confidence,
                )
            ]

        sections: list[PaperSection] = []

        # Preserve title/author/front-matter text before the first detected
        # heading instead of silently dropping it.
        first_candidate = candidates[0]
        if self.config.preserve_preamble:
            preamble_text, preamble_pages = self._slice_preamble(
                pages_text=pages_text,
                first_candidate=first_candidate,
                excluded_pages=excluded_pages,
            )
            if preamble_text.strip() and preamble_pages:
                sections.append(
                    PaperSection(
                        section_id=self._make_fallback_section_id(
                            document.document_id,
                            "front_matter",
                        ),
                        document_id=document.document_id,
                        section_type="other",
                        original_heading=None,
                        normalized_heading="front_matter",
                        level=1,
                        text=preamble_text,
                        start_page=preamble_pages[0],
                        end_page=preamble_pages[-1],
                        source_pages=tuple(preamble_pages),
                        parent_section_id=None,
                        child_section_ids=(),
                        confidence=self.config.fallback_section_confidence,
                    )
                )

        section_offset = len(sections)

        for index, candidate in enumerate(candidates):
            next_candidate = (
                candidates[index + 1]
                if index + 1 < len(candidates)
                else None
            )

            section_text, source_pages = self._slice_section_text(
                pages_text=pages_text,
                start_candidate=candidate,
                end_candidate=next_candidate,
            )

            section_id = self._make_section_id(
                document_id=document.document_id,
                section_index=index + section_offset,
                candidate=candidate,
            )

            sections.append(
                PaperSection(
                    section_id=section_id,
                    document_id=document.document_id,
                    section_type=candidate.section_type,
                    original_heading=candidate.text,
                    normalized_heading=candidate.normalized_heading,
                    level=max(1, min(9, candidate.level)),
                    text=section_text,
                    start_page=(
                        source_pages[0]
                        if source_pages
                        else candidate.page_number
                    ),
                    end_page=(
                        source_pages[-1]
                        if source_pages
                        else candidate.page_number
                    ),
                    source_pages=tuple(source_pages),
                    parent_section_id=None,
                    child_section_ids=(),
                    confidence=candidate.confidence,
                )
            )

        return sections

    def _slice_preamble(
        self,
        *,
        pages_text: Sequence[str],
        first_candidate: HeadingCandidate,
        excluded_pages: Optional[set[int]] = None,
    ) -> tuple[str, list[int]]:
        end_page_index = first_candidate.page_number - 1
        excluded_pages = excluded_pages or set()

        if end_page_index < 0 or end_page_index >= len(pages_text):
            return "", []

        pieces: list[str] = []

        for page_index in range(end_page_index):
            page_number = page_index + 1
            if page_number in excluded_pages:
                continue

            page_text = pages_text[page_index]
            if page_text.strip():
                pieces.append(page_text)

        prefix_end = max(
            0,
            min(
                first_candidate.start_offset,
                len(pages_text[end_page_index]),
            ),
        )
        prefix = pages_text[end_page_index][:prefix_end]
        if prefix.strip():
            pieces.append(prefix)

        text = "\n\n".join(
            piece.strip("\n")
            for piece in pieces
        ).strip()

        pages = [
            index + 1
            for index in range(end_page_index + 1)
            if index + 1 not in excluded_pages
            and pages_text[index].strip()
        ]
        return text, pages

    def _slice_section_text(
        self,
        *,
        pages_text: Sequence[str],
        start_candidate: HeadingCandidate,
        end_candidate: Optional[HeadingCandidate],
    ) -> tuple[str, list[int]]:
        """
        Slice section content without re-reading the PDF.

        The heading itself is excluded from section text because it is already
        preserved as original_heading. All following source text remains
        untouched except for structural page-boundary handling.
        """
        start_page_index = (
            start_candidate.page_number - 1
        )

        if start_page_index < 0 or start_page_index >= len(pages_text):
            return "", []

        # Start immediately after the heading line.
        first_page_text = pages_text[
            start_page_index
        ]

        start_offset = min(
            max(
                start_candidate.end_offset,
                0,
            ),
            len(first_page_text),
        )

        if end_candidate is None:
            end_page_index = len(pages_text) - 1
            end_offset = len(
                pages_text[end_page_index]
            )
        else:
            end_page_index = (
                end_candidate.page_number - 1
            )

            if end_page_index < 0 or end_page_index >= len(
                pages_text
            ):
                end_page_index = len(pages_text) - 1
                end_offset = len(
                    pages_text[end_page_index]
                )
            else:
                end_offset = min(
                    max(
                        end_candidate.start_offset,
                        0,
                    ),
                    len(
                        pages_text[end_page_index]
                    ),
                )

        if end_page_index < start_page_index:
            return "", []

        pieces: list[str] = []

        if start_page_index == end_page_index:
            pieces.append(
                first_page_text[start_offset:end_offset]
            )
        else:
            pieces.append(
                first_page_text[start_offset:]
            )

            for page_index in range(
                start_page_index + 1,
                end_page_index,
            ):
                pieces.append(
                    pages_text[page_index]
                )

            pieces.append(
                pages_text[end_page_index][:end_offset]
            )

        # Page separators are parser-generated structural delimiters. The
        # original page text itself remains otherwise untouched.
        text = "\n\n".join(
            piece.strip("\n")
            for piece in pieces
        ).strip()

        source_pages = list(
            range(
                start_candidate.page_number,
                end_page_index + 2,
            )
        )

        return text, source_pages

    @staticmethod
    def _make_section_id(
        *,
        document_id: str,
        section_index: int,
        candidate: HeadingCandidate,
    ) -> str:
        """
        Deterministic, collision-resistant ID.

        It incorporates the source document ID, structural index, page and
        heading text. It intentionally does not use random UUIDs.
        """
        material = (
            f"{document_id}|"
            f"{section_index}|"
            f"{candidate.page_number}|"
            f"{candidate.start_offset}|"
            f"{candidate.text}"
        )

        digest = hashlib.sha256(
            material.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()[:20]

        return (
            f"sec_{section_index:04d}_{digest}"
        )

    @staticmethod
    def _make_fallback_section_id(
        document_id: str,
        label: str,
    ) -> str:
        material = f"{document_id}|fallback|{label}"
        digest = hashlib.sha256(
            material.encode("utf-8", errors="replace")
        ).hexdigest()[:20]
        return f"sec_fallback_{label}_{digest}"

    # ------------------------------------------------------------------
    # Hierarchy reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_hierarchy(
        self,
        sections: Sequence[PaperSection],
    ) -> list[PaperSection]:
        if not sections:
            return []

        # First pass: make levels conservative and monotonically sensible.
        adjusted: list[PaperSection] = []

        previous_level = 1

        for index, section in enumerate(sections):
            level = section.level

            if index == 0:
                level = 1
            else:
                # Do not allow a section to jump from level 1 directly to
                # level 4 without a structural signal.
                if level > previous_level + 1:
                    level = previous_level + 1

                # If a top-level heading appears, preserve it.
                if section.level == 1:
                    level = 1

            previous_level = level

            adjusted.append(
                PaperSection(
                    section_id=section.section_id,
                    document_id=section.document_id,
                    section_type=section.section_type,
                    original_heading=section.original_heading,
                    normalized_heading=section.normalized_heading,
                    level=level,
                    text=section.text,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    source_pages=section.source_pages,
                    parent_section_id=None,
                    child_section_ids=(),
                    confidence=section.confidence,
                )
            )

        # Second pass: stack-based parent assignment.
        stack: list[PaperSection] = []
        parent_by_id: dict[str, Optional[str]] = {}
        children_by_id: dict[str, list[str]] = {
            section.section_id: []
            for section in adjusted
        }

        for section in adjusted:
            while stack and stack[-1].level >= section.level:
                stack.pop()

            parent_id = (
                stack[-1].section_id
                if stack
                else None
            )

            parent_by_id[
                section.section_id
            ] = parent_id

            if parent_id is not None:
                children_by_id[
                    parent_id
                ].append(section.section_id)

            stack.append(section)

        result: list[PaperSection] = []

        for section in adjusted:
            result.append(
                PaperSection(
                    section_id=section.section_id,
                    document_id=section.document_id,
                    section_type=section.section_type,
                    original_heading=section.original_heading,
                    normalized_heading=section.normalized_heading,
                    level=section.level,
                    text=section.text,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    source_pages=section.source_pages,
                    parent_section_id=parent_by_id[
                        section.section_id
                    ],
                    child_section_ids=tuple(
                        children_by_id[
                            section.section_id
                        ]
                    ),
                    confidence=section.confidence,
                )
            )

        logger.debug(
            "Section hierarchy reconstructed: %d sections.",
            len(result),
        )

        return result

    # ------------------------------------------------------------------
    # Status / warnings
    # ------------------------------------------------------------------

    def _determine_status(
        self,
        *,
        document: ExtractedDocumentLike,
        candidates: Sequence[HeadingCandidate],
        sections: Sequence[PaperSection],
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        if not candidates:
            warnings.append(
                "No reliable section headings were detected; "
                "document content was preserved as a fallback section."
            )

        page_statuses = [
            getattr(page, "extraction_status", None)
            for page in getattr(document, "pages", ())
        ]
        if any(
            status in {"partial", "ocr_required", "failed"}
            for status in page_statuses
            if status is not None
        ):
            warnings.append(
                "One or more pages have incomplete or non-native extraction status."
            )

        ambiguous_count = sum(
            1
            for candidate in candidates
            if candidate.warnings
        )

        if ambiguous_count:
            warnings.append(
                f"{ambiguous_count} heading candidate(s) "
                "have classification ambiguity."
            )

        if candidates:
            levels = [
                candidate.level
                for candidate in candidates
            ]

            if max(levels) > 1 and not any(
                candidate.numbering_depth
                and candidate.numbering_depth > 1
                for candidate in candidates
            ):
                warnings.append(
                    "Heading hierarchy was inferred without "
                    "explicit numbering for some sections."
                )

        extraction_status = getattr(
            document,
            "extraction_status",
            "success",
        )

        if extraction_status == "failed":
            return "failed", warnings

        if not sections:
            return "partial", warnings

        if (
            extraction_status in {
                "partial",
                "ocr_required",
            }
            or ambiguous_count > 0
            or not candidates
        ):
            return "partial", warnings

        return "success", warnings

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        *,
        result: StructuredResearchPaper,
        document: ExtractedDocumentLike,
    ) -> None:
        if result.document_id != document.document_id:
            raise SectionOutputValidationError(
                "Output document_id does not match input document_id."
            )

        ids = [
            section.section_id
            for section in result.sections
        ]

        if len(ids) != len(set(ids)):
            raise SectionOutputValidationError(
                "Section IDs are not unique."
            )

        valid_page_numbers = set(
            range(
                1,
                document.page_count + 1,
            )
        )

        section_positions = {
            section.section_id: index
            for index, section in enumerate(
                result.sections
            )
        }

        for index, section in enumerate(
            result.sections
        ):
            if index > 0:
                previous = result.sections[
                    index - 1
                ]

                if (
                    section.start_page is not None
                    and previous.start_page is not None
                    and section.start_page
                    < previous.start_page
                ):
                    raise SectionOutputValidationError(
                        "Section order violates page order."
                    )

            for page in section.source_pages:
                if page not in valid_page_numbers:
                    raise SectionOutputValidationError(
                        f"Invalid source page {page}."
                    )

            if section.start_page is not None:
                if section.start_page not in valid_page_numbers:
                    raise SectionOutputValidationError(
                        "Invalid section start_page."
                    )

            if section.end_page is not None:
                if section.end_page not in valid_page_numbers:
                    raise SectionOutputValidationError(
                        "Invalid section end_page."
                    )

            if (
                section.parent_section_id
                == section.section_id
            ):
                raise SectionOutputValidationError(
                    "Section cannot be its own parent."
                )

            if section.parent_section_id is not None:
                if (
                    section.parent_section_id
                    not in section_positions
                ):
                    raise SectionOutputValidationError(
                        "Parent section ID does not exist."
                    )

                if (
                    section_positions[
                        section.parent_section_id
                    ]
                    >= index
                ):
                    raise SectionOutputValidationError(
                        "Parent section must occur before child."
                    )

                parent = result.sections[
                    section_positions[
                        section.parent_section_id
                    ]
                ]

                if section.level != parent.level + 1:
                    raise SectionOutputValidationError(
                        "Child level must be exactly parent level + 1."
                    )

            for child_id in section.child_section_ids:
                if child_id not in section_positions:
                    raise SectionOutputValidationError(
                        "Child section ID does not exist."
                    )

                child = result.sections[
                    section_positions[child_id]
                ]

                if child.parent_section_id != section.section_id:
                    raise SectionOutputValidationError(
                        "Child/parent relationship is inconsistent."
                    )

                if child.level != section.level + 1:
                    raise SectionOutputValidationError(
                        "Child level is inconsistent."
                    )

        self._validate_hierarchy_acyclic(
            result.sections
        )

        if result.section_count != len(
            result.sections
        ):
            raise SectionOutputValidationError(
                "section_count is inconsistent."
            )

        if result.parsing_status not in VALID_PARSING_STATUSES:
            raise SectionOutputValidationError(
                "Invalid parsing status."
            )

    @staticmethod
    def _validate_hierarchy_acyclic(
        sections: Sequence[PaperSection],
    ) -> None:
        parent_map = {
            section.section_id:
                section.parent_section_id
            for section in sections
        }

        for section in sections:
            seen: set[str] = set()
            current = section.section_id

            while current is not None:
                if current in seen:
                    raise SectionOutputValidationError(
                        "Section hierarchy contains a cycle."
                    )

                seen.add(current)
                current = parent_map.get(
                    current
                )

    # ------------------------------------------------------------------
    # Utility heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        return len(
            re.findall(
                r"[\w]+(?:[-'][\w]+)*",
                text,
                flags=re.UNICODE,
            )
        )

    @staticmethod
    def _capitalization_signal(
        text: str,
    ) -> float:
        letters = [
            char
            for char in text
            if char.isalpha()
        ]

        if not letters:
            return 0.0

        uppercase = sum(
            char.isupper()
            for char in letters
        )

        ratio = uppercase / len(letters)

        if ratio >= 0.85:
            return 1.0

        if ratio >= 0.60:
            return 0.75

        if text[:1].isupper():
            return 0.40

        return 0.0

    @staticmethod
    def _looks_like_equation(
        text: str,
    ) -> bool:
        stripped = text.strip()

        # Do not reject ordinary headings containing a slash or hyphen.
        if (
            len(stripped) < 3
            or len(stripped.split()) > 12
        ):
            return False

        # Strong mathematical cues.
        math_symbols = sum(
            stripped.count(symbol)
            for symbol in (
                "∑",
                "∫",
                "√",
                "≤",
                "≥",
                "≈",
                "≠",
                "→",
                "←",
                "↔",
                "±",
                "×",
                "÷",
            )
        )

        if math_symbols >= 2:
            return True

        # Variable = expression lines are usually equations.
        if (
            "=" in stripped
            and len(stripped) <= 100
            and not re.match(
                r"^(?:results?|discussion|method|approach)\b",
                stripped,
                flags=re.IGNORECASE,
            )
        ):
            return True

        return False

    @staticmethod
    def _looks_like_list_item(
        text: str,
    ) -> bool:
        return bool(
            _LIST_RE.match(text)
        )

    @staticmethod
    def _dedupe_warnings(
        warnings: Iterable[str],
    ) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for warning in warnings:
            value = str(warning).strip()

            if not value or value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result


# ============================================================================
# Convenience API
# ============================================================================


def parse_sections(
    document: ExtractedDocumentLike,
    *,
    config: Optional[SectionParserConfig] = None,
) -> StructuredResearchPaper:
    """
    Functional convenience wrapper.

    Example
    -------
    paper = parse_sections(extracted_document)
    """
    return SectionParser(
        config=config
    ).parse(document)


# ============================================================================
# Test fixtures and self-test
# ============================================================================


@dataclass(frozen=True)
class _MockPage:
    page_number: int
    text: str
    char_count: int
    word_count: int
    extraction_status: str = "success"
    blocks: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _MockDocument:
    document_id: str
    filename: str
    page_count: int
    pages: tuple[_MockPage, ...]
    full_text: str
    metadata: Mapping[str, Any]
    extraction_status: str = "success"


def _mock_document(
    pages: Sequence[str],
    *,
    document_id: str = (
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ),
    metadata: Optional[Mapping[str, Any]] = None,
    extraction_status: str = "success",
) -> _MockDocument:
    mock_pages: list[_MockPage] = []

    for index, text in enumerate(
        pages,
        start=1,
    ):
        mock_pages.append(
            _MockPage(
                page_number=index,
                text=text,
                char_count=len(text),
                word_count=len(
                    re.findall(
                        r"[\w]+",
                        text,
                    )
                ),
                extraction_status="success",
            )
        )

    return _MockDocument(
        document_id=document_id,
        filename="test.pdf",
        page_count=len(mock_pages),
        pages=tuple(mock_pages),
        full_text="\n\n".join(pages),
        metadata=metadata or {},
        extraction_status=extraction_status,
    )


def run_self_test() -> None:
    """
    Model-free structural tests.

    No PDF file, LLM, GPU, FAISS, SPECTER2, OCR engine or internet is used.
    """
    parser = SectionParser()

    # ------------------------------------------------------------------
    # 1. Standard research paper
    # ------------------------------------------------------------------
    document = _mock_document(
        [
            (
                "A Robust Vision Transformer for Scientific Image Analysis\n"
                "John Doe, Example University\n\n"
                "Abstract\n"
                "We present a model for image analysis. "
                "The complete extracted abstract is preserved.\n\n"
                "Keywords:\n"
                "vision transformer; scientific imaging; deep learning\n\n"
                "1 Introduction\n"
                "This introduction explains the research problem.\n\n"
                "2 Methodology\n"
                "The methodology describes the experimental process.\n"
                "2.1 Dataset\n"
                "The dataset contains 2000 samples.\n"
                "2.2 Model\n"
                "The model uses a transformer architecture.\n\n"
            ),
            (
                "3 Results\n"
                "The results are reported without interpretation here.\n\n"
                "4 Discussion\n"
                "The discussion text remains source-preserved.\n\n"
                "5 Conclusion\n"
                "The conclusion is preserved exactly.\n\n"
                "References\n"
                "[1] Example reference.\n"
                "[2] Another reference.\n"
            ),
        ],
        metadata={
            "title": (
                "A Robust Vision Transformer for "
                "Scientific Image Analysis"
            )
        },
    )

    result = parser.parse(document)

    assert result.parsing_status == "success"
    assert result.title == (
        "A Robust Vision Transformer for "
        "Scientific Image Analysis"
    )

    types = [
        section.section_type
        for section in result.sections
    ]

    assert "abstract" in types
    assert "keywords" in types
    assert "introduction" in types
    assert "methodology" in types
    assert "dataset" in types
    assert "model" in types
    assert "results" in types
    assert "discussion" in types
    assert "conclusion" in types
    assert "references" in types

    # ------------------------------------------------------------------
    # 2. IEEE Roman numerals
    # ------------------------------------------------------------------
    ieee = _mock_document(
        [
            (
                "A Study of Scientific Retrieval\n\n"
                "I. INTRODUCTION\n"
                "Introduction body.\n\n"
                "II. RELATED WORK\n"
                "Related work body.\n\n"
                "III. METHODOLOGY\n"
                "Method body.\n\n"
                "IV. EXPERIMENTAL RESULTS\n"
                "Result body.\n\n"
                "V. CONCLUSION\n"
                "Conclusion body."
            )
        ]
    )

    ieee_result = parser.parse(ieee)

    assert [
        s.section_type
        for s in ieee_result.sections
    ] == [
        "other",
        "introduction",
        "related_work",
        "methodology",
        "results",
        "conclusion",
    ]

    assert all(
        section.level == 1
        for section in ieee_result.sections
    )

    # ------------------------------------------------------------------
    # 3. Decimal hierarchy
    # ------------------------------------------------------------------
    hierarchy = _mock_document(
        [
            (
                "2 Methodology\n"
                "Main method.\n\n"
                "2.1 Dataset\n"
                "Dataset content.\n\n"
                "2.2 Model\n"
                "Model content.\n\n"
                "2.2.1 Training\n"
                "Training content.\n\n"
                "3 Results\n"
                "Results content."
            )
        ]
    )

    hierarchy_result = parser.parse(hierarchy)

    method = hierarchy_result.sections[0]
    dataset = hierarchy_result.sections[1]
    model = hierarchy_result.sections[2]
    training = hierarchy_result.sections[3]
    results = hierarchy_result.sections[4]

    assert method.level == 1
    assert dataset.level == 2
    assert model.level == 2
    assert training.level == 3
    assert results.level == 1

    assert dataset.parent_section_id == method.section_id
    assert model.parent_section_id == method.section_id
    assert training.parent_section_id == model.section_id

    assert dataset.section_id in method.child_section_ids
    assert model.section_id in method.child_section_ids
    assert training.section_id in model.child_section_ids

    # ------------------------------------------------------------------
    # 4. Lettered headings
    # ------------------------------------------------------------------
    lettered = _mock_document(
        [
            (
                "A. INTRODUCTION\n"
                "Body.\n\n"
                "B. PROPOSED METHOD\n"
                "Method body.\n\n"
                "C. EXPERIMENTS\n"
                "Experiment body."
            )
        ]
    )

    lettered_result = parser.parse(lettered)

    assert [
        s.section_type
        for s in lettered_result.sections
    ] == [
        "introduction",
        "methodology",
        "experiments",
    ]

    # ------------------------------------------------------------------
    # 5. Unnumbered headings
    # ------------------------------------------------------------------
    unnumbered = _mock_document(
        [
            (
                "Abstract\n"
                "Abstract body.\n\n"
                "Introduction\n"
                "Introduction body.\n\n"
                "Results\n"
                "Results body.\n\n"
                "Conclusion\n"
                "Conclusion body."
            )
        ]
    )

    unnumbered_result = parser.parse(unnumbered)

    assert [
        s.section_type
        for s in unnumbered_result.sections
    ] == [
        "abstract",
        "introduction",
        "results",
        "conclusion",
    ]

    # ------------------------------------------------------------------
    # 6. False heading prevention
    # ------------------------------------------------------------------
    false_heading = _mock_document(
        [
            (
                "Introduction\n"
                "Deep learning models can achieve strong "
                "performance on difficult datasets.\n"
                "This sentence must remain body text.\n\n"
                "Methodology\n"
                "The method follows a deterministic procedure."
            )
        ]
    )

    false_result = parser.parse(
        false_heading
    )

    assert not any(
        section.normalized_heading
        == "deep_learning_models_can_achieve_strong_performance"
        for section in false_result.sections
    )

    # ------------------------------------------------------------------
    # 7. Figure/table/equation/list false positives
    # ------------------------------------------------------------------
    captions = _mock_document(
        [
            (
                "Results\n"
                "Figure 1: System architecture\n"
                "Table 1: Dataset statistics\n"
                "Algorithm 1: Training procedure\n"
                "x = y + 1\n"
                "- first contribution\n"
                "Actual result text."
            )
        ]
    )

    caption_result = parser.parse(captions)

    assert [
        section.section_type
        for section in caption_result.sections
    ] == ["results"]

    # ------------------------------------------------------------------
    # 8. Ambiguous heading
    # ------------------------------------------------------------------
    ambiguous = _mock_document(
        [
            (
                "Approach\n"
                "The approach is described here with enough surrounding "
                "content to make this a realistic paper page rather than "
                "a tiny synthetic document.\n\n"
                "Results\n"
                "Results text with additional experimental context."
            )
        ]
    )

    ambiguous_result = parser.parse(ambiguous)

    approach_sections = [
        section
        for section in ambiguous_result.sections
        if section.normalized_heading == "approach"
    ]

    # "Approach" is allowed only as conservative "other" when accepted;
    # it must never be silently forced to methodology.
    if approach_sections:
        assert approach_sections[0].section_type == "other"

    assert ambiguous_result.parsing_status == "partial"
    assert any(
        "ambiguity" in warning.lower()
        for warning in ambiguous_result.warnings
    )

    # ------------------------------------------------------------------
    # 9. Multi-page mapping
    # ------------------------------------------------------------------
    multipage = _mock_document(
        [
            (
                "1 Introduction\n"
                "Introduction starts on page one.\n"
                "More introduction."
            ),
            (
                "Continuation of introduction.\n"
                "Still introduction."
            ),
            (
                "2 Methodology\n"
                "Methodology starts on page three."
            ),
        ]
    )

    multipage_result = parser.parse(multipage)

    introduction = multipage_result.sections[0]

    assert introduction.start_page == 1
    assert introduction.end_page == 3
    assert introduction.source_pages == (
        1,
        2,
        3,
    )

    # ------------------------------------------------------------------
    # 10. References are terminal and reference entries are not sections.
    # ------------------------------------------------------------------
    references = _mock_document(
        [
            (
                "Conclusion\n"
                "Conclusion body.\n\n"
                "References\n"
                "[1] First paper.\n"
                "[2] Second paper.\n"
                "[3] Third paper."
            )
        ]
    )

    references_result = parser.parse(
        references
    )

    reference_sections = [
        section
        for section in references_result.sections
        if section.section_type == "references"
    ]

    assert len(reference_sections) == 1
    assert "[1] First paper." in reference_sections[0].text
    assert "[2] Second paper." in reference_sections[0].text
    assert not any(
        section.normalized_heading
        in {"1", "2", "3"}
        for section in references_result.sections
    )

    # ------------------------------------------------------------------
    # 11. Appendix
    # ------------------------------------------------------------------
    appendix = _mock_document(
        [
            (
                "Conclusion\n"
                "Conclusion.\n\n"
                "References\n"
                "[1] Reference.\n\n"
                "Appendix A\n"
                "Additional material."
            )
        ]
    )

    appendix_result = parser.parse(
        appendix
    )

    assert appendix_result.sections[-1].section_type == (
        "appendix"
    )

    # ------------------------------------------------------------------
    # 12. Table-of-contents protection
    # ------------------------------------------------------------------
    toc_document = _mock_document(
        [
            (
                "Machine Learning\n\n"
                "Thomas G Dietterich\n"
                "Department of Computer Science"
            ),
            (
                "Contents\n\n"
                "OVERVIEW\n"
                "PHILOSOPHICAL FOUNDATIONS\n"
                "THEORETICAL RESULTS ON LEARNING FROM EXAMPLES\n"
                "CONCLUDING REMARKS\n"
                "ACKNOWLEDGMENTS\n"
                "BIBLIOGRAPHY"
            ),
            (
                "OVERVIEW\n"
                "The overview body contains enough prose to represent "
                "the actual section.\n\n"
                "PHILOSOPHICAL FOUNDATIONS\n"
                "The foundations body follows the overview."
            ),
            (
                "CONCLUDING REMARKS\n"
                "Conclusion body.\n\n"
                "BIBLIOGRAPHY\n"
                "[1] A reference entry."
            ),
        ],
        metadata={
            "title": "/nfs/tesla/u6/tgd/papers/arcs/techreport.dvi"
        },
    )

    toc_result = parser.parse(toc_document)

    assert toc_result.title == "Machine Learning"
    assert not any(
        heading.page_number == 2
        for heading in toc_result.detected_headings
    )
    assert any(
        section.normalized_heading == "overview"
        and section.start_page == 3
        for section in toc_result.sections
    )
    assert any(
        section.section_type == "references"
        and section.start_page == 4
        for section in toc_result.sections
    )
    assert all(
        2 not in section.source_pages
        for section in toc_result.sections
        if section.normalized_heading != "front_matter"
    )

    # ------------------------------------------------------------------
    # 13. No-heading document
    # ------------------------------------------------------------------
    no_headings = _mock_document(
        [
            (
                "This is body text without reliable headings. "
                "It describes a research experiment in several "
                "sentences and should not be invented into sections."
            )
        ]
    )

    no_heading_result = parser.parse(
        no_headings
    )

    assert len(no_heading_result.sections) == 1
    assert no_heading_result.sections[0].section_type == "other"
    assert no_heading_result.sections[0].normalized_heading == "document_content"
    assert no_heading_result.parsing_status == "partial"
    assert any(
        "No reliable section headings were detected" in warning
        for warning in no_heading_result.warnings
    )

    # ------------------------------------------------------------------
    # 13. Empty section
    # ------------------------------------------------------------------
    empty_section = _mock_document(
        [
            (
                "Abstract\n"
                "\n"
                "Introduction\n"
                "Introduction body."
            )
        ]
    )

    empty_result = parser.parse(
        empty_section
    )

    assert empty_result.sections[0].section_type == (
        "abstract"
    )
    assert empty_result.sections[0].text == ""

    # ------------------------------------------------------------------
    # 14. Determinism
    # ------------------------------------------------------------------
    result_again = parser.parse(
        hierarchy
    )

    assert [
        (
            s.section_id,
            s.section_type,
            s.level,
            s.start_page,
            s.end_page,
            s.text,
        )
        for s in hierarchy_result.sections
    ] == [
        (
            s.section_id,
            s.section_type,
            s.level,
            s.start_page,
            s.end_page,
            s.text,
        )
        for s in result_again.sections
    ]

    # ------------------------------------------------------------------
    # 15. Unicode heading
    # ------------------------------------------------------------------
    unicode_doc = _mock_document(
        [
            (
                "3. Méthodologie\n"
                "Scientific text.\n\n"
                "4. Résultats\n"
                "Result text."
            )
        ]
    )

    unicode_result = parser.parse(
        unicode_doc
    )

    assert unicode_result.sections[0].original_heading == (
        "3. Méthodologie"
    )
    assert unicode_result.sections[1].original_heading == (
        "4. Résultats"
    )

    # ------------------------------------------------------------------
    # 16. Preamble preservation
    # ------------------------------------------------------------------
    preamble_doc = _mock_document(
        [
            (
                "A Strong Research Paper Title\n"
                "Author Name, Example University\n\n"
                "Abstract\n"
                "Abstract text."
            )
        ]
    )
    preamble_result = parser.parse(preamble_doc)
    assert preamble_result.sections[0].normalized_heading == "front_matter"
    assert "A Strong Research Paper Title" in preamble_result.sections[0].text
    assert preamble_result.sections[1].section_type == "abstract"

    # ------------------------------------------------------------------
    # 17. Two-line heading recovery
    # ------------------------------------------------------------------
    wrapped_heading = _mock_document(
        [
            (
                "1 Proposed\n"
                "Methodology\n"
                "Method content."
            )
        ]
    )
    wrapped_result = parser.parse(wrapped_heading)
    assert any(
        section.section_type == "methodology"
        for section in wrapped_result.sections
    )

    # ------------------------------------------------------------------
    # 18. PDF word-fragment heading recovery
    # ------------------------------------------------------------------
    # Legacy PDF extraction can split a heading word across isolated physical
    # lines, sometimes with an empty line between them:
    # "EXAM" / "" / "PLES" -> "EXAMPLES".
    fragment_heading = _mock_document(
        [
            (
                "OVERVIEW\n"
                "Overview body.\n\n"
                "THEORETICAL RESULTS ON LEARNING FROM EXAM\n"
                "\n"
                "PLES\n"
                "\n"
                "The theoretical section body starts here with enough "
                "scientific prose to represent a realistic extracted page. "
                "More body text is intentionally included for structural "
                "scoring.\n\n"
                "CONCLUSION\n"
                "Conclusion body."
            )
        ]
    )
    fragment_result = parser.parse(fragment_heading)

    fragment_headings = [
        heading.text
        for heading in fragment_result.detected_headings
    ]

    assert "THEORETICAL RESULTS ON LEARNING FROM EXAMPLES" in (
        fragment_headings
    )
    assert "THEORETICAL RESULTS ON LEARNING FROM EXAM" not in (
        fragment_headings
    )
    assert "PLES" not in fragment_headings
    fragment_sections = [
        section
        for section in fragment_result.sections
        if section.normalized_heading
        == "theoretical_results_on_learning_from_examples"
    ]
    assert fragment_sections
    assert fragment_sections[0].section_type == "other"
    assert fragment_result.parsing_status == "success"
    assert not any(
        "ambiguity" in warning.lower()
        for warning in fragment_result.warnings
    )

    # ------------------------------------------------------------------
    # 19. Do not merge unrelated uppercase headings
    # ------------------------------------------------------------------
    unrelated_headings = _mock_document(
        [
            (
                "RESULTS\n"
                "Results body.\n\n"
                "DISCUSSION\n"
                "Discussion body."
            )
        ]
    )
    unrelated_result = parser.parse(unrelated_headings)

    assert [
        section.original_heading
        for section in unrelated_result.sections
        if section.original_heading
    ] == ["RESULTS", "DISCUSSION"]

    # ------------------------------------------------------------------
    # 20. Malformed input
    # ------------------------------------------------------------------
    malformed = object()

    try:
        parser.parse(malformed)  # type: ignore[arg-type]
    except InvalidExtractedDocumentError:
        pass
    else:
        raise AssertionError(
            "Malformed input was not rejected."
        )

    # ------------------------------------------------------------------
    # 21. Status propagation
    # ------------------------------------------------------------------
    partial_doc = _mock_document(
        [
            (
                "Introduction\n"
                "Text."
            )
        ],
        extraction_status="partial",
    )

    partial_result = parser.parse(
        partial_doc
    )

    assert partial_result.parsing_status == "partial"

    print(
        "SectionParser self-test: PASSED "
        f"({len(result.sections)} standard-paper sections)"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
    run_self_test()