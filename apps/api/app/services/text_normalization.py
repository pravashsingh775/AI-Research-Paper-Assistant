"""Centralized, deterministic text normalization and validation for academic paper content.

Contracts:
- Strip U+0000 / NUL bytes which are strictly rejected by PostgreSQL TEXT/VARCHAR.
- Recover double-encoded Latin-1 mojibake when safe, without altering valid non-Latin-1 Unicode.
- Remove lone surrogates (U+D800 to U+DFFF) which violate valid UTF-8.
- Strictly preserve all scientific notation, mathematical symbols (e.g. α, β, γ, χ, μ, ∑, √, ≤, ≥, |ψ⟩),
  subscripts, superscripts, accented Latin characters (é, à, ö), multilingual scripts (Hindi, Chinese, Japanese, etc.),
  newlines (\\n), and meaningful whitespace.
- Provide structured observability metrics (e.g., removed NUL count) without leaking document content.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

logger = logging.getLogger("text_normalization")

# Regex pattern matching lone surrogate code points (U+D800 to U+DFFF)
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


class NormalizationResult(NamedTuple):
    text: str
    removed_nul_count: int
    normalization_applied: bool


def normalize_extracted_text(
    text: str | None,
    document_id: str | None = None,
    job_id: str | None = None,
) -> str:
    """Normalize extracted document text for safe persistence and retrieval.

    Safe for PostgreSQL TEXT/VARCHAR columns.
    Preserves all valid Unicode, scientific symbols, and meaningful layout.
    """
    cleaned, _ = normalize_and_audit_text(text, document_id=document_id, job_id=job_id)
    return cleaned


def normalize_and_audit_text(
    text: str | None,
    document_id: str | None = None,
    job_id: str | None = None,
) -> tuple[str, NormalizationResult]:
    """Normalize text and return audit metadata for structured observability."""
    if text is None:
        return "", NormalizationResult(text="", removed_nul_count=0, normalization_applied=False)

    if not isinstance(text, str):
        text = str(text)

    # 1. Attempt Latin-1 to UTF-8 mojibake recovery only if text is purely double-encoded.
    # Note: If text contains native characters outside latin-1 (Greek, Cyrillic, CJK, Devanagari, Math),
    # text.encode("latin1") will raise UnicodeEncodeError, in which case we preserve the original text untouched.
    recovered = text
    try:
        candidate = text.encode("latin1").decode("utf-8")
        if "\ufffd" not in candidate and len(candidate) > 0:
            recovered = candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # 2. Count and remove NUL bytes (0x00 / U+0000)
    nul_count = recovered.count("\x00")
    if nul_count > 0:
        recovered = recovered.replace("\x00", "")

    # 3. Strip lone surrogates if any
    surrogate_count = len(_SURROGATE_RE.findall(recovered))
    if surrogate_count > 0:
        recovered = _SURROGATE_RE.sub("", recovered)

    # 4. Observability: log structured metric if modifications were made without logging document content
    modified = (nul_count > 0) or (surrogate_count > 0) or (recovered != text)
    if nul_count > 0 or surrogate_count > 0:
        logger.info(
            "Text normalization applied: document_id=%s job_id=%s removed_nul_count=%d removed_surrogates=%d",
            document_id or "unknown",
            job_id or "unknown",
            nul_count,
            surrogate_count,
        )

    result = NormalizationResult(
        text=recovered,
        removed_nul_count=nul_count,
        normalization_applied=modified,
    )
    return recovered, result


def validate_text_for_persistence(
    text: str | None,
    field_name: str = "text",
) -> tuple[bool, str | None]:
    """Validate that text is 100% compliant with PostgreSQL UTF-8 constraints.

    Returns (True, None) if valid, or (False, error_reason) if invalid.
    """
    if text is None:
        return True, None
    if "\x00" in text:
        return False, f"Field '{field_name}' contains prohibited NUL byte (0x00 / U+0000)"
    if _SURROGATE_RE.search(text):
        return False, f"Field '{field_name}' contains unencodable lone surrogate code points"
    return True, None
