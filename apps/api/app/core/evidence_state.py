from __future__ import annotations

from enum import Enum


class EvidenceState(str, Enum):
    """Authoritative lifecycle state for research papers and evidence availability."""

    METADATA_ONLY = "metadata-only"
    PROCESSING = "processing"
    FULL_TEXT = "full-text"
    FAILED = "failed"

    @classmethod
    def from_str(cls, value: str | None) -> EvidenceState:
        if not value:
            return cls.METADATA_ONLY
        normalized = value.strip().lower()
        # Backward compatibility for legacy "private-upload" label
        if normalized == "private-upload":
            return cls.FULL_TEXT
        try:
            return cls(normalized)
        except ValueError:
            return cls.METADATA_ONLY


ALLOWED_TRANSITIONS: dict[EvidenceState, set[EvidenceState]] = {
    EvidenceState.METADATA_ONLY: {EvidenceState.PROCESSING, EvidenceState.FULL_TEXT},
    EvidenceState.PROCESSING: {
        EvidenceState.FULL_TEXT,
        EvidenceState.FAILED,
        EvidenceState.METADATA_ONLY,
    },
    EvidenceState.FULL_TEXT: {EvidenceState.PROCESSING},  # allow re-indexing
    EvidenceState.FAILED: {
        EvidenceState.PROCESSING,
        EvidenceState.METADATA_ONLY,
    },  # allow retry
}


def can_transition(current: str | EvidenceState, target: str | EvidenceState) -> bool:
    curr_state = EvidenceState.from_str(
        str(current.value if isinstance(current, EvidenceState) else current)
    )
    target_state = EvidenceState.from_str(
        str(target.value if isinstance(target, EvidenceState) else target)
    )
    if curr_state == target_state:
        return True
    return target_state in ALLOWED_TRANSITIONS.get(curr_state, set())


def supports_full_text_qa(state: str | EvidenceState | None) -> bool:
    """True if and only if full-text page-addressable chunks exist and are ready for cited Q&A."""
    if state is None:
        return False
    resolved = EvidenceState.from_str(
        str(state.value if isinstance(state, EvidenceState) else state)
    )
    return resolved == EvidenceState.FULL_TEXT
