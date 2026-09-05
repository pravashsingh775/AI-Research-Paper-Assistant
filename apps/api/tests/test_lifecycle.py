import pytest
from apps.api.app.core.evidence_state import (
    EvidenceState,
    can_transition,
    supports_full_text_qa,
)


def test_evidence_state_transitions() -> None:
    # Metadata-only can move to processing
    assert can_transition(EvidenceState.METADATA_ONLY, EvidenceState.PROCESSING) is True
    # Processing can move to full-text or failed
    assert can_transition(EvidenceState.PROCESSING, EvidenceState.FULL_TEXT) is True
    assert can_transition(EvidenceState.PROCESSING, EvidenceState.FAILED) is True
    # Failed can retry into processing
    assert can_transition(EvidenceState.FAILED, EvidenceState.PROCESSING) is True
    # Same state is always allowed
    assert can_transition(EvidenceState.FULL_TEXT, EvidenceState.FULL_TEXT) is True


def test_supports_full_text_qa() -> None:
    assert supports_full_text_qa(EvidenceState.FULL_TEXT) is True
    assert supports_full_text_qa("full-text") is True
    assert (
        supports_full_text_qa("private-upload") is True
    )  # backward compatibility alias
    assert supports_full_text_qa(EvidenceState.METADATA_ONLY) is False
    assert supports_full_text_qa(EvidenceState.PROCESSING) is False
    assert supports_full_text_qa(EvidenceState.FAILED) is False
    assert supports_full_text_qa(None) is False
