import pytest
from apps.api.app.providers.scholarly import (
    reconstruct_inverted_index,
    _normalize_openalex,
)


def test_reconstruct_inverted_index_correct_order() -> None:
    inv = {
        "We": [0],
        "evaluate": [1],
        "graph": [2],
        "neural": [3],
        "networks": [4],
        "on": [5],
        "the": [6],
        "Cora": [7],
        "dataset.": [8],
    }
    result = reconstruct_inverted_index(inv)
    assert result == "We evaluate graph neural networks on the Cora dataset."


def test_reconstruct_inverted_index_multiple_positions_per_word() -> None:
    inv = {
        "The": [0, 4],
        "model": [1],
        "outperformed": [2],
        "the": [3],
        "baseline.": [5],
    }
    result = reconstruct_inverted_index(inv)
    assert result == "The model outperformed the The baseline."


def test_reconstruct_inverted_index_empty_or_malformed() -> None:
    assert reconstruct_inverted_index({}) == ""
    assert reconstruct_inverted_index(None) == ""
    assert reconstruct_inverted_index("not a dict") == ""
    assert reconstruct_inverted_index({"bad": "not a list"}) == ""
    assert reconstruct_inverted_index({"word": [-1, "invalid"]}) == ""


def test_normalize_openalex_uses_reconstructed_abstract() -> None:
    raw_item = {
        "id": "https://openalex.org/W123456",
        "title": "A Sample Paper",
        "abstract_inverted_index": {
            "Deep": [0],
            "learning": [1],
            "improves": [2],
            "accuracy.": [3],
        },
        "authorships": [{"author": {"display_name": "Alice Smith"}}],
        "publication_date": "2024-01-15",
        "cited_by_count": 42,
        "primary_location": {"source": {"display_name": "NeurIPS"}},
        "open_access": {"is_oa": True},
    }
    normalized = _normalize_openalex(raw_item)
    assert normalized["id"] == "W123456"
    assert normalized["summary"] == "Deep learning improves accuracy."
    assert normalized["authors_raw"] == "Alice Smith"
    assert normalized["citation_count"] == 42
    assert normalized["venue"] == "NeurIPS"
    assert normalized["open_access"] is True
