"""
End-to-end retrieval integration tests.

Validates the critical contract:

    query
      -> DenseRetriever
      -> metadata_store
      -> CrossEncoderReranker
      -> Ranker
      -> final results

These tests intentionally use deterministic fakes. They do not require:
- a GPU
- Hugging Face downloads
- the production SPECTER2 model
- the production reranker model
- the production FAISS index
- the real 287k-paper metadata file

Run from the project root:

    python -m pytest tests/test_retrieval_integration.py -q

or, if this file is temporarily copied to the project root:

    python -m pytest test_retrieval_integration_final.py -q
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.research.search import ResearchSearchConfig, ResearchSearchService
from backend.retrieval.metadata_store import ResearchMetadataStore
from backend.retrieval.ranking import Ranker
from backend.retrieval.reranker import CrossEncoderReranker, FakeRerankerScorer
from backend.retrieval.retriever import (
    DenseRetriever,
    UploadedPaperRetriever,
)


# ---------------------------------------------------------------------------
# Deterministic test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeSearchResult:
    """Minimal FAISS-like result consumed by DenseRetriever."""

    document_id: str
    score: float
    index_position: int


class FakeEncoder:
    """Small deterministic query encoder."""

    model_name = "mock"
    model_version = "test"

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls = 0
        self.last_query: str | None = None

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        self.last_query = text
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeIndex:
    """Small FAISS-compatible index double."""

    dimension = 3

    def __init__(self, results: list[FakeSearchResult]) -> None:
        self.results = list(results)
        self.ntotal = len(self.results)
        self.calls = 0
        self.last_top_k: int | None = None
        self.last_exclude_ids: set[str] | None = None
        self.last_query_embedding: np.ndarray | None = None

    def search(
        self,
        query_embedding: Any,
        top_k: int,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[FakeSearchResult]:
        self.calls += 1
        self.last_top_k = top_k
        self.last_exclude_ids = set(exclude_ids or set())
        self.last_query_embedding = np.asarray(query_embedding)

        return [
            item
            for item in self.results
            if item.document_id not in self.last_exclude_ids
        ][:top_k]

    def validate(self) -> None:
        if self.ntotal != len(self.results):
            raise AssertionError("Fake index ntotal mismatch.")


# ---------------------------------------------------------------------------
# Fixtures / metadata
# ---------------------------------------------------------------------------


def write_metadata(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "document_count": 3,
        "documents": [
            {
                "document_id": "p1",
                "title": "Paper One",
                "summary": "Alpha",
            },
            {
                "document_id": "p2",
                "title": "Paper Two",
                "summary": "Beta",
            },
            {
                "document_id": "p3",
                "title": "Paper Three",
                "summary": "Gamma",
            },
        ],
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_uploaded_metadata(path: Path, document_id: str) -> None:
    payload = {
        "schema_version": "1",
        "document_id": document_id,
        "vector_count": 3,
        "embedding_dimension": 3,
        "chunks": [
            {
                "vector_position": 0,
                "chunk_id": "c0",
                "document_id": document_id,
                "section_id": "s0",
                "section_type": "abstract",
                "section_heading": "Abstract",
                "chunk_index": 0,
                "text": "alpha",
                "source_pages": [1],
                "metadata": {},
            },
            {
                "vector_position": 1,
                "chunk_id": "c1",
                "document_id": document_id,
                "section_id": "s1",
                "section_type": "method",
                "section_heading": "Method",
                "chunk_index": 1,
                "text": "beta",
                "source_pages": [2],
                "metadata": {},
            },
            {
                "vector_position": 2,
                "chunk_id": "c2",
                "document_id": document_id,
                "section_id": "s2",
                "section_type": "results",
                "section_heading": "Results",
                "chunk_index": 2,
                "text": "gamma",
                "source_pages": [3],
                "metadata": {},
            },
        ],
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_uploaded_manifest(path: Path, document_id: str) -> None:
    """Create a production-compatible uploaded-paper FAISS manifest."""
    payload = {
        "schema_version": "2",
        "paper_document_id": document_id,
        "document_id": document_id,
        "vector_count": 3,
        "document_count": 3,
        "embedding_dimension": 3,
        "document_ids": ["c0", "c1", "c2"],
        "vector_ids": ["c0", "c1", "c2"],
        "id_contract": {
            "document_id": "paper/document owner ID",
            "vector_id": "chunk_id",
        },
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_uploaded_test_index(
    paper_dir: Path,
    document_id: str,
) -> None:
    """Create the complete uploaded-paper fixture contract."""
    write_uploaded_metadata(
        paper_dir / "metadata.json",
        document_id,
    )
    write_uploaded_manifest(
        paper_dir / "index.index.manifest.json",
        document_id,
    )


# ---------------------------------------------------------------------------
# Research retrieval -> metadata -> reranker -> ranking
# ---------------------------------------------------------------------------


def test_metadata_to_reranker_bridge(tmp_path: Path) -> None:
    """
    Verify the most important production integration contract.

    FAISS results must retain stable IDs and retrieval scores.
    metadata_store must provide title/summary.
    reranking must consume that semantic metadata.
    ranking must return the reranker-authoritative Top-K.
    """

    metadata_path = tmp_path / "metadata.json"
    write_metadata(metadata_path)

    encoder = FakeEncoder()

    index = FakeIndex(
        [
            FakeSearchResult("p1", 0.91, 0),
            FakeSearchResult("p2", 0.89, 1),
            FakeSearchResult("p3", 0.87, 2),
        ]
    )

    retriever = DenseRetriever(
        encoder,
        index,
        default_candidate_k=3,
    )

    # Deliberately make the reranker disagree with FAISS:
    # p1 has the highest retrieval score but the lowest reranker score.
    fake_scorer = FakeRerankerScorer([0.20, 0.95, 0.90])

    reranker = CrossEncoderReranker(
        device="cpu",
        batch_size=2,
        max_length=128,
        final_k=2,
        scorer=fake_scorer,
    )

    metadata_store = ResearchMetadataStore(
        metadata_path,
        expected_document_count=3,
    )

    service = ResearchSearchService(
        retriever=retriever,
        reranker=reranker,
        ranker=Ranker(top_k=2),
        metadata_store=metadata_store,
        config=ResearchSearchConfig(
            candidate_k=3,
            rerank_k=2,
            final_k=2,
        ),
    )

    response = service.search("scientific retrieval")

    # Final ranking must follow the reranker rather than the initial FAISS
    # ordering.
    assert [result.document_id for result in response.results] == [
        "p2",
        "p3",
    ]

    assert response.candidates_retrieved == 3
    assert response.candidates_reranked == 2

    # The reranker must receive real semantic metadata, not only IDs.
    assert fake_scorer.last_pairs
    first_pair = fake_scorer.last_pairs[0]

    assert first_pair[0] == "scientific retrieval"
    assert first_pair[1].startswith("Title:\nPaper One")
    assert "Summary:\nAlpha" in first_pair[1]

    assert encoder.calls == 1
    assert index.calls == 1
    assert encoder.last_query == "scientific retrieval"
    assert index.last_top_k == 3

    # Verify the query vector contract.
    assert index.last_query_embedding is not None
    assert index.last_query_embedding.shape == (3,)


def test_research_metadata_store_returns_expected_documents(
    tmp_path: Path,
) -> None:
    """Verify stable ID -> metadata resolution independently."""

    metadata_path = tmp_path / "metadata.json"
    write_metadata(metadata_path)

    store = ResearchMetadataStore(
        metadata_path,
        expected_document_count=3,
    )

    # Support the expected public lookup API when available.
    if hasattr(store, "get"):
        p1 = store.get("p1")
    elif hasattr(store, "get_document"):
        p1 = store.get_document("p1")
    else:
        pytest.fail(
            "ResearchMetadataStore must expose get() or get_document()."
        )

    assert p1 is not None

    # Be tolerant of either a mapping-like or model-like metadata object.
    if isinstance(p1, dict):
        assert p1["document_id"] == "p1"
        assert p1["title"] == "Paper One"
        assert p1["summary"] == "Alpha"
    else:
        assert getattr(p1, "document_id") == "p1"
        assert getattr(p1, "title") == "Paper One"
        assert getattr(p1, "summary") == "Alpha"


# ---------------------------------------------------------------------------
# Uploaded-paper retrieval
# ---------------------------------------------------------------------------


def test_uploaded_retrieval_can_return_broad_candidate_pool(
    tmp_path: Path,
) -> None:
    """
    Uploaded-paper retrieval must be able to return a broad candidate pool
    before reranking.

    This is important because final_k must not prematurely truncate the
    candidate set before the cross-encoder gets a chance to reorder it.
    """

    document_id = "paper-1"
    paper_dir = tmp_path / document_id
    paper_dir.mkdir()

    write_uploaded_test_index(
        paper_dir,
        document_id,
    )

    encoder = FakeEncoder()

    uploaded_index = FakeIndex(
        [
            FakeSearchResult("c0", 0.90, 0),
            FakeSearchResult("c1", 0.80, 1),
            FakeSearchResult("c2", 0.70, 2),
        ]
    )

    retriever = UploadedPaperRetriever(
        encoder,
        uploaded_root=tmp_path,
        candidate_k=3,
        index_loader=lambda *_args, **_kwargs: uploaded_index,
    )

    results = retriever.retrieve_for_document(
        "what method was used?",
        document_id,
        top_k=1,
        candidate_k=3,
        return_candidate_pool=True,
    )

    assert len(results) == 3
    assert [item.chunk_id for item in results] == [
        "c0",
        "c1",
        "c2",
    ]

    # Retrieval must remain isolated to the requested uploaded document.
    assert all(
        item.document_id == document_id
        for item in results
    )

    assert encoder.calls == 1
    assert uploaded_index.calls == 1


def test_uploaded_retrieval_respects_candidate_pool_limit(
    tmp_path: Path,
) -> None:
    """Candidate_k must bound the pre-reranking uploaded-paper pool."""

    document_id = "paper-2"
    paper_dir = tmp_path / document_id
    paper_dir.mkdir()

    write_uploaded_test_index(
        paper_dir,
        document_id,
    )

    encoder = FakeEncoder()

    uploaded_index = FakeIndex(
        [
            FakeSearchResult("c0", 0.90, 0),
            FakeSearchResult("c1", 0.80, 1),
            FakeSearchResult("c2", 0.70, 2),
        ]
    )

    retriever = UploadedPaperRetriever(
        encoder,
        uploaded_root=tmp_path,
        candidate_k=3,
        index_loader=lambda *_args, **_kwargs: uploaded_index,
    )

    results = retriever.retrieve_for_document(
        "results",
        document_id,
        top_k=1,
        candidate_k=2,
        return_candidate_pool=True,
    )

    assert len(results) == 2
    assert [item.chunk_id for item in results] == ["c0", "c1"]
    assert uploaded_index.last_top_k == 2


# ---------------------------------------------------------------------------
# Contract / failure tests
# ---------------------------------------------------------------------------


def test_fake_index_is_dimension_compatible() -> None:
    """Catch accidental changes to the query embedding dimension."""

    encoder = FakeEncoder()
    index = FakeIndex([])

    assert encoder.dimension == index.dimension


def test_metadata_file_is_utf8_json(tmp_path: Path) -> None:
    """Metadata generated for integration tests must be valid JSON."""

    metadata_path = tmp_path / "metadata.json"
    write_metadata(metadata_path)

    payload = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "1"
    assert payload["document_count"] == 3
    assert len(payload["documents"]) == 3


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """
    Lightweight standalone runner.

    Pytest remains the preferred runner because tmp_path and assertion
    reporting are better there.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        test_metadata_to_reranker_bridge(root)
        test_research_metadata_store_returns_expected_documents(root)
        test_uploaded_retrieval_can_return_broad_candidate_pool(root)
        test_uploaded_retrieval_respects_candidate_pool_limit(root)
        test_fake_index_is_dimension_compatible()
        test_metadata_file_is_utf8_json(root)

    print("Retrieval integration self-test: PASSED")


if __name__ == "__main__":
    run_self_test()