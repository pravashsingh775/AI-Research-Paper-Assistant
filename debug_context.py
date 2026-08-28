from backend.rag.context import ContextBuilder
from types import SimpleNamespace

query = "What are the two general forms of bias in learning from examples?"

texts = [
    ("chk_000022", "Preference Bias"),
    ("chk_000009", "Restricted Hypothesis Space Bias"),
    ("chk_000023", "Preference Bias"),
    (
        "chk_000008",
        "There are two general forms of bias: "
        "restricted hypothesis space bias and preference bias."
    ),
]

builder = ContextBuilder()

for rank, (chunk_id, text) in enumerate(texts, start=1):
    candidate = SimpleNamespace(
        evidence_id=chunk_id,
        document_id="PAPER",
        paper_id=None,
        chunk_id=chunk_id,
        rank=rank,
        score=0.7,
        reranker_score=None,
        retrieval_score=None,
        section="",
        page=None,
        page_end=None,
        source_pages=(),
        title=None,
        text=text,
        source_metadata={},
    )

    result = builder._with_query_relevance(candidate, query)

    print(
        f"{chunk_id}: "
        f"overlap={result.query_overlap:.6f}, "
        f"phrase_match={result.query_phrase_match}"
    )
