from apps.api.app.services.rag import chunk_text, classify_question, embed, evidence_supports, hybrid_retrieve
from apps.api.app.services.reranker import rerank


def test_chunking_preserves_section_and_page() -> None:
    chunks = chunk_text("Methodology\n\nWe evaluate retrieval quality.", page=5)
    assert len(chunks) == 1
    assert chunks[0].section == "Methodology"
    assert chunks[0].page == 5
    assert len(chunks[0].embedding) == 384


def test_hybrid_retrieval_prefers_matching_chunk() -> None:
    chunks = chunk_text("Introduction\n\nVision models classify images.\n\nResults\n\nRetrieval quality improves.")
    results = hybrid_retrieve(chunks, "retrieval quality", limit=1)
    assert results[0].section == "Results"


def test_embeddings_are_stable_for_persisted_retrieval() -> None:
    assert embed("retrieval augmented generation") == embed("retrieval augmented generation")
    assert len(embed("retrieval augmented generation")) == 384


def test_reranker_has_explicit_fallback_mode() -> None:
    chunks = chunk_text("Methodology\n\nWe evaluate retrieval quality.")
    results, mode = rerank("retrieval quality", chunks, limit=1)
    assert results
    assert mode in {"cross-encoder", "hybrid-fallback"}


def test_dataset_retrieval_uses_methods_evidence_not_background() -> None:
    chunks = chunk_text("Introduction\n\nPrior work used ImageNet.\n\nMethods\n\nWe trained on 400 patient records collected from Hospital A.", page=4)
    selected = hybrid_retrieve(chunks, "What dataset was used?", limit=1)
    assert selected[0].section == "Methods"
    assert evidence_supports("What dataset was used?", selected)


def test_abstract_reference_to_another_study_is_not_dataset_evidence() -> None:
    chunks = chunk_text("Abstract\n\nPrior work used the MIMIC dataset; our contribution is a survey.", page=1)
    selected = hybrid_retrieve(chunks, "What dataset was used?", limit=1)
    # Abstract-only material cannot substantiate a current-paper dataset claim.
    assert selected[0].section == "Abstract"
    assert classify_question("What dataset was used?") == "dataset"
