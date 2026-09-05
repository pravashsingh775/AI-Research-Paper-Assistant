from __future__ import annotations

import pytest

from apps.api.app.services.rag import (
    Chunk,
    RAGConfig,
    chunk_text,
    classify_question,
    evidence_supports,
    hybrid_retrieve,
    is_primary_study_section,
)

SAMPLE_RESEARCH_PAPER = """
Abstract
Deep neural networks have shown strong performance across many tasks. Prior work used the ImageNet-1K dataset. In this work, we propose GraphSeq, a dual-path architecture for scientific graph classification.

Introduction
Scientific paper representation requires modeling citations and text. Smith et al. (2020) proposed GCN baselines trained on PubMed.

Methodology
Our methodology introduces a bi-directional transformer coupled with a graph attention network. The overall pipeline processes token sequences and adjacency matrices jointly.

Dataset
We evaluated GraphSeq on the Cora Citation Network dataset, containing 2,708 scientific publications and 5,429 links. Each paper is classified into one of seven subject categories.

Preprocessing
All abstracts were lowercased and tokenized using WordPiece with a vocabulary of 30,522 tokens. We filtered out isolated nodes with zero degree.

Model Architecture
The architecture comprises a 6-layer graph attention network (GAT) backbone with 8 attention heads and hidden dimension 256.

Training
The model was trained using the AdamW optimizer with an initial learning rate of 1e-4, weight decay 0.01, batch size 64, for 100 epochs on an NVIDIA A100 GPU.

Evaluation
We used classification accuracy, macro F1 score, and mean reciprocal rank (MRR) as our primary evaluation metrics with 5-fold cross-validation.

Results
GraphSeq achieved 88.4% accuracy on Cora, outperforming GCN (81.5%) and GAT (83.0%) by a significant margin with p < 0.01.

Strengths
Our approach offers linear computational complexity with respect to edge count and provides interpretable attention weights.

Limitations
The primary limitation is high GPU memory consumption during dense matrix operations on graphs with more than 100,000 nodes.

Future Work
Future work will investigate sparse tensor representations and distributed multi-GPU scaling for web-scale networks.
"""


@pytest.fixture
def paper_chunks() -> list[Chunk]:
    return chunk_text(SAMPLE_RESEARCH_PAPER, page=1)


def test_classify_all_11_question_types() -> None:
    questions = {
        "dataset": "What dataset was used in this study?",
        "methodology": "What is the primary methodology and approach?",
        "preprocessing": "How was data preprocessing and tokenization handled?",
        "model": "What model architecture and network was implemented?",
        "training": "What optimizer and training epochs were configured?",
        "evaluation": "Which evaluation metric and accuracy measures were used?",
        "results": "What performance results and improvement were reported?",
        "strengths": "What are the main strengths and advantages of this method?",
        "limitations": "What are the key limitations and weaknesses?",
        "future_work": "What directions for future work were proposed?",
        "metadata": "Who are the authors and venue of this paper?",
    }
    for expected_type, query in questions.items():
        predicted = classify_question(query)
        assert predicted == expected_type, (
            f"Failed on '{query}': expected '{expected_type}', got '{predicted}'"
        )


def test_retrieval_prefers_primary_study_over_background(
    paper_chunks: list[Chunk],
) -> None:
    # Query about dataset should retrieve the "Dataset" section (Cora), NOT the "Abstract" (ImageNet)
    results = hybrid_retrieve(
        paper_chunks, "What dataset was evaluated in this study?", limit=3
    )
    assert len(results) > 0
    top_chunk = results[0]
    assert "cora" in top_chunk.text.lower()
    assert top_chunk.section.lower() == "dataset"


def test_evidence_supports_grounded_claims(paper_chunks: list[Chunk]) -> None:
    # Grounded claim in the paper
    dataset_chunks = hybrid_retrieve(paper_chunks, "What dataset was used?", limit=3)
    assert evidence_supports("What dataset was used?", dataset_chunks) is True

    # Methodology claim in the paper
    method_chunks = hybrid_retrieve(
        paper_chunks, "What is the pipeline and methodology?", limit=3
    )
    assert (
        evidence_supports("What is the pipeline and methodology?", method_chunks)
        is True
    )


def test_evidence_abstains_on_unmentioned_facts(paper_chunks: list[Chunk]) -> None:
    # Completely unmentioned facts MUST fail support check
    unsupported_query = "What color were the laboratory walls during the experiment?"
    top_chunks = hybrid_retrieve(paper_chunks, unsupported_query, limit=3)
    assert evidence_supports(unsupported_query, top_chunks) is False

    unsupported_dataset = "What animal subjects were used in the clinical trial?"
    top_chunks_animal = hybrid_retrieve(paper_chunks, unsupported_dataset, limit=3)
    # The paper does not discuss animals or clinical trials
    assert evidence_supports(unsupported_dataset, top_chunks_animal, "dataset") is False


def test_primary_study_section_detection() -> None:
    assert is_primary_study_section("Methodology") is True
    assert is_primary_study_section("Dataset") is True
    assert is_primary_study_section("Results") is True
    assert is_primary_study_section("Abstract") is False
    assert is_primary_study_section("Related Work") is False
    assert is_primary_study_section("References") is False


def test_rag_config_custom_weights(paper_chunks: list[Chunk]) -> None:
    # Dense-heavy config
    dense_cfg = RAGConfig(dense_weight=0.9, lexical_weight=0.1)
    results_dense = hybrid_retrieve(
        paper_chunks, "optimizer AdamW", limit=2, config=dense_cfg
    )
    assert len(results_dense) > 0

    # Lexical-heavy config
    lexical_cfg = RAGConfig(dense_weight=0.1, lexical_weight=0.9)
    results_lexical = hybrid_retrieve(
        paper_chunks, "optimizer AdamW", limit=2, config=lexical_cfg
    )
    assert len(results_lexical) > 0
    assert "adamw" in results_lexical[0].text.lower()
