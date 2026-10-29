"""Evaluate retrieval and heuristic evidence support on the synthetic golden corpus.

This script does not invoke an LLM or evaluate generated answers or citations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.app.services.rag import (
    Chunk,
    chunk_text,
    evidence_supports,
    hybrid_retrieve,
)
from apps.api.tests.rag_golden_dataset import GOLDEN_QUERIES, PAPERS_CORPUS


def dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2**rel - 1) / math.log2(idx + 2)
        for idx, rel in enumerate(relevances[:k])
    )


def ndcg(retrieved_chunk_indices: list[int], expected_indices: list[int], k: int) -> float:
    if not expected_indices:
        return 1.0
    rels = [1 if idx in expected_indices else 0 for idx in retrieved_chunk_indices[:k]]
    actual_dcg = dcg(rels, k)
    ideal_rels = sorted([1] * min(len(expected_indices), k) + [0] * max(0, k - len(expected_indices)), reverse=True)
    ideal_dcg = dcg(ideal_rels, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_rag_pipeline() -> dict[str, float]:
    # 1. Chunk all papers in corpus
    processed_papers: dict[str, list[Chunk]] = {}
    for pid, pdata in PAPERS_CORPUS.items():
        processed_papers[pid] = chunk_text(pdata["text"])

    answerable_queries = [q for q in GOLDEN_QUERIES if q.answerable]
    unanswerable_queries = [q for q in GOLDEN_QUERIES if not q.answerable]

    recall_at_1: list[float] = []
    recall_at_3: list[float] = []
    recall_at_5: list[float] = []
    recall_at_10: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_5_list: list[float] = []
    ndcg_10_list: list[float] = []
    evidence_precisions: list[float] = []
    evidence_recalls: list[float] = []
    expected_fact_coverage: list[float] = []

    supported_answerable_queries = 0
    correct_abstentions = 0
    supported_unanswerable_queries = 0
    total_unanswerable = len(unanswerable_queries)
    total_abstention_decisions = 0

    print(
        f"[*] Evaluating {len(GOLDEN_QUERIES)} queries against "
        f"{len(PAPERS_CORPUS)} synthetic benchmark papers..."
    )

    for q in answerable_queries:
        chunks = processed_papers[q.target_paper_id]
        retrieved = hybrid_retrieve(chunks, q.question, limit=10)
        retrieved_indices = [c.index for c in retrieved]

        # Recall @ K
        has_at_1 = any(idx in q.expected_chunk_indices for idx in retrieved_indices[:1])
        has_at_3 = any(idx in q.expected_chunk_indices for idx in retrieved_indices[:3])
        has_at_5 = any(idx in q.expected_chunk_indices for idx in retrieved_indices[:5])
        has_at_10 = any(idx in q.expected_chunk_indices for idx in retrieved_indices[:10])

        recall_at_1.append(1.0 if has_at_1 else 0.0)
        recall_at_3.append(1.0 if has_at_3 else 0.0)
        recall_at_5.append(1.0 if has_at_5 else 0.0)
        recall_at_10.append(1.0 if has_at_10 else 0.0)

        # MRR
        rr = 0.0
        for rank, idx in enumerate(retrieved_indices, start=1):
            if idx in q.expected_chunk_indices:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # nDCG
        ndcg_5_list.append(ndcg(retrieved_indices, q.expected_chunk_indices, 5))
        ndcg_10_list.append(ndcg(retrieved_indices, q.expected_chunk_indices, 10))

        # Evidence Precision & Recall (at Top 5)
        top5 = retrieved_indices[:5]
        relevant_in_top5 = sum(1 for idx in top5 if idx in q.expected_chunk_indices)
        prec = relevant_in_top5 / len(top5) if top5 else 0.0
        rec = relevant_in_top5 / len(q.expected_chunk_indices) if q.expected_chunk_indices else 0.0
        evidence_precisions.append(prec)
        evidence_recalls.append(rec)

        # Evidence support & grounding
        supported = evidence_supports(q.question, retrieved[:5])
        if supported:
            supported_answerable_queries += 1
            retrieved_text = " ".join(c.text.lower() for c in retrieved[:5])
            facts_found = sum(1 for f in q.expected_facts if f.lower() in retrieved_text)
            coverage = facts_found / len(q.expected_facts) if q.expected_facts else 1.0
            expected_fact_coverage.append(coverage)
        else:
            expected_fact_coverage.append(0.0)
            total_abstention_decisions += 1

    for q in unanswerable_queries:
        chunks = processed_papers[q.target_paper_id]
        if q.distractor_paper_id:
            # Add distractor chunks to test entity confusion
            chunks = chunks + processed_papers[q.distractor_paper_id]

        retrieved = hybrid_retrieve(chunks, q.question, limit=5)
        supported = evidence_supports(q.question, retrieved)

        if not supported:
            correct_abstentions += 1
            total_abstention_decisions += 1
        else:
            supported_unanswerable_queries += 1

    metrics = {
        "recall@1": sum(recall_at_1) / len(recall_at_1),
        "recall@3": sum(recall_at_3) / len(recall_at_3),
        "recall@5": sum(recall_at_5) / len(recall_at_5),
        "recall@10": sum(recall_at_10) / len(recall_at_10),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg@5": sum(ndcg_5_list) / len(ndcg_5_list),
        "ndcg@10": sum(ndcg_10_list) / len(ndcg_10_list),
        "evidence_precision": sum(evidence_precisions) / len(evidence_precisions),
        "evidence_recall": sum(evidence_recalls) / len(evidence_recalls),
        "answerable_support_rate": supported_answerable_queries / len(answerable_queries),
        "expected_fact_coverage": sum(expected_fact_coverage) / len(expected_fact_coverage),
        "unanswerable_support_rate": (
            supported_unanswerable_queries / total_unanswerable
            if total_unanswerable
            else 0.0
        ),
        "abstention_precision": correct_abstentions / total_abstention_decisions if total_abstention_decisions else 1.0,
        "abstention_recall": correct_abstentions / total_unanswerable if total_unanswerable else 1.0,
    }

    print("\n" + "=" * 60)
    print("      RESEARCHPILOT AI — RETRIEVAL EVALUATION                 ")
    print("=" * 60)
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Recall@1:              {metrics['recall@1'] * 100:.2f}%")
    print(f"  Recall@3:              {metrics['recall@3'] * 100:.2f}%")
    print(f"  Recall@5:              {metrics['recall@5'] * 100:.2f}%")
    print(f"  Recall@10:             {metrics['recall@10'] * 100:.2f}%")
    print(f"  MRR:                   {metrics['mrr']:.4f}")
    print(f"  nDCG@5:                {metrics['ndcg@5']:.4f}")
    print(f"  nDCG@10:               {metrics['ndcg@10']:.4f}")
    print("\nRETRIEVED EVIDENCE:")
    print(f"  Evidence Precision@5:  {metrics['evidence_precision'] * 100:.2f}%")
    print(f"  Evidence Recall@5:     {metrics['evidence_recall'] * 100:.2f}%")
    print("\nHEURISTIC SUPPORT SCREENING (not generated-answer evaluation):")
    print(f"  Answerable Support:    {metrics['answerable_support_rate'] * 100:.2f}%")
    print(f"  Expected Fact Coverage:{metrics['expected_fact_coverage'] * 100:.2f}%")
    print(f"  Unanswerable Support:  {metrics['unanswerable_support_rate'] * 100:.2f}%")
    print(f"  Abstention Precision:  {metrics['abstention_precision'] * 100:.2f}%")
    print(f"  Abstention Recall:     {metrics['abstention_recall'] * 100:.2f}%")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_rag_pipeline()
