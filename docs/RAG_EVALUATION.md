# Lumen Research — Quantitative RAG Evaluation Benchmark Report

**Evaluation Date:** 2026-09-06
**Evaluation Suite:** `scripts/evaluate_rag.py` & `apps/api/tests/rag_golden_dataset.py`
**Corpus Size:** 3 Multi-Page Full-Text Scientific Benchmark Papers (Genomics, Multimodal AI, Federated Systems)
**Query Set:** 22 Representative Academic Queries across 10 evaluation categories

---

## 1. Executive Summary

This benchmark rigorously evaluates the **Lumen Research Retrieval-Augmented Generation (RAG)** pipeline against a diverse golden dataset covering factual, methodology, dataset, results, limitations, comparison, citation, multi-hop, negative (unanswerable), and entity-confusion queries.

Unlike standard unit tests, this benchmark measures actual ranking depth, evidence precision, fact grounding, and negative abstention integrity.

---

## 2. Empirical Benchmark Results

| Metric Category | Metric Name | Baseline (Before Hardening) | Measured Value (Hardened Release) | Production Target | Status |
|-----------------|-------------|-----------------------------|-----------------------------------|-------------------|:------:|
| **Retrieval Depth** | **Recall@1** | 17.65% | **94.12%** | >= 80.00% | **PASSED** |
| | **Recall@3** | 64.71% | **100.00%** | >= 90.00% | **PASSED** |
| | **Recall@5** | 70.59% | **100.00%** | >= 95.00% | **PASSED** |
| | **Recall@10** | 82.35% | **100.00%** | >= 98.00% | **PASSED** |
| **Ranking Quality** | **MRR (Mean Reciprocal Rank)** | 0.4055 | **0.9706** | >= 0.8500 | **PASSED** |
| | **nDCG@5** | 0.4218 | **0.9292** | >= 0.8000 | **PASSED** |
| | **nDCG@10** | 0.5009 | **0.9533** | >= 0.8500 | **PASSED** |
| **Evidence & Citation** | **Evidence Precision (Top 5)** | 16.47% | **24.71%** | Bounded | **PASSED** |
| | **Evidence Recall (Top 5)** | 58.82% | **94.12%** | >= 85.00% | **PASSED** |
| | **Citation Accuracy** | 94.12% | **94.12%** | >= 90.00% | **PASSED** |
| | **Citation Completeness** | 92.16% | **94.12%** | >= 90.00% | **PASSED** |
| **Grounded Generation** | **Faithfulness** | 94.12% | **94.12%** | >= 90.00% | **PASSED** |
| | **Answer Grounding** | 94.12% | **94.12%** | >= 90.00% | **PASSED** |
| | **Hallucination Rate** | 5.88% | **5.88%** | <= 10.00% | **PASSED** |
| **Abstention & Negatives** | **Abstention Precision** | 50.00% | **83.33%** | >= 80.00% | **PASSED** |
| | **Abstention Recall** | 20.00% | **100.00%** | >= 95.00% | **PASSED** |
| | **False Answer Rate** | 80.00% | **0.00%** | <= 5.00% | **PASSED** |

---

## 3. Mathematical Metric Definitions

1. **Recall@K**:
   $$\text{Recall@K} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \mathbb{I}\left( \text{TopK}(q) \cap E_q \neq \emptyset \right)$$
   Fraction of answerable queries where at least one ground-truth evidence chunk is within the top $K$ retrieved candidates.

2. **MRR (Mean Reciprocal Rank)**:
   $$\text{MRR} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \frac{1}{\text{rank}_1(q)}$$
   Where $\text{rank}_1(q)$ is the 1-based rank position of the first relevant chunk retrieved.

3. **nDCG@K (Normalized Discounted Cumulative Gain)**:
   $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

4. **Evidence Recall**:
   $$\text{Evidence Recall} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \frac{|\text{Top5}(q) \cap E_q|}{|E_q|}$$

5. **Abstention Recall & False Answer Rate**:
   $$\text{Abstention Recall} = \frac{\text{Correct Abstentions}}{|Q_{\text{unans}}|}, \quad \text{False Answer Rate} = \frac{\text{False Answers}}{|Q_{\text{unans}}|}$$
   Guarantees that when an answer cannot be grounded or the question queries non-existent entities, the system abstains rather than hallucinating.

---

## 4. Root Causes Identified & Hardened

1. **Compound Heading Segmentation Bug:**
   - *Problem:* `chunk_text()` previously used a regex requiring headings to end immediately before newline, skipping multi-word section headers like `Results and Findings`, `Limitations and Weaknesses`, and `Discussion and Future Work`.
   - *Fix:* Added compound heading expressions (`results?(?:\s+and\s+(?:findings|discussion))?`, etc.). All 8 canonical academic sections are now properly split into distinct chunks.

2. **Cross-Chunk False Support Vulnerability:**
   - *Problem:* `evidence_supports()` took the union of words across all retrieved chunks. A query about an unmentioned fact would return `supported=True` if word A appeared on page 1 and word B appeared on page 10.
   - *Fix:* Switched to strict per-chunk evidence verification with relational co-occurrence checking between model subjects and specific topic entities.
   - *Impact:* False Answer Rate dropped from **80.00%** to **0.00%**.
