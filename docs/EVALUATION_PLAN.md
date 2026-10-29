# Evaluation Plan

## Current baseline

`python scripts/evaluate_rag.py` evaluates hybrid retrieval and a rule-based evidence-support function against 22 queries and 3 synthetic papers. It does not evaluate generated answers or citation objects. Current measurements and caveats are in [RAG_EVALUATION.md](RAG_EVALUATION.md).

## Retrieval evaluation

Create a versioned dataset with:

- Stable query and paper identifiers.
- Human-reviewed relevant paper and evidence-span labels.
- Source and licensing metadata.
- Answerability labels and, when possible, multiple acceptable evidence spans.
- A held-out split not used to tune ranking weights or thresholds.

Compare BM25/lexical, dense, hybrid fusion, and hybrid plus reranking using Recall@K, Precision@K, MRR, and nDCG. Include per-query failures, corpus coverage, model/version/configuration, and confidence intervals. Evaluate ranking latency and indexing cost separately from relevance.

## Answer and citation evaluation

Run generation evaluation separately from retrieval. For each answerable and unanswerable query, record retrieved evidence IDs, generated answer, citation references, and a blinded human label for:

- Claim-level support and factual correctness.
- Citation existence, source identity, and passage/page alignment.
- Missing or contradictory evidence.
- Correct abstention when the corpus is insufficient.

Do not label a heuristic support decision as faithfulness, citation accuracy, or hallucination rate. Do not publish aggregate scores without denominator, annotation method, dataset version, and limitations.

## Product-level evaluation

- **Ingestion:** valid/invalid/malformed PDFs, extraction coverage, page mapping, duplicate detection, retries, and worker crash recovery.
- **Search UX:** query success, filters, stable ordering, empty/error/loading behavior, and source coverage.
- **Analysis:** compare structured fields against paper passages; mark absent vs. inferred information.
- **Operations:** cold/warm startup, readiness, queue delay, API latency/error rate, resource use, and backup restoration.

The local CSV/JSONL corpus is large but not a gold evaluation set. Verify its provenance and redistribution terms before using it as a public demo dataset.
