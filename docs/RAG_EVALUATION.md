# Retrieval and Evidence-Support Evaluation

## Reproducible run

- **Run date:** 2026-10-29
- **Command:** `python scripts/evaluate_rag.py`
- **Queries:** 22 source-defined queries (17 answerable, 5 unanswerable)
- **Corpus:** 3 synthetic, source-defined papers
- **Scope:** local retrieval and the `evidence_supports` heuristic; no LLM calls or generated answers

The script was run in the current workspace and returned:

| Metric | Result |
|---|---:|
| Recall@1 | 94.12% |
| Recall@3 | 100.00% |
| Recall@5 | 100.00% |
| Recall@10 | 100.00% |
| MRR | 0.9706 |
| nDCG@5 | 0.9292 |
| nDCG@10 | 0.9533 |
| Evidence precision@5 | 24.71% |
| Evidence recall@5 | 94.12% |
| Answerable-query support rate | 94.12% |
| Expected-fact coverage | 94.12% |
| Unanswerable-query support rate | 0.00% |
| Heuristic abstention precision | 83.33% |
| Heuristic abstention recall | 100.00% |

Recall@K is implemented as a hit rate: a query counts as a hit when at least one expected chunk appears in the first K results. Evidence precision/recall use the expected chunk labels. Support and abstention figures measure a rule-based classifier, not generated model responses.

## Limitations

- The three benchmark papers and their labels are synthetic fixtures in `apps/api/tests/rag_golden_dataset.py`; they are not external scientific papers.
- This evaluation does not call an LLM, generate answers, parse citations, or judge answer correctness. It therefore cannot establish citation accuracy, answer faithfulness, or a hallucination rate.
- The test set is small and not representative of the 287,421-row local corpus or user-uploaded PDFs.
- Results depend on the embedding provider selected by the current environment. Record that provider and model with future benchmark runs.
- Evidence precision@5 is 24.71%; most returned chunks are not labeled relevant for the test query. Ranking quality needs broader evaluation.
- The local corpus is metadata/abstract data. Its presence does not establish full-text coverage or licensing for redistribution.

## Next evaluation work

1. Add a versioned, human-reviewed dataset of real queries, relevant papers, and evidence spans with documented source/licensing.
2. Compare lexical, dense, hybrid, and reranked retrieval on a held-out split; report confidence intervals and query-level failures.
3. Evaluate generated responses and citations separately with a blinded human rubric. Keep abstention, evidence support, citation correctness, and factuality as distinct measures.
4. Record the embedding model/version, retrieval parameters, corpus snapshot, software versions, and exact run command for every result.
