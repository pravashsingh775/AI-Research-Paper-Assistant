# Architecture Notes

## Bounded contexts

### Discovery
`apps/worker/app/pipelines/discovery/` owns scholarly provider ingestion, local corpus import, normalization, deduplication, embedding, semantic retrieval, and reranking.

### Paper analysis
`apps/worker/app/pipelines/paper_analysis/` owns PDF extraction, section detection, chunking, embeddings, paper-scoped indexing, and source coordinates such as page and section.

### Application layer
`apps/api/app/services/` owns use cases: search, paper upload, summary requests, Q&A, evidence validation, and job submission. Routes should remain thin.

### Presentation
`apps/web/` owns the research workspace, search results, paper reader, analysis panels, citations, and job progress states.

## Storage boundaries

- PostgreSQL: users, papers, normalized metadata, jobs, citations, and audit records.
- pgvector or a dedicated vector store: chunk embeddings and retrieval metadata.
- Object storage: original PDFs, raw provider payloads, and derived artifacts.
- Redis: queue and short-lived cache.

Keep corpus ingestion idempotent. Every generated answer must retain the retrieved chunk IDs used to produce it, and validation must be a separate step before the answer reaches the UI.
