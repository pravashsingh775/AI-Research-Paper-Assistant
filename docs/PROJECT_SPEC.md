# ResearchPilot AI — Project Specification

## Product goal

Help researchers discover scientific literature, upload private PDFs, retrieve relevant passages, and compare papers without presenting unsupported conclusions as established facts.

## Current product boundary

The repository is a Next.js web application, a FastAPI API, a Python background worker, PostgreSQL/pgvector, Redis, and local or S3-compatible object storage. The existing stack is retained rather than replaced with the Vite baseline in the reference prompt; see [ADR 0001](ADR/0001-retain-existing-stack.md).

The current API uses `/api/...` routes, not `/api/v1/...`. The frontend includes discovery, paper-library, collection, comparison, trend, gap, idea, proposal, and similarity screens. The API supports registration/login, paper search and upload, background processing, Q&A/chat, collections, and analysis endpoints.

## Current capabilities and boundaries

- **Discovery:** Searches a local arXiv-derived metadata/abstract corpus and configured scholarly providers. Search filters and pagination are limited.
- **PDF ingestion:** Authenticated upload, PDF validation, object storage, Redis job dispatch, PyMuPDF extraction, chunking, and vector persistence are implemented. OCR is not provided.
- **Retrieval:** A hybrid retrieval path exists. The default local embedding provider is deterministic feature hashing; this is not a pretrained semantic embedding model. Sentence Transformers can be configured.
- **Q&A:** Answers can use retrieved chunks and a configured LLM provider. The support check is heuristic and is not a guarantee of factuality.
- **Research analysis:** Comparison, trend, gap, idea, proposal, and similarity endpoints exist. Their output is limited by the paper metadata/abstracts supplied and the configured model.
- **Persistence:** SQLAlchemy models and Alembic migrations exist for users, papers, documents, chunks, jobs, collections, and conversations.

## Product integrity requirements

1. Keep generated interpretations distinct from source claims.
2. Display provenance only when it exists; never fabricate pages, quotations, authors, links, or citations.
3. Label research gaps and generated ideas as hypotheses requiring human review.
4. Describe benchmark scope and limitations beside every reported score.
5. Never treat a successful local build as proof of production readiness.

## Acceptance gates for production claims

Production readiness requires, at minimum: durable/retryable ingestion, supported deployment configuration, resource-level authorization tests, reproducible migrations and backups, tested request limits and timeouts, verified production secrets, dependency locks, CI coverage of integration behavior, and independently reviewed retrieval/generation evaluation. These gates are not all met; see [PROJECT_REVIEW.md](PROJECT_REVIEW.md).
