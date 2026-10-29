# Repository Review — ResearchPilot AI

**Review date:** 2026-10-29
**Scope:** Tracked source, configuration, tests, documentation, and local corpus metadata; focused code inspection of API/authentication, retrieval, ingestion, worker, frontend, CI, and migrations. This review is not a penetration test or production acceptance.

## Executive assessment

The project is a substantial full-stack prototype with real API, database models/migrations, PDF processing, background work, provider integrations, and UI screens. It is not yet a production-ready platform. The most important current gaps are job durability, search reliability, embedding quality, reproducible service-backed tests, API versioning/filtering, and evidence-backed evaluation.

The original repository contained unsupported production and zero-hallucination claims. Those claims have been removed from the README and the retrieval evaluator now labels its rule-based measurements accurately. The historical acceptance and load-test artifacts are retained but are not current proof.

## Verification performed

| Check | Result |
|---|---|
| `python -m pytest apps/api/tests/ -m "not integration" -q` | 42 passed, 9 integration tests deselected, 2 deprecation warnings |
| `npm run build` | Passed; Next.js generated 14 routes. Local Node.js was 26.5.0; CI targets Node.js 20. |
| `python scripts/evaluate_rag.py` | Ran; 22 queries over 3 synthetic papers. Metrics are described in [RAG_EVALUATION.md](RAG_EVALUATION.md). |
| Full API/database/worker integration | Not run; Docker Desktop engine was unavailable. |
| Ruff | Focused ingestion/evaluation files and API import sorting pass. The broader changed-Python check reports 81 findings, primarily FastAPI `B008` defaults, unused imports, and broad exception handling; Python lint is not clean. |
| Frontend lint | Not verified; the configured `next lint` command opens an interactive setup prompt. |

The pre-change full-suite run produced 34 passed, 8 failed, and 3 errors. It exposed unauthenticated analytical routes and missing live services. The analytical routes were fixed and covered by the passing offline suite; service-backed failures still require a Docker host.

## Data inventory

The raw dataset and processed JSONL are local, ignored by Git, and were not copied or modified.

| Asset | Observed |
|---|---|
| `data/raw/corpus/arxiv_scientific_dataset.csv` | 430,180,631 bytes; 287,421 rows; 10 columns |
| `data/processed/arxiv_papers.jsonl` | 509,361,563 bytes; 287,421 lines; 0 malformed JSON rows |
| Required raw fields checked (`id`, `title`, `category_code`, dates, authors, summary) | No missing values; raw IDs unique |
| Processed identifiers | 287,421 unique IDs; no missing checked IDs/titles/category/date/summary/hash fields |
| Identical processed `content_hash` values | 73 repeated hash groups (146 records); the hash is over the normalized abstract, so these are candidate duplicate abstracts, not proof of duplicate full papers |
| Licensing/provenance | Not established by the checked-in manifest; do not redistribute until verified |

The hash count is an indication for duplicate review, not proof that all repeated records represent the same paper.

## Requirement gap report

| Requirement | Status and evidence | Severity | Recommended next action |
|---|---|---:|---|
| Authenticated analysis and tenant boundaries | **Improved:** unauthenticated comparison/trend/gap/idea/proposal/similarity endpoints previously accepted optional users; they now require auth. API/security unit tests pass. Live IDOR suite remains unrun. | High (residual integration proof) | Run `pytest -m integration` against the release stack and add authorization checks for every new route. |
| Search reliability and benchmark integrity | Historical `docs/API_BENCHMARK_RESULTS.json` records 46 HTTP 500 responses among 50 search requests while reporting 98% success; readiness also contains 503 responses in its status distribution. These historical figures were not reproduced. | High | Re-run with a defined success criterion, preserve request/error details safely, fix search failures, and regenerate the report. |
| Durable ingestion jobs | Worker uses Redis `RPUSH`/`BLPOP`; a job is removed before processing and has no acknowledgement or retry/dead-letter policy. A worker crash can lose a job. | High | Adopt a queue with acknowledgements/retries or implement a recoverable job state machine and crash-recovery tests. |
| Embedding quality | Observed local provider is `HashingEmbeddingProvider` with 384 dimensions. Feature hashing is deterministic but not semantic; SPECTER2 is not configured. | High | Add explicit model/version metadata and compare supported scientific embeddings on a held-out dataset. Reindex on model changes. |
| Evidence/citation integrity | Evidence support is heuristic. The local evaluation does not call an LLM or validate generated citations. | High | Validate citation IDs and coordinates against retrieved chunks; perform claim-level human evaluation. |
| Dataset quality and governance | Raw/processed records are parseable and have unique IDs; there are 73 repeated content hashes. Manifest lacks a verifiable license field. | Medium | Review duplicates, source provenance, license, timestamps, and update semantics before public redistribution. |
| API contracts | FastAPI routes and request models exist, but paths are `/api/...`, not `/api/v1/...`; several list/search operations lack documented pagination/filter contracts and typed response models. | Medium | Add a versioned router with compatibility/deprecation behavior and response schemas. |
| PDF processing | Upload, signature/structure validation, 15 MB API cap, PyMuPDF extraction, chunking, and background processing exist. OCR, extraction-quality diagnostics, and parser sandboxing are absent. | Medium | Add extraction diagnostics and resource isolation; make OCR an explicit optional capability. |
| Research intelligence | Analysis endpoints and pages exist; output quality depends on input abstracts/metadata and optional model providers. Corpus coverage and global novelty are not established. | Medium | Ground results in evidence passages, report missing fields, and label ideas/gaps as hypotheses. |
| Frontend honesty | The header previously claimed “Live Systems” without checking a service and login offered a demo account with no repository seeder. Status now polls readiness; the unsupported demo shortcut was removed. | Fixed in this change | Keep UI states tied to live API results and add browser tests. |
| Error information leakage | Readiness previously returned raw exception strings to clients. It now logs server-side and returns booleans. | Fixed in this change | Review other exception responses and logs; keep detailed errors out of client payloads. |
| CI and test coverage | CI now runs the offline suite and excludes explicitly marked infrastructure tests. Integration tests are not currently run by CI. | Medium | Add reproducible PostgreSQL/Redis/object-storage/API/worker integration CI. |
| Dependency reproducibility | JavaScript has a lockfile; Python dependencies are mostly ranges and no Python lockfile is checked in. CI and container Python versions differ. | Medium | Choose supported runtime versions and add a reproducible Python lock strategy. |
| Deployment and operations | Compose, Dockerfiles, health checks, and migrations exist. The production deployment, backup/restore, TLS, secrets, and service startup were not validated here. | High | Validate on a Docker host; document production-only configuration and run restore/deployment drills. |

## Architecture and maintainability observations

- The frontend is Next.js 15/React 19, not Vite; this is retained intentionally in [ADR 0001](ADR/0001-retain-existing-stack.md).
- `apps/api/app/main.py` combines route definitions, request contracts, and orchestration in a large module. Split by bounded context only after API behavior is covered by tests.
- Worker pipeline directories contain README placeholders; executable ingestion orchestration currently lives in `apps/worker/app/worker.py`.
- The Redis queue is a simple list and has at-most-once removal semantics.
- The app has an Alembic history and pgvector models; migrations were not applied locally because PostgreSQL was unavailable.
- Browser screens build, but several analysis routes are thin wrappers around a shared tool component.
- The default corpus is local and Git-ignored, which prevents accidental large-file commits; its license is unresolved.

## Changed in this review

- Required authentication on six analytical endpoints and made the paper-access helper require a user.
- Sanitized readiness failure details while retaining server-side exception logs.
- Replaced the misleading “Live Systems” indicator with live readiness states.
- Removed the login shortcut for an account that is not provisioned in the repository.
- Reworked retrieval metric labels and documentation so synthetic heuristic results are not represented as generated-answer or citation quality.
- Added repository specification, implementation plan, test strategy, security notes, evaluation plan, and a stack-retention ADR.
