# Implementation Plan

This plan reflects the inspected repository as of 2026-10-29. Status is based on code and executed checks, not prior report claims.

| Phase | Status | Exit criteria / next work |
|---|---|---|
| 1. Discovery and specification | Complete for this review | Inventory tracked code, inspect the existing corpus, document requirements and gaps. Revisit when the architecture or data source changes. |
| 2. Foundation | In progress | Keep the current app runnable; finish API contracts/versioning, configuration validation, consistent error handling, logging, and CI. Authentication enforcement and readiness error sanitization were hardened in this change. |
| 3. Dataset and ingestion | Partial | Add canonical duplicate handling and extraction-quality diagnostics; harden worker isolation and job lifecycle; evaluate scanned-PDF/OCR needs. |
| 4. Retrieval | Partial | Benchmark lexical, dense, hybrid, and reranked search against a held-out, human-reviewed dataset. Record the embedding model and prevent index/model incompatibility. |
| 5. RAG and citations | Partial | Validate every returned citation against retrieved chunk IDs and source coordinates; add human-judged answer/citation evaluation and explicit insufficient-evidence tests. |
| 6. Research intelligence | Partial | Ground comparisons and gap hypotheses in cited passages, mark inferred fields, and test missing-information behavior. |
| 7. Frontend integration | Partial | Add pagination/filter UX, source coverage and failure states, and browser-level tests. The readiness badge now reflects the API readiness endpoint; the nonexistent demo account shortcut was removed. |
| 8. Hardening | In progress | Continue security/privacy review, load tests, dependency and type checks, and integration tests against services. The current offline suite passes; service-backed tests are not run here. |
| 9. Deployment | Not verified | Start the full Compose stack on a Docker host, validate migrations/health/worker behavior, document secure production settings, backup/restore, and run a deployment smoke test. |
| 10. Portfolio evidence | Partial | Replace unsubstantiated benchmark and acceptance claims with reproducible evidence; add screenshots/demo only after capturing a real running deployment. |

## Immediate priority order

1. Make Redis jobs recoverable after worker crashes and define retry/dead-letter behavior.
2. Reproduce and resolve API search failures recorded in the existing benchmark artifact.
3. Add database-backed integration CI and repeat the tenant-isolation and ingestion lifecycle suite.
4. Version the API and add search/list pagination and filters without breaking existing clients.
5. Evaluate dense retrieval models and reranking using a held-out, licensed corpus.
6. Strengthen citation validation and user-visible evidence coverage before broadening model-generated analysis.

See [TEST_STRATEGY.md](TEST_STRATEGY.md), [SECURITY.md](SECURITY.md), and [EVALUATION_PLAN.md](EVALUATION_PLAN.md) for phase gates.
