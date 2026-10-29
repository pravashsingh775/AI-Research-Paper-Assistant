# Test Strategy

## Test layers

- **Unit:** Pure retrieval, normalization, evidence-state, storage validation, provider normalization, and API contract behavior. These must not need live services.
- **Database integration:** Alembic schema, PostgreSQL text persistence, pgvector, and transaction recovery.
- **System integration:** Live API, PostgreSQL, Redis, object storage, and worker lifecycle, including authorization boundaries.
- **Frontend:** Production build/type validation plus browser tests for real API success, empty, loading, and error states.
- **Evaluation:** Separate retrieval ranking measurements from evidence-support heuristics and generated-answer/citation judgment.

## Commands

From the repository root:

```powershell
python -m pytest apps/api/tests/ -m "not integration" -v
npm run build
python scripts/evaluate_rag.py
```

The integration-marked suite requires the API and backing services to be running:

```powershell
docker compose -f infra/docker/compose.yml up -d --build --wait
python -m pytest apps/api/tests/ -m integration -v
```

Use the repository's migration procedure before running database integration tests. Do not interpret skipped integration tests as passing.

## Current verification (2026-10-29)

- Offline backend tests: **42 passed, 9 integration tests deselected**, 2 dependency deprecation warnings.
- Next.js production build: **passed** with the installed Node.js 26.5.0 runtime. The workflow targets Node.js 20.
- Retrieval evaluation: **ran** on the three synthetic papers and 22 source-defined queries; see [RAG_EVALUATION.md](RAG_EVALUATION.md).
- Docker-backed integration suite: **not run** because the Docker Desktop engine was unavailable.
- Frontend lint: **not configured for non-interactive use**; `next lint` opens the ESLint setup prompt.
- Ruff: focused ingestion/evaluation files and API import sorting pass. A broader check across changed Python files reports 81 findings, primarily FastAPI `B008` defaults, unused imports, and broad exception handling; Python lint is not clean.

## CI policy

The CI unit job excludes only tests explicitly marked `integration`. The integration mark means the test needs live infrastructure; it is not a skip policy for assertions that can run offline. Add an integration job with PostgreSQL, Redis, object storage, API, and worker once it can be made deterministic.
