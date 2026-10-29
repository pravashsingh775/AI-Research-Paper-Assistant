# Local Setup

## Requirements

- Python 3.10 or later (the API container uses Python 3.12)
- Node.js 20 or later
- Docker Desktop / Docker Compose v2 for the complete stack
- Git

The local corpus is optional for startup and must be obtained separately. Its licensing is not documented in the repository; see [PROJECT_REVIEW.md](PROJECT_REVIEW.md).

## Start the complete development stack

From the repository root:

```powershell
Copy-Item .env.example .env
docker compose -f infra/docker/compose.yml up -d --build --wait
```

The API container runs Alembic migrations before starting Uvicorn. Check readiness and the generated API docs:

```powershell
Invoke-WebRequest http://localhost:8000/readiness
Start-Process http://localhost:8000/docs
Start-Process http://localhost:3000
```

The Compose file contains development-only credentials and is not a production deployment configuration. Change secrets before exposing any service.

## Run services locally without containerizing the API

Start PostgreSQL with pgvector, Redis, and an S3-compatible service first. Install the project and web dependencies:

```powershell
python -m pip install -e ".[dev]"
npm ci --prefix apps/web
```

Configure `.env` or process environment variables with host-reachable service endpoints. Run the migrations, API, worker, and web app in separate terminals:

```powershell
alembic upgrade head
python -m uvicorn apps.api.app.main:app --reload --port 8000
python -m apps.worker.app.worker
npm run dev:web
```

For local development, use a unique `JWT_SECRET` and `APP_ENV=development`. Provider API keys are optional; model-backed features need a supported provider key. Do not put keys in frontend build-time variables.

## Data ingestion

The source CSV is expected at `data/raw/corpus/arxiv_scientific_dataset.csv`. To create the normalized JSONL when no output exists:

```powershell
python scripts/ingest_arxiv_csv.py data\raw\corpus\arxiv_scientific_dataset.csv data\processed\arxiv_papers.jsonl
```

The importer refuses to replace an existing output unless `--overwrite` is explicitly supplied. It writes to a temporary file and atomically publishes the destination only after successful ingestion.

## Validation

```powershell
python -m pytest apps/api/tests/ -m "not integration" -v
npm run build
python scripts/evaluate_rag.py
```

The integration-marked API/database/worker tests need the live stack. See [TEST_STRATEGY.md](TEST_STRATEGY.md).

## Stop services

```powershell
docker compose -f infra/docker/compose.yml down
```

Omit `-v` to preserve the database and object-storage volumes.
