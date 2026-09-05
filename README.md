# AI Research Paper Assistant

A research discovery and paper analysis platform built around evidence-backed retrieval.

## Architecture

- `apps/web`: Next.js user interface for discovery, uploads, summaries, and cited Q&A.
- `apps/api`: FastAPI application API, authentication, persistence, and orchestration.
- `apps/worker`: asynchronous ingestion, indexing, PDF analysis, generation, and validation jobs.
- `packages/contracts`: shared request, response, event, and citation schemas.
- `packages/prompts`: versioned prompts and model configuration.
- `data`: local development data only; production corpus and PDFs belong in object storage.
  - `data/raw/corpus/`: source datasets, including the supplied CSV.
  - `data/staging/`: temporary validation and import files.
  - `data/processed/`: normalized JSONL generated from the CSV for downstream indexing.
  - `data/manifests/`: dataset metadata, row counts, and checksums.
- `infra`: local services, database migrations, and observability configuration.

## Core flow

1. Ingest local corpus and scholarly web results.
2. Normalize metadata and text, then deduplicate by DOI, identifiers, and content hash.
3. Create searchable chunks and embeddings in the retrieval index.
4. Upload a PDF, extract sections, chunk it, and index it in a private paper namespace.
5. Retrieve candidates, rerank them, and generate summaries or answers with page/section evidence.
6. Validate every generated claim against retrieved evidence before returning it.

## Development

Copy `.env.example` to `.env`, start dependencies with `docker compose -f infra/docker/compose.yml up -d`, then run the API and web app:

```powershell
python -m uvicorn apps.api.app.main:app --reload --port 8000
npm run dev:web
```

After starting PostgreSQL, initialize the persistent schema:

```powershell
alembic upgrade head
```

For a stable production-like local run, build once and serve the compiled app with `npm run build --prefix apps/web; npm run start --prefix apps/web`. On Windows, avoid running `next build` while a webpack development server is active because both processes write `.next` artifacts.

Open `http://localhost:3000`. Set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL` to enable Claude-backed Q&A; without a key, search and structured extraction still work locally.

Set `ARXIV_DATASET_PATH` to the supplied CSV location before running corpus ingestion. The source CSV is intentionally not committed to Git. Use `data/manifests/` for import manifests and checksums.

The ingestion flow keeps the original CSV in `data/raw/corpus/` and writes the normalized record stream to `data/processed/`:

```powershell
python scripts/ingest_arxiv_csv.py data/raw/corpus/arxiv_scientific_dataset.csv data/processed/arxiv_papers.jsonl
```
