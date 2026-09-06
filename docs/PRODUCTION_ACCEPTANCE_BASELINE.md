# Production Acceptance Baseline

**Project:** Lumen Research
**Date:** 2026-09-05
**Environment:** Windows 11 (Host) + WSL2 Docker Desktop (Engine v29.7.2)
**Python Runtime:** Python 3.10.0 (.venv) / Python 3.12 (Docker containers)
**Node Runtime:** Node.js v20+ / Next.js 15.5.25

---

## 1. Initial Test Suite Baseline

### Backend Tests (pytest)
- **Command:** `$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -m pytest apps/api/tests/ -v --durations=10`
- **Total Tests:** 31
- **Passed:** 31 (100%)
- **Failed:** 0
- **Errors:** 0
- **Skipped:** 0
- **Duration:** 3.51s
- **Warnings:** 2 (Starlette TestClient httpx deprecation warnings)
- **Slowest Tests:**
  - `apps/api/tests/test_security.py::test_authenticated_user_can_access_analytical_tools`: 0.09s
  - `apps/api/tests/test_security.py::test_analytical_tools_require_authentication`: 0.05s
  - `apps/api/tests/test_api.py::test_comparison_is_derived_from_supplied_papers`: 0.05s

### Frontend Build & Typecheck
- **Typecheck Command:** `npx tsc --noEmit` (in `apps/web`)
  - **Result:** Exit code 0 (0 errors, 0 warnings)
- **Build Command:** `npm run build` (in `apps/web`)
  - **Result:** 14 static pages generated cleanly in 3.1s:
    - `/`
    - `/_not-found`
    - `/collections`
    - `/collections/[id]`
    - `/compare`
    - `/login`
    - `/papers`
    - `/proposal`
    - `/register`
    - `/research-gaps`
    - `/research-ideas`
    - `/similarity-map`
    - `/trends`

---

## 2. Running Services Baseline

| Container | Image | Port | Initial Status |
|-----------|-------|------|----------------|
| `docker-postgres-1` | `pgvector/pgvector:pg16` | 5432 | Up (healthy) |
| `docker-redis-1` | `redis:7-alpine` | 6379 | Up (healthy) |
| `docker-minio-1` | `minio/minio:latest` | 9000, 9001 | Up |
| `docker-backend-1` | `docker-backend:latest` | 8000 | Up (healthy) |
| `docker-worker-1` | `docker-worker:latest` | - | Up |
| `docker-frontend-1` | `docker-frontend:latest` | 3000 | Up |

---

## 3. Initial Baseline Weaknesses Identified

1. **Missing Paper Deletion Endpoint:** No `DELETE /api/papers/{paper_id}` endpoint existed. Deleting a paper did not cascade to DB chunks, embeddings, chat history, or MinIO S3 objects.
2. **Missing Health Segregation:** Only `/health` existed. No `/liveness` (process-only) or `/readiness` (database, redis, storage check) endpoints.
3. **Correlation Tracking:** No request correlation ID middleware existed for tracing requests from API -> DB -> Redis -> Worker.
4. **Prompt Injection In Untrusted Papers:** Paper text was concatenated directly into LLM prompts without explicit `<untrusted_document_evidence>` XML boundaries or system guardrails against instruction overrides.
5. **LLM Retries & Backoff:** LLM calls had no bounded retry with exponential backoff for transient timeouts.
6. **Quantitative RAG Benchmarks:** RAG evaluation lacked automated Recall@K, MRR, nDCG, and abstention precision/recall calculation against a structured golden corpus.
7. **Container Images Outdated:** Docker container images were built before migration `20260903_storage_and_indexes` was added; containers need rebuilding.
