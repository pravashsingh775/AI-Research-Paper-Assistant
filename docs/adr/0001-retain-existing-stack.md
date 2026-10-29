# ADR 0001: Retain the Existing Application Stack

- **Status:** Accepted for the current implementation
- **Date:** 2026-10-29

## Context

The project already has a working Next.js 15/React 19 frontend, FastAPI backend, SQLAlchemy/Alembic data layer, PostgreSQL/pgvector, Redis queue, and Python worker. The target prompt lists Vite as a frontend baseline, but also says to preserve useful existing work and justify changes against the repository.

## Decision

Keep the current Next.js application and API/worker layout. Do not replace the frontend with Vite or perform a broad structural rewrite as part of this review. Keep existing API paths compatible while documenting that `/api/v1` versioning is still missing.

## Consequences

- Existing pages, styling, API client, Dockerfiles, and lockfile remain usable.
- The product avoids a risky framework migration unrelated to resolving current correctness gaps.
- Next.js-specific server/client behavior remains part of the stack and must be documented.
- The API still needs an explicit versioning plan; existing `/api` routes must be migrated compatibly.
- Missing baseline items such as TanStack Query, a frontend lint setup, and API response schemas remain tracked gaps rather than being added without demonstrated need.
