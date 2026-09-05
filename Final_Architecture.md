            # Lumen Research: Final System Architecture

            ## 1. System Purpose

            Lumen Research is an evidence-backed research workspace. It discovers papers from the local arXiv-derived corpus and scholarly providers, stores user-owned papers and conversations in PostgreSQL, indexes uploaded-paper chunks in pgvector, and exposes research tools through a Next.js application.

            The production rule is simple: the UI displays persisted or API-returned data only. AI-generated analysis is labelled as analysis, and evidence-limited responses explicitly state their limits.

            ## 2. Runtime Topology

            ```mermaid
            flowchart LR
                  Browser[Researcher browser] --> Web[Next.js web app\nlocalhost:3000]
                  Web -->|HTTP JSON / Bearer token| API[FastAPI API\nlocalhost:8000]
                  API --> DB[(PostgreSQL 16\n+ pgvector)]
                  API --> Redis[(Redis 7\nresearch-paper-jobs)]
                  Redis --> Worker[Async worker]
                  Worker --> DB
                  API -. optional .-> Anthropic[Anthropic Messages API]
                  API -. discovery .-> Scholarly[Scholarly providers]
                  Minio[MinIO object storage] -. reserved integration .-> API
            ```

            ### Services

            | Service | Responsibility | Persistence / boundary |
            |---|---|---|
            | `apps/web` | Search UI, authentication UX, paper workspace, collections, research tools | Browser token and UI state; no source-of-truth data |
            | `apps/api` | Auth, ownership checks, search, upload registration, analysis, chat, collections, tool endpoints | Async SQLAlchemy session to PostgreSQL; Redis enqueue |
            | `apps/worker` | Consume uploaded-paper jobs, extract/chunk/index/analyze documents | Redis consumer; writes job, document, chunks, and analysis records |
            | PostgreSQL + pgvector | Users, papers, documents, chunks, embeddings, analyses, chats, jobs, collections | Named Docker volume `postgres_data` |
            | Redis | Background job list `research-paper-jobs` | In-memory queue; jobs are persisted in PostgreSQL |
            | MinIO | Local object-storage service reserved by the Compose stack | Named Docker volume `minio_data`; current upload path stores extracted content in PostgreSQL |

            ## 3. Request and Data Flows

            ### Authentication

            ```mermaid
            sequenceDiagram
                  participant B as Browser
                  participant W as Next.js
                  participant A as FastAPI
                  participant P as PostgreSQL
                  B->>W: Submit email and password
                  W->>A: POST /api/auth/register or /api/auth/token
                  A->>P: Read/write users
                  A-->>W: JWT bearer access token
                  W-->>B: Store token for authenticated API requests
                  B->>A: Protected request with Authorization header
                  A->>P: Decode token and resolve user
            ```

            Passwords are hashed with `pwdlib`/Argon2. Protected resources verify both the token and the resource owner. The frontend clears the token and chat session on logout.

            ### Scholarly search

            ```mermaid
            flowchart TD
                  Q[Topic / research question] --> S[POST /api/search]
                  S --> Local[Read normalized local JSONL corpus]
                  S --> Provider[Optional scholarly provider lookup]
                  Local --> Rank[Term-based ranking and top-10 selection]
                  Provider --> Rank
                  Rank --> Metadata[Paper metadata + evidence-limited analysis]
                  Metadata --> UI[Ranked result cards]
            ```

            Search results are discovery records. A selected persisted paper can be loaded from `GET /api/papers/{paper_id}` for stored abstract, authors, metadata, and analysis.

            ### Private PDF ingestion

            ```mermaid
            sequenceDiagram
                  participant B as Browser
                  participant A as FastAPI
                  participant P as PostgreSQL
                  participant R as Redis
                  participant W as Worker
                  B->>A: POST /api/papers/upload (multipart PDF + JWT)
                  A->>P: Create Paper, Document, BackgroundJob
                  A->>R: RPUSH research-paper-jobs
                  A-->>B: paper_id + job_id + PENDING
                  W->>R: BLPOP research-paper-jobs
                  W->>P: Load document and job
                  W->>W: Extract PDF text with PyMuPDF
                  W->>W: Section-aware chunking and 384-dim embeddings
                  W->>P: Store DocumentChunk vectors
                  W->>W: Optional Anthropic analysis, otherwise evidence-limited analysis
                  W->>P: Store PaperAnalysis and COMPLETED job
                  B->>A: Poll GET /api/jobs/{job_id}
                  A-->>B: Progress, failure, or completed analysis
            ```

            The worker uses a persistent blocking Redis read and reconnects after Redis errors. The UI must keep uploads and Q&A visibly in a processing state until the job reports `COMPLETED`.

            ### Retrieval-augmented Q&A

            ```mermaid
            flowchart LR
                  Question[Question + paper_id + optional session_id] --> Auth[JWT + paper ownership]
                  Auth --> Vector[Embed question]
                  Vector --> PG[pgvector cosine search\nup to 30 candidate chunks]
                  PG --> Rerank[CrossEncoder rerank\nor explicit fallback]
                  Rerank --> Context[Selected chunks with page/section]
                  Context --> Model[Optional Anthropic answer\nor local evidence fallback]
                  Model --> Cite[Citations: page, section, chunk]
                  Cite --> Store[Store user and assistant messages]
                  Store --> Response[Answer + citations + retrieval metadata]
            ```

            Chat sessions are owned by the authenticated user and tied to a paper. PostgreSQL is the source of truth for chat history; browser state only remembers the current session identifier.

            ## 4. Relational Model

            ```mermaid
            erDiagram
                  USERS ||--o{ PAPERS : owns
                  USERS ||--o{ RESEARCH_COLLECTIONS : creates
                  USERS ||--o{ CHAT_SESSIONS : starts
                  USERS ||--o{ BACKGROUND_JOBS : submits
                  PAPERS ||--o{ DOCUMENTS : contains
                  PAPERS ||--o{ PAPER_ANALYSES : receives
                  PAPERS ||--o{ CHAT_SESSIONS : discusses
                  PAPERS ||--o{ COLLECTION_PAPERS : included
                  RESEARCH_COLLECTIONS ||--o{ COLLECTION_PAPERS : contains
                  DOCUMENTS ||--o{ DOCUMENT_CHUNKS : splits
                  CHAT_SESSIONS ||--o{ CHAT_MESSAGES : stores
            ```

            Important constraints:

            - `papers.owner_id` is nullable so public corpus papers and private uploads can coexist.
            - Private papers, collections, jobs, and chats are filtered by the authenticated user.
            - `document_chunks.embedding` is `vector(384)` in PostgreSQL.
            - Cascading foreign keys remove dependent documents, chunks, analyses, messages, and collection links with their parent records.
            - Alembic migration `20260901_initial` creates the schema and enables the `vector` extension.

            ## 5. Frontend Route Map

            | Route | Purpose | Backend contract |
            |---|---|---|
            | `/` | Discovery, ranked results, upload, analysis, Q&A | `/api/search`, `/api/papers/upload`, `/api/jobs/*`, `/api/chat` |
            | `/login`, `/register` | Token acquisition and account creation | `/api/auth/token`, `/api/auth/register` |
            | `/collections`, `/collections/[id]` | Collection CRUD and paper membership | `/api/collections*` |
            | `/compare` | Multi-paper comparison | `/api/compare` |
            | `/trends` | Trend analysis from selected records | `/api/trends` |
            | `/research-gaps` | Gap analysis | `/api/research-gaps` |
            | `/research-ideas` | Idea generation | `/api/research-ideas` |
            | `/proposal` | Proposal generation | `/api/proposals` |
            | `/similarity-map` | Similarity data from selected records | `/api/similarity-map` |

            The shared `AppHeader` exposes these routes and provides sign-in, account identity, and logout controls. `NEXT_PUBLIC_API_URL` selects the API origin; local development defaults to `http://localhost:8000`.

            ## 6. Deployment and Startup

            ```mermaid
            flowchart TD
                  Compose[docker compose up -d --build --wait] --> PGH[PostgreSQL healthcheck]
                  Compose --> RH[Redis healthcheck]
                  PGH --> Migration[Backend: alembic upgrade head]
                  RH --> Migration
                  Migration --> APIH[Backend /health]
                  APIH --> Frontend[Frontend starts]
                  PGH --> Worker[Worker starts]
                  RH --> Worker
            ```

            Local commands:

            ```powershell
            docker compose -f infra/docker/compose.yml up -d --build --wait
            docker compose -f infra/docker/compose.yml ps
            python -m pytest -q
            python -m compileall apps scripts
            npm run build
            ```

            Runtime evidence checks:

            ```powershell
            Invoke-WebRequest http://localhost:8000/health
            docker compose -f infra/docker/compose.yml exec redis redis-cli ping
            docker compose -f infra/docker/compose.yml exec postgres pg_isready -U research -d research
            docker compose -f infra/docker/compose.yml exec backend alembic current
            docker compose -f infra/docker/compose.yml logs --tail=100 worker backend
            ```

            ## 7. Current Production Boundaries

            Implemented and runtime-verified in the local Compose environment:

            - Next.js production build and responsive shared navigation.
            - FastAPI health endpoint and authenticated registration flow.
            - PostgreSQL 16 with pgvector and Alembic head migration.
            - Redis health and worker startup/reconnection behavior.
            - Local 384-dimensional hashing embeddings and pgvector retrieval path.
            - Evidence-limited fallback analysis and optional Anthropic model calls.

            Still dependent on a complete browser-level acceptance run:

            - Full UI workflow from search through upload, worker completion, persistent Q&A, collections, and all research tools.
            - Real CrossEncoder model download and reranking in the target deployment environment.
            - MinIO-backed binary storage migration, if object storage becomes the required upload source of truth.

            This document describes the current code and runtime honestly. It does not treat an endpoint or page as verified merely because its source file exists.
                 




                 
cd "C:\Users\PRAVASH\Desktop\ai_research_paper_assistant"
docker compose -f infra/docker/compose.yml up -d --build --wait
docker compose -f infra/docker/compose.yml ps

| #  | Search topic                                                   | Good for testing                     |
| -- | -------------------------------------------------------------- | ------------------------------------ |
| 1  | **Retrieval Augmented Generation for Large Language Models**   | RAG, datasets, evaluation            |
| 2  | **Large Language Models for Code Generation**                  | benchmarks, models, performance      |
| 3  | **Vision Transformers for Image Classification**               | datasets, architecture, metrics      |
| 4  | **Medical Image Classification using Deep Learning**           | medical datasets, methodology        |
| 5  | **Federated Learning for Privacy-Preserving Machine Learning** | distributed datasets, privacy        |
| 6  | **Graph Neural Networks for Node Classification**              | graph datasets, algorithms           |
| 7  | **Multimodal Large Language Models**                           | vision-language datasets             |
| 8  | **Deep Learning for Fake News Detection**                      | NLP datasets, classification         |
| 9  | **Reinforcement Learning for Autonomous Driving**              | environments, algorithms, evaluation |
| 10 | **Transformer Models for Machine Translation**                 | datasets, BLEU, architectures        |
