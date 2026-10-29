# API Reference

Base URL: `http://127.0.0.1:8000`

The current API routes use `/api/...`; they are not yet versioned under `/api/v1`. FastAPI's generated schema is available at `/openapi.json` and the interactive UI at `/docs`.

## Health

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | No | Basic process health |
| `GET /liveness` | No | Process liveness |
| `GET /readiness` | No | Database, Redis, and object-storage readiness; returns 503 when degraded |

## Authentication

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /api/auth/register` | No | Create an account |
| `POST /api/auth/token` | No | OAuth2 password-form login |
| `GET /api/auth/me` | Bearer token | Return the authenticated user |

## Papers, search, and ingestion

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /api/search` | Optional for public discovery | Search configured scholarly providers and the local corpus |
| `GET /api/papers` | Bearer token | List public corpus papers and the user's papers |
| `GET /api/papers/{paper_id}` | Bearer token | Get an accessible paper |
| `POST /api/papers/upload` | Bearer token | Upload a PDF; current API limit is 15 MB |
| `DELETE /api/papers/{paper_id}` | Bearer token | Delete an owned private paper and related records |
| `GET /api/jobs/{job_id}` | Bearer token | Get the user's ingestion job state |

Search currently accepts `{"topic": "explainable AI", "full_text_only": false}`. It does not expose the full year/author/category filters, pagination, or stable sort contract from the target specification.

Uploads use multipart field `file`. The API validates the PDF and stores it through the configured storage backend, then enqueues asynchronous processing. The worker extracts text and indexes chunks. OCR is not provided.

## Q&A and conversations

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /api/qa` | Bearer token | Ask about an uploaded document using `document_id` and `question` |
| `POST /api/chat` | Optional for public papers; owner required for private papers | Ask about a paper and optionally persist the conversation |
| `GET /api/chat/{session_id}` | Bearer token | Read a conversation |
| `GET /api/papers/{paper_id}/chats` | Bearer token | List chats for an accessible paper |
| `DELETE /api/chat/{session_id}` | Bearer token | Delete an owned conversation |

Responses and citation fields depend on the available extracted evidence and model configuration. Heuristic evidence checks do not prove answer correctness.

## Collections

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /api/collections` | Bearer token | Create a collection |
| `GET /api/collections` | Bearer token | List the user's collections |
| `GET /api/collections/{collection_id}` | Bearer token | Get one owned collection and its papers |
| `PATCH /api/collections/{collection_id}` | Bearer token | Update an owned collection |
| `DELETE /api/collections/{collection_id}` | Bearer token | Delete an owned collection |
| `POST /api/collections/{collection_id}/papers` | Bearer token | Add an accessible paper |
| `DELETE /api/collections/{collection_id}/papers/{paper_id}` | Bearer token | Remove a paper from an owned collection |

## Research analysis

These endpoints require authentication and accept a topic plus a bounded list of paper inputs:

| Method and path | Purpose |
|---|---|
| `POST /api/compare` | Compare selected papers |
| `POST /api/trends` | Summarize trends in selected papers |
| `POST /api/research-gaps` | Suggest candidate research gaps |
| `POST /api/research-ideas` | Suggest candidate research directions |
| `POST /api/proposals` | Generate a proposal draft |
| `POST /api/similarity-map` | Return similarity nodes and edges |

Treat model-generated outputs as hypotheses. They are limited to supplied metadata/abstracts and available model output; absence from the input is not proof of absence from the literature.

## AI settings

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /api/settings/verify-key` | Optional | Check a user-supplied provider key |
| `GET /api/settings/ai-status` | Optional | Report whether a server-side provider key is configured |

## Errors and limits

Request validation uses Pydantic. HTTP errors generally return FastAPI's `detail` shape; a single standardized error envelope and response schemas are not yet implemented. Expensive-operation rate limiting and API-level pagination are also outstanding.

For authorization expectations and current gaps, see [SECURITY.md](SECURITY.md) and [PROJECT_REVIEW.md](PROJECT_REVIEW.md).
