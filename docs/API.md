# API Reference

Base URL: `http://127.0.0.1:8000`

## `POST /api/search`

Request: `{ "topic": "explainable AI" }`

Returns up to ten deduplicated papers from OpenAlex, optional Semantic Scholar, and the local corpus. Metadata is provider-sourced. Ranking is a system signal based on query term matches and citation count.

## `POST /api/papers/upload`

Multipart field: `file`. Accepts PDF files up to 15 MB. Extracts text in memory and returns structured analysis. Files are not written to disk in this development implementation.

## `POST /api/papers/{paper_id}/ask`

Request: `{ "question": "What dataset was used?" }`. Answers are constrained to extracted paper text. With `ANTHROPIC_API_KEY`, Claude receives the extracted context; without it, the response is explicitly marked as evidence-limited.

## `GET /health`

Returns `{ "status": "ok" }`.
