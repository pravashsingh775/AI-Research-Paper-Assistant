# Worker service

The worker entry point is `app/worker.py`; the tracked `pipelines/` and `adapters/` directories contain documentation placeholders, not separate executable modules.

Start the consumer with `python -m apps.worker.app.worker`. It reads `research-paper-jobs` from Redis and uses PostgreSQL job/document records as the source of truth. The current Redis list consumer removes jobs before processing and does not provide acknowledgement/retry semantics; see [the project review](../../docs/PROJECT_REVIEW.md).
