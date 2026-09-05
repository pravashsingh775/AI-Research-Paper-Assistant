# Worker service

Runs queue-backed jobs. `pipelines/` contains domain transformations; `adapters/` contains provider, storage, embedding, vector, and model integrations.

The development consumer is started with `python -m apps.worker.app.worker`. It consumes `research-paper-jobs` from Redis and uses PostgreSQL job/document records as the source of truth.
