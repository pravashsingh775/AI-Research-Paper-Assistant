# Discovery pipeline

Stages: source ingestion -> normalization -> identifier resolution -> deduplication -> chunking -> embeddings -> indexing. Each stage should be idempotent and emit metrics.
