Background jobs should be thin orchestrators. Put business logic in the pipeline modules and keep adapters responsible for external systems.

Planned jobs:

- `ingest_corpus`: import, normalize, deduplicate, and index corpus records.
- `discover_sources`: query scholarly providers and persist normalized results.
- `analyze_paper`: extract, sectionize, chunk, embed, and index an uploaded PDF.
- `generate_answer`: retrieve, rerank, generate, and validate cited output.
