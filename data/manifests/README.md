Store import manifests, source CSV versions, record counts, and checksums here. Do not commit corpus files, PDFs, extracted text, embeddings, or secrets.

Expected local data layout:

```text
data/
	raw/corpus/       # Original CSV dataset
	staging/          # Temporary validation/import files
	processed/        # Normalized JSONL for indexing
	manifests/        # Metadata and checksums
```
