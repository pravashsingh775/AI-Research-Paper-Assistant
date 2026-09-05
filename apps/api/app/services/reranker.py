from __future__ import annotations

from functools import lru_cache

from apps.api.app.core.config import get_settings
from apps.api.app.services.rag import Chunk


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    # A request must not block while downloading a model. Deployments that want
    # cross-encoder reranking can pre-bake/cache the configured model; all
    # others use the explicit hybrid fallback immediately.
    return CrossEncoder(get_settings().reranker_model, local_files_only=True)


def rerank(query: str, chunks: list[Chunk], limit: int = 6) -> tuple[list[Chunk], str]:
    if not chunks:
        return [], "none"
    try:
        scores = _model().predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda item: float(item[0]), reverse=True)
        return [chunk for _, chunk in ranked[:limit]], "cross-encoder"
    except (ImportError, OSError, RuntimeError, ValueError):
        return chunks[:limit], "hybrid-fallback"
