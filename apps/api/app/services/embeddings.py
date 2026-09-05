from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from hashlib import blake2b
from typing import Sequence

from apps.api.app.core.config import get_settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Fast, deterministic, offline feature-hashing embedding provider.

    Uses BLAKE2b with fixed digest for zero-dependency, cross-process consistency.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            position = (
                int.from_bytes(
                    blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
                )
                % self._dimension
            )
            vector[position] += 1.0

        magnitude = math.sqrt(sum(val * val for val in vector)) or 1.0
        return [val / magnitude for val in vector]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Transformer-based dense embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded sentence-transformer model '{self.model_name}' (dim={self._dimension})"
            )
        except Exception as exc:
            logger.warning(
                f"Could not load sentence-transformer model '{self.model_name}': {exc}. Falling back to Hashing."
            )
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        if self._model is not None:
            try:
                emb = self._model.encode(text, normalize_embeddings=True)
                return [float(x) for x in emb]
            except Exception as exc:
                logger.error(f"Inference error with transformer model: {exc}")
        # Fallback to hashing
        hashing = HashingEmbeddingProvider(self._dimension)
        return hashing.embed_text(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is not None:
            try:
                embeddings = self._model.encode(list(texts), normalize_embeddings=True)
                return [[float(x) for x in emb] for emb in embeddings]
            except Exception as exc:
                logger.error(f"Batch inference error with transformer model: {exc}")
        hashing = HashingEmbeddingProvider(self._dimension)
        return [hashing.embed_text(t) for t in texts]


@lru_cache(maxsize=1)
def get_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    provider_type = (settings.embedding_provider or "local").lower().strip()

    if (
        provider_type in ("sentence-transformers", "transformer", "huggingface")
        and settings.embedding_model
    ):
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )

    return HashingEmbeddingProvider(dimension=settings.embedding_dimension)


def embed(text: str, dimension: int = 384) -> list[float]:
    """Top-level convenience embedding function matching legacy signature."""
    provider = get_embedding_provider()
    if provider.dimension == dimension:
        return provider.embed_text(text)
    return HashingEmbeddingProvider(dimension=dimension).embed_text(text)
