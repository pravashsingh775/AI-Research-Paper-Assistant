"""
Model-independent embedding contract for AI Research Paper Assistant.

This module is the abstraction layer only. It deliberately contains no
transformer/model loading, tokenization, SPECTER2 adapters, FAISS, reranking,
PDF parsing, RAG, LLM, summarization, or paper-analysis logic.

Concrete implementations (for example SPECTER2) implement EmbeddingEncoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np


class EmbeddingError(RuntimeError):
    """Base exception for embedding-interface failures."""


class InvalidEmbeddingInputError(EmbeddingError, ValueError):
    """Raised when query/document input violates the embedding contract."""


class EmbeddingValidationError(EmbeddingError, ValueError):
    """Raised when an embedding output violates the contract."""


class EmbeddingDimensionError(EmbeddingValidationError):
    """Raised when embedding dimensionality is inconsistent."""


@dataclass(frozen=True)
class EmbeddingInfo:
    """Minimal reproducibility metadata exposed by an encoder."""

    model_name: str
    embedding_dimension: int
    model_version: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    normalized: Optional[bool] = None


class EmbeddingEncoder(ABC):
    """
    Stable, model-independent embedding interface.

    Contract:
        embed_query(query) -> (dimension,)
        embed_document(document) -> (dimension,)
        embed_documents(documents) -> (n, dimension)

    Implementations must preserve input order:
        documents[i] <-> embeddings[i]
    """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed one non-empty search query."""
        raise NotImplementedError

    @abstractmethod
    def embed_document(self, document: str) -> np.ndarray:
        """Embed one non-empty document or chunk."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """Embed multiple documents/chunks while preserving input order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the verified vector dimensionality."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the concrete model/checkpoint identifier."""
        raise NotImplementedError

    @property
    def model_version(self) -> Optional[str]:
        """Return a verified model revision/version, if available."""
        return None

    @property
    def device(self) -> Optional[str]:
        """Return inference device, if known."""
        return None

    @property
    def dtype(self) -> Optional[str]:
        """Return output/model dtype, if known."""
        return None

    @property
    def normalized(self) -> Optional[bool]:
        """
        Return whether the implementation guarantees L2-normalized vectors.

        None means the implementation does not declare normalization.
        """
        return None

    def get_model_info(self) -> EmbeddingInfo:
        """Return lightweight, serializable model metadata."""
        return EmbeddingInfo(
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            model_version=self.model_version,
            device=self.device,
            dtype=self.dtype,
            normalized=self.normalized,
        )

    # ------------------------------------------------------------------
    # Input validation shared by concrete implementations
    # ------------------------------------------------------------------

    @staticmethod
    def validate_query(query: str) -> str:
        """Validate and strip a query without changing its semantics."""
        if query is None:
            raise InvalidEmbeddingInputError("Query cannot be None.")
        if not isinstance(query, str):
            raise InvalidEmbeddingInputError(
                f"Query must be a string, got {type(query).__name__}."
            )

        cleaned = query.strip()
        if not cleaned:
            raise InvalidEmbeddingInputError(
                "Query cannot be empty or whitespace-only."
            )
        return cleaned

    @staticmethod
    def validate_document(
        document: str,
        *,
        index: Optional[int] = None,
    ) -> str:
        """Validate and strip one document/chunk."""
        location = f" at index {index}" if index is not None else ""

        if document is None:
            raise InvalidEmbeddingInputError(
                f"Document{location} cannot be None."
            )
        if not isinstance(document, str):
            raise InvalidEmbeddingInputError(
                f"Document{location} must be a string, "
                f"got {type(document).__name__}."
            )

        cleaned = document.strip()
        if not cleaned:
            raise InvalidEmbeddingInputError(
                f"Document{location} cannot be empty or whitespace-only."
            )
        return cleaned

    @classmethod
    def validate_documents(
        cls,
        documents: Sequence[str],
        *,
        allow_empty_collection: bool = False,
    ) -> tuple[str, ...]:
        """Validate a document sequence without allocating model tensors."""
        if documents is None:
            raise InvalidEmbeddingInputError(
                "Document collection cannot be None."
            )

        if isinstance(documents, (str, bytes)):
            raise InvalidEmbeddingInputError(
                "documents must be a sequence of strings. "
                "Use embed_document() for one document."
            )

        if not isinstance(documents, Sequence):
            raise InvalidEmbeddingInputError(
                "documents must be a sequence of strings."
            )

        if len(documents) == 0:
            if allow_empty_collection:
                return ()
            raise InvalidEmbeddingInputError(
                "Document collection cannot be empty."
            )

        return tuple(
            cls.validate_document(document, index=index)
            for index, document in enumerate(documents)
        )

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dimension_value(dimension: int) -> int:
        """Validate an embedding dimension independently of a model."""
        if (
            not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, bool)
            or int(dimension) <= 0
        ):
            raise EmbeddingDimensionError(
                f"Embedding dimension must be a positive integer, got {dimension!r}."
            )
        return int(dimension)

    def validate_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        expected_count: Optional[int] = None,
        expected_dimension: Optional[int] = None,
        require_finite: bool = True,
        require_float: bool = True,
        normalized: Optional[bool] = None,
        normalization_tolerance: float = 1e-4,
    ) -> np.ndarray:
        """
        Validate a 2-D embedding matrix.

        Shape must be:
            (n_documents, embedding_dimension)

        No vector is silently repaired or normalized here; this method only
        validates the concrete encoder's output.
        """
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingValidationError(
                "Embeddings must be a numpy.ndarray, "
                f"got {type(embeddings).__name__}."
            )

        if embeddings.ndim != 2:
            raise EmbeddingValidationError(
                "Embedding matrix must be 2-D; "
                f"got shape {embeddings.shape}."
            )

        expected_dim = (
            self.embedding_dimension
            if expected_dimension is None
            else expected_dimension
        )
        expected_dim = self._validate_dimension_value(expected_dim)

        if embeddings.shape[1] != expected_dim:
            raise EmbeddingDimensionError(
                f"Expected embedding dimension {expected_dim}, "
                f"got {embeddings.shape[1]}."
            )

        if expected_count is not None:
            if (
                not isinstance(expected_count, (int, np.integer))
                or isinstance(expected_count, bool)
                or int(expected_count) < 0
            ):
                raise EmbeddingValidationError(
                    "expected_count must be a non-negative integer."
                )

            if embeddings.shape[0] != int(expected_count):
                raise EmbeddingValidationError(
                    f"Expected {expected_count} rows, "
                    f"got {embeddings.shape[0]}."
                )

        if require_float and embeddings.dtype.kind != "f":
            raise EmbeddingValidationError(
                "Embedding output must have a floating-point dtype; "
                f"got {embeddings.dtype}."
            )

        if require_finite and not np.isfinite(embeddings).all():
            raise EmbeddingValidationError(
                "Embedding output contains NaN or infinite values."
            )

        check_normalization = (
            self.normalized if normalized is None else normalized
        )

        if check_normalization:
            if normalization_tolerance <= 0:
                raise EmbeddingValidationError(
                    "normalization_tolerance must be > 0."
                )

            if embeddings.shape[0] > 0:
                norms = np.linalg.norm(
                    embeddings.astype(np.float64, copy=False),
                    axis=1,
                )

                if not np.isfinite(norms).all():
                    raise EmbeddingValidationError(
                        "Embedding norms contain NaN or infinite values."
                    )

                if np.any(norms <= np.finfo(np.float64).eps):
                    raise EmbeddingValidationError(
                        "Embedding output contains a zero/near-zero vector."
                    )

                max_error = float(np.max(np.abs(norms - 1.0)))
                if max_error > normalization_tolerance:
                    raise EmbeddingValidationError(
                        "Embedding vectors are not L2-normalized within "
                        f"tolerance. Maximum error={max_error:.6g}; "
                        f"tolerance={normalization_tolerance:.6g}."
                    )

        return embeddings

    def validate_embedding(
        self,
        embedding: np.ndarray,
        *,
        require_finite: bool = True,
        require_float: bool = True,
        normalized: Optional[bool] = None,
        normalization_tolerance: float = 1e-4,
    ) -> np.ndarray:
        """Validate one 1-D embedding vector."""
        if not isinstance(embedding, np.ndarray):
            raise EmbeddingValidationError(
                "Embedding must be a numpy.ndarray, "
                f"got {type(embedding).__name__}."
            )

        if embedding.ndim != 1:
            raise EmbeddingValidationError(
                f"Single embedding must be 1-D; got {embedding.shape}."
            )

        self.validate_embeddings(
            embedding.reshape(1, -1),
            expected_count=1,
            expected_dimension=self.embedding_dimension,
            require_finite=require_finite,
            require_float=require_float,
            normalized=normalized,
            normalization_tolerance=normalization_tolerance,
        )
        return embedding

    # ------------------------------------------------------------------
    # Optional convenience wrappers for concrete implementations
    # ------------------------------------------------------------------

    def encode_and_validate_query(self, query: str) -> np.ndarray:
        """
        Validate input, call the implementation, and validate its output.

        A concrete implementation should call this only if its public
        embed_query() method does not itself perform the same validation.
        """
        cleaned = self.validate_query(query)
        result = self.embed_query(cleaned)
        return self.validate_embedding(result)

    def encode_and_validate_document(self, document: str) -> np.ndarray:
        """Validate one document, invoke the implementation, validate output."""
        cleaned = self.validate_document(document)
        result = self.embed_document(cleaned)
        return self.validate_embedding(result)

    def encode_and_validate_documents(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:
        """Validate a batch, invoke the implementation, validate output."""
        cleaned = self.validate_documents(documents)
        result = self.embed_documents(cleaned)
        return self.validate_embeddings(
            result,
            expected_count=len(cleaned),
        )


class MockEmbeddingEncoder(EmbeddingEncoder):
    """
    Small deterministic fake encoder for tests.

    It is intentionally NOT semantic and must never be used for production
    retrieval. It allows FAISS/RAG/retrieval tests to run without downloading
    SPECTER2.
    """

    def __init__(
        self,
        embedding_dimension: int = 128,
        *,
        model_name: str = "mock-embedding",
        normalized: bool = True,
    ) -> None:
        self._embedding_dimension = self._validate_dimension_value(
            embedding_dimension
        )

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self._model_name = model_name.strip()
        self._normalized = bool(normalized)

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def normalized(self) -> bool:
        return self._normalized

    def _encode(self, text: str) -> np.ndarray:
        import hashlib

        cleaned = self.validate_document(text)

        seed = (
            f"{self._model_name}|{self._embedding_dimension}|{cleaned}"
        ).encode("utf-8")

        chunks: list[bytes] = []
        counter = 0
        required_bytes = self._embedding_dimension * 4

        while sum(len(chunk) for chunk in chunks) < required_bytes:
            chunks.append(
                hashlib.sha256(
                    seed + counter.to_bytes(4, byteorder="big")
                ).digest()
            )
            counter += 1

        raw = np.frombuffer(
            b"".join(chunks)[:required_bytes],
            dtype=np.uint32,
        )

        vector = (
            raw.astype(np.float64) / np.iinfo(np.uint32).max
        ) * 2.0 - 1.0

        if self._normalized:
            norm = np.linalg.norm(vector)
            if norm <= np.finfo(np.float64).eps:
                raise EmbeddingValidationError(
                    "Mock encoder generated a zero vector."
                )
            vector /= norm

        return vector.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        cleaned = self.validate_query(query)
        result = self._encode(cleaned)
        return self.validate_embedding(result)

    def embed_document(self, document: str) -> np.ndarray:
        cleaned = self.validate_document(document)
        result = self._encode(cleaned)
        return self.validate_embedding(result)

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        cleaned = self.validate_documents(documents)

        matrix = np.vstack(
            [self._encode(document) for document in cleaned]
        ).astype(np.float32, copy=False)

        return self.validate_embeddings(
            matrix,
            expected_count=len(cleaned),
        )


def smoke_test() -> None:
    """Run model-free tests for the complete embedding contract."""
    encoder = MockEmbeddingEncoder(embedding_dimension=128)

    query = encoder.embed_query("scientific information retrieval")
    document = encoder.embed_document("A paper about dense retrieval.")
    documents = encoder.embed_documents(
        [
            "Paper one about machine learning.",
            "Paper two about natural language processing.",
            "Paper three about information retrieval.",
        ]
    )

    assert query.shape == (128,)
    assert document.shape == (128,)
    assert documents.shape == (3, 128)

    assert np.isfinite(query).all()
    assert np.isfinite(document).all()
    assert np.isfinite(documents).all()

    assert np.allclose(np.linalg.norm(query), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(document), 1.0, atol=1e-4)
    assert np.allclose(
        np.linalg.norm(documents, axis=1),
        1.0,
        atol=1e-4,
    )

    # Invalid query.
    try:
        encoder.embed_query("   ")
    except InvalidEmbeddingInputError:
        pass
    else:
        raise AssertionError("Whitespace-only query was not rejected.")

    # Invalid batch.
    try:
        encoder.embed_documents([])
    except InvalidEmbeddingInputError:
        pass
    else:
        raise AssertionError("Empty document collection was not rejected.")

    # Wrong dimension.
    try:
        encoder.validate_embeddings(
            np.zeros((2, 127), dtype=np.float32),
            expected_count=2,
        )
    except EmbeddingDimensionError:
        pass
    else:
        raise AssertionError("Wrong embedding dimension was not rejected.")

    # NaN.
    bad = np.zeros((1, 128), dtype=np.float32)
    bad[0, 0] = np.nan

    try:
        encoder.validate_embeddings(bad)
    except EmbeddingValidationError:
        pass
    else:
        raise AssertionError("NaN vector was not rejected.")

    print("EmbeddingEncoder smoke test: PASSED")


if __name__ == "__main__":
    smoke_test()