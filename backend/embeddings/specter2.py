"""
SPECTER2 scientific embedding layer for AI Research Paper Assistant.

Single responsibility:
    text -> SPECTER2 (+ task adapter) -> embedding matrix

The implementation intentionally contains no FAISS, ranking, reranking, LLM,
PDF parsing, summarization, or application/business logic.

Official SPECTER2 usage requires the AdapterHub `adapters` package. The
retrieval setup uses:
    base model:     allenai/specter2_base
    document:       allenai/specter2      (proximity adapter)
    short query:    allenai/specter2_adhoc_query

The official model card specifies title + separator + abstract input and
CLS/first-token pooling. It also distinguishes the proximity adapter for
paper-to-paper retrieval from the adhoc-query adapter for short text queries.

This module preserves input order exactly. Therefore, for a downstream corpus:
    paper_id[i] <-> embeddings[i]
remains valid as long as the caller supplies records in the desired order.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import platform
import random
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

try:
    from .encoder import EmbeddingEncoder
except ImportError:  # pragma: no cover - standalone execution
    from encoder import EmbeddingEncoder

# Transformers and AdapterHub are imported lazily.
# This prevents optional/native torchvision DLL problems from breaking a
# simple module import before the application starts.


LOGGER = logging.getLogger(__name__)


class SPECTER2Error(RuntimeError):
    """Base exception for SPECTER2 embedding failures."""


class SPECTER2ConfigurationError(SPECTER2Error):
    """Raised when embedder configuration is invalid."""


class SPECTER2InputError(SPECTER2Error):
    """Raised when embedding input is invalid."""


class SPECTER2ValidationError(SPECTER2Error):
    """Raised when generated embeddings fail validation."""


def _load_transformers():
    """Import Transformers only when a tokenizer is actually needed."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except Exception as exc:
        raise SPECTER2Error(
            "Could not import Hugging Face Transformers. "
            "Check the installed transformers/torch/torchvision versions."
        ) from exc


def _load_adapter_model_class():
    """Import AdapterHub only when SPECTER2 model loading is requested."""
    try:
        from adapters import AutoAdapterModel
        return AutoAdapterModel
    except Exception as exc:
        raise SPECTER2Error(
            "Could not import AdapterHub `adapters`. "
            "Check compatibility with the installed Transformers stack."
        ) from exc


@dataclass(frozen=True)
class SPECTER2Config:
    """Configuration for the SPECTER2 embedder."""

    base_model: str = "allenai/specter2_base"
    document_adapter: str = "allenai/specter2"
    query_adapter: str = "allenai/specter2_adhoc_query"

    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512
    normalize: bool = True

    # Storage/output dtype. Model inference is not forcibly quantized.
    output_dtype: str = "float32"

    # Optional Hugging Face cache location. None means standard HF caching.
    cache_dir: Optional[Union[str, Path]] = None

    # Download/load options.
    local_files_only: bool = False
    trust_remote_code: bool = False

    # Runtime behavior.
    show_progress: bool = True
    deterministic: bool = True

    # CUDA inference can use autocast when requested. Default is conservative:
    # False means full model dtype is used and only output is cast to float32.
    use_autocast: bool = False

    # Validation tolerance for normalized vectors.
    normalization_tolerance: float = 1e-4

    # Automatically reduce the runtime batch size after a CUDA OOM.
    # This is intentionally local to one call and does not mutate config.batch_size.
    adaptive_batching: bool = True

    # Smallest batch size permitted by adaptive CUDA OOM recovery.
    min_batch_size: int = 1

    # Release cached CUDA allocator blocks after an OOM before retrying.
    clear_cuda_cache_on_oom: bool = True


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Machine-readable metadata describing one embedder instance."""

    base_model: str
    document_adapter: str
    query_adapter: str
    device: str
    dtype: str
    embedding_dimension: int
    max_length: int
    batch_size: int
    normalize: bool
    output_dtype: str
    transformers_version: Optional[str]
    adapters_version: Optional[str]
    torch_version: str
    python_version: str
    platform: str


@dataclass(frozen=True)
class EmbeddingBatch:
    """
    Output of corpus-oriented embedding.

    `ids[i]` always corresponds to `embeddings[i]`.
    """

    ids: tuple[str, ...]
    embeddings: np.ndarray


Record = Mapping[str, Any]
TextInput = Union[str, Sequence[str]]


class SPECTER2Embedder(EmbeddingEncoder):
    """
    Production-oriented SPECTER2 inference wrapper.

    Two official task adapters are kept available:

    * proximity adapter (`allenai/specter2`) for scientific paper/document
      embeddings used as retrieval candidates.
    * adhoc-query adapter (`allenai/specter2_adhoc_query`) for short natural
      language search queries.

    The base model and adapters are loaded once and reused for subsequent
    requests.

    Notes:
        - `max_length` defaults to 512 because the official SPECTER2 model
          usage demonstrates tokenizer truncation at 512. It remains
          configurable.
        - Embeddings are pooled from `last_hidden_state[:, 0, :]`, matching
          the official model-card inference procedure.
        - Normalization is applied after pooling and before returning vectors.
        - No claim of 100% semantic accuracy is made; retrieval quality must
          be measured empirically with a held-out evaluation set.
    """

    @property
    def embedding_dimension(self) -> int:
        return int(self._embedding_dimension)

    @property
    def model_name(self) -> str:
        return self.config.base_model

    @property
    def model_version(self) -> Optional[str]:
        # The configured checkpoint identifier is the strongest stable
        # identifier available without making an additional Hub request.
        return self.config.base_model

    @property
    def device(self):
        return self._device

    @property
    def dtype(self) -> str:
        return str(next(self.model.parameters()).dtype)

    @property
    def normalized(self) -> bool:
        return bool(self.config.normalize)

    @property
    def device_info(self) -> str:
        return str(self.device)

    def __init__(
        self,
        *,
        base_model: str = "allenai/specter2_base",
        document_adapter: str = "allenai/specter2",
        query_adapter: str = "allenai/specter2_adhoc_query",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True,
        output_dtype: str = "float32",
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        show_progress: bool = True,
        deterministic: bool = True,
        use_autocast: bool = False,
        normalization_tolerance: float = 1e-4,
        adaptive_batching: bool = True,
        min_batch_size: int = 1,
        clear_cuda_cache_on_oom: bool = True,
    ) -> None:
        self.config = SPECTER2Config(
            base_model=base_model,
            document_adapter=document_adapter,
            query_adapter=query_adapter,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            output_dtype=output_dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            show_progress=show_progress,
            deterministic=deterministic,
            use_autocast=use_autocast,
            normalization_tolerance=normalization_tolerance,
            adaptive_batching=adaptive_batching,
            min_batch_size=min_batch_size,
            clear_cuda_cache_on_oom=clear_cuda_cache_on_oom,
        )

        self._validate_config()

        self._device = self._resolve_device(self.config.device)
        self._configure_reproducibility()

        # AdapterHub changes the active adapter on the model. Protect the
        # adapter switch + forward pass from concurrent requests.
        self._inference_lock = threading.RLock()

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        self._embedding_dimension = self._infer_embedding_dimension()

        self._validate_model_runtime()

        LOGGER.info("SPECTER2 embedder initialized.")
        LOGGER.info("Base model: %s", self.config.base_model)
        LOGGER.info("Document adapter: %s", self.config.document_adapter)
        LOGGER.info("Query adapter: %s", self.config.query_adapter)
        LOGGER.info("Device: %s", self.device)
        LOGGER.info("Embedding dimension: %d", self.embedding_dimension)
        LOGGER.info("Max length: %d", self.config.max_length)
        LOGGER.info("Batch size: %d", self.config.batch_size)
        LOGGER.info("Normalize: %s", self.config.normalize)
        LOGGER.info("Output dtype: %s", self.config.output_dtype)

    # ------------------------------------------------------------------
    # Configuration / model loading
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not isinstance(self.config.base_model, str) or not self.config.base_model.strip():
            raise SPECTER2ConfigurationError("base_model must be a non-empty string.")

        if not isinstance(self.config.document_adapter, str) or not self.config.document_adapter.strip():
            raise SPECTER2ConfigurationError(
                "document_adapter must be a non-empty string."
            )

        if not isinstance(self.config.query_adapter, str) or not self.config.query_adapter.strip():
            raise SPECTER2ConfigurationError(
                "query_adapter must be a non-empty string."
            )

        if self.config.device not in {"auto", "cpu", "cuda"}:
            raise SPECTER2ConfigurationError(
                "device must be one of: 'auto', 'cpu', 'cuda'."
            )

        if self.config.batch_size < 1:
            raise SPECTER2ConfigurationError("batch_size must be >= 1.")

        if self.config.max_length < 2:
            raise SPECTER2ConfigurationError("max_length must be >= 2.")

        if self.config.output_dtype not in {"float32", "float64"}:
            raise SPECTER2ConfigurationError(
                "output_dtype must be 'float32' or 'float64'. "
                "float32 is recommended for FAISS/storage."
            )

        if not math.isfinite(float(self.config.normalization_tolerance)):
            raise SPECTER2ConfigurationError(
                "normalization_tolerance must be finite."
            )

        if self.config.normalization_tolerance <= 0:
            raise SPECTER2ConfigurationError(
                "normalization_tolerance must be > 0."
            )

        if not isinstance(self.config.adaptive_batching, bool):
            raise SPECTER2ConfigurationError(
                "adaptive_batching must be a boolean."
            )

        if self.config.min_batch_size < 1:
            raise SPECTER2ConfigurationError(
                "min_batch_size must be >= 1."
            )

        if not isinstance(self.config.clear_cuda_cache_on_oom, bool):
            raise SPECTER2ConfigurationError(
                "clear_cuda_cache_on_oom must be a boolean."
            )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            LOGGER.warning("CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise SPECTER2ConfigurationError(
                    "device='cuda' was requested, but CUDA is unavailable."
                )
            return torch.device("cuda")

        return torch.device("cpu")

    def _configure_reproducibility(self) -> None:
        if not self.config.deterministic:
            return

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms can reduce performance and may be
        # unsupported by some kernels. We intentionally do not force
        # torch.use_deterministic_algorithms(True) because inference does
        # not justify breaking otherwise-valid kernels.
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            pass

    def _hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = str(self.config.cache_dir)
        return kwargs

    def _load_tokenizer(self):
        LOGGER.info("Loading SPECTER2 tokenizer...")
        AutoTokenizer = _load_transformers()
        try:
            return AutoTokenizer.from_pretrained(
                self.config.base_model,
                **self._hf_kwargs(),
            )
        except Exception as exc:
            raise SPECTER2Error(
                f"Failed to load tokenizer '{self.config.base_model}'. "
                "Check the model name, Hugging Face cache/network access, "
                "and installed transformers version."
            ) from exc

    def _load_model(self):
        LOGGER.info("Loading SPECTER2 base model...")
        AutoAdapterModel = _load_adapter_model_class()
        try:
            model = AutoAdapterModel.from_pretrained(
                self.config.base_model,
                **self._hf_kwargs(),
            )

            LOGGER.info("Loading document/proximity adapter...")
            document_name = model.load_adapter(
                self.config.document_adapter,
                source="hf",
                load_as="specter2_document",
                set_active=False,
            )

            LOGGER.info("Loading adhoc-query adapter...")
            query_name = model.load_adapter(
                self.config.query_adapter,
                source="hf",
                load_as="specter2_query",
                set_active=False,
            )

            self._document_adapter_name = document_name
            self._query_adapter_name = query_name

            model.to(self.device)
            model.eval()

            return model

        except TypeError:
            # Some adapters versions do not accept every optional Hub
            # keyword in exactly the same combination. Retry only without
            # `source`, while preserving the official load_adapter API.
            try:
                model = AutoAdapterModel.from_pretrained(
                    self.config.base_model,
                    **self._hf_kwargs(),
                )
                document_name = model.load_adapter(
                    self.config.document_adapter,
                    load_as="specter2_document",
                    set_active=False,
                )
                query_name = model.load_adapter(
                    self.config.query_adapter,
                    load_as="specter2_query",
                    set_active=False,
                )
                self._document_adapter_name = document_name
                self._query_adapter_name = query_name
                model.to(self.device)
                model.eval()
                return model
            except Exception as exc:
                raise SPECTER2Error(
                    "Failed to load SPECTER2 base model/adapters. "
                    "Verify compatible versions of 'adapters', "
                    "'transformers', and the official checkpoints."
                ) from exc
        except Exception as exc:
            raise SPECTER2Error(
                "Failed to load SPECTER2 base model/adapters. "
                "Verify compatible versions of 'adapters', "
                "'transformers', and the official checkpoints."
            ) from exc

    def _validate_model_runtime(self) -> None:
        if not hasattr(self.model, "set_active_adapters"):
            raise SPECTER2Error(
                "Loaded model does not expose AdapterHub's "
                "set_active_adapters() API."
            )

        if not hasattr(self.model, "get_input_embeddings"):
            raise SPECTER2Error(
                "Loaded SPECTER2 model is missing expected transformer APIs."
            )

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration as exc:
            raise SPECTER2Error("Loaded model contains no parameters.") from exc

        if model_device.type != self.device.type:
            raise SPECTER2Error(
                f"Model device mismatch: expected {self.device}, got {model_device}."
            )

    def _infer_embedding_dimension(self) -> int:
        config = getattr(self.model, "config", None)

        candidates = [
            getattr(config, "hidden_size", None),
            getattr(config, "dim", None),
            getattr(config, "d_model", None),
        ]

        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value

        # Robust fallback: infer from one tiny real forward pass.
        try:
            test = self._encode_texts(
                ["A scientific paper about machine learning."],
                adapter=self._document_adapter_name,
                validate=True,
            )
            if test.ndim == 2 and test.shape[1] > 0:
                return int(test.shape[1])
        except Exception as exc:
            raise SPECTER2Error(
                "Could not determine SPECTER2 embedding dimensionality."
            ) from exc

        raise SPECTER2Error("Invalid embedding dimension.")

    # ------------------------------------------------------------------
    # Canonical text construction
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        # Avoid treating NaN/Inf as scientific text.
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return ""

        text = str(value)
        text = " ".join(text.split())
        return text.strip()

    @classmethod
    def build_paper_text(
        cls,
        title: Any,
        abstract: Any,
        *,
        separator: str = " [SEP] ",
    ) -> str:
        """
        Build canonical title+abstract text.

        The semantic content intentionally excludes IDs, dates, categories,
        filesystem paths, ranking scores, and other arbitrary metadata.

        The default separator is replaced by the tokenizer's actual
        `sep_token` in `_build_document_inputs`, because the official
        SPECTER2 procedure uses the tokenizer separator token.
        """
        title_text = cls._clean_text(title)
        abstract_text = cls._clean_text(abstract)

        if title_text and abstract_text:
            return f"{title_text}{separator}{abstract_text}"
        if title_text:
            return title_text
        return abstract_text

    def _build_document_inputs(
        self,
        title_abstract_pairs: Sequence[tuple[Any, Any]],
    ) -> list[str]:
        sep = self.tokenizer.sep_token or " [SEP] "
        return [
            self.build_paper_text(title, abstract, separator=sep)
            for title, abstract in title_abstract_pairs
        ]

    @classmethod
    def _validate_text_sequence(
        cls,
        texts: Sequence[Any],
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise SPECTER2InputError(
                "Expected a sequence of text strings, not one raw string."
            )

        try:
            values = list(texts)
        except TypeError as exc:
            raise SPECTER2InputError(
                "texts must be a string or a sequence of strings."
            ) from exc

        cleaned = [cls._clean_text(x) for x in values]

        if not allow_empty and any(not x for x in cleaned):
            bad = next(i for i, x in enumerate(cleaned) if not x)
            raise SPECTER2InputError(
                f"Input at index {bad} is empty after normalization."
            )

        return cleaned

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed one short research query using the official adhoc-query adapter.

        Returns:
            numpy.ndarray with shape `(embedding_dimension,)`.
        """
        cleaned = self._clean_text(query)
        if not cleaned:
            raise SPECTER2InputError("Query must contain non-whitespace text.")

        result = self._encode_texts(
            [cleaned],
            adapter=self._query_adapter_name,
            validate=True,
        )
        return result[0]

    def embed_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed multiple short research queries with the adhoc-query adapter.

        Input order is preserved exactly. This method is useful for evaluation,
        batch search, and offline retrieval benchmarks.
        """
        cleaned = self._validate_text_sequence(queries)
        if not cleaned:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=self._numpy_dtype,
            )

        return self._encode_texts(
            cleaned,
            adapter=self._query_adapter_name,
            batch_size=batch_size,
            validate=True,
        )

    def embed_document(self, document: Any, abstract: Any = "") -> np.ndarray:
        """Embed one document/chunk using the SPECTER2 document adapter.

        Backward compatibility:
            embed_document(title, abstract) -> canonical title+SEP+abstract

        Mode-2 compatibility:
            embed_document(chunk_text) -> chunk text directly

        The latter is important because ``SemanticChunk.text`` is the primary
        semantic input for uploaded-paper indexing. IDs and metadata are never
        embedded.
        """
        if abstract not in (None, ""):
            text = self._build_document_inputs([(document, abstract)])[0]
        else:
            text = self._clean_text(document)

        if not text:
            raise SPECTER2InputError(
                "Document/chunk must contain non-whitespace text."
            )

        result = self._encode_texts(
            [text],
            adapter=self._document_adapter_name,
            validate=True,
        )
        return result[0]

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Embed arbitrary document/chunk text using the proximity adapter.

        Input order is preserved exactly. This API is suitable for Mode 2
        chunks as well as already-built canonical document strings.

        Returns:
            float32/float64 numpy array of shape
            `(n_documents, embedding_dimension)`.
        """
        cleaned = self._validate_text_sequence(texts)
        if not cleaned:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=self._numpy_dtype,
            )

        return self._encode_texts(
            cleaned,
            adapter=self._document_adapter_name,
            batch_size=batch_size,
            validate=True,
        )

    def embed_documents_from_pairs(
        self,
        title_abstract_pairs: Sequence[tuple[Any, Any]],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Embed `(title, abstract)` pairs using the canonical document format.

        Input order is preserved exactly.
        """
        if isinstance(title_abstract_pairs, (str, bytes)):
            raise SPECTER2InputError(
                "title_abstract_pairs must be a sequence of (title, abstract) pairs."
            )

        pairs = list(title_abstract_pairs)
        texts = self._build_document_inputs(pairs)

        if any(not text for text in texts):
            bad = next(i for i, text in enumerate(texts) if not text)
            raise SPECTER2InputError(
                f"Paper at index {bad} has neither title nor abstract."
            )

        return self._encode_texts(
            texts,
            adapter=self._document_adapter_name,
            batch_size=batch_size,
            validate=True,
        )

    def embed_papers(
        self,
        records: Sequence[Record],
        *,
        id_field: str = "id",
        title_field: str = "title",
        abstract_field: str = "summary",
        batch_size: Optional[int] = None,
    ) -> EmbeddingBatch:
        """
        Embed paper records while explicitly preserving ID-to-row alignment.

        No sorting or reordering occurs.

        Result invariant:
            result.ids[i] corresponds exactly to
            result.embeddings[i].

        This helper is intentionally limited to in-memory record sequences.
        For the full 287K corpus, a separate streaming/chunked pipeline should
        read the processed dataset in batches and call this method per chunk.
        """
        if isinstance(records, (str, bytes)):
            raise SPECTER2InputError("records must be a sequence of mappings.")

        rows = list(records)
        ids: list[str] = []
        pairs: list[tuple[Any, Any]] = []

        for index, record in enumerate(rows):
            if not isinstance(record, Mapping):
                raise SPECTER2InputError(
                    f"Record {index} must be a mapping, got {type(record).__name__}."
                )

            if id_field not in record:
                raise SPECTER2InputError(
                    f"Record {index} is missing required ID field '{id_field}'."
                )

            raw_id = self._clean_text(record[id_field])
            if not raw_id:
                raise SPECTER2InputError(
                    f"Record {index} has an empty '{id_field}'."
                )

            ids.append(raw_id)
            pairs.append(
                (
                    record.get(title_field, ""),
                    record.get(abstract_field, ""),
                )
            )

        embeddings = self.embed_documents_from_pairs(
            pairs,
            batch_size=batch_size,
        )

        if len(ids) != embeddings.shape[0]:
            raise SPECTER2ValidationError(
                "ID/embedding alignment invariant failed: "
                f"{len(ids)} IDs vs {embeddings.shape[0]} embeddings."
            )

        return EmbeddingBatch(
            ids=tuple(ids),
            embeddings=embeddings,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @property
    def _numpy_dtype(self) -> np.dtype:
        return np.dtype(self.config.output_dtype)

    def _activate_adapter(self, adapter_name: str) -> None:
        try:
            self.model.set_active_adapters(adapter_name)
        except Exception as exc:
            raise SPECTER2Error(
                f"Could not activate adapter '{adapter_name}'."
            ) from exc

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        adapter: str,
        batch_size: Optional[int] = None,
        validate: bool = True,
    ) -> np.ndarray:
        """Encode texts in bounded batches with optional CUDA OOM recovery.

        The caller's requested/configured batch size is only a starting point.
        If a CUDA batch runs out of memory and adaptive batching is enabled, the
        same batch is retried at half the size until it succeeds or reaches
        ``min_batch_size``. This is especially useful on consumer GPUs with
        limited VRAM and avoids requiring callers to hard-code one fragile
        batch size.
        """
        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=self._numpy_dtype,
            )

        requested_batch = batch_size or self.config.batch_size
        if requested_batch < 1:
            raise SPECTER2ConfigurationError("batch_size must be >= 1.")

        if self.config.min_batch_size > requested_batch:
            raise SPECTER2ConfigurationError(
                "min_batch_size cannot exceed the requested batch size."
            )

        cleaned = self._validate_text_sequence(texts)
        if not cleaned:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=self._numpy_dtype,
            )

        chunks: list[np.ndarray] = []
        runtime_batch = requested_batch

        progress = None
        if self.config.show_progress:
            try:
                from tqdm.auto import tqdm

                progress = tqdm(
                    total=len(cleaned),
                    desc="SPECTER2 embedding",
                    unit="doc",
                )
            except ImportError:
                progress = None

        try:
            start = 0
            while start < len(cleaned):
                current_end = min(start + runtime_batch, len(cleaned))
                batch_texts = cleaned[start:current_end]

                LOGGER.debug(
                    "Encoding batch %d-%d / %d with batch_size=%d",
                    start,
                    current_end - 1,
                    len(cleaned),
                    runtime_batch,
                )

                try:
                    vectors = self._encode_batch(
                        batch_texts,
                        adapter=adapter,
                    )
                except SPECTER2Error as exc:
                    if not (
                        self.device.type == "cuda"
                        and self.config.adaptive_batching
                        and self._is_cuda_oom_error(exc)
                        and runtime_batch > self.config.min_batch_size
                    ):
                        raise

                    new_batch = max(
                        self.config.min_batch_size,
                        runtime_batch // 2,
                    )
                    if new_batch == runtime_batch:
                        raise

                    LOGGER.warning(
                        "CUDA OOM at batch_size=%d; retrying with batch_size=%d.",
                        runtime_batch,
                        new_batch,
                    )
                    runtime_batch = new_batch
                    if self.config.clear_cuda_cache_on_oom:
                        self._clear_cuda_memory()
                    continue

                if validate:
                    self.validate_embeddings(
                        vectors,
                        expected_count=len(batch_texts),
                        expected_dimension=self.embedding_dimension,
                        normalized=self.config.normalize,
                    )

                chunks.append(vectors)
                start = current_end

                if progress is not None:
                    progress.update(len(batch_texts))

                # Do not call empty_cache after every batch: it can significantly
                # reduce throughput. Python references are enough for normal use.
                if self.device.type == "cuda":
                    del vectors

        finally:
            if progress is not None:
                progress.close()

        if not chunks:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=self._numpy_dtype,
            )

        result = np.concatenate(chunks, axis=0)

        if validate:
            self.validate_embeddings(
                result,
                expected_count=len(cleaned),
                expected_dimension=self.embedding_dimension,
                normalized=self.config.normalize,
            )

        return result

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        """Return True when an exception represents CUDA out-of-memory."""
        message = str(exc).lower()
        return (
            "cuda out of memory" in message
            or "out of memory" in message
            or "cublas_status_alloc_failed" in message
        )

    @staticmethod
    def _clear_cuda_memory() -> None:
        """Best-effort cleanup after a CUDA allocation failure."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            # Cleanup must never mask the original inference error.
            pass

    def _encode_batch(
        self,
        texts: Sequence[str],
        *,
        adapter: str,
    ) -> np.ndarray:
        # Adapter selection mutates model state, therefore it must be atomic
        # with the corresponding forward pass.
        with self._inference_lock:
            self._activate_adapter(adapter)
            return self._encode_batch_locked(texts)

    def _encode_batch_locked(self, texts: Sequence[str]) -> np.ndarray:
        try:
            encoded = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
        except Exception as exc:
            raise SPECTER2Error(
                "SPECTER2 tokenizer failed for an input batch."
            ) from exc

        encoded = {
            key: value.to(self.device, non_blocking=True)
            for key, value in encoded.items()
        }

        try:
            with torch.inference_mode():
                if (
                    self.device.type == "cuda"
                    and self.config.use_autocast
                ):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)

            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise SPECTER2Error(
                    "SPECTER2 model output does not contain "
                    "'last_hidden_state'."
                )

            if hidden.ndim != 3:
                raise SPECTER2ValidationError(
                    f"Expected hidden state rank 3, got shape {tuple(hidden.shape)}."
                )

            # Official SPECTER2 inference uses the first token (CLS) as the
            # document representation.
            pooled = hidden[:, 0, :]

            if pooled.ndim != 2:
                raise SPECTER2ValidationError(
                    f"Expected pooled shape (batch, dim), got {tuple(pooled.shape)}."
                )

            if not torch.isfinite(pooled).all().item():
                raise SPECTER2ValidationError(
                    "SPECTER2 produced NaN/Inf values before normalization."
                )

            if self.config.normalize:
                norms = torch.linalg.vector_norm(
                    pooled,
                    ord=2,
                    dim=1,
                    keepdim=True,
                )

                if not torch.isfinite(norms).all().item():
                    raise SPECTER2ValidationError(
                        "SPECTER2 produced non-finite vector norms."
                    )

                if torch.any(norms <= torch.finfo(pooled.dtype).eps).item():
                    raise SPECTER2ValidationError(
                        "SPECTER2 produced a zero/near-zero embedding vector."
                    )

                pooled = pooled / norms

            # Always detach before crossing from torch to numpy.
            result = pooled.detach().float().cpu().numpy()

            if self.config.output_dtype == "float64":
                result = result.astype(np.float64, copy=False)
            else:
                result = result.astype(np.float32, copy=False)

            return result

        except SPECTER2Error:
            raise
        except RuntimeError as exc:
            if self.device.type == "cuda" and self._is_cuda_oom_error(exc):
                raise SPECTER2Error(
                    "CUDA out-of-memory during SPECTER2 inference. "
                    "Adaptive batching may retry automatically; if it still fails, "
                    "reduce batch_size and/or max_length."
                ) from exc
            raise SPECTER2Error(
                "SPECTER2 model inference failed."
            ) from exc
        finally:
            # Explicitly release input tensors. Model remains resident.
            del encoded

    # ------------------------------------------------------------------
    # Validation / information
    # ------------------------------------------------------------------

    def validate_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        expected_count: Optional[int] = None,
        expected_dimension: Optional[int] = None,
        expected_rows: Optional[int] = None,
        normalized: Optional[bool] = None,
    )-> np.ndarray:
        """
        Validate shape, dtype, finiteness and optional L2 normalization.

        Raises:
            SPECTER2ValidationError: on any invariant violation.
        """
        if expected_rows is not None:
            if expected_count is not None and expected_count != expected_rows:
                raise SPECTER2ValidationError(
                    "Conflicting row-count arguments: expected_count and expected_rows specify different values."
                )
            expected_count = expected_rows

        if not isinstance(embeddings, np.ndarray):
            raise SPECTER2ValidationError(
                f"Expected numpy.ndarray, got {type(embeddings).__name__}."
            )

        if embeddings.ndim != 2:
            raise SPECTER2ValidationError(
                f"Expected a 2D matrix, got shape {embeddings.shape}."
            )

        if expected_count is not None and embeddings.shape[0] != expected_count:
            raise SPECTER2ValidationError(
                f"Expected {expected_count} rows, got {embeddings.shape[0]}. "
            )

        dimension = expected_dimension or self.embedding_dimension
        if embeddings.shape[1] != dimension:
            raise SPECTER2ValidationError(
                f"Expected embedding dimension {dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if embeddings.dtype.kind != "f":
            raise SPECTER2ValidationError(
                f"Embeddings must use a floating dtype, got {embeddings.dtype}."
            )

        if not np.isfinite(embeddings).all():
            raise SPECTER2ValidationError(
                "Embedding matrix contains NaN or Inf."
            )

        should_be_normalized = (
            self.config.normalize if normalized is None else normalized
        )

        if should_be_normalized and len(embeddings):
            norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)

            if not np.isfinite(norms).all():
                raise SPECTER2ValidationError(
                    "Embedding norms contain NaN or Inf."
                )

            if np.any(norms <= np.finfo(np.float64).eps):
                raise SPECTER2ValidationError(
                    "Embedding matrix contains zero/near-zero vectors."
                )

            max_error = float(np.max(np.abs(norms - 1.0)))
            if max_error > self.config.normalization_tolerance:
                raise SPECTER2ValidationError(
                    "Embeddings are not L2-normalized within tolerance. "
                    f"Maximum norm error: {max_error:.6g}; "
                    f"tolerance: {self.config.normalization_tolerance:.6g}."
                )
        return embeddings

    def get_embedding_info(self) -> dict[str, Any]:
        """Backward/diagnostic alias returning serializable model metadata."""
        return self.get_model_info()

    def get_embedding_dimension(self) -> int:
        """Return the verified embedding dimensionality."""
        return self.embedding_dimension

    def get_model_info(self) -> dict[str, Any]:
        """Return JSON-serializable runtime/model metadata."""
        try:
            import adapters

            adapters_version = getattr(adapters, "__version__", None)
        except Exception:
            adapters_version = None

        try:
            import transformers

            transformers_version = getattr(transformers, "__version__", None)
        except Exception:
            transformers_version = None

        metadata = EmbeddingMetadata(
            base_model=self.config.base_model,
            document_adapter=self.config.document_adapter,
            query_adapter=self.config.query_adapter,
            device=str(self.device),
            dtype=str(next(self.model.parameters()).dtype),
            embedding_dimension=self.embedding_dimension,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            normalize=self.config.normalize,
            output_dtype=self.config.output_dtype,
            transformers_version=transformers_version,
            adapters_version=adapters_version,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )

        result = asdict(metadata)
        result.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": getattr(torch.version, "cuda", None),
            "adaptive_batching": self.config.adaptive_batching,
            "min_batch_size": self.config.min_batch_size,
            "use_autocast": self.config.use_autocast,
            }
        )

        if torch.cuda.is_available():
            try:
                index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                result.update(
                    {
                        "gpu_name": torch.cuda.get_device_name(index),
                        "gpu_memory_gb": round(
                            props.total_memory / (1024 ** 3), 2
                        ),
                    }
                )
            except Exception:
                pass

        return result

    def get_embedding_fingerprint(self) -> str:
        """
        Return a stable configuration fingerprint.

        This is NOT a fingerprint of model weights. It identifies the
        configured model/adapters/inference settings so a manifest can detect
        incompatible embedding runs.
        """
        payload = repr(
            (
                self.config.base_model,
                self.config.document_adapter,
                self.config.query_adapter,
                self.config.max_length,
                self.config.normalize,
                self.config.output_dtype,
                self.config.use_autocast,
                self.config.deterministic,
            )
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()[:16]


def validation_contract_smoke_test() -> None:
    """Model-free regression test for the generic embedding validation contract."""
    class _FakeEmbedder:
        embedding_dimension = 4
        config = type("Config", (), {
            "normalize": True,
            "normalization_tolerance": 1e-4,
        })()

    validator = SPECTER2Embedder.validate_embeddings.__get__(
        _FakeEmbedder(), _FakeEmbedder
    )
    valid = np.eye(4, dtype=np.float32)

    validator(valid, expected_count=4, expected_dimension=4, normalized=True)

    try:
        validator(valid, expected_count=3, expected_dimension=4, normalized=True)
    except SPECTER2ValidationError:
        pass
    else:
        raise AssertionError("expected_count validation did not reject mismatch.")

    # Backward compatibility for older callers.
    validator(valid, expected_rows=4, expected_dimension=4, normalized=True)
    LOGGER.info("SPECTER2 validation contract smoke test PASSED.")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic console logger for standalone use."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def environment_diagnostics() -> dict[str, Any]:
    """
    Return a lightweight environment snapshot without loading model weights.

    Safe to use when diagnosing Windows, CUDA, Transformers, or AdapterHub
    problems.
    """
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
    }

    if torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info.update(
                {
                    "current_device": index,
                    "gpu_name": torch.cuda.get_device_name(index),
                    "gpu_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                }
            )
        except Exception as exc:
            info["cuda_diagnostics_error"] = repr(exc)

    return info


def smoke_test() -> None:
    """
    Minimal real-model smoke test.

    Downloads the official checkpoints if they are not already cached.
    This intentionally tests:
        1. model loading
        2. query embedding
        3. document embedding
        4. batch embedding
        5. dimensions
        6. finite values
        7. normalization
        8. ID alignment
    """
    LOGGER.info("=== SPECTER2 SMOKE TEST START ===")
    LOGGER.info("Environment: %s", environment_diagnostics())

    embedder = SPECTER2Embedder(
        device="auto",
        batch_size=2,
        max_length=512,
        normalize=True,
    )

    query = embedder.embed_query(
        "deep learning methods for medical image segmentation"
    )

    document = embedder.embed_document(
        "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "We present a convolutional network architecture for biomedical image segmentation.",
    )

    queries = embedder.embed_queries(
        [
            "medical image segmentation",
            "transformer-based scientific retrieval",
        ],
        batch_size=2,
    )

    chunk = embedder.embed_document(
        "The methodology uses a transformer encoder for scientific retrieval."
    )

    documents = embedder.embed_documents(
        [
            "A study of transformer models for scientific information retrieval.",
            "A comparison of convolutional architectures for image classification.",
        ]
    )

    batch = embedder.embed_papers(
        [
            {
                "id": "paper-001",
                "title": "Scientific Retrieval with Neural Embeddings",
                "summary": "A study of dense retrieval for scientific literature.",
            },
            {
                "id": "paper-002",
                "title": "Transformer Models for Research Search",
                "summary": "An evaluation of transformer representations for academic search.",
            },
        ]
    )

    assert query.shape == (embedder.embedding_dimension,)
    assert document.shape == (embedder.embedding_dimension,)
    assert queries.shape == (2, embedder.embedding_dimension)
    assert chunk.shape == (embedder.embedding_dimension,)
    assert documents.shape == (2, embedder.embedding_dimension)
    assert batch.embeddings.shape == (2, embedder.embedding_dimension)
    assert batch.ids == ("paper-001", "paper-002")

    # Explicit final checks.
    assert np.isfinite(query).all()
    assert np.isfinite(document).all()
    assert np.isfinite(queries).all()
    assert np.isfinite(chunk).all()
    assert np.isfinite(documents).all()
    assert np.isfinite(batch.embeddings).all()

    if embedder.config.normalize:
        assert np.allclose(np.linalg.norm(query), 1.0, atol=1e-4)
        assert np.allclose(np.linalg.norm(document), 1.0, atol=1e-4)

    LOGGER.info("SPECTER2 smoke test PASSED.")
    LOGGER.info("Model info: %s", embedder.get_model_info())
    LOGGER.info("Embedding fingerprint: %s", embedder.get_embedding_fingerprint())

if __name__ == "__main__":
    configure_logging()
    LOGGER.info("Environment diagnostics: %s", environment_diagnostics())
    smoke_test()