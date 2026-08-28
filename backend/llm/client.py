"""
Production LLM provider boundary for the AI Research Paper Assistant.

Responsibilities
----------------
* validate an already-built LLMRequest
* resolve provider/model configuration
* communicate with an injected provider adapter
* enforce timeout/retry policy at the client boundary
* normalize transport-level provider responses
* preserve provider metadata and usage
* transport structured-output hints without becoming the application parser
* never modify application prompts
* never parse scientific claims
* never perform hallucination/factual validation

This module intentionally contains NO retrieval, FAISS, embeddings, reranking,
context construction, prompt generation, PDF processing, paper analysis, or
application-level output parsing.

Integration contract
--------------------
rag/pipeline.py calls:

    raw_response = llm_client.generate(llm_request)

The prompt builder's LLMRequest is treated as an opaque, duck-typed request
with the following expected attributes:

    messages
    task_type
    prompt_version
    response_schema
    temperature

The concrete prompt implementation already defines those fields. This client
does not create a duplicate request schema.

Provider implementation
-----------------------
A small provider protocol is used so the rest of the backend never depends
directly on a provider SDK. A standard-library OpenAI-compatible HTTP adapter
is included as the concrete transport because the supplied project snapshot
does not identify an installed provider SDK or a configured provider.

The adapter can be replaced/injected without changing RAG code.

Local Ollama is supported natively through Ollama's HTTP API. It does not
require an API key and is selected with provider=`ollama`. The default local
endpoint is http://127.0.0.1:11434.

Security
--------
* API credentials are never logged.
* Prompt bodies and raw model output are never logged by default.
* Retrieved paper text is transported as data only.
* No eval/exec/shell/dynamic code execution is performed.
"""

from __future__ import annotations

import json
import logging
import math
from email.utils import parsedate_to_datetime
from pathlib import Path
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from http.client import HTTPResponse
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from backend.rag.prompts import PromptBuilder, normalize_query_for_identity
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_MAX_SECONDS = 8.0
DEFAULT_JITTER_SECONDS = 0.2
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_STRUCTURED_OUTPUT_MODE = "schema"
MAX_PROVIDER_ERROR_BODY_BYTES = 4096
MAX_PROVIDER_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
MAX_URL_LENGTH = 2048
OLLAMA_STRUCTURED_OUTPUT_MODES = frozenset({"schema", "json", "none"})

RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


# ============================================================================
# Errors
# ============================================================================


class LLMError(RuntimeError):
    """Base error for the LLM communication layer."""


class LLMConfigurationError(LLMError, ValueError):
    """Invalid or incomplete LLM configuration."""


class LLMRequestError(LLMError, ValueError):
    """Invalid LLM request supplied by the application."""


class LLMAuthenticationError(LLMError):
    """Provider rejected credentials."""


class LLMRateLimitError(LLMError):
    """Provider rate-limited the request."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMTimeoutError(LLMError, TimeoutError):
    """Provider request exceeded the configured timeout."""


class LLMNetworkError(LLMError):
    """Network/transport failure."""


class LLMInvalidRequestError(LLMError):
    """Provider rejected the request as malformed/unsupported."""


class LLMProviderError(LLMError):
    """Provider-side failure that is not more specifically classified."""


class LLMEmptyResponseError(LLMError):
    """Provider returned no usable generated content."""


class LLMResponseError(LLMError):
    """Provider response existed but violated the transport contract."""


# ============================================================================
# Public normalized response
# ============================================================================


class FinishReason(str, Enum):
    """Common finish-reason values; unknown provider values remain strings."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMUsage:
    """Normalized token usage. Missing provider values remain None."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Common aliases for providers using prompt/completion terminology.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be None or a non-negative integer.")

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-normalized raw response.

    `raw_text` is the application parser's input.
    `raw_payload` is retained only when the provider adapter explicitly
    supplies a structured transport payload. It is never semantically parsed
    by this module.
    """

    raw_text: str
    provider: str
    model: str
    finish_reason: Optional[str]
    usage: Optional[LLMUsage]
    request_id: str
    provider_request_id: Optional[str]
    latency_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise LLMEmptyResponseError(
                "LLMResponse.raw_text must contain usable text."
            )
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")

    @property
    def text(self) -> str:
        """Parser-friendly alias."""
        return self.raw_text

    @property
    def content(self) -> str:
        """Parser-friendly alias."""
        return self.raw_text


# ============================================================================
# Client configuration
# ============================================================================


@dataclass(frozen=True)
class LLMClientConfig:
    """
    Provider configuration.

    No secret is stored in logs. `api_key` may be supplied by the application's
    existing configuration layer. If the application passes a key here, this
    object treats it as opaque secret material.

    `base_url` is intentionally generic. The included adapter speaks the
    OpenAI-compatible chat-completions wire format.
    """

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: Optional[float] = 0.0
    max_output_tokens: Optional[int] = 2048
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_max_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS
    jitter_seconds: float = DEFAULT_JITTER_SECONDS

    # Provider structured-output request mode. It is passed only when the
    # provider adapter supports it; parser.py remains the application parser.
    enable_structured_output: bool = True

    # Ollama compatibility mode:
    #   schema -> send the application JSON schema; if Ollama rejects that
    #             transport feature with HTTP 400/422, retry once WITHOUT the
    #             `format` field. This is the safest compatibility path for
    #             local Ollama installations/models that reject `format`.
    #   json   -> explicitly request Ollama generic JSON output. This remains
    #             available as an opt-in mode, but is NOT the default.
    #   none   -> do not send Ollama's `format` parameter.
    #
    # Schema mode is the production default. The application parser remains the
    # final enforcement layer, and Ollama's format transport is treated as an
    # optimization/constraint rather than a trust boundary.
    ollama_structured_output_mode: str = DEFAULT_OLLAMA_STRUCTURED_OUTPUT_MODE

    # Safe logging control. Even when True, message bodies and secrets remain
    # redacted; this only permits diagnostic metadata.
    debug_transport_metadata: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise LLMConfigurationError("provider must be non-empty.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMConfigurationError("model must be non-empty.")

        if self.api_key is not None and not isinstance(self.api_key, str):
            raise LLMConfigurationError("api_key must be a string or None.")

        if self.base_url is not None:
            if not isinstance(self.base_url, str) or not self.base_url.strip():
                raise LLMConfigurationError("base_url must be a non-empty string.")

        if self.temperature is not None:
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(float(self.temperature))
                or not 0.0 <= float(self.temperature) <= 2.0
            ):
                raise LLMConfigurationError(
                    "temperature must be None or a finite value in [0, 2]."
                )

        if self.max_output_tokens is not None:
            _validate_positive_int(
                self.max_output_tokens,
                "max_output_tokens",
                error_type=LLMConfigurationError,
            )

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise LLMConfigurationError(
                "timeout_seconds must be a positive number."
            )

        _validate_non_negative_int(
            self.max_retries,
            "max_retries",
            error_type=LLMConfigurationError,
        )

        for name, value in (
            ("backoff_base_seconds", self.backoff_base_seconds),
            ("backoff_max_seconds", self.backoff_max_seconds),
            ("jitter_seconds", self.jitter_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise LLMConfigurationError(
                    f"{name} must be a non-negative number."
                )

        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise LLMConfigurationError(
                "backoff_max_seconds must be >= backoff_base_seconds."
            )

        if self.ollama_structured_output_mode not in OLLAMA_STRUCTURED_OUTPUT_MODES:
            raise LLMConfigurationError(
                "ollama_structured_output_mode must be one of "
                f"{sorted(OLLAMA_STRUCTURED_OUTPUT_MODES)}."
            )


# ============================================================================
# Provider protocol
# ============================================================================


@dataclass(frozen=True)
class ProviderResponse:
    """
    Transport-normalized provider response consumed by LLMClient.

    Provider adapters may populate `raw_payload` with the decoded transport
    object. This is NOT application-level output parsing.
    """

    text: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[LLMUsage] = None
    provider_request_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_payload: Any = None


class LLMProviderAdapter(Protocol):
    """Provider-specific transport boundary."""

    @property
    def provider_name(self) -> str:
        ...

    def generate(
        self,
        request: Any,
        *,
        config: LLMClientConfig,
        request_id: str,
    ) -> ProviderResponse:
        ...


# ============================================================================
# Request helpers
# ============================================================================


def _validate_positive_int(
    value: Any,
    name: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise error_type(f"{name} must be a positive integer; got {value!r}.")


def _validate_non_negative_int(
    value: Any,
    name: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise error_type(
            f"{name} must be a non-negative integer; got {value!r}."
        )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_messages(request: Any) -> tuple[dict[str, str], ...]:
    messages = _get(request, "messages")
    if messages is None:
        raise LLMRequestError("LLM request must contain messages.")

    try:
        values = tuple(messages)
    except TypeError as exc:
        raise LLMRequestError("LLM request messages must be iterable.") from exc

    if not values:
        raise LLMRequestError("LLM request must contain at least one message.")

    normalized: list[dict[str, str]] = []

    for index, message in enumerate(values):
        role = _get(message, "role")
        content = _get(message, "content")

        if role not in {"system", "user", "assistant", "tool"}:
            raise LLMRequestError(
                f"Unsupported message role at index {index}: {role!r}."
            )

        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError(
                f"Message content at index {index} must be non-empty."
            )

        # IMPORTANT: content is copied verbatim except for type validation.
        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return tuple(normalized)


def _extract_request_query(messages: Sequence[Mapping[str, str]]) -> str:
    """Extract the exact canonical USER_QUERY from a PromptBuilder message."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        start = content.find("<USER_QUERY>")
        end = content.find("</USER_QUERY>", start + len("<USER_QUERY>"))
        if start < 0 or end < 0:
            continue
        query = content[start + len("<USER_QUERY>"):end]
        try:
            return normalize_query_for_identity(query)
        except ValueError as exc:
            raise LLMRequestError(
                "LLM request contains an empty USER_QUERY."
            ) from exc
    raise LLMRequestError(
        "LLM request user message does not contain a <USER_QUERY> block."
    )


def _validate_request(request: Any) -> tuple[dict[str, str], ...]:
    if request is None:
        raise LLMRequestError("LLM request cannot be None.")

    messages = _normalize_messages(request)

    task_type = _get(request, "task_type")
    if not isinstance(task_type, str) or not task_type.strip():
        raise LLMRequestError("LLM request task_type must be non-empty.")

    prompt_version = _get(request, "prompt_version")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise LLMRequestError(
            "LLM request prompt_version must be non-empty."
        )

    response_schema = _get(request, "response_schema")
    if response_schema is None or not isinstance(response_schema, Mapping):
        raise LLMRequestError(
            "LLM request response_schema must be a mapping."
        )

    # The schema is transported to providers as JSON. Validate that contract
    # here so provider adapters never leak raw TypeError/ValueError exceptions
    # caused by non-serializable application configuration.
    try:
        json.dumps(dict(response_schema), ensure_ascii=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMRequestError(
            "LLM request response_schema must be JSON-serializable."
        ) from exc

    temperature = _get(request, "temperature")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise LLMRequestError(
                "LLM request temperature must be None or in [0, 2]."
            )

    # The prompt builder is the single source of truth for request identity.
    # A provider must never receive a reconstructed/anonymous request because
    # that makes a 200 OK response impossible to correlate with the API query.
    request_id = _get(request, "request_id")
    query_fingerprint = _get(request, "query_fingerprint")
    evidence_fingerprint = _get(request, "evidence_fingerprint")

    if not isinstance(request_id, str) or not request_id.strip():
        raise LLMRequestError(
            "LLM request must contain the PromptBuilder request_id."
        )
    if not isinstance(query_fingerprint, str) or not query_fingerprint.strip():
        raise LLMRequestError(
            "LLM request must contain query_fingerprint."
        )
    if not isinstance(evidence_fingerprint, str) or not evidence_fingerprint.strip():
        raise LLMRequestError(
            "LLM request must contain evidence_fingerprint."
        )

    # Revalidate the exact request contract, including deterministic request_id.
    # This intentionally uses the same implementation as PromptBuilder rather
    # than duplicating the identity algorithm in this module.
    try:
        request_query = _extract_request_query(messages)
        PromptBuilder.verify_query_identity(request, request_query)
    except Exception as exc:
        raise LLMRequestError(
            "LLM request identity verification failed."
        ) from exc

    return messages


# ============================================================================
# OpenAI-compatible provider adapter
# ============================================================================


class OpenAICompatibleProvider:
    """
    Standard-library OpenAI-compatible chat-completions adapter.

    This is deliberately isolated from LLMClient. It can be replaced with an
    official SDK adapter later without changing rag/pipeline.py.

    Expected endpoint:
        {base_url}/chat/completions

    `base_url` may already include an API prefix such as `/v1`.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        user_agent: str = "AIResearchPaperAssistant/1.0",
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._user_agent = user_agent
        self._opener = opener or urllib_request.urlopen

    def generate(
        self,
        request: Any,
        *,
        config: LLMClientConfig,
        request_id: str,
    ) -> ProviderResponse:
        if not config.base_url:
            raise LLMConfigurationError(
                "base_url is required for the OpenAI-compatible provider."
            )

        if not config.api_key:
            raise LLMConfigurationError(
                "api_key is required for the OpenAI-compatible provider."
            )

        messages = _normalize_messages(request)

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
        }

        # PromptBuilder owns the task-specific temperature. The client only
        # propagates it; it does not invent task-specific behavior.
        request_temperature = _get(request, "temperature")
        effective_temperature = (
            request_temperature
            if request_temperature is not None
            else config.temperature
        )
        if effective_temperature is not None:
            payload["temperature"] = float(effective_temperature)

        if config.max_output_tokens is not None:
            payload["max_tokens"] = config.max_output_tokens

        response_schema = _get(request, "response_schema")
        if config.enable_structured_output and response_schema:
            # OpenAI-compatible providers vary. `json_schema` is sent only
            # through this adapter; application parsing remains parser.py.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "research_paper_assistant_response",
                    "strict": True,
                    "schema": dict(response_schema),
                },
            }

        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        endpoint = config.base_url.rstrip("/") + "/chat/completions"

        request_object = urllib_request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self._user_agent,
                "X-Request-ID": request_id,
            },
        )

        try:
            response = self._opener(
                request_object,
                timeout=float(config.timeout_seconds),
            )
            return self._decode_success_response(
                response,
                fallback_model=config.model,
            )
        except urllib_error.HTTPError as exc:
            raise self._map_http_error(exc) from exc
        except (TimeoutError, urllib_error.URLError) as exc:
            if _looks_like_timeout(exc):
                raise LLMTimeoutError(
                    "LLM provider request timed out."
                ) from exc
            raise LLMNetworkError(
                "LLM provider network request failed."
            ) from exc

    @staticmethod
    def _decode_success_response(
        response: HTTPResponse,
        *,
        fallback_model: str,
    ) -> ProviderResponse:
        try:
            raw_bytes = _read_http_response_bytes(
                response,
                max_bytes=MAX_PROVIDER_RESPONSE_BODY_BYTES,
            )
        except (TimeoutError, OSError) as exc:
            raise LLMNetworkError(
                "Failed while reading the LLM provider response."
            ) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not raw_bytes:
            raise LLMEmptyResponseError(
                "LLM provider returned an empty HTTP response."
            )

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "LLM provider returned a non-JSON response."
            ) from exc

        if not isinstance(payload, Mapping):
            raise LLMResponseError(
                "LLM provider response root must be a JSON object."
            )

        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(
            choices, (str, bytes)
        ) or not choices:
            raise LLMResponseError(
                "LLM provider response contains no choices."
            )

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMResponseError(
                "LLM provider choice has an invalid structure."
            )

        message = first_choice.get("message")
        text: Optional[str] = None

        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                text = content

            # Some compatible providers return structured content blocks.
            # This is transport normalization only; no scientific parsing.
            elif isinstance(content, Sequence) and not isinstance(
                content, (str, bytes)
            ):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
                if parts:
                    text = "".join(parts)

        if text is None:
            direct_text = first_choice.get("text")
            if isinstance(direct_text, str):
                text = direct_text

        if not isinstance(text, str) or not text.strip():
            raise LLMEmptyResponseError(
                "LLM provider returned no usable generated text."
            )

        usage = _parse_usage(payload.get("usage"))

        finish_reason = first_choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            model = fallback_model

        provider_request_id = payload.get("id")
        if provider_request_id is not None and not isinstance(
            provider_request_id, str
        ):
            provider_request_id = str(provider_request_id)

        metadata = {
            "object": payload.get("object"),
            "system_fingerprint": payload.get("system_fingerprint"),
        }
        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

        return ProviderResponse(
            text=text,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            provider_request_id=provider_request_id,
            metadata=metadata,
            raw_payload=payload,
        )

    @staticmethod
    def _map_http_error(exc: urllib_error.HTTPError) -> LLMError:
        status = int(getattr(exc, "code", 0) or 0)

        # Read only a bounded provider error body. It is intentionally NOT
        # included in the raised message because provider bodies may contain
        # secrets, request content, or sensitive data.
        try:
            exc.read(MAX_PROVIDER_ERROR_BODY_BYTES)
        except Exception:
            pass

        if status in {401, 403}:
            return LLMAuthenticationError(
                "LLM provider authentication/authorization failed."
            )

        retry_after = _parse_retry_after(exc.headers)

        if status == 429:
            return LLMRateLimitError(
                "LLM provider rate limit was reached.",
                retry_after_seconds=retry_after,
            )

        if status in {400, 406, 413, 415, 422}:
            return LLMInvalidRequestError(
                f"LLM provider rejected the request (HTTP {status})."
            )

        if status in RETRYABLE_HTTP_STATUS_CODES:
            return _RetryableProviderHTTPError(
                f"LLM provider returned transient HTTP {status}.",
                status_code=status,
                retry_after_seconds=retry_after,
            )

        return LLMProviderError(
            f"LLM provider returned HTTP {status}."
        )



def _normalize_ollama_base_url(base_url: Optional[str]) -> str:
    """Normalize a local Ollama base URL without changing its host."""
    value = (base_url or DEFAULT_OLLAMA_BASE_URL).strip()
    if not value:
        raise LLMConfigurationError("Ollama base_url must not be empty.")
    if len(value) > MAX_URL_LENGTH:
        raise LLMConfigurationError("Ollama base_url is too long.")

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            "Ollama base_url must be an absolute HTTP(S) URL."
        )
    if parsed.query or parsed.fragment:
        raise LLMConfigurationError(
            "Ollama base_url must not contain query or fragment components."
        )

    path = parsed.path.rstrip("/")
    if path == "/api":
        path = ""
    elif path.endswith("/api"):
        path = path[:-4].rstrip("/")

    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    ).rstrip("/")


class OllamaProvider:
    """
    Native Ollama HTTP adapter.

    Endpoint:
        {base_url}/api/chat

    Ollama is a local provider and therefore does not require an API key.
    The adapter uses the native Ollama chat API rather than pretending that
    Ollama is a cloud/OpenAI provider. This keeps local development fully
    offline from paid APIs while preserving the same LLMClient contract.
    """

    provider_name = "ollama"

    def __init__(
        self,
        *,
        user_agent: str = "AIResearchPaperAssistant/1.0",
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._user_agent = user_agent
        self._opener = opener or urllib_request.urlopen

    def generate(
        self,
        request: Any,
        *,
        config: LLMClientConfig,
        request_id: str,
    ) -> ProviderResponse:
        base_url = _normalize_ollama_base_url(config.base_url)
        messages = _normalize_messages(request)

        request_temperature = _get(request, "temperature")
        effective_temperature = (
            request_temperature
            if request_temperature is not None
            else config.temperature
        )

        options: dict[str, Any] = {}
        if effective_temperature is not None:
            options["temperature"] = float(effective_temperature)
        if config.max_output_tokens is not None:
            options["num_predict"] = int(config.max_output_tokens)

        response_schema = _get(request, "response_schema")
        mode = (
            config.ollama_structured_output_mode
            if config.enable_structured_output
            else "none"
        )

        def build_payload(output_mode: str) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": config.model,
                "messages": [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages
                ],
                "stream": False,
            }
            if options:
                payload["options"] = dict(options)

            if output_mode == "schema":
                if not isinstance(response_schema, Mapping) or not response_schema:
                    raise LLMRequestError(
                        "Ollama schema mode requires a non-empty response_schema."
                    )
                payload["format"] = dict(response_schema)
            elif output_mode == "json":
                payload["format"] = "json"

            return payload

        def send(payload: Mapping[str, Any]) -> ProviderResponse:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

            endpoint = base_url + "/api/chat"
            request_object = urllib_request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self._user_agent,
                    "X-Request-ID": request_id,
                    "X-Query-Fingerprint": str(_get(request, "query_fingerprint")),
                    "X-Evidence-Fingerprint": str(_get(request, "evidence_fingerprint")),
                },
            )

            try:
                response = self._opener(
                    request_object,
                    timeout=float(config.timeout_seconds),
                )
                return self._decode_success_response(
                    response,
                    fallback_model=config.model,
                )
            except urllib_error.HTTPError:
                raise

        # Some Ollama versions/models reject structured-output transport.
        # For this project, the application parser is the final schema
        # enforcement layer, so the safest compatibility fallback is to retry
        # once WITHOUT Ollama's `format` parameter.
        try:
            return send(build_payload(mode))
        except urllib_error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            if mode in {"schema", "json"} and status in {400, 422}:
                try:
                    exc.close()
                except Exception:
                    pass
                logger.warning(
                    "Ollama structured-%s request rejected; "
                    "retrying once without format: request_id=%s status=%d",
                    mode,
                    request_id,
                    status,
                )
                try:
                    return send(build_payload("none"))
                except urllib_error.HTTPError as fallback_exc:
                    raise self._map_http_error(fallback_exc) from fallback_exc
            raise self._map_http_error(exc) from exc
        except (TimeoutError, urllib_error.URLError) as exc:
            if _looks_like_timeout(exc):
                raise LLMTimeoutError("Ollama request timed out.") from exc
            raise LLMNetworkError("Ollama network request failed.") from exc

    @staticmethod
    def _decode_success_response(
        response: HTTPResponse,
        *,
        fallback_model: str,
    ) -> ProviderResponse:
        try:
            raw_bytes = _read_http_response_bytes(
                response,
                max_bytes=MAX_PROVIDER_RESPONSE_BODY_BYTES,
            )
        except (TimeoutError, OSError) as exc:
            raise LLMNetworkError(
                "Failed while reading the Ollama response."
            ) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not raw_bytes:
            raise LLMEmptyResponseError(
                "Ollama returned an empty HTTP response."
            )

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "Ollama returned a non-JSON response."
            ) from exc

        if not isinstance(payload, Mapping):
            raise LLMResponseError(
                "Ollama response root must be a JSON object."
            )

        # Native /api/chat response: {"message": {"role": "assistant",
        # "content": "..."}, "done": true, ...}
        message = payload.get("message")
        text: Optional[str] = None
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                text = content

        # Defensive fallback for compatible proxies.
        if text is None:
            direct_text = payload.get("response")
            if isinstance(direct_text, str):
                text = direct_text

        if not isinstance(text, str) or not text.strip():
            raise LLMEmptyResponseError(
                "Ollama returned no usable generated text."
            )

        prompt_tokens = _safe_non_negative_int(payload.get("prompt_eval_count"))
        output_tokens = _safe_non_negative_int(payload.get("eval_count"))
        total_tokens = (
            prompt_tokens + output_tokens
            if prompt_tokens is not None and output_tokens is not None
            else None
        )
        usage = (
            LLMUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
            )
            if any(v is not None for v in (prompt_tokens, output_tokens, total_tokens))
            else None
        )

        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            model = fallback_model

        finish_reason = payload.get("done_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        provider_request_id = payload.get("id")
        if provider_request_id is not None and not isinstance(
            provider_request_id, str
        ):
            provider_request_id = str(provider_request_id)

        metadata_keys = (
            "done",
            "created_at",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
        metadata = {
            key: payload.get(key)
            for key in metadata_keys
            if payload.get(key) is not None
        }

        metadata["structured_output_mode"] = "transport_only"
        return ProviderResponse(
            text=text,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            provider_request_id=provider_request_id,
            metadata=metadata,
            raw_payload=payload,
        )

    @staticmethod
    def _map_http_error(exc: urllib_error.HTTPError) -> LLMError:
        status = int(getattr(exc, "code", 0) or 0)

        # Never expose provider response bodies in exception messages.
        try:
            exc.read(MAX_PROVIDER_ERROR_BODY_BYTES)
        except Exception:
            pass

        retry_after = _parse_retry_after(exc.headers)

        if status == 429:
            return LLMRateLimitError(
                "Ollama rate limit was reached.",
                retry_after_seconds=retry_after,
            )

        if status in {400, 404, 405, 406, 413, 415, 422}:
            return LLMInvalidRequestError(
                f"Ollama rejected the request (HTTP {status})."
            )

        if status in {401, 403}:
            return LLMAuthenticationError(
                "Ollama rejected the request authorization."
            )

        if status in RETRYABLE_HTTP_STATUS_CODES:
            return _RetryableProviderHTTPError(
                f"Ollama returned transient HTTP {status}.",
                status_code=status,
                retry_after_seconds=retry_after,
            )

        return LLMProviderError(
            f"Ollama returned HTTP {status}."
        )


class _RetryableProviderHTTPError(LLMProviderError):
    """Internal marker for bounded retry handling."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _parse_retry_after(headers: Any) -> Optional[float]:
    """
    Parse a Retry-After header safely.

    RFC 9110 permits either:
      * delta-seconds, or
      * an HTTP-date.

    The returned value is deliberately not trusted as an unbounded sleep:
    ``LLMClient._retry_delay`` applies the configured maximum.
    """
    if headers is None:
        return None

    try:
        value = headers.get("Retry-After")
    except Exception:
        return None

    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # Preferred form: delta-seconds.
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = None

    if parsed is not None:
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return parsed

    # HTTP-date form.
    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at is None:
            return None
        if retry_at.tzinfo is None:
            return None
        delay = retry_at.timestamp() - time.time()
    except (TypeError, ValueError, OverflowError, OSError):
        return None

    if not math.isfinite(delay):
        return None

    return max(0.0, delay)


def _safe_non_negative_int(value: Any) -> Optional[int]:
    """Return a non-negative integer or None without coercing arbitrary data."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_usage(value: Any) -> Optional[LLMUsage]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None

    def integer(*names: str) -> Optional[int]:
        for name in names:
            candidate = value.get(name)
            if candidate is None:
                continue
            if isinstance(candidate, bool) or not isinstance(candidate, int):
                continue
            if candidate < 0:
                continue
            return candidate
        return None

    prompt_tokens = integer("prompt_tokens", "input_tokens")
    completion_tokens = integer("completion_tokens", "output_tokens")
    total_tokens = integer("total_tokens")

    return LLMUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ============================================================================
# Main client
# ============================================================================


class LLMClient:
    """
    Provider-independent LLM client.

    The provider adapter is injected. Therefore tests can use a fake provider
    and no network/API key is required.

    The client does NOT:
        * alter prompts
        * generate task-specific prompts
        * parse or repair application JSON
        * validate scientific claims
        * access retrieval systems

    Structured-output transport is only a provider constraint. The application
    parser remains authoritative after the provider response is received.
    """

    def __init__(
        self,
        config: LLMClientConfig,
        *,
        provider: Optional[LLMProviderAdapter] = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        if not isinstance(config, LLMClientConfig):
            raise LLMConfigurationError(
                "config must be an LLMClientConfig instance."
            )

        self.config = config
        self.provider = provider or OpenAICompatibleProvider()
        self._sleeper = sleeper
        self._random = random_source

        provider_name = getattr(self.provider, "provider_name", None)
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise LLMConfigurationError(
                "Provider adapter must expose a non-empty provider_name."
            )

        generate = getattr(self.provider, "generate", None)
        if not callable(generate):
            raise LLMConfigurationError(
                "Provider adapter must expose generate()."
            )

        self._validate_configuration_for_provider()

    def _validate_configuration_for_provider(self) -> None:
        provider_name = str(self.provider.provider_name).strip().lower()

        if provider_name in {"ollama", "ollama_local", "local_ollama"}:
            if not self.config.model:
                raise LLMConfigurationError(
                    "model is required for the configured Ollama provider."
                )
            # API keys are intentionally not required for local Ollama.
            return

        if provider_name == "openai_compatible":
            if not self.config.base_url:
                raise LLMConfigurationError(
                    "base_url is required for the configured "
                    "OpenAI-compatible provider."
                )
            if not self.config.api_key:
                raise LLMConfigurationError(
                    "api_key is required for the configured "
                    "OpenAI-compatible provider."
                )

    def generate(
        self,
        request: Any,
        *,
        session_id: Optional[str] = None,
        task_type: Optional[str] = None,
        query: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate one normalized raw response.

        `session_id`, `task_type`, and `query` are tracing metadata only.
        They do not alter the prompt or provider request.
        """
        messages = _validate_request(request)

        if task_type is not None and (
            not isinstance(task_type, str) or not task_type.strip()
        ):
            raise LLMRequestError("task_type must be a non-empty string or None.")

        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise LLMRequestError("session_id must be a non-empty string or None.")

        if query is not None and not isinstance(query, str):
            raise LLMRequestError("query must be a string or None.")

        request_task_type = _get(request, "task_type")
        if task_type is not None and task_type != request_task_type:
            raise LLMRequestError(
                "Explicit task_type does not match LLMRequest.task_type."
            )

        # The request identity is created by PromptBuilder and MUST be the same
        # identifier used for every provider attempt/retry. Generating a fresh
        # client-side UUID here was the critical observability/integrity flaw: the
        # request being logged could differ from the request that was built.
        request_id = str(_get(request, "request_id")).strip()

        request_query = _extract_request_query(messages)
        if query is not None:
            try:
                supplied_query = normalize_query_for_identity(query)
            except ValueError as exc:
                raise LLMRequestError("query must not be empty.") from exc
            if supplied_query != request_query:
                raise LLMRequestError(
                    "Explicit query does not match the LLMRequest USER_QUERY."
                )

        started = time.perf_counter()
        attempt = 0

        while True:
            attempt += 1

            try:
                provider_response = self.provider.generate(
                    request,
                    config=self.config,
                    request_id=request_id,
                )

                normalized = self._normalize_response(
                    provider_response,
                    request_id=request_id,
                    latency_seconds=time.perf_counter() - started,
                    session_id=session_id,
                )

                logger.info(
                    "LLM generation succeeded: request_id=%s provider=%s "
                    "model=%s task=%s latency=%.4fs attempts=%d%s",
                    request_id,
                    self.provider.provider_name,
                    self.config.model,
                    request_task_type,
                    normalized.latency_seconds,
                    attempt,
                    self._safe_usage_log_suffix(normalized.usage),
                )

                return normalized

            except LLMError as exc:
                if not self._should_retry(exc, attempt):
                    self._log_failure(
                        request_id=request_id,
                        task_type=request_task_type,
                        error=exc,
                        attempt=attempt,
                    )
                    raise

                delay = self._retry_delay(attempt, exc)

                logger.warning(
                    "Retrying transient LLM failure: request_id=%s "
                    "provider=%s attempt=%d/%d delay=%.3fs error=%s",
                    request_id,
                    self.provider.provider_name,
                    attempt,
                    self.config.max_retries + 1,
                    delay,
                    type(exc).__name__,
                )

                self._sleeper(delay)

            except Exception as exc:
                # Unknown adapter exceptions are provider failures. They are
                # never retried blindly because their semantics are unknown.
                self._log_failure(
                    request_id=request_id,
                    task_type=request_task_type,
                    error=exc,
                    attempt=attempt,
                )
                raise LLMProviderError(
                    "Unexpected LLM provider failure."
                ) from exc

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_response(
        self,
        response: ProviderResponse,
        *,
        request_id: str,
        latency_seconds: float,
        session_id: Optional[str] = None,
    ) -> LLMResponse:
        if not isinstance(response, ProviderResponse):
            raise LLMResponseError(
                "Provider adapter must return ProviderResponse."
            )

        if response.metadata is not None and not isinstance(
            response.metadata, Mapping
        ):
            raise LLMResponseError(
                "Provider response metadata must be a mapping."
            )

        if response.usage is not None and not isinstance(
            response.usage, LLMUsage
        ):
            raise LLMResponseError(
                "Provider response usage must be LLMUsage or None."
            )

        if not isinstance(response.text, str) or not response.text.strip():
            raise LLMEmptyResponseError(
                "Provider adapter returned empty generated text."
            )

        model = response.model or self.config.model
        if not isinstance(model, str) or not model.strip():
            raise LLMResponseError(
                "Provider response did not contain a usable model."
            )

        finish_reason = response.finish_reason
        if finish_reason is not None and not isinstance(
            finish_reason, str
        ):
            finish_reason = str(finish_reason)

        provider_request_id = response.provider_request_id
        if provider_request_id is not None and not isinstance(
            provider_request_id, str
        ):
            provider_request_id = str(provider_request_id)

        metadata = dict(response.metadata or {})

        # Request tracing metadata is safe and contains no user content.
        metadata["session_present"] = session_id is not None
        metadata["attempted_provider"] = self.provider.provider_name
        metadata["configured_provider"] = self.config.provider
        metadata["request_id"] = request_id

        return LLMResponse(
            raw_text=response.text,
            provider=self.provider.provider_name,
            model=model,
            finish_reason=finish_reason,
            usage=response.usage,
            request_id=request_id,
            provider_request_id=provider_request_id,
            latency_seconds=max(0.0, float(latency_seconds)),
            metadata=metadata,
            raw_payload=response.raw_payload,
        )

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def _should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt > self.config.max_retries:
            return False

        # Never retry authentication, invalid requests, malformed responses,
        # empty responses, or other deterministic application errors.
        if isinstance(
            exc,
            (
                LLMAuthenticationError,
                LLMInvalidRequestError,
                LLMEmptyResponseError,
                LLMResponseError,
                LLMConfigurationError,
                LLMRequestError,
            ),
        ):
            return False

        # Retry rate limiting only when bounded retries were explicitly
        # configured. The delay remains exponential rather than immediate.
        if isinstance(exc, LLMRateLimitError):
            return True

        if isinstance(
            exc,
            (
                LLMTimeoutError,
                LLMNetworkError,
                _RetryableProviderHTTPError,
            ),
        ):
            return True

        # A generic LLMProviderError is NOT automatically retried unless it is
        # the internal retryable HTTP marker.
        return False

    def _retry_delay(
        self,
        attempt: int,
        error: BaseException,
    ) -> float:
        retry_after = getattr(error, "retry_after_seconds", None)
        if isinstance(retry_after, (int, float)) and retry_after >= 0:
            return min(
                float(retry_after),
                self.config.backoff_max_seconds,
            )

        exponential = self.config.backoff_base_seconds * (2 ** (attempt - 1))
        bounded = min(exponential, self.config.backoff_max_seconds)

        jitter = (
            self._random() * self.config.jitter_seconds
            if self.config.jitter_seconds > 0
            else 0.0
        )

        return min(
            self.config.backoff_max_seconds,
            bounded + jitter,
        )

    # ------------------------------------------------------------------
    # Logging / tracing
    # ------------------------------------------------------------------


    @staticmethod
    def _safe_usage_log_suffix(usage: Optional[LLMUsage]) -> str:
        if usage is None:
            return ""

        total = usage.total_tokens
        return (
            f" total_tokens={total}"
            if isinstance(total, int)
            else ""
        )

    @staticmethod
    def _log_failure(
        *,
        request_id: str,
        task_type: Any,
        error: BaseException,
        attempt: int,
    ) -> None:
        # Never log exception strings from arbitrary providers: provider error
        # messages can contain request fragments or sensitive metadata.
        logger.error(
            "LLM generation failed: request_id=%s task=%s attempt=%d "
            "error_type=%s",
            request_id,
            task_type,
            attempt,
            type(error).__name__,
        )


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError)


def _read_http_response_bytes(response: Any, *, max_bytes: int) -> bytes:
    """
    Read a provider response with a bounded safety limit.

    Some test doubles expose read() without a size argument, so the helper
    gracefully falls back to read() for those objects. Real HTTPResponse
    objects support bounded reads.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")

    try:
        raw = response.read(max_bytes + 1)
    except TypeError:
        raw = response.read()

    if not isinstance(raw, (bytes, bytearray)):
        raise LLMResponseError("LLM provider returned an invalid response body.")

    if len(raw) > max_bytes:
        raise LLMResponseError("LLM provider response exceeded the safety limit.")

    return bytes(raw)


# ============================================================================
# Optional environment/config factory
# ============================================================================


def create_client_from_mapping(
    settings: Mapping[str, Any],
    *,
    provider_adapter: Optional[LLMProviderAdapter] = None,
) -> LLMClient:
    """
    Build an LLMClient from an existing application's configuration mapping.

    This is intentionally a mapping adapter rather than direct environment
    access. The project's backend/config.py can pass its already-resolved
    settings here without scattering os.getenv() through the codebase.

    Supported keys:
        LLM_PROVIDER / provider
        LLM_MODEL / model
        LLM_API_KEY / api_key
        LLM_BASE_URL / base_url
        LLM_TEMPERATURE / temperature
        LLM_MAX_OUTPUT_TOKENS / max_output_tokens
        LLM_TIMEOUT_SECONDS / timeout_seconds
        LLM_MAX_RETRIES / max_retries
        LLM_BACKOFF_BASE_SECONDS / backoff_base_seconds
        LLM_BACKOFF_MAX_SECONDS / backoff_max_seconds
        LLM_JITTER_SECONDS / jitter_seconds
        LLM_ENABLE_STRUCTURED_OUTPUT / enable_structured_output
        LLM_OLLAMA_STRUCTURED_OUTPUT_MODE / ollama_structured_output_mode
    """

    def first(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in settings and settings[name] is not None:
                return settings[name]
        return default

    provider = first("LLM_PROVIDER", "provider")
    model = first("LLM_MODEL", "model")
    api_key = first("LLM_API_KEY", "api_key")
    base_url = first("LLM_BASE_URL", "base_url")

    if not provider or not model:
        raise LLMConfigurationError(
            "LLM provider and model must be supplied by the application "
            "configuration layer."
        )

    config = LLMClientConfig(
        provider=str(provider),
        model=str(model),
        api_key=None if api_key is None else str(api_key),
        base_url=None if base_url is None else str(base_url),
        temperature=_coerce_optional_float(
            first("LLM_TEMPERATURE", "temperature", default=0.0),
            "temperature",
        ),
        max_output_tokens=_coerce_optional_int(
            first(
                "LLM_MAX_OUTPUT_TOKENS",
                "max_output_tokens",
                default=2048,
            ),
            "max_output_tokens",
        ),
        timeout_seconds=_coerce_float(
            first(
                "LLM_TIMEOUT_SECONDS",
                "timeout_seconds",
                default=DEFAULT_TIMEOUT_SECONDS,
            ),
            "timeout_seconds",
        ),
        max_retries=_coerce_int(
            first(
                "LLM_MAX_RETRIES",
                "max_retries",
                default=DEFAULT_MAX_RETRIES,
            ),
            "max_retries",
        ),
        backoff_base_seconds=_coerce_float(
            first(
                "LLM_BACKOFF_BASE_SECONDS",
                "backoff_base_seconds",
                default=DEFAULT_BACKOFF_BASE_SECONDS,
            ),
            "backoff_base_seconds",
        ),
        backoff_max_seconds=_coerce_float(
            first(
                "LLM_BACKOFF_MAX_SECONDS",
                "backoff_max_seconds",
                default=DEFAULT_BACKOFF_MAX_SECONDS,
            ),
            "backoff_max_seconds",
        ),
        jitter_seconds=_coerce_float(
            first(
                "LLM_JITTER_SECONDS",
                "jitter_seconds",
                default=DEFAULT_JITTER_SECONDS,
            ),
            "jitter_seconds",
        ),
        enable_structured_output=_coerce_bool(
            first(
                "LLM_ENABLE_STRUCTURED_OUTPUT",
                "enable_structured_output",
                default=True,
            )
        ),
        ollama_structured_output_mode=str(
            first(
                "LLM_OLLAMA_STRUCTURED_OUTPUT_MODE",
                "ollama_structured_output_mode",
                default=DEFAULT_OLLAMA_STRUCTURED_OUTPUT_MODE,
            )
        ).strip().lower(),
    )

    if provider_adapter is not None:
        adapter = provider_adapter
    else:
        normalized_provider = str(provider).strip().lower()
        if normalized_provider in {"ollama", "ollama_local", "local_ollama"}:
            adapter = OllamaProvider()
        elif normalized_provider in {
            "openai",
            "openai_compatible",
            "openai-compatible",
            "azure_openai",
            "azure-openai",
        }:
            adapter = OpenAICompatibleProvider()
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider: {provider!r}. "
                "Supported providers are 'ollama' and 'openai_compatible'."
            )

    return LLMClient(
        config,
        provider=adapter,
    )


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LLMConfigurationError(f"{name} must be an integer.")
    try:
        converted = int(value)
        return converted
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMConfigurationError(
            f"{name} must be an integer."
        ) from exc


def _coerce_optional_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _coerce_int(value, name)


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LLMConfigurationError(f"{name} must be numeric.")
    try:
        converted = float(value)
        if not math.isfinite(converted):
            raise LLMConfigurationError(f"{name} must be finite.")
        return converted
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMConfigurationError(
            f"{name} must be numeric."
        ) from exc


def _coerce_optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    return _coerce_float(value, name)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise LLMConfigurationError(
        "Boolean configuration value is invalid."
    )


# ============================================================================
# Deterministic mock provider for unit tests
# ============================================================================


class FakeLLMProvider:
    """Small dependency-injection provider for offline tests."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        responses: Optional[Sequence[ProviderResponse]] = None,
        failures: Optional[Sequence[BaseException]] = None,
    ) -> None:
        self._responses = list(responses or [])
        self._failures = list(failures or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        request: Any,
        *,
        config: LLMClientConfig,
        request_id: str,
    ) -> ProviderResponse:
        messages = _normalize_messages(request)

        # Capture exact prompt content for the critical prompt-integrity test.
        self.calls.append(
            {
                "request_id": request_id,
                "messages": messages,
                "task_type": _get(request, "task_type"),
                "prompt_version": _get(request, "prompt_version"),
                "response_schema": _get(request, "response_schema"),
                "query_fingerprint": _get(request, "query_fingerprint"),
                "evidence_fingerprint": _get(request, "evidence_fingerprint"),
            }
        )

        if self._failures:
            raise self._failures.pop(0)

        if not self._responses:
            return ProviderResponse(
                text="This is a test response.",
                model=config.model,
                finish_reason="stop",
                usage=LLMUsage(
                    input_tokens=10,
                    output_tokens=6,
                    total_tokens=16,
                ),
                provider_request_id="fake-provider-request",
            )

        return self._responses.pop(0)


# ============================================================================
# Offline tests
# ============================================================================


@dataclass(frozen=True)
class _FakeMessage:
    role: str
    content: str


@dataclass(frozen=True)
class _FakeRequest:
    messages: tuple[_FakeMessage, ...]
    task_type: str = "qa"
    prompt_version: str = "test-v1"
    response_schema: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    temperature: Optional[float] = 0.0
    request_id: str = "000000000000000000000000"
    query_fingerprint: str = "0000000000000000"
    evidence_fingerprint: str = "0000000000000000"


class _TimeoutProvider:
    provider_name = "fake-timeout"

    def generate(self, request: Any, *, config: LLMClientConfig, request_id: str) -> ProviderResponse:
        raise LLMTimeoutError("timeout")


class _AuthProvider:
    provider_name = "fake-auth"

    def generate(self, request: Any, *, config: LLMClientConfig, request_id: str) -> ProviderResponse:
        raise LLMAuthenticationError("auth failure")


class _TransientThenSuccessProvider:
    provider_name = "fake-transient"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: Any, *, config: LLMClientConfig, request_id: str) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            raise LLMNetworkError("temporary network failure")
        return ProviderResponse(
            text="This is a test response.",
            model=config.model,
            finish_reason="stop",
        )


class _MalformedProvider:
    provider_name = "fake-malformed"

    def generate(self, request: Any, *, config: LLMClientConfig, request_id: str) -> Any:
        return {"text": "not a ProviderResponse"}


def run_self_test() -> None:
    """Run all critical client tests without network, credentials or SDKs."""

    messages = (
        _FakeMessage(
            role="system",
            content="SYSTEM: answer only from supplied evidence.",
        ),
        _FakeMessage(
            role="user",
            content=(
                "<RESEARCH_EVIDENCE>\n"
                "Paper title: Test Paper\n"
                "Page: 3\n"
                "Dataset: MedSeg\n"
                "</RESEARCH_EVIDENCE>\n"
                "Question: What dataset was used?"
            ),
        ),
    )
    request = _FakeRequest(messages=messages)

    # Use the real production LLMRequest contract for client-boundary tests.
    # This prevents the test suite from accidentally validating a legacy request
    # shape that production PromptBuilder no longer emits.
    request = PromptBuilder().build_qa_prompt(
        {
            "formatted_context": (
                "[EVIDENCE 1]\nSOURCE_TEXT_BEGIN\nThe dataset is MedSeg.\n"
                "SOURCE_TEXT_END\n[END EVIDENCE 1]"
            ),
            "evidence_items": [
                {
                    "document_id": "test-paper",
                    "paper_id": "test-paper",
                    "chunk_id": "test-chunk",
                    "section": "Methods",
                    "page": 3,
                    "text": "The dataset is MedSeg.",
                }
            ],
        },
        "What dataset was used?",
    )

    config = LLMClientConfig(
        provider="fake",
        model="test-model",
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_output_tokens=512,
        timeout_seconds=5.0,
        max_retries=2,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        jitter_seconds=0.0,
    )

    # 1. Successful generation + normalized response.
    fake = FakeLLMProvider()
    client = LLMClient(
        config,
        provider=fake
    )
    response = client.generate(request)

    assert response.raw_text == "This is a test response."
    assert response.text == response.raw_text
    assert response.content == response.raw_text
    assert response.provider == "fake"
    assert response.model == "test-model"
    assert response.request_id == request.request_id
    assert response.usage is not None
    assert response.usage.total_tokens == 16
    assert len(fake.calls) == 1

    # 2. Critical prompt-integrity test: exact content must pass through.
    assert fake.calls[0]["messages"] == tuple(
        message.to_dict() for message in request.messages
    )
    assert fake.calls[0]["request_id"] == request.request_id
    assert fake.calls[0]["query_fingerprint"] == request.query_fingerprint
    assert fake.calls[0]["evidence_fingerprint"] == request.evidence_fingerprint

    # 3. Invalid requests.
    invalid_requests = [
        None,
        _FakeRequest(messages=()),
        _FakeRequest(
            messages=(
                _FakeMessage(role="system", content=""),
            )
        ),
        {
            "messages": (
                {"role": "system", "content": "ok"},
            ),
            "task_type": "qa",
            "prompt_version": "",
            "response_schema": {},
        },
    ]

    for invalid in invalid_requests:
        try:
            client.generate(invalid)
        except LLMRequestError:
            pass
        except (AttributeError, TypeError):
            # The test contract expects these to be normalized into
            # LLMRequestError rather than leaking Python internals.
            raise AssertionError(
                f"Invalid request leaked raw exception: {invalid!r}"
            )
        else:
            raise AssertionError(
                f"Invalid request was accepted: {invalid!r}"
            )

    # 3b. Non-JSON-serializable response schemas fail at the client boundary.
    bad_schema_request = _FakeRequest(
        messages=messages,
        response_schema={"type": "object", "bad": {1, 2, 3}},
    )
    try:
        client.generate(bad_schema_request)
    except LLMRequestError:
        pass
    else:
        raise AssertionError("Non-serializable response schema was accepted.")

    # 3c. A different API query must be rejected before provider dispatch.
    try:
        client.generate(request, query="What model was used?")
    except LLMRequestError:
        pass
    else:
        raise AssertionError("Mismatched API query was accepted.")

    # 3d. The client must use PromptBuilder's request identity, not mint a new ID.
    assert response.request_id == request.request_id
    assert fake.calls[0]["request_id"] == request.request_id
    assert fake.calls[0]["query_fingerprint"] == request.query_fingerprint
    assert fake.calls[0]["evidence_fingerprint"] == request.evidence_fingerprint

    # 4. Timeout classification.
    timeout_client = LLMClient(
        config,
        provider=_TimeoutProvider()
    )
    try:
        timeout_client.generate(request)
    except LLMTimeoutError:
        pass
    else:
        raise AssertionError("Timeout was not classified.")

    # 5. Authentication must never retry.
    auth_provider = _AuthProvider()
    auth_client = LLMClient(
        config,
        provider=auth_provider
    )
    try:
        auth_client.generate(request)
    except LLMAuthenticationError:
        pass
    else:
        raise AssertionError("Authentication failure was not classified.")

    # 6. Transient failure retries once and succeeds.
    transient_provider = _TransientThenSuccessProvider()
    transient_config = LLMClientConfig(
        provider="fake",
        model="test-model",
        timeout_seconds=5.0,
        max_retries=2,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        jitter_seconds=0.0,
    )
    transient_client = LLMClient(
        transient_config,
        provider=transient_provider,
        sleeper=lambda _: None,
        random_source=lambda: 0.0,
    )
    transient_response = transient_client.generate(request)
    assert transient_response.raw_text == "This is a test response."
    assert transient_provider.calls == 2

    # 7. Malformed provider response.
    malformed_client = LLMClient(
        config,
        provider=_MalformedProvider(),
    )
    try:
        malformed_client.generate(request)
    except LLMResponseError:
        pass
    else:
        raise AssertionError("Malformed provider response was accepted.")

    # 8. Empty provider response.
    empty_client = LLMClient(
        config,
        provider=FakeLLMProvider(
            responses=[
                ProviderResponse(
                    text="   ",
                    model="test-model",
                )
            ]
        ),
    )
    try:
        empty_client.generate(request)
    except LLMEmptyResponseError:
        pass
    else:
        raise AssertionError("Empty provider response was accepted.")

    # 9. Request ID propagation.
    propagated_provider = FakeLLMProvider()
    propagated_client = LLMClient(
        config,
        provider=propagated_provider
    )
    propagated_client.generate(request)
    assert propagated_provider.calls[0]["request_id"] == request.request_id

    # 10. Long prompt is transported without client-side truncation.
    long_text = "X" * 100_000
    long_request = PromptBuilder().build_qa_prompt(
        {
            "formatted_context": (
                "[EVIDENCE 1]\nSOURCE_TEXT_BEGIN\n"
                + long_text
                + "\nSOURCE_TEXT_END\n[END EVIDENCE 1]"
            ),
            "evidence_items": [
                {
                    "document_id": "long-paper",
                    "paper_id": "long-paper",
                    "chunk_id": "long-chunk",
                    "section": "Methods",
                    "page": 1,
                    "text": long_text,
                }
            ],
        },
        "What is in the evidence?",
    )
    long_provider = FakeLLMProvider()
    long_client = LLMClient(config, provider=long_provider)
    long_client.generate(long_request)
    assert long_text in long_provider.calls[0]["messages"][1]["content"]

    # 11. No retrieval/FAISS/embedding/PDF library imports from this module.
    # Inspect import statements rather than source text so architectural
    # terminology in documentation does not trigger a false positive.
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module.lower())

    forbidden_roots = {
        "faiss",
        "torch",
        "transformers",
        "sentence_transformers",
        "pymupdf",
        "fitz",
    }
    for module_name in imported_modules:
        root = module_name.split(".")[0]
        assert root not in forbidden_roots, (
            f"Forbidden dependency imported: {module_name}"
        )

    # 11b. Oversized provider bodies are rejected before JSON parsing.
    class _OversizedHTTPResponse:
        def __init__(self) -> None:
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            return b"X" * (size + 1)

        def close(self) -> None:
            self.closed = True

    try:
        OpenAICompatibleProvider._decode_success_response(
            _OversizedHTTPResponse(),
            fallback_model="test-model",
        )
    except LLMResponseError:
        pass
    else:
        raise AssertionError("Oversized provider response was accepted.")

    # 12. Configuration validation.
    try:
        LLMClientConfig(
            provider="",
            model="test-model",
        )
    except LLMConfigurationError:
        pass
    else:
        raise AssertionError("Empty provider configuration was accepted.")

    # 13. Local Ollama adapter payload/response normalization using a fake
    # transport. No Ollama process or network is required for this test.
    class _FakeHTTPResponse:
        def __init__(self, payload: Mapping[str, Any]) -> None:
            self._body = json.dumps(payload).encode("utf-8")
            self.closed = False

        def read(self) -> bytes:
            return self._body

        def close(self) -> None:
            self.closed = True

    captured: dict[str, Any] = {}

    def _fake_ollama_opener(req: Any, *, timeout: float) -> _FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "model": "qwen2.5:7b",
                "created_at": "2026-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"local test"}',
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 7,
                "total_duration": 123456,
            }
        )

    ollama_provider = OllamaProvider(opener=_fake_ollama_opener)
    ollama_config = LLMClientConfig(
        provider="ollama",
        model="qwen2.5:7b",
        api_key=None,
        base_url="http://127.0.0.1:11434/",
        temperature=0.0,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        jitter_seconds=0.0,
        enable_structured_output=True,
    )
    ollama_client = LLMClient(
        ollama_config,
        provider=ollama_provider
    )
    ollama_response = ollama_client.generate(request)
    assert ollama_response.provider == "ollama"
    assert ollama_response.model == "qwen2.5:7b"
    assert ollama_response.raw_text == '{"answer":"local test"}'
    assert ollama_response.usage is not None
    assert ollama_response.usage.total_tokens == 19
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["model"] == "qwen2.5:7b"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_predict"] == 256
    assert isinstance(captured["body"]["format"], dict)
    assert captured["body"]["format"] == dict(request.response_schema)
    assert captured["body"]["messages"] == [
        message.to_dict() for message in request.messages
    ]
    assert captured["headers"]["X-request-id"] == request.request_id
    assert captured["headers"]["X-query-fingerprint"] == request.query_fingerprint
    assert captured["headers"]["X-evidence-fingerprint"] == request.evidence_fingerprint
    assert ollama_response.request_id == request.request_id
    assert ollama_response.metadata["structured_output_mode"] == "transport_only"
    # Production default: send the canonical schema to Ollama. If a local
    # runtime rejects it, OllamaProvider has a bounded compatibility fallback.
    assert ollama_config.ollama_structured_output_mode == "schema"

    # 13b. OpenAI-compatible adapter preserves schema transport and metadata.
    openai_captured: dict[str, Any] = {}

    def _fake_openai_opener(req: Any, *, timeout: float) -> _FakeHTTPResponse:
        openai_captured["url"] = req.full_url
        openai_captured["body"] = json.loads(req.data.decode("utf-8"))
        openai_captured["headers"] = dict(req.headers)
        return _FakeHTTPResponse(
            {
                "id": "openai-test-id",
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"compatible"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                },
            }
        )

    openai_provider = OpenAICompatibleProvider(opener=_fake_openai_opener)
    openai_config = LLMClientConfig(
        provider="openai_compatible",
        model="test-model",
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        max_retries=0,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        jitter_seconds=0.0,
    )
    openai_client = LLMClient(
        openai_config,
        provider=openai_provider
    )
    openai_response = openai_client.generate(request)
    assert openai_response.raw_text == '{"answer":"compatible"}'
    assert openai_response.usage is not None
    assert openai_response.usage.total_tokens == 16
    assert openai_captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert openai_captured["body"]["response_format"]["type"] == "json_schema"
    assert (
        openai_captured["body"]["response_format"]["json_schema"]["schema"]
        == dict(request.response_schema)
    )

    # 14. Factory selects Ollama without an API key.
    factory_client = create_client_from_mapping(
        {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen2.5:7b",
            "LLM_BASE_URL": "http://127.0.0.1:11434/",
        }
    )
    assert factory_client.provider.provider_name == "ollama"
    assert factory_client.config.api_key is None

    # 15. Ollama schema compatibility fallback: HTTP 400 on schema
    # transparently retries once WITHOUT the format field.
    class _SchemaRejectingHTTPResponse:
        def __init__(self, payload: Mapping[str, Any]) -> None:
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def close(self) -> None:
            pass

    schema_calls: list[dict[str, Any]] = []

    def _schema_fallback_opener(req: Any, *, timeout: float) -> Any:
        body = json.loads(req.data.decode("utf-8"))
        schema_calls.append(body)
        if len(schema_calls) == 1:
            raise urllib_error.HTTPError(
                req.full_url,
                400,
                "schema unsupported",
                {"Content-Type": "application/json"},
                __import__("io").BytesIO(b'{"error":"unsupported format"}'),
            )
        return _SchemaRejectingHTTPResponse(
            {
                "model": "qwen2.5:7b",
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"fallback"}',
                },
                "done": True,
                "done_reason": "stop",
            }
        )

    fallback_provider = OllamaProvider(opener=_schema_fallback_opener)
    fallback_config = LLMClientConfig(
        provider="ollama",
        model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/api",
        max_retries=0,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        jitter_seconds=0.0,
        ollama_structured_output_mode="schema",
    )
    fallback_client = LLMClient(
        fallback_config,
        provider=fallback_provider
    )
    fallback_response = fallback_client.generate(request)
    assert fallback_response.raw_text == '{"answer":"fallback"}'
    assert len(schema_calls) == 2
    assert isinstance(schema_calls[0]["format"], dict)
    assert "format" not in schema_calls[1]

    # 16. Session presence is propagated as non-sensitive metadata.
    session_client = LLMClient(
        config,
        provider=FakeLLMProvider()
    )
    session_response = session_client.generate(
        request,
        session_id="session-123",
        task_type="qa",
        query="What dataset was used?",
    )
    assert session_response.metadata["session_present"] is True

    print("LLMClient self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()