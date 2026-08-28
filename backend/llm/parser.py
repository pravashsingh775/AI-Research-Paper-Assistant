"""
Application-level LLM output parser for the AI Research Paper Assistant.

Boundary
--------
    backend/llm/client.py
            ↓
    normalized/raw LLM response
            ↓
    backend/llm/parser.py
            ↓
    strict structured Mapping
            ↓
    validation/evidence.py
    validation/validator.py
            ↓
    rag/pipeline.py / API / frontend

This module is deliberately NOT responsible for:
    * retrieval / FAISS / embeddings / reranking
    * context construction
    * prompt generation
    * LLM API calls
    * PDF processing
    * scientific/factual verification
    * hallucination detection
    * evidence-to-document matching

The parser performs transport extraction + application-level structural
validation only. It does not decide whether a scientific claim is true.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Public exceptions
# ============================================================================


class LLMParserError(RuntimeError):
    """Base exception for application-level LLM parsing failures."""


class LLMParserInputError(LLMParserError, TypeError):
    """Input cannot be interpreted as an LLM response."""


class LLMParserJSONError(LLMParserError, ValueError):
    """The LLM response is not valid JSON in the supported formats."""


class LLMParserSchemaError(LLMParserError, ValueError):
    """The parsed JSON does not satisfy the project's task schema."""


class LLMParserTaskError(LLMParserError, ValueError):
    """The requested task type is unsupported or missing."""


# ============================================================================
# Immutable parsed result
# ============================================================================


@dataclass(frozen=True)
class ParsedLLMOutput(Mapping[str, Any]):
    """
    Typed metadata wrapper around the exact structured application payload.

    The object implements Mapping so existing validators/API code can consume
    it exactly like a normal dictionary:

        result["answer"]
        result.get("evidence")
        dict(result)

    `data` contains ONLY the schema-defined LLM fields. Transport metadata is
    kept separately and is never inserted into the application's structured
    answer, which preserves the prompt's `additionalProperties: false`
    contract.
    """

    data: Mapping[str, Any]
    task_type: str
    raw_text: str
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    provider_request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        if not isinstance(self.data, Mapping):
            raise TypeError("ParsedLLMOutput.data must be a mapping.")

        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("ParsedLLMOutput.task_type must be non-empty.")

        if not isinstance(self.raw_text, str):
            raise TypeError("ParsedLLMOutput.raw_text must be a string.")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def to_dict(self) -> dict[str, Any]:
        """Return the pure application payload without transport metadata."""
        return _deep_copy_json_value(self.data)

    def to_record(self) -> dict[str, Any]:
        """
        Return a JSON-safe record containing payload + parser metadata.

        This is intended for diagnostics/API serialization, not as the input
        to the evidence validator.
        """
        return {
            "task_type": self.task_type,
            "data": self.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "provider_request_id": self.provider_request_id,
            "finish_reason": self.finish_reason,
            "metadata": _deep_copy_json_value(self.metadata),
        }

    @property
    def grounded(self) -> Optional[bool]:
        value = self.data.get("grounded")
        return value if isinstance(value, bool) else None

    @property
    def evidence(self) -> tuple[Mapping[str, Any], ...]:
        value = self.data.get("evidence")
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))


# ============================================================================
# Project task/schema integration
# ============================================================================


SUPPORTED_TASK_TYPES = frozenset(
    {
        "qa",
        "summary",
        "methodology",
        "dataset",
        "model",
        "findings",
        "strengths",
        "weaknesses",
        "comparison",
    }
)


def _load_project_schema(task_type: str) -> Mapping[str, Any]:
    """
    Reuse the authoritative schemas from rag/prompts.py.

    The parser intentionally does not duplicate the project's output schemas.
    If prompts.py is unavailable during an isolated unit test, a small exact
    compatibility schema is available as a fallback. In normal application
    execution, the project schema is the source of truth.
    """
    try:
        from backend.rag.prompts import get_output_schema
    except ImportError:
        try:
            from rag.prompts import get_output_schema  # type: ignore
        except ImportError:
            return _fallback_schema(task_type)

    try:
        schema = get_output_schema(task_type)
    except Exception as exc:
        raise LLMParserTaskError(
            f"Unable to resolve output schema for task_type={task_type!r}."
        ) from exc

    if not isinstance(schema, Mapping):
        raise LLMParserSchemaError(
            "PromptBuilder returned a non-mapping output schema."
        )

    return _deep_copy_json_value(schema)


def _fallback_schema(task_type: str) -> Mapping[str, Any]:
    """
    Exact self-contained fallback for environments where backend.rag.prompts
    cannot be imported.

    The fallback MUST match the canonical contract:
      * `question` is optional compatibility output.
      * `grounded` is required.
      * `evidence` is an array of integer evidence references.
      * provenance is NEVER part of the model output.
    """
    question = {"type": "string", "minLength": 1}
    grounded = {"type": "boolean"}
    evidence = {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
    }
    string = {"type": "string"}
    strings = {"type": "array", "items": {"type": "string"}}

    def schema(
        name: str,
        properties: Mapping[str, Any],
        required: Sequence[str],
    ) -> dict[str, Any]:
        merged = {
            "question": question,
            "grounded": grounded,
            "evidence": evidence,
        }
        merged.update(properties)
        return {
            "name": name,
            "type": "object",
            "additionalProperties": False,
            "properties": merged,
            "required": list(dict.fromkeys([*required, "grounded", "evidence"])),
        }

    schemas = {
        "qa": schema(
            "research_paper_qa",
            {"answer": string},
            ["answer"],
        ),
        "summary": schema(
            "research_paper_summary",
            {
                "summary": string,
                "problem": string,
                "methodology": string,
                "dataset": string,
                "model": string,
                "key_findings": strings,
                "limitations": strings,
            },
            [
                "summary", "problem", "methodology", "dataset", "model",
                "key_findings", "limitations",
            ],
        ),
        "methodology": schema(
            "research_paper_methodology",
            {
                "methodology": string,
                "major_steps": strings,
                "models_algorithms": strings,
                "experimental_setup": string,
            },
            ["methodology", "major_steps", "models_algorithms", "experimental_setup"],
        ),
        "dataset": schema(
            "research_paper_dataset",
            {
                "dataset_name": string,
                "source": string,
                "size": string,
                "number_of_classes": string,
                "splits": string,
                "preprocessing": string,
                "augmentation": string,
                "other_details": strings,
            },
            [
                "dataset_name", "source", "size", "number_of_classes", "splits",
                "preprocessing", "augmentation", "other_details",
            ],
        ),
        "model": schema(
            "research_paper_model",
            {
                "model_name": string,
                "architecture": string,
                "major_components": strings,
                "training_approach": string,
                "hyperparameters": strings,
                "baselines": strings,
            },
            [
                "model_name", "architecture", "major_components",
                "training_approach", "hyperparameters", "baselines",
            ],
        ),
        "findings": schema(
            "research_paper_findings",
            {
                "key_findings": strings,
                "reported_metrics": strings,
                "comparisons": strings,
                "experimental_conclusions": strings,
            },
            [
                "key_findings", "reported_metrics", "comparisons",
                "experimental_conclusions",
            ],
        ),
        "strengths": schema(
            "research_paper_strengths",
            {
                "strengths": strings,
                "author_stated_strengths": strings,
                "inferred_strengths": strings,
            },
            ["strengths", "author_stated_strengths", "inferred_strengths"],
        ),
        "weaknesses": schema(
            "research_paper_weaknesses",
            {
                "weaknesses": strings,
                "author_stated_limitations": strings,
                "inferred_limitations": strings,
            },
            ["weaknesses", "author_stated_limitations", "inferred_limitations"],
        ),
        "comparison": schema(
            "research_paper_comparison",
            {
                "comparison": strings,
                "paper_specific_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "paper_label": {"type": "string"},
                            "document_id": {"type": ["string", "null"]},
                            "findings": strings,
                        },
                        "required": [
                            "paper_label",
                            "document_id",
                            "findings",
                        ],
                    },
                },
            },
            ["comparison", "paper_specific_findings"],
        ),
    }

    if task_type not in schemas:
        raise LLMParserTaskError(
            f"Unsupported task_type={task_type!r}. "
            f"Supported values: {sorted(schemas)!r}."
        )

    return schemas[task_type]


# ============================================================================
# Parser
# ============================================================================


class LLMOutputParser:
    """
    Strict, deterministic application-level LLM output parser.

    Primary API
    -----------
        parse(raw_output, *, task_type=None, request=None)

    `raw_output` may be:
        * backend.llm.client.LLMResponse
        * a compatible object exposing raw_text/content/text
        * a mapping containing raw_text/content/text
        * a JSON string
        * a markdown-fenced JSON string
        * a directly structured mapping

    The parser does not call the LLM and does not consult retrieval/context.
    """

    def __init__(
        self,
        *,
        supported_tasks: Optional[Sequence[str]] = None,
        allow_markdown_fences: bool = True,
        allow_direct_mapping: bool = True,
        allow_python_literal_fallback: bool = False,
        allow_embedded_json: bool = True,
        max_input_characters: int = 2_000_000,
        max_json_object_characters: int = 512_000,
    ) -> None:
        tasks = frozenset(supported_tasks or SUPPORTED_TASK_TYPES)

        unknown = tasks.difference(SUPPORTED_TASK_TYPES)
        if unknown:
            raise LLMParserTaskError(
                f"Unsupported parser task(s): {sorted(unknown)!r}."
            )

        if (
            isinstance(max_input_characters, bool)
            or not isinstance(max_input_characters, int)
            or max_input_characters <= 0
        ):
            raise ValueError(
                "max_input_characters must be a positive integer."
            )

        if (
            isinstance(max_json_object_characters, bool)
            or not isinstance(max_json_object_characters, int)
            or max_json_object_characters <= 0
            or max_json_object_characters > max_input_characters
        ):
            raise ValueError(
                "max_json_object_characters must be a positive integer "
                "not greater than max_input_characters."
            )

        self.supported_tasks = tasks
        self.allow_markdown_fences = bool(allow_markdown_fences)
        self.allow_direct_mapping = bool(allow_direct_mapping)
        self.allow_python_literal_fallback = bool(
            allow_python_literal_fallback
        )
        self.allow_embedded_json = bool(allow_embedded_json)
        self.max_input_characters = max_input_characters
        self.max_json_object_characters = max_json_object_characters

    def parse(
        self,
        raw_output: Any,
        *,
        request: Any = None,
        task_type: Optional[str] = None,
    ) -> ParsedLLMOutput:
        """
        Parse and structurally validate one LLM response.

        The method intentionally accepts `request` for compatibility with
        rag/pipeline.py but does not use the request to modify the output.
        If task_type is omitted, it is resolved from the request when possible.
        """
        normalized_task = self._resolve_task_type(
            task_type=task_type,
            request=request,
            raw_output=raw_output,
        )

        extracted = self._extract_transport_response(raw_output)

        if isinstance(extracted.payload, Mapping):
            if not self.allow_direct_mapping:
                raise LLMParserInputError(
                    "Direct structured mappings are disabled."
                )
            payload = _deep_copy_json_value(extracted.payload)
        elif isinstance(extracted.payload, str):
            payload = self._parse_json_text(extracted.payload)
        else:
            raise LLMParserInputError(
                "LLM output must be text or a structured mapping."
            )

        if not isinstance(payload, Mapping):
            raise LLMParserSchemaError(
                "LLM output root must be a JSON object."
            )

        schema = _load_project_schema(normalized_task)
        _validate_against_schema(
            payload,
            schema,
            path="$",
        )

        clean_payload = _deep_copy_json_value(payload)

        result = ParsedLLMOutput(
            data=clean_payload,
            task_type=normalized_task,
            raw_text=extracted.raw_text,
            provider=extracted.provider,
            model=extracted.model,
            request_id=extracted.request_id,
            provider_request_id=extracted.provider_request_id,
            finish_reason=extracted.finish_reason,
            metadata=extracted.metadata,
        )

        logger.debug(
            "LLM output parsed successfully: task=%s provider=%s model=%s",
            normalized_task,
            extracted.provider,
            extracted.model,
        )

        return result

    # ------------------------------------------------------------------
    # Task resolution
    # ------------------------------------------------------------------

    def _resolve_task_type(
        self,
        *,
        task_type: Optional[str],
        request: Any,
        raw_output: Any,
    ) -> str:
        candidate = task_type

        if candidate is None and request is not None:
            candidate = _read(request, "task_type")

        if candidate is None:
            candidate = _read(raw_output, "task_type")

        if not isinstance(candidate, str):
            raise LLMParserTaskError(
                "task_type is required. Supply task_type or a request "
                "containing task_type."
            )

        normalized = candidate.strip().lower()

        if normalized not in self.supported_tasks:
            raise LLMParserTaskError(
                f"Unsupported task_type={normalized!r}. "
                f"Supported values: {sorted(self.supported_tasks)!r}."
            )

        return normalized

    # ------------------------------------------------------------------
    # Transport extraction
    # ------------------------------------------------------------------

    def _extract_transport_response(
        self,
        raw_output: Any,
    ) -> "_ExtractedResponse":
        if raw_output is None:
            raise LLMParserInputError("raw_output cannot be None.")

        if isinstance(raw_output, str):
            if not raw_output.strip():
                raise LLMParserJSONError(
                    "LLM output text is empty or whitespace-only."
                )
            return _ExtractedResponse(
                payload=raw_output,
                raw_text=raw_output,
                provider=None,
                model=None,
                request_id=None,
                provider_request_id=None,
                finish_reason=None,
                metadata={},
            )

        if isinstance(raw_output, ParsedLLMOutput):
            return _ExtractedResponse(
                payload=raw_output.to_dict(),
                raw_text=raw_output.raw_text,
                provider=raw_output.provider,
                model=raw_output.model,
                request_id=raw_output.request_id,
                provider_request_id=raw_output.provider_request_id,
                finish_reason=raw_output.finish_reason,
                metadata=raw_output.metadata,
            )

        provider = _read(raw_output, "provider")
        model = _read(raw_output, "model")
        request_id = _read(raw_output, "request_id")
        provider_request_id = _read(raw_output, "provider_request_id")
        finish_reason = _read(raw_output, "finish_reason")
        metadata = _read(raw_output, "metadata", {})

        raw_text = _read(raw_output, "raw_text", None)
        if raw_text is None:
            raw_text = _read(raw_output, "content", None)
        if raw_text is None:
            raw_text = _read(raw_output, "text", None)

        if isinstance(raw_output, Mapping):
            if raw_text is None:
                return _ExtractedResponse(
                    payload=raw_output,
                    raw_text=_stable_json(raw_output),
                    provider=_string_or_none(provider),
                    model=_string_or_none(model),
                    request_id=_string_or_none(request_id),
                    provider_request_id=_string_or_none(
                        provider_request_id
                    ),
                    finish_reason=_string_or_none(finish_reason),
                    metadata=_safe_metadata(metadata),
                )

        if isinstance(raw_text, Mapping):
            return _ExtractedResponse(
                payload=raw_text,
                raw_text=_stable_json(raw_text),
                provider=_string_or_none(provider),
                model=_string_or_none(model),
                request_id=_string_or_none(request_id),
                provider_request_id=_string_or_none(provider_request_id),
                finish_reason=_string_or_none(finish_reason),
                metadata=_safe_metadata(metadata),
            )

        if not isinstance(raw_text, str):
            raise LLMParserInputError(
                "Could not extract raw_text/content/text from LLM output."
            )

        if not raw_text.strip():
            raise LLMParserJSONError(
                "LLM output text is empty or whitespace-only."
            )

        if len(raw_text) > self.max_input_characters:
            raise LLMParserInputError(
                "LLM output exceeds the configured parser input limit."
            )

        return _ExtractedResponse(
            payload=raw_text,
            raw_text=raw_text,
            provider=_string_or_none(provider),
            model=_string_or_none(model),
            request_id=_string_or_none(request_id),
            provider_request_id=_string_or_none(provider_request_id),
            finish_reason=_string_or_none(finish_reason),
            metadata=_safe_metadata(metadata),
        )

    # ------------------------------------------------------------------
    # JSON decoding
    # ------------------------------------------------------------------

    def _parse_json_text(self, text: str) -> Mapping[str, Any]:
        candidate = text.strip()

        if not candidate:
            raise LLMParserJSONError(
                "LLM output text is empty or whitespace-only."
            )

        if self.allow_markdown_fences:
            candidate = _unwrap_markdown_json(candidate)

        # Fast path: canonical JSON.
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as first_error:
            parsed = None

            if self.allow_embedded_json:
                candidates = _extract_json_object_candidates(
                    candidate,
                    max_object_characters=self.max_json_object_characters,
                )
                valid: list[Mapping[str, Any]] = []

                for fragment in candidates:
                    try:
                        value = json.loads(fragment)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, Mapping):
                        valid.append(value)

                if len(valid) == 1:
                    parsed = valid[0]
                elif len(valid) > 1:
                    raise LLMParserJSONError(
                        "LLM output contains multiple valid JSON objects; "
                        "the parser cannot determine which object is authoritative."
                    ) from first_error

            if parsed is None and self.allow_python_literal_fallback:
                try:
                    value = ast.literal_eval(candidate)
                except (ValueError, SyntaxError) as exc:
                    raise LLMParserJSONError(
                        "LLM output is neither valid JSON nor an allowed "
                        "Python literal."
                    ) from exc
                if not isinstance(value, Mapping):
                    raise LLMParserSchemaError(
                        "Structured LLM output must be a JSON object."
                    )
                parsed = value

            # Never convert arbitrary prose into a synthetic answer/evidence
            # object. Doing so would fabricate grounding/provenance and could
            # turn an invalid provider response into a false scientific answer.
            if parsed is None:
                raise LLMParserJSONError(
                    "LLM output does not contain one unambiguous valid JSON "
                    "object in a supported format."
                ) from first_error

        if not isinstance(parsed, Mapping):
            raise LLMParserSchemaError(
                "Structured LLM output must be a JSON object."
            )

        return parsed


@dataclass(frozen=True)
class _ExtractedResponse:
    payload: Any
    raw_text: str
    provider: Optional[str]
    model: Optional[str]
    request_id: Optional[str]
    provider_request_id: Optional[str]
    finish_reason: Optional[str]
    metadata: Mapping[str, Any]


# ============================================================================
# Deterministic embedded-JSON extraction
# ============================================================================

def _extract_json_object_candidates(
    text: str,
    *,
    max_object_characters: int,
) -> list[str]:
    candidates: list[str] = []
    length = len(text)
    cursor = 0

    while cursor < length:
        start = text.find("{", cursor)
        if start < 0:
            break

        depth = 0
        in_string = False
        escaped = False
        completed = False

        for index in range(start, length):
            current = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
                if index - start + 1 > max_object_characters:
                    cursor = index + 1
                    break
            elif current == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[start:index + 1]
                    if len(fragment) <= max_object_characters:
                        candidates.append(fragment)
                    cursor = index + 1
                    completed = True
                    break
        else:
            cursor = start + 1

        if not completed and cursor <= start:
            cursor = start + 1

    return candidates


# ============================================================================
# Schema validation
# ============================================================================


def _validate_against_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
) -> None:
    if not isinstance(schema, Mapping):
        raise LLMParserSchemaError(
            f"{path}: schema definition must be an object."
        )

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_type(
        value,
        expected_type,
    ):
        raise LLMParserSchemaError(
            f"{path}: expected type {expected_type!r}, "
            f"got {_json_type_name(value)!r}."
        )

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get(
            "additionalProperties",
            True,
        )

        if not isinstance(properties, Mapping):
            raise LLMParserSchemaError(
                f"{path}: schema properties must be an object."
            )

        if not isinstance(required, Sequence) or isinstance(
            required,
            (str, bytes),
        ):
            raise LLMParserSchemaError(
                f"{path}: schema required must be an array."
            )

        for field_name in required:
            if field_name not in value:
                raise LLMParserSchemaError(
                    f"{path}: missing required field {field_name!r}."
                )

        if additional_properties is False:
            unknown = set(value).difference(properties)
            if unknown:
                raise LLMParserSchemaError(
                    f"{path}: unexpected field(s): {sorted(unknown)!r}."
                )

        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                _validate_against_schema(
                    child_value,
                    child_schema,
                    path=f"{path}.{key}",
                )

    if isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema is not None:
            if not isinstance(items_schema, Mapping):
                raise LLMParserSchemaError(
                    f"{path}: array items schema must be an object."
                )

            for index, child_value in enumerate(value):
                _validate_against_schema(
                    child_value,
                    items_schema,
                    path=f"{path}[{index}]",
                )

    minimum = schema.get("minimum")
    if minimum is not None and _is_json_number(value):
        if value < minimum:
            raise LLMParserSchemaError(
                f"{path}: value {value!r} is below minimum {minimum!r}."
            )


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(
            _matches_json_type(value, item)
            for item in expected
        )

    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )
    if expected == "number":
        return _is_json_number(value)

    return False


def _is_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


# ============================================================================
# Input/output helpers
# ============================================================================


def _read(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _safe_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return _deep_copy_json_value(value)


def _deep_copy_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise LLMParserSchemaError(
                "Structured output contains a non-finite numeric value."
            )
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _deep_copy_json_value(child)
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [_deep_copy_json_value(child) for child in value]

    if isinstance(value, tuple):
        return [_deep_copy_json_value(child) for child in value]

    raise LLMParserSchemaError(
        f"Structured output contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise LLMParserInputError(
            "Structured LLM output is not JSON serializable."
        ) from exc


def _unwrap_markdown_json(text: str) -> str:
    value = text.strip()

    if not value.startswith("```") or not value.endswith("```"):
        return value

    lines = value.splitlines()

    if len(lines) < 3:
        return value

    opening = lines[0].strip().lower()
    closing = lines[-1].strip()

    if closing != "```":
        return value

    if opening in {"```json", "```jsonc", "```javascript", "```js"}:
        return "\n".join(lines[1:-1]).strip()

    body = "\n".join(lines[1:-1]).strip()
    if body.startswith("{") and body.endswith("}"):
        return body

    return value


# ============================================================================
# Backward-compatible functional API
# ============================================================================


_DEFAULT_PARSER = LLMOutputParser()


def parse_llm_output(
    raw_output: Any,
    *,
    task_type: Optional[str] = None,
    request: Any = None,
) -> ParsedLLMOutput:
    """Functional convenience wrapper around the production parser."""
    return _DEFAULT_PARSER.parse(
        raw_output,
        task_type=task_type,
        request=request,
    )


parse = parse_llm_output


# ============================================================================
# Offline self-test
# ============================================================================


@dataclass(frozen=True)
class _FakeRequest:
    task_type: str


def _qa_payload() -> dict[str, Any]:
    # Canonical QA model output: no question echo and no provenance objects.
    return {
        "answer": "The paper used MedSeg.",
        "grounded": True,
        "evidence": [1],
    }


def run_self_test() -> None:
    parser = LLMOutputParser()
    result = parser.parse(json.dumps(_qa_payload()), task_type="qa")
    assert isinstance(result, ParsedLLMOutput)
    assert result["answer"] == "The paper used MedSeg."
    assert result["grounded"] is True
    assert result["evidence"] == [1]

    # Optional compatibility question echo is accepted and structurally checked.
    with_question = dict(_qa_payload())
    with_question["question"] = "What dataset was used?"
    result_with_question = parser.parse(
        json.dumps(with_question),
        task_type="qa",
    )
    assert result_with_question["question"] == "What dataset was used?"

    # Provenance objects must never be accepted as model evidence.
    bad_provenance = dict(_qa_payload())
    bad_provenance["evidence"] = [{"chunk_id": "c1", "page": 2}]
    try:
        parser.parse(json.dumps(bad_provenance), task_type="qa")
    except LLMParserSchemaError:
        pass
    else:
        raise AssertionError("Expected provenance evidence object to fail.")

    print("LLMOutputParser self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()