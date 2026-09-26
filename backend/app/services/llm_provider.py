"""Structured completion clients for question planning and answer synthesis."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Protocol

from pydantic import SecretStr

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    name: str
    is_production_grade: bool

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]: ...


class LLMError(RuntimeError):
    """Base exception for structured completion failures."""


class LLMConfigurationError(LLMError):
    """Required provider configuration is absent or invalid."""


class LLMResponseError(LLMError):
    """A provider returned output that cannot satisfy the requested schema."""


def _matches_json_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> bool:
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/"):
        target: Any = root_schema
        for component in reference[2:].split("/"):
            if not isinstance(target, dict) or component not in target:
                return False
            target = target[component]
        return isinstance(target, dict) and _matches_json_schema(
            value, target, root_schema
        )

    if "anyOf" in schema and not any(
        _matches_json_schema(value, option, root_schema)
        for option in schema["anyOf"]
    ):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "const" in schema and value != schema["const"]:
        return False

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        if any(key not in value for key in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        return all(
            key not in properties
            or _matches_json_schema(item, properties[key], root_schema)
            for key, item in value.items()
        )
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        if len(value) < schema.get("minItems", 0):
            return False
        item_schema = schema.get("items")
        return not isinstance(item_schema, dict) or all(
            _matches_json_schema(item, item_schema, root_schema) for item in value
        )
    if expected_type == "string":
        return isinstance(value, str) and len(value) <= schema.get(
            "maxLength", len(value)
        )
    if expected_type == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= schema.get("minimum", value)
            and value <= schema.get("maximum", value)
        )
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "default":
            continue
        if isinstance(value, dict):
            strict[key] = _strict_json_schema(value)
        elif isinstance(value, list):
            strict[key] = [
                _strict_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            strict[key] = value

    properties = strict.get("properties")
    if strict.get("type") == "object" and isinstance(properties, dict):
        strict["additionalProperties"] = False
        strict["required"] = list(properties)
    return strict


class FakeLLMClient:
    """Deterministic client that performs no real analysis and exists only for tests."""

    name = "fake"
    is_production_grade = False

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        seed = json.dumps(
            {
                "model": model,
                "system": system,
                "user": user,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        properties = schema.get("properties", {})

        if "intent" in properties:
            try:
                payload = json.loads(user)
                question = str(payload.get("question", user))
            except (json.JSONDecodeError, AttributeError):
                question = user
            return {
                "intent": "semantic_evidence",
                "semantic_query": question[:512],
                "app_ids": None,
                "countries": ["US"],
                "ratings": None,
                "date_from": None,
                "date_to": None,
                "top_k": 8,
                "group_by": None,
                "requested_metrics": [],
                "needs_semantic_search": True,
                "needs_aggregation": False,
                "planner_notes": f"test-{digest[:12]}",
            }

        if "answer" in properties:
            return {
                "answer": f"Deterministic test response {digest[:12]}",
                "findings": [],
            }

        return {"test_digest": digest}


class OpenAILLMClient:
    """OpenAI strict-JSON client with bounded retries."""

    name = "openai"
    is_production_grade = True

    def __init__(
        self,
        api_key: SecretStr | None,
        *,
        timeout_seconds: int = 60,
        client: Any | None = None,
    ) -> None:
        if api_key is None:
            raise LLMConfigurationError(
                "APP_OPENAI_API_KEY is required when APP_LLM_PROVIDER=openai"
            )
        if timeout_seconds < 1:
            raise LLMConfigurationError("APP_LLM_TIMEOUT_SECONDS must be positive")

        if client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key.get_secret_value(),
                timeout=timeout_seconds,
                max_retries=0,
            )
        else:
            self._client = client

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        response: Any | None = None
        reasoning_kwargs = (
            {"reasoning_effort": reasoning_effort}
            if reasoning_effort is not None
            else {}
        )
        for attempt in range(5):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "strict": True,
                            "schema": _strict_json_schema(schema),
                        },
                    },
                    max_completion_tokens=max_output_tokens,
                    **reasoning_kwargs,
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code == 429 or (
                    isinstance(status_code, int) and status_code >= 500
                )
                if not retryable or attempt == 4:
                    if status_code is None:
                        # The exception type is safe to surface and is usually the
                        # only clue for a timeout or connection failure; the message
                        # is dropped because it can embed request details.
                        raise LLMError(
                            "LLM provider request failed "
                            f"({type(exc).__name__})"
                        ) from None
                    raise LLMError(
                        f"LLM provider request failed with HTTP status {status_code}"
                    ) from None
                time.sleep(min(0.5 * (2**attempt), 4.0))

        if response is None:
            raise LLMError("LLM provider request failed")

        usage = getattr(response, "usage", None)
        if on_usage is not None and usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if (
                total_tokens is None
                and input_tokens is not None
                and output_tokens is not None
            ):
                total_tokens = input_tokens + output_tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is None:
                prompt_details = getattr(usage, "input_tokens_details", None)
            cached_input_tokens = (
                getattr(prompt_details, "cached_tokens", None)
                if prompt_details is not None
                else None
            )
            on_usage(
                {
                    "provider": self.name,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cached_input_tokens": cached_input_tokens,
                }
            )

        # A reasoning model can spend the whole token budget before emitting output,
        # which arrives as a successful response with empty content. Reported as
        # truncation so the budget, rather than the model's JSON, is what gets blamed.
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise LLMResponseError(
                "LLM provider truncated the response before returning complete JSON; "
                "increase APP_LLM_MAX_OUTPUT_TOKENS"
            )

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
            raise LLMResponseError(
                "LLM provider returned invalid structured JSON"
            ) from None
        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM provider returned a non-object JSON value")
        if not _matches_json_schema(parsed, schema, schema):
            raise LLMResponseError(
                "LLM provider returned JSON that does not match the requested schema"
            )
        return parsed


def build_llm_client(settings: Settings) -> LLMClient:
    """Construct the configured client and log only its public name."""
    if settings.llm_provider == "fake":
        client: LLMClient = FakeLLMClient()
    elif settings.llm_provider == "openai":
        client = OpenAILLMClient(
            settings.openai_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    else:
        raise LLMConfigurationError(
            "APP_LLM_PROVIDER must be either 'fake' or 'openai'"
        )
    logger.info("LLM provider selected: %s", client.name)
    return client
