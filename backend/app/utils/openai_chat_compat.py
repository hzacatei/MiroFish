"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Anthropic's OpenAI compatibility layer rejects {"type": "json_object"} with
# HTTP 400 "response_format.type must be json_schema" and does not set the
# param/code fields the generic unsupported-parameter check looks for.
# Some local servers use the same sentence, or a Pydantic-style
# "Input should be 'json_schema'".
_JSON_SCHEMA_TYPE_REJECTION = re.compile(
    r"response_format\b.*(?:must be|input should be).*json_schema\b",
    re.IGNORECASE | re.DOTALL,
)


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    return client.chat.completions.create(**kwargs)


def _message_rejects_json_object(message: str) -> bool:
    """True when the provider refuses json_object and requires json_schema."""

    normalized = message.replace("`", " ").replace("'", " ").replace('"', " ")
    return _JSON_SCHEMA_TYPE_REJECTION.search(normalized) is not None


def is_response_format_unsupported(error: Exception) -> bool:
    """Detect an explicit provider rejection of JSON response_format."""

    if getattr(error, "status_code", None) not in {400, 422}:
        return False

    body = getattr(error, "body", None)
    if isinstance(body, str):
        return _message_rejects_json_object(body)
    if not isinstance(body, dict):
        return False

    details = body.get("error", body)
    if isinstance(details, str):
        return _message_rejects_json_object(details)
    if not isinstance(details, dict):
        return False

    param = str(details.get("param") or "").strip().lower()
    if param == "response_format" or param.startswith("response_format."):
        return True

    message = str(details.get("message") or "")
    if _message_rejects_json_object(message):
        return True

    lowered = message.lower()
    if "response_format" not in lowered:
        return False

    code = str(details.get("code") or "").lower()
    unsupported_codes = {
        "unsupported_parameter",
        "unsupported_value",
        "unknown_parameter",
        "invalid_parameter",
    }
    unsupported_phrases = (
        "not support",
        "unsupported",
        "unknown parameter",
        "unrecognized parameter",
    )
    return code in unsupported_codes or any(
        phrase in lowered for phrase in unsupported_phrases
    )


def create_chat_completion_with_json_fallback(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """Send one completion, then one more without response_format if it is rejected.

    LLMClient.chat_json keeps this negotiation itself so a later content retry
    does not send json_object again. Profile and simulation-config generation
    call the OpenAI SDK directly and use this helper for the same one-shot drop.
    """

    active_format = response_format
    while True:
        try:
            return create_chat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=active_format,
            )
        except Exception as error:
            if active_format is not None and is_response_format_unsupported(error):
                logger.warning(
                    "LLM provider explicitly rejected response_format; "
                    "retrying once without JSON mode"
                )
                active_format = None
                continue
            raise


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
