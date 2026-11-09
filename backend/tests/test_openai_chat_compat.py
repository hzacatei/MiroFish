from types import SimpleNamespace

import pytest

from app.utils.openai_chat_compat import (
    create_chat_completion,
    create_chat_completion_with_json_fallback,
    extract_chat_completion_text,
    is_gpt5_family,
    is_response_format_unsupported,
)


class ProviderError(RuntimeError):
    def __init__(self, *, status_code, body):
        super().__init__("provider error")
        self.status_code = status_code
        self.body = body


class CompletionRecorder:
    def __init__(self, result=None, error=None):
        self.result = result or object()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def client_for(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_gpt5_uses_completion_token_limit_without_temperature():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    result = create_chat_completion(
        client_for(recorder),
        model="gpt-5-2025-08-07",
        messages=messages,
        temperature=0.2,
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert result is recorder.result
    assert recorder.calls == [
        {
            "model": "gpt-5-2025-08-07",
            "messages": messages,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


def test_legacy_model_preserves_original_request_shape():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion(
        client_for(recorder),
        model="third-party-chat-model",
        messages=messages,
        temperature=0.7,
        max_tokens=456,
        response_format={"type": "json_object"},
    )

    assert recorder.calls == [
        {
            "model": "third-party-chat-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 456,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_error_is_propagated_without_guessing_or_retrying():
    provider_error = RuntimeError("unsupported max_tokens due to a server outage")
    recorder = CompletionRecorder(error=provider_error)

    with pytest.raises(RuntimeError) as captured:
        create_chat_completion(
            client_for(recorder),
            model="legacy-model",
            messages=[],
            max_tokens=10,
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        (" GPT-5.1-mini ", True),
        ("gpt-4.1", False),
        ("my-gpt-5-proxy", False),
        (None, False),
    ],
)
def test_gpt5_family_detection(model, expected):
    assert is_gpt5_family(model) is expected


def test_json_fallback_retries_once_without_response_format():
    anthropic_error = ProviderError(
        status_code=400,
        body={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "response_format.type must be json_schema",
            },
        },
    )
    recorder = CompletionRecorder()
    recorder.error = anthropic_error
    original_create = recorder.create

    def create(**kwargs):
        if len(recorder.calls) == 0:
            return original_create(**kwargs)
        recorder.error = None
        return original_create(**kwargs)

    recorder.create = create
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion_with_json_fallback(
        client_for(recorder),
        model="claude-sonnet",
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    assert recorder.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in recorder.calls[1]
    assert len(recorder.calls) == 2


def test_json_fallback_does_not_loop_when_the_retry_also_fails():
    anthropic_error = ProviderError(
        status_code=400,
        body={
            "error": {
                "message": "response_format.type must be json_schema",
            }
        },
    )
    recorder = CompletionRecorder(error=anthropic_error)

    with pytest.raises(ProviderError):
        create_chat_completion_with_json_fallback(
            client_for(recorder),
            model="claude-sonnet",
            messages=[],
            response_format={"type": "json_object"},
        )

    assert len(recorder.calls) == 2
    assert "response_format" not in recorder.calls[1]


def test_schema_validation_error_is_not_a_json_object_rejection():
    error = ProviderError(
        status_code=400,
        body={
            "error": {
                "message": "response_format.json_schema.schema is invalid",
            }
        },
    )
    assert is_response_format_unsupported(error) is False


def test_extracts_text_from_supported_content_shapes():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "first"}},
                        {"content": " second"},
                        SimpleNamespace(text=" third"),
                    ]
                )
            )
        ]
    )

    assert extract_chat_completion_text(response) == "first second third"
    assert extract_chat_completion_text(SimpleNamespace(choices=[])) == ""
