"""Later simulation stages must use the same JSON-mode fallback as ontology."""

from types import SimpleNamespace

from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.simulation_config_generator import SimulationConfigGenerator


class ProviderError(RuntimeError):
    def __init__(self, *, status_code, body):
        super().__init__(body["error"]["message"])
        self.status_code = status_code
        self.body = body


def _anthropic_rejection():
    return ProviderError(
        status_code=400,
        body={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "response_format.type must be json_schema",
            },
            "request_id": "req_mock_anthropic",
        },
    )


def _completion(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ]
    )


class _Completions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _anthropic_rejection()
        return _completion(self.payload)


def _client(payload):
    completions = _Completions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_simulation_config_survives_anthropic_json_object_rejection():
    client, completions = _client('{"hours": 1}')
    generator = object.__new__(SimulationConfigGenerator)
    generator.client = client
    generator.model_name = "claude-sonnet"

    assert generator._call_llm_with_retry("prompt", "system") == {"hours": 1}
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]
    assert completions.calls[0]["model"] == "claude-sonnet"


def test_profile_generation_survives_anthropic_json_object_rejection():
    client, completions = _client('{"bio": "synthetic", "persona": "observer"}')
    generator = object.__new__(OasisProfileGenerator)
    generator.client = client
    generator.model_name = "claude-sonnet"

    result = generator._generate_profile_with_llm(
        entity_name="Ada",
        entity_type="Person",
        entity_summary="A synthetic person.",
        entity_attributes={},
        context="A one-paragraph fixture.",
    )

    assert result["bio"] == "synthetic"
    assert result["persona"] == "observer"
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]
