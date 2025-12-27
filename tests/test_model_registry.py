from __future__ import annotations

from src.model_registry import list_openai_chat_models
from src.provider_registry import parse_model_id


def test_list_openai_chat_models_filters_non_chat_models():
    ids = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "o4-mini",
        "text-embedding-3-small",
        "omni-moderation-latest",
        "whisper-1",
        "gpt-image-1",
        "some-unknown-model",
    ]

    out = list_openai_chat_models(ids)
    assert [m.id for m in out] == ["gpt-4.1", "gpt-4.1-mini", "o4-mini"]


def test_parse_model_id_defaults_to_openai_when_no_prefix():
    pm = parse_model_id("gpt-4.1")
    assert pm.provider == "openai"
    assert pm.model == "gpt-4.1"


def test_parse_model_id_with_provider_prefix():
    pm = parse_model_id("ollama:llama3")
    assert pm.provider == "ollama"
    assert pm.model == "llama3"
