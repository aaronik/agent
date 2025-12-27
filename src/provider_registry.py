from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model: str

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_model_id(raw: str) -> ProviderModel:
    """Parse a model id.

    Supported:
    - "openai:gpt-4.1"  -> provider=openai, model=gpt-4.1
    - "ollama:llama3"   -> provider=ollama, model=llama3
    - "gpt-4.1"         -> provider=openai (default), model=gpt-4.1

    This keeps backwards compatibility with existing UX where model ids were
    passed without an explicit provider.
    """

    if ":" in raw:
        provider, model = raw.split(":", 1)
        return ProviderModel(provider=provider, model=model)

    return ProviderModel(provider="openai", model=raw)


def format_model_id(provider: str, model: str) -> str:
    return f"{provider}:{model}"
