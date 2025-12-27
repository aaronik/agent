from __future__ import annotations

import os
from typing import Iterable


def get_available_models() -> list[str]:
    """Return available model ids.

    Output is provider-prefixed, e.g.:
    - openai:gpt-4.1
    - ollama:llama3.1:latest

    Best-effort:
    - OpenAI requires OPENAI_API_KEY.
    - Ollama defaults to http://localhost:11434 (override with OLLAMA_URL).
    """

    models: list[str] = []

    # OpenAI
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.models.list()
        ids = [m.id for m in resp.data]

        from src.model_registry import list_openai_chat_models

        for mi in list_openai_chat_models(ids):
            models.append(f"openai:{mi.id}")
    except Exception:
        pass

    # Ollama
    try:
        import requests

        ollama_url = os.getenv("OLLAMA_URL") or "http://localhost:11434"
        resp = requests.get(ollama_url.rstrip("/") + "/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        for m in (data.get("models") or []):
            name = m.get("name")
            if name:
                models.append(f"ollama:{name}")
    except Exception:
        pass

    # Stable ordering for display + completion.
    return sorted(set(models))


def filter_completions(candidates: Iterable[str], *, prefix: str) -> list[str]:
    """Return candidates that start with prefix (case-sensitive)."""

    if not prefix:
        return list(candidates)
    return [c for c in candidates if c.startswith(prefix)]
