from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Provider = Literal["openai", "ollama"]


@dataclass(frozen=True)
class ModelContextInfo:
    max_context_tokens: int | None
    source: str


def _openai_context_from_model_name(model: str) -> ModelContextInfo:
    # OpenAI's Models API does not currently expose context window size.
    # Hardcode known context windows for common models.

    model_lower = model.lower()

    # GPT-5.2 models (400k context)
    if "gpt-5.2" in model_lower or "gpt5.2" in model_lower:
        return ModelContextInfo(max_context_tokens=400_000, source="hardcoded:gpt-5.2")

    # GPT-4o models (128k context)
    if "gpt-4o" in model_lower:
        return ModelContextInfo(max_context_tokens=128_000, source="hardcoded:gpt-4o")

    # GPT-4 Turbo models (128k context)
    if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower or "gpt-4-0125" in model_lower:
        return ModelContextInfo(max_context_tokens=128_000, source="hardcoded:gpt-4-turbo")

    # GPT-4 base models (8k context)
    if "gpt-4-" in model_lower or model_lower == "gpt-4":
        return ModelContextInfo(max_context_tokens=8_192, source="hardcoded:gpt-4")

    # GPT-3.5 Turbo (16k context for newer versions, 4k for older)
    if "gpt-3.5-turbo-16k" in model_lower:
        return ModelContextInfo(max_context_tokens=16_384, source="hardcoded:gpt-3.5-turbo-16k")
    if "gpt-3.5-turbo" in model_lower:
        return ModelContextInfo(max_context_tokens=16_384, source="hardcoded:gpt-3.5-turbo")

    # O1 models (128k context)
    if model_lower.startswith("o1-") or model_lower.startswith("o1"):
        return ModelContextInfo(max_context_tokens=128_000, source="hardcoded:o1")

    # Unknown model - return None to fall back to default
    return ModelContextInfo(max_context_tokens=None, source="openai-models-api:no-context")


def _ollama_context_from_api(*, ollama_url: str, model: str) -> ModelContextInfo:
    import requests

    # Ollama exposes model_info including e.g. llama.context_length.
    resp = requests.post(
        ollama_url.rstrip("/") + "/api/show",
        json={"name": model},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()

    mi = data.get("model_info") or {}
    if not isinstance(mi, dict):
        return ModelContextInfo(max_context_tokens=None, source="ollama:/api/show:bad-model_info")

    # Common key for llama-family models.
    ctx = mi.get("llama.context_length")
    if isinstance(ctx, int) and ctx > 0:
        return ModelContextInfo(max_context_tokens=ctx, source="ollama:/api/show:model_info:llama.context_length")

    # Best-effort scan for any context-ish key.
    for k, v in mi.items():
        if not isinstance(v, int) or v <= 0:
            continue
        lk = str(k).lower()
        if "context" in lk or lk.endswith(".ctx") or "ctx" in lk:
            return ModelContextInfo(max_context_tokens=v, source=f"ollama:/api/show:model_info:{k}")

    return ModelContextInfo(max_context_tokens=None, source="ollama:/api/show:no-context")


def get_model_context_info(*, provider: str, model: str) -> ModelContextInfo:
    """Return best-effort model context info.

    Behavior:
    - Prefer an in-memory cache (populated opportunistically at runtime).
    - If not cached, query explicitly where possible (Ollama).
    - If not available (OpenAI), return None.
    """

    from src.model_context_cache import get_model_context_cache

    p = (provider or "").strip().lower()
    m = (model or "").strip()

    cache = get_model_context_cache()
    cached = cache.get(provider=p, model=m)
    if cached is not None:
        return ModelContextInfo(max_context_tokens=cached, source="cache")

    if p == "ollama":
        import os

        ollama_url = os.getenv("OLLAMA_URL") or "http://localhost:11434"
        try:
            info = _ollama_context_from_api(ollama_url=ollama_url, model=m)
            if info.max_context_tokens is not None:
                cache.set(provider=p, model=m, max_context_tokens=info.max_context_tokens)
            return info
        except Exception:
            return ModelContextInfo(max_context_tokens=None, source="ollama:/api/show:error")

    # Default: treat as OpenAI.
    return _openai_context_from_model_name(m)
