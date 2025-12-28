from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelContextCache:
    # Provider + model_name -> max context tokens
    _cache: dict[tuple[str, str], int]

    def __init__(self) -> None:
        self._cache = {}

    def get(self, *, provider: str, model: str) -> int | None:
        key = (provider.strip().lower(), model.strip())
        return self._cache.get(key)

    def set(self, *, provider: str, model: str, max_context_tokens: int) -> None:
        if max_context_tokens <= 0:
            return
        key = (provider.strip().lower(), model.strip())
        self._cache[key] = int(max_context_tokens)


_model_context_cache = ModelContextCache()


def get_model_context_cache() -> ModelContextCache:
    return _model_context_cache
