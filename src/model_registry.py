from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str


def list_openai_chat_models(model_ids: list[str]) -> list[ModelInfo]:
    """Filter OpenAI model ids to those likely usable with ChatCompletions.

    We keep this conservative to avoid offering models that will error at runtime.

    Notes:
    - OpenAI's /v1/models returns many model types (embeddings, images, audio, etc.)
    - This CLI uses langchain_openai.ChatOpenAI (chat completions / responses wrapper)
    """

    deny_prefixes = (
        "whisper-",
        "tts-",
        "gpt-image-",
        "omni-moderation-",
        "text-embedding-",
        "text-search-",
        "dall-e-",
    )

    deny_substrings = (
        "embedding",
        "moderation",
        "realtime",
        "audio",
        "vision-preview",
    )

    out: list[ModelInfo] = []
    for mid in model_ids:
        if any(mid.startswith(p) for p in deny_prefixes):
            continue
        if any(s in mid for s in deny_substrings):
            continue
        # Keep known chat families.
        if mid.startswith("gpt-") or mid.startswith("o"):
            out.append(ModelInfo(id=mid))

    # Stable ordering for display/testing.
    out.sort(key=lambda m: m.id)
    return out
