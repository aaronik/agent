from __future__ import annotations

from typing import Iterable, List

from langchain_core.messages import AIMessage, BaseMessage


def normalize_for_token_count(messages: Iterable[BaseMessage]) -> List[BaseMessage]:
    """Normalize messages so langchain_openai token counting won't crash.

    LangChain's OpenAI token counter (used by `trim_messages`) expects
    `AIMessage.content` to be either a string or a list of *known* content
    blocks (e.g. {type: "text", text: ...}).

    When routing via OpenAI Responses API, LangChain can emit non-standard
    blocks like {type: "function_call", ...}. Those are valid Responses
    artifacts but are not recognized by the token counter, causing:
        ValueError: Unrecognized content block type

    We don't want these blocks to participate in context trimming anyway.
    The agent state already carries tool calls via `AIMessage.tool_calls` and
    tool results via `ToolMessage`.

    Strategy:
    - For any AIMessage whose content is a list, keep only text blocks and
      join them into a single string.
    - Preserve metadata (usage, response/additional kwargs) so cost tracking
      still works.
    """

    out: list[BaseMessage] = []

    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            text_parts: list[str] = []
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str) and t:
                        text_parts.append(t)

            out.append(
                AIMessage(
                    content="\n".join(text_parts),
                    tool_calls=getattr(m, "tool_calls", None),
                    response_metadata=m.response_metadata,
                    additional_kwargs=m.additional_kwargs,
                    usage_metadata=getattr(m, "usage_metadata", None),
                )
            )
            continue

        out.append(m)

    return out
