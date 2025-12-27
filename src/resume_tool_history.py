from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent_runner import extract_result_preview
from src.tool_status_display import ToolStatus, ToolStatusDisplay


def _coerce_args(tool_call: dict) -> dict:
    args = tool_call.get("args")
    if isinstance(args, dict):
        return args

    # Some serializations use a JSON string in `arguments`.
    arguments = tool_call.get("arguments")
    if isinstance(arguments, str):
        try:
            import json

            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"arguments": arguments}

    return {}


def replay_tool_history(messages: Iterable[BaseMessage], display: ToolStatusDisplay) -> None:
    """Reconstruct the tool status panels from a saved message history.

    Goal: when resuming, show previous tool calls/results in the same formatted
    way as during live execution, without printing raw JSON.

    We rebuild the tool timeline from:
    - AIMessage.tool_calls (call metadata)
    - ToolMessage(tool_call_id=...) (results)

    Assumptions:
    - If we see a result without a registered call, we register a placeholder.
    - All resumed calls are treated as DONE unless tool output indicates error.
    """

    id_to_call: dict[str, dict] = {}

    # 1) Collect calls in order.
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if not isinstance(tc_id, str) or not tc_id:
                    continue
                id_to_call.setdefault(tc_id, tc)

    # 2) Register them so we can update statuses.
    if id_to_call:
        display.register_calls(
            [
                {
                    "id": tc_id,
                    "name": str(tc.get("name") or "tool"),
                    "args": _coerce_args(tc),
                }
                for tc_id, tc in id_to_call.items()
            ]
        )

    # 3) Apply results in order of appearance.
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue

        tc_id = getattr(m, "tool_call_id", None)
        if not isinstance(tc_id, str) or not tc_id:
            continue

        # Register placeholder if we never saw the call.
        if display.get_tool_call(tc_id) is None:
            display.register_calls(
                [
                    {
                        "id": tc_id,
                        "name": str(getattr(m, "name", None) or "tool"),
                        "args": {},
                    }
                ]
            )

        # Preview logic (match live UI).
        if getattr(m, "name", None) == "search_replace":
            preview = m.content if isinstance(m.content, str) else ""
        else:
            preview = extract_result_preview(m.content)

        if isinstance(m.content, str) and "(exit code:" in m.content:
            try:
                code_str = m.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                code = int(code_str)
            except Exception:
                code = None

            if code is not None and code != 0:
                display.update_status(tc_id, ToolStatus.ERROR, m.content)
            else:
                display.update_status(tc_id, ToolStatus.DONE, preview)
        else:
            display.update_status(tc_id, ToolStatus.DONE, preview)
