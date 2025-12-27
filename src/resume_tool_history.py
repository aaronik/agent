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

    Important:
    ToolStatusDisplay removes calls from its active set after DONE/ERROR.
    If we pre-register *all* calls first, then apply results, only the last
    printed panel may appear (earlier ones get popped before printing).

    So during resume we must replay strictly in appearance order:
    - register calls as we encounter them
    - update them as we encounter results

    Also important:
    Rich Live auto-refresh will keep repainting the terminal region. After a
    resume replay we should ensure Live is finalized so the user's input prompt
    doesn't flicker.
    """

    # Cache call metadata so we can fill placeholders if a ToolMessage appears
    # without its corresponding AIMessage tool_calls envelope.
    id_to_call: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id:
                    id_to_call.setdefault(tc_id, tc)

    for m in messages:
        # Register tool calls as they appear.
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls = [tc for tc in m.tool_calls if isinstance(tc, dict)]
            if tool_calls:
                display.register_calls(
                    [
                        {
                            "id": tc_id,
                            "name": str(tc.get("name") or "tool"),
                            "args": _coerce_args(tc),
                        }
                        for tc in tool_calls
                        if isinstance((tc_id := tc.get("id")), str) and tc_id
                    ]
                )
            continue

        # Apply results as they appear.
        if not isinstance(m, ToolMessage):
            continue

        tc_id = getattr(m, "tool_call_id", None)
        if not isinstance(tc_id, str) or not tc_id:
            continue

        if display.get_tool_call(tc_id) is None:
            tc = id_to_call.get(tc_id) or {}
            display.register_calls(
                [
                    {
                        "id": tc_id,
                        "name": str(tc.get("name") or getattr(m, "name", None) or "tool"),
                        "args": _coerce_args(tc),
                    }
                ]
            )

        # Preview logic (match live UI).
        if getattr(m, "name", None) == "search_replace":
            preview = m.content if isinstance(m.content, str) else ""
        else:
            preview = extract_result_preview(m.content)

        # Detect non-zero exit codes for run_shell_command.
        if isinstance(m.content, str) and "(exit code:" in m.content:
            try:
                code_str = m.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                code = int(code_str)
            except Exception:
                code = None

            if code is not None and code != 0:
                display.update_status(tc_id, ToolStatus.ERROR, m.content)
                continue

        display.update_status(tc_id, ToolStatus.DONE, preview)

    # Don't leave a Live region auto-refreshing after resume replay.
    display.finalize()
