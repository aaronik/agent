from langchain_core.messages import AIMessage, ToolMessage

from src.resume_tool_history import replay_tool_history
from src.tool_status_display import ToolStatusDisplay


def test_replay_tool_history_finalizes_live(monkeypatch):
    monkeypatch.delenv("AGENT_NO_LIVE", raising=False)

    display = ToolStatusDisplay()

    messages = [
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "run_shell_command", "args": {"cmd": "echo hi"}}]),
        ToolMessage(content="hi", tool_call_id="t1"),
    ]

    replay_tool_history(messages, display)

    # Resume replay should not leave a Rich Live region running (it causes
    # terminal repaints/flicker while user is typing).
    assert display.live is None
