from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class _DummyTokenUsage:
    def __init__(self, model: str = "dummy") -> None:
        self.model = model

    def ingest_from_messages(self, _messages):
        return None

    def print_panel(self):
        return None


class _CaptureDisplay:
    def __init__(self):
        self.registered: list[list[dict]] = []
        self.updated: list[tuple[str, str]] = []

    def clear(self):
        return None

    def register_calls(self, tool_calls: list) -> None:
        self.registered.append(tool_calls)

    def update_status(self, tool_call_id: str, status, result=None) -> None:
        # Record the enum name for easy assertion.
        self.updated.append((tool_call_id, getattr(status, "name", str(status))))

    def get_tool_call(self, tool_call_id: str):
        # Not needed for this test.
        return None


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_resume_replays_tool_history_into_display(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a session file containing a prior tool call/result.
    from src.session_store import save_messages

    session_id = "s1"
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="ignored",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "run_shell_command",
                    "args": {"cmd": "echo hi", "timeout": 30},
                }
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call_1", name="run_shell_command"),
        AIMessage(content="final answer"),
    ]
    save_messages(session_id, messages)

    import main as main_mod

    importlib.reload(main_mod)

    disp = _CaptureDisplay()
    monkeypatch.setattr(main_mod, "_build_display", lambda: disp)

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    # Immediately exit after replay.
    def _fake_prompt(_session, *, completer=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kwargs: (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model"),
    )

    with pytest.raises(SystemExit):
        main_mod.main(["--resume", session_id])

    # We should have registered the call and marked it done.
    assert disp.registered
    flat_registered = [tc for batch in disp.registered for tc in batch]
    assert any(tc.get("id") == "call_1" and tc.get("name") == "run_shell_command" for tc in flat_registered)

    assert ("call_1", "DONE") in disp.updated
