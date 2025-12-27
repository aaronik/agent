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


class _DummyDisplay:
    def clear(self):
        return None


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _patch_main(monkeypatch: pytest.MonkeyPatch, *, user_inputs: list[str], agent_reply: str):
    import main as main_mod

    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    inputs = user_inputs.copy()

    def _fake_prompt(_session, *, completer=None):
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kwargs: (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model"),
    )

    import src.agent_runner as agent_runner

    def _fake_run_agent_with_display(_agent, state):
        assert any(isinstance(m, HumanMessage) for m in state.messages)
        return [AIMessage(content=agent_reply)]

    monkeypatch.setattr(agent_runner, "run_agent_with_display", _fake_run_agent_with_display)

    return main_mod


def test_resume_replay_does_not_print_tool_call_json(tmp_home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
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

    # Reload main to isolate patches.
    import main as main_mod

    importlib.reload(main_mod)
    main_mod = _patch_main(monkeypatch, user_inputs=["next"], agent_reply="r")

    with pytest.raises(SystemExit):
        main_mod.main(["--resume", session_id])

    out = capsys.readouterr().out

    assert "final answer" in out
    assert "run_shell_command" not in out
    assert "call_1" not in out
    assert "tool_calls" not in out
