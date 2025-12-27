from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage


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


def test_followup_after_tool_call_content_blocks_does_not_crash(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    import main as main_mod

    # Reload main early so subsequent monkeypatching affects the module instance
    # used by the test.
    importlib.reload(main_mod)

    # Keep UI inert.
    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    # Two user turns, then exit.
    inputs = ["first", "second"]

    def _fake_prompt(_session, *, completer=None):
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    # Minimal agent wiring.
    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kwargs: (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model"),
    )

    import src.agent_runner as agent_runner

    # First agent turn returns an AIMessage whose *content* includes a Responses-style
    # function_call block. This used to crash on the *next* prompt because trimming
    # calls langchain_openai's token counter.
    calls = {"n": 0}

    def _fake_run_agent_with_display(_agent, _state):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                AIMessage(
                    content=[
                        {
                            "type": "function_call",
                            "name": "run_shell_command",
                            "arguments": '{"cmd":"echo hi","timeout":30}',
                            "call_id": "call_1",
                            "status": "completed",
                            "id": "fc_1",
                        }
                    ]
                )
            ]
        return [AIMessage(content="ok")]

    monkeypatch.setattr(agent_runner, "run_agent_with_display", _fake_run_agent_with_display)

    with pytest.raises(SystemExit):
        main_mod.main([])

    # Sanity: both turns executed.
    assert calls["n"] == 2
