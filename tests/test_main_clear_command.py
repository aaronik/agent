from __future__ import annotations

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
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_clear_command_resets_session(tmp_home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    import main as main_mod

    disp = _DummyDisplay()
    monkeypatch.setattr(main_mod, "_build_display", lambda: disp)

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    # First turn: user says hi; then /clear; then new user says again.
    inputs = ["hi", "/clear", "again"]

    def _fake_prompt(_session, *, completer=None):
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kw: (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model"),
    )

    import src.agent_runner as agent_runner

    def _fake_run_agent_with_display(_agent, state, cancel_token=None):
        # Ensure /clear isn't sent as HumanMessage
        assert not any(isinstance(m, HumanMessage) and m.content == "/clear" for m in state.messages)
        return [AIMessage(content="ok")]

    monkeypatch.setattr(agent_runner, "run_agent_with_display", _fake_run_agent_with_display)

    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "ok" in out

    # Display was cleared at least once.
    assert disp.clear_calls >= 1

    # Ensure we created a new session file after clear (latest_session changes).
    latest = (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    session_path = tmp_home / ".agent" / "sessions" / f"{latest}.json"
    assert session_path.exists()

    saved = session_path.read_text(encoding="utf-8")
    assert "again" in saved
