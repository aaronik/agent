from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage


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


def test_help_command_prints_available_commands(monkeypatch: pytest.MonkeyPatch, capsys):
    import main as main_mod

    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    inputs = ["/help", "hi"]

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

    monkeypatch.setattr(agent_runner, "run_agent_with_display", lambda *_a, **_k: [AIMessage(content="ok")])

    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "Available commands:" in out
    assert "/help" in out
    assert "/models" in out
    assert "/clear" in out
