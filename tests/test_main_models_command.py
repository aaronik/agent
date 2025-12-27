from __future__ import annotations

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


def test_models_command_lists_models(monkeypatch: pytest.MonkeyPatch, capsys):
    import main as main_mod

    # Avoid prompt_toolkit + tool UI.
    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    inputs = ["/models", "hi"]

    def _fake_prompt(_session, *, completer=None):
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    # Make `/models` deterministic.
    monkeypatch.setattr(main_mod, "_print_available_models", lambda: print("openai:gpt-4.1"))

    # Avoid creating a real agent/model.
    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kwargs: (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model"),
    )

    import src.agent_runner as agent_runner

    def _fake_run_agent_with_display(_agent, state):
        assert any(isinstance(m, HumanMessage) for m in state.messages)
        return [AIMessage(content="ok")]

    monkeypatch.setattr(agent_runner, "run_agent_with_display", _fake_run_agent_with_display)

    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "openai:gpt-4.1" in out
    assert "ok" in out


def test_models_command_switches_model(monkeypatch: pytest.MonkeyPatch, capsys):
    import main as main_mod

    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())

    class _DummySession:
        pass

    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())

    inputs = ["/models openai:gpt-4.1", "hi"]

    def _fake_prompt(_session, *, completer=None):
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt)
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    calls: list[str | None] = []

    def _fake_build_agent_and_deps(*, model_override=None, list_models=False):
        calls.append(model_override)
        return (object(), main_mod.SimpleTokenCounter(), _DummyTokenUsage(), "dummy-model")

    monkeypatch.setattr(main_mod, "_build_agent_and_deps", _fake_build_agent_and_deps)

    import src.agent_runner as agent_runner

    monkeypatch.setattr(agent_runner, "run_agent_with_display", lambda _a, _s: [AIMessage(content="ok")])

    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "𜱟 dummy-model" in out
    # First call: initial boot (None), second call: switch.
    assert calls[0] is None
    assert calls[1] == "openai:gpt-4.1"
