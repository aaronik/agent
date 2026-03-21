from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


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


def _patch_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_inputs: list[str],
    agent_reply: str,
    seen_human_transcripts: list[list[str]] | None = None,
):
    """Patch main module to avoid network, prompt_toolkit, and expensive imports."""

    import main as main_mod

    # Prevent tool status UI / prompt_toolkit.
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

    # Prevent background pricing preload.
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    # Avoid creating a real model/agent.
    monkeypatch.setattr(
        main_mod,
        "_build_agent_and_deps",
        lambda **_kwargs: (
            object(),
            main_mod.SimpleTokenCounter(),
            _DummyTokenUsage(),
            "dummy-model",
        ),
    )

    # Stub agent runner: just emit one AI message each turn.
    import src.agent_runner as agent_runner

    def _fake_run_agent_with_display(_agent, state):
        # Make sure the agent sees history.
        human_messages = [
            str(m.content) for m in state.messages if isinstance(m, HumanMessage)
        ]
        assert human_messages
        if seen_human_transcripts is not None:
            seen_human_transcripts.append(human_messages)
        return [AIMessage(content=agent_reply)]

    monkeypatch.setattr(
        agent_runner, "run_agent_with_display", _fake_run_agent_with_display
    )

    return main_mod


def test_main_creates_session_and_autosaves(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["hello", "bye"],
        agent_reply="world",
    )

    # Run one turn, then exit via KeyboardInterrupt from prompt.
    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "world" in out

    latest = (
        (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    )
    session_path = tmp_home / ".agent" / "sessions" / f"{latest}.json"
    assert session_path.exists()

    # Sanity check saved transcript contains our initial input.
    saved = session_path.read_text(encoding="utf-8")
    assert "hello" in saved


def test_main_resume_loads_history_and_does_not_crash(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    # First run creates a session.
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["first"],
        agent_reply="r1",
    )
    with pytest.raises(SystemExit):
        main_mod.main([])

    latest = (
        (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    )

    # Reload main module to clear any cached state and reapply patches.
    importlib.reload(main_mod)

    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["second"],
        agent_reply="r2",
    )

    with pytest.raises(SystemExit):
        main_mod.main(["--resume", latest])

    out = capsys.readouterr().out
    assert "r2" in out
    assert "r1" in out
    assert "first" in out
    assert "[SYSTEM INFO]" not in out

    # Ensure the resumed session file was updated and contains both turns.
    session_path = tmp_home / ".agent" / "sessions" / f"{latest}.json"
    saved = session_path.read_text(encoding="utf-8")
    assert "first" in saved
    assert "second" in saved


def test_main_auto_resumes_latest_session_by_default(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    first_run_humans: list[list[str]] = []
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["first"],
        agent_reply="r1",
        seen_human_transcripts=first_run_humans,
    )
    with pytest.raises(SystemExit):
        main_mod.main([])

    latest = (
        (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    )
    assert first_run_humans == [["first"]]

    importlib.reload(main_mod)

    second_run_humans: list[list[str]] = []
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["second"],
        agent_reply="r2",
        seen_human_transcripts=second_run_humans,
    )

    with pytest.raises(SystemExit):
        main_mod.main([])

    out = capsys.readouterr().out
    assert "r1" in out
    assert "r2" in out
    assert "first" in out
    assert second_run_humans == [["first", "second"]]

    session_path = tmp_home / ".agent" / "sessions" / f"{latest}.json"
    saved = session_path.read_text(encoding="utf-8")
    assert "first" in saved
    assert "second" in saved


def test_new_flag_starts_fresh_session_even_when_latest_exists(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["first"],
        agent_reply="r1",
    )
    with pytest.raises(SystemExit):
        main_mod.main([])

    original_latest = (
        (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    )

    importlib.reload(main_mod)

    fresh_run_humans: list[list[str]] = []
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=["second"],
        agent_reply="r2",
        seen_human_transcripts=fresh_run_humans,
    )

    with pytest.raises(SystemExit):
        main_mod.main(["--new"])

    new_latest = (
        (tmp_home / ".agent" / "latest_session").read_text(encoding="utf-8").strip()
    )
    assert new_latest != original_latest
    assert fresh_run_humans == [["second"]]


def test_prefill_history_populates_session_history() -> None:
    import main as main_mod

    session = main_mod._build_prompt_session()
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="first"),
        AIMessage(content="ok"),
        HumanMessage(content="second"),
    ]

    main_mod._prefill_history_from_messages(session, messages)
    assert list(session.history.get_strings()) == ["first", "second"]


def test_single_mode_does_not_save(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    main_mod = _patch_main_dependencies(
        monkeypatch,
        user_inputs=[],
        agent_reply="x",
    )

    # In single mode, we provide the query on argv to avoid prompt.
    rc = main_mod.main(["--single", "hi"])
    assert rc == 0

    assert not (tmp_home / ".agent").exists()
