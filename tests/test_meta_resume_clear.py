from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage


class _DummyDisplay:
    def clear(self):
        return None


class _DummySession:
    def __init__(self) -> None:
        self.history = None


class _DummyTokenUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._seen_message_ids = set()

        # If True, ingest_from_messages will set non-zero usage.
        self.simulate_nonzero_on_ingest = True

    def ingest_from_messages(self, messages):
        # Simulate resumed usage being non-zero when configured.
        if self.simulate_nonzero_on_ingest:
            self.prompt_tokens = 123
            self.completion_tokens = 456

    def total_cost(self):
        # Cost should reflect current token counters.
        if self.prompt_tokens == 0 and self.completion_tokens == 0:
            return 0.0
        return 1.2345


def _patch_main(monkeypatch: pytest.MonkeyPatch, *, prompt_inputs: list[str]):
    import main as main_mod

    # Minimal deps.
    monkeypatch.setattr(main_mod, "_build_display", lambda: _DummyDisplay())
    monkeypatch.setattr(main_mod, "_build_prompt_session", lambda: _DummySession())
    monkeypatch.setattr(main_mod, "preload_litellm_cost_map", lambda: None)

    # Provide stable token counter.
    token_counter = main_mod.SimpleTokenCounter()
    tu = _DummyTokenUsage()

    monkeypatch.setattr(main_mod, "_build_agent_and_deps", lambda **_k: (object(), token_counter, tu, "dummy-model"))

    # Capture prompt meta line each time we prompt.
    meta_lines: list[str] = []
    real_prompt_boxed = main_mod._prompt_boxed

    inputs = prompt_inputs.copy()

    def _fake_prompt_boxed(session, *, completer=None, state=None, token_counter=None, **kwargs):
        if state is not None and token_counter is not None:
            meta_lines.append(
                main_mod._format_cost_and_context_line(
                    state=state,
                    token_counter=token_counter,
                    max_context_tokens=main_mod.MAX_CONTEXT_TOKENS,
                )
            )
        if not inputs:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    monkeypatch.setattr(main_mod, "_prompt_boxed", _fake_prompt_boxed)

    # No-op agent runner.
    import src.agent_runner as agent_runner

    monkeypatch.setattr(agent_runner, "run_agent_with_display", lambda _a, _s, **_k: [AIMessage(content="ok")])

    # Avoid memory file scanning noise.
    monkeypatch.setattr(main_mod, "load_all_agents_memory", lambda: "")

    return main_mod, tu, meta_lines


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_resume_rehydrates_cost_into_prompt_meta(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a session with one assistant message that includes token usage metadata
    # so resume has something to ingest.
    from src.session_store import save_messages

    session_id = "s1"
    messages = [HumanMessage(content="hi"), AIMessage(content="there")]
    save_messages(session_id, messages)

    main_mod, tu, meta_lines = _patch_main(monkeypatch, prompt_inputs=["/resume s1", "next"])

    with pytest.raises(SystemExit):
        main_mod.main([])

    # We should have prompted at least once after resume, and the cost should reflect ingest.
    assert any("Cost: $1.2345" in ml for ml in meta_lines)


def _parse_remaining_tokens(meta_line: str) -> int:
    # "Cost: $...   Context 87% (113,100/130,000 tokens)   Model: gpt-5.2"
    import re

    m = re.search(r"\(([-0-9,]+)/([-0-9,]+) tokens\)", meta_line)
    assert m is not None, f"Could not parse remaining tokens from: {meta_line!r}"
    return int(m.group(1).replace(",", ""))


def test_clear_resets_cost_and_context_and_updates_rendered_meta(tmp_home: Path, monkeypatch: pytest.MonkeyPatch):
    main_mod, tu, meta_lines = _patch_main(monkeypatch, prompt_inputs=["hello", "/clear", "world"])

    # Let ingest set a non-zero cost after the first assistant turn so we can
    # verify /clear resets the *rendered* cost line too.
    tu.simulate_nonzero_on_ingest = True

    with pytest.raises(SystemExit):
        main_mod.main([])

    # We should have captured multiple prompt meta lines.
    assert len(meta_lines) >= 2

    # Find the first prompt meta *after a turn has completed* (i.e. after we
    # have ingested non-zero usage). The very first prompt meta (before the first
    # turn) is expected to be $0.
    first_meta = next((ml for ml in meta_lines if "Cost: $1.2345" in ml), meta_lines[0])

    # Look for the first meta line after /clear: it should be the one whose cost
    # is reset to 0.
    clear_metas = [ml for ml in meta_lines if "Cost: $0.0000" in ml]
    assert clear_metas, f"Expected a post-/clear meta line with $0 cost. Got: {meta_lines!r}"
    after_clear_meta = clear_metas[0]

    assert "Cost: $0.0000" not in first_meta  # cost was non-zero after first turn
    assert "Cost: $0.0000" in after_clear_meta

    # Context remaining should increase after clear (conversation history wiped).
    assert _parse_remaining_tokens(after_clear_meta) > _parse_remaining_tokens(first_meta)

    # Also ensure internal counters were reset at clear time.
    assert tu._seen_message_ids == set()
