from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_save_and_load_roundtrip(tmp_home: Path) -> None:
    from src.session_store import load_messages, save_messages

    session_id = "test-123"
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]

    save_messages(session_id, messages)
    loaded_id, loaded = load_messages(session_id)

    assert loaded_id == session_id
    assert [m.type for m in loaded] == ["system", "human", "ai"]
    assert [m.content for m in loaded] == ["sys", "hi", "hello"]


def test_load_latest_pointer(tmp_home: Path) -> None:
    from src.session_store import load_messages, save_messages

    save_messages("s1", [SystemMessage(content="a")])
    save_messages("s2", [SystemMessage(content="b")])

    loaded_id, loaded = load_messages(None)
    assert loaded_id == "s2"
    assert loaded[0].content == "b"
