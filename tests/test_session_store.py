from __future__ import annotations

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


def test_list_session_ids_sorted_desc(tmp_home: Path) -> None:
    from src.session_store import list_session_ids, save_messages

    save_messages("20240101-000000", [SystemMessage(content="a")])
    save_messages("20240102-000000", [SystemMessage(content="b")])

    assert list_session_ids()[:2] == ["20240102-000000", "20240101-000000"]


def test_list_session_labels_includes_preview(tmp_home: Path) -> None:
    from src.session_store import list_session_labels, save_messages

    save_messages(
        "20240102-000000",
        [
            SystemMessage(content="sys"),
            HumanMessage(content="hello\nworld"),
            AIMessage(content="ok"),
        ],
    )

    labels = list_session_labels(max_preview_len=80)
    assert any(l.startswith("20240102-000000\t") for l in labels)
    assert any("hello world" in l for l in labels)
