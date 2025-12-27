import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def _agent_state_dir() -> Path:
    """Return the directory for persisted agent state.

    Per requirements, this is always $HOME/.agent/.
    """

    home = os.path.expanduser("~")
    return Path(home) / ".agent"


def _sessions_dir() -> Path:
    return _agent_state_dir() / "sessions"


def _latest_session_path() -> Path:
    return _agent_state_dir() / "latest_session"


def _ensure_dirs() -> None:
    _sessions_dir().mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    """Return a simple, sortable, filesystem-safe session id."""

    return time.strftime("%Y%m%d-%H%M%S")


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def list_session_ids() -> list[str]:
    """List saved session ids (most recent first)."""

    _ensure_dirs()
    sessions_dir = _sessions_dir()
    if not sessions_dir.exists():
        return []

    ids: list[str] = []
    for p in sessions_dir.glob("*.json"):
        ids.append(p.stem)

    # Session ids are sortable timestamps; keep newest first.
    ids.sort(reverse=True)
    return ids


def _first_human_message_preview(session_path: Path, *, max_len: int = 80) -> str | None:
    """Best-effort extraction of the first human message content from a session file."""

    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return None

    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("type") != "human":
            continue
        data = m.get("data")
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if isinstance(content, str):
            preview = " ".join(content.split())
            if len(preview) > max_len:
                preview = preview[: max_len - 1] + "…"
            return preview

    return None


def list_session_labels(*, max_preview_len: int = 80) -> list[str]:
    """List sessions as labels suitable for UI completion.

    Format:
        <session_id>\t<first human message preview>

    The tab keeps the id easy to parse while still showing context.
    """

    _ensure_dirs()
    sessions_dir = _sessions_dir()
    if not sessions_dir.exists():
        return []

    labels: list[str] = []
    for p in sessions_dir.glob("*.json"):
        sid = p.stem
        preview = _first_human_message_preview(p, max_len=max_preview_len)
        if preview:
            labels.append(f"{sid}\t{preview}")
        else:
            labels.append(sid)

    # Sort by id desc (timestamp-based).
    labels.sort(key=lambda s: s.split("\t", 1)[0], reverse=True)
    return labels


def save_messages(session_id: str, messages: Iterable[BaseMessage]) -> None:
    """Persist messages to disk and update the global latest pointer."""

    _ensure_dirs()

    payload = {
        "session_id": session_id,
        "messages": messages_to_dict(list(messages)),
    }

    session_path = _session_path(session_id)
    tmp_path = session_path.with_suffix(session_path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(session_path)

    # Update latest pointer (best-effort atomic-ish).
    latest_path = _latest_session_path()
    latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    latest_tmp.write_text(session_id + "\n", encoding="utf-8")
    latest_tmp.replace(latest_path)


def load_messages(session_id: Optional[str] = None) -> tuple[str, list[BaseMessage]]:
    """Load a session by id, or the latest session if id is None."""

    _ensure_dirs()

    if session_id is None:
        latest = _latest_session_path()
        if not latest.exists():
            raise FileNotFoundError("No latest session found")
        session_id = latest.read_text(encoding="utf-8").strip()
        if not session_id:
            raise FileNotFoundError("Latest session pointer was empty")

    session_path = _session_path(session_id)
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    payload = json.loads(session_path.read_text(encoding="utf-8"))
    messages = messages_from_dict(payload["messages"])
    return session_id, messages


@dataclass
class SessionAutosaver:
    """Background autosaver that coalesces frequent save requests.

    Autosave requests are handled on a background thread to avoid UI jitter.
    The autosaver keeps only the latest requested message list.
    """

    session_id: str

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._latest_messages: list[BaseMessage] | None = None
        self._thread = threading.Thread(
            target=self._run, name="session-autosaver", daemon=True
        )
        self._thread.start()

    def request_save(self, messages: Iterable[BaseMessage]) -> None:
        # Copy into a list so the caller can continue mutating their state.
        with self._lock:
            self._latest_messages = list(messages)
        self._event.set()

    def close(self, timeout: float = 2.0) -> None:
        # Best effort flush.
        self._stop = True
        self._event.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()

            with self._lock:
                messages = self._latest_messages
                self._latest_messages = None

            if messages is not None:
                try:
                    save_messages(self.session_id, messages)
                except Exception:
                    # Autosave should never crash the CLI.
                    pass

            if self._stop:
                with self._lock:
                    if self._latest_messages is None:
                        return
