from __future__ import annotations

import threading


class AgentCancelled(RuntimeError):
    """Raised when a user cancels the current agent turn (SIGINT)."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise AgentCancelled(self.reason or "cancelled")
