from io import StringIO

from rich.console import Console
from rich.live import Live

from src.tool_status_display import ToolStatusDisplay, ToolStatus


class _FakeLive:
    """Minimal Live stand-in to record updates without touching the terminal."""

    def __init__(self):
        self.updates: list[object] = []
        self.started = False
        self.stopped = False
        self.refreshed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update(self, renderable):
        self.updates.append(renderable)

    def refresh(self):
        self.refreshed = True


def test_live_mode_removes_finished_tool_from_active_calls(monkeypatch):
    monkeypatch.delenv("AGENT_NO_LIVE", raising=False)

    display = ToolStatusDisplay()
    display.register_calls(
        [
            {"id": "t1", "name": "run_shell_command", "args": {"cmd": "echo hi"}},
        ]
    )

    assert display.get_tool_call("t1") is not None

    display.update_status("t1", ToolStatus.RUNNING)
    assert display.get_tool_call("t1") is not None

    display.update_status("t1", ToolStatus.DONE, "hi")
    assert display.get_tool_call("t1") is None


def test_no_live_mode_keeps_tool_call_state(monkeypatch):
    monkeypatch.setenv("AGENT_NO_LIVE", "1")

    display = ToolStatusDisplay()
    display.register_calls(
        [
            {"id": "t1", "name": "run_shell_command", "args": {"cmd": "echo hi"}},
        ]
    )

    assert display.get_tool_call("t1") is not None
    display.update_status("t1", ToolStatus.DONE, "hi")
    assert display.get_tool_call("t1") is not None


def test_live_mode_clears_live_region_and_prints_final_panel_once(monkeypatch):
    """Codify scrollback behavior.

    On DONE/ERROR we should:
    - clear the live region (update(Text(""))) so the running panel isn't left behind
    - print exactly one final panel to the console (scrollback)
    """

    monkeypatch.delenv("AGENT_NO_LIVE", raising=False)

    out = StringIO()
    display = ToolStatusDisplay()
    display.console = Console(file=out, force_terminal=True, color_system=None, width=120)

    # Avoid real terminal Live. We just want to see that we clear and print.
    fake_live = _FakeLive()
    display.live = fake_live  # prevent start_live from creating a real Live

    # Register a tool and move it to running.
    display.register_calls(
        [
            {"id": "t1", "name": "run_shell_command", "args": {"cmd": "echo hi"}},
        ]
    )
    display.update_status("t1", ToolStatus.RUNNING)

    # Clear output so we only measure DONE behavior.
    out.truncate(0)
    out.seek(0)

    display.update_status("t1", ToolStatus.DONE, "hi")

    text = out.getvalue()

    # Final panel printed once.
    assert text.count("run_shell_command") == 1

    # Live region cleared at least once.
    assert any(getattr(r, "plain", None) == "" for r in fake_live.updates)


def test_start_live_creates_rich_live_when_enabled(monkeypatch):
    """Sanity: ensure we still use Rich Live in normal operation."""

    monkeypatch.delenv("AGENT_NO_LIVE", raising=False)

    out = StringIO()
    display = ToolStatusDisplay()
    display.console = Console(file=out, force_terminal=True, color_system=None, width=120)

    display.start_live()
    assert display.live is not None
    assert isinstance(display.live, Live)

    display.finalize()
