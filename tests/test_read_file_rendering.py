import os
import tempfile

from rich.console import Console

from src.tool_status_display import ToolStatusDisplay, ToolStatus


def test_read_file_panel_renders_syntax_highlighted(monkeypatch):
    monkeypatch.setenv("AGENT_NO_LIVE", "1")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "example.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n")

        display = ToolStatusDisplay()
        display.console = Console(record=True, force_terminal=True, width=80)

        tc_id = "1"
        display.register_calls([
            {"id": tc_id, "name": "read_file", "args": {"path": path}}
        ])
        display.update_status(tc_id, ToolStatus.DONE, f"[FILE]: {path}\n" + "def add(a, b):\n    return a + b\n")

        out = display.console.export_text()

        assert "def add" in out
        assert "return a + b" in out
