from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
import os

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markup import escape
from rich.text import Text


class ToolStatus(Enum):
    """Status for a tool call.

    Keep indicators ASCII/boring and color subtle; the panel border provides
    most of the visual structure.
    """

    PENDING = ("...", "grey50")
    RUNNING = (">", "cyan")
    DONE = ("OK", "green")
    ERROR = ("ERR", "red")


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]
    status: ToolStatus
    result: Optional[str] = None


def _format_args_one_line(args: Dict[str, Any], max_len: int = 120) -> str:
    """Format tool args as a single line, escaping Rich markup."""
    if not args:
        return ""
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    args_str = escape(args_str)
    if len(args_str) > max_len:
        return args_str[: max_len - 3] + "..."
    return args_str


def _extract_unified_diff(text: str) -> str | None:
    """Best-effort extraction of a unified diff from a tool result string."""
    if not text:
        return None

    marker = "\nDiff:\n"
    if marker in text:
        return text.split(marker, 1)[1].strip("\n") or None

    stripped = text.lstrip()
    if stripped.startswith("--- ") and "\n+++ " in stripped:
        return stripped.strip("\n")

    return None


def _remove_diff_from_result(text: str) -> str:
    """Remove the embedded diff from a tool result string (if present)."""
    if not text:
        return ""
    marker = "\nDiff:\n"
    if marker in text:
        return text.split(marker, 1)[0].rstrip("\n")
    return text


def _extract_file_marker(text: str) -> tuple[str | None, str]:
    """Extract a [FILE]: <path> marker from tool output.

    Returns:
        (path, remainder_text)
    """

    if not text:
        return None, ""

    marker = "[FILE]: "
    if text.startswith(marker):
        rest = text[len(marker):]
        if "\n" in rest:
            path, remainder = rest.split("\n", 1)
            return path.strip(), remainder
        return rest.strip(), ""

    return None, text


def _guess_lexer_from_path(path: str) -> str:
    """Best-effort lexer name based on file extension."""

    _, ext = os.path.splitext(path.lower())

    # Common cases we hit in this repo.
    if ext in {".py"}:
        return "python"
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext in {".js"}:
        return "javascript"
    if ext in {".ts"}:
        return "typescript"
    if ext in {".json"}:
        return "json"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext in {".toml"}:
        return "toml"
    if ext in {".sh", ".bash", ".zsh"}:
        return "bash"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".css"}:
        return "css"
    if ext in {".diff", ".patch"}:
        return "diff"

    return "text"


class ToolStatusDisplay:
    def __init__(self):
        self.console = Console()

        # Calls currently shown in the live updating area.
        self._active_tool_calls: dict[str, ToolCall] = {}

        self.live: Optional[Live] = None
        self._cost_line: Optional[str] = None

    def _live_enabled(self) -> bool:
        """Return True if Rich Live updates should be used.

        Disable via env var to make output append-only.

        Set AGENT_NO_LIVE=1 to disable.
        """

        return os.getenv("AGENT_NO_LIVE", "").strip().lower() not in {"1", "true", "yes", "on"}

    def start_live(self) -> None:
        if not self._live_enabled():
            return

        if self.live is None:
            # Keep Live running in normal screen mode.
            self.live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=10,
                transient=False,
                screen=False,
                vertical_overflow="visible",
                auto_refresh=True,
            )
            self.live.start()

    def _update_live(self) -> None:
        if self.live:
            self.live.update(self._create_display())

    def _clear_live_region(self) -> None:
        """Clear the current live region so it doesn't remain in scrollback."""
        if self.live:
            # Render an empty string into the live region, then refresh.
            self.live.update(Text(""))
            try:
                self.live.refresh()
            except Exception:
                pass

    def register_calls(self, tool_calls: list) -> None:
        for call in tool_calls:
            tc = ToolCall(
                id=call["id"],
                name=call["name"],
                args=call["args"],
                status=ToolStatus.PENDING,
                result=None,
            )
            self._active_tool_calls[tc.id] = tc

        if self._live_enabled():
            self.start_live()
            self._update_live()
        else:
            for tc in self._active_tool_calls.values():
                self.console.print(self._tool_panel(tc))

    def update_status(self, tool_call_id: str, status: ToolStatus, result: Optional[str] = None) -> None:
        tc = self._active_tool_calls.get(tool_call_id)
        if tc is None:
            return

        tc.status = status
        if result is not None and result != "":
            tc.result = result

        if not self._live_enabled():
            self.console.print(self._tool_panel(tc))
            return

        self.start_live()

        if status in {ToolStatus.DONE, ToolStatus.ERROR}:
            # 1) Show final state in the live region briefly.
            self._update_live()
            if self.live:
                try:
                    self.live.refresh()
                except Exception:
                    pass

            # 2) Remove from live set and clear the live region so the running
            #    panel doesn't get "left behind" in scrollback.
            self._active_tool_calls.pop(tool_call_id, None)
            self._update_live()
            self._clear_live_region()

            # 3) Print the final panel exactly once (append-only scrollback).
            self.console.print(self._tool_panel(tc))

            # 4) Continue live display if there are other active tools.
            self._update_live()
        else:
            self._update_live()

    def add_communication(self, message: str) -> None:
        panel = Panel(message, title="Agent Communication", border_style="grey37", padding=(1, 2))

        if self._live_enabled() and self.live:
            self._clear_live_region()
            self.console.print(panel)
            self._update_live()
        else:
            self.console.print(panel)

    def update_cost(self, token_usage) -> None:
        try:
            input_cost = token_usage.prompt_cost()
            output_cost = token_usage.completion_cost()
            total = token_usage.total_cost()
            self._cost_line = f"Cost — input: ${input_cost:.4f} | output: ${output_cost:.4f} | total: ${total:.4f}"
        except Exception:
            self._cost_line = None

        self._update_live()

    def clear(self) -> None:
        if self.live:
            self.live.stop()
            self.live = None

        self._active_tool_calls = {}
        self._cost_line = None

    def _tool_panel(self, tc: ToolCall):
        if tc.name == "communicate":
            msg = (tc.result or "").strip("\n")
            return Text(escape(msg), style="italic dim")

        icon, color = tc.status.value
        header = Text()
        header.append(tc.name, style="cyan")
        header.append("  ")

        exit_code: int | None = None
        raw_result = tc.result or ""
        if "(exit code:" in raw_result:
            try:
                code_str = raw_result.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                exit_code = int(code_str)
            except Exception:
                exit_code = None

        result_text = raw_result
        if "(exit code:" in result_text:
            result_text = result_text.split("(exit code:", 1)[0].rstrip()

        if tc.status == ToolStatus.DONE:
            header.append(f"[{icon} Done]", style=color)
        elif tc.status == ToolStatus.ERROR:
            if exit_code is not None:
                header.append(f"[{icon} Done ({exit_code})]", style=color)
            else:
                header.append(f"[{icon} Done]", style=color)
        else:
            header.append(f"[{icon} {tc.status.name.title()}]", style=color)

        body_parts: list[Any] = []

        args_line = _format_args_one_line(tc.args)
        if args_line:
            body_parts.append(Text(args_line, style="dim"))

        diff_text = _extract_unified_diff(result_text)
        if diff_text:
            result_text = _remove_diff_from_result(result_text)

        file_path_marker, result_text = _extract_file_marker(result_text)

        if tc.name == "read_file":
            file_path = file_path_marker or tc.args.get("path")
            if isinstance(file_path, str) and file_path:
                lexer_name = _guess_lexer_from_path(file_path)
                body_parts.append(
                    Syntax(
                        result_text,
                        lexer_name,
                        theme="native",
                        line_numbers=False,
                        word_wrap=False,
                        code_width=None,
                        tab_size=4,
                    )
                )
            elif result_text:
                body_parts.append(Text(escape(result_text), style="white"))
        elif result_text:
            body_parts.append(Text(escape(result_text), style="white"))

        if diff_text:
            body_parts.append(Syntax(diff_text, "diff", theme="native", line_numbers=False))

        if not body_parts:
            body_parts.append(Text("", style="dim"))

        from rich.console import Group

        return Panel(
            Group(*body_parts),
            title=header,
            border_style="grey37",
            padding=(1, 2),
        )

    def _create_display(self):
        from rich.console import Group

        renderables: list[Any] = []
        for tc in self._active_tool_calls.values():
            renderables.append(self._tool_panel(tc))

        if self._cost_line and renderables:
            renderables.append(Panel(self._cost_line, title="Running Cost", border_style="grey37", padding=(0, 1)))

        return Group(*renderables)

    def finalize(self) -> None:
        if self.live:
            self._clear_live_region()
            self.live.stop()
            self.live = None

    def get_tool_call(self, tool_call_id: str):
        return self._active_tool_calls.get(tool_call_id)

    def add_diff(self, filename: str, diff_text: str):
        syntax = Syntax(diff_text or "", "diff", theme="native", line_numbers=False)
        panel = Panel(syntax, title=f"{filename} — Diff", border_style="grey37", padding=(1, 2))

        if self._live_enabled() and self.live:
            self._clear_live_region()
            self.console.print(panel)
            self._update_live()
        else:
            self.console.print(panel)


_display = ToolStatusDisplay()


def get_tool_status_display() -> ToolStatusDisplay:
    return _display
