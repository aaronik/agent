from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass

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

    # If the result itself is already a diff, detect common headers.
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


class ToolStatusDisplay:
    def __init__(self):
        self.console = Console()
        # Sequence of tool batches and communications.
        # Keeping it as a sequence preserves ordering across streaming.
        self.display_sequence: list[dict] = []
        self.live: Optional[Live] = None
        self._cost_line: Optional[str] = None

    def start_live(self):
        """Start the live display"""
        if not self.live:
            # NOTE: Rich Live uses an alternate screen by default. When the
            # renderable exceeds the available terminal height, Rich shows an
            # overflow indicator (often "…" / red dots) and the output stops
            # scrolling until Live is stopped.
            #
            # We want normal terminal scrollback behavior so users can keep
            # scrolling while many tool panels are emitted.
            self.live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=10,
                transient=False,
                # Keep normal scrollback (no alternate screen), and don't show
                # the overflow ellipsis indicator when the renderable exceeds
                # terminal height.
                screen=False,
                vertical_overflow="visible",
                auto_refresh=True,
            )
            self.live.start()

    def register_calls(self, tool_calls: list):
        """Register tool calls from an AIMessage"""
        # Find or create current tool batch
        if not self.display_sequence or self.display_sequence[-1]["type"] != "tools":
            self.display_sequence.append({"type": "tools", "tool_calls": {}})

        current = self.display_sequence[-1]
        for call in tool_calls:
            current["tool_calls"][call["id"]] = ToolCall(
                id=call["id"],
                name=call["name"],
                args=call["args"],
                status=ToolStatus.PENDING,
                result=None,
            )

        self.start_live()
        if self.live:
            self.live.update(self._create_display())

    def update_status(self, tool_call_id: str, status: ToolStatus, result: Optional[str] = None):
        """Update the status of a tool call"""
        for item in self.display_sequence:
            if item.get("type") == "tools" and tool_call_id in item.get("tool_calls", {}):
                tc: ToolCall = item["tool_calls"][tool_call_id]
                tc.status = status
                if result is not None and result != "":
                    tc.result = result

                if self.live:
                    self.live.update(self._create_display())
                break

    def add_communication(self, message: str):
        """Add a communication message to the display"""
        self.display_sequence.append({"type": "communication", "message": message})

        if self.live:
            self.live.update(self._create_display())

    def update_cost(self, token_usage):
        """Update the running cost line shown in the display."""
        try:
            input_cost = token_usage.prompt_cost()
            output_cost = token_usage.completion_cost()
            total = token_usage.total_cost()
            self._cost_line = f"Cost — input: ${input_cost:.4f} | output: ${output_cost:.4f} | total: ${total:.4f}"
        except Exception:
            self._cost_line = None

        if self.live:
            self.live.update(self._create_display())

    def clear(self):
        """Clear the display and reset state"""
        if self.live:
            self.live.stop()
            self.live = None

        self.display_sequence = []
        self._cost_line = None

    def _tool_panel(self, tc: ToolCall):
        """Render a single tool call.

        Most tools are panels. The `communicate` tool is rendered as unboxed,
        muted text so it reads like an aside but still maintains correct
        ordering in the tool stream.
        """
        # Special-case: communicate is basically a UI note. Render it inline.
        if tc.name == "communicate":
            msg = (tc.result or "").strip("\n")
            # No prefix: the styling is the differentiator.
            # Italic support varies by terminal, but Rich will degrade gracefully.
            return Text(escape(msg), style="italic dim")

        icon, color = tc.status.value
        header = Text()
        header.append(tc.name, style="cyan")
        header.append("  ")
        header.append(f"[{icon} {tc.status.name.title()}]", style=color)

        body_parts: list[Any] = []

        args_line = _format_args_one_line(tc.args)
        if args_line:
            body_parts.append(Text(args_line, style="dim"))

        result_text = tc.result or ""
        diff_text = _extract_unified_diff(result_text)
        if diff_text:
            result_text = _remove_diff_from_result(result_text)

        if result_text.strip():
            body_parts.append(Text(escape(result_text), style="white"))

        if diff_text:
            # Diff should feel like part of the tool output, not a separate event.
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
        """Create the complete display."""
        from rich.console import Group

        tool_renderables: list[Any] = []
        comm_renderables: list[Any] = []

        for item in self.display_sequence:
            if item.get("type") == "tools":
                for tc in item.get("tool_calls", {}).values():
                    tool_renderables.append(self._tool_panel(tc))
            else:  # communication
                if "renderable" in item:
                    comm_renderables.append(item["renderable"])
                else:
                    comm_renderables.append(
                        Panel(
                            item["message"],
                            title="Agent Communication",
                            border_style="grey37",
                            padding=(1, 2),
                        )
                    )

        renderables: list[Any] = []
        renderables.extend(tool_renderables)

        if self._cost_line:
            renderables.append(Panel(self._cost_line, title="Running Cost", border_style="grey37", padding=(0, 1)))

        renderables.extend(comm_renderables)

        return Group(*renderables)

    def finalize(self):
        """Finalize the display (no more updates)"""
        if self.live:
            self.live.stop()
            self.live = None

        if self.display_sequence:
            self.console.print()

    def get_tool_call(self, tool_call_id: str):
        """Return the ToolCall object for the given id or None"""
        for item in self.display_sequence:
            if item.get("type") == "tools":
                if tool_call_id in item.get("tool_calls", {}):
                    return item["tool_calls"][tool_call_id]
        return None

    def add_diff(self, filename: str, diff_text: str):
        """Add a rich diff panel to the display.

        Deprecated directionally: prefer embedding diffs in the tool panel.
        Kept for backwards compatibility with tool implementations.
        """
        syntax = Syntax(diff_text or "", "diff", theme="native", line_numbers=False)
        panel = Panel(syntax, title=f"{filename} — Diff", border_style="grey37", padding=(1, 2))
        self.display_sequence.append({"type": "communication", "renderable": panel})


# Global instance
_display = ToolStatusDisplay()


def get_tool_status_display() -> ToolStatusDisplay:
    """Get the global tool status display instance"""
    return _display
