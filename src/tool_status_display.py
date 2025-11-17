from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box
from rich.syntax import Syntax
from rich.markup import escape


class ToolStatus(Enum):
    # Use a more subdued, professional palette (avoid bold/bright colors)
    PENDING = ("⏳", "grey50")
    RUNNING = ("▶️", "cyan")
    DONE = ("✓", "green")
    ERROR = ("✗", "red")


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]
    status: ToolStatus
    result: Optional[str] = None


class ToolStatusDisplay:
    def __init__(self):
        self.console = Console()
        self.display_sequence: list[dict] = []  # Sequence of tables and communications
        self.live: Optional[Live] = None
        self._cost_line: Optional[str] = None

    def start_live(self):
        """Start the live display"""
        if not self.live:
            self.live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=10,
                transient=False
            )
            self.live.start()

    def register_calls(self, tool_calls: list):
        """Register tool calls from an AIMessage"""
        # Find or create current table
        if not self.display_sequence or self.display_sequence[-1]["type"] != "table":
            # Need a new table
            new_table = {"type": "table", "tool_calls": {}}
            self.display_sequence.append(new_table)

        # Add to last table
        current_table = self.display_sequence[-1]
        for call in tool_calls:
            current_table["tool_calls"][call["id"]] = ToolCall(
                id=call["id"],
                name=call["name"],
                args=call["args"],
                status=ToolStatus.PENDING,
                result=None
            )

        # Start live display if not already started
        self.start_live()
        # Update display with new calls
        if self.live:
            self.live.update(self._create_display())

    def update_status(self, tool_call_id: str, status: ToolStatus, result: Optional[str] = None):
        """Update the status of a tool call"""
        # Find the tool call in any table
        for item in self.display_sequence:
            if item["type"] == "table" and tool_call_id in item["tool_calls"]:
                item["tool_calls"][tool_call_id].status = status
                if result:
                    item["tool_calls"][tool_call_id].result = result

                # Update live display if active
                if self.live:
                    self.live.update(self._create_display())
                break

    def add_communication(self, message: str):
        """Add a communication message to the display"""
        comm = {"type": "communication", "message": message}
        self.display_sequence.append(comm)

        # Update live display if active
        if self.live:
            self.live.update(self._create_display())

    def update_cost(self, token_usage):
        """Update the running cost line shown in the display.

        token_usage: object with prompt_cost(), completion_cost(), total_cost() methods
        """
        try:
            input_cost = token_usage.prompt_cost()
            output_cost = token_usage.completion_cost()
            total = token_usage.total_cost()
            self._cost_line = f"Cost — input: ${input_cost:.4f} | output: ${output_cost:.4f} | total: ${total:.4f}"
        except Exception:
            # Be resilient: if token_usage doesn't have expected methods, skip
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

    def _create_table(self, tool_calls: Dict[str, ToolCall]) -> Table:
        """Create a Rich table for the current tool calls"""
        # Get terminal width and calculate available space
        terminal_width = self.console.width
        # Reserve space for table borders, padding, and columns
        # Tool: ~15, Arguments: variable, Status: ~10, Result: remainder
        reserved_width = 40  # borders, padding, tool name, status
        available_for_content = max(terminal_width - reserved_width, 40)

        # Split remaining space between Arguments and Result
        args_width = min(50, available_for_content // 2)
        result_width = min(60, available_for_content - args_width)

        table = Table(
            title="🛠️ Tool Execution",
            box=box.ROUNDED,
            show_header=True,
            header_style="cyan",
            title_justify="left",
            border_style="grey37",
            title_style="bold",
            width=terminal_width - 2  # Ensure table fits in terminal
        )

        # Left-align columns for a cleaner, more professional look
        table.add_column("Tool", style="cyan", no_wrap=True, max_width=15, justify="left")
        table.add_column("Arguments", style="white", max_width=args_width, no_wrap=True, justify="left")
        table.add_column("Status", justify="left", style="", no_wrap=True)
        table.add_column("Result", style="dim", max_width=result_width, no_wrap=False, justify="left")

        for tc in tool_calls.values():
            # Format args - escape Rich markup to prevent MarkupError
            args_str = ", ".join(f"{k}={v}" for k, v in tc.args.items())
            args_str = escape(args_str)

            # Get status icon and color
            icon, color = tc.status.value
            status_display = f"[{color}]{icon} {tc.status.name.title()}[/{color}]"

            # Format result - escape Rich markup to prevent MarkupError
            result_display = escape(tc.result) if tc.result else ""

            table.add_row(
                tc.name,
                args_str,
                status_display,
                result_display
            )

        # Use more muted colors for panels and syntax themes

        return table

    def _create_display(self):
        """Create the complete display with tables and communications"""
        from rich.console import Group

        # We'll render communications first so they appear "on top" of the running cost
        # panel in the combined live view. This ensures messages like the user banner
        # are visible above the cost panel instead of being pushed outside the live
        # view area.
        comm_renderables = []
        other_renderables = []

        for item in self.display_sequence:
            if item["type"] == "table":
                # Only render table if it has tool calls
                if item["tool_calls"]:
                    other_renderables.append(self._create_table(item["tool_calls"]))
            else:  # communication
                if "renderable" in item:
                    comm_renderables.append(item["renderable"])
                else:
                    comm_renderables.append(
                        Panel(
                            item["message"],
                            title="💭 Agent Communication",
                            border_style="grey37",
                            padding=(1, 2)
                        )
                    )

        renderables = []

        # Add communications first so they appear above the cost line/panels
        renderables.extend(comm_renderables)

        # Add cost line after communications
        if self._cost_line:
            renderables.append(Panel(self._cost_line, title="💰 Running Cost", border_style="grey37", padding=(0, 1)))

        # Then add the rest (tables, etc.)
        renderables.extend(other_renderables)

        return Group(*renderables)

    def finalize(self):
        """Finalize the display (no more updates)"""
        # Stop live display and show final state
        if self.live:
            self.live.stop()
            self.live = None

        # Print a newline for spacing
        if self.display_sequence:
            self.console.print()

    def get_tool_call(self, tool_call_id: str):
        """Return the ToolCall object for the given id or None"""
        for item in self.display_sequence:
            if item.get("type") == "table":
                if tool_call_id in item["tool_calls"]:
                    return item["tool_calls"][tool_call_id]
        return None

    def add_diff(self, filename: str, diff_text: str):
        """Add a rich diff panel to the display.

        filename: short label for the diff (typically basename)
        diff_text: unified diff text to render with syntax highlighting
        """
        # Create a Syntax renderable for nice diff coloring
        syntax = Syntax(diff_text or "", "diff", theme="native", line_numbers=False)
        panel = Panel(syntax, title=f"🧾 {filename} — Diff", border_style="grey37", padding=(1, 2))

        comm = {"type": "communication", "renderable": panel}
        self.display_sequence.append(comm)


# Global instance
_display = ToolStatusDisplay()


def get_tool_status_display() -> ToolStatusDisplay:
    """Get the global tool status display instance"""
    return _display
