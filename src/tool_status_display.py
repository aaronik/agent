from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box


class ToolStatus(Enum):
    PENDING = ("⏳", "dim yellow")
    RUNNING = ("▶️", "bold cyan")
    DONE = ("✓", "bold green")
    ERROR = ("✗", "bold red")


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

    def clear(self):
        """Clear the display and reset state"""
        if self.live:
            self.live.stop()
            self.live = None

        self.display_sequence = []

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
            header_style="bold cyan",
            border_style="blue",
            title_style="bold magenta",
            width=terminal_width - 2  # Ensure table fits in terminal
        )

        table.add_column("Tool", style="cyan", no_wrap=True, max_width=15)
        table.add_column("Arguments", style="white", max_width=args_width, no_wrap=True)
        table.add_column("Status", justify="center", style="bold", no_wrap=True)
        table.add_column("Result", style="dim white", max_width=result_width, no_wrap=False)

        for tc in tool_calls.values():
            # Format args
            args_str = ", ".join(f"{k}={v}" for k, v in tc.args.items())

            # Get status icon and color
            icon, color = tc.status.value
            status_display = f"[{color}]{icon} {tc.status.name.title()}[/{color}]"

            # Format result - single line only
            result_display = tc.result or ""

            table.add_row(
                tc.name,
                args_str,
                status_display,
                result_display
            )

        return table

    def _create_display(self):
        """Create the complete display with tables and communications"""
        from rich.console import Group

        renderables = []

        for item in self.display_sequence:
            if item["type"] == "table":
                # Only render table if it has tool calls
                if item["tool_calls"]:
                    renderables.append(self._create_table(item["tool_calls"]))
            else:  # communication
                renderables.append(
                    Panel(
                        item["message"],
                        title="💭 Agent Communication",
                        border_style="blue",
                        padding=(1, 2)
                    )
                )

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


# Global instance
_display = ToolStatusDisplay()


def get_tool_status_display() -> ToolStatusDisplay:
    """Get the global tool status display instance"""
    return _display
