"""Tests for the tool status display."""
import pytest
from src.tool_status_display import ToolStatusDisplay, ToolStatus
from rich.errors import MarkupError


def test_display_handles_result_with_rich_markup_chars():
    """Test that tool results containing Rich markup characters are properly escaped.

    This test ensures that results containing [ or ] characters don't cause
    MarkupError when rendering the display.
    """
    display = ToolStatusDisplay()

    # Register a tool call
    tool_calls = [{
        "id": "test_call_1",
        "name": "test_tool",
        "args": {"param": "value"}
    }]
    display.register_calls(tool_calls)

    # Update with a result that contains Rich markup-like characters
    # This simulates real-world scenarios where tool results might contain
    # code snippets, logs, or other content with brackets
    problematic_result = "Error: closing tag '[/{color}]' at position 5614"
    display.update_status("test_call_1", ToolStatus.DONE, result=problematic_result)

    # This should not raise MarkupError
    try:
        rendered = display._create_display()
        # Force rendering to trigger any markup errors
        from io import StringIO
        from rich.console import Console
        temp_console = Console(file=StringIO(), width=120, legacy_windows=False)
        temp_console.print(rendered)
    except MarkupError as e:
        pytest.fail(f"MarkupError should not be raised when rendering results with brackets: {e}")
    finally:
        display.clear()


def test_display_handles_various_bracket_patterns():
    """Test various patterns of brackets that might appear in tool results."""
    display = ToolStatusDisplay()

    test_cases = [
        "[bold]Some text[/bold]",  # Valid Rich markup that should be escaped
        "Error: [/bold] without opening",  # Unmatched closing tag
        "Text with [brackets]",  # Simple brackets
        "Multiple [levels [of [brackets]]]",  # Nested brackets
        "[",  # Single opening bracket
        "]",  # Single closing bracket
        "Normal text [/{variable}] with f-string-like syntax",  # The actual error pattern
    ]

    for idx, result_text in enumerate(test_cases):
        tool_calls = [{
            "id": f"test_call_{idx}",
            "name": "test_tool",
            "args": {"param": "value"}
        }]
        display.register_calls(tool_calls)
        display.update_status(f"test_call_{idx}", ToolStatus.DONE, result=result_text)

    # Should not raise MarkupError
    try:
        rendered = display._create_display()
        from io import StringIO
        from rich.console import Console
        temp_console = Console(file=StringIO(), width=120, legacy_windows=False)
        temp_console.print(rendered)
    except MarkupError as e:
        pytest.fail(f"MarkupError should not be raised: {e}")
    finally:
        display.clear()
