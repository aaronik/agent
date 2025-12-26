from typing import Set, List

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from src.tool_status_display import get_tool_status_display, ToolStatus


def extract_result_preview(content: str, max_lines: int = 3, max_line_length: int = 80) -> str:
    """Extract a preview from tool result content with multiline support."""
    if not isinstance(content, str):
        return ""

    # Hide the internal exit-code marker from the visible preview.
    # The display title will still surface it.
    content = content.replace("(exit code:", "(exit code:")
    if "(exit code:" in content:
        content = content.split("(exit code:", 1)[0].rstrip()

    # Get first N non-empty lines
    lines = content.split("\n")
    preview_lines = []

    for line in lines[: max_lines * 3]:  # Look through more lines to find non-empty ones
        stripped = line.strip()
        if stripped:
            # Remove control characters
            cleaned = "".join(char for char in stripped if char.isprintable())
            # Truncate each line individually
            if len(cleaned) > max_line_length:
                cleaned = cleaned[: max_line_length - 3] + "..."
            preview_lines.append(cleaned)

        if len(preview_lines) >= max_lines:
            break

    # Join with actual newlines for multiline display
    return "\n".join(preview_lines)


def process_agent_chunk(messages: List[BaseMessage], tool_call_ids_seen: Set[str], display):
    """Process messages from the agent node."""
    for msg in messages:
        # Handle tool calls in AIMessage
        if isinstance(msg, AIMessage) and msg.tool_calls:
            new_calls = [tc for tc in msg.tool_calls if tc["id"] not in tool_call_ids_seen]
            if new_calls:
                for tc in new_calls:
                    tool_call_ids_seen.add(tc["id"])

                display.register_calls(new_calls)
                # Mark as running since they're about to execute
                for tc in new_calls:
                    display.update_status(tc["id"], ToolStatus.RUNNING)

        # Handle tool results in ToolMessage
        elif isinstance(msg, ToolMessage):
            preview = extract_result_preview(msg.content)
            # If the tool indicates a non-zero exit code, show it as an error.
            # run_shell_command appends "(exit code: X)" on failure.
            if isinstance(msg.content, str) and "(exit code:" in msg.content:
                try:
                    code_str = msg.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                    code = int(code_str)
                except Exception:
                    code = None

                if code is not None and code != 0:
                    # Preserve the raw content (with exit-code marker) so the
                    # panel title can show `Done (X)`. The preview function
                    # hides the marker from the panel body.
                    display.update_status(msg.tool_call_id, ToolStatus.ERROR, msg.content)
                else:
                    display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)
            else:
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


def process_tools_chunk(messages: List[BaseMessage], display):
    """Process messages from the tools node."""
    # Mark pending tools as running
    for item in display.display_sequence:
        if item["type"] == "tools":
            for tool_id, tool_call in item["tool_calls"].items():
                if tool_call.status == ToolStatus.PENDING:
                    display.update_status(tool_id, ToolStatus.RUNNING)

    # Process tool results
    for msg in messages:
        if isinstance(msg, ToolMessage):
            preview = extract_result_preview(msg.content)
            # If the tool indicates a non-zero exit code, show it as an error.
            # run_shell_command appends "(exit code: X)" on failure.
            if isinstance(msg.content, str) and "(exit code:" in msg.content:
                try:
                    code_str = msg.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                    code = int(code_str)
                except Exception:
                    code = None

                if code is not None and code != 0:
                    # Preserve the raw content (with exit-code marker) so the
                    # panel title can show `Done (X)`. The preview function
                    # hides the marker from the panel body.
                    display.update_status(msg.tool_call_id, ToolStatus.ERROR, msg.content)
                else:
                    display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)
            else:
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


def run_agent_with_display(agent, state, recursion_limit: int = 200):
    """Run the agent with live status display"""
    display = get_tool_status_display()
    display.clear()

    final_messages = []
    tool_call_ids_seen = set()

    # Convert state to the format expected by create_agent
    # Handle both dict and AgentState object
    if isinstance(state, dict):
        agent_input = state
    else:
        agent_input = {"messages": state.messages}

    for chunk in agent.stream(agent_input, {"recursion_limit": recursion_limit}):
        if "model" in chunk:
            messages = chunk["model"].get("messages", [])
            final_messages.extend(messages)
            process_agent_chunk(messages, tool_call_ids_seen, display)

        elif "tools" in chunk:
            messages = chunk["tools"].get("messages", [])
            final_messages.extend(messages)
            process_tools_chunk(messages, display)

    display.finalize()

    return final_messages
