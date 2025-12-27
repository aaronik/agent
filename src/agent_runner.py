from typing import Set, List

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from src.tool_status_display import get_tool_status_display, ToolStatus


def extract_result_preview(content: str, max_lines: int = 30, max_line_length: int = 120) -> str:
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
    preview_lines: list[str] = []

    scanned = 0
    for line in lines[: max_lines * 3]:  # Look through more lines to find non-empty ones
        scanned += 1
        line_has_content = line != ""
        if line_has_content:
            # Preserve indentation; only drop non-printing control chars.
            cleaned = "".join(char for char in line if (char.isprintable() or char in {"\t"}))
            # Truncate each line individually
            if len(cleaned) > max_line_length:
                cleaned = cleaned[: max_line_length - 3] + "..."
            preview_lines.append(cleaned)

        if len(preview_lines) >= max_lines:
            break

    # Join with actual newlines for multiline display
    preview = "\n".join(preview_lines)

    # Add a visual truncation indicator if there was more content beyond what
    # we included in the preview.
    remaining_has_content = any(l != "" for l in lines[scanned:])
    if remaining_has_content:
        if preview and not preview.endswith("\n"):
            preview += "\n"
        preview += "…"

    return preview


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
            if getattr(msg, "name", None) == "search_replace":
                preview = msg.content if isinstance(msg.content, str) else ""
            else:
                preview = extract_result_preview(msg.content)

            if isinstance(msg.content, str) and "(exit code:" in msg.content:
                try:
                    code_str = msg.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                    code = int(code_str)
                except Exception:
                    code = None

                if code is not None and code != 0:
                    display.update_status(msg.tool_call_id, ToolStatus.ERROR, msg.content)
                else:
                    display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)
            else:
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


def process_tools_chunk(messages: List[BaseMessage], display):
    """Process messages from the tools node."""
    # Mark pending tools as running.
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) == "search_replace":
                preview = msg.content if isinstance(msg.content, str) else ""
            else:
                preview = extract_result_preview(msg.content)

            if isinstance(msg.content, str) and "(exit code:" in msg.content:
                try:
                    code_str = msg.content.split("(exit code:", 1)[1].split(")", 1)[0].strip()
                    code = int(code_str)
                except Exception:
                    code = None

                if code is not None and code != 0:
                    display.update_status(msg.tool_call_id, ToolStatus.ERROR, msg.content)
                else:
                    display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)
            else:
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


from src.cancel import CancelToken, AgentCancelled


def run_agent_with_display(agent, state, recursion_limit: int = 200, cancel_token: CancelToken | None = None):
    """Run the agent with live status display"""
    display = get_tool_status_display()
    display.clear()

    final_messages = []
    tool_call_ids_seen = set()

    # Convert state to the format expected by create_agent
    if isinstance(state, dict):
        agent_input = state
    else:
        agent_input = {"messages": state.messages}

    try:
        for chunk in agent.stream(agent_input, {"recursion_limit": recursion_limit}):
            if cancel_token is not None:
                cancel_token.check()

            if "model" in chunk:
                messages = chunk["model"].get("messages", [])
                final_messages.extend(messages)
                process_agent_chunk(messages, tool_call_ids_seen, display)

            elif "tools" in chunk:
                messages = chunk["tools"].get("messages", [])
                final_messages.extend(messages)
                process_tools_chunk(messages, display)
    except AgentCancelled:
        display.finalize()
        return []

    display.finalize()

    return final_messages
