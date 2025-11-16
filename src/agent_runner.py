from typing import Set, List
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from src.tool_status_display import get_tool_status_display, ToolStatus


def extract_result_preview(content: str, max_lines: int = 3, max_line_length: int = 80) -> str:
    """Extract a preview from tool result content with multiline support"""
    if not isinstance(content, str):
        return ""

    # Get first N non-empty lines
    lines = content.split("\n")
    preview_lines = []

    for line in lines[:max_lines * 3]:  # Look through more lines to find non-empty ones
        stripped = line.strip()
        if stripped:
            # Remove control characters
            cleaned = "".join(char for char in stripped if char.isprintable())
            # Truncate each line individually
            if len(cleaned) > max_line_length:
                cleaned = cleaned[:max_line_length - 3] + "..."
            preview_lines.append(cleaned)

        if len(preview_lines) >= max_lines:
            break

    # Join with actual newlines for multiline display
    return "\n".join(preview_lines)


def process_agent_chunk(messages: List[BaseMessage], tool_call_ids_seen: Set[str], display, communicate_calls: Set[str]):
    """Process messages from the agent node"""
    for msg in messages:
        # Handle tool calls in AIMessage
        if isinstance(msg, AIMessage) and msg.tool_calls:
            new_calls = [tc for tc in msg.tool_calls if tc["id"] not in tool_call_ids_seen]
            if new_calls:
                # Separate communicate calls from normal tool calls
                normal_calls = []
                for tc in new_calls:
                    tool_call_ids_seen.add(tc["id"])
                    if tc["name"] == "communicate":
                        # Track communicate calls separately
                        communicate_calls.add(tc["id"])
                    else:
                        normal_calls.append(tc)

                # Only register non-communicate calls with the display
                if normal_calls:
                    display.register_calls(normal_calls)
                    # Mark as running since they're about to execute
                    for tc in normal_calls:
                        display.update_status(tc["id"], ToolStatus.RUNNING)

        # Handle tool results in ToolMessage
        elif isinstance(msg, ToolMessage):
            # Skip communicate tool results here (handled in process_tools_chunk)
            if msg.tool_call_id not in communicate_calls:
                preview = extract_result_preview(msg.content)
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


def process_tools_chunk(messages: List[BaseMessage], display, communicate_calls: Set[str]):
    """Process messages from the tools node"""
    # Mark pending tools as running (check all tables in sequence)
    for item in display.display_sequence:
        if item["type"] == "table":
            for tool_id, tool_call in item["tool_calls"].items():
                if tool_call.status == ToolStatus.PENDING:
                    display.update_status(tool_id, ToolStatus.RUNNING)

    # Process tool results
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Handle communicate tool specially
            if msg.tool_call_id in communicate_calls:
                # Add communication to display (it will update live)
                display.add_communication(msg.content)
            else:
                # Normal tool result handling
                preview = extract_result_preview(msg.content)
                display.update_status(msg.tool_call_id, ToolStatus.DONE, preview)


def run_agent_with_display(agent, state, recursion_limit: int = 200):
    """Run the agent with live status display"""
    display = get_tool_status_display()
    display.clear()

    final_messages = []
    tool_call_ids_seen = set()
    communicate_calls = set()  # Track communicate tool calls separately

    # Stream agent execution
    # Keep the running cost updated as messages arrive
    token_usage = None
    try:
        # attempt to get token usage cache from state if present
        token_usage = getattr(state, 'token_usage', None)
    except Exception:
        token_usage = None

    for chunk in agent.stream(state, {"recursion_limit": recursion_limit}):
        if "agent" in chunk:
            messages = chunk["agent"].get("messages", [])
            final_messages.extend(messages)
            process_agent_chunk(messages, tool_call_ids_seen, display, communicate_calls)
            # Update running cost if token_usage is available on the state
            if token_usage is not None:
                try:
                    # First try incremental ingestion if available
                    if hasattr(token_usage, 'ingest_messages_incremental'):
                        token_usage.ingest_messages_incremental(messages)
                    else:
                        token_usage.ingest_from_messages(messages)

                    display.update_cost(token_usage)
                except Exception:
                    # Don't let cost reporting break execution
                    pass

        elif "tools" in chunk:
            messages = chunk["tools"].get("messages", [])
            final_messages.extend(messages)
            process_tools_chunk(messages, display, communicate_calls)
            if token_usage is not None:
                try:
                    if hasattr(token_usage, 'ingest_messages_incremental'):
                        token_usage.ingest_messages_incremental(messages)
                    else:
                        token_usage.ingest_from_messages(messages)
                    display.update_cost(token_usage)
                except Exception:
                    pass

    display.finalize()

    return final_messages
