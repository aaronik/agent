from typing import Set, List

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from src.tool_status_display import get_tool_status_display, ToolStatus


def extract_result_preview(content: str, max_lines: int = 30, max_line_length: int = 120) -> str:
    """Extract a preview from tool result content with multiline support.

    This is used for *tool panel* previews in the CLI.

    Goals:
    - Preserve blank lines between content (so file output / logs look right).
    - Drop leading/trailing blank lines (keeps panels compact).
    - Cap by max_lines of non-empty lines.
    """
    if not isinstance(content, str):
        return ""

    # Hide the internal exit-code marker from the visible preview.
    if "(exit code:" in content:
        content = content.split("(exit code:", 1)[0].rstrip()

    raw_lines = content.split("\n")

    # Find first/last non-empty line to drop leading/trailing blank lines.
    first_non_empty = None
    last_non_empty = None
    for i, line in enumerate(raw_lines):
        if line != "":
            first_non_empty = i
            break
    for i in range(len(raw_lines) - 1, -1, -1):
        if raw_lines[i] != "":
            last_non_empty = i
            break

    if first_non_empty is None or last_non_empty is None:
        return ""

    lines = raw_lines[first_non_empty : last_non_empty + 1]

    preview_lines: list[str] = []
    non_empty_count = 0

    for line in lines:
        # If we've already collected max_lines non-empty lines, stop *before*
        # adding any further lines (including blank lines) so we don't end up
        # with a stray blank line right before the truncation marker.
        if non_empty_count >= max_lines:
            break

        if line != "":
            non_empty_count += 1

        cleaned = "".join(char for char in line if (char.isprintable() or char in {"\t"}))
        if len(cleaned) > max_line_length:
            cleaned = cleaned[: max_line_length - 3] + "..."
        preview_lines.append(cleaned)

    # Determine truncation: any remaining non-empty content after we stopped.
    truncated = any(l != "" for l in lines[len(preview_lines) :])

    preview = "\n".join(preview_lines)

    if truncated:
        preview = preview.rstrip("\n")
        if preview:
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
                # Responses API can emit intermediate AIMessage objects whose
                # content is a list of content-block dicts (reasoning/text/etc).
                # We only want the final text rendered in the main transcript.
                if messages:
                    last = messages[-1]
                    if isinstance(last, AIMessage) and isinstance(last.content, list):
                        text_parts: list[str] = []
                        for block in last.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                t = block.get("text")
                                if isinstance(t, str) and t:
                                    text_parts.append(t)
                        if text_parts:
                            # Preserve usage_metadata so cost tracking works.
                            messages = [
                                *messages[:-1],
                                AIMessage(
                                    content="\n".join(text_parts),
                                    response_metadata=last.response_metadata,
                                    additional_kwargs=last.additional_kwargs,
                                    usage_metadata=getattr(last, "usage_metadata", None),
                                ),
                            ]
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
