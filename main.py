import signal
import argparse
from dataclasses import dataclass, field
from typing import List
from langchain_openai import ChatOpenAI
from langchain.schema import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
)
from langgraph.prebuilt import create_react_agent
import src.tools as tools
from src.constants import system_string
from src.util import TokenUsage, sys_git_ls, sys_ls, sys_pwd, sys_uname
from static.pricing import pricing
from src.tool_status_display import get_tool_status_display, ToolStatus
from langchain_core.messages import AIMessage, ToolMessage

HUMAN = "\n--- 🤷‍♂️🤷🤷‍♀️ User 🤷‍♂️🤷🤷‍♀️ ---\n"
ROBOT = "\n--- 🤖🤖🤖 AI 🤖🤖🤖 ---\n"
TOOLS = "\n--- 🛠️🪚✒️ Tools used 🛠️🪚✒️ ---\n"

MODEL = "gpt-5-mini"
model = ChatOpenAI(model=MODEL)

tools = [
    # tools.search_text,
    # tools.search_text_alternative,
    # tools.search_images,
    tools.fetch,
    tools.read_file,
    tools.write_file,
    tools.search_replace,
    tools.gen_image,
    tools.run_shell_command,
    tools.spawn,
    # tools.screenshot_and_upload,
    # tools.summarize_response,
]

agent = create_react_agent(model, tools=tools)


@dataclass
class AgentState:
    messages: List[BaseMessage] = field(default_factory=list)


token_usage = TokenUsage(
    model=f"openai:{model.model_name}",
    pricing=pricing
)


# Handle Ctrl-C: print total tokens and exit
def signal_handler(*_):
    token_usage.print()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Personal Command Line Agent')
    parser.add_argument('--single', '-s', action='store_true',
                       help='Run single invocation without prompting for more input')
    parser.add_argument('query', nargs='*', help='Query to process')

    args = parser.parse_args()

    print("\n𜱜\n𜱟 " + MODEL)

    # Initial user input from command line or prompt
    if args.query:
        user_input = " ".join(args.query)
    else:
        user_input = input("What's up? ")

    # Initialize state with typed messages
    state = AgentState(messages=[
        SystemMessage(content=system_string),
        SystemMessage(content=f"[SYSTEM INFO] uname -a: {sys_uname()}"),
        SystemMessage(content=f"[SYSTEM INFO] pwd: {sys_pwd()}"),
        SystemMessage(content=f"[SYSTEM INFO] ls -l: {sys_ls()}"),
        SystemMessage(content=f"[SYSTEM INFO] git ls-files: {sys_git_ls()}"),
        HumanMessage(content=user_input),
    ])

    print(TOOLS)

    while True:
        # Get the tool status display
        display = get_tool_status_display()
        display.clear()

        # Stream the agent execution
        final_messages = []
        tool_call_ids_seen = set()

        for chunk in agent.stream(state, {"recursion_limit": 200}):
            # Extract messages from the chunk
            if "agent" in chunk:
                messages = chunk["agent"].get("messages", [])
                final_messages.extend(messages)

                for msg in messages:
                    # Check for tool calls in AIMessage
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        # Register new tool calls
                        new_calls = [tc for tc in msg.tool_calls if tc["id"] not in tool_call_ids_seen]
                        if new_calls:
                            display.register_calls(new_calls)
                            # Immediately mark as running since they're about to execute
                            for tc in new_calls:
                                tool_call_ids_seen.add(tc["id"])
                                display.update_status(tc["id"], ToolStatus.RUNNING)

                    # Check for tool results in ToolMessage
                    elif isinstance(msg, ToolMessage):
                        # Extract result preview (first line or first 80 chars)
                        result_preview = msg.content
                        if isinstance(result_preview, str):
                            lines = result_preview.split("\n")
                            result_preview = lines[0] if lines else ""
                            if len(result_preview) > 80:
                                result_preview = result_preview[:77] + "..."

                        display.update_status(msg.tool_call_id, ToolStatus.DONE, result_preview)

            elif "tools" in chunk:
                # Tools node - mark all as running first, then extract results
                for tool_id in display.tool_calls.keys():
                    if display.tool_calls[tool_id].status == ToolStatus.PENDING:
                        display.update_status(tool_id, ToolStatus.RUNNING)

                # Extract ToolMessages
                messages = chunk["tools"].get("messages", [])
                final_messages.extend(messages)

                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        # Extract result preview
                        result_preview = msg.content
                        if isinstance(result_preview, str):
                            lines = result_preview.split("\n")
                            result_preview = lines[0] if lines else ""
                            if len(result_preview) > 80:
                                result_preview = result_preview[:77] + "..."

                        display.update_status(msg.tool_call_id, ToolStatus.DONE, result_preview)

        # Finalize the display
        display.finalize()

        # Update state with all messages
        state = AgentState(messages=state.messages + final_messages)

        output = state.messages[-1] if state.messages else None

        print(
            ROBOT,
            output.content if output else None
        )

        token_usage.ingest_from_messages(state.messages)

        # Exit after single invocation if --single flag is used
        if args.single:
            break

        # Get next user input
        print(HUMAN)
        user_input = input("Anything else? ")

        print(TOOLS)

        # Append new user message to state
        state.messages.append(HumanMessage(content=user_input))
