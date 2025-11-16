import signal
import argparse
from dataclasses import dataclass, field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
)
from langchain.agents import create_agent
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
import src.tools as tools
from src.constants import system_string
from src.util import TokenUsage, sys_git_ls, sys_ls, sys_pwd, sys_uname
from src.agent_runner import run_agent_with_display
from src.tool_status_display import get_tool_status_display
from src import claude_memory

HUMAN = "\n--- 🤷‍♂️🤷🤷‍♀️ User 🤷‍♂️🤷🤷‍♀️ ---\n\n"
ROBOT = "\n--- 🤖🤖🤖 AI 🤖🤖🤖 ---\n\n"

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
    tools.communicate,
    tools.spawn,
    # tools.screenshot_and_upload,
    # tools.summarize_response,
]

agent = create_agent(model, tools=tools)


@dataclass
class AgentState:
    messages: List[BaseMessage] = field(default_factory=list)
    token_usage: 'TokenUsage | None' = None


token_usage = TokenUsage(
    model=model.model_name
)

# Ensure the display updates with running cost as tokens accumulate
display = get_tool_status_display()


# Handle Ctrl-C: clean up display, print total tokens and exit
def signal_handler(*_):
    display = get_tool_status_display()
    display.clear()
    token_usage.print()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Create prompt session with vim mode
session = PromptSession(editing_mode=EditingMode.VI)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Personal Command Line Agent')
    parser.add_argument('--single', '-s', action='store_true',
                       help='Run single invocation without prompting for more input')
    parser.add_argument('query', nargs='*', help='Query to process')

    args = parser.parse_args()

    # Print a tall robot and the model name
    print("\n𜱜\n𜱟 " + MODEL)

    # Discover CLAUDE.md memory files and print line counts for each
    memory_files = claude_memory.find_all_claude_md_files()
    if memory_files:
        print("\nMemory files found:")
        for mf in memory_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                print(f"- {mf}: {len(lines)} lines")
            except Exception:
                print(f"- {mf}: (could not read)")
    else:
        print("\nNo CLAUDE.md memory files found")

    # Initial user input from command line or prompt
    if args.query:
        user_input = " ".join(args.query)
    else:
        user_input = session.prompt("What's up? ")

    # Initialize state with typed messages
    state = AgentState(
        messages=[
            SystemMessage(content=system_string),
            SystemMessage(content=f"[SYSTEM INFO] uname -a: {sys_uname()}"),
            SystemMessage(content=f"[SYSTEM INFO] pwd: {sys_pwd()}"),
            SystemMessage(content=f"[SYSTEM INFO] ls -l: {sys_ls()}"),
            SystemMessage(content=f"[SYSTEM INFO] git ls-files: {sys_git_ls()}"),
            HumanMessage(content=user_input),
        ],
        token_usage=token_usage
    )

    while True:
        # Run agent with live status display
        new_messages = run_agent_with_display(agent, state)

        # Update state with new messages
        state = AgentState(messages=state.messages + new_messages, token_usage=token_usage)

        output = state.messages[-1] if state.messages else None

        print(
            ROBOT,
            output.content if output else None
        )

        token_usage.ingest_from_messages(state.messages)

        # Exit after single invocation if --single flag is used
        if args.single:
            token_usage.print()
            break

        # Get next user input
        print(HUMAN)
        user_input = session.prompt("Anything else? ")

        # Append new user message to state
        state.messages.append(HumanMessage(content=user_input))
