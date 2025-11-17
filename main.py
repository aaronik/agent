import signal
import argparse
from dataclasses import dataclass, field
from typing import List
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    trim_messages,
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
import sys

HUMAN = "\n--- 🤷‍♂️🤷🤷‍♀️ User 🤷‍♂️🤷🤷‍♀️ ---\n\n"
ROBOT = "\n--- 🤖🤖🤖 AI 🤖🤖🤖 ---\n\n"

# Default model
MODEL = "gpt-5-mini"
MAX_CONTEXT_TOKENS = 150000


class SimpleTokenCounter:
    """Simple token counter for Ollama models that approximates tokens as characters / 4"""

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Approximate token count using character length / 4"""
        return sum(len(str(msg.content)) // 4 for msg in messages)


# If the user set Ollama configuration, prefer that local model
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

if OLLAMA_URL and OLLAMA_MODEL:
    # Use the Ollama model name and point the OpenAI-compatible client at the Ollama API base URL.
    # ChatOpenAI accepts a `base_url` parameter which will be used as the OpenAI base; Ollama
    # implements an OpenAI-like subset of the API at /v1, so this works reasonably well for simple usage.
    MODEL = OLLAMA_MODEL
    # Instantiate ChatOpenAI but direct it to the Ollama server
    # We intentionally do not set an API key here (Ollama typically doesn't require one for localhost).
    ollama_base_url = OLLAMA_URL.rstrip('/') + '/v1'
    model = ChatOpenAI(model=MODEL, base_url=ollama_base_url, api_key="dummy")
    # Use simple token counter for Ollama models since they don't support OpenAI's token counting
    token_counter = SimpleTokenCounter()
else:
    # Default: use the managed OpenAI-compatible client
    model = ChatOpenAI(model=MODEL)
    # Use the model itself for accurate token counting
    token_counter = model

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


# Graceful exit helper used for Ctrl-C handling
def graceful_exit() -> None:
    """Clean up UI and print token usage before exiting."""
    try:
        display.clear()
    except Exception:
        # Fallback: try obtaining a fresh display instance
        try:
            get_tool_status_display().clear()
        except Exception:
            pass
    try:
        token_usage.print()
    except Exception:
        pass
    print("Goodbye!")
    sys.exit(0)


# Handle Ctrl-C: clean up display, print total tokens and exit
def signal_handler(*_):
    graceful_exit()


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
        try:
            user_input = session.prompt("What's up? ")
        except (KeyboardInterrupt, SystemExit):
            graceful_exit()

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
        # Trim messages to stay within token limits while preserving system messages
        trimmed_messages = trim_messages(
            state.messages,
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",
            token_counter=token_counter,
            # Always keep system messages at the start
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        # Create trimmed state for agent execution
        trimmed_state = AgentState(messages=trimmed_messages, token_usage=token_usage)

        # Run agent with live status display
        new_messages = run_agent_with_display(agent, trimmed_state)

        # Update state with new messages
        state = AgentState(messages=state.messages + new_messages, token_usage=token_usage)

        output = state.messages[-1] if state.messages else None

        print(
            ROBOT,
            output.content if output else None
        )

        token_usage.ingest_from_messages(state.messages)

        # Show persistent cost display after each turn
        print()  # Add spacing
        token_usage.print_panel()

        # Exit after single invocation if --single flag is used
        if args.single:
            break

        # Get next user input
        print(HUMAN)
        try:
            user_input = session.prompt("Anything else? ")
        except (KeyboardInterrupt, SystemExit):
            graceful_exit()

        # Append new user message to state
        state.messages.append(HumanMessage(content=user_input))
