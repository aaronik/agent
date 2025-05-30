import sys
import signal
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

model = ChatOpenAI(model="gpt-4.1")

tools = [
    tools.search_text,
    # tools.search_text_alternative,
    # tools.search_images,
    tools.fetch,
    tools.read_file,
    tools.write_file,
    tools.apply_diff,
    tools.gen_image,
    tools.run_shell_command,
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
    # Initial user input from command line or prompt
    if len(sys.argv) < 2:
        user_input = input("What's up? ")
    else:
        user_input = " ".join(sys.argv[1:])

    # Initialize state with typed messages
    state = AgentState(messages=[
        SystemMessage(content=system_string),
        SystemMessage(content=f"[SYSTEM INFO] uname -a: {sys_uname()}"),
        SystemMessage(content=f"[SYSTEM INFO] pwd: {sys_pwd()}"),
        SystemMessage(content=f"[SYSTEM INFO] ls -l: {sys_ls()}"),
        SystemMessage(content=f"[SYSTEM INFO] git ls-files: {sys_git_ls()}"),
        HumanMessage(content=user_input),
    ])

    print("\n---\n")

    while True:
        # Run the agent
        new_state_dict = agent.invoke(state, {"recursion_limit": 200})

        state = AgentState(messages=new_state_dict["messages"])

        output = state.messages[-1] if state.messages else None

        print(
            "\n---\n",
            output.content if output else None
        )

        token_usage.ingest_from_messages(state.messages)

        # Get next user input
        print("\n---\n")
        user_input = input("Anything else? ")
        print("\n---\n")

        # Append new user message to state
        state.messages.append(HumanMessage(content=user_input))
