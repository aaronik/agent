import sys
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import src.tools as tools
from src.constants import system_string
from src.util import sys_ls, sys_pwd, sys_uname

model = ChatOpenAI(model="gpt-4.1-mini")

tools = [
    tools.search_text,
    tools.search_images,
    tools.fetch,
    tools.read_file,
    tools.write_file,
    tools.apply_diff,
    tools.gen_image,
    tools.run_shell_command,
]

agent = create_react_agent(model, tools=tools)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query as command line argument.")
        sys.exit(1)

    # Initial user input from command line
    user_input = sys.argv[1] or input("What's up? ")

    # State for conversation/messages
    state = {"messages": [
        ("system", system_string),
        ("system", f"[SYSTEM INFO] uname -a:\n{sys_uname()}"),
        ("system", f"[SYSTEM INFO] pwd:\n{sys_pwd()}"),
        ("system", f"[SYSTEM INFO] ls -l:\n{sys_ls()}"),
        ("user", user_input),
    ]}

    while True:
        # Run the agent with invoke
        state = agent.invoke(state)

        output = None
        if "messages" in state and len(state["messages"]) > 0:
            output = state["messages"][-1]

        print(
            "Answer:",
            output.content
            if hasattr(output, 'content') and output is not None
            else output
        )

        # Get next user input
        user_input = input("Anything else? ")

        # Append new user message to state
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(("user", user_input))
