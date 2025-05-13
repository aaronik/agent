import sys
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools import search_text, fetch

model = ChatOpenAI(model="gpt-4.1-mini")

tools = [search_text, fetch]

agent = create_react_agent(model, tools=tools)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query as command line argument.")
        sys.exit(1)

    # Initial user input from command line
    user_input = sys.argv[1]

    # State for conversation/messages
    state = {"messages": [("user", user_input)]}

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
        user_input = input("Enter next query (or type 'exit' to quit): ")
        if user_input.strip().lower() == 'exit':
            print("Exiting.")
            break

        # Append new user message to state
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(("user", user_input))
