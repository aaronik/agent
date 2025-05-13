from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from src.tools import search_text, fetch

# Define our tools
search_tool = Tool(
    name="SearchText",
    func=search_text,
    description=(
        "Useful for when you need to answer questions by searching "
        "the web for text results"
    )
)

fetch_tool = Tool(
    name="FetchContent",
    func=fetch,
    description="Provide fully qualified url"
)

# Initialize the chat model
chat = ChatOpenAI(temperature=0)

# Create the agent with the tools, allowing iterative tool calls
agent = initialize_agent(
    tools=[search_tool, fetch_tool],
    llm=chat,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    verbose=True
)


if __name__ == "__main__":
    print("Enter your query (type 'exit' to quit):")
    while True:
        query = input("\nQuery: ")
        if query.lower() == 'exit':
            break
        # Run the agent on the query
        result = agent.run(query)
        print("\nAnswer:", result)
