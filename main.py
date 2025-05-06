import os
import sys
import openai
import argparse
import requests
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition


def fetch(url: str):
    """Fetch content from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching URL {url}: {e}"

tool_fetch = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="fetch",
        description="Fetches a document from the web",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch data from"}
            },
            "required": ["url"]
        }
    )
)

def main():
    parser = argparse.ArgumentParser(
        description="AI-powered CLI automation tool"
    )
    _ = parser.add_argument(
        'task',
        type=str,
        nargs=argparse.REMAINDER,
        help="Describe the CLI task or flow you want to automate."
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable.")
        sys.exit(1)
    openai.api_key = api_key

    user_input = ' '.join(args.task) if args.task else ""
    if not user_input:
        print("Please enter a description of the CLI task to automate.")
        sys.exit(1)

    prompt = "You are a helpful assistant"

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        tools=[tool_fetch],
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            if function_name == "fetch":
                print("Fetching: " + arguments)

    else:
        print(message.content)


if __name__ == "__main__":
    main()

