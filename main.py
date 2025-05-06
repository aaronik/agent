import os
import sys
import openai
import argparse
from pydantic import BaseModel
import requests


def fetch(url: str):
    """Fetch content from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching URL {url}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered CLI automation tool")
    _ = parser.add_argument(
        'task',
        nargs=argparse.REMAINDER,
        help="Describe the CLI task or flow you want to automate."
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable.")
        sys.exit(1)

    user_input = ' '.join(args.task) if args.task else ""
    if not user_input:
        print("Please enter a description of the CLI task to automate.")
        sys.exit(1)

    openai.api_key = api_key

    prompt = "You are an assistant that writes bash scripts to automate tasks as described by the user."

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        tools=[
            {
                "name": "fetch",
                "description": "Fetches a document from the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch data from"}
                    },
                    "required": ["url"]
                }
            }
        ],
        function_call="auto"  # Let the model choose which function to call
    )

    message = response.choices[0].message

    if message.function_call:
        function_name = message.function_call.name
        arguments = message.function_call.arguments

        if function_name == "fetch":
            # Handle fetch function call as before
            bash_script = message.content
            print("\nGenerated Bash Script:\n")
            print(bash_script)

        elif function_name == "include_tools":
            import json
            args = json.loads(arguments)
            tools_list = args.get("tools", [])
            # Generate include statements for the specified tools
            includes = []
            for tool in tools_list:
                # Simple example: map tool names to bash commands or similar includes
                if tool.lower() == "curl":
                    includes.append("# Ensure curl is installed\ncommand -v curl >/dev/null 2>&1 || { echo >&2 'curl is not installed. Aborting.'; exit 1; }")
                elif tool.lower() == "jq":
                    includes.append("# Ensure jq is installed\ncommand -v jq >/dev/null 2>&1 || { echo >&2 'jq is not installed. Aborting.'; exit 1; }")
                else:
                    includes.append(f"# Include tool: {tool} (custom handling may be required)")
            include_section = "\n".join(includes)
            print("\nIncluded Tools Section:\n")
            print(include_section)

    else:
        # No function call, just print content
        bash_script = message.content
        print("\nGenerated Bash Script:\n")
        print(bash_script)


if __name__ == "__main__":
    main()

