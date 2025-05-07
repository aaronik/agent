from typing import Any, Dict
from time import sleep
import json
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition


client = OpenAI()
starting_assistant = ""
starting_thread = ""

def fetch(url: str, max_length: int = 1000000):
    """Fetch content from the given URL with a size limit."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        if len(text) > max_length:
            return text[:max_length] + "\n\n[Content truncated due to size limitations]"
        return text
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

def create_assistant():
    if starting_assistant == "":
        my_assistant = client.beta.assistants.create(
            instructions="You are a helpful assistant.",
            name="MyQuickstartAssistant",
            model="gpt-3.5-turbo",
            tools=[tool_fetch],
        )
    else:
        my_assistant = client.beta.assistants.retrieve(starting_assistant)

    return my_assistant

def create_thread():
    empty_thread = client.beta.threads.create()
    return empty_thread


def send_message(thread_id: str, message: str):
    thread_message = client.beta.threads.messages.create(
        thread_id,
        role="user",
        content=message,
    )
    return thread_message


def run_assistant(thread_id: str, assistant_id: str):
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )
    return run


def get_newest_message(thread_id: str):
    thread_messages = client.beta.threads.messages.list(thread_id)
    return thread_messages.data[0]


def get_run_status(thread_id: str, run_id: str):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status


def run_action(thread_id: str, run_id: str):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    if not run.required_action:
        return

    for tool in run.required_action.submit_tool_outputs.tool_calls:

        if tool.function.name == "fetch":
            arguments: dict[str, Any] = json.loads(tool.function.arguments)
            url: str = arguments["url"]

            print("using tool [fetch], url: " + url)
            resp = fetch(url)

            _ = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=[
                    {
                        "tool_call_id": tool.id,
                        "output": resp,
                    },
                ],
            )
        else:
            raise Exception(
                f"Unsupported function call: {tool.function.name} provided."
            )


def main():
    my_assistant = create_assistant()
    my_thread = create_thread()

    while True:
        user_message = input("Enter your message: ")
        if user_message.lower() == "exit":
            break

        _ = send_message(my_thread.id, user_message)
        run = run_assistant(my_thread.id, my_assistant.id)

        while run.status != "completed":
            run.status = get_run_status(my_thread.id, run.id)

            # If assistant needs to call a function, it will enter the "requires_action" state
            if run.status == "requires_action":
                run_action(my_thread.id, run.id)

            sleep(1)
            print("⏳", end="\r", flush=True)

        sleep(0.5)

        response = get_newest_message(my_thread.id)
        for content in response.content:
            obj = content.to_dict()["text"]
            print("\n---\n")
            print(obj["value"])


if __name__ == "__main__":
    main()
