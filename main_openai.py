import inspect
from typing import Any, Callable
from time import sleep
import json
from openai import OpenAI
from openai.types.beta.threads import RequiredActionFunctionToolCall
from openai.types.chat import ChatCompletionToolParam

from src import tools


client = OpenAI()
starting_assistant = ""
starting_thread = ""


def function_to_tool(fn: Callable) -> ChatCompletionToolParam:
    """
    Helper for tools.
    Normally to make a tool you need a function and a json schema.
    function_to_tool creates that schema from the function itself.
    """

    # Mapping from python type names to OpenAI-compatible types
    type_map = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
    }

    def get_openai_type(annotation):
        # Handle annotations given as types
        if hasattr(annotation, '__name__'):
            return type_map.get(annotation.__name__, 'string')
        # Fallback for other cases
        return 'string'

    sig = inspect.signature(fn)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": fn.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": get_openai_type(param.annotation)}
                    for name, param in sig.parameters.items()
                    if param.annotation != inspect.Parameter.empty
                },
                "required": [
                    name for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                ]
            }
        }
    }


# Define the tools
tool_fetch = function_to_tool(tools.fetch)
tool_search = function_to_tool(tools.search_text)


def create_assistant():
    if starting_assistant == "":
        my_assistant = client.beta.assistants.create(
            instructions="You are a helpful assistant.",
            name="MyQuickstartAssistant",
            model="gpt-3.5-turbo",
            tools=[tool_fetch, tool_search],
        )
    else:
        my_assistant = client.beta.assistants.retrieve(starting_assistant)

    return my_assistant


def create_thread():
    empty_thread = client.beta.threads.create()
    return empty_thread


def run_assistant(thread_id: str, assistant_id: str):
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )
    return run


def send_message(thread_id: str, message: str):
    thread_message = client.beta.threads.messages.create(
        thread_id,
        role="user",
        content=message,
    )
    return thread_message


def get_newest_message(thread_id: str):
    thread_messages = client.beta.threads.messages.list(thread_id)
    return thread_messages.data[0]


def get_run_status(thread_id: str, run_id: str):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status


def handle_tool_call(
    tool: RequiredActionFunctionToolCall,
    thread_id: str,
    run_id: str
):
    arguments: dict[str, Any] = json.loads(tool.function.arguments)
    url: str = arguments["url"]

    if tool.function.name == "fetch":
        resp = tools.fetch(url)

        # Tool outputs need to have a specific relationship to the message that
        # invoked them, which the framework does in this call
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=[
                {
                    "tool_call_id": tool.id,
                    "output": resp,
                },
            ],
        )

    elif tool.function.name == "search_text":
        resp = tools.fetch(url)

        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
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


def run_action(thread_id: str, run_id: str):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    if not run.required_action:
        return

    for tool in run.required_action.submit_tool_outputs.tool_calls:
        handle_tool_call(tool, thread_id, run_id)


def main():
    my_assistant = create_assistant()
    thread = create_thread()

    while True:
        user_message = input("Enter your message: ")
        if user_message.lower() == "exit":
            break

        send_message(thread.id, user_message)
        run = run_assistant(thread.id, my_assistant.id)

        while run.status != "completed":
            run.status = get_run_status(thread.id, run.id)

            # If assistant needs to call a function, it will enter the
            # "requires_action" state
            if run.status == "requires_action":
                run_action(thread.id, run.id)

            sleep(1)
            print("⏳", end="\r", flush=True)

        sleep(0.5)

        response = get_newest_message(thread.id)
        content = response.content[0].model_dump()
        print(content["text"]["value"])


if __name__ == "__main__":
    main()
