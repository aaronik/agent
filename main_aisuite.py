import sys
import signal
import aisuite as ai

import src.tools as tools
from openai.types.chat.chat_completion import ChatCompletion

from src.util import (
    TokenUsage,
    get_sys_info,
    message_from_choice,
    message_from_user_input,
    print_token_usage,
)


cli_input = " ".join(sys.argv[1:])
user_request = cli_input if cli_input else input("What'll it be, boss? ")

system_info = get_sys_info()
system_string = (
    "* You are a CLI agent tool. You're run from the command line."
    "* Cite all sources and you include links in every citation."
    "* Use as many shell commands as needed if it helps meet"
    "the user's request."
    f"* The user's system is: uname -a => {system_info}"
)

messages: list[ai.Message] = [
    ai.Message(
        role="system",
        content=system_string
    ),
    ai.Message(
        role="user",
        content=user_request
    )
]


token_usage = TokenUsage(
    prompt_tokens=0,
    completion_tokens=0
)


# Handle Ctrl-C: print total tokens and exit
def signal_handler(*_):
    print_token_usage(token_usage)
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

client = ai.Client()

while True:
    response: ChatCompletion = client.chat.completions.create(
        model="openai:gpt-4.1-mini",
        messages=messages,
        tools=[
            tools.fetch,
            tools.search_text,
            tools.search_images,
            tools.shell_command,
            tools.printz,
            tools.gen_image,
        ],
        max_turns=20  # Maximum number of back-and-forth tool calls
    )

    if response.usage:
        token_usage.prompt_tokens += response.usage.prompt_tokens
        token_usage.completion_tokens += response.usage.completion_tokens

    for i, choice in enumerate(response.choices):
        messages.append(message_from_choice(choice))

        print("\n---\n")
        print(choice.message.content)

    print("\n---\n")
    user_input = input("Anything else? ")

    user_message = message_from_user_input(user_input)
    messages.append(user_message)

    print("\n---\n")
