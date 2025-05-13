import sys
import signal
import aisuite as ai

import src.tools as tools
from src.constants import system_string
from openai.types.chat.chat_completion import ChatCompletion

from src.util import (
    TokenUsage,
    sys_ls,
    sys_pwd,
    sys_uname,
    message_from_choice,
    message_from_user_input,
    print_token_usage,
)

MODEL = "openai:gpt-4.1-mini"
# MODEL = "openai:o4-mini"
# MODEL = "openai:gpt-3.5-turbo"
# MODEL = "ollama:llama3.1:latest"  # Doesn't seem to accept tool calls

cli_input = " ".join(sys.argv[1:])
user_request = cli_input if cli_input else input("What's up? ")

# Main chat state
messages: list[ai.Message] = [
    ai.Message(
        role="system",
        content=system_string
    ),
    ai.Message(
        role="system",
        content=f"[SYSTEM INFO] uname -a:\n{sys_uname()}"
    ),
    ai.Message(
        role="system",
        content=f"[SYSTEM INFO] pwd:\n{sys_pwd()}"
    ),
    ai.Message(
        role="system",
        content=f"[SYSTEM INFO] ls -l:\n{sys_ls()}"
    ),
    ai.Message(
        role="user",
        content=user_request
    ),
]

# Our token usage state, periodically updated throughout session
token_usage = TokenUsage(
    prompt_tokens=0,
    completion_tokens=0
)


# Handle Ctrl-C: print total tokens and exit
def signal_handler(*_):
    print_token_usage(MODEL, token_usage)
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

client = ai.Client()

MAX_RETRIES = 5


def trim_messages_for_context_limit(
        messages: list[ai.Message],
        retries: int,
        max_retries: int,
        max_trim: int = 2
):
    """
    Trim recent user and assistant messages to reduce context length.
    Keeps at least the first system message and user request.
    Returns updated messages and whether trimming was done.
    """
    if retries >= max_retries:
        return
    if len(messages) > 4:
        messages = messages[:-max_trim]
        messages.append(ai.Message(
            role='system',
            content=(
                f"[SYSTEM NOTE] Previous messages trimmed"
                "due to context length limit "
                f"after {retries} retry(ies)."
            )
        ))
        print(
            f"[INFO] Context length exceeded,"
            f"trimming recent messages. Retry {retries}/{max_retries}")
        return
    print("[ERROR] Cannot trim messages further. Aborting.")
    return


while True:
    retries = 0
    while True:
        try:
            response: ChatCompletion = client.chat.completions.create(
                model=MODEL,
                messages=[message.model_dump() for message in messages],
                tools=[
                    tools.fetch,
                    tools.search_text,
                    tools.search_images,
                    tools.run_shell_command,
                    tools.gen_image,
                    tools.read_file,
                    tools.write_file,
                    tools.apply_diff,
                    tools.build_trim_message(messages),
                ],
                max_turns=50
            )
            break
        except Exception as e:
            err_str = str(e)
            # Check if it's a JSON error string with
            # .code context_length_exceeded and try trimming messages
            if 'context_length_exceeded' in err_str:
                retries += 1
                trim_messages_for_context_limit(
                    messages, retries, MAX_RETRIES
                )
            else:
                raise e

    if hasattr(response, 'usage') and response.usage is not None:
        token_usage.prompt_tokens += response.usage.prompt_tokens
        token_usage.completion_tokens += response.usage.completion_tokens

    choice = response.choices[0]

    messages.append(message_from_choice(choice))

    print("\n---\n")
    print(choice.message.content)

    print("\n---\n")
    user_input = input("Anything else? ")

    user_message = message_from_user_input(user_input)
    messages.append(user_message)

    print("\n---\n")
