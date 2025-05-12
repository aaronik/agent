import sys
import signal
import aisuite as ai

import src.tools as tools
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
user_request = cli_input if cli_input else input("What'll it be, boss? ")

system_string = (
    "You are a CLI agent tool. You're run from the command line."
    "Your purpose is to automate tasks for the user."
    "You have been supplied with a series of tools to get the job done."
    "Prefer taking action over asking the user permission."
    "Aggressively trim any messages that aren't pertinent to latest chat."
    "Remember to:"
    "- Cite all sources include links in every citation."
)

# Main chat state
messages: list[ai.Message] = [
    ai.Message(
        role="system",
        content=system_string
    ),
    ai.Message(
        role="user",
        content=user_request
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
    )
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

while True:
    print("\n---\n")
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
                    tools.printz,
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
            # Check if it's a JSON error string with .code context_length_exceeded
            # and try trimming messages
            if 'context_length_exceeded' in err_str and retries < MAX_RETRIES:
                retries += 1
                # Remove recent user and assistant messages except system
                # Keep at least first system message and user request
                # Remove 2 messages per retry if possible
                if len(messages) > 4:
                    messages = messages[:-2]
                    # Add a system note about trimming
                    messages.append(ai.Message(
                        role='system',
                        content=f"[SYSTEM NOTE] Previous messages trimmed due to context length limit after {retries} retry(ies)."
                    ))
                    print(f"[INFO] Context length exceeded, trimming recent messages. Retry {retries}/{MAX_RETRIES}")
                    continue
                else:
                    print("[ERROR] Cannot trim messages further. Aborting.")
                    raise
            else:
                raise

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
